"""
AI research over HTTP: who may start a session, and every limit on cost.
"""

import threading

import pytest

from fastapi.testclient import TestClient

from api import deps
from api.auth import TokenVerifier
from test_auth import URL, FakeJWKS, bearer, token
from test_research import TREND, FakeClient, call

# Distinct inboxes: research quotas follow the email, not the account.
ALICE = bearer(token("alice", email="alice@example.com"))
BOB = bearer(token("bob", email="bob@example.com"))
BODY = {"symbol": "DEMO-REGIME", "trials": 3}
SCRIPT = [TREND, call("finish", pick=1, notes="Trend held; Sharpe 9.9.")]


def make_client(tmp_path, research=True, client=None, **overrides):
    settings = dict(provider="local", data_root="sampledata",
                    cache_root=str(tmp_path / "cache"), admin_token="",
                    rate_limit=10_000, user_rate_limit=10_000,
                    max_inflight=1_000, client_ip_header="",
                    trust_proxy_hops=0, supabase_url=URL,
                    supabase_publishable_key="sb_publishable_test",
                    anthropic_api_key="", research_daily_limit=3,
                    research_global_daily_limit=50, research_max_queue=3,
                    database_url=f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    deps.verifier = TokenVerifier(URL, jwks_client=FakeJWKS())
    if research:
        deps.research_client = client or FakeClient(list(SCRIPT))
    from api.main import app
    return TestClient(app)


def start(c, headers=ALICE, body=BODY):
    return c.post("/research", json=body, headers=headers)


def finish(c, headers=ALICE):
    r = start(c, headers)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deps.research_jobs.wait(job_id, timeout=60)
    return job_id


class BlockingClient(FakeClient):
    """Holds the session open after its first trial until released."""

    def __init__(self, script):
        super().__init__(script)
        self.release = threading.Event()

    def create(self, **kw):
        if len(self.requests) == 1:
            self.release.wait(10)
        return super().create(**kw)


# ── Access ─────────────────────────────────────────────────────────────────

def test_off_without_an_api_key(tmp_path):
    with make_client(tmp_path, research=False) as c:
        assert c.get("/config").json()["research"] is None
        assert start(c).status_code == 503


def test_config_advertises_research_and_its_limits(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/config").json()["research"] == {
            "daily_limit": 3, "max_trials": 8}


def test_anonymous_visitors_cannot_spend_tokens(tmp_path):
    with make_client(tmp_path) as c:
        assert c.post("/research", json=BODY).status_code == 401


def test_unusable_symbol_is_refused_before_any_cost(tmp_path):
    client = FakeClient(list(SCRIPT))
    with make_client(tmp_path, client=client) as c:
        r = start(c, body={"symbol": "NOSUCH"})
        assert r.status_code == 422
        assert c.get("/me", headers=ALICE).json()["research"]["used"] == 0
    assert client.requests == []


# ── A full session ─────────────────────────────────────────────────────────

def test_session_runs_and_is_saved(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c)
        deps.research_jobs = type(deps.research_jobs)(workers=1)  # "restart"
        got = c.get(f"/research/{job_id}").json()
        mine = c.get("/me/research", headers=ALICE).json()
    st = got["state"]
    assert got["status"] == "done" and st["final"]["trial"] == 1
    assert st["holdout_window"][0] > st["research_window"][1]
    assert "9.9" not in st["notes"]                 # model's number scrubbed
    assert st["input_tokens"] > 0
    assert "owner_id" not in got and "alice" not in str(got)
    assert mine["sessions"][0]["job_id"] == job_id
    assert mine["quota"] == {"used": 1, "limit": 3, "remaining": 2}


def test_progress_is_visible_while_running(tmp_path):
    client = BlockingClient(list(SCRIPT))
    with make_client(tmp_path, client=client) as c:
        job_id = start(c).json()["job_id"]
        for _ in range(200):
            live = c.get(f"/research/{job_id}").json()
            if (live.get("state") or {}).get("trials"):
                break
            threading.Event().wait(0.02)
        client.release.set()
        deps.research_jobs.wait(job_id, timeout=30)
    assert live["status"] == "running" and len(live["state"]["trials"]) == 1


# ── Cost limits ────────────────────────────────────────────────────────────

def test_daily_quota_per_user(tmp_path):
    with make_client(tmp_path, research_daily_limit=2) as c:
        for _ in range(2):
            deps.research_client = FakeClient(list(SCRIPT))
            finish(c)
        r = start(c)
        assert r.status_code == 429 and "Daily research limit" in r.json()["detail"]
        deps.research_client = FakeClient(list(SCRIPT))
        assert start(c, BOB).status_code == 202      # bob has his own


def test_one_session_at_a_time_per_user(tmp_path):
    client = BlockingClient(list(SCRIPT))
    with make_client(tmp_path, client=client) as c:
        first = start(c).json()["job_id"]
        second = start(c)
        client.release.set()
        deps.research_jobs.wait(first, timeout=30)
    assert second.status_code == 429 and "already" in second.json()["detail"]


def test_concurrent_requests_cannot_both_slip_past_the_quota(tmp_path):
    """Check-then-insert is locked; without it, a burst all reads used=0."""
    client = BlockingClient(list(SCRIPT))
    with make_client(tmp_path, client=client, research_daily_limit=1) as c:
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(
            start(c).status_code)) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        client.release.set()
    assert sorted(codes) == [202] + [429] * 5


def test_site_wide_daily_ceiling(tmp_path):
    with make_client(tmp_path, research_global_daily_limit=1) as c:
        finish(c, ALICE)
        r = start(c, BOB)
    assert r.status_code == 429 and "capacity" in r.json()["detail"]


def test_model_outage_fails_cleanly_and_is_not_charged(tmp_path):
    class Down(FakeClient):
        def create(self, **kw):
            raise ConnectionError("api.anthropic.com unreachable, key=sk-ant-x")

    with make_client(tmp_path, client=Down([])) as c:
        job_id = finish(c)
        got = c.get(f"/research/{job_id}").json()
        quota = c.get("/me", headers=ALICE).json()["research"]
    assert got["status"] == "failed"
    assert "could not be reached" in got["error"] and "sk-ant" not in got["error"]
    assert quota["used"] == 0


def test_restart_releases_a_stuck_session(tmp_path):
    with make_client(tmp_path) as c:
        deps.runs.research_create("feedfacefeedface", "alice", "X", "", "m")
        assert deps.runs.research_in_flight("alice") == 1
        deps.runs = deps._make_runs()                # process restart
        assert deps.runs.research_in_flight("alice") == 0
        assert c.get("/research/feedfacefeedface").json()["status"] == "failed"


# ── Provider selection ─────────────────────────────────────────────────────

def _settings(**kw):
    s = deps.settings
    for k, v in dict(anthropic_api_key="", deepseek_api_key="",
                     research_provider="anthropic", research_model="").items():
        setattr(s, k, v)
    for k, v in kw.items():
        setattr(s, k, v)


def test_deepseek_key_selects_deepseek_endpoint_and_model():
    _settings(deepseek_api_key="ds-test", research_provider="deepseek")
    client = deps._make_research_client()
    assert str(client.base_url).rstrip("/") == "https://api.deepseek.com/anthropic"
    assert deps.research_model() == "deepseek-flash"
    opts = deps.research_request_options()
    assert opts == {"thinking": {"type": "disabled"},
                    "extra_body": {"temperature": 0.3}}


def test_anthropic_gets_no_temperature():
    """The Anthropic SDK has no temperature argument; sending one in the
    body would be an unknown field."""
    _settings(anthropic_api_key="sk-test", research_provider="anthropic")
    assert deps._make_research_client() is not None
    assert deps.research_model() == "claude-haiku-5-5"
    assert deps.research_request_options() == {"thinking": {"type": "disabled"}}


def test_provider_without_its_key_is_off():
    _settings(anthropic_api_key="sk-test", research_provider="deepseek")
    assert deps._make_research_client() is None


def test_model_override_wins():
    _settings(deepseek_api_key="ds", research_provider="deepseek",
              research_model="deepseek-v4-pro")
    assert deps.research_model() == "deepseek-v4-pro"


# ── One inbox, one research quota (layer 1) ────────────────────────────────

from api.identity import canonical_email  # noqa: E402


@pytest.mark.parametrize("variant", [
    "jane.doe@gmail.com", "JANE.DOE@GMAIL.COM", "janedoe@gmail.com",
    "j.a.n.e.d.o.e@gmail.com", "jane.doe+research@gmail.com",
    "janedoe+1+2@googlemail.com", "  jane.doe@gmail.com "])
def test_gmail_variants_are_one_inbox(variant):
    assert canonical_email(variant) == "janedoe@gmail.com"


def test_plus_tags_fold_everywhere_but_dots_only_at_gmail():
    assert canonical_email("bob+x@outlook.com") == "bob@outlook.com"
    assert canonical_email("bo.b@outlook.com") == "bo.b@outlook.com"
    assert canonical_email("bob@outlook.com") != canonical_email("bob@gmail.com")


def test_variant_accounts_share_one_research_quota(tmp_path):
    """Three accounts, one inbox: three sessions in all, not three each."""
    accounts = [bearer(token(f"acct{i}", email=e)) for i, e in enumerate(
        ["jane.doe@gmail.com", "janedoe+2@gmail.com", "j.a.n.e.doe@googlemail.com"])]
    with make_client(tmp_path, research_daily_limit=2) as c:
        for acct in accounts[:2]:
            deps.research_client = FakeClient(list(SCRIPT))
            finish(c, acct)
        r = start(c, accounts[2])
        quota = c.get("/me", headers=accounts[2]).json()["research"]
        other = start(c, bearer(token("someone", email="someone.else@gmail.com")))
    assert r.status_code == 429 and "Daily research limit" in r.json()["detail"]
    assert quota == {"used": 2, "limit": 2, "remaining": 0}
    assert other.status_code == 202


def test_emails_are_not_stored(tmp_path):
    from sqlalchemy import select
    from api.store import research_identities
    with make_client(tmp_path) as c:
        finish(c, bearer(token("x", email="private.person@gmail.com")))
        with deps.runs.engine.connect() as conn:
            stored = [r[0] for r in conn.execute(select(research_identities.c.identity))]
    assert len(stored) == 1 and "@" not in stored[0] and "private" not in stored[0]
    assert len(stored[0]) == 32


# ── Throwaway inboxes (layer 2) ────────────────────────────────────────────

from api.identity import is_disposable  # noqa: E402


@pytest.mark.parametrize("email,blocked", [
    ("x@mailinator.com", True), ("X@MAILINATOR.COM", True),
    ("x@eu.mailinator.com", True), ("x@yopmail.fr", True),
    ("x@gmail.com", False), ("x@notmailinator.com", False),
    ("x@mailinator.com.example.org", False), ("", False), ("no-at-sign", False),
])
def test_disposable_domains(email, blocked):
    assert is_disposable(email) is blocked


def test_disposable_inbox_cannot_start_research(tmp_path):
    client = FakeClient(list(SCRIPT))
    with make_client(tmp_path, client=client) as c:
        r = start(c, bearer(token("burner", email="burner@guerrillamail.com")))
        quota_used = deps.runs.research_usage(
            __import__("datetime").datetime(2000, 1, 1,
                                            tzinfo=__import__("datetime").timezone.utc))
    assert r.status_code == 403 and "disposable" in r.json()["detail"]
    assert client.requests == [] and quota_used == 0


def test_blocklist_ships_with_the_app():
    """A missing file would quietly disable the check (it fails open)."""
    from api.identity import _BLOCKLIST, _disposable_domains
    assert _BLOCKLIST.is_file() and len(_disposable_domains()) >= 40


# ── One network, one cap across accounts (layer 3) ─────────────────────────

def account(n):
    return bearer(token(f"net{n}", email=f"person{n}@example.com"))


def net(ip):
    return {"CF-Connecting-IP": ip}


def start_from(c, who, ip):
    deps.research_client = FakeClient(list(SCRIPT))
    return c.post("/research", json=BODY, headers={**who, **net(ip)})


def test_many_accounts_on_one_network_share_a_cap(tmp_path):
    with make_client(tmp_path, research_network_daily_limit=2,
                     visitor_key="k", client_ip_header="cf-connecting-ip") as c:
        codes = []
        for n in range(3):
            r = start_from(c, account(n), "203.0.113.7")
            codes.append(r.status_code)
            if r.status_code == 202:
                deps.research_jobs.wait(r.json()["job_id"], timeout=60)
        elsewhere = start_from(c, account(9), "198.51.100.1")
    assert codes == [202, 202, 429]
    assert elsewhere.status_code == 202


def test_rotating_ipv6_is_one_network(tmp_path):
    with make_client(tmp_path, research_network_daily_limit=1,
                     visitor_key="k", client_ip_header="cf-connecting-ip") as c:
        first = start_from(c, account(1), "2001:db8:9:9::1")
        deps.research_jobs.wait(first.json()["job_id"], timeout=60)
        second = start_from(c, account(2), "2001:db8:9:9::abcd")
    assert second.status_code == 429 and "network" in second.json()["detail"]


def test_network_cap_without_a_key_counts_in_memory(tmp_path):
    with make_client(tmp_path, research_network_daily_limit=1,
                     visitor_key="", client_ip_header="cf-connecting-ip") as c:
        first = start_from(c, account(1), "203.0.113.8")
        deps.research_jobs.wait(first.json()["job_id"], timeout=60)
        second = start_from(c, account(2), "203.0.113.8")
    assert first.status_code == 202 and second.status_code == 429


def test_networks_are_stored_as_pseudonyms(tmp_path):
    from sqlalchemy import select
    from api.store import research_networks
    with make_client(tmp_path, visitor_key="k",
                     client_ip_header="cf-connecting-ip") as c:
        r = start_from(c, account(1), "203.0.113.9")
        deps.research_jobs.wait(r.json()["job_id"], timeout=60)
        with deps.runs.engine.connect() as conn:
            stored = [x[0] for x in conn.execute(select(research_networks.c.network))]
    assert len(stored) == 1 and "203.0.113" not in stored[0] and len(stored[0]) == 32


def test_a_rejected_request_does_not_use_up_the_network(tmp_path):
    """The account quota refuses first; the network allowance is untouched."""
    with make_client(tmp_path, research_network_daily_limit=2, research_daily_limit=1,
                     visitor_key="k", client_ip_header="cf-connecting-ip") as c:
        a = account(1)
        r = start_from(c, a, "203.0.113.10")
        deps.research_jobs.wait(r.json()["job_id"], timeout=60)
        refused = start_from(c, a, "203.0.113.10")          # account quota spent
        other = start_from(c, account(2), "203.0.113.10")   # network has room
    assert refused.status_code == 429 and "Daily research limit" in refused.json()["detail"]
    assert other.status_code == 202
