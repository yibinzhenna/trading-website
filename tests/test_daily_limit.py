"""
Daily backtest allowance: per address for visitors, per account for users.
"""

import threading
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from api import deps
from api.auth import TokenVerifier
from quantlab.jobs import JobStore
from test_auth import URL, FakeJWKS, bearer, token
from test_store import FakeJob

BODY = {"symbol": "SPY", "kind": "momentum"}
ALICE, BOB = bearer(token("alice")), bearer(token("bob"))


def make_client(tmp_path, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), admin_token="",
                    rate_limit=10_000, user_rate_limit=10_000,
                    max_inflight=1_000, client_ip_header="",
                    trust_proxy_hops=0, supabase_url=URL,
                    supabase_publishable_key="sb_publishable_test",
                    anthropic_api_key="", daily_limit=50, user_daily_limit=200,
                    database_url=f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    deps.verifier = TokenVerifier(URL, jwks_client=FakeJWKS())
    from api.main import app
    return TestClient(app)


def post(c, headers=None, body=BODY):
    return c.post("/backtest", json=body, headers=headers or {})


def settle():
    for job in list(deps.jobs._jobs):
        deps.jobs.wait(job, timeout=30)


# ── Visitors ───────────────────────────────────────────────────────────────

def test_visitor_daily_cap(tmp_path):
    with make_client(tmp_path, daily_limit=3) as c:
        codes = [post(c).status_code for _ in range(4)]
        r = post(c)
    assert codes == [202, 202, 202, 429]
    assert "Sign in" in r.json()["detail"]
    assert int(r.headers["Retry-After"]) > 3600      # a day-scale wait


def test_invalid_requests_do_not_use_the_allowance(tmp_path):
    with make_client(tmp_path, daily_limit=1) as c:
        bad = post(c, body={"symbol": "SPY", "kind": "momentum",
                            "params": {"lookbak": 3}})
        good = post(c)
    assert bad.status_code == 422 and good.status_code == 202


def test_zero_disables_the_cap(tmp_path):
    with make_client(tmp_path, daily_limit=0) as c:
        assert {post(c).status_code for _ in range(5)} == {202}


# ── Accounts ───────────────────────────────────────────────────────────────

def test_account_daily_cap_is_per_account(tmp_path):
    with make_client(tmp_path, user_daily_limit=2) as c:
        codes = [post(c, ALICE).status_code for _ in range(3)]
        bob = post(c, BOB).status_code
        settle()
        me = c.get("/me", headers=ALICE).json()["backtests"]
    assert codes == [202, 202, 429] and bob == 202
    assert me == {"used": 2, "limit": 2, "remaining": 0}


def test_account_count_survives_a_restart(tmp_path):
    """Render's free tier sleeps and restarts; a memory-only count would
    hand out a fresh allowance every time it woke."""
    with make_client(tmp_path, user_daily_limit=2) as c:
        post(c, ALICE), post(c, ALICE)
        settle()
        deps.jobs = JobStore(workers=1, on_finish=deps._persist)
        r = post(c, ALICE)
    assert r.status_code == 429


def test_runs_older_than_a_day_do_not_count(tmp_path):
    with make_client(tmp_path, user_daily_limit=1) as c:
        old = FakeJob("0ld0ld0ld0ld0ld0", (datetime.now(timezone.utc)
                      - timedelta(hours=25)).isoformat(timespec="milliseconds"))
        old.meta["owner_id"] = "alice"
        deps.runs.save(old)
        assert post(c, ALICE).status_code == 202


def test_parallel_burst_cannot_exceed_the_cap(tmp_path):
    with make_client(tmp_path, user_daily_limit=3) as c:
        codes = []
        threads = [threading.Thread(target=lambda: codes.append(
            post(c, ALICE).status_code)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert sorted(codes) == [202] * 3 + [429] * 7


def test_signing_in_does_not_inherit_the_visitor_count(tmp_path):
    with make_client(tmp_path, daily_limit=1, user_daily_limit=5) as c:
        assert post(c).status_code == 202
        assert post(c).status_code == 429
        assert post(c, ALICE).status_code == 202


# ── Submitters do not wait on each other (issue #4) ────────────────────────

def test_one_slow_account_does_not_hold_up_others(tmp_path, monkeypatch):
    """The daily check for an account queries the database. It used to run
    under one site-wide lock, so a slow query for one user stalled every
    other user's and every visitor's submission behind it."""
    import time
    from api import security
    from fastapi import Request

    def stripe(key):
        return hash(key) % len(security._STRIPES)

    # A second user and a visitor address on different stripes from alice.
    other = next(f"u{i}" for i in range(500)
                 if stripe(f"user:u{i}") != stripe("user:alice"))
    with make_client(tmp_path, user_daily_limit=50, daily_limit=50) as c:
        original = deps.runs.owner_runs_since
        gate = threading.Event()

        def slow(owner_id, since):
            if owner_id == "alice":
                gate.wait(5)
            return original(owner_id, since)

        monkeypatch.setattr(deps.runs, "owner_runs_since", slow)
        alice = threading.Thread(target=lambda: post(c, ALICE))
        alice.start()
        time.sleep(0.2)                       # alice is now inside her check
        t = time.monotonic()
        other_code = post(c, bearer(token(other))).status_code
        visitor_code = post(c).status_code
        elapsed = time.monotonic() - t
        gate.set()
        alice.join(10)
    assert other_code == 202 and visitor_code == 202
    assert elapsed < 2.0, f"waited {elapsed:.1f}s behind another user"


def test_the_same_account_is_still_serialised():
    """Striping must keep one submitter's requests on one lock, or the
    parallel-burst guarantee breaks."""
    from api import security
    from starlette.requests import Request
    req = Request({"type": "http", "headers": [], "client": ("1.2.3.4", 1)})
    alice = type("U", (), {"id": "alice"})()
    assert security.submission_lock(req, alice) is security.submission_lock(req, alice)
    assert security.submission_lock(req, None) is security.submission_lock(req, None)
