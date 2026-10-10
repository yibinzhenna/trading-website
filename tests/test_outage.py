"""
Database outages: the site degrades, it does not go down.

The database is made unreachable by pointing SQLite into a directory that
does not exist — the same OperationalError a paused Supabase project or a
dropped connection produces — and "comes back" when the directory appears.
"""

import time

import pytest
from fastapi.testclient import TestClient

from api import deps, store
from api.auth import TokenVerifier
from api.store import RunStore, StoreUnavailable
from test_auth import URL, FakeJWKS, bearer, token

BODY = {"symbol": "SPY", "kind": "momentum"}
ALICE = bearer(token("alice"))


def down_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'not-yet' / 'runs.db').as_posix()}"


def make_client(tmp_path, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), admin_token="",
                    rate_limit=10_000, user_rate_limit=10_000,
                    max_inflight=1_000, client_ip_header="", trust_proxy_hops=0,
                    supabase_url=URL, supabase_publishable_key="sb_publishable_test",
                    user_daily_limit=50, database_url=down_url(tmp_path))
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    deps.verifier = TokenVerifier(URL, jwks_client=FakeJWKS())
    from api.main import app
    return TestClient(app)


def health_status(c, want=None, timeout=10):
    """/health answers at once with the last known status; poll until the
    background probe has reported (or reports `want`)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = c.get("/health")
        assert r.status_code == 200
        status = r.json()["database_status"]
        if status != "unknown" and (want is None or status == want):
            return status
        time.sleep(0.05)
    return status


def finish(c, headers=None):
    r = c.post("/backtest", json=BODY, headers=headers or {})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert deps.jobs.wait(job_id, timeout=30).status == "done"
    return job_id


# ── The store ──────────────────────────────────────────────────────────────

def test_creating_the_store_does_not_touch_the_database(tmp_path):
    """It used to connect in the constructor with no timeout: an unreachable
    database held startup for over two minutes, then crashed it."""
    t = time.monotonic()
    s = RunStore("postgresql://u:p@127.0.0.1:1/never")
    assert time.monotonic() - t < 1.0
    s.close()


def test_postgres_connections_time_out(tmp_path):
    s = RunStore("postgresql://u:p@db.invalid:5432/x")
    assert s.engine.dialect.name == "postgresql"
    # connect_args reach the driver as conninfo keywords.
    assert s.engine.url.drivername == "postgresql+psycopg"
    assert store.CONNECT_TIMEOUT <= 10
    s.close()


def test_failures_trip_a_breaker_that_fails_fast(tmp_path, monkeypatch):
    s = RunStore(down_url(tmp_path))
    with pytest.raises(StoreUnavailable):
        s.get("x")
    attempts = []
    monkeypatch.setattr(s.engine, "connect",
                        lambda *a, **k: attempts.append(1) or 1 / 0)
    with pytest.raises(StoreUnavailable):
        s.get("x")
    assert attempts == []                   # did not even try
    assert s.check() == "unavailable"


def test_store_recovers_when_the_database_returns(tmp_path, monkeypatch):
    s = RunStore(down_url(tmp_path))
    assert s.check() == "unavailable"
    (tmp_path / "not-yet").mkdir()
    monkeypatch.setattr(s, "_down_until", 0.0)
    assert s.check() == "ok"
    assert s.count() == 0


def test_warm_never_raises(tmp_path):
    RunStore(down_url(tmp_path)).warm()


# ── The site during an outage ──────────────────────────────────────────────

def test_health_stays_up_and_reports_the_database(tmp_path):
    with make_client(tmp_path) as c:
        assert health_status(c) == "unavailable"


def test_health_never_waits_on_the_database(tmp_path, monkeypatch):
    """Even while a probe is stuck connecting, /health answers at once."""
    import threading
    stuck = threading.Event()
    with make_client(tmp_path) as c:
        monkeypatch.setattr(deps.runs, "check", lambda: stuck.wait(5))
        monkeypatch.setattr(deps.runs, "_health", ("unknown", float("-inf")))
        t = time.monotonic()
        assert c.get("/health").status_code == 200
        assert c.get("/health").status_code == 200
        assert time.monotonic() - t < 1.0
        stuck.set()


def test_backtests_keep_working(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c)
        got = c.get(f"/backtest/{job_id}")
        chart = c.get(f"/backtest/{job_id}/equity")
    assert got.status_code == 200 and got.json()["status"] == "done"
    assert chart.status_code == 200


def test_signed_in_backtests_keep_working(tmp_path):
    """The daily cap fails open on the database (memory still counts);
    per-minute limits and the capacity cap still hold."""
    with make_client(tmp_path) as c:
        assert finish(c, ALICE)


def test_saved_data_returns_a_clear_503(tmp_path):
    with make_client(tmp_path) as c:
        for method, path in (("GET", "/backtest/0123456789abcdef"),
                             ("GET", "/me/runs"),
                             ("DELETE", "/runs/0123456789abcdef")):
            r = c.request(method, path, headers=ALICE)
            assert r.status_code == 503, path
            assert "temporarily unavailable" in r.json()["detail"]
            assert r.headers["Retry-After"] == "15"


def test_account_page_still_loads(tmp_path):
    with make_client(tmp_path) as c:
        r = c.get("/me", headers=ALICE)
    assert r.status_code == 200 and r.json()["email"] == "a@example.com"


def test_research_refuses_rather_than_spending_untracked(tmp_path):
    """Quotas live in the database. Without it a session would cost money
    with nothing counting it, so research fails closed."""
    from test_research import FakeClient
    with make_client(tmp_path, data_root="sampledata") as c:
        deps.research_client = FakeClient([])
        r = c.post("/research", json={"symbol": "DEMO-REGIME"}, headers=ALICE)
    assert r.status_code == 503


def test_the_site_recovers_on_its_own(tmp_path, monkeypatch):
    with make_client(tmp_path) as c:
        assert c.get("/me/runs", headers=ALICE).status_code == 503
        (tmp_path / "not-yet").mkdir()
        monkeypatch.setattr(deps.runs, "_down_until", 0.0)
        monkeypatch.setattr(deps.runs, "_health", ("unavailable", float("-inf")))
        job_id = finish(c, ALICE)
        assert health_status(c, want="ok") == "ok"
        mine = c.get("/me/runs", headers=ALICE).json()
    assert [r["job_id"] for r in mine] == [job_id]
