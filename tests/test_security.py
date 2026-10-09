"""Admin access, client identification and rate limiting."""

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from api import deps
from api.security import RateLimiter, client_key

ADMIN = {"Authorization": "Bearer s3cret"}
BODY = {"symbol": "SPY", "kind": "momentum"}


def make_client(tmp_path, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), admin_token="s3cret",
                    rate_limit=10_000, rate_window=60, max_inflight=1_000,
                    trust_proxy_hops=0)
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    from api.main import app
    return TestClient(app)


# ── Admin endpoints ────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path", [("GET", "/cache"),
                                         ("DELETE", "/cache"),
                                         ("GET", "/jobs")])
def test_admin_endpoints_refuse_anonymous(tmp_path, method, path):
    with make_client(tmp_path) as c:
        assert c.request(method, path).status_code == 403


@pytest.mark.parametrize("method,path", [("GET", "/cache"),
                                         ("DELETE", "/cache"),
                                         ("GET", "/jobs")])
def test_admin_endpoints_accept_the_token(tmp_path, method, path):
    with make_client(tmp_path) as c:
        assert c.request(method, path, headers=ADMIN).status_code == 200


def test_wrong_token_refused(tmp_path):
    with make_client(tmp_path) as c:
        r = c.get("/jobs", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 403


def test_token_must_use_bearer_scheme(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/jobs", headers={"Authorization": "s3cret"}).status_code == 403


def test_admin_disabled_when_no_token_configured(tmp_path):
    """Closed by default: a fresh deploy without a secret exposes nothing,
    and no header value — including an empty bearer — gets in."""
    with make_client(tmp_path, admin_token="") as c:
        assert c.get("/jobs").status_code == 403
        assert c.get("/jobs", headers={"Authorization": "Bearer "}).status_code == 403
        assert "not set" in c.get("/jobs").json()["detail"]


def test_public_endpoints_stay_public(tmp_path):
    with make_client(tmp_path) as c:
        for path in ("/health", "/strategies", "/symbols", "/providers", "/"):
            assert c.get(path).status_code == 200, path


def test_job_readable_by_its_id_without_a_token(tmp_path):
    """The id is the capability: 64 random bits."""
    with make_client(tmp_path) as c:
        job_id = c.post("/backtest", json=BODY).json()["job_id"]
        assert len(job_id) == 16
        assert c.get(f"/backtest/{job_id}").status_code == 200


# ── Rate limiting ──────────────────────────────────────────────────────────

def test_per_client_limit_returns_429_with_retry_after(tmp_path):
    with make_client(tmp_path, rate_limit=3) as c:
        codes = [c.post("/backtest", json=BODY).status_code for _ in range(4)]
        assert codes == [202, 202, 202, 429]
        r = c.post("/backtest", json=BODY)
        assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_global_capacity_cap(tmp_path, monkeypatch):
    with make_client(tmp_path, max_inflight=2) as c:
        monkeypatch.setattr(deps.jobs, "in_flight", lambda: 2)
        r = c.post("/backtest", json=BODY)
        assert r.status_code == 429 and "capacity" in r.json()["detail"]


def test_limiter_window_expires():
    lim = RateLimiter(limit=2, window=0.05)
    assert lim.check("a")[0] and lim.check("a")[0]
    assert not lim.check("a")[0]
    import time
    time.sleep(0.06)
    assert lim.check("a")[0]


def test_limiter_buckets_are_independent():
    lim = RateLimiter(limit=1, window=60)
    assert lim.check("a")[0]
    assert not lim.check("a")[0]
    assert lim.check("b")[0]


def test_limiter_forgets_idle_clients():
    lim = RateLimiter(limit=5, window=0.01)
    for i in range(50):
        lim.check(f"visitor-{i}")
    import time
    time.sleep(0.02)
    lim.check("trigger-sweep")
    assert len(lim._hits) == 1


# ── Client identification ──────────────────────────────────────────────────

def request_with(xff=None, peer="10.0.0.1"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_socket_address_used_when_no_proxy_trusted(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=0, cache_root=str(tmp_path))
    assert client_key(request_with("1.2.3.4")) == "10.0.0.1"


def test_rightmost_trusted_entry_used_behind_one_proxy(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=1, cache_root=str(tmp_path))
    # Client claimed 6.6.6.6; the proxy appended what it really saw.
    assert client_key(request_with("6.6.6.6, 203.0.113.9")) == "203.0.113.9"


def test_spoofed_leftmost_entry_cannot_choose_a_bucket(tmp_path):
    """Taking the left-most entry would let a client rotate fake addresses
    to dodge the limit. Only the proxy-appended entry counts."""
    deps.reset_for_tests(trust_proxy_hops=1, cache_root=str(tmp_path))
    a = client_key(request_with("1.1.1.1, 203.0.113.9"))
    b = client_key(request_with("2.2.2.2, 203.0.113.9"))
    assert a == b == "203.0.113.9"


def test_short_chain_falls_back_to_socket(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=2, cache_root=str(tmp_path))
    assert client_key(request_with("203.0.113.9")) == "10.0.0.1"
