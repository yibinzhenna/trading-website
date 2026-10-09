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
                    trust_proxy_hops=0, client_ip_header="",
                    supabase_url="", supabase_publishable_key="")
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

def request_with(xff=None, peer="10.0.0.1", cf=None):
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    if cf:
        headers.append((b"cf-connecting-ip", cf.encode()))
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_socket_address_used_when_no_proxy_trusted(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=0, client_ip_header="", cache_root=str(tmp_path))
    assert client_key(request_with("1.2.3.4")) == "10.0.0.1"


def test_rightmost_trusted_entry_used_behind_one_proxy(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=1, client_ip_header="", cache_root=str(tmp_path))
    # Client claimed 6.6.6.6; the proxy appended what it really saw.
    assert client_key(request_with("6.6.6.6, 203.0.113.9")) == "203.0.113.9"


def test_spoofed_leftmost_entry_cannot_choose_a_bucket(tmp_path):
    """Taking the left-most entry would let a client rotate fake addresses
    to dodge the limit. Only the proxy-appended entry counts."""
    deps.reset_for_tests(trust_proxy_hops=1, client_ip_header="", cache_root=str(tmp_path))
    a = client_key(request_with("1.1.1.1, 203.0.113.9"))
    b = client_key(request_with("2.2.2.2, 203.0.113.9"))
    assert a == b == "203.0.113.9"


def test_short_chain_falls_back_to_socket(tmp_path):
    deps.reset_for_tests(trust_proxy_hops=2, client_ip_header="", cache_root=str(tmp_path))
    assert client_key(request_with("203.0.113.9")) == "10.0.0.1"


# ── Edge-set client header (Render / Cloudflare) ───────────────────────────

def test_edge_header_preferred_over_forwarded_for(tmp_path):
    deps.reset_for_tests(client_ip_header="cf-connecting-ip",
                         trust_proxy_hops=1, cache_root=str(tmp_path))
    r = request_with("6.6.6.6, 198.51.100.7, 10.1.1.1", cf="203.0.113.9")
    assert client_key(r) == "203.0.113.9"


def test_forwarded_for_spoof_cannot_move_the_bucket_with_edge_header(tmp_path):
    """The live failure: a fake X-Forwarded-For escaped the limit. With the
    edge header configured, whatever the client writes there is ignored."""
    deps.reset_for_tests(client_ip_header="cf-connecting-ip",
                         trust_proxy_hops=0, cache_root=str(tmp_path))
    a = client_key(request_with("9.9.9.9, edge, lb", cf="203.0.113.9"))
    b = client_key(request_with("1.2.3.4, edge2, lb2", cf="203.0.113.9"))
    assert a == b == "203.0.113.9"


def test_rotating_proxy_pool_does_not_split_one_client(tmp_path):
    """hops=1 on Render picked a rotating infrastructure address, so one
    client got a fresh bucket per request. The edge header is stable."""
    deps.reset_for_tests(client_ip_header="cf-connecting-ip",
                         trust_proxy_hops=0, cache_root=str(tmp_path))
    keys = {client_key(request_with(f"203.0.113.9, edge, lb-{i}",
                                    cf="203.0.113.9")) for i in range(10)}
    assert keys == {"203.0.113.9"}


def test_missing_edge_header_falls_back_to_socket(tmp_path):
    deps.reset_for_tests(client_ip_header="cf-connecting-ip",
                         trust_proxy_hops=0, cache_root=str(tmp_path))
    assert client_key(request_with("1.2.3.4")) == "10.0.0.1"


def test_limit_holds_end_to_end_via_edge_header(tmp_path):
    with make_client(tmp_path, rate_limit=2,
                     client_ip_header="cf-connecting-ip") as c:
        h = {"CF-Connecting-IP": "203.0.113.9"}
        codes = [c.post("/backtest", json=BODY,
                        headers=dict(h, **{"X-Forwarded-For": f"{i}.{i}.{i}.{i}"})
                        ).status_code for i in range(1, 4)]
        assert codes == [202, 202, 429]


# ── Browser hardening ──────────────────────────────────────────────────────
# Sessions live in localStorage, readable by any script on the page, so
# controlling which scripts can run is what protects accounts.

def test_security_headers_on_every_page(tmp_path):
    with make_client(tmp_path) as c:
        for path in ("/", "/health", "/static/app.js"):
            h = c.get(path).headers
            assert "frame-ancestors 'none'" in h["content-security-policy"], path
            assert h["x-content-type-options"] == "nosniff"
            assert h["x-frame-options"] == "DENY"
            assert h["referrer-policy"] == "no-referrer"


def test_csp_allows_no_inline_code_or_eval(tmp_path):
    with make_client(tmp_path) as c:
        csp = c.get("/").headers["content-security-policy"]
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "script-src 'self' https://cdn.jsdelivr.net" in csp


def test_csp_lets_the_page_reach_supabase_only_when_configured(tmp_path):
    with make_client(tmp_path) as c:
        assert "connect-src 'self';" in c.get("/").headers["content-security-policy"]
        deps.settings.supabase_url = "https://proj.supabase.co"
        csp = c.get("/").headers["content-security-policy"]
    assert "connect-src 'self' https://proj.supabase.co;" in csp


def test_page_has_no_inline_code_and_pins_external_scripts(tmp_path):
    """The CSP would block inline code anyway; this catches it before a
    deploy does. Every CDN script must carry an integrity hash."""
    import re
    from pathlib import Path
    html = Path("api/templates/index.html").read_text(encoding="utf-8")
    assert "style=" not in html and "<style" not in html
    assert not re.search(r"\son[a-z]+=", html)
    for tag in re.findall(r"<script[^>]*>", html):
        assert "src=" in tag, f"inline script: {tag}"
        if "https://" in tag:
            assert 'integrity="sha384-' in tag and "crossorigin" in tag, tag
    js = Path("api/static/account.js").read_text(encoding="utf-8")
    assert "s.integrity = SDK_INTEGRITY" in js


def test_hsts_only_over_https(tmp_path):
    with make_client(tmp_path) as c:
        assert "strict-transport-security" not in c.get("/").headers
        h = c.get("/", headers={"X-Forwarded-Proto": "https"}).headers
    assert h["strict-transport-security"].startswith("max-age=31536000")


def test_account_responses_are_not_cached(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/me").headers["cache-control"] == "no-store"
        assert "cache-control" not in c.get("/strategies").headers


def test_other_sites_cannot_call_the_api_from_a_browser(tmp_path):
    """No CORS by default: a foreign page cannot use its visitors' browsers
    to run backtests or read responses."""
    with make_client(tmp_path) as c:
        r = c.options("/backtest", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        get = c.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    assert "access-control-allow-origin" not in get.headers


def test_job_ids_are_64_random_bits():
    from quantlab.jobs import Job
    ids = {Job("t").id for _ in range(2000)}
    assert len(ids) == 2000
    assert all(len(i) == 16 and int(i, 16) >= 0 for i in ids)
    # A truncated uuid4 always had '4' at position 12.
    assert len({i[12] for i in ids}) > 1
