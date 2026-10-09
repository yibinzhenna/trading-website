"""
API tests.

Every one runs against the local fixture provider through a temp cache, so
the suite stays hermetic: no network, no API key, no shared state between
runs.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from api import deps
    deps.reset_for_tests(
        provider="local",
        data_root="tests/fixtures",
        cache_root=str(tmp_path_factory.mktemp("cache")),
    )
    from api.main import app
    with TestClient(app) as c:
        yield c


def run_to_completion(client, payload, timeout=60):
    """Submit and poll, the way a real client would."""
    r = client.post("/backtest", json=payload)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    for _ in range(timeout * 20):
        got = client.get(f"/backtest/{job_id}").json()
        if got["status"] in ("done", "failed"):
            return got
    raise AssertionError("job never finished")


# ── Metadata ───────────────────────────────────────────────────────────────

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider"] == "local"


def test_strategies_lists_every_kind_with_defaults(client):
    body = client.get("/strategies").json()
    kinds = {s["kind"] for s in body["strategies"]}
    assert kinds == {"momentum", "mean_reversion", "trend_following",
                     "breakout", "volatility"}
    momentum = next(s for s in body["strategies"] if s["kind"] == "momentum")
    assert set(momentum["params"]) == {"lookback", "move_pct"}


def test_providers_reports_local_available(client):
    names = {p["name"]: p for p in client.get("/providers").json()}
    assert names["local"]["available"] is True


def test_openapi_schema_builds(client):
    assert client.get("/openapi.json").status_code == 200


# ── Submit / poll ──────────────────────────────────────────────────────────

def test_submit_returns_202_and_a_job_id(client):
    r = client.post("/backtest", json={"symbol": "SPY",
                                       "kind": "trend_following"})
    assert r.status_code == 202
    assert r.json()["status"] in ("queued", "running")


def test_full_backtest_produces_a_graded_result(client):
    got = run_to_completion(client, {
        "symbol": "SPY", "kind": "trend_following",
        "params": {"fast": 10, "slow": 30},
        "cost_model": {"slippage_bps": 10},
    })
    assert got["status"] == "done", got.get("error")
    res = got["result"]
    for key in ("sharpe", "max_drawdown_pct", "excess_return_pct", "passed",
                "checks", "out_of_sample", "consistency", "likely_overfit"):
        assert key in res
    assert res["symbol"] == "SPY" and res["provider"] == "local"
    assert all({"name", "passed", "detail"} <= set(c) for c in res["checks"])


def test_equity_curves_are_stripped_from_the_response(client):
    """They are large and no summary view uses them."""
    res = run_to_completion(client, {"symbol": "SPY",
                                     "kind": "momentum"})["result"]
    assert "equity" not in res
    assert "equity" not in res["out_of_sample"]
    assert all("equity" not in f for f in res["folds"])


def test_custom_criteria_change_the_verdict(client):
    base = {"symbol": "SPY", "kind": "trend_following"}
    loose = run_to_completion(client, base)["result"]
    strict = run_to_completion(
        client, dict(base, criteria={"min_sharpe": 50}))["result"]

    def sharpe_ok(res):
        return next(c["passed"] for c in res["checks"]
                    if c["name"].startswith("Sharpe"))

    assert sharpe_ok(loose) and not sharpe_ok(strict)


def test_unknown_job_is_404(client):
    assert client.get("/backtest/nope").status_code == 404


def test_jobs_listing(client):
    run_to_completion(client, {"symbol": "SPY", "kind": "breakout"})
    jobs = client.get("/jobs").json()
    assert jobs and jobs[0]["kind"] == "backtest"


# ── Validation: reject before queueing ─────────────────────────────────────

def test_unknown_kind_rejected(client):
    r = client.post("/backtest", json={"symbol": "SPY", "kind": "astrology"})
    assert r.status_code == 422


def test_unknown_param_rejected_not_ignored(client):
    """A typo'd parameter must not silently run with defaults."""
    r = client.post("/backtest", json={
        "symbol": "SPY", "kind": "momentum", "params": {"lookbak": 10}})
    assert r.status_code == 422
    assert "lookbak" in r.text


def test_fast_above_slow_rejected(client):
    r = client.post("/backtest", json={
        "symbol": "SPY", "kind": "trend_following",
        "params": {"fast": 50, "slow": 10}})
    assert r.status_code == 422


def test_bad_symbol_rejected(client):
    r = client.post("/backtest", json={"symbol": "../etc/passwd",
                                       "kind": "momentum"})
    assert r.status_code == 422


def test_negative_cash_rejected(client):
    r = client.post("/backtest", json={"symbol": "SPY", "kind": "momentum",
                                       "cash": -5})
    assert r.status_code == 422


def test_symbol_is_normalised_to_upper(client):
    res = run_to_completion(client, {"symbol": "spy",
                                     "kind": "momentum"})["result"]
    assert res["symbol"] == "SPY"


# ── Failures surface as a failed job, not a 500 ────────────────────────────

def test_insufficient_bars_fails_the_job_with_a_clear_message(client):
    # TINY has 10 bars, below the 60-bar minimum.
    got = run_to_completion(client, {"symbol": "TINY", "kind": "momentum"})
    assert got["status"] == "failed"
    assert "bars" in got["error"].lower()


def test_unknown_symbol_fails_cleanly(client):
    got = run_to_completion(client, {"symbol": "NOSUCH", "kind": "momentum"})
    assert got["status"] == "failed" and "bars" in got["error"].lower()


# ── Cache ──────────────────────────────────────────────────────────────────

def test_cache_populated_after_a_run(client):
    run_to_completion(client, {"symbol": "SPY", "kind": "momentum"})
    assert client.get("/cache").json()["entries"] >= 1


def test_cache_can_be_cleared(client):
    run_to_completion(client, {"symbol": "SPY", "kind": "momentum"})
    assert client.request("DELETE", "/cache").json()["removed"] >= 1
