"""Frontend and equity-series tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from api import deps
    deps.reset_for_tests(provider="local", data_root="tests/fixtures",
                         cache_root=str(tmp_path_factory.mktemp("cache")),
        admin_token="test-admin", rate_limit=10_000, max_inflight=1_000)
    from api.main import app
    with TestClient(app) as c:
        yield c


def finished(client, payload):
    jid = client.post("/backtest", json=payload).json()["job_id"]
    for _ in range(1200):
        got = client.get(f"/backtest/{jid}").json()
        if got["status"] in ("done", "failed"):
            return jid, got
    raise AssertionError("never finished")


# ── Static assets ──────────────────────────────────────────────────────────

def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "quantcave" in r.text


def test_css_and_js_served(client):
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_index_references_its_assets(client):
    html = client.get("/").text
    assert "/static/app.css" in html and "/static/app.js" in html


def test_page_has_every_result_region(client):
    html = client.get("/").text
    for anchor in ('id="verdict-badge"', 'id="stats"', 'id="chart"',
                   'id="gates"', 'id="robust"', 'id="folds"', 'id="overfit"'):
        assert anchor in html, anchor


def test_cash_input_default_is_valid_for_its_step(client):
    """Regression: min=1 with step=100 made the default 1000 invalid, and the
    browser silently refused to submit the form."""
    html = client.get("/").text
    line = next(l for l in html.splitlines() if 'id="cash"' in l)
    assert 'step="any"' in line or 'min="0"' in line


# ── Equity series ──────────────────────────────────────────────────────────

def test_summary_response_excludes_the_series(client):
    """It dwarfs the summary and only the chart needs it."""
    _, got = finished(client, {"symbol": "SPY", "kind": "trend_following"})
    assert "_series" not in got["result"]


def test_equity_endpoint_returns_aligned_series(client):
    jid, got = finished(client, {"symbol": "SPY", "kind": "trend_following"})
    assert got["status"] == "done"
    s = client.get(f"/backtest/{jid}/equity").json()
    assert set(s) == {"t", "strategy", "benchmark"}
    assert len(s["t"]) == len(s["benchmark"]) > 0
    assert all(isinstance(x, str) for x in s["t"][:3])


def test_benchmark_series_matches_reported_return(client):
    """The chart and the stat tiles must not disagree."""
    jid, got = finished(client, {"symbol": "SPY", "kind": "momentum",
                                 "cash": 1000})
    s = client.get(f"/backtest/{jid}/equity").json()
    charted = (s["benchmark"][-1] - 1000) / 1000 * 100
    assert charted == pytest.approx(got["result"]["benchmark_return_pct"],
                                    abs=0.01)


def test_equity_404_for_unknown_job(client):
    assert client.get("/backtest/nope/equity").status_code == 404


def test_equity_409_while_job_unfinished(client):
    jid = client.post("/backtest",
                      json={"symbol": "SPY", "kind": "momentum"}).json()["job_id"]
    r = client.get(f"/backtest/{jid}/equity")
    assert r.status_code in (200, 409)   # may already have finished
