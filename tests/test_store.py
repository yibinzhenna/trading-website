"""
Run persistence: results outlive the process that computed them.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api import deps
from api.store import RunStore, normalise_url
from quantlab.jobs import JobStore

BODY = {"symbol": "SPY", "kind": "trend_following",
        "params": {"fast": 10, "slow": 30}}


def make_client(tmp_path, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), admin_token="",
                    rate_limit=10_000, max_inflight=1_000,
                    database_url=f"sqlite:///{(tmp_path / 'runs.db').as_posix()}")
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    from api.main import app
    return TestClient(app)


def finish(client, payload):
    job_id = client.post("/backtest", json=payload).json()["job_id"]
    job = deps.jobs.wait(job_id, timeout=60)
    assert job.status in ("done", "failed")
    return job_id


def restart():
    """What a redeploy does to memory: a fresh job store, same database."""
    deps.jobs.shutdown(wait=True)
    deps.jobs = JobStore(workers=1, on_finish=deps._persist)


# ── Through the API ────────────────────────────────────────────────────────

def test_result_survives_a_restart(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c, BODY)
        before = c.get(f"/backtest/{job_id}").json()
        restart()
        after = c.get(f"/backtest/{job_id}").json()
    assert after["status"] == "done"
    assert after["result"] == before["result"]
    assert after["meta"]["persisted"] is True


def test_equity_survives_a_restart(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c, BODY)
        before = c.get(f"/backtest/{job_id}/equity").json()
        restart()
        r = c.get(f"/backtest/{job_id}/equity")
    assert r.status_code == 200 and r.json() == before
    assert len(r.json()["strategy"]) == len(r.json()["t"])


def test_saved_run_carries_its_request(tmp_path):
    """A shared link has to reproduce the form, not just the numbers."""
    with make_client(tmp_path) as c:
        job_id = finish(c, dict(BODY, cash=2500))
        restart()
        req = c.get(f"/backtest/{job_id}").json()["meta"]["request"]
    assert req["symbol"] == "SPY" and req["cash"] == 2500
    assert req["params"] == {"fast": 10, "slow": 30}
    assert req["cost_model"]["slippage_bps"] == 10


def test_series_not_duplicated_into_the_summary(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c, BODY)
        restart()
        res = c.get(f"/backtest/{job_id}").json()["result"]
    assert "_series" not in res


def test_failed_run_is_kept_without_its_traceback(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c, {"symbol": "TINY", "kind": "momentum"})
        live = c.get(f"/backtest/{job_id}").json()
        restart()
        saved = c.get(f"/backtest/{job_id}").json()
        equity = c.get(f"/backtest/{job_id}/equity")
    for body in (live, saved):
        assert body["status"] == "failed" and "bars" in body["error"]
        assert "traceback" not in body["meta"]
    assert equity.status_code == 404


def test_unknown_run_is_404_after_restart(tmp_path):
    with make_client(tmp_path) as c:
        restart()
        assert c.get("/backtest/0123456789abcdef").status_code == 404
        assert c.get("/backtest/0123456789abcdef/equity").status_code == 404


def test_health_reports_the_database(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/health").json()["database"] == "sqlite"


def test_a_failing_store_does_not_fail_the_job(tmp_path, monkeypatch):
    """The result was computed; losing the copy must not lose the answer."""
    with make_client(tmp_path) as c:
        def broken(job):
            raise RuntimeError("database down")
        monkeypatch.setattr(deps.runs, "save", broken)
        job_id = finish(c, BODY)
        assert c.get(f"/backtest/{job_id}").json()["status"] == "done"


# ── The store itself ───────────────────────────────────────────────────────

class FakeJob:
    def __init__(self, id, submitted_at, status="done"):
        self.id, self.status = id, status
        self.kind = "backtest"
        self.submitted_at = submitted_at
        self.started_at = self.finished_at = submitted_at
        self.duration_sec = 0.1
        self.result = {"passed": False, "_series": {"t": []}}
        self.error = None
        self.meta = {"symbol": "X", "kind": "momentum", "interval": "day",
                     "request": {"symbol": "X"}}


def iso(dt):
    return dt.isoformat(timespec="milliseconds")


def store(tmp_path, **kw):
    return RunStore(f"sqlite:///{(tmp_path / 's.db').as_posix()}", **kw)


def test_save_is_idempotent(tmp_path):
    s = store(tmp_path)
    job = FakeJob("a" * 16, iso(datetime.now(timezone.utc)))
    s.save(job)
    s.save(job)
    assert s.count() == 1


def test_save_does_not_mutate_the_live_result(tmp_path):
    """The in-memory job still serves the chart from `_series`."""
    job = FakeJob("b" * 16, iso(datetime.now(timezone.utc)))
    store(tmp_path).save(job)
    assert "_series" in job.result


def test_retention_prunes_old_runs(tmp_path):
    s = store(tmp_path, retention_days=30)
    now = datetime.now(timezone.utc)
    s.save(FakeJob("old" + "0" * 13, iso(now - timedelta(days=31))))
    s.save(FakeJob("new" + "0" * 13, iso(now - timedelta(days=1))))
    assert s.prune() == 1
    assert s.get("old" + "0" * 13) is None and s.get("new" + "0" * 13)


def test_retention_zero_keeps_everything(tmp_path):
    s = store(tmp_path, retention_days=0)
    s.save(FakeJob("c" * 16, iso(datetime(2000, 1, 1, tzinfo=timezone.utc))))
    assert s.prune() == 0 and s.count() == 1


def test_timestamps_round_trip_with_timezone(tmp_path):
    s = store(tmp_path)
    stamp = "2026-01-02T03:04:05.123+00:00"
    s.save(FakeJob("d" * 16, stamp))
    assert s.get("d" * 16)["submitted_at"] == stamp


def test_owner_listing_is_newest_first_within_a_second(tmp_path):
    """Two quick runs used to share a timestamp to the second, leaving their
    order to the random id."""
    s = store(tmp_path)
    t0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    for i, (ms, run_id) in enumerate([(100, "f" * 16), (900, "0" * 16)]):
        job = FakeJob(run_id, iso(t0 + timedelta(milliseconds=ms)))
        job.meta["owner_id"] = "alice"
        s.save(job)
    assert [r["job_id"] for r in s.list_for_owner("alice")] == ["0" * 16, "f" * 16]
    assert s.list_for_owner("bob") == []


def test_infinite_metrics_save_as_null(tmp_path):
    """No losing trades means an infinite profit factor. Postgres JSON has no
    Infinity; the serializer is strict, so this fails here if it would there."""
    s = store(tmp_path)
    job = FakeJob("e" * 16, iso(datetime.now(timezone.utc)))
    job.result = {"passed": True, "profit_factor": float("inf"),
                  "trade_t": float("nan"), "folds": [{"sharpe": float("-inf")}]}
    s.save(job)
    res = s.get("e" * 16)["result"]
    assert res["profit_factor"] is None and res["trade_t"] is None
    assert res["folds"][0]["sharpe"] is None
    assert job.result["profit_factor"] == float("inf")    # live copy untouched


@pytest.mark.parametrize("url,expected", [
    ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
    ("sqlite:///x.db", "sqlite:///x.db"),
])
def test_url_normalisation(url, expected):
    assert normalise_url(url) == expected


# ── Closing ────────────────────────────────────────────────────────────────

def test_closed_store_refuses_new_work(tmp_path):
    from api.store import StoreUnavailable
    s = store(tmp_path)
    s.count()
    s.close()
    with pytest.raises(StoreUnavailable):
        s.count()
    s.close()                                  # idempotent


def test_close_waits_for_work_in_progress(tmp_path, monkeypatch):
    """A save racing a shutdown used to hand its connection back to a
    disposed pool, leaking it."""
    import threading
    import time
    s = store(tmp_path)
    s.count()
    started, release = threading.Event(), threading.Event()
    original = type(s).count.__wrapped__

    def slow_count(self):
        started.set()
        release.wait(5)
        return original(self)

    from api import store as store_mod
    monkeypatch.setattr(store_mod.RunStore, "count",
                        store_mod._guarded(slow_count))
    worker = threading.Thread(target=s.count)
    worker.start()
    started.wait(5)
    closer = threading.Thread(target=s.close)
    t = time.monotonic()
    closer.start()
    time.sleep(0.2)
    assert closer.is_alive()                   # waiting for the count
    release.set()
    closer.join(5)
    worker.join(5)
    assert not closer.is_alive() and time.monotonic() - t < 5
