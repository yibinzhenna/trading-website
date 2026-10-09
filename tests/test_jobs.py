"""Job store tests."""

import time

from quantlab.jobs import DONE, FAILED, QUEUED, RUNNING, JobStore


def test_submit_returns_immediately_with_an_id():
    store = JobStore(workers=1)
    job = store.submit("test", lambda: time.sleep(0.2) or 42)
    assert job.id and job.status in (QUEUED, RUNNING)
    store.shutdown(wait=True)


def test_result_available_after_completion():
    store = JobStore(workers=1)
    job = store.submit("test", lambda x: x * 2, 21)
    done = store.wait(job.id, timeout=5)
    assert done.status == DONE and done.result == 42
    store.shutdown(wait=True)


def test_failure_is_captured_not_raised():
    store = JobStore(workers=1)

    def boom():
        raise ValueError("deliberate")

    job = store.wait(store.submit("test", boom).id, timeout=5)
    assert job.status == FAILED
    assert "ValueError: deliberate" in job.error
    assert "traceback" in job.meta
    store.shutdown(wait=True)


def test_unknown_job_is_none():
    store = JobStore(workers=1)
    assert store.get("nope") is None
    store.shutdown(wait=True)


def test_duration_is_recorded():
    store = JobStore(workers=1)
    job = store.wait(store.submit("test", lambda: time.sleep(0.05)).id,
                     timeout=5)
    assert job.duration_sec is not None and job.duration_sec >= 0
    store.shutdown(wait=True)


def test_list_is_newest_first_and_omits_results():
    store = JobStore(workers=1)
    ids = [store.submit("test", lambda i=i: i).id for i in range(3)]
    for i in ids:
        store.wait(i, timeout=5)
    listed = store.list()
    assert listed[0]["job_id"] == ids[-1]
    assert "result" not in listed[0]
    store.shutdown(wait=True)


def test_history_is_bounded():
    store = JobStore(workers=2, max_jobs=5)
    for i in range(20):
        store.wait(store.submit("test", lambda i=i: i).id, timeout=5)
    assert len(store.list(limit=100)) <= 5
    store.shutdown(wait=True)


def test_concurrent_jobs_both_complete():
    store = JobStore(workers=2)
    a = store.submit("test", lambda: time.sleep(0.1) or "a")
    b = store.submit("test", lambda: time.sleep(0.1) or "b")
    assert store.wait(a.id, timeout=5).result == "a"
    assert store.wait(b.id, timeout=5).result == "b"
    store.shutdown(wait=True)
