"""
Background job runner.

Backtests are CPU-bound and slow — a walk-forward across four folds plus a
parameter sweep takes real seconds to minutes. They cannot run inside a
request handler.

The shape that matters is submit -> id -> poll, because retrofitting that
onto a synchronous API is painful. The thing that runs the work is not
important yet, so this uses a thread pool: no Redis, no broker, no Docker
for a developer who just wants to run the server.

`JobStore` is the seam. Swapping in arq or Celery later means writing one
class with the same four methods and changing the wiring — the API and the
frontend never notice.
"""

import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
TERMINAL = (DONE, FAILED)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    __slots__ = ("id", "kind", "status", "result", "error", "submitted_at",
                 "started_at", "finished_at", "meta")

    def __init__(self, kind, meta=None):
        self.id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.status = QUEUED
        self.result = None
        self.error = None
        self.submitted_at = _now()
        self.started_at = None
        self.finished_at = None
        self.meta = meta or {}

    @property
    def duration_sec(self):
        if not self.started_at:
            return None
        end = self.finished_at or _now()
        return round(
            (datetime.fromisoformat(end)
             - datetime.fromisoformat(self.started_at)).total_seconds(), 3)

    def to_dict(self, include_result=True):
        out = {"job_id": self.id, "kind": self.kind, "status": self.status,
               "submitted_at": self.submitted_at,
               "started_at": self.started_at,
               "finished_at": self.finished_at,
               "duration_sec": self.duration_sec,
               "meta": self.meta}
        if self.status == FAILED:
            out["error"] = self.error
        if include_result and self.status == DONE:
            out["result"] = self.result
        return out


class JobStore:
    """Thread-pool backed job store with a bounded history.

    Completed jobs are kept so a client can poll after the fact, but only the
    most recent `max_jobs` — an unbounded dict is a memory leak with a slow
    fuse.
    """

    def __init__(self, workers=2, max_jobs=500):
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="job")
        self.max_jobs = max_jobs

    def submit(self, kind, fn, *args, meta=None, **kwargs):
        job = Job(kind, meta)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self.max_jobs:
                oldest, old = next(iter(self._jobs.items()))
                if old.status not in TERMINAL:
                    break          # never evict work still in flight
                self._jobs.pop(oldest)
        self._pool.submit(self._run, job, fn, args, kwargs)
        return job

    def _run(self, job, fn, args, kwargs):
        job.started_at = _now()
        job.status = RUNNING
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            # The message is for the client; the traceback is for the log.
            job.error = f"{type(e).__name__}: {e}"
            job.meta["traceback"] = traceback.format_exc(limit=6)
            job.finished_at = _now()
            job.status = FAILED
        else:
            job.result = result
            job.finished_at = _now()
            job.status = DONE
        # `status` is written last on both paths, deliberately. Pollers key
        # off it, so flipping it before result/error are populated lets a
        # client observe a terminal job with missing fields.

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit=50):
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.to_dict(include_result=False) for j in reversed(jobs)][:limit]

    def wait(self, job_id, timeout=30, poll=0.05):
        """Block until terminal. For tests and synchronous callers only —
        the HTTP layer polls instead."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.get(job_id)
            if job is None:
                return None
            if job.status in TERMINAL:
                return job
            time.sleep(poll)
        return self.get(job_id)

    def shutdown(self, wait=False):
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
