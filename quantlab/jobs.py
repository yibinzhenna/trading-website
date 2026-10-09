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

import logging
import secrets
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

log = logging.getLogger("quantlab.jobs")

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
TERMINAL = (DONE, FAILED)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Job:
    __slots__ = ("id", "kind", "status", "result", "error", "submitted_at",
                 "started_at", "finished_at", "meta")

    def __init__(self, kind, meta=None, job_id=None):
        # 64 random bits. The id is the only key to a result, so it must be
        # unguessable; a truncated uuid4 spends 4 of these on its version.
        self.id = job_id or secrets.token_hex(8)
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

    def __init__(self, workers=2, max_jobs=500, on_finish=None):
        self._jobs = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="job")
        self.max_jobs = max_jobs
        # Called with each job once it is terminal, on the worker thread.
        # Persistence hooks in here; its failures are logged, never raised,
        # because a result that could not be saved is still a result.
        self.on_finish = on_finish

    def submit(self, kind, fn, *args, meta=None, job_id=None, **kwargs):
        """Queue `fn(*args, **kwargs)`. `job_id` lets a caller record the
        job elsewhere (a database row, say) before it can possibly finish."""
        job = Job(kind, meta, job_id)
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
        if self.on_finish is not None:
            try:
                self.on_finish(job)
            except Exception:
                log.exception("on_finish failed for job %s", job.id)

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def forget(self, job_id):
        """Drop a terminal job from memory. Running jobs are kept."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status in TERMINAL:
                del self._jobs[job_id]
                return True
        return False

    def list(self, limit=50):
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.to_dict(include_result=False) for j in reversed(jobs)][:limit]

    def in_flight(self):
        """Jobs queued or running, across the whole store."""
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j.status not in TERMINAL)

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
