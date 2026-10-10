"""
Run persistence.

The job store is process memory: a restart, a redeploy or the 500-job
history cap forgets every result. This keeps finished backtests in a
database so a result outlives the process that computed it, which is what
makes a result link worth sharing.

One table, SQLAlchemy Core, two dialects:

* SQLite by default — zero setup for development and tests.
* Postgres when `DATABASE_URL` is set — the deploy. Render, Neon and Supabase
  hand out `postgres://` URLs; they are normalised to the psycopg 3 driver.

The schema is created on startup. When it first changes (users, with auth),
that is the point to bring in Alembic, not before.
"""

import functools
import json
import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, Integer,
                        MetaData, String, Table, Text, create_engine, delete,
                        func, insert, or_, select, update)
from sqlalchemy.exc import InterfaceError, OperationalError

log = logging.getLogger("quantlab.store")

metadata = MetaData()

runs = Table(
    "runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("status", String(16), nullable=False),
    Column("symbol", String(16), nullable=False, index=True),
    Column("kind", String(32), nullable=False),
    Column("bar_interval", String(8), nullable=False),
    Column("passed", Boolean),
    Column("submitted_at", DateTime(timezone=True), nullable=False, index=True),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("request", JSON, nullable=False),
    Column("result", JSON),          # summary, as GET /backtest returns it
    Column("series", JSON),          # equity curves, as GET .../equity returns it
    Column("error", Text),
    Column("duration_sec", Float),
    # Filled once accounts exist. Present now so adding auth does not need
    # a migration on a table already holding data.
    Column("owner_id", String(64), index=True),
)

# AI research sessions. A row is written when a session is *accepted*, not
# when it finishes, so quotas count work in flight and survive a restart.
research = Table(
    "research", metadata,
    Column("id", String(32), primary_key=True),
    Column("owner_id", String(64), nullable=False, index=True),
    Column("status", String(16), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("goal", Text),
    Column("model", String(64)),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("finished_at", DateTime(timezone=True)),
    Column("state", JSON),           # trials, holdout verdict, notes
    Column("error", Text),
    Column("input_tokens", Integer, nullable=False, default=0),
    Column("output_tokens", Integer, nullable=False, default=0),
)

TABLES = ("runs", "research")
IN_FLIGHT = ("queued", "running")


# Give up on an unreachable database after this long, and then stop trying
# for RETRY_AFTER seconds so every request does not wait out the timeout.
CONNECT_TIMEOUT = 5
RETRY_AFTER = 15
_CONNECTIVITY = (OperationalError, InterfaceError)


class StoreUnavailable(Exception):
    """The database cannot be reached right now. Features that need saved
    data fail with this; everything else — running backtests — carries on."""


def _guarded(fn):
    """Make the schema exist before first use, fail fast while the database
    is known to be down, and turn connection failures into StoreUnavailable.
    Other database errors are bugs and propagate unchanged."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._busy:
            if self._closed:
                raise StoreUnavailable("the store is closed")
            self._active += 1
        try:
            self._ensure()
            return fn(self, *args, **kwargs)
        except _CONNECTIVITY as e:
            self._trip(e)
            raise StoreUnavailable("the database is unreachable") from e
        finally:
            with self._busy:
                self._active -= 1
                self._busy.notify_all()
    return wrapper


def normalise_url(url):
    """Point Postgres URLs at the psycopg 3 driver SQLAlchemy should use."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def _finite(obj):
    """Non-finite floats become None, recursively.

    The engine reports an unbeaten profit factor or a zero-variance t-stat as
    infinity. Postgres JSON rejects `Infinity` outright, so the row would not
    save. None is also what the API already sends for these (JSON has no
    infinity), so a saved run reads back exactly as the live one did.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _ts(value):
    return datetime.fromisoformat(value) if value else None


def _iso(value):
    if value is None:
        return None
    if value.tzinfo is None:          # SQLite drops the offset on the way out
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat(timespec="milliseconds")


class RunStore:
    """Finished backtests, keyed by job id."""

    def __init__(self, url, retention_days=30, prune_every=100):
        """Does not touch the database. The schema is set up on first use, or
        by `warm()` in the background at startup: a server must be able to
        start, and serve backtests, while its database is down. It used to
        connect here with no timeout, so an unreachable database held
        startup for over two minutes and then crashed it."""
        url = normalise_url(url)
        # Strict on every dialect, so SQLite in tests rejects what Postgres
        # would reject in production instead of quietly storing `Infinity`.
        kwargs = {"pool_pre_ping": True,
                  "json_serializer": lambda o: json.dumps(o, allow_nan=False)}
        if url.startswith("sqlite"):
            # Writes come from job threads, reads from request threads.
            kwargs["connect_args"] = {"check_same_thread": False}
        else:
            kwargs["connect_args"] = {"connect_timeout": CONNECT_TIMEOUT}
            if ":6543/" in url:
                # Supabase's transaction pooler hands each transaction to a
                # different backend, so server-side prepared statements
                # break. The session pooler (5432) is the right choice for a
                # long-lived server; this keeps the wrong one from failing
                # obscurely.
                kwargs["connect_args"]["prepare_threshold"] = None
        self.engine = create_engine(url, **kwargs)
        self.retention_days = retention_days
        self.prune_every = prune_every
        self._saves = 0
        self._ready = False
        self._init_lock = threading.Lock()
        self._down_until = 0.0
        self._health = ("unknown", float("-inf"))
        self._probing = threading.Lock()
        self._closed = False
        # Operations in progress, so close() can wait for them.
        self._busy = threading.Condition()
        self._active = 0

    # ── Availability ──────────────────────────────────────────────────────

    def _ensure(self):
        """Fast-fail while tripped; otherwise create the schema once."""
        if self._closed:
            raise StoreUnavailable("the store is closed")
        if time.monotonic() < self._down_until:
            raise StoreUnavailable("the database is unreachable")
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            # Another thread may have just failed while this one waited.
            if time.monotonic() < self._down_until:
                raise StoreUnavailable("the database is unreachable")
            try:
                metadata.create_all(self.engine)
                self._lock_down()
                RunStore.research_mark_interrupted.__wrapped__(self)
                RunStore.prune.__wrapped__(self)
            except _CONNECTIVITY as e:
                self._trip(e)
                raise StoreUnavailable("the database is unreachable") from e
            self._ready = True
            log.info("run store ready (%s)", self.dialect)

    def _trip(self, error):
        already = time.monotonic() < self._down_until
        self._down_until = time.monotonic() + RETRY_AFTER
        if not already:
            log.warning("database unreachable; retrying in %ss: %s",
                        RETRY_AFTER, str(error).splitlines()[0][:200])

    def warm(self):
        """Set up the schema now if possible. For a background thread at
        startup; never raises."""
        try:
            self._ensure()
        except StoreUnavailable:
            pass

    def health(self, max_age=10.0):
        """The last known status — "ok", "unavailable" or "unknown" — returned
        immediately. A stale status is refreshed in the background. A health
        endpoint that waited on a connection timeout could itself time out,
        and the host would restart a server still serving backtests."""
        status, at = self._health
        if time.monotonic() - at >= max_age and self._probing.acquire(blocking=False):
            def probe():
                try:
                    self.check()
                finally:
                    self._probing.release()
            threading.Thread(target=probe, name="store-probe", daemon=True).start()
        return status

    def check(self):
        """Probe the database now and record the result. Blocks for up to
        the connect timeout; for background use and tests."""
        try:
            self._ensure()
            with self.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            status = "ok"
        except StoreUnavailable:
            status = "unavailable"
        except _CONNECTIVITY as e:
            self._trip(e)
            status = "unavailable"
        self._health = (status, time.monotonic())
        return status

    def _lock_down(self):
        """Close the table to Supabase's auto-generated REST API.

        Supabase exposes every table in `public` over HTTP, reachable with
        the publishable key, which ships to every browser. Without row-level
        security that is the whole runs table, readable and writable by
        anyone. RLS with no policies denies those roles everything; this
        server connects as the table owner and is unaffected.
        """
        if self.dialect != "postgresql":
            return
        for table in TABLES:
            try:
                with self.engine.begin() as conn:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
            except Exception:
                log.exception("could not enable row-level security on %s; on "
                              "Supabase the table may be publicly reachable",
                              table)

    @property
    def dialect(self):
        return self.engine.dialect.name

    @_guarded
    def save(self, job, status=None):
        """Record a terminal job. Idempotent: saving twice keeps the first.

        The request comes from `job.meta["request"]`; the equity curves come
        out of the result's private `_series` key and go to their own column,
        so a summary read never drags them along.
        """
        result = dict(job.result) if isinstance(job.result, dict) else None
        series = result.pop("_series", None) if result else None
        meta = {k: v for k, v in job.meta.items() if k != "traceback"}
        row = {
            "id": job.id,
            # Passed in by the job store, which saves before it publishes.
            "status": status or job.status,
            "symbol": meta.get("symbol", ""),
            "kind": meta.get("kind", ""),
            "bar_interval": meta.get("interval", "day"),
            "passed": (result or {}).get("passed"),
            "submitted_at": _ts(job.submitted_at),
            "started_at": _ts(job.started_at),
            "finished_at": _ts(job.finished_at),
            "request": _finite(meta.get("request") or {}),
            "result": _finite(result),
            "series": _finite(series),
            "error": job.error,
            "duration_sec": job.duration_sec,
            # A column, never echoed back to whoever holds the link.
            "owner_id": meta.get("owner_id"),
        }
        with self.engine.begin() as conn:
            if conn.execute(select(runs.c.id).where(runs.c.id == job.id)).first():
                return
            conn.execute(insert(runs).values(**row))
        self._saves += 1
        if self.prune_every and self._saves % self.prune_every == 0:
            self.prune()

    @_guarded
    def get(self, run_id):
        """The run in the same shape as JobStore's `to_dict()`, or None."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(runs).where(runs.c.id == run_id)).mappings().first()
        if row is None:
            return None
        out = {
            "job_id": row["id"], "kind": "backtest", "status": row["status"],
            "submitted_at": _iso(row["submitted_at"]),
            "started_at": _iso(row["started_at"]),
            "finished_at": _iso(row["finished_at"]),
            "duration_sec": row["duration_sec"],
            "meta": {"symbol": row["symbol"], "kind": row["kind"],
                     "interval": row["bar_interval"], "request": row["request"],
                     "persisted": True},
        }
        if row["status"] == "failed":
            out["error"] = row["error"]
        else:
            out["result"] = row["result"]
        return out

    @_guarded
    def owner_runs_since(self, owner_id, since):
        """(id, submitted_at ISO) of an owner's runs since `since`."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(runs.c.id, runs.c.submitted_at).where(
                    runs.c.owner_id == owner_id,
                    runs.c.submitted_at >= since)).all()
        return [(r[0], _iso(r[1])) for r in rows]

    @_guarded
    def list_for_owner(self, owner_id, limit=50):
        """A user's runs, newest first, summarised for a listing."""
        cols = (runs.c.id, runs.c.status, runs.c.symbol, runs.c.kind,
                runs.c.passed, runs.c.submitted_at, runs.c.result,
                runs.c.error)
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(*cols).where(runs.c.owner_id == owner_id)
                .order_by(runs.c.submitted_at.desc(), runs.c.id)
                .limit(limit)).mappings().all()
        out = []
        for r in rows:
            res = r["result"] or {}
            out.append({
                "job_id": r["id"], "status": r["status"],
                "symbol": r["symbol"], "kind": r["kind"],
                "passed": r["passed"],
                "submitted_at": _iso(r["submitted_at"]),
                "total_return_pct": res.get("total_return_pct"),
                "excess_return_pct": res.get("excess_return_pct"),
                "oos_sharpe": res.get("oos_sharpe"),
                "error": r["error"],
            })
        return out

    @_guarded
    def delete(self, run_id, owner_id):
        """Delete a run if, and only if, it belongs to `owner_id`."""
        with self.engine.begin() as conn:
            return conn.execute(
                delete(runs).where(runs.c.id == run_id,
                                   runs.c.owner_id == owner_id)).rowcount > 0

    @_guarded
    def series(self, run_id):
        with self.engine.connect() as conn:
            return conn.execute(
                select(runs.c.series).where(runs.c.id == run_id)).scalar()

    @_guarded
    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(runs)).scalar()

    @_guarded
    def prune(self):
        """Delete runs past retention. Free Postgres tiers are a few hundred
        MB, and one run with its curves is tens of KB."""
        if not self.retention_days:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self.engine.begin() as conn:
            removed = conn.execute(
                delete(runs).where(runs.c.submitted_at < cutoff)).rowcount
            removed += conn.execute(
                delete(research).where(research.c.created_at < cutoff,
                                       research.c.status.notin_(IN_FLIGHT))
            ).rowcount
        if removed:
            log.info("pruned %d runs older than %d days",
                     removed, self.retention_days)
        return removed

    # ── Research sessions ─────────────────────────────────────────────────

    @_guarded
    def research_create(self, run_id, owner_id, symbol, goal, model):
        with self.engine.begin() as conn:
            conn.execute(insert(research).values(
                id=run_id, owner_id=owner_id, status="queued", symbol=symbol,
                goal=goal, model=model, created_at=datetime.now(timezone.utc),
                input_tokens=0, output_tokens=0))

    @_guarded
    def research_finish(self, run_id, status, state=None, error=None):
        state = _finite(state) if state else None
        with self.engine.begin() as conn:
            conn.execute(update(research).where(research.c.id == run_id).values(
                status=status, state=state, error=error,
                finished_at=datetime.now(timezone.utc),
                input_tokens=(state or {}).get("input_tokens", 0),
                output_tokens=(state or {}).get("output_tokens", 0)))

    @_guarded
    def research_set_status(self, run_id, status):
        with self.engine.begin() as conn:
            conn.execute(update(research).where(research.c.id == run_id)
                         .values(status=status))

    @_guarded
    def research_get(self, run_id):
        with self.engine.connect() as conn:
            row = conn.execute(select(research).where(
                research.c.id == run_id)).mappings().first()
        return None if row is None else self._research_out(row)

    @_guarded
    def research_list(self, owner_id, limit=20):
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(research).where(research.c.owner_id == owner_id)
                .order_by(research.c.created_at.desc()).limit(limit)
            ).mappings().all()
        out = []
        for r in rows:
            final = (r["state"] or {}).get("final") or {}
            out.append({"job_id": r["id"], "status": r["status"],
                        "symbol": r["symbol"],
                        "created_at": _iso(r["created_at"]),
                        "trials": len((r["state"] or {}).get("trials", [])),
                        "passed": final.get("passed"),
                        "pick": final.get("kind")})
        return out

    @_guarded
    def research_usage(self, since, owner_id=None):
        """Sessions that count against a quota since `since`.

        A session that failed before spending a token (the model was down,
        say) is not charged: the user got nothing for it.
        """
        q = select(func.count()).select_from(research).where(
            research.c.created_at >= since,
            or_(research.c.status != "failed", research.c.input_tokens > 0))
        if owner_id is not None:
            q = q.where(research.c.owner_id == owner_id)
        with self.engine.connect() as conn:
            return conn.execute(q).scalar()

    @_guarded
    def research_in_flight(self, owner_id):
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(research).where(
                research.c.owner_id == owner_id,
                research.c.status.in_(IN_FLIGHT))).scalar()

    @_guarded
    def research_mark_interrupted(self):
        """Sessions run in this process. After a restart, any still marked in
        flight died with the old process; close them so they stop blocking
        their owner's next session."""
        with self.engine.begin() as conn:
            conn.execute(update(research).where(
                research.c.status.in_(IN_FLIGHT)).values(
                status="failed", error="Interrupted by a server restart.",
                finished_at=datetime.now(timezone.utc)))

    @staticmethod
    def _research_out(row):
        return {"job_id": row["id"], "kind": "research", "status": row["status"],
                "symbol": row["symbol"], "goal": row["goal"],
                "model": row["model"],
                "created_at": _iso(row["created_at"]),
                "finished_at": _iso(row["finished_at"]),
                "error": row["error"], "state": row["state"]}

    def close(self):
        """Release every connection. New operations are refused at once;
        operations, schema setup and health probes already in progress are
        waited for (up to 10 s), because a connection they hold would
        otherwise be returned to a disposed pool and never closed.
        Idempotent."""
        with self._busy:
            self._closed = True
            self._busy.wait_for(lambda: self._active == 0, timeout=10)
        with self._init_lock, self._probing:
            self.engine.dispose()
