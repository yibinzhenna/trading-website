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

import json
import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, MetaData,
                        String, Table, Text, create_engine, delete, func,
                        insert, select)

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
    return value.isoformat(timespec="seconds")


class RunStore:
    """Finished backtests, keyed by job id."""

    def __init__(self, url, retention_days=30, prune_every=100):
        url = normalise_url(url)
        # Strict on every dialect, so SQLite in tests rejects what Postgres
        # would reject in production instead of quietly storing `Infinity`.
        kwargs = {"pool_pre_ping": True,
                  "json_serializer": lambda o: json.dumps(o, allow_nan=False)}
        if url.startswith("sqlite"):
            # Writes come from job threads, reads from request threads.
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kwargs)
        self.retention_days = retention_days
        self.prune_every = prune_every
        self._saves = 0
        metadata.create_all(self.engine)
        self.prune()

    @property
    def dialect(self):
        return self.engine.dialect.name

    def save(self, job):
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
            "status": job.status,
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
        }
        with self.engine.begin() as conn:
            if conn.execute(select(runs.c.id).where(runs.c.id == job.id)).first():
                return
            conn.execute(insert(runs).values(**row))
        self._saves += 1
        if self.prune_every and self._saves % self.prune_every == 0:
            self.prune()

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

    def series(self, run_id):
        with self.engine.connect() as conn:
            return conn.execute(
                select(runs.c.series).where(runs.c.id == run_id)).scalar()

    def count(self):
        with self.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(runs)).scalar()

    def prune(self):
        """Delete runs past retention. Free Postgres tiers are a few hundred
        MB, and one run with its curves is tens of KB."""
        if not self.retention_days:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        with self.engine.begin() as conn:
            removed = conn.execute(
                delete(runs).where(runs.c.submitted_at < cutoff)).rowcount
        if removed:
            log.info("pruned %d runs older than %d days",
                     removed, self.retention_days)
        return removed

    def close(self):
        self.engine.dispose()
