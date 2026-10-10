"""
quantlab HTTP API.

Submit a backtest, poll for it, get a graded result. Finished runs are
written to a database (`api.store`), so a result link keeps working after
the process that computed it is gone. No accounts yet.

Run it with::

    uvicorn api.main:app --reload --workers 1

One worker, deliberately. Jobs *in flight* live in process memory, so a
second worker would accept a submission on one process and be asked for it
on another, returning 404 for a job that is running perfectly well next door.
Finished runs are in the database and readable from anywhere.
"""

import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import deps, schemas
from api.store import StoreUnavailable
from api.auth import User, optional_user, require_user
from api.security import (check_daily_limit, daily_gate, daily_usage,
                          enforce_submission_limits, require_admin)
from quantlab import StrategySpec, __version__, backtest, engine
from quantlab.providers import ProviderError
from quantlab.research import run_research
from quantlab.strategies import StrategyError, compile_strategy

@asynccontextmanager
async def lifespan(_app):
    # Schema setup in the background: startup must not wait on the database.
    threading.Thread(target=deps.runs.warm, name="store-warm",
                     daemon=True).start()
    yield
    deps.jobs.shutdown(wait=False)
    deps.research_jobs.shutdown(wait=False)
    deps.runs.close()


def _optional(fn, *args):
    """For informational extras: None while the database is down."""
    try:
        return fn(*args)
    except StoreUnavailable:
        return None


app = FastAPI(
    lifespan=lifespan,
    title="quantlab",
    version=__version__,
    summary="Strategy backtesting with overfit detection",
)

if deps.settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=deps.settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )


def _csp():
    """Content-Security-Policy for every response.

    The session token lives in localStorage, where any script on the page
    can read it, so the real defence for accounts is controlling which
    scripts run at all: this origin, plus pinned and hashed files from one
    CDN. No inline script, no inline style, no eval. Network calls may go
    only here and to the Supabase project. No site may frame the page, which
    rules out clickjacking the sign-in dialog.
    """
    connect = ["'self'"]
    if deps.settings.supabase_url:
        connect.append(deps.settings.supabase_url)
    return "; ".join([
        "default-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src " + " ".join(connect),
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    h = response.headers
    # FastAPI's interactive docs run an inline script and need their own
    # policy; they render only the schema this app generates.
    if not request.url.path.startswith(("/docs", "/redoc")):
        h.setdefault("Content-Security-Policy", _csp())
    h.setdefault("X-Content-Type-Options", "nosniff")
    h.setdefault("X-Frame-Options", "DENY")
    # Run links carry the run id in the query string; never send it on.
    h.setdefault("Referrer-Policy", "no-referrer")
    h.setdefault("Permissions-Policy",
                 "camera=(), microphone=(), geolocation=(), payment=()")
    if request.url.scheme == "https" or \
            request.headers.get("x-forwarded-proto") == "https":
        h.setdefault("Strict-Transport-Security",
                     "max-age=31536000; includeSubDomains")
    # Per-user responses (run lists, account details) must not be stored by
    # a shared cache or left on a shared computer's disk.
    if request.url.path.startswith(("/me", "/runs")):
        h.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(StoreUnavailable)
async def store_unavailable(_request, _exc):
    """Anything that needs saved data, while the database is unreachable."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503, headers={"Retry-After": "15"},
        content={"detail": "Saved results are temporarily unavailable. "
                           "New backtests still work; try again shortly."})


# ── Metadata ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=schemas.Health, tags=["meta"])
def health():
    in_flight = deps.jobs.in_flight()
    return schemas.Health(
        version=__version__,
        provider=deps.settings.provider,
        cache_entries=deps.cache.stats()["entries"],
        jobs_in_flight=in_flight,
        database=deps.runs.dialect,
        # Reported, not enforced: a 503 here would make the host restart a
        # server that is still serving backtests perfectly well.
        database_status=deps.runs.health(),
    )


@app.get("/strategies", response_model=schemas.StrategyList, tags=["meta"])
def strategies():
    """Every strategy kind with its parameter names and defaults."""
    return schemas.strategy_catalog()


@app.get("/symbols", tags=["meta"])
def symbols(provider: str | None = Query(None)):
    """What the active provider can serve.

    `symbols` is null for providers that cannot enumerate — a remote vendor
    covering thousands of tickers — meaning "try any", not "none available".
    `default` is a symbol known to produce a meaningful result, so a first
    visit does not open on an error.
    """
    try:
        prov = deps.build_provider(provider)
    except ProviderError as e:
        raise HTTPException(400, f"Provider unavailable: {e}") from e
    syms = prov.symbols()
    default = None
    if syms:
        default = next((s for s in syms if s == "DEMO-REGIME"), syms[0])
    return {"provider": prov.name, "symbols": syms, "default": default}


@app.get("/providers", response_model=list[schemas.ProviderInfo], tags=["meta"])
def providers():
    return deps.provider_status()


# Operator tools. They reveal what other people queried and can wipe state,
# so they sit behind QUANTLAB_ADMIN_TOKEN and are disabled when it is unset.

@app.get("/cache", tags=["admin"], dependencies=[Depends(require_admin)])
def cache_stats():
    return deps.cache.stats()


@app.delete("/cache", tags=["admin"], dependencies=[Depends(require_admin)])
def cache_clear(provider: str | None = Query(None)):
    return {"removed": deps.cache.clear(provider)}


# ── Backtests ──────────────────────────────────────────────────────────────

def _run_backtest(req: schemas.BacktestRequest):
    """Executed on a worker thread. Raises are captured by the job store."""
    provider = deps.build_provider(req.provider)
    bars = provider.bars(req.symbol, req.interval,
                         limit=deps.settings.max_bars)

    if len(bars) < deps.settings.min_bars:
        raise ValueError(
            f"{req.symbol}: only {len(bars)} bars available for interval "
            f"'{req.interval}'; need at least {deps.settings.min_bars} for a "
            "meaningful backtest")

    result = backtest(
        bars,
        StrategySpec(kind=req.kind, params=req.params, symbol=req.symbol),
        cash=req.cash,
        cost_model=req.cost_model.model_dump(),
        interval=req.interval,
        criteria=req.criteria.as_dict() or None,
    )
    # Keep the equity curve on the job for the chart endpoint, but strip it
    # from the summary payload: it is by far the largest field and no summary
    # view reads it.
    curve = result.pop("equity", None)
    bench = engine.buy_and_hold(bars, req.cash)
    result["_series"] = {
        "t": [b["t"].date().isoformat() for b in bars],
        "strategy": [round(v, 4) for v in (curve or [])],
        "benchmark": [round(v, 4) for v in bench],
    }
    for section in ("in_sample", "out_of_sample"):
        result.get(section, {}).pop("equity", None)
    for fold in result.get("folds", []):
        fold.pop("equity", None)

    result["checks"] = [{"name": n, "passed": ok, "detail": d}
                        for n, ok, d in result["checks"]]
    result["symbol"] = req.symbol
    result["provider"] = provider.name
    return result


@app.post("/backtest", response_model=schemas.JobRef, status_code=202,
          tags=["backtest"],
          dependencies=[Depends(enforce_submission_limits)])
def submit_backtest(req: schemas.BacktestRequest, request: Request,
                    user: User | None = Depends(optional_user)):
    """Queue a backtest. Returns immediately with a job id to poll."""
    # Compile now so a bad spec fails fast with a 422 rather than becoming a
    # job that fails thirty seconds later. compile_strategy, not resolved():
    # cross-parameter rules like fast < slow are enforced at compile time.
    try:
        compile_strategy(StrategySpec(kind=req.kind, params=req.params))
    except StrategyError as e:
        raise HTTPException(422, f"Invalid strategy: {e}") from e

    try:
        deps.build_provider(req.provider)
    except ProviderError as e:
        raise HTTPException(400, f"Provider unavailable: {e}") from e

    # The request rides along in meta so a saved run can repopulate the form
    # it came from — that is what makes a shared link reproducible.
    with daily_gate:
        check_daily_limit(request, user)
        job = deps.jobs.submit(
            "backtest", _run_backtest, req,
            meta={"symbol": req.symbol, "kind": req.kind,
                  "interval": req.interval,
                  "request": req.model_dump(mode="json"),
                  "owner_id": user.id if user else None})
    return schemas.JobRef(**job.to_dict(include_result=False))


def _public(payload):
    """Strip what a public reader should not get: the equity curves (served
    separately), server tracebacks (paths and source lines, for the log) and
    who submitted it (a link is shareable; the account behind it is not)."""
    payload = dict(payload)
    payload["meta"] = {k: v for k, v in payload.get("meta", {}).items()
                       if k not in ("traceback", "owner_id")}
    if isinstance(payload.get("result"), dict):
        payload["result"] = {k: v for k, v in payload["result"].items()
                             if k != "_series"}
    return payload


@app.get("/backtest/{job_id}", response_model=schemas.JobStatus,
         tags=["backtest"])
def get_backtest(job_id: str):
    """Memory first — it is the only place a job in flight exists — then the
    database, for anything finished before the last restart or evicted from
    the in-memory history."""
    job = deps.jobs.get(job_id)
    if job is not None:
        return schemas.JobStatus(**_public(job.to_dict()))
    saved = deps.runs.get(job_id)
    if saved is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return schemas.JobStatus(**_public(saved))


@app.get("/backtest/{job_id}/equity", tags=["backtest"])
def get_equity(job_id: str):
    """Equity curve and benchmark, for charting.

    Served separately because it dwarfs the summary and only one view needs it.
    """
    job = deps.jobs.get(job_id)
    if job is not None:
        if job.status != "done":
            raise HTTPException(409, f"Job is {job.status}, not done")
        series = (job.result or {}).get("_series")
    else:
        series = deps.runs.series(job_id)
        if series is None and deps.runs.get(job_id) is None:
            raise HTTPException(404, f"No such job: {job_id}")
    if not series:
        raise HTTPException(404, "No series recorded for this job")
    return series


# ── Accounts ───────────────────────────────────────────────────────────────

@app.get("/config", tags=["meta"])
def client_config():
    """What the browser needs to offer sign-in, or `auth: null` when
    accounts are off. The publishable key is designed to be public."""
    s = deps.settings
    research = None
    if deps.research_enabled():
        research = {"daily_limit": s.research_daily_limit,
                    "max_trials": s.research_max_trials}
    if not (deps.verifier and s.supabase_publishable_key):
        return {"auth": None, "research": None}
    return {"auth": {"provider": "supabase", "url": s.supabase_url,
                     "publishable_key": s.supabase_publishable_key},
            "research": research}


@app.get("/me", tags=["account"])
def me(user: User = Depends(require_user)):
    return {"id": user.id, "email": user.email,
            "rate_limit": deps.settings.user_rate_limit,
            "rate_window": deps.settings.rate_window,
            "backtests": daily_usage(user),
            "research": _optional(_research_quota, user)}


@app.get("/me/runs", tags=["account"])
def my_runs(limit: int = Query(50, ge=1, le=200),
            user: User = Depends(require_user)):
    """Your saved runs, newest first. Only finished runs appear: a run is
    written when it completes."""
    return deps.runs.list_for_owner(user.id, limit)


@app.delete("/runs/{run_id}", status_code=204, tags=["account"])
def delete_run(run_id: str, user: User = Depends(require_user)):
    """Delete one of your runs, and with it the link to it.

    Someone else's run and a run that does not exist both return 404, so
    the endpoint cannot be used to probe which ids exist.
    """
    if not deps.runs.delete(run_id, user.id):
        raise HTTPException(404, f"No such run: {run_id}")
    deps.jobs.forget(run_id)


# ── AI research ────────────────────────────────────────────────────────────
# Costs real money per session, so: signed-in users only, a daily quota per
# user, a daily ceiling across everyone, one session at a time per user, and
# a short queue. Every limit is checked before anything is spent.

_research_gate = threading.Lock()


def _day_ago():
    return datetime.now(timezone.utc) - timedelta(days=1)


def _research_quota(user):
    if not deps.research_enabled():
        return None
    used = deps.runs.research_usage(_day_ago(), user.id)
    limit = deps.settings.research_daily_limit
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def _require_research():
    if deps.research_client is None:
        raise HTTPException(503, "AI research is not enabled on this server")
    if deps.verifier is None:
        raise HTTPException(503, "AI research needs accounts enabled")


def _run_research_job(job_id, client, bars, req, max_trials):
    deps.runs.research_set_status(job_id, "running")

    def progress(state):
        job = deps.research_jobs.get(job_id)
        if job is not None:
            # A snapshot: the poller reads it while this thread appends.
            job.meta["state"] = {**state, "trials": list(state["trials"])}

    try:
        return run_research(
            client, bars, req.symbol, req.goal,
            model=deps.research_model(), max_trials=max_trials,
            token_budget=deps.settings.research_token_budget,
            request_options=deps.research_request_options(),
            on_progress=progress)
    except ValueError:
        raise
    except Exception as e:
        # API errors can carry request details; the client gets a plain
        # message and the log gets the rest.
        import logging
        logging.getLogger("quantlab.research").exception("research %s", job_id)
        raise RuntimeError("The research model could not be reached. "
                           "Try again later.") from e


@app.post("/research", status_code=202, tags=["research"])
def start_research(req: schemas.ResearchRequest,
                   user: User = Depends(require_user)):
    """Start an AI research session on a symbol. Poll GET /research/{id}."""
    _require_research()
    s = deps.settings
    max_trials = min(req.trials, s.research_max_trials)

    # Bars first: an unusable symbol must fail here, free, not as a session
    # that bills tokens and then cannot split its data.
    try:
        bars = deps.build_provider().bars(req.symbol, "day", limit=s.max_bars)
    except ProviderError as e:
        raise HTTPException(400, f"Provider unavailable: {e}") from e
    if len(bars) < 2 * s.min_bars:
        raise HTTPException(
            422, f"{req.symbol}: only {len(bars)} bars; research needs at "
                 f"least {2 * s.min_bars} to keep a holdout")

    # Check-then-insert under one lock, or two quick requests both pass.
    with _research_gate:
        if deps.runs.research_in_flight(user.id):
            raise HTTPException(
                429, "You already have a research session running.")
        quota = _research_quota(user)
        if quota["remaining"] <= 0:
            raise HTTPException(
                429, f"Daily research limit reached ({quota['limit']} per "
                     "24 hours).", headers={"Retry-After": "3600"})
        if deps.runs.research_usage(_day_ago()) >= s.research_global_daily_limit:
            raise HTTPException(
                429, "The site has reached today's research capacity. "
                     "Try again tomorrow.", headers={"Retry-After": "3600"})
        if deps.research_jobs.in_flight() >= s.research_max_queue:
            raise HTTPException(
                429, "The research queue is full. Try again in a few minutes.",
                headers={"Retry-After": "60"})
        job_id = secrets.token_hex(8)
        deps.runs.research_create(job_id, user.id, req.symbol, req.goal,
                                  deps.research_model())

    deps.research_jobs.submit(
        "research", _run_research_job, job_id, deps.research_client, bars,
        req, max_trials, job_id=job_id,
        meta={"symbol": req.symbol, "goal": req.goal})
    return {"job_id": job_id, "status": "queued",
            "quota": _research_quota(user)}


@app.get("/research/{job_id}", tags=["research"])
def get_research(job_id: str):
    """A session, live while it runs, then from the database. Readable by
    anyone holding the id, like a backtest link; the owner is not shown."""
    job = deps.research_jobs.get(job_id)
    try:
        saved = deps.runs.research_get(job_id)
    except StoreUnavailable:
        if job is None:
            raise
        saved = None          # a session in memory can still be shown
    if job is None and saved is None:
        raise HTTPException(404, f"No such research session: {job_id}")
    if job is not None and job.status not in ("done", "failed"):
        out = dict(saved or {"job_id": job_id, "kind": "research",
                             "symbol": job.meta.get("symbol"),
                             "goal": job.meta.get("goal")})
        out.update(status=job.status, state=job.meta.get("state"))
        return out
    if saved is not None:
        return saved
    out = {"job_id": job_id, "kind": "research", "status": job.status,
           "symbol": job.meta.get("symbol"), "goal": job.meta.get("goal"),
           "error": job.error,
           "state": job.result if job.status == "done" else job.meta.get("state")}
    return out


@app.get("/me/research", tags=["research"])
def my_research(limit: int = Query(20, ge=1, le=100),
                user: User = Depends(require_user)):
    return {"sessions": deps.runs.research_list(user.id, limit),
            "quota": _research_quota(user)}


@app.get("/jobs", response_model=list[schemas.JobStatus], tags=["admin"],
         dependencies=[Depends(require_admin)])
def list_jobs(limit: int = Query(25, ge=1, le=200)):
    return [schemas.JobStatus(**j) for j in deps.jobs.list(limit)]


# ── Frontend ───────────────────────────────────────────────────────────────
# Served from the same app so there is one process to run and no CORS in
# development. A separate static host is a Phase 4 concern.

_HERE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_HERE / "templates" / "index.html")
