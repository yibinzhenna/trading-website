"""
quantlab HTTP API.

No auth and no database — both are Phase 4. This layer exists to prove the
shape: submit a backtest, poll for it, get a graded result.

Run it with::

    uvicorn api.main:app --reload --workers 1

One worker, deliberately. Jobs live in process memory, so a second worker
would accept a submission on one process and be asked for it on another,
returning 404 for a job that is running perfectly well next door.
"""

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import deps, schemas
from api.security import enforce_submission_limits, require_admin
from quantlab import StrategySpec, __version__, backtest, engine
from quantlab.providers import ProviderError
from quantlab.strategies import StrategyError, compile_strategy

@asynccontextmanager
async def lifespan(_app):
    yield
    deps.jobs.shutdown(wait=False)


app = FastAPI(
    lifespan=lifespan,
    title="quantlab",
    version=__version__,
    summary="Strategy backtesting with overfit detection",
)

# Wide open for local development. Phase 4 narrows this to the real origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Metadata ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=schemas.Health, tags=["meta"])
def health():
    in_flight = deps.jobs.in_flight()
    return schemas.Health(
        version=__version__,
        provider=deps.settings.provider,
        cache_entries=deps.cache.stats()["entries"],
        jobs_in_flight=in_flight,
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
def submit_backtest(req: schemas.BacktestRequest):
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

    job = deps.jobs.submit(
        "backtest", _run_backtest, req,
        meta={"symbol": req.symbol, "kind": req.kind,
              "interval": req.interval})
    return schemas.JobRef(**job.to_dict(include_result=False))


@app.get("/backtest/{job_id}", response_model=schemas.JobStatus,
         tags=["backtest"])
def get_backtest(job_id: str):
    job = deps.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    payload = job.to_dict()
    if isinstance(payload.get("result"), dict):
        payload["result"] = {k: v for k, v in payload["result"].items()
                             if k != "_series"}
    return schemas.JobStatus(**payload)


@app.get("/backtest/{job_id}/equity", tags=["backtest"])
def get_equity(job_id: str):
    """Equity curve and benchmark, for charting.

    Served separately because it dwarfs the summary and only one view needs it.
    """
    job = deps.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    if job.status != "done":
        raise HTTPException(409, f"Job is {job.status}, not done")
    series = (job.result or {}).get("_series")
    if not series:
        raise HTTPException(404, "No series recorded for this job")
    return series


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
