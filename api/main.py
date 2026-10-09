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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import deps, schemas
from quantlab import StrategySpec, __version__, backtest
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
    in_flight = sum(1 for j in deps.jobs.list(limit=200)
                    if j["status"] in ("queued", "running"))
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


@app.get("/providers", response_model=list[schemas.ProviderInfo], tags=["meta"])
def providers():
    return deps.provider_status()


@app.get("/cache", tags=["meta"])
def cache_stats():
    return deps.cache.stats()


@app.delete("/cache", tags=["meta"])
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
    # The equity curve is large and the summary endpoints never use it.
    # A client that wants it can ask for the series separately later.
    result.pop("equity", None)
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
          tags=["backtest"])
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
    return schemas.JobStatus(**job.to_dict())


@app.get("/jobs", response_model=list[schemas.JobStatus], tags=["backtest"])
def list_jobs(limit: int = Query(25, ge=1, le=200)):
    return [schemas.JobStatus(**j) for j in deps.jobs.list(limit)]
