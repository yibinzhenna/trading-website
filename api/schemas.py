"""
Request and response models.

Validation happens here, before a job is queued. A bad spec should come back
as a 422 in milliseconds, not as a failed job thirty seconds later.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from quantlab.strategies import DEFAULTS, KINDS

Kind = Literal["momentum", "mean_reversion", "trend_following",
               "breakout", "volatility"]


class CostModel(BaseModel):
    commission: float = Field(0.0, ge=0, le=1000,
                              description="Flat fee per fill")
    slippage_bps: float = Field(10.0, ge=0, le=1000,
                                description="Basis points, applied per side")


class Criteria(BaseModel):
    """Accept/reject gates. Omitted fields fall back to engine defaults."""
    min_sharpe: float | None = Field(None, ge=-10, le=10)
    max_drawdown_pct: float | None = Field(None, gt=0, le=100)
    min_trades: int | None = Field(None, ge=0, le=10_000)
    min_profit_factor: float | None = Field(None, ge=0, le=100)
    must_beat_benchmark: bool | None = None
    min_confidence: Literal[0.90, 0.95, 0.99] | None = Field(
        None, description="One-sided confidence that mean trade return > 0")

    def as_dict(self):
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BacktestRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    kind: Kind
    params: dict[str, float | int] = Field(default_factory=dict)
    interval: Literal["day", "hour", "5min", "1min"] = "day"
    cash: float = Field(1000.0, gt=0, le=1e9)
    provider: str | None = Field(
        None, description="Overrides the server default")
    cost_model: CostModel = Field(default_factory=CostModel)
    criteria: Criteria = Field(default_factory=Criteria)

    @field_validator("symbol")
    @classmethod
    def _clean_symbol(cls, v):
        s = v.strip().upper()
        if not all(c.isalnum() or c in ".-" for c in s):
            raise ValueError("symbol may only contain letters, digits, . and -")
        return s

    @field_validator("params")
    @classmethod
    def _known_params(cls, v, info):
        """Reject unknown parameter names rather than silently ignoring them.

        A typo'd `lookbak` that quietly runs with defaults produces a result
        the caller will misread as their configuration.
        """
        kind = (info.data or {}).get("kind")
        if kind and kind in DEFAULTS:
            unknown = set(v) - set(DEFAULTS[kind])
            if unknown:
                raise ValueError(
                    f"unknown params for {kind}: {sorted(unknown)}; "
                    f"expected {sorted(DEFAULTS[kind])}")
        return v


class ResearchRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=12)
    goal: str = Field("", max_length=300,
                      description="Optional steer for the model, in words")
    trials: int = Field(6, ge=2, le=20)

    _clean_symbol = field_validator("symbol")(
        BacktestRequest._clean_symbol.__func__)


class JobRef(BaseModel):
    job_id: str
    status: str
    kind: str
    submitted_at: str
    meta: dict[str, Any] = Field(default_factory=dict)


class JobStatus(JobRef):
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class StrategyInfo(BaseModel):
    kind: str
    params: dict[str, float | int]


class StrategyList(BaseModel):
    strategies: list[StrategyInfo]


class ProviderInfo(BaseModel):
    name: str
    available: bool
    detail: str = ""


class Health(BaseModel):
    status: str = "ok"
    version: str
    provider: str
    cache_entries: int
    jobs_in_flight: int
    database: str = Field(description="Dialect where runs are kept")
    database_status: str = Field("ok", description='"ok" or "unavailable"')


def strategy_catalog():
    """Every kind with its parameters and defaults — enough for a client to
    build a form without hardcoding anything."""
    return StrategyList(strategies=[
        StrategyInfo(kind=k, params=DEFAULTS[k]) for k in KINDS])
