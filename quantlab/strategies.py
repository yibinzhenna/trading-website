"""
Strategy definitions and the compiler that turns them into signal functions.

A signal function has the shape::

    signal(i, bars, in_position) -> "BUY" | "SELL" | None

and may only read ``bars[:i+1]``. The engine fills at the *next* bar's open,
so a signal can never act on a price it has not seen. That contract is the
single most important thing in this module; everything else is arithmetic.

Two fixes relative to the desktop prototype this was lifted from:

1. Breakout compared the close against a window that included the current
   bar's own high. Since high >= close by definition, it fired essentially
   never — measured at 0 signals across 300 bars in three regimes. It now
   compares against the window *ending at the previous bar*.

2. A single `threshold` field meant different things per strategy: a percent
   move for momentum, an RSI level for mean reversion. At threshold=2 that
   produced 68 momentum signals and 0 mean-reversion signals. Each strategy
   now reads its own named parameter.
"""

from dataclasses import dataclass, field

KINDS = ("momentum", "mean_reversion", "trend_following",
         "breakout", "volatility")

# Per-kind defaults. Named, so no value has to mean two things.
DEFAULTS = {
    "momentum":        {"lookback": 10, "move_pct": 2.0},
    "mean_reversion":  {"rsi_period": 14, "rsi_low": 30.0, "rsi_high": 70.0},
    "trend_following": {"fast": 10, "slow": 30},
    "breakout":        {"lookback": 20},
    "volatility":      {"fast": 10, "slow": 30, "spread_pct": 1.0},
}


class StrategyError(ValueError):
    """A spec that cannot be compiled into a signal function."""


@dataclass
class StrategySpec:
    kind: str
    params: dict = field(default_factory=dict)
    name: str = ""
    symbol: str = "SPY"

    def resolved(self):
        """Defaults for this kind, overridden by whatever was supplied."""
        if self.kind not in KINDS:
            raise StrategyError(
                f"Unknown kind {self.kind!r}; expected one of {KINDS}")
        merged = dict(DEFAULTS[self.kind])
        merged.update({k: v for k, v in (self.params or {}).items()
                       if v is not None})
        return merged


# ── Indicators ─────────────────────────────────────────────────────────────

def sma(bars, i, n):
    """Simple moving average of the n closes ending at bar i."""
    if n <= 0 or i + 1 < n:
        return None
    return sum(b["c"] for b in bars[i + 1 - n:i + 1]) / n


def rsi(bars, i, n=14):
    """Wilder-style RSI over the n changes ending at bar i."""
    if n <= 0 or i < n:
        return None
    gains = losses = 0.0
    for k in range(i - n + 1, i + 1):
        delta = bars[k]["c"] - bars[k - 1]["c"]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - 100.0 / (1 + rs)


# ── Compiler ───────────────────────────────────────────────────────────────

def compile_strategy(spec):
    """StrategySpec (or an equivalent dict) -> signal function."""
    if isinstance(spec, dict):
        spec = StrategySpec(kind=spec.get("kind", ""),
                            params=spec.get("params") or {},
                            name=spec.get("name", ""),
                            symbol=spec.get("symbol", "SPY"))
    p = spec.resolved()

    if spec.kind == "momentum":
        look, move = int(p["lookback"]), abs(float(p["move_pct"]))

        def signal(i, bars, in_position):
            if i < look:
                return None
            past = bars[i - look]["c"]
            if not past:
                return None
            change = (bars[i]["c"] - past) / past * 100.0
            if not in_position and change >= move:
                return "BUY"
            if in_position and change <= -move:
                return "SELL"
            return None

    elif spec.kind == "mean_reversion":
        period = int(p["rsi_period"])
        low, high = float(p["rsi_low"]), float(p["rsi_high"])
        if not 0 < low < high < 100:
            raise StrategyError(
                f"Need 0 < rsi_low < rsi_high < 100, got {low} / {high}")

        def signal(i, bars, in_position):
            value = rsi(bars, i, period)
            if value is None:
                return None
            if not in_position and value <= low:
                return "BUY"
            if in_position and value >= high:
                return "SELL"
            return None

    elif spec.kind == "trend_following":
        fast, slow = int(p["fast"]), int(p["slow"])
        if fast >= slow:
            raise StrategyError(f"fast ({fast}) must be < slow ({slow})")

        def signal(i, bars, in_position):
            f, s = sma(bars, i, fast), sma(bars, i, slow)
            if f is None or s is None:
                return None
            if not in_position and f > s:
                return "BUY"
            if in_position and f < s:
                return "SELL"
            return None

    elif spec.kind == "breakout":
        look = int(p["lookback"])

        def signal(i, bars, in_position):
            # Window ENDS at the previous bar. Including bar i would compare
            # its close against its own high, which can only tie.
            if i < look:
                return None
            window = bars[i - look:i]
            high = max(b["h"] for b in window)
            low = min(b["l"] for b in window)
            close = bars[i]["c"]
            if not in_position and close > high:
                return "BUY"
            if in_position and close < low:
                return "SELL"
            return None

    elif spec.kind == "volatility":
        fast, slow = int(p["fast"]), int(p["slow"])
        spread_pct = abs(float(p["spread_pct"]))
        if fast >= slow:
            raise StrategyError(f"fast ({fast}) must be < slow ({slow})")

        def signal(i, bars, in_position):
            f, s = sma(bars, i, fast), sma(bars, i, slow)
            if f is None or s is None or not s:
                return None
            spread = abs(f - s) / s * 100.0
            if not in_position and f > s and spread >= spread_pct:
                return "BUY"
            if in_position and f < s:
                return "SELL"
            return None

    else:                                    # unreachable: resolved() guards
        raise StrategyError(f"Unknown kind {spec.kind!r}")

    signal.spec = spec
    signal.resolved_params = p
    return signal
