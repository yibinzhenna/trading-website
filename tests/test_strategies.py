"""
Strategy compiler tests.

Includes regression tests for the two defects carried out of the desktop
prototype: breakout never firing, and `threshold` meaning different things
in different strategies.
"""

import pytest

from quantlab import engine
from quantlab.strategies import (DEFAULTS, KINDS, StrategyError, StrategySpec,
                                 compile_strategy, rsi, sma)


def ramp(n=120, start=100.0, step=0.5):
    """Steadily rising bars with a realistic high > close > low."""
    out = []
    for i in range(n):
        c = start + i * step
        out.append({"t": i, "o": c, "h": c * 1.004, "l": c * 0.996,
                    "c": c, "v": 1000})
    return out


def sawtooth(n=120, base=100.0, amp=5.0, period=20):
    out = []
    for i in range(n):
        c = base + (amp if (i // period) % 2 else -amp)
        out.append({"t": i, "o": c, "h": c * 1.004, "l": c * 0.996,
                    "c": c, "v": 1000})
    return out


# ── Indicators ─────────────────────────────────────────────────────────────

def test_sma_known_answer():
    bars = [{"c": v, "o": v, "h": v, "l": v, "t": i}
            for i, v in enumerate([1, 2, 3, 4, 5])]
    assert sma(bars, 4, 5) == pytest.approx(3.0)


def test_sma_returns_none_before_enough_history():
    bars = ramp(10)
    assert sma(bars, 2, 5) is None


def test_rsi_all_gains_is_100():
    assert rsi(ramp(40), 30, 14) == pytest.approx(100.0)


def test_rsi_bounded():
    bars = sawtooth(120)
    values = [rsi(bars, i, 14) for i in range(20, 120)]
    assert all(0.0 <= v <= 100.0 for v in values if v is not None)


# ── Compiler contract ──────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_compiles_and_is_callable(kind):
    signal = compile_strategy(StrategySpec(kind))
    bars = ramp(120)
    assert signal(60, bars, False) in ("BUY", "SELL", None)


def test_unknown_kind_rejected():
    with pytest.raises(StrategyError):
        compile_strategy(StrategySpec("astrology"))


def test_fast_must_be_below_slow():
    with pytest.raises(StrategyError):
        compile_strategy(StrategySpec("trend_following",
                                      {"fast": 30, "slow": 10}))


def test_rsi_bounds_validated():
    with pytest.raises(StrategyError):
        compile_strategy(StrategySpec("mean_reversion",
                                      {"rsi_low": 80, "rsi_high": 20}))


def test_dict_spec_accepted():
    signal = compile_strategy({"kind": "momentum",
                               "params": {"lookback": 5, "move_pct": 1.0}})
    assert signal.resolved_params["lookback"] == 5


def test_defaults_applied_when_params_omitted():
    signal = compile_strategy(StrategySpec("momentum"))
    assert signal.resolved_params == DEFAULTS["momentum"]


# ── Regressions ────────────────────────────────────────────────────────────

def test_breakout_actually_fires():
    """Regression: the window used to include the current bar's own high,
    and since high >= close by definition it fired ~never. Measured at 0
    signals over 300 bars before the fix."""
    signal = compile_strategy(StrategySpec("breakout", {"lookback": 20}))
    bars = ramp(200)
    fired = sum(1 for i in range(len(bars) - 1)
                if signal(i, bars, False) == "BUY")
    assert fired > 0, "breakout produced no signals on a rising series"


def test_breakout_does_not_peek_at_current_bar():
    """The comparison window must end at the previous bar."""
    bars = ramp(60)
    signal = compile_strategy(StrategySpec("breakout", {"lookback": 10}))
    i = 30
    before = signal(i, bars, False)
    bars[i] = dict(bars[i], h=bars[i]["c"] * 10)   # spike only this bar's high
    assert signal(i, bars, False) == before


def test_threshold_is_not_shared_across_kinds():
    """Regression: one `threshold` field meant a percent move for momentum
    and an RSI level for mean reversion. Each kind now has its own name."""
    assert "move_pct" in DEFAULTS["momentum"]
    assert "rsi_low" in DEFAULTS["mean_reversion"]
    assert "threshold" not in DEFAULTS["momentum"]
    assert "threshold" not in DEFAULTS["mean_reversion"]


def test_mean_reversion_fires_on_its_own_scale():
    """With named parameters, mean reversion triggers on an RSI level
    instead of silently never firing."""
    signal = compile_strategy(StrategySpec(
        "mean_reversion", {"rsi_low": 45.0, "rsi_high": 55.0}))
    bars = sawtooth(200)
    fired = sum(1 for i in range(len(bars) - 1)
                if signal(i, bars, False) == "BUY")
    assert fired > 0


# ── Signals respect the no-lookahead contract ──────────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_signal_only_reads_history(kind):
    """Mutating bars after i must not change the decision at i."""
    signal = compile_strategy(StrategySpec(kind))
    bars = ramp(150)
    i = 100
    before = signal(i, bars, False)
    for j in range(i + 1, len(bars)):
        bars[j] = dict(bars[j], c=bars[j]["c"] * 5, h=bars[j]["h"] * 5)
    assert signal(i, bars, False) == before


@pytest.mark.parametrize("kind", KINDS)
def test_end_to_end_through_the_engine(kind):
    result = engine.evaluate(ramp(200), compile_strategy(StrategySpec(kind)),
                             cash=1000.0, cost_model={"slippage_bps": 10})
    assert result["bars"] == 200
    assert "sharpe" in result and "excess_return_pct" in result
