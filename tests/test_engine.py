"""
Engine tests.

Two of these matter more than the rest. `test_no_lookahead` and
`test_fills_at_next_open` guard the property that makes a backtest mean
anything: a strategy must not be able to act on information it would not
have had. Every other metric is downstream of that being true.
"""

import math

import pytest

from quantlab import engine


def flat(n=10, price=100.0):
    return [{"t": i, "o": price, "h": price, "l": price, "c": price, "v": 0}
            for i in range(n)]


def ramp(n=20, start=100.0, step=1.0):
    return [{"t": i, "o": start + i * step, "h": start + i * step + 0.5,
             "l": start + i * step - 0.5, "c": start + i * step, "v": 0}
            for i in range(n)]


# ── Known answers ──────────────────────────────────────────────────────────

def test_max_drawdown_known_answer():
    # peak 120, trough 60 -> 50%
    assert engine.max_drawdown([100, 120, 60, 90]) == pytest.approx(50.0)


def test_max_drawdown_monotonic_is_zero():
    assert engine.max_drawdown([100, 110, 120]) == pytest.approx(0.0)


def test_sharpe_flat_equity_is_zero():
    assert engine.sharpe([100, 100, 100, 100]) == 0.0


def test_sharpe_sign_follows_drift():
    up = engine.sharpe([100, 101, 100.5, 102, 101.5, 103])
    down = engine.sharpe([100, 99, 99.5, 98, 98.5, 97])
    assert up > 0 > down


def test_sortino_ignores_upside_volatility():
    # No down bars means no downside deviation to divide by.
    assert engine.sortino([100, 110, 120, 130]) == 0.0


def test_sparse_returns_do_not_inflate_ratios():
    """A mostly-cash curve has near-zero variance, which would otherwise
    produce a meaningless four-digit Sharpe."""
    assert engine.sharpe([100, 100, 100, 101, 101]) == 0.0
    assert engine.sortino([100, 100, 100, 101, 101]) == 0.0


def test_trade_stats_known_answer():
    stats = engine.trade_stats([10, -5, 20, -5])
    assert stats["trades"] == 4
    assert stats["win_rate"] == pytest.approx(50.0)
    assert stats["profit_factor"] == pytest.approx(3.0)   # 30 won / 10 lost


def test_profit_factor_with_no_losses_is_infinite():
    assert math.isinf(engine.trade_stats([5, 5])["profit_factor"])


def test_trade_stats_empty():
    stats = engine.trade_stats([])
    assert stats["trades"] == 0 and stats["win_rate"] == 0.0


# ── The guards ─────────────────────────────────────────────────────────────

def test_no_lookahead():
    """The signal function must never be handed the final bar."""
    bars = ramp(20)
    widest = []

    def peeker(i, b, in_position):
        widest.append(i)
        return None

    engine.run(bars, peeker)
    assert max(widest) == len(bars) - 2, (
        "signal saw a bar it would not have had at decision time")


def test_fills_at_next_open():
    """A BUY on bar i must fill at bar i+1's open, not bar i's close."""
    bars = [
        {"t": 0, "o": 10, "h": 10, "l": 10, "c": 10, "v": 0},
        {"t": 1, "o": 50, "h": 50, "l": 50, "c": 50, "v": 0},
        {"t": 2, "o": 50, "h": 50, "l": 50, "c": 50, "v": 0},
    ]
    equity, _ = engine.run(bars, lambda i, b, p: "BUY" if i == 0 else None,
                           cash=100.0)
    # Filled at 50 -> 2 shares -> still 100. Filling at the close of 10 would
    # have bought 10 shares and shown 500.
    assert equity[-1] == pytest.approx(100.0)


# ── Costs ──────────────────────────────────────────────────────────────────

def test_costs_reduce_pnl():
    bars = flat(10)
    signal = lambda i, b, p: "BUY" if i == 0 else ("SELL" if i == 5 else None)
    free = engine.run(bars, signal, cash=1000.0)[1]
    paid = engine.run(bars, signal, cash=1000.0,
                      cost_model={"commission": 5, "slippage_bps": 50})[1]
    assert paid[0] < free[0]


def test_slippage_applies_to_both_sides():
    bars = flat(10)
    signal = lambda i, b, p: "BUY" if i == 0 else ("SELL" if i == 5 else None)
    low = engine.run(bars, signal, cash=1000.0,
                     cost_model={"slippage_bps": 10})[1][0]
    high = engine.run(bars, signal, cash=1000.0,
                      cost_model={"slippage_bps": 100})[1][0]
    assert high < low


# ── Evaluate / benchmark / grade ───────────────────────────────────────────

def test_buy_and_hold_tracks_price():
    bars = ramp(10, start=100.0, step=10.0)   # 100 -> 190
    curve = engine.buy_and_hold(bars, cash=1000.0)
    assert curve[-1] == pytest.approx(1900.0)


def test_evaluate_reports_excess_against_benchmark():
    bars = ramp(60)
    never = lambda i, b, p: None
    result = engine.evaluate(bars, never, cash=1000.0)
    # Sitting in cash through a rising market must show negative excess.
    assert result["benchmark_return_pct"] > 0
    assert result["excess_return_pct"] < 0


STRONG = {"sharpe": 1.5, "max_drawdown_pct": 12.0, "profit_factor": 1.8,
          "trades": 50, "excess_return_pct": 4.0, "trade_t": 5.0,
          "trade_t_critical": 1.68, "mean_trade_return_pct": 1.2}


def test_grade_passes_a_strong_result():
    passed, checks = engine.grade(dict(STRONG))
    assert passed and all(ok for _, ok, _ in checks)


def test_good_ratios_without_significance_fail():
    passed, checks = engine.grade(dict(STRONG, trade_t=0.8))
    assert not passed
    assert any("significant" in n for n, ok, _ in checks if not ok)


def test_t_critical_known_values():
    assert engine.t_critical(1) == pytest.approx(6.314)
    assert engine.t_critical(11) == pytest.approx(1.796, abs=0.002)


def test_two_trades_need_overwhelming_consistency():
    assert engine.trade_significance([0.05, 0.04])["significant"]
    assert not engine.trade_significance([0.05, 0.01])["significant"]


def test_grade_names_every_failure():
    passed, checks = engine.grade({
        "sharpe": 0.3, "max_drawdown_pct": 45.0, "profit_factor": 0.8,
        "trades": 5, "excess_return_pct": -2.0})
    assert not passed
    assert len([c for c in checks if not c[1]]) == 5


def test_trade_floor_rejects_a_couple_of_bets():
    passed, checks = engine.grade(dict(STRONG, trades=2))
    assert not passed
    assert any(n.startswith("Trades") for n, ok, _ in checks if not ok)


# ── Robustness ─────────────────────────────────────────────────────────────

def test_split_test_partitions_without_overlap():
    bars = ramp(100)
    never = lambda i, b, p: None
    split = engine.split_test(bars, never, train_frac=0.7)
    assert split["split_at"] == 70
    assert split["in_sample"]["bars"] == 70
    assert split["out_of_sample"]["bars"] == 30


def test_walk_forward_returns_each_fold():
    bars = ramp(120)
    folds = engine.walk_forward(bars, lambda i, b, p: None, folds=4)
    assert len(folds) == 4


def test_sensitivity_sweeps_every_value():
    bars = ramp(80)
    rows = engine.sensitivity(
        bars, lambda v: (lambda i, b, p: "BUY" if i == v else None),
        [5, 10, 15])
    assert [r["param"] for r in rows] == [5, 10, 15]
