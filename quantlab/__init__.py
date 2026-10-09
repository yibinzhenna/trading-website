"""
quantlab — strategy backtesting and validation.

    from quantlab import backtest, StrategySpec
    from quantlab.providers import get_provider

    bars = get_provider("local", root="tests/fixtures").bars("SPY")
    result = backtest(bars, StrategySpec("trend_following",
                                         {"fast": 10, "slow": 30}))
    print(result["sharpe"], result["excess_return_pct"])
"""

from . import engine
from .engine import (DEFAULT_CRITERIA, buy_and_hold, cagr, evaluate, grade,
                     max_drawdown, run, sensitivity, sharpe, sortino,
                     split_test, trade_stats, walk_forward)
from .strategies import (DEFAULTS, KINDS, StrategyError, StrategySpec,
                         compile_strategy, rsi, sma)

__version__ = "0.1.0"

__all__ = [
    "engine", "run", "evaluate", "split_test", "walk_forward", "sensitivity",
    "grade", "buy_and_hold", "sharpe", "sortino", "max_drawdown", "cagr",
    "trade_stats", "DEFAULT_CRITERIA",
    "StrategySpec", "StrategyError", "compile_strategy", "sma", "rsi",
    "KINDS", "DEFAULTS", "backtest",
]


def backtest(bars, spec, cash=1000.0, cost_model=None, interval="day",
             criteria=None):
    """Compile a spec, score it, and grade it in one call.

    Returns the full metric set plus `passed` and `checks`, and the
    robustness breakdown that decides whether an edge is real or fitted.
    """
    signal = compile_strategy(spec)
    result = evaluate(bars, signal, cash, cost_model, interval)
    split = split_test(bars, signal, cash, cost_model, interval)
    folds = walk_forward(bars, signal, 4, cash, cost_model, interval)
    passed, checks = grade(result, criteria)

    positive = sum(1 for f in folds if f["total_return_pct"] > 0)
    result.update({
        "passed": passed,
        "checks": checks,
        "in_sample": split["in_sample"],
        "out_of_sample": split["out_of_sample"],
        "oos_sharpe": split["out_of_sample"]["sharpe"],
        "folds": folds,
        "consistency": f"{positive}/{len(folds)}" if folds else "0/0",
        # In-sample success with out-of-sample failure is the signature of a
        # fitted curve, so say so rather than leaving it to be noticed.
        "likely_overfit": (split["in_sample"]["sharpe"] > 0
                           and split["out_of_sample"]["sharpe"] <= 0),
    })
    return result
