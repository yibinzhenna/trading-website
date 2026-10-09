"""End-to-end: provider -> compile -> backtest -> grade."""

from quantlab import KINDS, StrategySpec, backtest
from quantlab.providers import LocalProvider

BARS = LocalProvider("tests/fixtures").bars("SPY")


def test_full_pipeline_produces_a_graded_result():
    r = backtest(BARS, StrategySpec("trend_following",
                                    {"fast": 10, "slow": 30}),
                 cost_model={"slippage_bps": 10})
    for key in ("sharpe", "max_drawdown_pct", "excess_return_pct", "passed",
                "checks", "in_sample", "out_of_sample", "folds",
                "consistency", "likely_overfit"):
        assert key in r
    assert isinstance(r["passed"], bool)


def test_every_kind_runs_end_to_end():
    for kind in KINDS:
        r = backtest(BARS, StrategySpec(kind))
        assert r["bars"] == len(BARS)


def test_ranking_key_is_out_of_sample():
    r = backtest(BARS, StrategySpec("momentum"))
    assert r["oos_sharpe"] == r["out_of_sample"]["sharpe"]


def test_overfit_flag_is_set_when_oos_fails():
    r = backtest(BARS, StrategySpec("momentum"))
    expected = (r["in_sample"]["sharpe"] > 0
                and r["out_of_sample"]["sharpe"] <= 0)
    assert r["likely_overfit"] == expected
