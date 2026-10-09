"""
The bundled demo data, held to what it is documented to show. If a refactor
made the planted edge stop passing, or let a random walk pass, the live site
would quietly start teaching the wrong thing.
"""

import pytest

from quantlab import KINDS, StrategySpec, backtest
from quantlab.providers import LocalProvider

PROVIDER = LocalProvider("sampledata")


def run(symbol, kind):
    return backtest(PROVIDER.bars(symbol), StrategySpec(kind),
                    cost_model={"slippage_bps": 10})


@pytest.mark.parametrize("kind", ["momentum", "trend_following",
                                  "breakout", "volatility"])
def test_planted_edge_is_found(kind):
    r = run("DEMO-REGIME", kind)
    assert r["passed"] and r["oos_sharpe"] > 1.0


def test_wrong_tool_rejected_on_planted_edge():
    r = run("DEMO-REGIME", "mean_reversion")
    assert not r["passed"] and r["total_return_pct"] < 0


@pytest.mark.parametrize("symbol", ["DEMO-TREND", "DEMO-CHOP", "DEMO-BEAR"])
@pytest.mark.parametrize("kind", KINDS)
def test_random_walks_produce_no_passes(symbol, kind):
    """A random walk has no edge; a pass here would be a false positive."""
    assert not run(symbol, kind)["passed"]
