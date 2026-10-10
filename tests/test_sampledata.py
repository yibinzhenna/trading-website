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


# ── The AI-research holdout ────────────────────────────────────────────────
# A 30% holdout must hold enough trend trades for the significance gate, or
# research could never say PASS. 15 years is what that takes.

HOLDOUT = 0.3


def holdout(kind):
    from quantlab import compile_strategy, engine
    bars = PROVIDER.bars("DEMO-REGIME")
    cut = int(len(bars) * (1 - HOLDOUT))
    res = engine.evaluate(bars, compile_strategy(StrategySpec(kind)), 1000.0,
                          {"slippage_bps": 10}, start=cut)
    return engine.grade(res)[0], res


def test_regime_series_is_long_enough_for_a_holdout():
    assert len(PROVIDER.bars("DEMO-REGIME")) == 3780


@pytest.mark.parametrize("kind", ["momentum", "trend_following",
                                  "breakout", "volatility"])
def test_planted_edge_passes_on_the_holdout(kind):
    passed, res = holdout(kind)
    assert passed and res["trades"] >= 5


def test_wrong_tool_fails_on_the_holdout():
    passed, res = holdout("mean_reversion")
    assert not passed and res["total_return_pct"] < 0


def test_generator_reproduces_the_committed_file():
    import json
    import sys
    sys.path.insert(0, "scripts")
    from make_sampledata import regime_rows
    with open("sampledata/DEMO-REGIME.jsonl", encoding="utf-8") as fh:
        on_disk = [json.loads(line) for line in fh]
    assert regime_rows() == on_disk
