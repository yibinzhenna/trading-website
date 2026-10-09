# quantlab

Strategy backtesting with honest overfit detection.

Most retail backtesters will happily show you an in-sample curve fit and call
it a strategy. This one splits the data, tests out-of-sample, walks forward,
and tells you when a result is fitted noise.

Phase 1 of a planned web platform. Right now it is a dependency-free Python
package with no web layer — that comes next.

## Install

```bash
pip install -e ".[dev]"
pytest
```

No dependencies beyond the standard library. `pytest` is the only dev extra.

## Use

```python
from quantlab import backtest, StrategySpec
from quantlab.providers import get_provider

bars = get_provider("local", root="tests/fixtures").bars("SPY")

result = backtest(bars,
                  StrategySpec("trend_following", {"fast": 10, "slow": 30}),
                  cost_model={"slippage_bps": 10})

print(result["sharpe"], result["excess_return_pct"], result["passed"])
for name, ok, detail in result["checks"]:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")
```

## What it guarantees

**No look-ahead.** A signal function sees only `bars[:i+1]`, and fills happen
at the *next* bar's open — never the close that produced the signal. Both are
enforced by tests, because every other number is meaningless if either breaks.

**Costs are real.** Slippage applies to both sides of a trade; commission is
charged per fill. A strategy that only works at zero cost fails here.

**Benchmarked.** Every result carries buy-and-hold over the same window.
Absolute return says nothing on its own — a strategy can post a 2.0 Sharpe
and still lose badly to doing nothing.

## Robustness

| Check | What it catches |
|---|---|
| In-sample / out-of-sample split | Parameters fitted to the whole history |
| Walk-forward (4 folds) | Edge that exists in one regime only |
| Parameter sensitivity sweep | A lone peak that collapses either side |
| `likely_overfit` flag | In-sample pass with out-of-sample failure |

Ranking uses **out-of-sample** Sharpe, since in-sample performance is the part
most easily curve-fitted.

## Strategies

| Kind | Parameters |
|---|---|
| `momentum` | `lookback`, `move_pct` |
| `mean_reversion` | `rsi_period`, `rsi_low`, `rsi_high` |
| `trend_following` | `fast`, `slow` |
| `breakout` | `lookback` |
| `volatility` | `fast`, `slow`, `spread_pct` |

Each parameter is named for the strategy that uses it. An earlier version
shared one `threshold` field across all of them, which meant a percent move
for momentum and an RSI level for mean reversion — at `threshold=2` that
produced 68 momentum signals and zero mean-reversion signals.

## Providers

The engine never talks to a vendor. It asks a `DataProvider` for bars.

| Provider | Needs | Notes |
|---|---|---|
| `local` | Nothing | CSV and JSONL on disk. Used by every test. |
| `alphavantage` | `ALPHAVANTAGE_API_KEY` | Free tier is daily bars only — intraday and options are premium, verified by probe. |

Adding one means implementing a single method:

```python
class MyProvider(DataProvider):
    def bars(self, symbol, interval="day", limit=None):
        ...
        return self._finish(rows, limit)
```

## Known issues

**The `min_trades: 30` gate is miscalibrated for daily bars.** Strategies of
this kind generate roughly 2–14 trades across a year of daily data, so the
default criteria reject everything regardless of quality. The gate exists for
a good reason — a Sharpe over five trades is noise — but the threshold needs
to be set against the actual bar count and holding period. Pass your own
`criteria` to `backtest()` until this is resolved.

## Licensing note

Market data redistribution is restricted by every major vendor. Free API tiers
permit personal use, not display to third parties. Options data is stricter
still: OPRA charges a redistribution fee independent of user count, with an
exemption for historical-only products.

`data/` and `cache/` are gitignored. Do not commit vendor data.

## Layout

```
quantlab/
  engine.py            metrics, simulation, robustness, grading
  strategies.py        specs, indicators, the compiler
  providers/
    base.py            DataProvider interface
    local.py           CSV / JSONL
    alphavantage.py    daily bars
tests/                 64 tests, hermetic, no network
```
