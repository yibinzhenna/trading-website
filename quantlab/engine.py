"""
Backtesting engine and performance analytics.

Every number here is COMPUTED from historical bars. Nothing is ever asked of
a language model — a model with no engine will happily invent a plausible
Sharpe ratio, and a fabricated metric you act on is worse than no metric.

The engine takes bars from a provider (see quantlab.providers) and never
talks to a data vendor itself.

Conventions: returns are per-bar simple returns, Sharpe and Sortino are
annualised against a zero risk-free rate, drawdown is peak-to-trough on the
equity curve.
"""

import math

# Bars per year, used to annualise. Approximate but consistent across runs.
BARS_PER_YEAR = {
    "1min": 252 * 390,
    "5min": 252 * 78,
    "10min": 252 * 39,
    "hour": 252 * 7,
    "day": 252,
    "week": 52,
}

# A strategy must clear every one of these to be called deployable.
# min_trades is not in the source framework and is deliberate: a Sharpe
# computed over a handful of trades is noise, not evidence.
DEFAULT_CRITERIA = {
    "min_sharpe": 1.0,
    "max_drawdown_pct": 30.0,
    "must_beat_benchmark": True,
    "min_trades": 30,
    "min_profit_factor": 1.0,
}


# ── Metrics ────────────────────────────────────────────────────────────────

def _returns(equity):
    out = []
    for i in range(1, len(equity)):
        prev = equity[i - 1]
        out.append((equity[i] - prev) / prev if prev else 0.0)
    return out


def _stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _too_sparse(rets):
    """A curve that barely moves has near-zero variance, which inflates any
    risk-adjusted ratio. Mostly-cash strategies hit this; report 0 instead of
    a meaningless four-digit Sharpe."""
    return sum(1 for r in rets if r != 0) < 2


def sharpe(equity, interval="day"):
    """Annualised Sharpe, zero risk-free rate."""
    rets = _returns(equity)
    sd = _stdev(rets)
    if not rets or sd == 0 or _too_sparse(rets):
        return 0.0
    mean = sum(rets) / len(rets)
    return (mean / sd) * math.sqrt(BARS_PER_YEAR.get(interval, 252))


def sortino(equity, interval="day"):
    """Annualised Sortino — penalises downside deviation only."""
    rets = _returns(equity)
    if not rets:
        return 0.0
    downside = [r for r in rets if r < 0]
    dd = _stdev(downside) if len(downside) > 1 else 0.0
    if dd == 0 or _too_sparse(rets):
        return 0.0
    mean = sum(rets) / len(rets)
    return (mean / dd) * math.sqrt(BARS_PER_YEAR.get(interval, 252))


def max_drawdown(equity):
    """Worst peak-to-trough decline, as a positive percentage."""
    peak, worst = equity[0] if equity else 0.0, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak:
            worst = max(worst, (peak - v) / peak)
    return worst * 100.0


def cagr(equity, bars, interval="day"):
    if len(equity) < 2 or equity[0] <= 0:
        return 0.0
    years = len(bars) / BARS_PER_YEAR.get(interval, 252)
    if years <= 0:
        return 0.0
    growth = equity[-1] / equity[0]
    if growth <= 0:
        return -100.0
    return (growth ** (1 / years) - 1) * 100.0


def trade_stats(trades):
    """Win rate and profit factor from closed-trade P&L."""
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else
                         (float("inf") if gross_win else 0.0),
        "avg_win": (gross_win / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_loss / len(losses)) if losses else 0.0,
    }


# ── Engine ─────────────────────────────────────────────────────────────────

def run(bars, signal_fn, cash=1000.0, cost_model=None, interval="day"):
    """Walk bars once, long-only, one position at a time.

    signal_fn(i, bars, in_position) -> "BUY" | "SELL" | None. It may only read
    bars[:i+1]; slicing that way is what keeps look-ahead bias out.

    Fills are modelled at the NEXT bar's open, never the signal bar's close —
    you cannot trade a price you have not seen yet.
    """
    cost = cost_model or {}
    per_trade = float(cost.get("commission", 0.0))
    slip_bps = float(cost.get("slippage_bps", 0.0))

    equity, trades = [cash], []
    shares, entry_px, entry_cost = 0.0, 0.0, 0.0

    for i in range(len(bars) - 1):
        price = bars[i + 1]["o"]          # fill at next open
        action = signal_fn(i, bars, shares > 0)

        if action == "BUY" and shares == 0:
            fill = price * (1 + slip_bps / 10000.0)
            spend = equity[-1] - per_trade
            if spend > 0 and fill > 0:
                shares = spend / fill
                entry_px, entry_cost = fill, spend + per_trade
        elif action == "SELL" and shares > 0:
            fill = price * (1 - slip_bps / 10000.0)
            proceeds = shares * fill - per_trade
            trades.append(proceeds - entry_cost)
            shares, entry_px, entry_cost = 0.0, 0.0, 0.0
            equity.append(proceeds)
            continue

        mark = bars[i + 1]["c"]
        equity.append(shares * mark if shares > 0 else equity[-1])

    # Close anything still open at the final bar.
    if shares > 0 and bars:
        fill = bars[-1]["c"] * (1 - slip_bps / 10000.0)
        proceeds = shares * fill - per_trade
        trades.append(proceeds - entry_cost)
        equity.append(proceeds)

    return equity, trades


def buy_and_hold(bars, cash=1000.0):
    """Benchmark equity curve. Beating this after costs is the bar."""
    if not bars:
        return [cash]
    shares = cash / bars[0]["c"]
    return [shares * b["c"] for b in bars]


def evaluate(bars, signal_fn, cash=1000.0, cost_model=None,
             interval="day", benchmark_bars=None):
    """Full metric set for one strategy over one window."""
    equity, trades = run(bars, signal_fn, cash, cost_model, interval)
    bench = buy_and_hold(benchmark_bars or bars, cash)

    total = ((equity[-1] - cash) / cash * 100.0) if cash else 0.0
    bench_total = ((bench[-1] - cash) / cash * 100.0) if cash else 0.0

    res = {
        "total_return_pct": total,
        "benchmark_return_pct": bench_total,
        "excess_return_pct": total - bench_total,
        "cagr_pct": cagr(equity, bars, interval),
        "sharpe": sharpe(equity, interval),
        "sortino": sortino(equity, interval),
        "max_drawdown_pct": max_drawdown(equity),
        "benchmark_max_drawdown_pct": max_drawdown(bench),
        "final_equity": equity[-1],
        "bars": len(bars),
        "equity": equity,
    }
    res.update(trade_stats(trades))
    return res


def split_test(bars, signal_fn, cash=1000.0, cost_model=None,
               interval="day", train_frac=0.7):
    """In-sample / out-of-sample split.

    A strategy that only works in-sample is a fitted curve, not an edge.
    """
    cut = max(1, int(len(bars) * train_frac))
    return {
        "in_sample": evaluate(bars[:cut], signal_fn, cash, cost_model, interval),
        "out_of_sample": evaluate(bars[cut:], signal_fn, cash, cost_model, interval),
        "split_at": cut,
    }


def walk_forward(bars, signal_fn, folds=4, cash=1000.0,
                 cost_model=None, interval="day"):
    """Sequential non-overlapping folds — consistency across regimes."""
    n = len(bars) // folds
    out = []
    for f in range(folds):
        chunk = bars[f * n:(f + 1) * n] if f < folds - 1 else bars[f * n:]
        if len(chunk) > 2:
            out.append(evaluate(chunk, signal_fn, cash, cost_model, interval))
    return out


def sensitivity(bars, make_signal_fn, param_values, cash=1000.0,
                cost_model=None, interval="day"):
    """Sweep one parameter. A peak that collapses either side is overfit."""
    rows = []
    for v in param_values:
        r = evaluate(bars, make_signal_fn(v), cash, cost_model, interval)
        rows.append({"param": v, "sharpe": r["sharpe"],
                     "total_return_pct": r["total_return_pct"],
                     "max_drawdown_pct": r["max_drawdown_pct"],
                     "trades": r["trades"]})
    return rows


# ── Verdict ────────────────────────────────────────────────────────────────

def grade(result, criteria=None):
    """Apply the accept/reject gates. Returns (passed, [(name, ok, detail)])."""
    c = dict(DEFAULT_CRITERIA)
    c.update(criteria or {})
    checks = [
        ("Sharpe >= %.2f" % c["min_sharpe"],
         result["sharpe"] >= c["min_sharpe"],
         "%.2f" % result["sharpe"]),
        ("Max drawdown <= %.0f%%" % c["max_drawdown_pct"],
         result["max_drawdown_pct"] <= c["max_drawdown_pct"],
         "%.1f%%" % result["max_drawdown_pct"]),
        ("Profit factor >= %.2f" % c["min_profit_factor"],
         result["profit_factor"] >= c["min_profit_factor"],
         "%.2f" % result["profit_factor"]),
        ("Trades >= %d (sample size)" % c["min_trades"],
         result["trades"] >= c["min_trades"],
         "%d" % result["trades"]),
    ]
    if c["must_beat_benchmark"]:
        checks.append(("Beats benchmark after costs",
                       result["excess_return_pct"] > 0,
                       "%+.2f%% vs benchmark" % result["excess_return_pct"]))
    return all(ok for _, ok, _ in checks), checks
