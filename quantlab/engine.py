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
    # A floor, not a sample-size test. Below this there is no "strategy" to
    # evaluate, just a couple of bets. The significance test below is what
    # decides whether the trades that did happen could plausibly be luck.
    "min_trades": 3,
    "min_profit_factor": 1.0,
    # One-sided confidence that mean trade return > 0. Replaces a fixed
    # min_trades of 30, which no daily-bar strategy over a year could reach,
    # so it rejected everything regardless of quality.
    "min_confidence": 0.95,
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

def run(bars, signal_fn, cash=1000.0, cost_model=None, interval="day",
        with_returns=False):
    """Walk bars once, long-only, one position at a time.

    signal_fn(i, bars, in_position) -> "BUY" | "SELL" | None. It may only read
    bars[:i+1]; slicing that way is what keeps look-ahead bias out.

    Fills are modelled at the NEXT bar's open, never the signal bar's close —
    you cannot trade a price you have not seen yet.

    Returns (equity, trades), or (equity, trades, trade_returns) when
    `with_returns` is set. Trade returns are P&L over capital committed, so
    later trades are not weighted more heavily just because equity compounded.
    """
    cost = cost_model or {}
    per_trade = float(cost.get("commission", 0.0))
    slip_bps = float(cost.get("slippage_bps", 0.0))

    equity, trades, rets = [cash], [], []
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
            rets.append((proceeds - entry_cost) / entry_cost if entry_cost else 0.0)
            shares, entry_px, entry_cost = 0.0, 0.0, 0.0
            equity.append(proceeds)
            continue

        mark = bars[i + 1]["c"]
        equity.append(shares * mark if shares > 0 else equity[-1])

    # Close anything still open at the final bar. This replaces that bar's
    # mark rather than appending: one point per bar, so the curve lines up
    # with the dates and the benchmark. Appending gave the last bar two.
    if shares > 0 and bars:
        fill = bars[-1]["c"] * (1 - slip_bps / 10000.0)
        proceeds = shares * fill - per_trade
        trades.append(proceeds - entry_cost)
        rets.append((proceeds - entry_cost) / entry_cost if entry_cost else 0.0)
        equity[-1] = proceeds

    if with_returns:
        return equity, trades, rets
    return equity, trades


# ── Significance ───────────────────────────────────────────────────────────
#
# One-sided Student-t critical values. The engine has no dependencies, so a
# table rather than scipy. Values between rows are interpolated in 1/df, which
# is close to linear for the t distribution and accurate to the third decimal
# across this range.

_T_TABLE = {
    0.90: {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440,
           7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372, 12: 1.356, 15: 1.341,
           20: 1.325, 25: 1.316, 30: 1.310, 40: 1.303, 60: 1.296,
           120: 1.289},
    0.95: {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
           7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 12: 1.782, 15: 1.753,
           20: 1.725, 25: 1.708, 30: 1.697, 40: 1.684, 60: 1.671,
           120: 1.658},
    0.99: {1: 31.821, 2: 6.965, 3: 4.541, 4: 3.747, 5: 3.365, 6: 3.143,
           7: 2.998, 8: 2.896, 9: 2.821, 10: 2.764, 12: 2.681, 15: 2.602,
           20: 2.528, 25: 2.485, 30: 2.457, 40: 2.423, 60: 2.390,
           120: 2.358},
}
_T_INF = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}


def t_critical(df, confidence=0.95):
    """One-sided critical t for `df` degrees of freedom.

    Grows sharply as df shrinks: 6.31 at df=1, 2.02 at df=5, 1.70 at df=30
    for 95%. That growth is the whole point — a small sample has to show far
    more consistency to count as evidence.
    """
    if confidence not in _T_TABLE:
        raise ValueError(f"confidence must be one of {sorted(_T_TABLE)}")
    if df < 1:
        return float("inf")
    table = _T_TABLE[confidence]
    if df in table:
        return table[df]
    keys = sorted(table)
    if df > keys[-1]:
        lo_df, lo_t = keys[-1], table[keys[-1]]
        # Interpolate from the last row toward the normal limit in 1/df.
        w = (1 / lo_df - 1 / df) / (1 / lo_df)
        return lo_t + w * (_T_INF[confidence] - lo_t)
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    w = (1 / lo - 1 / df) / (1 / lo - 1 / hi)
    return table[lo] + w * (table[hi] - table[lo])


def trade_significance(trade_returns, confidence=0.95):
    """Is the mean trade return distinguishable from zero?

    A one-sided t-test on per-trade returns. This is what a minimum trade
    count was crudely standing in for: whether the result could plausibly be
    luck. Unlike a fixed count, the bar rises automatically as the sample
    shrinks, so two trades can pass only with overwhelming consistency.

    Returns t, the critical value, the sample size, and the verdict.
    """
    n = len(trade_returns)
    out = {"n": n, "t": 0.0, "critical": float("inf"),
           "mean": 0.0, "significant": False, "confidence": confidence}
    if n < 2:
        return out                      # a t-test needs at least two points
    mean = sum(trade_returns) / n
    sd = _stdev(trade_returns)
    out["mean"] = mean
    out["critical"] = t_critical(n - 1, confidence)
    if sd == 0:
        # Every trade identical. Consistent winners are significant, anything
        # else is not — there is no variance to test against.
        out["t"] = float("inf") if mean > 0 else 0.0
    else:
        out["t"] = mean / (sd / math.sqrt(n))
    out["significant"] = mean > 0 and out["t"] >= out["critical"]
    return out


def buy_and_hold(bars, cash=1000.0):
    """Benchmark equity curve. Beating this after costs is the bar."""
    if not bars:
        return [cash]
    shares = cash / bars[0]["c"]
    return [shares * b["c"] for b in bars]


def evaluate(bars, signal_fn, cash=1000.0, cost_model=None,
             interval="day", benchmark_bars=None, confidence=0.95, start=0):
    """Full metric set for one strategy over one window.

    `start` measures only `bars[start:]`, with the bars before it used as
    history. Without it a later window starts cold: a 30-bar moving average
    sees nothing for its first 30 bars, so a short window loses much of its
    length before the strategy can act. The strategy still cannot trade
    before the window — its first possible fill is at `bars[start]`'s open —
    and signals only ever read bars up to the current one.
    """
    measured = len(bars) - start
    if start > 0:
        inner = signal_fn

        def signal_fn(i, b, in_position):
            return inner(i, b, in_position) if i >= start - 1 else None

    equity, trades, rets = run(bars, signal_fn, cash, cost_model, interval,
                               with_returns=True)
    if start > 0:
        # From the close before the window, when the account is still cash.
        equity = equity[start - 1:]
        bars = bars[start - 1:]
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
        "bars": measured,
        "equity": equity,
    }
    res.update(trade_stats(trades))
    sig = trade_significance(rets, confidence)
    res.update({
        "trade_t": sig["t"],
        "trade_t_critical": sig["critical"],
        "trade_significant": sig["significant"],
        "mean_trade_return_pct": sig["mean"] * 100.0,
    })
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

def _fmt_t(t):
    return "inf" if t == float("inf") else "%.2f" % t


def grade(result, criteria=None):
    """Apply the accept/reject gates. Returns (passed, [(name, ok, detail)]).

    The significance check needs the trade-level fields `evaluate` produces.
    A result built by hand without them falls back to a recomputation from
    nothing, which fails — absent evidence is not evidence of an edge.
    """
    c = dict(DEFAULT_CRITERIA)
    c.update(criteria or {})
    conf = c.get("min_confidence", 0.95)

    t = result.get("trade_t", 0.0)
    crit = result.get("trade_t_critical", float("inf"))
    # Recompute the critical value if the caller changed the confidence, so a
    # custom confidence is honoured rather than silently using the default.
    n = result.get("trades", 0)
    if n >= 2:
        crit = t_critical(n - 1, conf)
    significant = bool(result.get("mean_trade_return_pct", 0) > 0
                       and t >= crit)

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
        ("Trades >= %d" % c["min_trades"],
         result["trades"] >= c["min_trades"],
         "%d" % result["trades"]),
        ("Edge significant at %d%%" % round(conf * 100),
         significant,
         "t=%s, need %s over %d trades" % (_fmt_t(t), _fmt_t(crit), n)),
    ]
    if c["must_beat_benchmark"]:
        checks.append(("Beats benchmark after costs",
                       result["excess_return_pct"] > 0,
                       "%+.2f%% vs benchmark" % result["excess_return_pct"]))
    return all(ok for _, ok, _ in checks), checks
