"""
The research loop: the model proposes, the engine measures.

The obvious way to build this is to let a language model try strategies and
report the best one. That is an overfitting machine. Try enough parameter
sets on the same bars and one of them looks brilliant by chance, and a model
asked to summarise will happily quote its numbers back as findings.

So the loop is built around three rules:

1. **A holdout the model never sees.** The last `holdout_frac` of the bars
   is cut off before the model is involved. Every trial runs on the research
   window only; the model's pick is then evaluated on the holdout exactly
   once. That one-shot result is the verdict.

2. **The engine is the only source of numbers.** The model chooses kinds and
   parameters through a tool, and receives measurements back. Its notes are
   commentary, shown as such, and metric figures it tries to state in them
   are removed.

3. **Hard budgets.** A trial cap, a token budget per run, and a final turn
   that may only call `finish`. The loop cannot run away with the bill.

The model client is injected (anything with `messages.create(...)` shaped
like the Anthropic SDK), so the loop is testable with a scripted fake.
"""

import json
import re

from . import engine
from .strategies import DEFAULTS, KINDS, StrategyError, StrategySpec, \
    compile_strategy

DEFAULT_MODEL = "claude-haiku-5-5"

SYSTEM = """You are the research assistant in quantlab, a backtesting tool \
whose purpose is telling real edges from fitted curves.

You propose trading-strategy configurations one at a time with the \
run_backtest tool. The engine runs each on the research window of the \
data and returns its measurements. You never see the final part of the data: \
it is a holdout, and your chosen trial will be judged on it exactly once.

What generalises to unseen data:
- Positive out-of-sample Sharpe *within* the research window, not just a \
high overall Sharpe.
- Consistency across walk-forward folds.
- Beating buy-and-hold after costs.
- Settings in a broad region that works, not a single sharp peak. Nudging a \
parameter by one to chase a better number is fitting noise; avoid it.
- Enough trades for the result to mean something.

Use your trials to explore genuinely different ideas before refining one. \
When you are done, or when told the budget is spent, call finish with the \
trial number you trust most to hold up on unseen data, and brief notes on \
why. Picking a trial that failed is allowed if nothing passed; say so.

The notes are commentary, displayed beside the engine's measurements. Do not \
state metric values in them (no Sharpe, return or drawdown figures): the \
interface shows the measured numbers, and anything you write is not a \
measurement. Refer to trials by number."""


def tools(max_trials):
    params_doc = "; ".join(
        f"{k}: {', '.join(f'{p} (default {v})' for p, v in DEFAULTS[k].items())}"
        for k in KINDS)
    return [
        {
            "name": "run_backtest",
            "description": (
                "Backtest one configuration on the research window and get "
                f"its measurements. At most {max_trials} trials per session. "
                f"Parameters by kind — {params_doc}. Omitted parameters use "
                "the default."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "hypothesis": {
                        "type": "string",
                        "description": "One sentence: what this trial tests.",
                    },
                },
                "required": ["kind", "hypothesis"],
            },
        },
        {
            "name": "finish",
            "description": "End the session and nominate one trial for the "
                           "holdout test.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pick": {"type": "integer",
                             "description": "Trial number to evaluate."},
                    "notes": {"type": "string",
                              "description": "Why this trial, qualitatively. "
                                             "No metric values."},
                },
                "required": ["pick", "notes"],
            },
        },
    ]


# ── Keeping the model's prose honest ───────────────────────────────────────

_METRIC_WORDS = (r"sharpe|sortino|return|returns|drawdown|profit factor|"
                 r"win rate|cagr|excess|t-stat|t stat")
_PATTERNS = [
    re.compile(r"[-+]?\d+(?:\.\d+)?\s*%"),
    re.compile(rf"\b(?:{_METRIC_WORDS})\b[^.\n]{{0,24}}?[-+]?\d+(?:\.\d+)?",
               re.IGNORECASE),
    re.compile(rf"[-+]?\d+(?:\.\d+)?\s*(?:{_METRIC_WORDS})\b", re.IGNORECASE),
]


def scrub_metrics(text, limit=1200):
    """Remove metric figures from model prose.

    Instructed not to, a model will still sometimes write "a Sharpe of 1.4".
    That number would sit on the page looking like a measurement. The UI
    shows the engine's numbers; prose keeps parameter values ("fast 10")
    but loses anything that reads as a result.
    """
    text = (text or "")[:limit]
    for pat in _PATTERNS:
        text = pat.sub("[see table]", text)
    return text


# ── Measurements the model sees ────────────────────────────────────────────

def _r(x, nd=2):
    return None if x is None or x != x or x in (float("inf"), float("-inf")) \
        else round(x, nd)


def summarise(result):
    """The compact view of a research-window backtest sent to the model and
    shown in the trials table. Rounded: more digits invite fitting noise."""
    return {
        "passed": result["passed"],
        "failed_checks": [n for n, ok, _ in result["checks"] if not ok],
        "sharpe": _r(result["sharpe"]),
        "oos_sharpe": _r(result["oos_sharpe"]),
        "in_sample_sharpe": _r(result["in_sample"]["sharpe"]),
        "excess_return_pct": _r(result["excess_return_pct"], 1),
        "max_drawdown_pct": _r(result["max_drawdown_pct"], 1),
        "trades": result["trades"],
        "profit_factor": _r(result["profit_factor"]),
        "folds_positive": result["consistency"],
        "likely_overfit": result["likely_overfit"],
    }


def _fallback_pick(trials):
    """If the model never nominates a valid trial, the engine picks: best
    research-window out-of-sample Sharpe, preferring trials that passed."""
    ok = [t for t in trials if t.get("summary")]
    if not ok:
        return None
    return max(ok, key=lambda t: (t["summary"]["passed"],
                                  t["summary"]["oos_sharpe"] or -1e9))["n"]


class BudgetExceeded(Exception):
    pass


# ── The loop ───────────────────────────────────────────────────────────────

def run_research(client, bars, symbol, goal="", *, model=DEFAULT_MODEL,
                 max_trials=6, token_budget=60_000, holdout_frac=0.3,
                 cash=1000.0, cost_model=None, criteria=None,
                 min_research_bars=60, min_holdout_bars=30,
                 on_progress=None):
    """Run a research session. Returns a dict; raises on unusable input.

    `on_progress(state)` is called after every trial so a poller can watch
    the session unfold.
    """
    cost_model = cost_model or {"slippage_bps": 10}
    cut = int(len(bars) * (1 - holdout_frac))
    research, holdout = bars[:cut], bars[cut:]
    if len(research) < min_research_bars or len(holdout) < min_holdout_bars:
        raise ValueError(
            f"{symbol}: {len(bars)} bars is too few to split into a research "
            f"window of {min_research_bars}+ and a holdout of "
            f"{min_holdout_bars}+")

    state = {
        "symbol": symbol, "goal": goal, "model": model,
        "research_window": [research[0]["t"].date().isoformat(),
                            research[-1]["t"].date().isoformat()],
        "holdout_window": [holdout[0]["t"].date().isoformat(),
                           holdout[-1]["t"].date().isoformat()],
        "trials": [], "input_tokens": 0, "output_tokens": 0,
        "stop_reason": None,
    }

    def progress():
        if on_progress:
            on_progress(state)

    intro = (f"Symbol: {symbol}. Research window: {state['research_window'][0]}"
             f" to {state['research_window'][1]} ({len(research)} daily bars). "
             f"Costs: {cost_model.get('slippage_bps', 0)} bps slippage per side. "
             f"You have {max_trials} trials.")
    if goal:
        intro += f"\n\nThe user's goal, in their words: {goal[:300]}"
    messages = [{"role": "user", "content": intro}]
    tool_defs = tools(max_trials)
    pick, notes = None, ""

    for _turn in range(max_trials + 2):
        spent = state["input_tokens"] + state["output_tokens"]
        final_turn = (len(state["trials"]) >= max_trials
                      or spent >= token_budget)
        if spent >= token_budget * 1.5:
            state["stop_reason"] = "token budget"
            break
        choice = ({"type": "tool", "name": "finish"} if final_turn
                  else {"type": "any"})

        resp = client.messages.create(
            model=model, max_tokens=1024, system=SYSTEM, tools=tool_defs,
            tool_choice=choice, messages=messages)
        state["input_tokens"] += resp.usage.input_tokens
        state["output_tokens"] += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            state["stop_reason"] = "model stopped without a tool call"
            break

        results, finished = [], False
        for use in uses:
            if use.name == "finish":
                finished = True
                pick = use.input.get("pick")
                notes = use.input.get("notes", "")
                results.append(_tool_result(use.id, "Session finished."))
                continue
            if len(state["trials"]) >= max_trials:
                results.append(_tool_result(
                    use.id, "Trial budget spent. Call finish.", error=True))
                continue
            trial = _run_trial(len(state["trials"]) + 1, use.input, research,
                               cash, cost_model, criteria)
            state["trials"].append(trial)
            body = trial["summary"] if trial.get("summary") else \
                {"error": trial["error"]}
            results.append(_tool_result(
                use.id, json.dumps({"trial": trial["n"], **body}),
                error="error" in trial))
            progress()
        messages.append({"role": "user", "content": results})
        if finished:
            state["stop_reason"] = "model finished"
            break
    else:
        state["stop_reason"] = state["stop_reason"] or "turn limit"

    valid = {t["n"] for t in state["trials"] if t.get("summary")}
    state["picked_by"] = "model" if pick in valid else "engine fallback"
    if pick not in valid:
        pick = _fallback_pick(state["trials"])
    state["notes"] = scrub_metrics(notes)

    if pick is None:
        state["final"] = None
        progress()
        return state

    chosen = next(t for t in state["trials"] if t["n"] == pick)
    state["final"] = _holdout(chosen, holdout, cash, cost_model, criteria,
                              len(state["trials"]))
    progress()
    return state


def _tool_result(tool_use_id, content, error=False):
    out = {"type": "tool_result", "tool_use_id": tool_use_id,
           "content": content}
    if error:
        out["is_error"] = True
    return out


def _run_trial(n, raw, bars, cash, cost_model, criteria):
    """One proposal through the engine. Invalid proposals become an error
    the model can read and correct, not an exception."""
    from . import backtest
    kind = raw.get("kind")
    params = raw.get("params") or {}
    trial = {"n": n, "kind": kind, "params": params,
             "hypothesis": (raw.get("hypothesis") or "")[:300]}
    try:
        if kind not in KINDS:
            raise StrategyError(f"unknown kind {kind!r}")
        unknown = set(params) - set(DEFAULTS[kind])
        if unknown:
            raise StrategyError(f"unknown params for {kind}: {sorted(unknown)}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in params.values()):
            raise StrategyError("parameter values must be numbers")
        if any(not 0 < v <= 1000 for v in params.values()):
            raise StrategyError("parameter values must be between 0 and 1000")
        spec = StrategySpec(kind=kind, params=params)
        compile_strategy(spec)
        result = backtest(bars, spec, cash=cash, cost_model=cost_model,
                          criteria=criteria)
    except Exception as e:
        # Model input: whatever went wrong goes back to the model as a
        # readable error rather than ending the session.
        trial["error"] = f"{type(e).__name__}: {e}"[:300]
        return trial
    trial["params"] = StrategySpec(kind=kind, params=params).resolved()
    trial["summary"] = summarise(result)
    return trial


def _holdout(trial, holdout, cash, cost_model, criteria, n_trials):
    """The one-shot test on bars the model never saw."""
    signal = compile_strategy(StrategySpec(kind=trial["kind"],
                                           params=trial["params"]))
    res = engine.evaluate(holdout, signal, cash, cost_model)
    passed, checks = engine.grade(res, criteria)
    return {
        "trial": trial["n"], "kind": trial["kind"], "params": trial["params"],
        "passed": passed,
        "checks": [{"name": n, "passed": ok, "detail": d}
                   for n, ok, d in checks],
        "total_return_pct": _r(res["total_return_pct"]),
        "benchmark_return_pct": _r(res["benchmark_return_pct"]),
        "excess_return_pct": _r(res["excess_return_pct"]),
        "sharpe": _r(res["sharpe"]),
        "max_drawdown_pct": _r(res["max_drawdown_pct"]),
        "trades": res["trades"],
        "bars": res["bars"],
        "trials_tried": n_trials,
    }
