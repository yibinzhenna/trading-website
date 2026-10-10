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

Token economy
-------------
Every call is **stateless**: one user message holding a compact trial log,
rather than a replayed conversation of tool calls and results. The model
sees the same information — every configuration and its measurements — at
a fraction of the tokens, and there is no prior turn whose reasoning would
have to be passed back.

Each call's message is the previous call's message plus new log lines,
appended. System prompt and tools never vary, not even with the trial
budget. Both properties matter to prefix caching (DeepSeek's is automatic
and keyed on exact prefixes): almost all of every call after the first is a
repeat.

The model client is injected (anything with `messages.create(...)` shaped
like the Anthropic SDK), so the loop is testable with a scripted fake. It
serves both Anthropic and DeepSeek's Anthropic-compatible endpoint.
"""

import re

from . import engine
from .strategies import DEFAULTS, KINDS, StrategySpec, compile_strategy

MODELS = {"anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"}
DEFAULT_MODEL = MODELS["anthropic"]

# A tool call is ~70 tokens and finish notes ~150. Generous for that, tight
# enough that a model which starts thinking out loud is cut short.
MAX_OUTPUT_TOKENS = 600
HYPOTHESIS_CHARS = 100
NOTES_CHARS = 600

SYSTEM = """You are quantlab's strategy research assistant. Find a configuration that will hold up on data you cannot see.

Each run_backtest call tests one configuration on the research window and adds a row to the trial log. When the log's trial budget is spent, call finish. Your pick is then tested once on a later holdout you never see; only that test decides PASS or FAIL.

Row fields: S Sharpe; is/oos Sharpe on the window's first 70%/last 30%; xs % return over buy-and-hold; dd % max drawdown; n trades; pf profit factor; folds positive walk-forward folds; then PASS or the failed gates; overfit means is>0 but oos<=0.

Prefer oos>0, most folds positive, xs>0, enough trades to be significant, and a broad working region of parameters over a sharp peak.

Rules:
1. Explore different kinds before refining one. Never repeat a logged configuration.
2. Change parameters meaningfully; fast 10 to 11 only fits noise.
3. hypothesis: at most 12 words.
4. finish: the trial most likely to survive the holdout, even if it failed. notes: at most 3 sentences, trials by number, no metric values."""


def _param_doc():
    return "; ".join(
        f"{k}: " + ", ".join(f"{p}={v}" for p, v in DEFAULTS[k].items())
        for k in KINDS)


TOOLS = [
    {
        "name": "run_backtest",
        "description": (f"Test one configuration. Defaults: {_param_doc()}. "
                        "Omitted parameters use the default. Example: "
                        '{"kind":"breakout","params":{"lookback":40},'
                        '"hypothesis":"Longer channels avoid false breaks"}'),
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(KINDS)},
                "params": {"type": "object",
                           "additionalProperties": {"type": "number"}},
                "hypothesis": {"type": "string"},
            },
            "required": ["kind", "hypothesis"],
        },
    },
    {
        "name": "finish",
        "description": "Nominate one trial for the holdout test.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pick": {"type": "integer"},
                "notes": {"type": "string"},
            },
            "required": ["pick", "notes"],
        },
    },
]


def tools(_max_trials=None):
    """The tool definitions. Identical for every session, so the cached
    prefix is shared across sessions too; the budget lives in the log."""
    return TOOLS


# ── Keeping the model's prose honest ───────────────────────────────────────

_METRIC_WORDS = (r"sharpe|sortino|return|returns|drawdown|profit factor|"
                 r"win rate|cagr|excess|t-stat|t stat")
_PATTERNS = [
    re.compile(r"[-+]?\d+(?:\.\d+)?\s*%"),
    re.compile(rf"\b(?:{_METRIC_WORDS})\b[^.\n]{{0,24}}?[-+]?\d+(?:\.\d+)?",
               re.IGNORECASE),
    re.compile(rf"[-+]?\d+(?:\.\d+)?\s*(?:{_METRIC_WORDS})\b", re.IGNORECASE),
]


# Numbers that are references, not results: fold counts ("4/4 folds") and
# trial numbers ("Trials 1, 2 and 6", "#3"). Shielded before scrubbing, or a
# metric pattern's look-ahead swallows them: "Sharpe across 4/4 folds" came
# out as "[see table]/4 folds".
_PROTECTED = re.compile(
    r"\b\d+\s*/\s*\d+\s*folds?\b"
    r"|\btrials?\s+\d+(?:\s*(?:,|and|&|or)\s*\d+)*"
    r"|#\d+",
    re.IGNORECASE)
_SHIELD = 0xE000          # Unicode private use: never in model prose


def scrub_metrics(text, limit=NOTES_CHARS):
    """Remove metric figures from model prose.

    Instructed not to, a model will still sometimes write "a Sharpe of 1.4".
    That number would sit on the page looking like a measurement. The UI
    shows the engine's numbers; prose keeps parameter values ("fast 10"),
    trial numbers and fold counts, but loses anything that reads as a result.
    """
    text = (text or "")[:limit]
    shielded = []

    def shield(m):
        shielded.append(m.group(0))
        return chr(_SHIELD + len(shielded) - 1)

    text = _PROTECTED.sub(shield, text)
    for pat in _PATTERNS:
        text = pat.sub("[see table]", text)
    for i, original in enumerate(shielded):
        text = text.replace(chr(_SHIELD + i), original)
    return text


def _cap_words(text, words=12, chars=HYPOTHESIS_CHARS):
    return " ".join((text or "").split()[:words])[:chars]


# ── Measurements ───────────────────────────────────────────────────────────

def _r(x, nd=2):
    return None if x is None or x != x or x in (float("inf"), float("-inf")) \
        else round(x, nd)


def summarise(result):
    """The structured view of a research-window backtest, kept in the
    session state for the trials table and the fallback pick."""
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


_GATE_SHORT = (("Sharpe", "sharpe"), ("Max drawdown", "drawdown"),
               ("Profit factor", "pf"), ("Trades", "trades"),
               ("Edge significant", "significance"), ("Beats", "benchmark"))


def _gate(name):
    return next((short for prefix, short in _GATE_SHORT
                 if name.startswith(prefix)), name)


def _num(v, nd=2, sign=False):
    if v is None:
        return "inf"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def row(trial, left):
    """One trial as a log line: ~40 tokens where the JSON was ~100.

    #3 trend_following fast=20 slow=60 | S 1.23 is 1.50 oos 0.81 xs +12.4
    dd 6.1 n 9 pf 1.80 folds 3/4 | FAIL sharpe,trades | 3 left
    """
    def val(v):    # error rows can hold whatever the model sent
        return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(
            v, bool) else repr(v)[:20]
    params = " ".join(f"{k}={val(v)}" for k, v in (trial["params"] or {}).items())
    head = f"#{trial['n']} {trial['kind']} {params}".rstrip()
    m = trial.get("summary")
    if not m:
        return f"{head} | ERROR {trial['error']} | {left} left"
    verdict = "PASS" if m["passed"] else \
        "FAIL " + ",".join(_gate(n) for n in m["failed_checks"])
    if m["likely_overfit"]:
        verdict += " overfit"
    return (f"{head} | S {_num(m['sharpe'])} is {_num(m['in_sample_sharpe'])} "
            f"oos {_num(m['oos_sharpe'])} xs {_num(m['excess_return_pct'], 1, True)} "
            f"dd {_num(m['max_drawdown_pct'], 1)} n {m['trades']} "
            f"pf {_num(m['profit_factor'])} folds {m['folds_positive']} "
            f"| {verdict} | {left} left")


def _fallback_pick(trials):
    """If the model never nominates a valid trial, the engine picks: best
    research-window out-of-sample Sharpe, preferring trials that passed."""
    ok = [t for t in trials if t.get("summary")]
    if not ok:
        return None
    return max(ok, key=lambda t: (t["summary"]["passed"],
                                  t["summary"]["oos_sharpe"] or -1e9))["n"]


# ── The loop ───────────────────────────────────────────────────────────────

def run_research(client, bars, symbol, goal="", *, model=DEFAULT_MODEL,
                 max_trials=6, token_budget=60_000, holdout_frac=0.3,
                 cash=1000.0, cost_model=None, criteria=None,
                 min_research_bars=60, min_holdout_bars=30,
                 request_options=None, on_progress=None):
    """Run a research session. Returns a dict; raises on unusable input.

    `request_options` are passed to every `messages.create` call: how the
    provider is asked not to think out loud, its temperature, and so on.
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
        "trials": [], "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "thinking_blocks": 0, "stop_reason": None,
    }

    def progress():
        if on_progress:
            on_progress(state)

    # Stable first, variable last: the header never changes within a
    # session, and log lines are only ever appended after it.
    log = [
        f"Symbol {symbol}. Research window {state['research_window'][0]} to "
        f"{state['research_window'][1]}, {len(research)} daily bars. Costs "
        f"{cost_model.get('slippage_bps', 0)} bps slippage per side. Trial "
        f"budget {max_trials}.",
    ]
    if goal:
        log.append(f"User's goal: {goal[:300]}")
    log.append("Trial log:")
    seen = {}
    pick, notes = None, ""
    options = request_options or {}

    for _call in range(max_trials + 2):
        spent = state["input_tokens"] + state["output_tokens"]
        if spent >= token_budget * 1.5:
            state["stop_reason"] = "token budget"
            break
        final_turn = len(state["trials"]) >= max_trials or spent >= token_budget
        if final_turn and log[-1] != "Budget spent: call finish.":
            log.append("Budget spent: call finish.")
        choice = ({"type": "tool", "name": "finish"} if final_turn
                  else {"type": "any"})

        resp = client.messages.create(
            model=model, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM,
            tools=TOOLS, tool_choice=choice,
            messages=[{"role": "user", "content": "\n".join(log)}],
            **options)
        _account(state, resp)

        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            state["stop_reason"] = "model stopped without a tool call"
            break

        finished = False
        for use in uses:
            if use.name == "finish":
                finished = True
                pick = use.input.get("pick")
                notes = use.input.get("notes", "")
                continue
            if len(state["trials"]) >= max_trials:
                continue
            trial = _run_trial(len(state["trials"]) + 1, use.input, research,
                               cash, cost_model, criteria, seen)
            state["trials"].append(trial)
            log.append(row(trial, max_trials - len(state["trials"])))
            progress()
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
    state["final"] = _holdout(chosen, bars, cut, cash, cost_model, criteria,
                              len(state["trials"]))
    progress()
    return state


def _account(state, resp):
    """Token accounting. Cache reads are counted inside input_tokens by the
    budget (they are still tokens sent) and reported separately because
    they are billed at a small fraction of the price."""
    usage = resp.usage
    state["calls"] += 1
    state["input_tokens"] += usage.input_tokens
    state["output_tokens"] += usage.output_tokens
    state["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    # Thinking was asked to be off. If a provider ignores that, this says so
    # rather than leaving it to be inferred from a surprising bill.
    state["thinking_blocks"] += sum(
        1 for b in resp.content if getattr(b, "type", "") == "thinking")


def _run_trial(n, raw, bars, cash, cost_model, criteria, seen):
    """One proposal through the engine. Invalid proposals become an error
    the model can read and correct, not an exception."""
    from . import backtest
    kind = raw.get("kind")
    params = raw.get("params") or {}
    trial = {"n": n, "kind": kind, "params": params,
             "hypothesis": _cap_words(raw.get("hypothesis"))}
    try:
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        unknown = set(params) - set(DEFAULTS[kind])
        if unknown:
            raise ValueError(f"unknown params for {kind}: {sorted(unknown)}")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in params.values()):
            raise ValueError("parameter values must be numbers")
        if any(not 0 < v <= 1000 for v in params.values()):
            raise ValueError("parameter values must be between 0 and 1000")
        spec = StrategySpec(kind=kind, params=params)
        resolved = spec.resolved()
        key = (kind, tuple(sorted(resolved.items())))
        if key in seen:
            raise ValueError(f"duplicate of #{seen[key]}")
        compile_strategy(spec)
        result = backtest(bars, spec, cash=cash, cost_model=cost_model,
                          criteria=criteria)
    except Exception as e:
        # Model input: whatever went wrong goes back to the model as a
        # readable error rather than ending the session.
        trial["error"] = str(e)[:200]
        return trial
    seen[key] = n
    trial["params"] = resolved
    trial["summary"] = summarise(result)
    return trial


# Gates that measure how much evidence there is rather than how good it is.
_SAMPLE_GATES = ("Trades", "Edge significant")


def holdout_verdict(passed, checks):
    """"pass", "fail", or "inconclusive".

    Inconclusive: every performance gate passed, and only the sample-size
    gates failed. "Not enough trades to tell" is a different finding from
    "evidence against", and a short holdout produces it often; reporting it
    as FAIL would misstate what the test found. It is never a pass.
    """
    if passed:
        return "pass"
    failed = [c[0] if isinstance(c, tuple) else c["name"]
              for c in checks
              if not (c[1] if isinstance(c, tuple) else c["passed"])]
    if failed and all(name.startswith(_SAMPLE_GATES) for name in failed):
        return "inconclusive"
    return "fail"


def _holdout(trial, bars, cut, cash, cost_model, criteria, n_trials):
    """The one-shot test on bars the model never saw.

    Warm-started: indicators read the research window as history, so the
    strategy can act from the holdout's first bar instead of spending its
    lookback sitting idle. It cannot trade before the holdout, and nothing
    from the holdout reaches a decision made before it.
    """
    signal = compile_strategy(StrategySpec(kind=trial["kind"],
                                           params=trial["params"]))
    res = engine.evaluate(bars, signal, cash, cost_model, start=cut)
    passed, checks = engine.grade(res, criteria)
    return {
        "trial": trial["n"], "kind": trial["kind"], "params": trial["params"],
        "passed": passed,
        "verdict": holdout_verdict(passed, checks),
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
