"""
The research loop, driven by a scripted fake model.

Nothing here calls a real model: the fake replays tool calls, which pins the
loop's guarantees — the holdout is never shown, budgets hold, bad proposals
are survivable, and the model's numbers never reach the page.
"""

from types import SimpleNamespace

import pytest

from quantlab.providers import LocalProvider
from quantlab.research import run_research, scrub_metrics

BARS = LocalProvider("sampledata").bars("DEMO-REGIME")


def call(name, **inp):
    return ("tool", name, inp)


class FakeClient:
    """Replays a script of tool calls. Records every request it receives."""

    def __init__(self, script, tokens=(500, 100)):
        self.script = list(script)
        self.requests = []
        self.tokens = tokens
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        forced = kw["tool_choice"].get("name")
        step = self.script.pop(0) if self.script else call(
            "finish", pick=1, notes="out of script")
        if forced == "finish" and step[1] != "finish":
            step = call("finish", pick=1, notes="forced")
        content = []
        if step[0] == "tool":
            content.append(SimpleNamespace(type="tool_use", id=f"t{len(self.requests)}",
                                           name=step[1], input=step[2]))
        else:
            content.append(SimpleNamespace(type="text", text=step[1]))
        return SimpleNamespace(content=content, usage=SimpleNamespace(
            input_tokens=self.tokens[0], output_tokens=self.tokens[1]))


def research(script, **kw):
    client = FakeClient(script)
    return client, run_research(client, BARS, "DEMO-REGIME", **kw)


TREND = call("run_backtest", kind="trend_following",
             params={"fast": 10, "slow": 30}, hypothesis="Trend persists.")
REVERT = call("run_backtest", kind="mean_reversion", params={},
              hypothesis="Prices snap back.")


def test_model_pick_is_judged_on_the_holdout():
    client, out = research([TREND, REVERT,
                            call("finish", pick=1, notes="Trend held up.")])
    assert [t["n"] for t in out["trials"]] == [1, 2]
    assert out["picked_by"] == "model" and out["final"]["trial"] == 1
    assert out["final"]["kind"] == "trend_following"
    assert out["final"]["trials_tried"] == 2
    assert out["stop_reason"] == "model finished"


def test_holdout_bars_never_reach_the_model():
    """Every trial runs on the research window, and nothing sent to the
    model mentions a date inside the holdout."""
    client, out = research([TREND, call("finish", pick=1, notes="ok")])
    holdout_start = out["holdout_window"][0]
    sent = repr([r["messages"] for r in client.requests])
    assert holdout_start not in sent
    assert out["research_window"][1] < holdout_start
    cut = int(len(BARS) * 0.7)
    assert out["final"]["bars"] == len(BARS) - cut


def test_trial_budget_forces_finish():
    client, out = research([TREND] * 10, max_trials=3)
    assert len(out["trials"]) == 3
    forced = [r["tool_choice"] for r in client.requests]
    assert forced[-1] == {"type": "tool", "name": "finish"}
    assert out["final"] is not None


def test_token_budget_forces_finish():
    client = FakeClient([TREND] * 10, tokens=(40_000, 1_000))
    out = run_research(client, BARS, "DEMO-REGIME", max_trials=8,
                       token_budget=60_000)
    assert len(out["trials"]) <= 2
    assert client.requests[-1]["tool_choice"]["name"] == "finish"


def test_bad_proposals_come_back_as_errors_not_crashes():
    client, out = research([
        call("run_backtest", kind="astrology", params={}, hypothesis="x"),
        call("run_backtest", kind="momentum", params={"lookbak": 5}, hypothesis="x"),
        call("run_backtest", kind="trend_following",
             params={"fast": 50, "slow": 10}, hypothesis="x"),
        call("run_backtest", kind="breakout", params={"lookback": 1e9}, hypothesis="x"),
        call("run_backtest", kind="breakout", params={"lookback": "20"}, hypothesis="x"),
        TREND,
        call("finish", pick=1, notes="the first one"),
    ], max_trials=8)
    errors = [t for t in out["trials"] if "error" in t]
    assert len(errors) == 5
    # Trial 1 was an error, so the model's pick is invalid: engine picks.
    assert out["picked_by"] == "engine fallback"
    assert out["final"]["trial"] == 6
    # The model was told what went wrong, in the log it reads.
    log = client.requests[-1]["messages"][0]["content"]
    assert "#1 astrology | ERROR unknown kind 'astrology'" in log
    assert "#5 breakout lookback='20' | ERROR" in log


# ── Token economy ──────────────────────────────────────────────────────────

SIX = [call("run_backtest", kind=k, params=p, hypothesis="h") for k, p in [
    ("trend_following", {"fast": 10, "slow": 30}), ("breakout", {"lookback": 20}),
    ("momentum", {}), ("volatility", {}), ("trend_following", {"fast": 20, "slow": 60}),
    ("mean_reversion", {})]] + [call("finish", pick=1, notes="n")]


def test_every_call_is_one_stateless_message():
    """No replayed tool calls: nothing a provider could demand back."""
    client, _ = research(list(SIX))
    for req in client.requests:
        assert len(req["messages"]) == 1
        assert req["messages"][0]["role"] == "user"
        assert isinstance(req["messages"][0]["content"], str)


def test_each_message_extends_the_last_one():
    """Append-only, so each call's message is a prefix-cache hit on the
    previous one."""
    client, _ = research(list(SIX))
    msgs = [r["messages"][0]["content"] for r in client.requests]
    for before, after in zip(msgs, msgs[1:]):
        assert after.startswith(before) and len(after) > len(before)


def test_system_and_tools_are_identical_across_sessions():
    """Different symbol, goal and budget: same cached prefix."""
    a = FakeClient([call("finish", pick=1, notes="x")])
    b = FakeClient([call("finish", pick=1, notes="x")])
    run_research(a, BARS, "DEMO-REGIME", max_trials=3)
    run_research(b, BARS[:400], "OTHER", goal="anything", max_trials=8)
    for key in ("system", "tools"):
        assert a.requests[0][key] == b.requests[0][key]


def test_log_rows_are_compact():
    client, out = research(list(SIX))
    log = client.requests[-1]["messages"][0]["content"]
    rows = [line for line in log.splitlines() if line.startswith("#")]
    assert len(rows) == 6
    assert all(len(r) < 220 for r in rows), rows   # ~60 tokens at most
    assert rows[0].startswith("#1 trend_following fast=10 slow=30 | S ")
    assert rows[0].endswith("| 5 left")
    # The model's own hypotheses stay out of what it is sent back.
    assert "hypothesis" not in log and "\nh\n" not in log


def test_session_size_stays_small():
    """A guard against the prompt quietly growing back. 3.6 characters per
    token is a fair average for this mix of English and figures."""
    client, _ = research(list(SIX))
    sent = sum(len(r["system"]) + len(str(r["tools"]))
               + len(r["messages"][0]["content"]) for r in client.requests)
    assert sent / 3.6 < 6000, f"~{sent / 3.6:.0f} input tokens per session"


def test_request_options_reach_every_call():
    opts = {"thinking": {"type": "disabled"}, "extra_body": {"temperature": 0.3}}
    client = FakeClient(list(SIX))
    run_research(client, BARS, "DEMO-REGIME", request_options=opts)
    assert all(r["thinking"] == {"type": "disabled"} for r in client.requests)
    assert all(r["extra_body"] == {"temperature": 0.3} for r in client.requests)
    assert all(r["max_tokens"] <= 600 for r in client.requests)


def test_thinking_blocks_are_counted_if_a_provider_ignores_the_request():
    class Thinker(FakeClient):
        def create(self, **kw):
            resp = super().create(**kw)
            resp.content.insert(0, SimpleNamespace(type="thinking", thinking="hmm"))
            return resp

    out = run_research(Thinker(list(SIX)), BARS, "DEMO-REGIME")
    assert out["thinking_blocks"] == out["calls"] == 7


def test_duplicate_configuration_is_refused_without_a_backtest():
    client, out = research([
        TREND,
        call("run_backtest", kind="trend_following", params={"slow": 30, "fast": 10},
             hypothesis="again"),
        call("run_backtest", kind="trend_following", params={}, hypothesis="defaults"),
        call("finish", pick=1, notes="x")])
    assert out["trials"][1]["error"] == "duplicate of #1"
    # {} resolves to the defaults fast=10 slow=30: also a duplicate.
    assert out["trials"][2]["error"] == "duplicate of #1"


def test_hypothesis_is_capped_at_twelve_words():
    client, out = research([call("run_backtest", kind="breakout",
                                 params={}, hypothesis="word " * 40),
                            call("finish", pick=1, notes="x")])
    assert len(out["trials"][0]["hypothesis"].split()) == 12


def test_engine_picks_when_the_model_never_finishes():
    client, out = research([TREND, REVERT, ("text", "I am done.")])
    assert out["stop_reason"] == "model stopped without a tool call"
    assert out["picked_by"] == "engine fallback"
    assert out["final"]["trial"] in (1, 2)


def test_no_valid_trials_means_no_verdict():
    client, out = research([
        call("run_backtest", kind="astrology", params={}, hypothesis="x"),
        call("finish", pick=1, notes="nothing worked")])
    assert out["final"] is None


def test_too_few_bars_is_refused_before_any_model_call():
    client = FakeClient([TREND])
    with pytest.raises(ValueError, match="too few"):
        run_research(client, BARS[:80], "DEMO-REGIME")
    assert client.requests == []


def test_progress_reported_per_trial():
    seen = []
    client = FakeClient([TREND, REVERT, call("finish", pick=2, notes="x")])
    run_research(client, BARS, "DEMO-REGIME",
                 on_progress=lambda s: seen.append(len(s["trials"])))
    assert seen[:2] == [1, 2] and seen[-1] == 2


def test_goal_is_passed_but_truncated():
    client = FakeClient([call("finish", pick=1, notes="x")])
    run_research(client, BARS, "DEMO-REGIME", goal="g" * 1000)
    intro = client.requests[0]["messages"][0]["content"]
    assert "g" * 300 in intro and "g" * 301 not in intro


def test_model_notes_lose_their_numbers():
    client, out = research([TREND, call(
        "finish", pick=1,
        notes="Trial 1 had a Sharpe of 1.85 and returned 34.2% with fast 10.")])
    assert "1.85" not in out["notes"] and "34.2" not in out["notes"]
    assert "fast 10" in out["notes"] and "Trial 1" in out["notes"]


@pytest.mark.parametrize("text,gone", [
    ("a sharpe ratio of 2.1", "2.1"),
    ("returns were 12%", "12%"),
    ("drawdown: -8.5", "8.5"),
    ("1.4 Sharpe", "1.4"),
    ("profit factor near 3", "near 3"),
])
def test_scrub_metrics(text, gone):
    assert gone not in scrub_metrics(text)


def test_scrub_keeps_parameters_and_trial_numbers():
    text = "Trial 3 used fast 12 and slow 40; lookback 20 was steadier."
    assert scrub_metrics(text) == text


# ── Holdout verdict ────────────────────────────────────────────────────────

from quantlab.research import holdout_verdict  # noqa: E402


def gates(**failed):
    names = ["Sharpe >= 1.00", "Max drawdown <= 30%", "Profit factor >= 1.00",
             "Trades >= 3", "Edge significant at 95%", "Beats benchmark after costs"]
    return [(n, n.split()[0] not in failed, "") for n in names]


def test_verdict_pass():
    assert holdout_verdict(True, gates()) == "pass"


def test_too_few_trades_alone_is_inconclusive_not_fail():
    assert holdout_verdict(False, gates(Trades=1)) == "inconclusive"
    assert holdout_verdict(False, gates(Trades=1, Edge=1)) == "inconclusive"


def test_any_performance_failure_is_a_fail():
    assert holdout_verdict(False, gates(Sharpe=1)) == "fail"
    assert holdout_verdict(False, gates(Trades=1, Beats=1)) == "fail"


def test_verdict_reads_api_shaped_checks_too():
    checks = [{"name": n, "passed": ok, "detail": d} for n, ok, d in gates(Edge=1)]
    assert holdout_verdict(False, checks) == "inconclusive"


def test_planted_edge_can_now_pass_research():
    """Issue #2: on 500 bars the holdout made ~1 trade and nothing could
    pass. A trend pick on the 15-year series passes the one-shot test."""
    client, out = research([TREND, call("finish", pick=1, notes="trend")])
    f = out["final"]
    assert f["verdict"] == "pass" and f["passed"] and f["trades"] >= 5


def test_wrong_pick_still_fails_research():
    client, out = research([REVERT, call("finish", pick=1, notes="dip buying")])
    assert out["final"]["verdict"] == "fail"


# ── Scrubbing keeps references intact (issue #11) ──────────────────────────

def test_fold_counts_survive_next_to_a_metric_word():
    """The live session read "positive out-of-sample [see table]/4 folds"."""
    text = ("Trials 1, 2, 4, 5 and 6 all passed with positive out-of-sample "
            "Sharpe across 4/4 folds; trial 3 failed badly.")
    assert scrub_metrics(text) == text


@pytest.mark.parametrize("text,kept,gone", [
    ("Sharpe stayed above 1.5 in trial 2 across 3/4 folds.",
     ["trial 2", "3/4 folds"], ["1.5"]),
    ("#4 beat #2 with a sharpe ratio of 2.1.", ["#4", "#2"], ["2.1"]),
    ("The trial 12 return of 40 beat trial 3.", ["trial 12", "trial 3"], ["40"]),
    ("Trials 2 and 5 returned 30% over 4/4 folds.", ["Trials 2 and 5", "4/4 folds"], ["30%"]),
])
def test_references_kept_metrics_removed(text, kept, gone):
    out = scrub_metrics(text)
    assert all(k in out for k in kept), out
    assert not any(g in out for g in gone), out


# ── Token accounting (issue #10) ───────────────────────────────────────────

class CachingClient(FakeClient):
    """Reports usage the way the Messages API does: input_tokens excludes
    tokens read from or written to the cache."""

    def create(self, **kw):
        resp = super().create(**kw)
        resp.usage = SimpleNamespace(input_tokens=100, output_tokens=50,
                                     cache_read_input_tokens=600,
                                     cache_creation_input_tokens=20)
        return resp


def test_cached_tokens_count_toward_the_total():
    out = run_research(CachingClient(list(SIX)), BARS, "DEMO-REGIME")
    calls = out["calls"]
    assert out["input_tokens"] == calls * (100 + 600 + 20)
    assert out["cache_read_tokens"] == calls * 600
    assert out["cache_read_tokens"] < out["input_tokens"]   # a share of it


def test_cached_tokens_count_toward_the_budget():
    """Uncached input alone would never reach this budget; with cache reads
    counted it is spent after three calls, and finish is forced."""
    client = CachingClient(list(SIX))
    out = run_research(client, BARS, "DEMO-REGIME", max_trials=8,
                       token_budget=2000)
    assert client.requests[-1]["tool_choice"] == {"type": "tool", "name": "finish"}
    assert len(out["trials"]) < 6
