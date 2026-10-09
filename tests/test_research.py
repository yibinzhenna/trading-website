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
    # The model was told what went wrong, flagged as an error.
    # [0] intro, [1] first proposal, [2] its result.
    results = client.requests[-1]["messages"][2]["content"]
    assert results[0]["is_error"] and "astrology" in results[0]["content"]


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
