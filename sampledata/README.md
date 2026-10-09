# Sample data

Synthetic bars, so a fresh clone or deploy works with no API key and no
provider account. Real market data carries redistribution restrictions that
would make committing it to a public repository a licensing problem;
generated data carries none.

**None of these are real prices.** Do not read anything into a backtest
against them beyond what each one was built to show.

| Symbol | Built as | What a correct engine should do |
|---|---|---|
| `DEMO-REGIME` | Alternating ~90-day up and down trends | **Trend strategies pass, mean reversion fails** |
| `DEMO-TREND` | Random walk, upward drift | Everything fails |
| `DEMO-CHOP` | Random walk, no drift | Everything fails |
| `DEMO-BEAR` | Random walk, downward drift | Everything fails |

## Why three of the four should fail everything

A random walk has no exploitable structure, so no strategy has an edge on
one. If a strategy *passed* on `DEMO-TREND`, `DEMO-CHOP` or `DEMO-BEAR`, the
gates would be leaking a false positive. All three failing is the evidence
that the gates work, not a sign the site is broken.

## Why one should pass

Without a series that contains a real edge, a visitor could never see what a
PASS looks like, and could not tell "this rejects bad strategies" apart from
"this is broken." `DEMO-REGIME` has a planted trend-following edge. With
**default parameters** — nothing tuned to the data — the four trend-type
strategies pass, out-of-sample as well as in-sample. Mean reversion loses
heavily, because buying dips in a sustained downtrend is ruinous.

Being honest about how it was made: the series parameters (segment length,
drift, volatility) were chosen after trying several and keeping one where the
edge was clear. That is planting a known answer, like a unit test fixture.
What was not done is tuning the *strategy* parameters to the data — that is
the line between a fixture and a curve fit.
