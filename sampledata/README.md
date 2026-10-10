# Sample data

Synthetic bars, so a fresh clone or deploy works with no API key and no
provider account. Real market data carries redistribution restrictions that
would make committing it to a public repository a licensing problem;
generated data carries none.

**None of these are real prices.** Do not read anything into a backtest
against them beyond what each one was built to show.

| Symbol | Built as | What a correct engine should do |
|---|---|---|
| `DEMO-REGIME` | Alternating ~90-day up and down trends, 15 years | **Trend strategies pass, mean reversion fails** |
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

## Why DEMO-REGIME is 15 years long

AI research holds back the last 30% of a series as a holdout and judges the
model's pick there once. With ~90-day regimes a trend strategy trades about
once per cycle, so a short holdout cannot hold enough trades for the
significance gate: at 500 bars every pick made about one trade and nothing
could ever pass. The length was chosen after testing — 750, 1,000 and 1,260
bars all fell short, 2,520 left momentum just below the bar, and 3,780 (15
years) is the shortest tried at which the four trend strategies pass the
holdout and mean reversion fails it. As above, only the series was sized;
no strategy parameter was tuned. `scripts/make_sampledata.py` regenerates it
exactly, and a test checks the committed file against it.
