"""
Regenerate sampledata/DEMO-REGIME.jsonl.

    python scripts/make_sampledata.py

The recipe is fixed: seed 21, alternating 90-bar up and down segments, drift
+/-0.4% a day, 1% daily volatility, business days ending 2025-12-01. Only
the length was chosen, and it was chosen after testing (see
sampledata/README.md): 3,780 bars is the shortest of the lengths tried at
which a 30% holdout holds enough trend trades for the significance gate.

The three random-walk series (DEMO-TREND, -CHOP, -BEAR) were generated
separately and are not reproduced here.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

BARS = 3780
END = date(2025, 12, 1)
SEED = 21
SEGMENT = 90
DRIFT = 0.004
VOL = 0.010


def business_days(n, end):
    days, d = [], end
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days[::-1]


def regime_rows(n=BARS):
    rng = random.Random(SEED)
    px, rows = 100.0, []
    for i, day in enumerate(business_days(n, END)):
        drift = DRIFT if (i // SEGMENT) % 2 == 0 else -DRIFT
        px *= 1 + rng.gauss(drift, VOL)
        rows.append({"t": f"{day.isoformat()}T00:00:00Z",
                     "o": round(px * 0.999, 4), "h": round(px * 1.006, 4),
                     "l": round(px * 0.994, 4), "c": round(px, 4),
                     "v": 1_000_000})
    return rows


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sampledata" / "DEMO-REGIME.jsonl"
    rows = regime_rows()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{out.name}: {len(rows)} bars, {rows[0]['t'][:10]} to {rows[-1]['t'][:10]}")
