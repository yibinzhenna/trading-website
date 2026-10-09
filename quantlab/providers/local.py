"""
Local file provider — CSV and JSONL, no network, no API key.

Two jobs:

1. Hermetic tests. Every engine test runs against fixtures on disk, so the
   suite needs no key, no network, and gives the same answer every run.
2. Reading data you already recorded. The JSONL layout matches what the
   desktop recorder writes, so captured sessions are backtestable directly.
"""

import csv
import json
import os

from .base import DataProvider, ProviderError


class LocalProvider(DataProvider):
    """Reads bars from a directory of per-symbol files.

    Looks for ``<root>/<SYMBOL>.jsonl``, ``<SYMBOL>.csv``, or any
    ``<SYMBOL>_*.jsonl`` (the recorder's one-file-per-day layout), and
    concatenates them in order.
    """

    name = "local"

    # Accepts the recorder's short keys and conventional long ones alike.
    FIELDS = {
        "t": ("t", "time", "timestamp", "date", "datetime", "begins_at"),
        "o": ("o", "open", "open_price"),
        "h": ("h", "high", "high_price"),
        "l": ("l", "low", "low_price"),
        "c": ("c", "close", "close_price", "adj_close"),
        "v": ("v", "volume"),
    }

    def __init__(self, root="data"):
        self.root = root

    def _files(self, symbol):
        if not os.path.isdir(self.root):
            raise ProviderError(f"No such data directory: {self.root}")
        sym = symbol.upper()
        hits = []
        for name in sorted(os.listdir(self.root)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".jsonl", ".csv", ".json"):
                continue
            if stem.upper() == sym or stem.upper().startswith(sym + "_"):
                hits.append(os.path.join(self.root, name))
        return hits

    @classmethod
    def _pick(cls, row, key):
        for alias in cls.FIELDS[key]:
            if alias in row and row[alias] not in (None, ""):
                return row[alias]
        return None

    def _row_to_bar(self, row):
        t = self._pick(row, "t")
        c = self._pick(row, "c")
        if t is None or c is None:
            return None
        # Bar-only sources sometimes omit OHLV; fall back to close so a
        # close-only series still backtests rather than being discarded.
        return self._bar(
            t,
            self._pick(row, "o") or c,
            self._pick(row, "h") or c,
            self._pick(row, "l") or c,
            c,
            self._pick(row, "v") or 0.0,
        )

    def bars(self, symbol, interval="day", limit=None):
        out = []
        for path in self._files(symbol):
            try:
                with open(path, encoding="utf-8") as fh:
                    if path.lower().endswith(".csv"):
                        rows = csv.DictReader(fh)
                    else:
                        rows = (json.loads(line) for line in fh if line.strip())
                    for row in rows:
                        bar = self._row_to_bar(row)
                        if bar:
                            out.append(bar)
            except (OSError, json.JSONDecodeError) as e:
                raise ProviderError(f"Reading {path}: {e}") from e
        return self._finish(out, limit)
