"""
Data provider interface.

The backtest engine never talks to a vendor directly. It asks a provider for
bars and gets back a plain list of dicts, so swapping Alpha Vantage for Tiingo
or a local file changes one line of wiring and nothing else.

A Bar is a dict, deliberately, rather than a dataclass or DataFrame: it keeps
the engine dependency-free and makes fixtures trivial to write by hand.

    {"t": datetime, "o": float, "h": float, "l": float, "c": float, "v": float}

`t` is timezone-aware UTC. Bars are returned oldest-first.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class ProviderError(RuntimeError):
    """Raised when a provider cannot return usable data.

    Distinct from an empty result: an empty list means the vendor had nothing
    for that symbol and window, which is a legitimate answer. This means the
    request failed — bad key, rate limit, network, malformed response.
    """


class DataProvider(ABC):
    """Anything the engine can backtest against."""

    name = "base"

    @abstractmethod
    def bars(self, symbol, interval="day", limit=None):
        """Historical OHLCV, oldest first.

        :param symbol:   ticker, e.g. "SPY"
        :param interval: "day" | "hour" | "5min"
        :param limit:    keep only the most recent N bars, after fetching
        :returns:        list[Bar]; empty if the vendor has nothing
        :raises ProviderError: the request itself failed
        """

    def symbols(self):
        """Symbols this provider can serve, or None if it cannot enumerate.

        A remote vendor covers thousands of tickers and offers no cheap way to
        list them, so None means "any ticker, ask and see" — not "none".
        """
        return None

    # ── helpers for implementations ────────────────────────────────────────

    @staticmethod
    def _parse_ts(value):
        """Vendor timestamps into aware UTC datetimes.

        Accepts ISO strings with or without a zone, with Z or an offset, and
        plain dates. A naive timestamp is assumed UTC rather than local, so
        results do not change with the machine running the backtest.
        """
        if isinstance(value, datetime):
            dt = value
        else:
            text = str(value).strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                dt = datetime.strptime(text[:10], "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None \
            else dt.astimezone(timezone.utc)

    @classmethod
    def _bar(cls, t, o, h, l, c, v=0.0):
        """Build a Bar, or return None if any field is unusable.

        Providers skip the Nones. One malformed row should not abort a fetch,
        but it must never reach the engine as a zero.
        """
        try:
            bar = {"t": cls._parse_ts(t), "o": float(o), "h": float(h),
                   "l": float(l), "c": float(c), "v": float(v or 0.0)}
        except (TypeError, ValueError):
            return None
        if bar["c"] <= 0 or bar["o"] <= 0:
            return None
        return bar

    @staticmethod
    def _finish(bars, limit=None):
        """Sort oldest-first, drop duplicate timestamps, apply the limit."""
        seen, out = set(), []
        for b in sorted(bars, key=lambda x: x["t"]):
            if b["t"] in seen:
                continue
            seen.add(b["t"])
            out.append(b)
        return out[-limit:] if limit else out
