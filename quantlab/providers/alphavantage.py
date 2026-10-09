"""
Alpha Vantage provider.

Only ``TIME_SERIES_DAILY`` at ``outputsize=compact`` works on the free tier —
100 daily bars. Probed and confirmed: intraday, ``outputsize=full``, and
``HISTORICAL_OPTIONS`` all return a premium-endpoint notice rather than data.

Alpha Vantage answers with HTTP 200 and an explanatory body when it refuses,
so a bare status check passes on a request that returned nothing usable.
Those bodies are detected and raised as ProviderError.
"""

import json
import os
import urllib.parse
import urllib.request

from .base import DataProvider, ProviderError

BASE = "https://www.alphavantage.co/query"

# Keys Alpha Vantage uses to explain a refusal instead of returning data.
_REFUSAL_KEYS = ("Note", "Information", "Error Message")

_SERIES = {
    "day": ("TIME_SERIES_DAILY", "Time Series (Daily)", None),
    "hour": ("TIME_SERIES_INTRADAY", "Time Series (60min)", "60min"),
    "5min": ("TIME_SERIES_INTRADAY", "Time Series (5min)", "5min"),
    "1min": ("TIME_SERIES_INTRADAY", "Time Series (1min)", "1min"),
}


class AlphaVantageProvider(DataProvider):
    """Daily bars on the free tier; intraday needs a premium key."""

    name = "alphavantage"

    def __init__(self, api_key=None, outputsize="compact", timeout=40):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ProviderError("Set ALPHAVANTAGE_API_KEY or pass api_key.")
        self.outputsize = outputsize
        self.timeout = timeout

    def _get(self, **params):
        params["apikey"] = self.api_key
        url = BASE + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode())
        except Exception as e:
            raise ProviderError(f"Alpha Vantage request failed: {e}") from e

        for key in _REFUSAL_KEYS:
            if key in payload:
                raise ProviderError(f"Alpha Vantage [{key}]: {payload[key]}")
        return payload

    def bars(self, symbol, interval="day", limit=None):
        if interval not in _SERIES:
            raise ProviderError(
                f"Unsupported interval {interval!r}; "
                f"expected one of {sorted(_SERIES)}")
        function, series_key, av_interval = _SERIES[interval]

        params = {"function": function, "symbol": symbol.upper(),
                  "outputsize": self.outputsize}
        if av_interval:
            params["interval"] = av_interval

        payload = self._get(**params)
        series = payload.get(series_key)
        if series is None:
            # Shape changed, or the response was something unanticipated.
            keys = ", ".join(list(payload)[:4])
            raise ProviderError(
                f"No '{series_key}' in response (saw: {keys})")

        out = []
        for stamp, row in series.items():
            bar = self._bar(stamp,
                            row.get("1. open"), row.get("2. high"),
                            row.get("3. low"), row.get("4. close"),
                            row.get("5. volume"))
            if bar:
                out.append(bar)
        return self._finish(out, limit)
