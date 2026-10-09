"""
Disk cache for provider bars.

Built before the API rather than after, because development alone will exhaust
a free vendor tier in an afternoon — the same backtest re-run with different
parameters asks for the same bars every time.

Entries are JSONL under ``<root>/<provider>/<SYMBOL>_<interval>.jsonl`` with a
sidecar ``.meta.json`` holding the fetch time. A stale entry is still returned
when the provider fails: yesterday's bars beat an error page.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone

DEFAULT_TTL = 60 * 60 * 12          # 12h — daily bars change once a day


def _safe(part):
    """Make a path component safe without silently colliding."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(part))


class BarCache:
    def __init__(self, root="cache", ttl=DEFAULT_TTL):
        self.root = root
        self.ttl = ttl

    def _paths(self, provider, symbol, interval):
        folder = os.path.join(self.root, _safe(provider))
        stem = f"{_safe(symbol.upper())}_{_safe(interval)}"
        return (os.path.join(folder, stem + ".jsonl"),
                os.path.join(folder, stem + ".meta.json"))

    def age(self, provider, symbol, interval):
        """Seconds since the entry was written, or None if absent."""
        _, meta_path = self._paths(provider, symbol, interval)
        try:
            with open(meta_path, encoding="utf-8") as fh:
                return time.time() - float(json.load(fh)["fetched_at"])
        except (OSError, ValueError, KeyError):
            return None

    def read(self, provider, symbol, interval, allow_stale=False):
        """Cached bars, or None on a miss or expiry.

        `allow_stale` ignores the TTL — used as a fallback when the provider
        itself has failed.
        """
        age = self.age(provider, symbol, interval)
        if age is None or (not allow_stale and age > self.ttl):
            return None
        data_path, _ = self._paths(provider, symbol, interval)
        bars = []
        try:
            with open(data_path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["t"] = datetime.fromisoformat(row["t"])
                    bars.append(row)
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            return None
        return bars or None

    def write(self, provider, symbol, interval, bars):
        data_path, meta_path = self._paths(provider, symbol, interval)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # Write to a temp file then replace, so a crash mid-write cannot
        # leave a half-file that later reads as a cache hit. The temp name is
        # unique per write: two requests filling the same entry at once used
        # to share one, and interleaved their bytes.
        lines = []
        for b in bars:
            row = dict(b)
            row["t"] = b["t"].isoformat()
            lines.append(json.dumps(row, separators=(",", ":")) + "\n")
        meta = json.dumps({"fetched_at": time.time(), "rows": len(bars),
                           "written": datetime.now(timezone.utc).isoformat()})
        if not (self._atomic_write(data_path, "".join(lines))
                and self._atomic_write(meta_path, meta)):
            return 0
        return len(bars)

    @staticmethod
    def _atomic_write(path, text):
        """Write-then-rename. Returns False if a concurrent writer won the
        rename; it was storing the same bars, so losing costs nothing."""
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path),
                                   prefix=os.path.basename(path) + ".",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
            return True
        except PermissionError:
            # Windows refuses to replace a file another thread holds open.
            return False
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def clear(self, provider=None):
        """Drop cached entries. Returns how many files were removed."""
        target = os.path.join(self.root, _safe(provider)) if provider else self.root
        removed = 0
        for folder, _, names in os.walk(target):
            for name in names:
                if name.endswith((".jsonl", ".meta.json", ".tmp")):
                    os.remove(os.path.join(folder, name))
                    removed += 1
        return removed

    def stats(self):
        entries, total = [], 0
        for folder, _, names in os.walk(self.root):
            for name in names:
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(folder, name)
                size = os.path.getsize(path)
                total += size
                entries.append({
                    "provider": os.path.basename(folder),
                    "key": name[:-6],
                    "bytes": size,
                })
        return {"entries": len(entries), "bytes": total, "items": entries}


class CachedProvider:
    """Wraps any DataProvider with read-through caching.

    Satisfies the same interface, so the engine cannot tell the difference.
    """

    def __init__(self, provider, cache=None):
        self.provider = provider
        self.cache = cache or BarCache()
        self.name = getattr(provider, "name", "unknown")

    def symbols(self):
        fn = getattr(self.provider, "symbols", None)
        return fn() if fn else None

    def bars(self, symbol, interval="day", limit=None):
        hit = self.cache.read(self.name, symbol, interval)
        if hit is not None:
            return hit[-limit:] if limit else hit

        try:
            fresh = self.provider.bars(symbol, interval)
        except Exception:
            # Vendor down, rate limited, key exhausted. Stale data is a far
            # better answer than a failed backtest.
            stale = self.cache.read(self.name, symbol, interval,
                                    allow_stale=True)
            if stale is not None:
                return stale[-limit:] if limit else stale
            raise

        if fresh:
            self.cache.write(self.name, symbol, interval, fresh)
        return fresh[-limit:] if limit else fresh
