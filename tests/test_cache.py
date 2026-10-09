"""Cache tests — hermetic, tmp_path only."""

import time

import pytest

from quantlab.cache import BarCache, CachedProvider
from quantlab.providers import LocalProvider, ProviderError

BARS = LocalProvider("tests/fixtures").bars("SPY")


def test_write_then_read_roundtrip(tmp_path):
    c = BarCache(str(tmp_path))
    assert c.write("local", "SPY", "day", BARS) == len(BARS)
    got = c.read("local", "SPY", "day")
    assert len(got) == len(BARS)
    assert got[0]["t"] == BARS[0]["t"]
    assert got[0]["c"] == pytest.approx(BARS[0]["c"])


def test_miss_returns_none(tmp_path):
    assert BarCache(str(tmp_path)).read("local", "NOPE", "day") is None


def test_expired_entry_is_a_miss(tmp_path):
    c = BarCache(str(tmp_path), ttl=0)
    c.write("local", "SPY", "day", BARS)
    time.sleep(0.01)
    assert c.read("local", "SPY", "day") is None


def test_stale_entry_readable_when_explicitly_allowed(tmp_path):
    c = BarCache(str(tmp_path), ttl=0)
    c.write("local", "SPY", "day", BARS)
    time.sleep(0.01)
    assert c.read("local", "SPY", "day", allow_stale=True) is not None


def test_clear_removes_entries(tmp_path):
    c = BarCache(str(tmp_path))
    c.write("local", "SPY", "day", BARS)
    assert c.stats()["entries"] == 1
    assert c.clear() > 0
    assert c.stats()["entries"] == 0


def test_symbols_do_not_collide(tmp_path):
    c = BarCache(str(tmp_path))
    c.write("local", "SPY", "day", BARS)
    c.write("local", "QQQ", "day", BARS[:10])
    assert len(c.read("local", "SPY", "day")) == len(BARS)
    assert len(c.read("local", "QQQ", "day")) == 10


def test_intervals_do_not_collide(tmp_path):
    c = BarCache(str(tmp_path))
    c.write("local", "SPY", "day", BARS)
    c.write("local", "SPY", "5min", BARS[:5])
    assert len(c.read("local", "SPY", "day")) == len(BARS)
    assert len(c.read("local", "SPY", "5min")) == 5


# ── CachedProvider ─────────────────────────────────────────────────────────

class Counting:
    """Counts upstream fetches so we can prove the cache is used."""
    name = "counting"

    def __init__(self, bars=None, fail=False):
        self.calls = 0
        self._bars = bars if bars is not None else BARS
        self.fail = fail

    def bars(self, symbol, interval="day", limit=None):
        self.calls += 1
        if self.fail:
            raise ProviderError("upstream is down")
        return self._bars


def test_second_call_does_not_hit_upstream(tmp_path):
    up = Counting()
    p = CachedProvider(up, BarCache(str(tmp_path)))
    first, second = p.bars("SPY"), p.bars("SPY")
    assert up.calls == 1
    assert len(first) == len(second) == len(BARS)


def test_limit_applies_to_cached_result(tmp_path):
    p = CachedProvider(Counting(), BarCache(str(tmp_path)))
    p.bars("SPY")
    assert len(p.bars("SPY", limit=7)) == 7


def test_stale_cache_rescues_a_failing_provider(tmp_path):
    """Yesterday's bars beat an error page."""
    cache = BarCache(str(tmp_path), ttl=0)
    good = CachedProvider(Counting(), cache)
    good.bars("SPY")
    time.sleep(0.01)
    broken = CachedProvider(Counting(fail=True), cache)
    broken.name = "counting"
    assert len(broken.bars("SPY")) == len(BARS)


def test_failure_with_no_cache_propagates(tmp_path):
    p = CachedProvider(Counting(fail=True), BarCache(str(tmp_path)))
    with pytest.raises(ProviderError):
        p.bars("SPY")


def test_empty_upstream_is_not_cached(tmp_path):
    up = Counting(bars=[])
    p = CachedProvider(up, BarCache(str(tmp_path)))
    p.bars("SPY")
    p.bars("SPY")
    assert up.calls == 2, "an empty result must not be cached as an answer"


def test_concurrent_writes_to_one_entry_do_not_collide(tmp_path):
    """Two requests filling the same entry used to share a temp file."""
    import threading
    from datetime import datetime, timedelta
    cache = BarCache(str(tmp_path), ttl=3600)
    t0 = datetime(2025, 1, 1)
    bars = [{"t": t0 + timedelta(days=i), "o": 1.0, "h": 1.0, "l": 1.0,
             "c": 1.0, "v": 0} for i in range(500)]
    errors = []

    def write():
        try:
            cache.write("local", "SPY", "day", bars)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(cache.read("local", "SPY", "day")) == 500
    assert not list(tmp_path.rglob("*.tmp"))
