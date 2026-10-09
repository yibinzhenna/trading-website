"""
Server wiring: settings, provider construction, shared singletons.

Providers are built per request from a small cache rather than at import
time, so a missing API key only breaks the endpoints that need it instead of
preventing the server from starting.
"""

import os

from quantlab.cache import BarCache, CachedProvider
from quantlab.jobs import JobStore
from quantlab.providers import ProviderError, get_provider


class Settings:
    def __init__(self):
        self.provider = os.getenv("QUANTLAB_PROVIDER", "local")
        # sampledata/ is committed, so a fresh clone or deploy works with
        # no key and no provider account. Point this at your own data.
        self.data_root = os.getenv("QUANTLAB_DATA_ROOT", "sampledata")
        # Hosts with an ephemeral or read-only filesystem should point this
        # somewhere writable, or accept that every run refetches.
        self.cache_root = os.getenv("QUANTLAB_CACHE_ROOT", "cache")
        self.cache_ttl = int(os.getenv("QUANTLAB_CACHE_TTL", 60 * 60 * 12))
        self.workers = int(os.getenv("QUANTLAB_WORKERS", 2))
        self.max_bars = int(os.getenv("QUANTLAB_MAX_BARS", 5000))
        self.min_bars = int(os.getenv("QUANTLAB_MIN_BARS", 60))


settings = Settings()
cache = BarCache(settings.cache_root, settings.cache_ttl)
jobs = JobStore(workers=settings.workers)

_PROVIDER_KWARGS = {"local": lambda s: {"root": s.data_root}}
_providers = {}


def build_provider(name=None):
    """A cached, cache-wrapped provider. Raises ProviderError if unusable."""
    key = (name or settings.provider).lower()
    if key not in _providers:
        kwargs = _PROVIDER_KWARGS.get(key, lambda s: {})(settings)
        _providers[key] = CachedProvider(get_provider(key, **kwargs), cache)
    return _providers[key]


def provider_status():
    """Which providers this server can actually serve, and why not."""
    out = []
    for name in ("local", "alphavantage"):
        try:
            build_provider(name)
            out.append({"name": name, "available": True, "detail": ""})
        except ProviderError as e:
            out.append({"name": name, "available": False, "detail": str(e)})
        except Exception as e:
            out.append({"name": name, "available": False,
                        "detail": f"{type(e).__name__}: {e}"})
    return out


def reset_for_tests(**overrides):
    """Point the module at temp dirs and a fresh job pool.

    The job store is rebuilt, not reused: app shutdown drains its thread pool,
    so a second test module sharing the singleton would hit "cannot schedule
    new futures after shutdown" the moment it submitted anything.
    """
    global cache, jobs
    for k, v in overrides.items():
        setattr(settings, k, v)
    cache = BarCache(settings.cache_root, settings.cache_ttl)
    jobs = JobStore(workers=settings.workers)
    _providers.clear()
    return cache
