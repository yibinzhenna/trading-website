"""
Server wiring: settings, provider construction, shared singletons.

Providers are built per request from a small cache rather than at import
time, so a missing API key only breaks the endpoints that need it instead of
preventing the server from starting.
"""

import os

from api.store import RunStore
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
        # Unset disables admin endpoints entirely: closed by default.
        self.admin_token = os.getenv("QUANTLAB_ADMIN_TOKEN", "")
        # Backtest submissions per client per window.
        self.rate_limit = int(os.getenv("QUANTLAB_RATE_LIMIT", 20))
        self.rate_window = int(os.getenv("QUANTLAB_RATE_WINDOW", 60))
        # Queued + running jobs across everyone. Protects the one instance.
        self.max_inflight = int(os.getenv("QUANTLAB_MAX_INFLIGHT", 8))
        # Header an edge proxy sets and overwrites, naming the real client.
        # Preferred over X-Forwarded-For. On Render: cf-connecting-ip.
        self.client_ip_header = os.getenv("QUANTLAB_CLIENT_IP_HEADER", "").lower()
        # Reverse proxies whose X-Forwarded-For entries can be trusted.
        # 0 = use the socket address (local development).
        self.trust_proxy_hops = int(os.getenv("QUANTLAB_TRUST_PROXY_HOPS", 0))
        # Where finished runs are kept. SQLite for development; set a
        # Postgres URL in production. Hosts with an ephemeral disk lose a
        # SQLite file on every restart, which is the problem this solves.
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///quantlab.db")
        # Runs older than this are deleted. 0 keeps them forever.
        self.run_retention_days = int(os.getenv("QUANTLAB_RUN_RETENTION_DAYS", 30))
        # Supabase Auth. Unset URL = accounts off, everyone anonymous.
        # The publishable key is public by design: it ships to the browser.
        self.supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.supabase_publishable_key = (os.getenv("SUPABASE_PUBLISHABLE_KEY")
                                         or os.getenv("SUPABASE_ANON_KEY", ""))
        # Only for projects still signing sessions with the legacy HS256
        # secret. Projects on asymmetric signing keys leave this unset.
        self.supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET", "")
        # Signed-in users get their own, larger allowance.
        self.user_rate_limit = int(os.getenv("QUANTLAB_USER_RATE_LIMIT", 60))
        # Other origins allowed to call the API from a browser, comma
        # separated. Empty (default): none. The UI is same-origin and needs
        # no CORS; opening it up lets any page run backtests from its
        # visitors' browsers, spreading load across their IP limits.
        self.cors_origins = [o.strip() for o in
                             os.getenv("QUANTLAB_CORS_ORIGINS", "").split(",")
                             if o.strip()]


settings = Settings()
cache = BarCache(settings.cache_root, settings.cache_ttl)


def _persist(job):
    if job.kind == "backtest":
        runs.save(job)


def _make_runs():
    return RunStore(settings.database_url, settings.run_retention_days)


runs = _make_runs()
jobs = JobStore(workers=settings.workers, on_finish=_persist)


def _make_limiter(limit=None):
    # Imported here: api.security imports this module.
    from api.security import RateLimiter
    return RateLimiter(limit or settings.rate_limit, settings.rate_window)


def _make_verifier():
    if not settings.supabase_url:
        return None
    from api.auth import TokenVerifier
    return TokenVerifier(settings.supabase_url, settings.supabase_jwt_secret)


limiter = _make_limiter()
user_limiter = _make_limiter(settings.user_rate_limit)
verifier = _make_verifier()

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
    global cache, jobs, limiter, runs, user_limiter, verifier
    for k, v in overrides.items():
        setattr(settings, k, v)
    cache = BarCache(settings.cache_root, settings.cache_ttl)
    runs.close()
    runs = _make_runs()
    jobs = JobStore(workers=settings.workers, on_finish=_persist)
    limiter = _make_limiter()
    user_limiter = _make_limiter(settings.user_rate_limit)
    verifier = _make_verifier()
    _providers.clear()
    return cache
