"""
Server wiring: settings, provider construction, shared singletons.

Providers are built per request from a small cache rather than at import
time, so a missing API key only breaks the endpoints that need it instead of
preventing the server from starting.
"""

import os

from api.store import RunStore
from quantlab.cache import BarCache, CachedProvider
from quantlab.research import MODELS
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
        # Largest request body accepted, checked before it is read. Every
        # real request is under 1 KB; an unbounded one could exhaust memory.
        self.max_body_bytes = int(os.getenv("QUANTLAB_MAX_BODY_BYTES", 64 * 1024))
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
        # Backtests per rolling 24 hours. 0 = no daily cap. Anonymous
        # visitors are counted per address in memory (no IPs are stored, so
        # this resets on restart); accounts are counted from the database.
        self.daily_limit = int(os.getenv("QUANTLAB_DAILY_LIMIT", 50))
        self.user_daily_limit = int(os.getenv("QUANTLAB_USER_DAILY_LIMIT", 200))
        # Secret for pseudonymising visitor addresses so their daily count
        # can be kept in the database and survive restarts. Unset: the count
        # stays in memory and resets with the server.
        self.visitor_key = os.getenv("QUANTLAB_VISITOR_KEY", "")
        # Other origins allowed to call the API from a browser, comma
        # separated. Empty (default): none. The UI is same-origin and needs
        # no CORS; opening it up lets any page run backtests from its
        # visitors' browsers, spreading load across their IP limits.
        # AI research. Needs an Anthropic key *and* accounts: sessions cost
        # money, so they are tied to a signed-in user with a daily quota.
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        # Who runs AI research: "anthropic" or "deepseek". Unset: DeepSeek
        # if its key is present, otherwise Anthropic.
        self.research_provider = (
            os.getenv("QUANTLAB_RESEARCH_PROVIDER")
            or ("deepseek" if self.deepseek_api_key else "anthropic")).lower()
        # Empty = the provider's default (see quantlab.research.MODELS).
        self.research_model = os.getenv("QUANTLAB_RESEARCH_MODEL", "")
        # DeepSeek only; the Anthropic SDK takes no temperature. Low keeps
        # proposals and tool calls consistent; zero would make a model that
        # repeats itself repeat itself exactly.
        self.research_temperature = float(
            os.getenv("QUANTLAB_RESEARCH_TEMPERATURE", 0.3))
        self.research_daily_limit = int(
            os.getenv("QUANTLAB_RESEARCH_DAILY_LIMIT", 3))
        # Across all users: the ceiling on what one day can cost.
        self.research_global_daily_limit = int(
            os.getenv("QUANTLAB_RESEARCH_GLOBAL_DAILY_LIMIT", 50))
        self.research_max_trials = int(os.getenv("QUANTLAB_RESEARCH_MAX_TRIALS", 8))
        self.research_token_budget = int(
            os.getenv("QUANTLAB_RESEARCH_TOKEN_BUDGET", 60_000))
        self.research_max_queue = int(os.getenv("QUANTLAB_RESEARCH_MAX_QUEUE", 3))
        self.cors_origins = [o.strip() for o in
                             os.getenv("QUANTLAB_CORS_ORIGINS", "").split(",")
                             if o.strip()]


settings = Settings()
cache = BarCache(settings.cache_root, settings.cache_ttl)


def _persist(job, status):
    if job.kind == "backtest":
        runs.save(job, status)


def _make_runs():
    return RunStore(settings.database_url, settings.run_retention_days)


def _persist_research(job, status):
    state = job.meta.get("state")
    if status == "done":
        runs.research_finish(job.id, "done", job.result)
    else:
        runs.research_finish(job.id, "failed", state, job.error)


DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"


def _make_research_client():
    """A Messages-API client for the configured provider, or None when
    research is not configured. DeepSeek is reached through its
    Anthropic-compatible endpoint, so one client and one code path serve
    both."""
    provider = settings.research_provider
    key = {"anthropic": settings.anthropic_api_key,
           "deepseek": settings.deepseek_api_key}.get(provider)
    if not key:
        return None
    import anthropic
    kwargs = {"api_key": key, "max_retries": 2, "timeout": 60.0}
    if provider == "deepseek":
        kwargs["base_url"] = DEEPSEEK_BASE_URL
    return anthropic.Anthropic(**kwargs)


def research_model():
    return settings.research_model or MODELS.get(settings.research_provider, "")


def research_request_options():
    """Per-call options, tuned per provider.

    Thinking is switched off explicitly. DeepSeek turns it on by default at
    high effort: thousands of output tokens per call spent reasoning about
    which moving-average lengths to try, and with tools present its API
    demands that reasoning be passed back on later turns. The loop's calls
    are stateless, so nothing would break if a provider ignored this — the
    session state counts any thinking blocks that come back.
    """
    options = {"thinking": {"type": "disabled"}}
    if settings.research_provider == "deepseek":
        options["extra_body"] = {"temperature": settings.research_temperature}
    return options


def research_enabled():
    return research_client is not None and verifier is not None


runs = _make_runs()
jobs = JobStore(workers=settings.workers, on_finish=_persist)
# Its own single worker: a research session holds a thread for a minute or
# more, and must never starve ordinary backtests of theirs.
research_jobs = JobStore(workers=1, on_finish=_persist_research)
research_client = _make_research_client()


def _make_limiter(limit=None):
    # Imported here: api.security imports this module.
    from api.security import RateLimiter
    return RateLimiter(limit or settings.rate_limit, settings.rate_window)


def _make_verifier():
    if not settings.supabase_url:
        return None
    from api.auth import TokenVerifier
    return TokenVerifier(settings.supabase_url, settings.supabase_jwt_secret)


def _make_daily_limiter():
    from api.security import RateLimiter
    return RateLimiter(max(1, settings.daily_limit), 24 * 60 * 60)


limiter = _make_limiter()
user_limiter = _make_limiter(settings.user_rate_limit)
daily_limiter = _make_daily_limiter()
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
    global cache, jobs, limiter, runs, user_limiter, verifier, daily_limiter, \
        research_jobs, research_client
    for k, v in overrides.items():
        setattr(settings, k, v)
    cache = BarCache(settings.cache_root, settings.cache_ttl)
    # Let the previous test's jobs finish first. Their results are saved to
    # whatever `runs` is when they finish, so a straggler would otherwise
    # land in the next test's fresh database.
    jobs.shutdown(wait=True)
    research_jobs.shutdown(wait=True)
    runs.close()
    runs = _make_runs()
    jobs = JobStore(workers=settings.workers, on_finish=_persist)
    research_jobs = JobStore(workers=1, on_finish=_persist_research)
    research_client = _make_research_client()
    limiter = _make_limiter()
    user_limiter = _make_limiter(settings.user_rate_limit)
    daily_limiter = _make_daily_limiter()
    verifier = _make_verifier()
    _providers.clear()
    return cache
