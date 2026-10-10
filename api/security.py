"""
Access control and rate limiting.

Two different problems, kept apart on purpose:

* Admin endpoints (cache inspection and clearing, the job listing) are
  operator tools. They need a secret, not a rate limit. With no secret
  configured they are disabled outright, so a fresh deploy is closed by
  default rather than open by default.

* Backtest submission is public by design. It needs throttling, not a secret:
  backtests are CPU-bound and one script could otherwise saturate the single
  instance for everyone.

Individual jobs stay readable by id. Ids are 64 random bits, so knowing one
means you submitted it or were handed it — a capability, not a guessable key.
"""

import hmac
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, Request

from api import deps
from api.auth import optional_user
from api.store import StoreUnavailable


# ── Admin ──────────────────────────────────────────────────────────────────

def require_admin(authorization: str | None = Header(None)):
    """FastAPI dependency: `Authorization: Bearer <QUANTLAB_ADMIN_TOKEN>`.

    Unset token -> 403 for everyone. Compared in constant time so response
    timing does not reveal how much of a guess was right.
    """
    expected = deps.settings.admin_token
    if not expected:
        raise HTTPException(
            403, "Admin endpoints are disabled: QUANTLAB_ADMIN_TOKEN is not set")
    scheme, _, supplied = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
            supplied.strip().encode(), expected.encode()):
        raise HTTPException(403, "Admin token required",
                            headers={"WWW-Authenticate": "Bearer"})


# ── Client identity ────────────────────────────────────────────────────────

def client_key(request: Request):
    """Best available identity for the caller, for rate limiting only.

    In order of preference:

    1. A header set by a trusted edge proxy that overwrites any client-sent
       value (`client_ip_header`). On Render that is Cloudflare's
       `CF-Connecting-IP`: Cloudflare replaces it on every request, so a
       client cannot choose its own value.
    2. X-Forwarded-For, read from the right, `trust_proxy_hops` entries in.
       Only correct when every proxy appends exactly one entry and the hop
       count is exact.
    3. The socket address.

    Why not X-Forwarded-For on Render: requests pass Cloudflare and then
    Render's load balancer, so it arrives as `client, edge, lb`. The left-most
    entry is client-written, so trusting it lets anyone pick their own bucket.
    The right-most entries come from rotating infrastructure pools, so trusting
    those gives every request a fresh bucket. Both fail *open* — verified
    against the live deploy, where hops=1 let 22 consecutive requests through
    a limit of 20.
    """
    header = deps.settings.client_ip_header
    if header:
        value = request.headers.get(header, "").strip()
        if value:
            return value

    hops = deps.settings.trust_proxy_hops
    if hops > 0:
        chain = [h.strip() for h in
                 request.headers.get("x-forwarded-for", "").split(",")
                 if h.strip()]
        if len(chain) >= hops:
            return chain[-hops]
    return request.client.host if request.client else "unknown"


# ── Rate limiting ──────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding-window limiter: at most `limit` hits per `window` seconds.

    In process memory, like the job store, and for the same reason: one
    worker. A multi-instance deployment needs this in Redis alongside jobs.
    """

    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self._hits = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def check(self, key):
        """Record a hit. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            return True, 0

    def _sweep(self, now):
        """Drop idle keys now and then, so one-off visitors do not
        accumulate forever."""
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        for key in [k for k, q in self._hits.items()
                    if not q or now - q[-1] >= self.window]:
            del self._hits[key]


# ── Daily backtest allowance ───────────────────────────────────────────────
# Checked inside the submit handler, after validation and under a lock held
# with the submission itself: a request rejected as invalid costs nothing,
# and a burst of parallel requests cannot all read the same count and slip
# past.
#
# The lock is per submitter, not global. The check for an account queries
# the database, and one site-wide lock held across that query made every
# signed-in user's submission — and every visitor's behind them — wait on
# everyone else's round trip. Only one submitter's own requests need to be
# serialised against each other, so locks are striped by submitter: the
# same key always maps to the same lock; different keys rarely share one.

_STRIPES = tuple(threading.Lock() for _ in range(64))
DAY = timedelta(days=1)


def submitter_key(request, user):
    return f"user:{user.id}" if user is not None else f"ip:{client_key(request)}"


def submission_lock(request, user):
    """The lock to hold across the daily check and the submission."""
    return _STRIPES[hash(submitter_key(request, user)) % len(_STRIPES)]


def _user_runs_today(user):
    """An account's backtests in the last 24 hours: saved runs plus anything
    still in memory (in flight, or finished but not yet written)."""
    since = datetime.now(timezone.utc) - DAY
    try:
        seen = dict(deps.runs.owner_runs_since(user.id, since))
    except StoreUnavailable:
        # Fail open on the daily cap: backtests do not need the database,
        # and the per-minute limit and global capacity cap still hold.
        # Counting what is in memory keeps a floor under it.
        seen = {}
    seen.update(deps.jobs.submitted_since(since.isoformat(timespec="milliseconds"),
                                          owner_id=user.id))
    return sorted(seen.values())


def daily_usage(user):
    """{used, limit, remaining} for an account, or None with no daily cap."""
    limit = deps.settings.user_daily_limit
    if not limit:
        return None
    used = len(_user_runs_today(user))
    return {"used": used, "limit": limit, "remaining": max(0, limit - used)}


def visitor_id(request):
    """A pseudonym for a signed-out visitor's address: HMAC-SHA256 under a
    server secret, truncated. The address itself is never stored. A plain
    hash would not do — every IPv4 address can be hashed in minutes — but
    without the key the pseudonym cannot be reversed. None without a key."""
    key = deps.settings.visitor_key
    if not key:
        return None
    return hmac.new(key.encode(), client_key(request).encode(),
                    "sha256").hexdigest()[:32]


def _visitor_check(request, limit):
    """Daily allowance for a visitor, counted in the database so restarts do
    not reset it. Returns False if it could not be counted that way."""
    visitor = visitor_id(request)
    if visitor is None:
        return False
    now = datetime.now(timezone.utc)
    try:
        stamps = deps.runs.visitor_usage_since(visitor, now - DAY)
        if len(stamps) >= limit:
            oldest = datetime.fromisoformat(stamps[0])
            retry = max(1, int((oldest + DAY - now).total_seconds()) + 1)
            raise HTTPException(
                429, f"Daily backtest limit reached ({limit} per 24 hours). "
                     "Sign in for a larger allowance.",
                headers={"Retry-After": str(retry)})
        deps.runs.visitor_record(visitor, now)
    except StoreUnavailable:
        return False
    return True


def check_daily_limit(request: Request, user):
    """Raise 429 once the 24-hour allowance is spent. Call while holding
    `submission_lock(request, user)`, immediately before submitting."""
    if user is not None:
        limit = deps.settings.user_daily_limit
        if not limit:
            return
        stamps = _user_runs_today(user)
        if len(stamps) >= limit:
            oldest = datetime.fromisoformat(stamps[0])
            retry = max(1, int((oldest + DAY - datetime.now(timezone.utc))
                               .total_seconds()) + 1)
            raise HTTPException(
                429, f"Daily backtest limit reached ({limit} per 24 hours).",
                headers={"Retry-After": str(retry)})
        return
    if not deps.settings.daily_limit:
        return
    if _visitor_check(request, deps.settings.daily_limit):
        return
    # No key configured, or the database is down: count in memory, which
    # resets when the server restarts.
    allowed, retry = deps.daily_limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            429, f"Daily backtest limit reached ({deps.settings.daily_limit} "
                 "per 24 hours). Sign in for a larger allowance.",
            headers={"Retry-After": str(retry)})


def enforce_submission_limits(request: Request,
                              user=Depends(optional_user)):
    """Gate for POST /backtest: global capacity first, then per-client rate.

    The global cap is the one that actually protects the instance. Per-client
    limits are fairness, and a determined client can rotate addresses; a cap
    on work in flight holds regardless of who is asking.

    Signed-in users are limited per account rather than per address, with a
    larger allowance. An account is a far better identity than an IP: a
    whole office shares one address, and one person can hop between several.
    """
    in_flight = deps.jobs.in_flight()
    if in_flight >= deps.settings.max_inflight:
        raise HTTPException(
            429, "The server is at capacity. Try again in a few seconds.",
            headers={"Retry-After": "5"})

    if user is not None:
        allowed, retry = deps.user_limiter.check(f"user:{user.id}")
    else:
        allowed, retry = deps.limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            429, f"Too many backtests. Try again in {retry}s.",
            headers={"Retry-After": str(retry)})
