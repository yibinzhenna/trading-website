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

from fastapi import Header, HTTPException, Request

from api import deps


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

    Behind a reverse proxy every request arrives from the proxy's address, so
    the real client sits in X-Forwarded-For. That header is client-writable:
    each proxy *appends* the address it saw, so only entries added by proxies
    you trust are honest. With `trust_proxy_hops = n`, the n-th entry from the
    right is the address the outermost trusted proxy received from.

    Taking the left-most entry — the common mistake — lets anyone pick their
    own bucket by sending a fake header. If hops is set too high the result is
    an internal address shared by everyone, which fails strict (one shared
    bucket), never open.
    """
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


def enforce_submission_limits(request: Request):
    """Gate for POST /backtest: global capacity first, then per-client rate.

    The global cap is the one that actually protects the instance. Per-client
    limits are fairness, and a determined client can rotate addresses; a cap
    on work in flight holds regardless of who is asking.
    """
    in_flight = deps.jobs.in_flight()
    if in_flight >= deps.settings.max_inflight:
        raise HTTPException(
            429, "The server is at capacity. Try again in a few seconds.",
            headers={"Retry-After": "5"})

    allowed, retry = deps.limiter.check(client_key(request))
    if not allowed:
        raise HTTPException(
            429, f"Too many backtests. Try again in {retry}s.",
            headers={"Retry-After": str(retry)})
