"""
Who is behind an account, as far as quotas are concerned.

Supabase gives every account a unique id, but one person can open many
accounts with variants of one inbox: `jane+1@gmail.com`, `jane+2@...`,
`j.a.n.e@gmail.com` and `jane@googlemail.com` all deliver to the same
mailbox. Quotas that matter (AI research) are counted per *identity*: the
canonical form of the email, so those variants share one allowance.

The canonical email is never stored. It is pseudonymised with HMAC under the
same server secret as visitor addresses, domain-separated so an identity
pseudonym can never equal a visitor one.
"""

import functools
import hashlib
import hmac
from pathlib import Path

from api import deps

_BLOCKLIST = Path(__file__).with_name("disposable_domains.txt")

# Providers where dots in the local part are ignored for delivery.
_DOTS_IGNORED = {"gmail.com": "gmail.com", "googlemail.com": "gmail.com"}


def canonical_email(email):
    """Lower-case, drop a "+tag", and for Gmail drop dots and fold
    googlemail.com into gmail.com. Anything without an "@" is returned
    lower-cased and otherwise unchanged."""
    email = (email or "").strip().lower()
    local, at, domain = email.rpartition("@")
    if not at or not local or not domain:
        return email
    local = local.split("+", 1)[0]
    if domain in _DOTS_IGNORED:
        local = local.replace(".", "")
        domain = _DOTS_IGNORED[domain]
    return f"{local}@{domain}"


@functools.lru_cache(maxsize=1)
def _disposable_domains():
    try:
        lines = _BLOCKLIST.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(line.strip().lower() for line in lines
                     if line.strip() and not line.lstrip().startswith("#"))


def is_disposable(email):
    """Whether the email's domain, or any parent of it, is a known throwaway
    provider: `x@mailinator.com` and `x@eu.mailinator.com` both are."""
    domain = (email or "").strip().lower().rpartition("@")[2].rstrip(".")
    if not domain:
        return False
    blocked = _disposable_domains()
    parts = domain.split(".")
    return any(".".join(parts[i:]) in blocked for i in range(len(parts) - 1))


def identity_key(user):
    """A 32-hex pseudonym for the person behind `user`.

    Keyed with QUANTLAB_VISITOR_KEY when set. Without it this falls back to
    an unkeyed hash, which still groups variants correctly but could be
    matched against a guessed address — set the key in production.
    """
    basis = canonical_email(user.email) if user.email else f"id:{user.id}"
    message = f"identity:{basis}".encode()
    secret = deps.settings.visitor_key
    if secret:
        return hmac.new(secret.encode(), message, "sha256").hexdigest()[:32]
    return hashlib.sha256(message).hexdigest()[:32]
