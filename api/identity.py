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

import hashlib
import hmac

from api import deps

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
