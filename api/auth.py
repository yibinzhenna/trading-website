"""
Accounts, via Supabase Auth.

The browser signs in with supabase-js and sends its access token as
`Authorization: Bearer <jwt>`. This module verifies the token and turns it
into a `User`. Verification is local: tokens are checked against the
project's public signing keys (JWKS), fetched once and cached, so a request
never waits on a round trip to Supabase.

Accounts are optional. Anonymous visitors still run backtests under the
per-IP limit; signing in adds a run history and a larger allowance.

With `SUPABASE_URL` unset, auth is off: every request is anonymous and the
frontend hides the sign-in UI. Local development and the test suite need no
Supabase project.
"""

from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException

from api import deps

AUDIENCE = "authenticated"


@dataclass(frozen=True)
class User:
    id: str
    email: str | None = None


class InvalidToken(Exception):
    pass


class TokenVerifier:
    """Checks a Supabase access token: signature, expiry, issuer, audience.

    Asymmetric keys (ES256, RS256) are verified against the project's JWKS.
    HS256 is accepted only if the legacy shared secret is configured — older
    projects still sign with it. The algorithm list passed to `decode` is
    always the single one the key belongs to, so a token cannot talk the
    verifier into checking an RS256 signature with an HMAC secret.
    """

    ASYMMETRIC = ("ES256", "RS256")

    def __init__(self, supabase_url, jwt_secret="", jwks_client=None):
        self.issuer = supabase_url.rstrip("/") + "/auth/v1"
        self.secret = jwt_secret
        # Supabase caches keys for up to 20 minutes on its side before a
        # rotation takes effect; matching that here is enough.
        self.jwks = jwks_client or jwt.PyJWKClient(
            self.issuer + "/.well-known/jwks.json",
            cache_keys=True, lifespan=600)

    def verify(self, token):
        try:
            alg = jwt.get_unverified_header(token).get("alg")
            if alg in self.ASYMMETRIC:
                key = self.jwks.get_signing_key_from_jwt(token).key
            elif alg == "HS256" and self.secret:
                key = self.secret
            else:
                raise InvalidToken(f"unsupported signing algorithm: {alg}")
            claims = jwt.decode(
                token, key, algorithms=[alg], audience=AUDIENCE,
                issuer=self.issuer, options={"require": ["exp", "sub"]})
        except jwt.PyJWKClientConnectionError as e:
            raise ConnectionError(f"could not fetch signing keys: {e}") from e
        except jwt.PyJWTError as e:
            raise InvalidToken(str(e)) from e
        # The anon and service_role keys are JWTs from the same issuer.
        # Only a signed-in user's session token counts as a user.
        if claims.get("role") != AUDIENCE:
            raise InvalidToken("not a user session token")
        return User(id=claims["sub"], email=claims.get("email"))


def optional_user(authorization: str | None = Header(None)):
    """The signed-in user, or None for an anonymous request.

    A token that is present but bad is a 401, not a silent downgrade to
    anonymous: an expired session should prompt a refresh, not quietly run
    under someone's IP limit and land outside their history.
    """
    verifier = deps.verifier
    if verifier is None or not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        return verifier.verify(token.strip())
    except InvalidToken as e:
        raise HTTPException(401, f"Invalid or expired session: {e}",
                            headers={"WWW-Authenticate": "Bearer"}) from e
    except ConnectionError as e:
        raise HTTPException(503, "Sign-in is temporarily unavailable") from e


def require_user(user: User | None = Depends(optional_user)):
    if deps.verifier is None:
        raise HTTPException(503, "Accounts are not enabled on this server")
    if user is None:
        raise HTTPException(401, "Sign in required",
                            headers={"WWW-Authenticate": "Bearer"})
    return user
