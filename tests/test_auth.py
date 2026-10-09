"""
Accounts: token verification, ownership, and per-user limits.

Tokens are minted here with a throwaway P-256 key, standing in for the
project's signing key, and the verifier is pointed at a fake key set. That
exercises every check the real path makes without a Supabase project.
"""

import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

from api import deps
from api.auth import InvalidToken, TokenVerifier

URL = "https://proj.supabase.co"
ISSUER = URL + "/auth/v1"
KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = ec.generate_private_key(ec.SECP256R1())
BODY = {"symbol": "SPY", "kind": "momentum"}


class FakeJWKS:
    def __init__(self, key=KEY, down=False):
        self.key, self.down = key, down

    def get_signing_key_from_jwt(self, token):
        if self.down:
            raise jwt.PyJWKClientConnectionError("network unreachable")
        return SimpleNamespace(key=self.key.public_key())


def token(sub="user-1", email="a@example.com", key=KEY, alg="ES256",
          ttl=3600, **overrides):
    claims = {"sub": sub, "email": email, "aud": "authenticated",
              "role": "authenticated", "iss": ISSUER,
              "exp": int(time.time()) + ttl}
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm=alg)


def bearer(tok):
    return {"Authorization": f"Bearer {tok}"}


def verifier(**kw):
    return TokenVerifier(URL, jwks_client=kw.pop("jwks", FakeJWKS()), **kw)


# ── Verification ───────────────────────────────────────────────────────────

def test_valid_token_yields_the_user():
    user = verifier().verify(token())
    assert user.id == "user-1" and user.email == "a@example.com"


@pytest.mark.parametrize("bad", [
    pytest.param(dict(ttl=-10), id="expired"),
    pytest.param(dict(iss="https://evil.supabase.co/auth/v1"), id="issuer"),
    pytest.param(dict(aud="something-else"), id="audience"),
    pytest.param(dict(role="anon"), id="anon-key"),
    pytest.param(dict(role="service_role"), id="service-role"),
    pytest.param(dict(key=OTHER_KEY), id="foreign-key"),
])
def test_bad_tokens_rejected(bad):
    with pytest.raises(InvalidToken):
        verifier().verify(token(**bad))


def test_unsigned_token_rejected():
    forged = jwt.encode({"sub": "x", "aud": "authenticated", "iss": ISSUER,
                         "role": "authenticated",
                         "exp": int(time.time()) + 60}, None, algorithm="none")
    with pytest.raises(InvalidToken):
        verifier().verify(forged)


def test_hs256_refused_without_the_legacy_secret():
    """Otherwise a token could pick HMAC and sign with any guessable key."""
    with pytest.raises(InvalidToken):
        verifier().verify(token(key="guessable" * 4, alg="HS256"))


def test_hs256_accepted_with_the_legacy_secret():
    secret = "legacy-project-secret-at-least-32-bytes!"
    user = verifier(jwt_secret=secret).verify(
        token(key=secret, alg="HS256"))
    assert user.id == "user-1"


def test_garbage_is_invalid_not_a_crash():
    with pytest.raises(InvalidToken):
        verifier().verify("not.a.jwt")


# ── Through the API ────────────────────────────────────────────────────────

def make_client(tmp_path, auth=True, jwks=None, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), admin_token="",
                    rate_limit=10_000, user_rate_limit=10_000,
                    max_inflight=1_000, client_ip_header="",
                    trust_proxy_hops=0, supabase_url="",
                    supabase_publishable_key="",
                    database_url=f"sqlite:///{(tmp_path / 'r.db').as_posix()}")
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    if auth:
        deps.settings.supabase_url = URL
        deps.settings.supabase_publishable_key = "sb_publishable_test"
        deps.verifier = TokenVerifier(URL, jwks_client=jwks or FakeJWKS())
    from api.main import app
    return TestClient(app)


def finish(c, headers=None, body=BODY):
    r = c.post("/backtest", json=body, headers=headers or {})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    assert deps.jobs.wait(job_id, timeout=60).status in ("done", "failed")
    return job_id


def test_config_offers_sign_in_when_configured(tmp_path):
    with make_client(tmp_path) as c:
        auth = c.get("/config").json()["auth"]
    assert auth == {"provider": "supabase", "url": URL,
                    "publishable_key": "sb_publishable_test"}


def test_config_says_no_accounts_when_unconfigured(tmp_path):
    with make_client(tmp_path, auth=False) as c:
        assert c.get("/config").json() == {"auth": None, "research": None}
        assert c.get("/me").status_code == 503


def test_me_requires_a_session(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/me").status_code == 401
        assert c.get("/me", headers=bearer(token())).json()["email"] == "a@example.com"


def test_anonymous_backtests_still_work(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c)
        assert c.get(f"/backtest/{job_id}").json()["status"] == "done"


def test_bad_token_is_401_not_a_silent_anonymous_run(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/backtest", json=BODY, headers=bearer(token(ttl=-10)))
    assert r.status_code == 401


def test_signing_service_outage_is_503(tmp_path):
    with make_client(tmp_path, jwks=FakeJWKS(down=True)) as c:
        assert c.get("/me", headers=bearer(token())).status_code == 503


def test_runs_belong_to_whoever_submitted_them(tmp_path):
    alice, bob = bearer(token("alice")), bearer(token("bob"))
    with make_client(tmp_path) as c:
        mine = finish(c, alice)
        finish(c)                                  # anonymous
        alices = c.get("/me/runs", headers=alice).json()
        bobs = c.get("/me/runs", headers=bob).json()
    assert [r["job_id"] for r in alices] == [mine]
    assert alices[0]["symbol"] == "SPY" and "total_return_pct" in alices[0]
    assert bobs == []


def test_owner_is_not_revealed_to_link_holders(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c, bearer(token("alice")))
        live = c.get(f"/backtest/{job_id}").json()
        deps.jobs.forget(job_id)
        saved = c.get(f"/backtest/{job_id}").json()
    for body in (live, saved):
        assert "owner_id" not in body["meta"]
        assert "alice" not in str(body)


def test_only_the_owner_can_delete(tmp_path):
    alice, bob = bearer(token("alice")), bearer(token("bob"))
    with make_client(tmp_path) as c:
        job_id = finish(c, alice)
        assert c.delete(f"/runs/{job_id}", headers=bob).status_code == 404
        assert c.delete(f"/runs/{job_id}").status_code == 401
        assert c.get(f"/backtest/{job_id}").status_code == 200
        assert c.delete(f"/runs/{job_id}", headers=alice).status_code == 204
        # Gone from memory and the database: the link is dead.
        assert c.get(f"/backtest/{job_id}").status_code == 404
        assert c.get("/me/runs", headers=alice).json() == []


def test_anonymous_runs_cannot_be_claimed_by_deleting(tmp_path):
    with make_client(tmp_path) as c:
        job_id = finish(c)
        r = c.delete(f"/runs/{job_id}", headers=bearer(token("mallory")))
    assert r.status_code == 404


# ── Per-user limits ────────────────────────────────────────────────────────

def test_users_get_their_own_allowance(tmp_path):
    with make_client(tmp_path, rate_limit=1, user_rate_limit=3) as c:
        anon = [c.post("/backtest", json=BODY).status_code for _ in range(2)]
        user = [c.post("/backtest", json=BODY,
                       headers=bearer(token("alice"))).status_code
                for _ in range(4)]
    assert anon == [202, 429]
    assert user == [202, 202, 202, 429]


def test_one_users_limit_does_not_touch_anothers(tmp_path):
    with make_client(tmp_path, user_rate_limit=1) as c:
        a = [c.post("/backtest", json=BODY,
                    headers=bearer(token("alice"))).status_code
             for _ in range(2)]
        b = c.post("/backtest", json=BODY,
                   headers=bearer(token("bob"))).status_code
    assert a == [202, 429] and b == 202
