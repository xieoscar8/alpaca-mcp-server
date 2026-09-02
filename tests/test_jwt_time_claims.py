"""Direct verified JWTs, refreshed access tokens, and principal exclusion."""

import time
from unittest.mock import AsyncMock

import pytest
from authlib.jose import jwt
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from alpaca_mcp_server import authentication

ISSUER = "https://issuer.example"
AUDIENCE = "https://mcp.example/mcp"
NOW = 2_000_000_000
MISSING = object()
INVALID = ["2000000600", True, False, None, float("nan"), float("inf"),
           float("-inf"), {}, []]


@pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
@pytest.mark.parametrize("value", INVALID)
def test_numericdate_rejects_malformed_values(monkeypatch, claim, value):
    monkeypatch.setattr(authentication.time, "time", lambda: NOW)
    assert not authentication._valid_time_claims({"exp": NOW + 600, claim: value})


@pytest.mark.parametrize("value,allowed", [
    (NOW + 600, True), (NOW + 0.5, True), (NOW, True),
    (NOW - 59.999, True), (NOW - 60, False), (NOW - 60.001, False),
    (NOW - 600, False), (MISSING, False),
])
def test_expiration_gate_boundary(monkeypatch, value, allowed):
    # Upstream JWTVerifier is stricter: it rejects already-expired JWTs even
    # inside skew. This gate must never bypass that earlier rejection.
    monkeypatch.setattr(authentication.time, "time", lambda: NOW)
    claims = {} if value is MISSING else {"exp": value}
    assert authentication._valid_time_claims(claims) is allowed


@pytest.mark.parametrize("claim", ["nbf", "iat"])
@pytest.mark.parametrize("value,allowed", [
    (MISSING, True), (NOW - 300, True), (NOW, True), (NOW + 59.999, True),
    (NOW + 60, True), (NOW + 60.001, False), (NOW + 300, False),
])
def test_optional_time_boundary(monkeypatch, claim, value, allowed):
    monkeypatch.setattr(authentication.time, "time", lambda: NOW)
    claims = {"exp": NOW + 600}
    if value is not MISSING:
        claims[claim] = value
    assert authentication._valid_time_claims(claims) is allowed


@pytest.fixture
def signed_provider(monkeypatch):
    """Real direct adapter and default RSA verifier; only JWKS I/O is mocked."""
    keys = RSAKeyPair.generate()
    now = int(time.time())
    monkeypatch.setattr(authentication.time, "time", lambda: now)
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    provider = authentication.HardenedAuthKitProvider(
        issuer=ISSUER, base_url="https://mcp.example", audience=AUDIENCE,
        required_scopes=["openid"],
    )
    monkeypatch.setattr(provider.token_verifier, "_get_verification_key",
                        AsyncMock(return_value=keys.public_key))
    provider.get_routes(mcp_path="/mcp")

    def sign(overrides=None, missing=None):
        claims = {"iss": ISSUER, "sub": "user-a", "aud": AUDIENCE,
                  "scope": "openid", "exp": now + 600, "iat": now,
                  **(overrides or {})}
        claims.pop(missing, None)
        return jwt.encode({"alg": "RS256"}, claims, keys.private_key.get_secret_value()).decode()

    return provider, sign, now


async def assert_no_principal(monkeypatch, provider, bearer):
    value = await provider.verify_token(bearer)
    assert value is None  # authentication middleware never receives an identity
    monkeypatch.setattr(authentication, "get_access_token", lambda: value)
    with pytest.raises(authentication.PrincipalError):
        authentication.authenticated_principal()


@pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
@pytest.mark.parametrize("value", INVALID)
async def test_signed_malformed_time_never_creates_principal(
    monkeypatch, signed_provider, claim, value,
):
    provider, sign, _ = signed_provider
    await assert_no_principal(monkeypatch, provider, sign({claim: value}))


@pytest.mark.parametrize("case,allowed", [
    ("missing-exp", False), ("expired", False), ("expired-inside-skew", False),
    ("valid", True), ("fractional", True), ("missing-nbf", True), ("past-nbf", True),
    ("future-nbf", False), ("skew-nbf", True), ("missing-iat", True),
    ("past-iat", True), ("future-iat", False), ("skew-iat", True),
])
async def test_signed_time_policy(monkeypatch, signed_provider, case, allowed):
    provider, sign, now = signed_provider
    overrides = {
        "expired": {"exp": now - 300}, "expired-inside-skew": {"exp": now - 1},
        "fractional": {"exp": now + 600.5}, "past-nbf": {"nbf": now - 300},
        "future-nbf": {"nbf": now + 300}, "skew-nbf": {"nbf": now + 60},
        "past-iat": {"iat": now - 300}, "future-iat": {"iat": now + 300},
        "skew-iat": {"iat": now + 60},
    }.get(case, {})
    missing = case.removeprefix("missing-") if case.startswith("missing-") else None
    bearer = sign(overrides, missing)
    if not allowed:
        await assert_no_principal(monkeypatch, provider, bearer)
    else:
        value = await provider.verify_token(bearer)
        assert value is not None and value.subject == "user-a"
        monkeypatch.setattr(authentication, "get_access_token", lambda: value)
        assert authentication.authenticated_principal().startswith("oauth-v2-")


@pytest.mark.parametrize("case,allowed", [
    ("valid", True), ("missing-exp", False), ("future-nbf", False),
    ("future-iat", False), ("malformed", False), ("expired", False),
])
async def test_refreshed_access_token_obeys_time_gate_without_cached_identity(
    monkeypatch, signed_provider, case, allowed,
):
    # Refresh belongs to WorkOS/Claude now. Present successive signed access
    # tokens directly; do not emulate a server-side token endpoint or refresh.
    provider, sign, now = signed_provider
    original = await provider.verify_token(sign())
    assert original is not None
    monkeypatch.setattr(authentication, "get_access_token", lambda: original)
    principal = authentication.authenticated_principal()
    overrides = {"future-nbf": {"nbf": now + 300}, "future-iat": {"iat": now + 300},
                 "malformed": {"exp": None}, "expired": {"exp": now - 300}}.get(case, {})
    refreshed = sign(overrides, "exp" if case == "missing-exp" else None)
    if allowed:
        value = await provider.verify_token(refreshed)
        assert value.subject == "user-a"
        monkeypatch.setattr(authentication, "get_access_token", lambda: value)
        assert authentication.authenticated_principal() == principal
    else:
        await assert_no_principal(monkeypatch, provider, refreshed)
