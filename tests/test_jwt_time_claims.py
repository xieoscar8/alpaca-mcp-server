"""Verified-claims gate, real signed token swap/refresh, and principal exclusion."""

import time
from unittest.mock import AsyncMock, Mock

import pytest
from authlib.jose import jwt
from fastmcp.server.auth import OIDCProxy
from fastmcp.server.auth.oauth_proxy.models import JTIMapping, UpstreamTokenSet
from fastmcp.server.auth.oidc_proxy import OIDCConfiguration
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from key_value.aio.stores.memory import MemoryStore

from alpaca_mcp_server import authentication

ISSUER = "https://issuer.example"
AUDIENCE = "https://mcp.example"
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
def signed_proxy(monkeypatch):
    """Real project OIDC adapter, RSA verifier, and proxy stores; no IdP network."""
    keys = RSAKeyPair.generate()
    now = int(time.time())
    monkeypatch.setattr(authentication.time, "time", lambda: now)
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    discovery = OIDCConfiguration(
        issuer=ISSUER, authorization_endpoint=ISSUER + "/authorize",
        token_endpoint=ISSUER + "/token", jwks_uri=ISSUER + "/jwks",
        response_types_supported=["code"], subject_types_supported=["public"],
        id_token_signing_alg_values_supported=["RS256"],
    )
    monkeypatch.setattr(OIDCProxy, "get_oidc_configuration", lambda *args: discovery)
    provider = authentication.SubjectBoundOIDCProxy(
        config_url=ISSUER + "/.well-known/openid-configuration",
        client_id="test-app", client_secret="test-only-secret", audience=AUDIENCE,
        base_url=AUDIENCE, required_scopes=["openid"], verify_id_token=False,
        client_storage=MemoryStore(),
        jwt_signing_key="test-only-signing-key-with-at-least-32-characters",
        forward_resource=False, extra_token_params={"resource": AUDIENCE},
    )
    # Static public key instead of remote JWKS; all signature/claim verification
    # and the installed OAuthProxy.load_access_token implementation remain real.
    provider._token_validator = JWTVerifier(
        public_key=keys.public_key, issuer=ISSUER, audience=AUDIENCE,
        algorithm="RS256", required_scopes=["openid"],
    )
    provider.get_routes(mcp_path="/mcp")

    def sign(overrides=None, missing=None):
        claims = {"iss": ISSUER, "sub": "user-a", "aud": AUDIENCE,
                  "scope": "openid", "exp": now + 600, "iat": now,
                  **(overrides or {})}
        claims.pop(missing, None)
        return jwt.encode({"alg": "RS256"}, claims, keys.private_key.get_secret_value()).decode()

    async def reference(upstream, *, refresh=False):
        await provider._upstream_token_store.put(key="upstream", value=UpstreamTokenSet(
            upstream_token_id="upstream", access_token=upstream,
            refresh_token="test-only-refresh" if refresh else None,
            refresh_token_expires_at=now + 3600 if refresh else None,
            expires_at=now - 300 if refresh else now + 600,
            token_type="Bearer", scope="openid", client_id="same-client", created_at=now,
        ))
        await provider._jti_mapping_store.put(key="reference", value=JTIMapping(
            jti="reference", upstream_token_id="upstream", created_at=now,
        ))
        return provider.jwt_issuer.issue_access_token(
            client_id="same-client", scopes=["openid"], jti="reference", expires_in=600,
        )

    return provider, sign, reference, now


async def assert_no_principal(monkeypatch, provider, bearer):
    value = await provider.load_access_token(bearer)
    assert value is None  # authentication middleware never receives an identity
    monkeypatch.setattr(authentication, "get_access_token", lambda: value)
    with pytest.raises(authentication.PrincipalError):
        authentication.authenticated_principal()


@pytest.mark.parametrize("claim", ["exp", "nbf", "iat"])
@pytest.mark.parametrize("value", INVALID)
async def test_signed_malformed_time_never_creates_principal(
    monkeypatch, signed_proxy, claim, value,
):
    provider, sign, reference, _ = signed_proxy
    await assert_no_principal(monkeypatch, provider, await reference(sign({claim: value})))


@pytest.mark.parametrize("case,allowed", [
    ("missing-exp", False), ("expired", False), ("expired-inside-skew", False),
    ("valid", True), ("fractional", True), ("missing-nbf", True), ("past-nbf", True),
    ("future-nbf", False), ("skew-nbf", True), ("missing-iat", True),
    ("past-iat", True), ("future-iat", False), ("skew-iat", True),
])
async def test_signed_time_policy(monkeypatch, signed_proxy, case, allowed):
    provider, sign, reference, now = signed_proxy
    overrides = {
        "expired": {"exp": now - 300}, "expired-inside-skew": {"exp": now - 1},
        "fractional": {"exp": now + 600.5}, "past-nbf": {"nbf": now - 300},
        "future-nbf": {"nbf": now + 300}, "skew-nbf": {"nbf": now + 60},
        "past-iat": {"iat": now - 300}, "future-iat": {"iat": now + 300},
        "skew-iat": {"iat": now + 60},
    }.get(case, {})
    missing = case.removeprefix("missing-") if case.startswith("missing-") else None
    bearer = await reference(sign(overrides, missing))
    if not allowed:
        await assert_no_principal(monkeypatch, provider, bearer)
    else:
        value = await provider.load_access_token(bearer)
        assert value is not None and value.subject == "user-a"
        monkeypatch.setattr(authentication, "get_access_token", lambda: value)
        assert authentication.authenticated_principal().startswith("oauth-v2-")


@pytest.mark.parametrize("case,allowed", [
    ("valid", True), ("missing-exp", False), ("future-nbf", False),
    ("future-iat", False), ("malformed", False), ("expired", False),
])
async def test_real_transparent_refresh_obeys_time_gate(
    monkeypatch, signed_proxy, case, allowed,
):
    provider, sign, reference, now = signed_proxy
    overrides = {"future-nbf": {"nbf": now + 300}, "future-iat": {"iat": now + 300},
                 "malformed": {"exp": None}, "expired": {"exp": now - 300}}.get(case, {})
    refreshed = sign(overrides, "exp" if case == "missing-exp" else None)
    upstream = Mock(refresh_token=AsyncMock(return_value={
        "access_token": refreshed, "refresh_token": "test-only-rotated-refresh",
        "expires_in": 600,
    }))
    monkeypatch.setattr(provider, "_create_upstream_oauth_client", lambda: upstream)
    bearer = await reference(sign({"exp": now - 300}), refresh=True)
    if allowed:
        assert (await provider.load_access_token(bearer)).subject == "user-a"
    else:
        await assert_no_principal(monkeypatch, provider, bearer)
    upstream.refresh_token.assert_awaited_once()
    assert upstream.refresh_token.call_args.kwargs["resource"] == AUDIENCE
