"""Direct AuthKit discovery, configured authority and real signed session tests."""

import hashlib
import time
from unittest.mock import AsyncMock

import pytest
from authlib.jose import jwt
from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from starlette.testclient import TestClient

from alpaca_mcp_server import authentication as auth

ISSUER = "https://issuer.example"
BASE = "https://mcp.example"
RESOURCE = BASE + "/mcp"


def provider(**overrides):
    return auth.HardenedAuthKitProvider(**{
        "issuer": ISSUER, "base_url": BASE, "audience": RESOURCE,
        "required_scopes": ["alpaca:safe-write"], **overrides,
    })


@pytest.fixture
def signed(monkeypatch):
    keys = RSAKeyPair.generate()
    p = provider()
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", RESOURCE)
    monkeypatch.setattr(p.token_verifier, "_get_verification_key",
                        AsyncMock(return_value=keys.public_key))
    now = int(time.time())
    monkeypatch.setattr(auth.time, "time", lambda: now)

    def sign(changes=None, missing=None):
        claims = {"iss": ISSUER, "sub": "user-a", "aud": RESOURCE,
                  "client_id": "same-claude-client", "scope": "alpaca:safe-write",
                  "exp": now + 600, "iat": now, **(changes or {})}
        claims.pop(missing, None)
        return jwt.encode({"alg": "RS256"}, claims, keys.private_key.get_secret_value()).decode()

    return p, sign, now


def test_discovery_and_challenge_without_oauth_state(monkeypatch):
    for key in ("DATABASE_URL", "ALPACA_MCP_OIDC_CLIENT_ID", "ALPACA_MCP_OIDC_CLIENT_SECRET",
                "ALPACA_MCP_JWT_SIGNING_KEY", "ALPACA_MCP_OIDC_CONFIG_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"ALPACA_MCP_AUTHKIT_ISSUER": ISSUER,
                       "ALPACA_MCP_PUBLIC_BASE_URL": BASE,
                       "ALPACA_MCP_OIDC_AUDIENCE": RESOURCE,
                       "ALPACA_MCP_OAUTH_SCOPES": "alpaca:safe-write"}.items():
        monkeypatch.setenv(key, value)
    p = auth.build_managed_oidc_provider()
    app = FastMCP("discovery", auth=p).http_app(path="/mcp", stateless_http=False)
    with TestClient(app) as client:
        response = client.post("/mcp", json={})
        assert response.status_code == 401
        assert f'resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"' in (
            response.headers["www-authenticate"]
        )
        document = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert document == {"resource": RESOURCE, "authorization_servers": [ISSUER],
                            "scopes_supported": ["alpaca:safe-write"],
                            "bearer_methods_supported": ["header"]}
        for path in ("/authorize", "/token", "/register", "/auth/callback", "/consent"):
            assert client.get(path).status_code == 404
            assert client.post(path, json={}).status_code == 404
    assert p.token_verifier.audience == RESOURCE
    assert not hasattr(p, "_client_storage")
    assert not hasattr(p, "jwt_issuer")


@pytest.mark.parametrize("key", ["ALPACA_MCP_AUTHKIT_ISSUER", "ALPACA_MCP_PUBLIC_BASE_URL",
                                 "ALPACA_MCP_OIDC_AUDIENCE", "ALPACA_MCP_OAUTH_SCOPES"])
@pytest.mark.parametrize("bad", [None, "", " "])
def test_each_hosted_setting_required(monkeypatch, key, bad):
    values = {"ALPACA_MCP_AUTHKIT_ISSUER": ISSUER, "ALPACA_MCP_PUBLIC_BASE_URL": BASE,
              "ALPACA_MCP_OIDC_AUDIENCE": RESOURCE, "ALPACA_MCP_OAUTH_SCOPES": "openid"}
    for k, v in values.items():
        monkeypatch.setenv(k, v)
    if bad is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, bad)
    with pytest.raises(auth.AuthenticationConfigurationError):
        auth.build_managed_oidc_provider()


@pytest.mark.parametrize("changes", [
    {"issuer": ISSUER + "/"}, {"issuer": ISSUER + "/tenant"},
    {"issuer": "http://issuer.example"}, {"issuer": "https://user:pass@issuer.example"},
    {"issuer": ISSUER + "?x"}, {"issuer": "https://ISSUER.example"},
    {"base_url": BASE + "/"}, {"base_url": "http://mcp.example"},
    {"audience": "https://other.example/mcp"}, {"audience": BASE},
    {"audience": RESOURCE + "#fragment"}, {"audience": RESOURCE + "?aud=x"},
    {"required_scopes": []}, {"required_scopes": [" "]}, {"required_scopes": ["a b"]},
])
def test_ambiguous_config_is_rejected(changes):
    with pytest.raises(auth.AuthenticationConfigurationError):
        provider(**changes)


def test_actual_mount_must_match_configured_resource():
    with pytest.raises(auth.AuthenticationConfigurationError):
        FastMCP("wrong-path", auth=provider()).http_app(path="/other")


@pytest.mark.parametrize("changes,missing", [
    ({"iss": ISSUER + "/"}, None), ({"iss": "https://other.example"}, None),
    ({"aud": BASE}, None), ({"aud": RESOURCE + "/"}, None),
    ({"scope": "openid"}, None), ({"scope": ""}, None),
    ({}, "sub"), ({"sub": None}, None), ({"sub": ""}, None), ({"sub": " "}, None),
])
async def test_signed_identity_issuer_resource_and_scope_rejections(signed, changes, missing):
    p, sign, _ = signed
    assert await p.verify_token(sign(changes, missing)) is None


async def test_signature_and_jwks_failure_are_closed(signed, monkeypatch):
    p, sign, _ = signed
    token = sign()
    wrong_key = RSAKeyPair.generate()
    monkeypatch.setattr(p.token_verifier, "_get_verification_key",
                        AsyncMock(return_value=wrong_key.public_key))
    assert await p.verify_token(token) is None
    monkeypatch.setattr(p.token_verifier, "_get_verification_key",
                        AsyncMock(side_effect=RuntimeError("test failure")))
    assert await p.verify_token(token) is None
    assert await p.verify_token("malformed") is None


async def test_resource_a_token_cannot_authenticate_b(signed, monkeypatch):
    a, sign, _ = signed
    b = provider(base_url="https://other.example", audience="https://other.example/mcp")
    monkeypatch.setattr(b.token_verifier, "_get_verification_key",
                        a.token_verifier._get_verification_key)
    assert await a.verify_token(sign()) is not None
    assert await b.verify_token(sign()) is None


@pytest.mark.parametrize("claim", ["nbf", "iat"])
@pytest.mark.parametrize("offset,allowed", [(59, True), (60, True), (60.001, False), (61, False)])
async def test_direct_future_boundaries(signed, claim, offset, allowed):
    p, sign, now = signed
    assert bool(await p.verify_token(sign({claim: now + offset}))) is allowed


@pytest.mark.parametrize("offset", [1, 30, 59, 60, 61])
async def test_no_expiration_grace(signed, offset):
    p, sign, now = signed
    assert await p.verify_token(sign({"exp": now - offset})) is None


def test_real_session_identity_and_caller_resource_cannot_override(signed):
    p, sign, now = signed
    calls = []
    mcp = FastMCP("direct-session", auth=p)

    @mcp.tool
    def identity() -> str:
        value = auth.authenticated_principal()
        calls.append(value)
        return value

    app = mcp.http_app(path="/mcp", stateless_http=False, json_response=True)
    with TestClient(app) as c:
        original = sign()
        headers = {"Authorization": "Bearer " + original,
                   "Accept": "application/json, text/event-stream"}
        r = c.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"}}})
        assert r.status_code == 200
        headers.update({"Mcp-Session-Id": r.headers["mcp-session-id"],
                        "MCP-Protocol-Version": "2025-06-18"})
        assert c.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "identity", "arguments": {}}}
        assert c.post("/mcp", headers=headers, json=request).status_code == 200
        expected = "oauth-v2-" + hashlib.sha256((ISSUER + "\0user-a").encode()).hexdigest()
        assert calls == [expected]
        for bearer, status in [(sign(missing="exp"), 401),
                               (sign({"nbf": now + 61}), 401),
                               (sign({"iat": now + 61}), 401),
                               (sign({"sub": "user-b"}), 404),
                               (sign({"aud": "https://attacker.example/mcp"}), 401),
                               ("malformed", 401), ("", 401)]:
            r = c.post("/mcp?resource=https://attacker.example/mcp&audience=anything",
                       headers={**headers, "Authorization": "Bearer " + bearer}, json=request)
            assert r.status_code == status
            assert calls == [expected]
        # A renewed token for the same user/client and the original token both
        # retain identity; neither a failed attack nor new exp changes the hash.
        for bearer in (sign({"exp": now + 1200}), original):
            assert c.post("/mcp", headers={**headers, "Authorization": "Bearer " + bearer},
                          json=request).status_code == 200
        assert calls == [expected] * 3
