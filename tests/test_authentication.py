"""Authentication configuration, endpoint protection, and principal tests."""

from __future__ import annotations

from importlib.metadata import version
import time
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlparse

import pytest
from authlib.jose import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth.oauth_proxy import OAuthProxy
from fastmcp.server.auth.oauth_proxy.models import OAuthTransaction
from fastmcp.server.auth.oidc_proxy import OIDCConfiguration, OIDCProxy
from key_value.aio.stores.memory import MemoryStore
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
from starlette.testclient import TestClient

from alpaca_mcp_server import authentication
from alpaca_mcp_server.authentication import (
    AuthenticationConfigurationError,
    PrincipalError,
    authenticated_principal,
    build_managed_oidc_provider,
)
from alpaca_mcp_server.server import build_server


AUDIENCE = "https://mcp.example/mcp"


@pytest.mark.parametrize("body,expected", [
    ("a=1&b=2", 200),
    ("a=1&b=2&c=3", 400),
    ("a=" + "x" * 17, 400),
])
def test_urlencoded_form_limits_use_real_starlette_parser(body, expected):
    """OAuth uses Request.form(); bounded inputs exercise its actual ASGI path."""
    async def parse_form(request):
        async with request.form(max_fields=2, max_part_size=16) as form:
            return Response(str(len(form)))

    app = Starlette(routes=[Route("/form", parse_form, methods=["POST"])])
    with TestClient(app) as client:
        response = client.post(
            "/form", content=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == expected


def test_oauth_routes_with_starlette_lifespan():
    """Real OAuth routes reject invalid requests without contacting an IdP."""
    from fastmcp import FastMCP

    proxy = OAuthProxy(
        upstream_authorization_endpoint="https://idp.example/authorize",
        upstream_token_endpoint="https://idp.example/token",
        upstream_client_id="test-app",
        upstream_client_secret="test-only-client-secret",
        token_verifier=StaticTokenVerifier({}),
        base_url="https://mcp.example",
        client_storage=MemoryStore(),
        jwt_signing_key="test-only-signing-key-with-at-least-32-characters",
    )
    app = FastMCP("auth-compatibility", auth=proxy).http_app()
    with TestClient(app) as client:
        assert client.get("/authorize").status_code == 400
        assert client.get("/auth/callback").status_code == 400
        assert client.post("/token", data={"grant_type": "invalid"}).status_code in (400, 401)
        metadata = client.get("/.well-known/oauth-authorization-server")
        assert metadata.status_code == 200
        assert metadata.json()["token_endpoint"] == "https://mcp.example/token"
        resource = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resource.status_code == 200
        assert resource.json()["resource"] == AUDIENCE
        registration = client.post("/register", json={
            "client_name": "test-client", "redirect_uris": ["https://client.example/callback"],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        })
        assert registration.status_code == 201
        assert registration.json()["client_id"]
    # Construction only: no long-lived SSE connection or external traffic.
    assert FastMCP("sse-test", auth=proxy).http_app(transport="sse") is not None


@pytest.mark.asyncio
async def test_upstream_refresh_preserves_fixed_resource_and_state(monkeypatch):
    from fastmcp.server.auth.oauth_proxy.models import UpstreamTokenSet

    proxy = OAuthProxy(
        upstream_authorization_endpoint="https://idp.example/authorize",
        upstream_token_endpoint="https://idp.example/token",
        upstream_client_id="test-app", upstream_client_secret="test-only-secret",
        token_verifier=StaticTokenVerifier({}), base_url="https://mcp.example",
        client_storage=MemoryStore(),
        jwt_signing_key="test-only-signing-key-with-at-least-32-characters",
        forward_resource=False, extra_token_params={"resource": AUDIENCE},
    )
    upstream = Mock(refresh_token=AsyncMock(return_value={
        "access_token": "test-refreshed-access", "refresh_token": "test-refreshed-refresh",
        "expires_in": 300,
    }))
    monkeypatch.setattr(proxy, "_create_upstream_oauth_client", lambda: upstream)
    old = UpstreamTokenSet(
        upstream_token_id="refresh-test", access_token="test-old-access",
        refresh_token="test-old-refresh", refresh_token_expires_at=time.time() + 3600,
        expires_at=time.time() - 1, token_type="Bearer", scope="openid",
        client_id="same-client", created_at=time.time(),
    )
    new = await proxy._try_transparent_refresh(old)
    upstream.refresh_token.assert_awaited_once()
    assert upstream.refresh_token.call_args.kwargs["resource"] == AUDIENCE
    assert new.client_id == "same-client"
    assert (await proxy._upstream_token_store.get(key="refresh-test")).access_token == "test-refreshed-access"


@pytest.mark.asyncio
async def test_multipart_excessive_headers_rejected_by_request_form():
    from python_multipart.exceptions import MultipartParseError

    body = (b"--boundary\r\nContent-Disposition: form-data; name=\"a\"\r\n"
            + b"X-Test: value\r\n" * 9 + b"\r\nvalue\r\n--boundary--\r\n")
    request = Request({
        "type": "http", "headers": [
            (b"content-type", b"multipart/form-data; boundary=boundary"),
        ],
    }, receive=AsyncMock(return_value={"type": "http.request", "body": body}))
    with pytest.raises(MultipartParseError, match="header"):
        await request.form()


def test_semicolon_payload_is_data_not_quadratic_field_separator():
    async def parse_form(request):
        async with request.form(max_fields=1, max_part_size=512) as form:
            assert list(form) == ["value"]
            return Response(form["value"])

    payload = "a;" * 128
    with TestClient(Starlette(routes=[Route("/form", parse_form, methods=["POST"])])) as client:
        response = client.post("/form", content="value=" + payload,
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        assert response.status_code == 200
        assert response.text == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", [None, "", "other-user"])
async def test_subject_bridge_rejects_explicit_invalid_subject(monkeypatch, subject):
    from fastmcp.server.auth.auth import AccessToken

    value = AccessToken(token="test", client_id="same-client", scopes=[], subject=subject,
                        claims={"iss": "https://issuer.example", "sub": "user-1", "aud": AUDIENCE})
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(OIDCProxy, "load_access_token", AsyncMock(return_value=value))
    provider = object.__new__(authentication.SubjectBoundOIDCProxy)
    assert await provider.load_access_token("test") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"iss": None}, {"sub": None}, {"sub": " "}, {"aud": None},
    {"aud": "https://wrong.example"}, {"aud": [AUDIENCE, 42]},
])
async def test_subject_bridge_rejects_malformed_verified_claims(monkeypatch, overrides):
    from fastmcp.server.auth.auth import AccessToken

    value = AccessToken(token="test", client_id="same-client", scopes=[], claims={
        "iss": "https://issuer.example", "sub": "user-1", "aud": AUDIENCE, **overrides,
    })
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(OIDCProxy, "load_access_token", AsyncMock(return_value=value))
    provider = object.__new__(authentication.SubjectBoundOIDCProxy)
    assert await provider.load_access_token("test") is None


def test_stateful_session_binds_verified_user_not_only_oauth_client(
    monkeypatch, configured_oidc_env, oidc_jwt_material,
):
    from fastmcp import FastMCP

    private_key, public_key = oidc_jwt_material
    discovery = OIDCConfiguration(
        issuer="https://issuer.example", authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token", jwks_uri="https://issuer.example/jwks",
        response_types_supported=["code"], subject_types_supported=["public"],
        id_token_signing_alg_values_supported=["RS256"],
    )
    monkeypatch.setattr(OIDCProxy, "get_oidc_configuration", lambda *args: discovery)
    monkeypatch.setattr(authentication, "PostgreSQLStore", lambda **kwargs: MemoryStore())
    verifier = JWTVerifier(public_key=public_key, issuer="https://issuer.example",
                           audience=AUDIENCE, algorithm="RS256")

    async def verified_upstream(self, value):
        return await verifier.verify_token(value)

    # Only bypass proxy-token storage/exchange; signatures, subject bridge,
    # authentication middleware and the stateful HTTP session manager are real.
    monkeypatch.setattr(OIDCProxy, "load_access_token", verified_upstream)
    provider = build_managed_oidc_provider()
    app = FastMCP("session-test", auth=provider).http_app(
        path="/mcp", stateless_http=False, json_response=True,
    )
    a = _signed_token(private_key, sub="user-a", client_id="same-client",
                      scope="alpaca:read alpaca:safe-write")
    b = _signed_token(private_key, sub="user-b", client_id="same-client",
                      scope="alpaca:read alpaca:safe-write")
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + a, "Accept": "application/json, text/event-stream"}
        response = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "session-test", "version": "1"},
            },
        })
        assert response.status_code == 200
        headers["Mcp-Session-Id"] = response.headers["mcp-session-id"]
        headers["MCP-Protocol-Version"] = "2025-06-18"
        assert client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "method": "notifications/initialized",
        }).status_code == 202
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        assert client.post("/mcp", headers=headers, json=request).status_code == 200
        assert client.post("/mcp", headers={**headers, "Authorization": "Bearer " + b},
                           json=request).status_code == 404
        for auth in ("", "Bearer invalid"):
            assert client.post("/mcp", headers={**headers, "Authorization": auth},
                               json=request).status_code == 401
        assert client.post("/mcp", headers=headers, json=request).status_code == 200


def token(
    *,
    issuer="https://issuer.example",
    subject="user-1",
    claim_subject="user-1",
    audience=AUDIENCE,
):
    return Mock(subject=subject, claims={"iss": issuer, "sub": claim_subject, "aud": audience})


def test_principal_is_stable_and_binds_issuer_and_subject(monkeypatch):
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(authentication, "get_access_token", lambda: token())
    first = authenticated_principal()
    assert first == authenticated_principal() and first.startswith("oauth-v2-")
    monkeypatch.setattr(
        authentication, "get_access_token", lambda: token(issuer="https://other.example")
    )
    assert authenticated_principal() != first


@pytest.mark.parametrize(
    "value",
    [
        None,
        token(subject=None),
        token(subject=""),
        token(issuer=None),
        token(claim_subject="other"),
        token(claim_subject=None),
    ],
)
def test_missing_or_inconsistent_authenticated_identity_fails(monkeypatch, value):
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(authentication, "get_access_token", lambda: value)
    with pytest.raises(PrincipalError):
        authenticated_principal()


def test_hosted_oidc_configuration_missing_fails_closed(monkeypatch):
    for name in (
        "ALPACA_MCP_OAUTH_SCOPES",
        "ALPACA_MCP_OIDC_CONFIG_URL",
        "ALPACA_MCP_OIDC_CLIENT_ID",
        "ALPACA_MCP_OIDC_CLIENT_SECRET",
        "ALPACA_MCP_OIDC_AUDIENCE",
        "ALPACA_MCP_PUBLIC_BASE_URL",
        "ALPACA_MCP_JWT_SIGNING_KEY",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AuthenticationConfigurationError):
        build_server(hosted_mode=True)


@pytest.mark.parametrize("secret", ["", "short", "a" * 64])
def test_hosted_ownership_secret_must_be_strong(monkeypatch, secret):
    verifier = StaticTokenVerifier({})
    monkeypatch.setenv("ALPACA_SAFE_OWNERSHIP_SECRET", secret)
    with pytest.raises(AuthenticationConfigurationError):
        build_server(hosted_mode=True, auth_provider=verifier)


def test_hosted_secrets_must_be_independent(monkeypatch):
    shared = "shared-secret-with-enough-variety-123456789"
    monkeypatch.setenv("ALPACA_SAFE_OWNERSHIP_SECRET", shared)
    monkeypatch.setenv("ALPACA_MCP_JWT_SIGNING_KEY", shared)
    with pytest.raises(AuthenticationConfigurationError):
        build_server(hosted_mode=True, auth_provider=StaticTokenVerifier({}))


@pytest.fixture
def configured_oidc_env(monkeypatch):
    values = {
        "ALPACA_MCP_OAUTH_SCOPES": "alpaca:read alpaca:safe-write",
        "ALPACA_MCP_OIDC_CONFIG_URL": "https://idp.example/.well-known/openid-configuration",
        "ALPACA_MCP_OIDC_CLIENT_ID": "client",
        "ALPACA_MCP_OIDC_CLIENT_SECRET": "secret",
        "ALPACA_MCP_OIDC_AUDIENCE": AUDIENCE,
        "ALPACA_MCP_PUBLIC_BASE_URL": "https://mcp.example",
        "ALPACA_MCP_JWT_SIGNING_KEY": "test-signing-key-with-32-plus-characters-123",
        "DATABASE_URL": "postgresql://test:test@localhost/test",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_managed_oidc_provider_uses_locked_fastmcp_architecture(monkeypatch, configured_oidc_env):
    constructor = Mock(return_value=Mock())
    storage = Mock(return_value=Mock())
    monkeypatch.setattr(authentication, "SubjectBoundOIDCProxy", constructor)
    monkeypatch.setattr(authentication, "PostgreSQLStore", storage)
    assert build_managed_oidc_provider() is constructor.return_value
    kwargs = constructor.call_args.kwargs
    assert kwargs["required_scopes"] == ["alpaca:read", "alpaca:safe-write"]
    assert kwargs["audience"] == AUDIENCE
    assert kwargs["verify_id_token"] is False
    assert kwargs["forward_resource"] is False
    assert kwargs["extra_authorize_params"] == {"resource": AUDIENCE}
    assert kwargs["extra_token_params"] == {"resource": AUDIENCE}
    assert kwargs["redirect_path"] == "/auth/callback"
    assert kwargs["client_storage"] is storage.return_value
    assert kwargs["jwt_signing_key"] == "test-signing-key-with-32-plus-characters-123"
    assert kwargs["require_authorization_consent"] is True


@pytest.mark.parametrize("audience", [None, "", " ", " https://mcp.example/mcp"])
def test_hosted_requires_configured_audience(monkeypatch, configured_oidc_env, audience):
    if audience is None:
        monkeypatch.delenv("ALPACA_MCP_OIDC_AUDIENCE")
    else:
        monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", audience)
    constructor = Mock()
    monkeypatch.setattr(authentication, "SubjectBoundOIDCProxy", constructor)
    monkeypatch.setattr(authentication, "PostgreSQLStore", Mock())
    with pytest.raises(AuthenticationConfigurationError):
        build_managed_oidc_provider()
    constructor.assert_not_called()


def test_real_oidc_proxy_wires_jwks_issuer_audience_and_resource(
    monkeypatch, configured_oidc_env
):
    discovery = OIDCConfiguration(
        issuer="https://issuer.example",
        authorization_endpoint="https://issuer.example/authorize",
        token_endpoint="https://issuer.example/token",
        jwks_uri="https://issuer.example/jwks",
        response_types_supported=["code"],
        subject_types_supported=["public"],
        id_token_signing_alg_values_supported=["RS256"],
    )
    monkeypatch.setattr(OIDCProxy, "get_oidc_configuration", lambda *args: discovery)
    monkeypatch.setattr(authentication, "PostgreSQLStore", lambda **kwargs: MemoryStore())
    provider = build_managed_oidc_provider()
    assert isinstance(provider, OIDCProxy)
    assert provider._token_validator.audience == AUDIENCE
    assert provider._token_validator.issuer == "https://issuer.example"
    assert provider._token_validator.jwks_uri == "https://issuer.example/jwks"
    assert provider._token_validator.required_scopes == ["alpaca:read", "alpaca:safe-write"]
    assert provider._verify_id_token is False
    assert provider._forward_resource is False
    assert provider._extra_authorize_params["resource"] == AUDIENCE
    assert provider._extra_token_params["resource"] == AUDIENCE
    provider.set_mcp_path("/mcp")
    assert str(provider._resource_url) == AUDIENCE


@pytest.mark.parametrize(
    "audience",
    [None, "https://wrong.example/mcp", 42, {"resource": AUDIENCE}, [], [AUDIENCE, 42]],
)
def test_principal_rejects_missing_wrong_or_malformed_audience(monkeypatch, audience):
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(authentication, "get_access_token", lambda: token(audience=audience))
    with pytest.raises(PrincipalError):
        authenticated_principal()


def test_principal_accepts_expected_audience_in_list(monkeypatch):
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(
        authentication,
        "get_access_token",
        lambda: token(audience=["https://another.example", AUDIENCE]),
    )
    assert authenticated_principal().startswith("oauth-v2-")


def test_principal_rejects_missing_server_audience(monkeypatch):
    monkeypatch.delenv("ALPACA_MCP_OIDC_AUDIENCE", raising=False)
    monkeypatch.setattr(authentication, "get_access_token", lambda: token())
    with pytest.raises(PrincipalError):
        authenticated_principal()


def test_fastmcp_oauth_proxy_has_browser_bound_consent_cookie_patch():
    assert version("fastmcp") == "3.2.4"
    proxy = object.__new__(OAuthProxy)
    proxy._is_https = True
    proxy._upstream_client_secret = None
    proxy._jwt_signing_key = b"test-only-signing-key-with-32-bytes-minimum"
    txn_id = "transaction-1"
    consent_token = "consent-token-1"
    empty_request = Request({"type": "http", "headers": []})
    assert not proxy._verify_consent_binding_cookie(empty_request, txn_id, consent_token)

    response = Response()
    proxy._set_consent_binding_cookie(empty_request, response, txn_id, consent_token)
    cookie = response.headers["set-cookie"].split(";", 1)[0]
    bound_request = Request(
        {"type": "http", "headers": [(b"cookie", cookie.encode("ascii"))]}
    )
    assert proxy._verify_consent_binding_cookie(bound_request, txn_id, consent_token)
    assert not proxy._verify_consent_binding_cookie(bound_request, txn_id, "forged")


@pytest.fixture
def oidc_jwt_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _signed_token(private_key, **overrides):
    now = int(time.time())
    claims = {
        "iss": "https://issuer.example",
        "sub": "user-1",
        "aud": AUDIENCE,
        "scope": "openid alpaca:safe-write",
        "iat": now,
        "exp": now + 300,
    }
    claims.update(overrides)
    return jwt.encode({"alg": "RS256"}, claims, private_key).decode()


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_valid_oidc_identity_and_audience(oidc_jwt_material):
    private_key, public_key = oidc_jwt_material
    verifier = JWTVerifier(
        public_key=public_key,
        issuer="https://issuer.example",
        audience=AUDIENCE,
        algorithm="RS256",
        required_scopes=["openid", "alpaca:safe-write"],
    )
    value = await verifier.verify_token(_signed_token(private_key))
    assert value is not None
    assert value.claims["sub"] == "user-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"iss": "https://wrong.example"},
        {"iss": None},
        {"aud": "https://wrong.example/mcp"},
        {"aud": None},
        {"aud": 42},
        {"aud": {"resource": AUDIENCE}},
        {"exp": 1},
        {"scope": "openid"},
    ],
)
async def test_jwt_verifier_rejects_invalid_claims(oidc_jwt_material, overrides):
    private_key, public_key = oidc_jwt_material
    verifier = JWTVerifier(
        public_key=public_key,
        issuer="https://issuer.example",
        audience=AUDIENCE,
        algorithm="RS256",
        required_scopes=["openid", "alpaca:safe-write"],
    )
    assert await verifier.verify_token(_signed_token(private_key, **overrides)) is None


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_expected_audience_in_list(oidc_jwt_material):
    private_key, public_key = oidc_jwt_material
    verifier = JWTVerifier(
        public_key=public_key,
        issuer="https://issuer.example",
        audience=AUDIENCE,
        algorithm="RS256",
        required_scopes=["openid", "alpaca:safe-write"],
    )
    value = await verifier.verify_token(
        _signed_token(private_key, aud=["https://another.example", AUDIENCE])
    )
    assert value is not None


@pytest.mark.asyncio
async def test_real_access_token_claims_drive_principal_not_client_id(
    oidc_jwt_material, monkeypatch
):
    private_key, public_key = oidc_jwt_material
    verifier = JWTVerifier(
        public_key=public_key, issuer="https://issuer.example", audience=AUDIENCE,
        algorithm="RS256", required_scopes=["openid"],
    )
    value = await verifier.verify_token(_signed_token(private_key, client_id="client-a"))
    assert value is not None
    monkeypatch.setenv("ALPACA_MCP_OIDC_AUDIENCE", AUDIENCE)
    monkeypatch.setattr(OIDCProxy, "load_access_token", AsyncMock(return_value=value))
    provider = object.__new__(authentication.SubjectBoundOIDCProxy)
    value = await provider.load_access_token("mock-proxy-token")
    assert value is not None and value.subject == "user-1"
    monkeypatch.setattr(authentication, "get_access_token", lambda: value)
    first = authenticated_principal()
    value.client_id = "client-b"
    assert authenticated_principal() == first


@pytest.mark.asyncio
async def test_callback_requires_same_browser_before_upstream_token_exchange(monkeypatch):
    """Run the installed callback handler; only the upstream IdP is mocked."""
    proxy = OAuthProxy(
        upstream_authorization_endpoint="https://idp.example/authorize",
        upstream_token_endpoint="https://idp.example/token",
        upstream_client_id="upstream-app",
        upstream_client_secret="test-only-client-secret",
        token_verifier=StaticTokenVerifier({}),
        base_url="https://mcp.example",
        client_storage=MemoryStore(),
        jwt_signing_key="test-only-signing-key-with-at-least-32-characters",
        require_authorization_consent=True,
        forward_resource=False,
        extra_authorize_params={"resource": AUDIENCE},
        extra_token_params={"resource": AUDIENCE},
    )
    transaction = OAuthTransaction(
        txn_id="txn", client_id="claude-test",
        client_redirect_uri="https://claude.ai/api/mcp/auth_callback",
        client_state="client-state", code_challenge="challenge", code_challenge_method="S256",
        scopes=["openid"], created_at=time.time(), consent_token="browser-binding",
        resource="https://attacker.example",
    )
    await proxy._transaction_store.put(key="txn", value=transaction)
    upstream = Mock(fetch_token=AsyncMock(return_value={
        "access_token": "mock-upstream-token", "token_type": "Bearer", "expires_in": 300,
    }))
    factory = Mock(return_value=upstream)
    monkeypatch.setattr(proxy, "_create_upstream_oauth_client", factory)
    request = Request({
        "type": "http", "headers": [], "query_string": b"state=txn&code=idp-code",
    })
    denied = await proxy._handle_idp_callback(request)
    assert denied.status_code == 403
    factory.assert_not_called()
    assert await proxy._transaction_store.get(key="txn") is not None

    response = Response()
    proxy._set_consent_binding_cookie(request, response, "txn", "browser-binding")
    cookie = response.headers["set-cookie"].split(";", 1)[0]
    bound = Request({
        "type": "http", "headers": [(b"cookie", cookie.encode())],
        "query_string": b"state=txn&code=idp-code",
    })
    allowed = await proxy._handle_idp_callback(bound)
    assert allowed.status_code == 302
    assert allowed.headers["location"].startswith(transaction.client_redirect_uri + "?")
    upstream.fetch_token.assert_awaited_once()
    assert upstream.fetch_token.call_args.kwargs["resource"] == AUDIENCE
    query = parse_qs(urlparse(proxy._build_upstream_authorize_url(
        "txn", transaction.model_dump()
    )).query)
    assert query["resource"] == [AUDIENCE]


def test_remote_endpoint_protects_initialize_list_and_call(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_TOOLSETS", "")
    monkeypatch.setenv(
        "ALPACA_SAFE_OWNERSHIP_SECRET", "independent-ownership-secret-not-production-1234"
    )
    monkeypatch.setenv("ALPACA_MCP_JWT_SIGNING_KEY", "independent-jwt-secret-not-production-1234")
    verifier = StaticTokenVerifier(
        {
            "valid": {
                "client_id": "claude-test",
                "scopes": ["alpaca:mcp"],
                "subject": "user-1",
                "claims": {"iss": "https://issuer.example", "sub": "user-1"},
            }
        },
        required_scopes=["alpaca:mcp"],
    )
    server = build_server(
        hosted_mode=True,
        auth_provider=verifier,
        principal_provider=lambda: "test-principal",
    )
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search_readme", "arguments": {"query": "safe"}},
        },
    ]
    with TestClient(app) as client:
        assert all(client.post("/mcp", json=body).status_code == 401 for body in requests)
        headers = {"Authorization": "Bearer valid", "Accept": "application/json"}
        assert all(
            client.post("/mcp", json=body, headers=headers).status_code == 200 for body in requests
        )
