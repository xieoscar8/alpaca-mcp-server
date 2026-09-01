"""Authentication configuration, endpoint protection, and principal tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from starlette.testclient import TestClient

from alpaca_mcp_server import authentication
from alpaca_mcp_server.authentication import (
    AuthenticationConfigurationError,
    PrincipalError,
    authenticated_principal,
    build_managed_oidc_provider,
)
from alpaca_mcp_server.server import build_server


def token(*, issuer="https://issuer.example", subject="user-1", claim_subject="user-1"):
    return Mock(subject=subject, claims={"iss": issuer, "sub": claim_subject})


def test_principal_is_stable_and_binds_issuer_and_subject(monkeypatch):
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
    ],
)
def test_missing_or_inconsistent_authenticated_identity_fails(monkeypatch, value):
    monkeypatch.setattr(authentication, "get_access_token", lambda: value)
    with pytest.raises(PrincipalError):
        authenticated_principal()


def test_hosted_oidc_configuration_missing_fails_closed(monkeypatch):
    for name in (
        "ALPACA_MCP_OAUTH_SCOPES",
        "ALPACA_MCP_OIDC_CONFIG_URL",
        "ALPACA_MCP_OIDC_CLIENT_ID",
        "ALPACA_MCP_OIDC_CLIENT_SECRET",
        "ALPACA_MCP_PUBLIC_BASE_URL",
        "ALPACA_MCP_JWT_SIGNING_KEY",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AuthenticationConfigurationError):
        build_server(hosted_mode=True)


def test_managed_oidc_provider_uses_locked_fastmcp_architecture(monkeypatch):
    values = {
        "ALPACA_MCP_OAUTH_SCOPES": "alpaca:read alpaca:safe-write",
        "ALPACA_MCP_OIDC_CONFIG_URL": "https://idp.example/.well-known/openid-configuration",
        "ALPACA_MCP_OIDC_CLIENT_ID": "client",
        "ALPACA_MCP_OIDC_CLIENT_SECRET": "secret",
        "ALPACA_MCP_PUBLIC_BASE_URL": "https://mcp.example",
        "ALPACA_MCP_JWT_SIGNING_KEY": "test-signing-key",
        "DATABASE_URL": "postgresql://test:test@localhost/test",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    constructor = Mock(return_value=Mock())
    storage = Mock(return_value=Mock())
    monkeypatch.setattr(authentication, "OIDCProxy", constructor)
    monkeypatch.setattr(authentication, "PostgreSQLStore", storage)
    assert build_managed_oidc_provider() is constructor.return_value
    kwargs = constructor.call_args.kwargs
    assert kwargs["required_scopes"] == ["alpaca:read", "alpaca:safe-write"]
    assert kwargs["redirect_path"] == "/auth/callback"
    assert kwargs["client_storage"] is storage.return_value
    assert kwargs["jwt_signing_key"] == "test-signing-key"
    assert kwargs["require_authorization_consent"] is True


def test_remote_endpoint_protects_initialize_list_and_call(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ALPACA_TOOLSETS", "")
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
