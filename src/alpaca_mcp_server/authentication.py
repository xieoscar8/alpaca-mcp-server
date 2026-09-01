"""Managed OIDC configuration and authenticated Safe Trading principals."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

from fastmcp.server.auth import AuthProvider, OIDCProxy
from fastmcp.server.dependencies import get_access_token
from key_value.aio.stores.postgresql import PostgreSQLStore


class AuthenticationConfigurationError(RuntimeError):
    """Raised when hosted authentication cannot be configured safely."""


class PrincipalError(RuntimeError):
    """Raised when a validated authenticated principal is unavailable."""


PrincipalProvider = Callable[[], str]


def _required(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or value != value.strip():
        raise AuthenticationConfigurationError("Hosted OAuth configuration is incomplete")
    return value


def build_managed_oidc_provider() -> AuthProvider:
    """Build FastMCP's managed OIDC proxy without exposing configuration values."""
    try:
        scopes = _required("ALPACA_MCP_OAUTH_SCOPES").split()
        if not scopes:
            raise AuthenticationConfigurationError("Hosted OAuth configuration is incomplete")
        client_storage = PostgreSQLStore(
            url=_required("DATABASE_URL"),
            table_name="safe_v2_oauth_state",
        )
        return OIDCProxy(
            config_url=_required("ALPACA_MCP_OIDC_CONFIG_URL"),
            client_id=_required("ALPACA_MCP_OIDC_CLIENT_ID"),
            client_secret=_required("ALPACA_MCP_OIDC_CLIENT_SECRET"),
            base_url=_required("ALPACA_MCP_PUBLIC_BASE_URL"),
            required_scopes=scopes,
            redirect_path="/auth/callback",
            client_storage=client_storage,
            jwt_signing_key=_required("ALPACA_MCP_JWT_SIGNING_KEY"),
            require_authorization_consent=True,
        )
    except AuthenticationConfigurationError:
        raise
    except Exception as exc:
        raise AuthenticationConfigurationError("Hosted OAuth configuration is invalid") from exc


def authenticated_principal() -> str:
    """Derive a stable opaque principal from a validated token's issuer and subject."""
    token = get_access_token()
    if token is None:
        raise PrincipalError("Authenticated identity is unavailable")
    issuer = token.claims.get("iss")
    claim_subject = token.claims.get("sub")
    subject = token.subject
    if (
        not isinstance(issuer, str)
        or not issuer
        or issuer != issuer.strip()
        or not isinstance(subject, str)
        or not subject
        or subject != subject.strip()
        or (claim_subject is not None and claim_subject != subject)
    ):
        raise PrincipalError("Authenticated issuer and subject are required")
    identity = f"{issuer}\0{subject}".encode()
    return "oauth-v2-" + hashlib.sha256(identity).hexdigest()


def local_principal_provider(principal: str) -> PrincipalProvider:
    """Create the explicit non-hosted/test identity path."""

    def provide() -> str:
        if not principal:
            raise PrincipalError("Local Safe Trading principal is unavailable")
        return principal

    return provide
