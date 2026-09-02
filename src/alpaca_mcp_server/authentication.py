"""Managed OIDC configuration and authenticated Safe Trading principals."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable

from fastmcp.server.auth import AuthProvider, OIDCProxy
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.dependencies import get_access_token
from key_value.aio.stores.postgresql import PostgreSQLStore


class AuthenticationConfigurationError(RuntimeError):
    """Raised when hosted authentication cannot be configured safely."""


class PrincipalError(RuntimeError):
    """Raised when a validated authenticated principal is unavailable."""


PrincipalProvider = Callable[[], str]


class SubjectBoundOIDCProxy(OIDCProxy):
    """Bridge FastMCP 3.2 verified claims to the MCP SDK session identity."""

    async def load_access_token(self, token: str) -> AccessToken | None:
        verified = await super().load_access_token(token)
        if verified is None:
            return None
        issuer = verified.claims.get("iss")
        subject = verified.claims.get("sub")
        audience = os.environ.get("ALPACA_MCP_OIDC_AUDIENCE", "")
        if (
            not isinstance(issuer, str) or not issuer or issuer != issuer.strip()
            or not isinstance(subject, str) or not subject or subject != subject.strip()
            or not audience or audience != audience.strip()
            or not _audience_matches(verified.claims.get("aud"), audience)
        ):
            return None
        # FastMCP 3.2.4 leaves the SDK's new subject field UNSET. Only bridge
        # that case; explicit null/empty/conflicting subjects remain invalid.
        if "subject" in verified.model_fields_set and verified.subject != subject:
            return None
        return verified.model_copy(update={"subject": subject})


def validate_secret(value: str, name: str) -> str:
    """Reject empty, short, whitespace-padded, or trivial hosted secrets."""
    if (
        not value
        or value != value.strip()
        or len(value) < 32
        or len(set(value)) < 8
        or value.lower() in {"change-me", "changeme", "test-secret", "secret"}
    ):
        raise AuthenticationConfigurationError(f"{name} is not securely configured")
    return value


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
        audience = _required("ALPACA_MCP_OIDC_AUDIENCE")
        return SubjectBoundOIDCProxy(
            config_url=_required("ALPACA_MCP_OIDC_CONFIG_URL"),
            client_id=_required("ALPACA_MCP_OIDC_CLIENT_ID"),
            client_secret=_required("ALPACA_MCP_OIDC_CLIENT_SECRET"),
            audience=audience,
            base_url=_required("ALPACA_MCP_PUBLIC_BASE_URL"),
            required_scopes=scopes,
            verify_id_token=False,
            redirect_path="/auth/callback",
            # WorkOS uses RFC 8707 resource, not the OAuth client ID. Keep both
            # authorization and token/refresh requests pinned to server configuration.
            forward_resource=False,
            extra_authorize_params={"resource": audience},
            extra_token_params={"resource": audience},
            client_storage=client_storage,
            jwt_signing_key=validate_secret(
                _required("ALPACA_MCP_JWT_SIGNING_KEY"), "Hosted JWT signing key"
            ),
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
    audience = token.claims.get("aud")
    expected_audience = os.environ.get("ALPACA_MCP_OIDC_AUDIENCE", "")
    # FastMCP 3.2 AccessToken carries the verified OIDC subject in claims.
    # Some verifier implementations also expose ``subject``; if present it must agree.
    subject = getattr(token, "subject", claim_subject)
    if (
        not isinstance(issuer, str)
        or not issuer
        or issuer != issuer.strip()
        or not isinstance(subject, str)
        or not subject
        or subject != subject.strip()
        or not isinstance(claim_subject, str)
        or not claim_subject
        or claim_subject != claim_subject.strip()
        or claim_subject != subject
        or not expected_audience
        or expected_audience != expected_audience.strip()
        or not _audience_matches(audience, expected_audience)
    ):
        raise PrincipalError("Authenticated issuer, subject, and configured audience are required")
    identity = f"{issuer}\0{subject}".encode()
    return "oauth-v2-" + hashlib.sha256(identity).hexdigest()


def _audience_matches(value: object, expected: str) -> bool:
    """Accept the configured audience as a scalar or one member of a JWT audience list."""
    if isinstance(value, str):
        return value == expected
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, str) for item in value) and expected in value
    return False


def local_principal_provider(principal: str) -> PrincipalProvider:
    """Create the explicit non-hosted/test identity path."""

    def provide() -> str:
        if not principal:
            raise PrincipalError("Local Safe Trading principal is unavailable")
        return principal

    return provide
