"""Managed OIDC configuration and authenticated Safe Trading principals."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections.abc import Callable
from urllib.parse import urlsplit

from fastmcp.server.auth import AuthProvider
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.providers.workos import AuthKitProvider
from fastmcp.server.dependencies import get_access_token
from mcp.server.auth.routes import cors_middleware
from pydantic import AnyHttpUrl
from starlette.responses import JSONResponse
from starlette.routing import Route


class AuthenticationConfigurationError(RuntimeError):
    """Raised when hosted authentication cannot be configured safely."""


class PrincipalError(RuntimeError):
    """Raised when a validated authenticated principal is unavailable."""


PrincipalProvider = Callable[[], str]

# Bound ordinary clock drift without granting minutes of premature token use.
# Server-only policy: never sourced from request arguments or environment variables.
JWT_CLOCK_SKEW_SECONDS = 60


def _valid_time_claims(claims: dict) -> bool:
    """Check only verified claims; do not decode or re-verify JWTs here.

    The upstream verifier's stricter expiration check remains authoritative.
    This additional gate never rescues a token rejected upstream. No maximum
    lifetime is imposed: that requires a separate provider-backed policy.
    """
    now = time.time()
    for name in ("exp", "nbf", "iat"):
        if name not in claims:
            if name == "exp":
                return False
            continue
        value = claims[name]
        # JSON NumericDate permits fractional seconds, not strings or booleans.
        # Integers are finite without converting potentially huge ints to float.
        if type(value) not in (int, float):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if name == "exp":
            if now >= value + JWT_CLOCK_SKEW_SECONDS:
                return False
        elif value > now + JWT_CLOCK_SKEW_SECONDS:
            return False
    return True


def _https_url(value: str, *, origin: bool = False) -> str:
    """Require canonical server configuration, never normalize token issuers."""
    try:
        parsed = urlsplit(value)
        canonical = str(AnyHttpUrl(value))
        if (
            not value or value != value.strip() or parsed.scheme != "https"
            or not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "?" in value or "#" in value
            or any(ord(c) <= 32 or ord(c) == 127 for c in value)
            or (origin and parsed.path != "")
            or canonical != value + ("/" if parsed.path == "" else "")
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise AuthenticationConfigurationError("Hosted HTTPS URL is invalid") from exc
    return value


class HardenedAuthKitProvider(AuthKitProvider):
    """Direct AuthKit resource server with Phase 6 time and identity guards.

    No client secret, reference tokens, refresh mediation or OAuth state store.
    Keep the default trusted JWTVerifier and its audience auto-binding; a custom
    verifier argument is deliberately not exposed by this hosted adapter.
    """

    def __init__(self, *, issuer: str, base_url: str, audience: str,
                 required_scopes: list[str]):
        self.expected_issuer = _https_url(issuer, origin=True)
        self.expected_audience = _https_url(audience)
        _https_url(base_url, origin=True)
        if not audience.startswith(base_url + "/") or audience == base_url + "/":
            raise AuthenticationConfigurationError("Resource must include the MCP endpoint path")
        if not required_scopes or any(
            not isinstance(scope, str) or not re.fullmatch(r"[\x21\x23-\x5b\x5d-\x7e]+", scope)
            for scope in required_scopes
        ):
            raise AuthenticationConfigurationError("Hosted scopes are required")
        super().__init__(authkit_domain=issuer, base_url=base_url,
                         required_scopes=list(required_scopes))
        # Pin even before http_app()/get_routes() is called; never have an
        # audience-free validation window during construction.
        self.token_verifier.audience = audience

    def set_mcp_path(self, mcp_path: str | None) -> None:
        if str(self._get_resource_url(mcp_path)) != self.expected_audience:
            raise AuthenticationConfigurationError("MCP endpoint and configured resource disagree")
        super().set_mcp_path(mcp_path)

    def get_routes(self, mcp_path: str | None = None) -> list[Route]:
        routes = super().get_routes(mcp_path)

        async def metadata(request):
            # AnyHttpUrl serializes a root issuer with a trailing slash. Emit
            # our exact configured issuer instead, without changing JWT claims
            # or the canonical principal algorithm.
            return JSONResponse({
                "resource": self.expected_audience,
                "authorization_servers": [self.expected_issuer],
                "scopes_supported": self.token_verifier.scopes_supported,
                "bearer_methods_supported": ["header"],
            })

        return [
            Route(route.path, endpoint=cors_middleware(metadata, ["GET", "OPTIONS"]),
                  methods=["GET", "OPTIONS"])
            if route.path.startswith("/.well-known/oauth-protected-resource") else route
            for route in routes
        ]

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            verified = await super().verify_token(token)
        except Exception:
            # Includes malformed NumericDate overflow and JWKS/network errors.
            # No token or exception text is exposed; never fall back to anonymous.
            return None
        if verified is None:
            return None
        # The trusted verifier has checked signature/issuer/audience/scopes.
        # This gate only sees verified claims, including newly refreshed tokens.
        if not _valid_time_claims(verified.claims):
            return None
        issuer = verified.claims.get("iss")
        subject = verified.claims.get("sub")
        if (
            issuer != self.expected_issuer
            or not isinstance(subject, str) or not subject or subject != subject.strip()
            or not _audience_matches(verified.claims.get("aud"), self.expected_audience)
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
    """Build direct AuthKit auth. Kept under its existing factory API name."""
    try:
        scopes = _required("ALPACA_MCP_OAUTH_SCOPES").split()
        if not scopes:
            raise AuthenticationConfigurationError("Hosted OAuth configuration is incomplete")
        return HardenedAuthKitProvider(
            issuer=_required("ALPACA_MCP_AUTHKIT_ISSUER"),
            audience=_required("ALPACA_MCP_OIDC_AUDIENCE"),
            base_url=_required("ALPACA_MCP_PUBLIC_BASE_URL"),
            required_scopes=scopes,
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
