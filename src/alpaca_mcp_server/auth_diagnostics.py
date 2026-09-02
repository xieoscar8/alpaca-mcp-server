"""Temporary read-only summary of the current authenticated request context."""

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.dependencies import get_access_token


def auth_claims_summary() -> dict[str, bool | str]:
    """Construct only allowlisted values; never serialize token or claim data."""
    empty: dict[str, bool | str] = {
        "verified_context_present": False,
        "org_id_present": False,
        "role_present": False,
        "role_value": "absent",
        "permissions_present": False,
        "permissions_shape_valid": False,
        "paper_trading_present": False,
        "alpaca_role_present": False,
        "alpaca_role_value": "absent",
    }
    try:
        token = get_access_token()
        if not isinstance(token, AccessToken) or not isinstance(token.claims, dict):
            return empty
        claims = token.claims
        role_present = "role" in claims
        role = claims.get("role")
        role_value = "absent"
        if role_present:
            role_value = "other_or_invalid"
            if isinstance(role, str) and role == "member":
                role_value = "member"
            elif isinstance(role, str) and role == "paper-trader":
                role_value = "paper-trader"
        alpaca_role_present = "alpaca_role" in claims
        alpaca_role = claims.get("alpaca_role")
        alpaca_role_value = "absent"
        if alpaca_role_present:
            alpaca_role_value = "other_or_invalid"
            if isinstance(alpaca_role, str) and alpaca_role == "paper-trader":
                alpaca_role_value = "paper-trader"
            elif isinstance(alpaca_role, str) and alpaca_role == "member":
                alpaca_role_value = "member"
        permissions = claims.get("permissions")
        # Intentionally mirror the production gate without changing it.
        shape_valid = (
            isinstance(permissions, list)
            and bool(permissions)
            and all(
                isinstance(item, str) and item and item == item.strip()
                for item in permissions
            )
        )
        return {
            "verified_context_present": True,
            "org_id_present": "org_id" in claims,
            "role_present": role_present,
            "role_value": role_value,
            "permissions_present": "permissions" in claims,
            "permissions_shape_valid": shape_valid,
            "paper_trading_present": shape_valid and "paper-trading" in permissions,
            "alpaca_role_present": alpaca_role_present,
            "alpaca_role_value": alpaca_role_value,
        }
    except Exception:
        return empty


def register_auth_diagnostic_tool(server: FastMCP) -> None:
    """Register only on the authenticated hosted server, never local stdio."""
    @server.tool(annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    })
    def debug_auth_claims_summary() -> dict[str, bool | str]:
        """Return a temporary fixed, redacted summary of authenticated claim presence."""
        return auth_claims_summary()
