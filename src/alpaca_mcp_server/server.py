"""
Alpaca MCP Server v2 — FastMCP + OpenAPI

Builds MCP tools from Alpaca's OpenAPI specs at process init time.
No hand-crafted tool functions except for overrides (e.g., order placement).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.server.auth import AuthProvider
from fastmcp.server.providers.openapi.routing import MCPType

from .authentication import (
    AuthenticationConfigurationError,
    PrincipalProvider,
    authenticated_principal,
    build_managed_oidc_provider,
    local_principal_provider,
    validate_secret,
)
from .readme_docs import ReadMeClientFactory, register_readme_docs_tools
from .risk_store import PostgresRiskStore, RiskStore, UnavailableRiskStore
from .security import TrustBoundaryMiddleware
from .tool_registry import TOOL_DESCRIPTIONS, TOOL_NAMES
from .toolsets import OVERRIDE_OPERATION_IDS, TOOLSETS, get_active_operations

SPECS_DIR = Path(__file__).parent / "specs"

MARKET_DATA_BASE_URL = "https://data.alpaca.markets"


def strip_v_from_version(release_version: str) -> str:
    if release_version[:1] == "v" and release_version[1:2].isdigit():
        return release_version[1:]

    return release_version


def get_mcp_user_agent() -> str:
    release_version = version("alpaca-mcp-server")
    return f"APCA-MCP-TRADING/{strip_v_from_version(release_version)}"


_OPENAPI_LITERAL_FIELDS = frozenset({"default", "example", "examples"})
_OPENAPI_NAMED_MAP_FIELDS = frozenset({"properties", "schemas"})


def _strip_openapi_vendor_extensions(value: Any) -> Any:
    """Remove extensions without dropping x-prefixed schema data."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key.startswith("x-"):
                continue
            if key in _OPENAPI_LITERAL_FIELDS:
                cleaned[key] = item
            elif key in _OPENAPI_NAMED_MAP_FIELDS and isinstance(item, dict):
                cleaned[key] = {
                    name: _strip_openapi_vendor_extensions(entry) for name, entry in item.items()
                }
            else:
                cleaned[key] = _strip_openapi_vendor_extensions(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_openapi_vendor_extensions(item) for item in value]
    return value


def _load_spec(name: str) -> dict[str, Any]:
    path = SPECS_DIR / f"{name}.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    return _strip_openapi_vendor_extensions(spec)


def _make_filter(allowed_ops: set[str]):
    """Create a route_map_fn that includes only allowlisted operationIds."""

    def filter_fn(route, _default_type):
        if route.operation_id in allowed_ops and route.operation_id not in OVERRIDE_OPERATION_IDS:
            return MCPType.TOOL
        return MCPType.EXCLUDE

    return filter_fn


def _get_read_operation_ids(spec: dict[str, Any]) -> set[str]:
    """Return operationIds for GET endpoints only."""
    return {
        operation["operationId"]
        for path_item in spec.get("paths", {}).values()
        for method, operation in path_item.items()
        if method.lower() == "get" and isinstance(operation, dict) and "operationId" in operation
    }


def _safe_mode_enabled(*, hosted_mode: bool = False) -> bool:
    """V2 never registers legacy writes, including stdio."""
    return True


def _make_customizer(descriptions: dict[str, str]):
    """Create an mcp_component_fn that overrides descriptions where provided."""

    def customizer(route, component):
        if route.operation_id in descriptions:
            component.description = descriptions[route.operation_id]

    return customizer


def _build_auth_headers() -> dict[str, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY", "")
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    user_agent = os.environ.get("ALPACA_MCP_USER_AGENT")
    if user_agent is None:
        user_agent = get_mcp_user_agent()
    if user_agent.strip():
        headers["User-Agent"] = user_agent.strip()
    return headers


def _get_trading_base_url() -> str:
    from .paper import require_paper

    return require_paper()


def _ensure_scheme(url: str) -> str:
    """Prepend ``https://`` when the URL has no scheme (common .env misconfiguration)."""
    if url and "://" not in url:
        return f"https://{url}"
    return url


def _parse_toolsets() -> set[str] | None:
    raw = os.environ.get("ALPACA_TOOLSETS", "").strip()
    if not raw:
        return None
    return {t.strip() for t in raw.split(",") if t.strip()}


def _make_api_client(base_url: str, headers: dict[str, str]) -> httpx.AsyncClient:
    client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)
    if "User-Agent" not in headers:
        client.headers.pop("User-Agent", None)
    return client


def build_server(
    readme_client_factory: ReadMeClientFactory | None = None,
    risk_store: RiskStore | None = None,
    *,
    hosted_mode: bool = False,
    auth_provider: AuthProvider | None = None,
    principal_provider: PrincipalProvider | None = None,
) -> FastMCP:
    """Construct the Alpaca MCP server from OpenAPI specs."""
    active_toolsets = _parse_toolsets()
    spec_ops = get_active_operations(active_toolsets)
    safe_mode = _safe_mode_enabled(hosted_mode=hosted_mode)
    ownership_secret: str | None = None
    if hosted_mode:
        auth_provider = auth_provider or build_managed_oidc_provider()
        principal_provider = principal_provider or authenticated_principal
        ownership_secret = validate_secret(
            os.environ.get("ALPACA_SAFE_OWNERSHIP_SECRET", ""), "Safe ownership secret"
        )
        if risk_store is None and not os.environ.get("DATABASE_URL", "").strip():
            raise AuthenticationConfigurationError("Hosted risk database is required")
    else:
        principal_provider = principal_provider or local_principal_provider(
            os.environ.get("ALPACA_SAFE_PRINCIPAL", "")
        )

    auth_headers = _build_auth_headers()
    trading_base = _get_trading_base_url() if "trading" in spec_ops or hosted_mode else None
    data_base = _ensure_scheme(os.environ.get("DATA_API_URL", MARKET_DATA_BASE_URL)).rstrip("/")

    clients: list[httpx.AsyncClient] = []
    if risk_store is None:
        database_url = os.environ.get("DATABASE_URL", "")
        risk_store = PostgresRiskStore(database_url) if database_url else UnavailableRiskStore()

    trading_client: httpx.AsyncClient | None = None
    if "trading" in spec_ops:
        trading_client = _make_api_client(trading_base, auth_headers)
        clients.append(trading_client)

    data_client: httpx.AsyncClient | None = None
    if "market-data" in spec_ops:
        data_client = _make_api_client(data_base, auth_headers)
        clients.append(data_client)

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[dict]:
        try:
            await risk_store.open()
            if trading_client is not None and safe_mode:
                from .reconciliation import reconcile_pending
                from .safe_overrides import _proof

                ownership_secret = os.environ.get("ALPACA_SAFE_OWNERSHIP_SECRET", "")
                if ownership_secret:
                    await reconcile_pending(
                        trading_client,
                        risk_store,
                        ownership_proof=lambda operation: _proof(operation, ownership_secret),
                    )
            yield {}
        finally:
            await risk_store.close()
            for c in clients:
                await c.aclose()

    main = FastMCP("Alpaca MCP Server", lifespan=lifespan, auth=auth_provider)
    main.add_middleware(TrustBoundaryMiddleware())

    if trading_client is not None:
        allowed = spec_ops["trading"]
        spec = _load_spec("trading-api")
        if safe_mode:
            allowed = allowed & _get_read_operation_ids(spec)
        sub = FastMCP.from_openapi(
            spec,
            client=trading_client,
            name="Alpaca Trading",
            mcp_names=TOOL_NAMES,
            route_map_fn=_make_filter(allowed),
            mcp_component_fn=_make_customizer(TOOL_DESCRIPTIONS),
            validate_output=False,
        )
        main.mount(sub)

    if data_client is not None:
        allowed = spec_ops["market-data"]
        spec = _load_spec("market-data-api")
        sub = FastMCP.from_openapi(
            spec,
            client=data_client,
            name="Alpaca Market Data",
            mcp_names=TOOL_NAMES,
            route_map_fn=_make_filter(allowed),
            mcp_component_fn=_make_customizer(TOOL_DESCRIPTIONS),
            validate_output=False,
        )
        main.mount(sub)

    active_ts = active_toolsets if active_toolsets is not None else set(TOOLSETS.keys())

    if trading_client is not None and "trading" in active_ts:
        _register_safe_trading_overrides(
            main, trading_client, risk_store, principal_provider, ownership_secret
        )

    if data_client is not None and active_ts & {"stock-data", "crypto-data"}:
        _register_market_data_overrides(main, data_client)

    register_readme_docs_tools(main, client_factory=readme_client_factory)

    return main


def _register_safe_trading_overrides(
    server: FastMCP,
    trading_client: httpx.AsyncClient,
    risk_store: RiskStore,
    principal_provider: PrincipalProvider,
    ownership_secret: str | None = None,
) -> None:
    """Register the only write tools permitted by Safe Trading V2."""
    from .safe_overrides import register_safe_trading_tools

    register_safe_trading_tools(
        server,
        trading_client,
        risk_store,
        principal_provider=principal_provider,
        ownership_secret=(
            ownership_secret
            if ownership_secret is not None
            else os.environ.get("ALPACA_SAFE_OWNERSHIP_SECRET", "")
        ),
    )


def _register_market_data_overrides(server: FastMCP, data_client: httpx.AsyncClient) -> None:
    """Register hand-crafted override tools for historical market data."""
    from .market_data_overrides import register_market_data_tools

    register_market_data_tools(server, data_client)
