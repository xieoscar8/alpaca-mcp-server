"""Server-enforced Safe Trading V1 write tools."""

from __future__ import annotations

import os
import re
import uuid
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

import httpx
from fastmcp import FastMCP

DEFAULT_MAX_ORDER_NOTIONAL = Decimal(100)
DEFAULT_MIN_CRYPTO_NOTIONAL = Decimal(10)
DEFAULT_MAX_CLOSE_MARKET_VALUE = Decimal(25)
PAPER_ONLY_ERROR = "Safe Trading V1 write operations are Paper-only."
ORDER_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$")
POSITION_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}|[A-Za-z0-9][A-Za-z0-9.-]*(?:/[A-Za-z0-9][A-Za-z0-9.-]*)?)$"
)


def _positive_decimal(value: object, field: str) -> tuple[Decimal | None, dict | None]:
    text = str(value)
    if text != text.strip():
        return None, _error(f"{field} must not contain surrounding whitespace")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None, _error(f"{field} must be a valid positive decimal")
    if not number.is_finite() or number <= 0:
        return None, _error(f"{field} must be a finite positive decimal")
    return number, None


def _safe_limit(name: str, default: Decimal, *, is_minimum: bool = False) -> Decimal:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        return default
    if not value.is_finite() or value <= 0:
        return default
    # V1 environment overrides may tighten safety, never expand its envelope.
    return max(value, default) if is_minimum else min(value, default)


def _paper_trade_enabled() -> bool:
    return os.environ.get("ALPACA_PAPER_TRADE", "").strip().lower() in {
        "true", "1", "yes"
    }


def _audit(**extra: object) -> dict:
    result = {
        "safe_mode": True,
        "paper_trade": _paper_trade_enabled(),
        "max_order_notional": str(
            _safe_limit("ALPACA_SAFE_MAX_ORDER_NOTIONAL", DEFAULT_MAX_ORDER_NOTIONAL)
        ),
        "min_crypto_notional": str(
            _safe_limit(
                "ALPACA_SAFE_MIN_CRYPTO_NOTIONAL",
                DEFAULT_MIN_CRYPTO_NOTIONAL,
                is_minimum=True,
            )
        ),
        "max_close_market_value": str(
            _safe_limit("ALPACA_SAFE_MAX_CLOSE_MARKET_VALUE", DEFAULT_MAX_CLOSE_MARKET_VALUE)
        ),
    }
    result.update(extra)
    return result


def _error(message: str, **extra: object) -> dict:
    return {"error": {"message": message}, **_audit(**extra)}


def _paper_gate() -> dict | None:
    return None if _paper_trade_enabled() else _error(PAPER_ONLY_ERROR, risk_checks=["paper_only_failed"])


def _client_order_id(value: str | None) -> tuple[str | None, dict | None]:
    order_id = value if value is not None else f"safe-{uuid.uuid4()}"
    if (
        not isinstance(order_id, str)
        or not 1 <= len(order_id) <= 128
        or order_id != order_id.strip()
        or not order_id.isprintable()
    ):
        return None, _error(
            "client_order_id must be 1-128 printable characters without surrounding whitespace"
        )
    return order_id, None


def _api_error(message: str, response: httpx.Response, **extra: object) -> dict:
    try:
        detail = response.json()
    except ValueError:
        detail = {"raw": response.text}
    return _error(message, http_status=response.status_code, detail=detail, **extra)


async def _place(client: httpx.AsyncClient, body: dict, audit: dict) -> dict:
    try:
        response = await client.post("/v2/orders", json=body)
    except httpx.ReadTimeout:
        return _error(
            "The order MAY have been placed. Do not retry the POST; query by client_order_id.",
            uncertain=True,
            client_order_id=body["client_order_id"],
            **audit,
        )
    if response.is_error:
        return _api_error("API rejected the order", response, **audit)
    return {"result": response.json(), **_audit(**audit)}


def register_safe_trading_tools(server: FastMCP, client: httpx.AsyncClient) -> None:
    annotations = {
        "readOnlyHint": False, "destructiveHint": True,
        "idempotentHint": False, "openWorldHint": True,
    }

    @server.tool(annotations={"title": "Safely Place Stock Order", **annotations})
    async def safe_place_stock_order(
        symbol: str,
        side: str,
        qty: str | None = None,
        notional: str | None = None,
        type: str = "limit",
        time_in_force: str = "day",
        limit_price: str | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Place a buy-only stock limit order within the server-side notional cap."""
        gate = _paper_gate()
        if gate:
            return gate
        if side != "buy":
            return _error("Safe stock orders only allow side=buy")
        if type != "limit":
            return _error("Safe stock orders only allow type=limit")
        if limit_price is None:
            return _error("limit_price is required")
        if (qty is None) == (notional is None):
            return _error("Provide exactly one of qty or notional")
        price, error = _positive_decimal(limit_price, "limit_price")
        if error:
            return error
        amount_input = qty if qty is not None else notional
        amount, error = _positive_decimal(amount_input, "qty" if qty is not None else "notional")
        if error:
            return error
        estimated = amount * price if qty is not None else amount
        maximum = _safe_limit("ALPACA_SAFE_MAX_ORDER_NOTIONAL", DEFAULT_MAX_ORDER_NOTIONAL)
        if estimated > maximum:
            return _error("Estimated notional exceeds the safe maximum", estimated_notional=str(estimated))
        order_id, error = _client_order_id(client_order_id)
        if error:
            return error
        body = {"symbol": symbol, "side": "buy", "type": "limit", "time_in_force": time_in_force,
                "limit_price": str(limit_price), "client_order_id": order_id}
        body["qty" if qty is not None else "notional"] = str(amount_input)
        return await _place(client, body, {
            "risk_checks": ["paper_only", "buy_only", "limit_only", "notional_cap"],
            "estimated_notional": str(estimated),
        })

    @server.tool(annotations={"title": "Safely Place Crypto Order", **annotations})
    async def safe_place_crypto_order(
        symbol: str,
        side: str,
        qty: str | None = None,
        notional: str | None = None,
        type: str = "limit",
        time_in_force: str = "gtc",
        limit_price: str | None = None,
        client_order_id: str | None = None,
    ) -> dict:
        """Place a buy-only crypto limit order within the safe notional range."""
        gate = _paper_gate()
        if gate:
            return gate
        if side != "buy":
            return _error("Safe crypto orders only allow side=buy")
        if type != "limit":
            return _error("Safe crypto orders only allow type=limit")
        if time_in_force not in {"gtc", "ioc"}:
            return _error("Safe crypto orders only allow time_in_force=gtc or ioc")
        if limit_price is None:
            return _error("limit_price is required")
        if (qty is None) == (notional is None):
            return _error("Provide exactly one of qty or notional")
        price, error = _positive_decimal(limit_price, "limit_price")
        if error:
            return error
        amount_input = qty if qty is not None else notional
        amount, error = _positive_decimal(amount_input, "qty" if qty is not None else "notional")
        if error:
            return error
        estimated = amount * price if qty is not None else amount
        minimum = _safe_limit(
            "ALPACA_SAFE_MIN_CRYPTO_NOTIONAL",
            DEFAULT_MIN_CRYPTO_NOTIONAL,
            is_minimum=True,
        )
        maximum = _safe_limit("ALPACA_SAFE_MAX_ORDER_NOTIONAL", DEFAULT_MAX_ORDER_NOTIONAL)
        if estimated < minimum or estimated > maximum:
            return _error("Estimated crypto notional is outside the safe range", estimated_notional=str(estimated))
        order_id, error = _client_order_id(client_order_id)
        if error:
            return error
        body = {"symbol": symbol, "side": "buy", "type": "limit", "time_in_force": time_in_force,
                "limit_price": str(limit_price), "client_order_id": order_id}
        body["qty" if qty is not None else "notional"] = str(amount_input)
        return await _place(client, body, {
            "risk_checks": ["paper_only", "buy_only", "limit_only", "notional_range"],
            "estimated_notional": str(estimated),
        })

    @server.tool(annotations={"title": "Safely Cancel One Order", **annotations})
    async def safe_cancel_order(order_id: str) -> dict:
        """Pre-check and cancel exactly one explicitly identified order."""
        gate = _paper_gate()
        if gate:
            return gate
        if not isinstance(order_id, str) or not ORDER_ID_RE.fullmatch(order_id):
            return _error("A single explicit order_id is required")
        path = f"/v2/orders/{quote(order_id, safe='')}"
        try:
            check = await client.get(path)
        except httpx.ReadTimeout:
            return _error("Order pre-check timed out; cancellation was not attempted")
        if check.is_error:
            return _api_error("Order pre-check failed; cancellation was not attempted", check)
        try:
            checked_id = check.json()["id"]
        except (KeyError, TypeError, ValueError):
            return _error("Order pre-check response was malformed; cancellation was not attempted")
        if checked_id != order_id:
            return _error("Order pre-check returned a different ID; cancellation was not attempted")
        try:
            response = await client.delete(path)
        except httpx.ReadTimeout:
            return _error(
                "The cancellation MAY have been submitted. Do not retry; query the order.",
                uncertain=True,
            )
        if response.is_error:
            return _api_error("API rejected the cancellation", response)
        return {"result": response.json() if response.content else {"cancelled_order_id": order_id},
                **_audit(risk_checks=["paper_only", "single_order_precheck", "single_cancel"])}

    @server.tool(annotations={"title": "Safely Close One Position", **annotations})
    async def safe_close_position(symbol_or_asset_id: str) -> dict:
        """Fully close one small position after a server-side market-value check."""
        gate = _paper_gate()
        if gate:
            return gate
        if (
            not isinstance(symbol_or_asset_id, str)
            or symbol_or_asset_id.lower() == "all"
            or not POSITION_ID_RE.fullmatch(symbol_or_asset_id)
        ):
            return _error("A single explicit symbol or asset_id is required")
        path = f"/v2/positions/{quote(symbol_or_asset_id, safe='')}"
        try:
            check = await client.get(path)
        except httpx.ReadTimeout:
            return _error("Position pre-check timed out; close was not attempted")
        if check.is_error:
            return _api_error("Position pre-check failed; close was not attempted", check)
        try:
            market_value_raw = check.json()["market_value"]
        except (KeyError, TypeError, ValueError):
            return _error("Position market_value is missing; close was not attempted")
        market_value_text = str(market_value_raw)
        if market_value_text != market_value_text.strip():
            return _error("Position market_value is invalid; close was not attempted")
        try:
            market_value = abs(Decimal(market_value_text))
        except (InvalidOperation, ValueError):
            return _error("Position market_value is invalid; close was not attempted")
        if not market_value.is_finite():
            return _error("Position market_value is invalid; close was not attempted")
        maximum = _safe_limit("ALPACA_SAFE_MAX_CLOSE_MARKET_VALUE", DEFAULT_MAX_CLOSE_MARKET_VALUE)
        if market_value > maximum:
            return _error("Position market value exceeds the safe close maximum", market_value=str(market_value))
        try:
            response = await client.delete(path)
        except httpx.ReadTimeout:
            return _error(
                "The position close MAY have been submitted. Do not retry; query the position and orders.",
                uncertain=True, risk_checks=["paper_only", "single_position_precheck", "full_close"],
            )
        if response.is_error:
            return _api_error("API rejected the position close", response)
        return {"result": response.json() if response.content else {"closed": symbol_or_asset_id},
                **_audit(risk_checks=["paper_only", "single_position_precheck", "market_value_cap", "full_close"],
                         market_value=str(market_value))}
