"""Server-enforced Safe Trading V2 write tools."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from decimal import Decimal, InvalidOperation
from urllib.parse import quote

import httpx
from fastmcp import FastMCP

from .authentication import PrincipalError, PrincipalProvider, has_paper_trading_permission
from .paper import paper_client, paper_enabled
from .reconciliation import reconcile_pending
from .risk_store import Operation, RiskLimitExceeded, RiskLimits, RiskStore, RiskStoreError

MAX_ORDER = Decimal(100)
MIN_CRYPTO = Decimal(10)
HARD_OPEN_ORDERS = 5
HARD_OPEN_NOTIONAL = Decimal(300)
HARD_DAILY_NOTIONAL = Decimal(500)
HARD_SYMBOL_NOTIONAL = Decimal(200)
ORDER_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)
STOCK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
CRYPTO_RE = re.compile(r"^[A-Za-z0-9]+/[A-Za-z0-9]+$")
OWNER_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _paper() -> bool:
    return paper_enabled()


def _error(message: str, **extra: object) -> dict:
    return {
        "error": {"message": message},
        "safe_mode": True,
        "safe_trading_version": "v2",
        "paper_trade": _paper(),
        **extra,
    }


def _number(value: object, field: str) -> tuple[Decimal | None, dict | None]:
    if not isinstance(value, str) or not 1 <= len(value) <= 64 or not value.isascii():
        return None, _error(f"{field} must be a bounded decimal string")
    text = value
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text) is None:
        return None, _error(f"{field} must be a valid positive decimal")
    if text != text.strip():
        return None, _error(f"{field} must not contain surrounding whitespace")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None, _error(f"{field} must be a valid positive decimal")
    if not number.is_finite() or number <= 0:
        return None, _error(f"{field} must be a finite positive decimal")
    if len(number.as_tuple().digits) > 12 or not -12 <= number.as_tuple().exponent <= 12:
        return None, _error(f"{field} exceeds decimal precision or exponent bounds")
    if not -12 <= number.adjusted() <= 12:
        return None, _error(f"{field} exceeds decimal magnitude bounds")
    return number, None


def _decimal_limit(name: str, hard: Decimal) -> Decimal:
    raw = os.environ.get(name)
    if raw is None:
        return hard
    value, error = _number(raw, name)
    return hard if error else min(value, hard)


def _integer_limit(name: str, hard: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return hard
    try:
        value = int(raw)
    except ValueError:
        return hard
    return min(value, hard) if value > 0 and str(value) == raw else hard


def _limits() -> RiskLimits:
    return RiskLimits(
        max_open_orders=_integer_limit("ALPACA_SAFE_MAX_OPEN_ORDERS", HARD_OPEN_ORDERS),
        max_open_notional=_decimal_limit("ALPACA_SAFE_MAX_OPEN_NOTIONAL", HARD_OPEN_NOTIONAL),
        max_daily_submitted_notional=_decimal_limit(
            "ALPACA_SAFE_MAX_DAILY_SUBMITTED_NOTIONAL", HARD_DAILY_NOTIONAL
        ),
        max_symbol_open_notional=_decimal_limit(
            "ALPACA_SAFE_MAX_SYMBOL_OPEN_NOTIONAL", HARD_SYMBOL_NOTIONAL
        ),
    )


def _symbol(value: object, crypto: bool) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 32 or value != value.strip() or not value.isprintable():
        return False
    return (CRYPTO_RE if crypto else STOCK_RE).fullmatch(value) is not None


def _canonical_symbol(value: str) -> str:
    """Collapse only broker-safe case-equivalent symbol forms."""
    return value.upper()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _request_fingerprint(*, body: dict[str, str], asset_type: str) -> str:
    intent = {"asset_type": asset_type, **body}
    serialized = json.dumps(intent, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _owner_field(value: object) -> bool:
    return isinstance(value, str) and OWNER_RE.fullmatch(value) is not None


def _client_id(secret: str, principal: str, strategy: str, key: str) -> str:
    data = f"v2\0{principal}\0{strategy}\0{key}".encode()
    return "safe-v2-" + hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()


def _proof(operation: Operation, secret: str) -> bool:
    expected = _client_id(
        secret, operation.principal, operation.strategy_id, operation.idempotency_key
    )
    return hmac.compare_digest(expected, operation.client_order_id)


def _replay(operation: Operation) -> dict:
    return {
        "safe_mode": True,
        "safe_trading_version": "v2",
        "paper_trade": True,
        "idempotent_replay": True,
        "client_order_id": operation.client_order_id,
        "order_id": operation.order_id,
        "status": operation.status,
        "uncertain": operation.uncertain,
        "estimated_notional": str(operation.requested_notional),
    }


async def _post(
    client: httpx.AsyncClient, store: RiskStore, body: dict, operation: Operation
) -> dict:
    if not paper_client(client):
        return _error("Safe Trading requires a verified Paper transport")
    try:
        response = await client.post("/v2/orders", json=body)
    except httpx.RequestError:
        try:
            await store.mark_uncertain(operation.client_order_id)
        except RiskStoreError:
            pass
        return _error(
            "Order MAY have been placed; do not retry automatically",
            uncertain=True,
            client_order_id=operation.client_order_id,
        )
    if response.status_code == 408 or response.status_code >= 500:
        try:
            await store.mark_uncertain(operation.client_order_id)
        except RiskStoreError:
            pass
        return _error(
            "Order MAY have been placed; do not retry automatically",
            uncertain=True,
            client_order_id=operation.client_order_id,
            http_status=response.status_code,
        )
    if response.is_error:
        try:
            await store.mark_rejected(operation.client_order_id)
        except RiskStoreError:
            return _error("Order rejected but risk-state update failed", uncertain=True)
        return _error("API rejected the order", http_status=response.status_code)
    try:
        payload = response.json()
        order_id = str(payload["id"])
        returned_client_id = payload["client_order_id"]
    except (KeyError, TypeError, ValueError):
        order_id, returned_client_id = "", None
    if not ORDER_ID_RE.fullmatch(order_id) or returned_client_id != operation.client_order_id:
        try:
            await store.mark_uncertain(operation.client_order_id)
        except RiskStoreError:
            pass
        return _error(
            "Order response ownership mismatch; reservation remains uncertain",
            uncertain=True,
            client_order_id=operation.client_order_id,
        )
    try:
        await store.mark_submitted(operation.client_order_id, order_id)
    except RiskStoreError:
        try:
            await store.mark_uncertain(operation.client_order_id)
        except RiskStoreError:
            pass
        return _error(
            "Order accepted but ownership persistence failed",
            uncertain=True,
            client_order_id=operation.client_order_id,
        )
    return {
        "result": payload,
        "safe_mode": True,
        "safe_trading_version": "v2",
        "paper_trade": True,
        "client_order_id": operation.client_order_id,
        "estimated_notional": str(operation.requested_notional),
        "risk_checks": ["paper_only", "durable_reservation", "cumulative_limits"],
    }


async def _reserve(
    *,
    client: httpx.AsyncClient,
    store: RiskStore,
    principal_provider: PrincipalProvider,
    secret: str,
    strategy: str,
    key: str,
    symbol: str,
    asset_type: str,
    estimated: Decimal,
    body: dict,
    reconcile_before_write: bool,
) -> dict:
    if not _owner_field(strategy) or not _owner_field(key):
        return _error("strategy_id and idempotency_key must be 1-64 safe characters")
    if not paper_client(client):
        return _error("Safe Trading requires a verified Paper transport")
    try:
        principal = principal_provider()
    except PrincipalError:
        return _error("Authenticated Safe V2 principal is unavailable")
    if not principal or not secret:
        return _error("Safe V2 server ownership configuration is unavailable")
    if reconcile_before_write:
        await reconcile_pending(
            client,
            store,
            ownership_proof=lambda operation: _proof(operation, secret),
            principal=principal,
        )
    client_order_id = _client_id(secret, principal, strategy, key)
    request_fingerprint = _request_fingerprint(body=body, asset_type=asset_type)
    try:
        reservation = await store.reserve(
            principal=principal,
            strategy_id=strategy,
            idempotency_key=key,
            client_order_id=client_order_id,
            symbol=symbol,
            asset_type=asset_type,
            request_fingerprint=request_fingerprint,
            requested_notional=estimated,
            limits=_limits(),
        )
    except RiskLimitExceeded as exc:
        return _error(str(exc), risk_checks=["cumulative_limit_failed"])
    except RiskStoreError:
        return _error("Safe V2 risk reservation failed; order was not sent")
    if not reservation.created:
        return _replay(reservation.operation)
    body["client_order_id"] = client_order_id
    return await _post(client, store, body, reservation.operation)


def register_safe_trading_tools(
    server: FastMCP,
    client: httpx.AsyncClient,
    store: RiskStore,
    *,
    principal_provider: PrincipalProvider,
    ownership_secret: str,
    reconcile_before_write: bool = True,
) -> None:
    annotations = {
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    }

    @server.tool(annotations={"title": "Safely Place Stock Order", **annotations})
    async def safe_place_stock_order(
        symbol: str,
        side: str,
        strategy_id: str,
        idempotency_key: str,
        qty: str | None = None,
        notional: str | None = None,
        type: str = "limit",
        time_in_force: str = "day",
        limit_price: str | None = None,
    ) -> dict:
        """Place a V2-owned cumulative-risk-limited stock order."""
        if not has_paper_trading_permission():
            return _error("Safe Trading requires paper-trading permission")
        if not paper_client(client):
            return _error("Safe Trading V2 write operations are Paper-only.")
        if not _symbol(symbol, False):
            return _error("Invalid single stock symbol")
        symbol = _canonical_symbol(symbol)
        if side != "buy":
            return _error("Safe stock orders only allow side=buy")
        if type != "limit":
            return _error("Safe stock orders only allow type=limit")
        if time_in_force not in {"day", "gtc"}:
            return _error("Stock TIF must be day or gtc")
        if limit_price is None:
            return _error("limit_price is required")
        if (qty is None) == (notional is None):
            return _error("Provide exactly one of qty or notional")
        price, error = _number(limit_price, "limit_price")
        if error:
            return error
        amount_input = qty if qty is not None else notional
        amount, error = _number(amount_input, "qty" if qty is not None else "notional")
        if error:
            return error
        estimated = amount * price if qty is not None else amount
        if estimated > MAX_ORDER:
            return _error("Estimated notional exceeds $100")
        body = {
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": _decimal_text(price),
            "qty" if qty is not None else "notional": _decimal_text(amount),
        }
        # Read-only asset proof precedes reconciliation and reservation.
        if re.fullmatch(r"[A-Z0-9.]+[0-9]{6}[CP][0-9]{8}", symbol):
            return _error("Options are not allowed by the stock tool")
        try:
            asset_response = await client.get(f"/v2/assets/{quote(symbol, safe='')}")
            if asset_response.status_code != 200:
                return _error("Equity asset verification failed")
            asset = asset_response.json()
            if (
                not isinstance(asset, dict)
                or asset.get("symbol") != symbol
                or asset.get("class", asset.get("asset_class")) != "us_equity"
                or ("class" in asset and asset["class"] != "us_equity")
                or ("asset_class" in asset and asset["asset_class"] != "us_equity")
                or asset.get("tradable") is not True
                or asset.get("status") != "active"
            ):
                return _error("A verified active tradable equity is required")
        except (httpx.RequestError, ValueError, TypeError):
            return _error("Equity asset verification failed")
        return await _reserve(
            client=client,
            store=store,
            principal_provider=principal_provider,
            secret=ownership_secret,
            strategy=strategy_id,
            key=idempotency_key,
            symbol=symbol,
            asset_type="stock",
            estimated=estimated,
            body=body,
            reconcile_before_write=reconcile_before_write,
        )

    @server.tool(annotations={"title": "Safely Place Crypto Order", **annotations})
    async def safe_place_crypto_order(
        symbol: str,
        side: str,
        strategy_id: str,
        idempotency_key: str,
        qty: str | None = None,
        notional: str | None = None,
        type: str = "limit",
        time_in_force: str = "gtc",
        limit_price: str | None = None,
    ) -> dict:
        """Place a V2-owned cumulative-risk-limited crypto order."""
        if not has_paper_trading_permission():
            return _error("Safe Trading requires paper-trading permission")
        if not paper_client(client):
            return _error("Safe Trading V2 write operations are Paper-only.")
        if not _symbol(symbol, True):
            return _error("Invalid single crypto pair")
        symbol = _canonical_symbol(symbol)
        if side != "buy":
            return _error("Safe crypto orders only allow side=buy")
        if type != "limit":
            return _error("Safe crypto orders only allow type=limit")
        if time_in_force not in {"gtc", "ioc"}:
            return _error("Crypto TIF must be gtc or ioc")
        if limit_price is None:
            return _error("limit_price is required")
        if (qty is None) == (notional is None):
            return _error("Provide exactly one of qty or notional")
        price, error = _number(limit_price, "limit_price")
        if error:
            return error
        amount_input = qty if qty is not None else notional
        amount, error = _number(amount_input, "qty" if qty is not None else "notional")
        if error:
            return error
        estimated = amount * price if qty is not None else amount
        if estimated < MIN_CRYPTO or estimated > MAX_ORDER:
            return _error("Estimated crypto notional must be between $10 and $100")
        body = {
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": _decimal_text(price),
            "qty" if qty is not None else "notional": _decimal_text(amount),
        }
        return await _reserve(
            client=client,
            store=store,
            principal_provider=principal_provider,
            secret=ownership_secret,
            strategy=strategy_id,
            key=idempotency_key,
            symbol=symbol,
            asset_type="crypto",
            estimated=estimated,
            body=body,
            reconcile_before_write=reconcile_before_write,
        )

    @server.tool(annotations={"title": "Safely Cancel Owned Order", **annotations})
    async def safe_cancel_order(order_id: str, strategy_id: str) -> dict:
        """Cancel one order proven to belong to this principal and strategy."""
        if not has_paper_trading_permission():
            return _error("Safe Trading requires paper-trading permission")
        if not paper_client(client):
            return _error("Safe Trading V2 write operations are Paper-only.")
        if not ORDER_ID_RE.fullmatch(order_id):
            return _error("A single order UUID is required")
        if not _owner_field(strategy_id):
            return _error("A valid strategy_id is required")
        try:
            principal = principal_provider()
        except PrincipalError:
            return _error("Authenticated Safe V2 principal is unavailable")
        if not principal or not ownership_secret:
            return _error("Ownership configuration unavailable")
        path = f"/v2/orders/{quote(order_id, safe='')}"
        try:
            response = await client.get(path)
        except httpx.RequestError:
            return _error("Order pre-check timed out")
        if response.status_code != 200:
            return _error("Order pre-check failed")
        try:
            fetched = response.json()
            fetched_id = fetched["id"]
            client_id = fetched["client_order_id"]
        except (KeyError, TypeError, ValueError):
            return _error("Malformed order pre-check response")
        if fetched_id != order_id or not isinstance(client_id, str):
            return _error("Alpaca order mismatch")
        if not client_id.startswith("safe-v2-"):
            return _error("Order is not Safe V2-owned")
        try:
            operation = await store.get_by_client_order_id(client_id)
        except RiskStoreError:
            return _error("Ownership lookup failed")
        if operation is None or not _proof(operation, ownership_secret):
            return _error("Ownership proof invalid")
        if fetched.get("symbol") != operation.symbol:
            return _error("Alpaca and ownership symbols mismatch")
        if operation.principal != principal or operation.strategy_id != strategy_id:
            return _error("Order belongs to another principal or strategy")
        if operation.status == "uncertain" and operation.order_id is None:
            try:
                operation = await store.reconcile_uncertain_order(client_id, order_id)
            except RiskStoreError:
                return _error("Uncertain ownership reconciliation failed")
        if operation.status != "submitted" or operation.uncertain:
            return _error("Order ownership state does not permit cancellation")
        if operation.order_id != order_id or operation.client_order_id != client_id:
            return _error("Alpaca and ownership records mismatch")
        try:
            await store.begin_cancel(client_id)
        except RiskStoreError:
            return _error("Cancellation reservation failed; DELETE was not sent")
        if not paper_client(client):
            return _error("Safe Trading requires a verified Paper transport")
        try:
            deleted = await client.delete(path)
        except httpx.RequestError:
            return _error("Cancellation MAY have been submitted", uncertain=True)
        if deleted.status_code == 408 or deleted.status_code >= 500:
            return _error(
                "Cancellation MAY have been submitted",
                uncertain=True,
                http_status=deleted.status_code,
            )
        if deleted.is_error:
            try:
                await store.mark_cancel_rejected(client_id)
            except RiskStoreError:
                return _error("Cancellation rejection persistence failed", uncertain=True)
            return _error("API rejected cancellation", http_status=deleted.status_code)
        if deleted.status_code != 204:
            return _error("Cancellation outcome is unconfirmed", uncertain=True)
        # Only later broker-verified reconciliation may release reserved risk.
        return {
            "result": {"cancel_requested_order_id": order_id},
            "status": "cancel_uncertain",
            "uncertain": True,
            "safe_mode": True,
            "safe_trading_version": "v2",
            "paper_trade": True,
            "risk_checks": ["ownership_proof", "durable_ownership"],
        }
