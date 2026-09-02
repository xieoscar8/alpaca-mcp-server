"""Pure-local Safe Trading V2 ownership, risk, and self-attack tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from alpaca_mcp_server.reconciliation import reconcile_pending
from alpaca_mcp_server.risk_store import (
    ACTIVE_STATUSES,
    Operation,
    Reservation,
    RiskLimitExceeded,
    RiskStoreError,
)
from alpaca_mcp_server.safe_overrides import _limits, _proof, register_safe_trading_tools
from alpaca_mcp_server.server import _load_spec, build_server
from alpaca_mcp_server.tool_registry import TOOL_NAMES

ORDER_1 = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture(autouse=True)
def safe_env(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
    monkeypatch.setenv("ALPACA_SAFE_PRINCIPAL", "principal-a")
    monkeypatch.setenv(
        "ALPACA_SAFE_OWNERSHIP_SECRET", "test-ownership-secret-not-production-123456"
    )


class CaptureServer:
    def __init__(self):
        self.tools = {}

    def tool(self, **_kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return decorator


class FakeClient:
    def __init__(self):
        self.calls = []
        self.post_timeout = False
        self.delete_timeout = False
        self.delete_error = None
        self.get_timeout = False
        self.post_status = 200
        self.delete_status = 204
        self.get_status = 200
        self.get_payload = {}

    async def post(self, path, **kwargs):
        self.calls.append(("POST", path, kwargs))
        if self.post_timeout:
            raise httpx.ReadTimeout("timeout")
        body = kwargs["json"]
        payload = {"id": ORDER_1, "client_order_id": body["client_order_id"]}
        return httpx.Response(self.post_status, content=json.dumps(payload).encode())

    async def get(self, path, **kwargs):
        self.calls.append(("GET", path, kwargs))
        if self.get_timeout:
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(self.get_status, content=json.dumps(self.get_payload).encode())

    async def delete(self, path, **kwargs):
        self.calls.append(("DELETE", path, kwargs))
        if self.delete_timeout:
            raise httpx.ReadTimeout("timeout")
        if self.delete_error is not None:
            raise self.delete_error("ambiguous delete")
        return httpx.Response(self.delete_status, content=b"")


class FakeStore:
    def __init__(self, rows=None, lock=None):
        self.rows = rows if rows is not None else {}
        self.lock = lock or asyncio.Lock()
        self.unavailable = False
        self.transaction_failure = False

    async def open(self):
        pass

    async def close(self):
        pass

    async def reserve(self, **kwargs):
        if self.unavailable:
            raise RiskStoreError("unavailable")
        async with self.lock:
            if self.transaction_failure:
                raise RiskStoreError("transaction")
            identity = (kwargs["principal"], kwargs["strategy_id"], kwargs["idempotency_key"])
            for row in self.rows.values():
                if (row.principal, row.strategy_id, row.idempotency_key) == identity:
                    if (
                        row.client_order_id != kwargs["client_order_id"]
                        or row.symbol != kwargs["symbol"]
                        or row.request_fingerprint != kwargs["request_fingerprint"]
                        or row.requested_notional != kwargs["requested_notional"]
                    ):
                        raise RiskStoreError("idempotency mismatch")
                    return Reservation(row, False)
            active = [r for r in self.rows.values() if r.status in ACTIVE_STATUSES]
            limits = kwargs["limits"]
            amount = kwargs["requested_notional"]
            if len(active) + 1 > limits.max_open_orders:
                raise RiskLimitExceeded("open orders")
            if (
                sum((r.reserved_notional for r in active), Decimal(0)) + amount
                > limits.max_open_notional
            ):
                raise RiskLimitExceeded("open notional")
            symbol_total = sum(
                (r.reserved_notional for r in active if r.symbol == kwargs["symbol"]), Decimal(0)
            )
            if symbol_total + amount > limits.max_symbol_open_notional:
                raise RiskLimitExceeded("symbol notional")
            daily = sum(
                (r.requested_notional for r in self.rows.values() if r.status != "rejected"),
                Decimal(0),
            )
            if daily + amount > limits.max_daily_submitted_notional:
                raise RiskLimitExceeded("daily notional")
            row = Operation(
                kwargs["principal"],
                kwargs["strategy_id"],
                kwargs["idempotency_key"],
                kwargs["client_order_id"],
                None,
                kwargs["symbol"],
                kwargs["asset_type"],
                kwargs["request_fingerprint"],
                amount,
                amount,
                "reserved",
                False,
                0,
            )
            self.rows[row.client_order_id] = row
            return Reservation(row, True)

    async def _set(self, key, **changes):
        if self.unavailable:
            raise RiskStoreError("unavailable")
        row = self.rows.get(key)
        if row is None:
            raise RiskStoreError("missing")
        row = replace(row, state_version=row.state_version + 1, **changes)
        self.rows[key] = row
        return row

    async def mark_submitted(self, key, order_id, submitted_at=None):
        return await self._set(key, order_id=order_id, status="submitted")

    async def mark_uncertain(self, key):
        await self._set(key, status="uncertain", uncertain=True)

    async def mark_rejected(self, key):
        await self._set(key, status="rejected", reserved_notional=Decimal(0))

    async def get_by_client_order_id(self, key):
        if self.unavailable:
            raise RiskStoreError("unavailable")
        return self.rows.get(key)

    async def list_reconcilable(self, principal=None):
        if self.unavailable:
            raise RiskStoreError("unavailable")
        return [
            row
            for row in self.rows.values()
            if row.status in ACTIVE_STATUSES and (principal is None or row.principal == principal)
        ]

    async def reconcile_uncertain_order(self, key, order_id):
        row = self.rows.get(key)
        if row is None or row.status != "uncertain" or row.order_id is not None:
            raise RiskStoreError("not reconcilable")
        return await self._set(key, order_id=order_id, status="submitted", uncertain=False)

    async def reconcile_verified(self, operation, *, order_id, broker_status, target_status):
        async with self.lock:
            current = self.rows.get(operation.client_order_id)
            if (
                current is None
                or current.status != operation.status
                or current.state_version != operation.state_version
                or current.principal != operation.principal
                or current.strategy_id != operation.strategy_id
                or current.symbol != operation.symbol
                or (current.order_id is not None and current.order_id != order_id)
            ):
                raise RiskStoreError("mismatch")
            terminal = target_status in {"filled", "expired", "cancelled", "rejected"}
            return await self._set(
                operation.client_order_id,
                order_id=order_id,
                broker_status=broker_status,
                status=target_status,
                uncertain=False,
                reserved_notional=Decimal(0) if terminal else current.reserved_notional,
            )

    async def mark_cancelled(self, key):
        await self._set(key, status="cancelled", reserved_notional=Decimal(0))

    async def begin_cancel(self, key):
        row = self.rows.get(key)
        if row is None or row.status != "submitted":
            raise RiskStoreError("not cancellable")
        return await self._set(key, status="cancel_uncertain", uncertain=True)

    async def mark_cancel_rejected(self, key):
        row = self.rows.get(key)
        if row is None or row.status != "cancel_uncertain":
            raise RiskStoreError("not cancellation uncertain")
        return await self._set(key, status="submitted", uncertain=False)

    async def mark_cancel_uncertain(self, key):
        await self._set(key, status="cancel_uncertain", uncertain=True)


def make_tools(
    client=None, store=None, principal="principal-a", secret="test-secret-not-production"
):
    server = CaptureServer()
    client = client or FakeClient()
    store = store or FakeStore()
    register_safe_trading_tools(
        server,
        client,
        store,
        principal_provider=lambda: principal,
        ownership_secret=secret,
        reconcile_before_write=False,
    )
    return server.tools, client, store


async def place(tools, *, key="key-1", strategy="strategy-1", symbol="AAPL", notional="50"):
    return await tools["safe_place_stock_order"](
        symbol, "buy", strategy, key, notional=notional, limit_price="10"
    )


@pytest.mark.asyncio
async def test_only_three_v2_writes_are_exposed(monkeypatch):
    monkeypatch.delenv("ALPACA_SAFE_MODE", raising=False)
    names = {tool.name for tool in await build_server().list_tools()}
    expected = {"safe_place_stock_order", "safe_place_crypto_order", "safe_cancel_order"}
    write_names = {
        TOOL_NAMES.get(operation["operationId"], operation["operationId"])
        for path_item in _load_spec("trading-api")["paths"].values()
        for method, operation in path_item.items()
        if method.lower() in {"post", "put", "patch", "delete"}
        and isinstance(operation, dict)
        and "operationId" in operation
    }
    write_names.update(
        {
            "place_stock_order",
            "place_crypto_order",
            "place_option_order",
            "safe_place_stock_order",
            "safe_place_crypto_order",
            "safe_cancel_order",
            "safe_close_position",
        }
    )
    assert names & write_names == expected
    assert "safe_close_position" not in names
    assert not (
        {
            "place_stock_order",
            "place_crypto_order",
            "place_option_order",
            "cancel_all_orders",
            "close_position",
            "close_all_positions",
            "replace_order_by_id",
            "update_account_config",
        }
        & names
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": "sell", "notional": "10", "limit_price": "10"},
        {"side": "buy", "notional": "10", "limit_price": "10", "type": "market"},
        {"side": "buy", "notional": "10", "limit_price": "10", "time_in_force": "ioc"},
        {"side": "buy", "notional": "100.01", "limit_price": "10"},
        {"side": "buy", "notional": "NaN", "limit_price": "10"},
        {"side": "buy", "notional": "10", "limit_price": " 10"},
    ],
)
async def test_v1_stock_validation_regression(kwargs):
    tools, client, _ = make_tools()
    result = await tools["safe_place_stock_order"]("AAPL", kwargs.pop("side"), "s", "k", **kwargs)
    assert "error" in result and not client.calls


@pytest.mark.asyncio
async def test_same_idempotency_key_posts_once_and_replays():
    tools, client, store = make_tools()
    first = await place(tools)
    second = await place(tools)
    assert "error" not in first and second["idempotent_replay"]
    assert [c[0] for c in client.calls].count("POST") == 1 and len(store.rows) == 1


@pytest.mark.asyncio
async def test_idempotency_reuse_with_changed_inputs_fails_closed():
    tools, client, _ = make_tools()
    await place(tools)
    result = await place(tools, symbol="MSFT")
    assert "error" in result and [c[0] for c in client.calls].count("POST") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [" key", "key ", "key\n", "*", "a/b"])
async def test_idempotency_format_attacks_post_zero(key):
    tools, client, _ = make_tools()
    assert "error" in await place(tools, key=key)
    assert not client.calls


@pytest.mark.asyncio
async def test_restart_reconstructs_idempotency():
    shared = {}
    lock = asyncio.Lock()
    tools, client, _ = make_tools(store=FakeStore(shared, lock))
    await place(tools)
    restarted, client2, _ = make_tools(store=FakeStore(shared, lock))
    result = await place(restarted)
    assert result["idempotent_replay"] and not client2.calls and len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unavailable", "transaction_failure"])
async def test_store_failure_posts_zero(failure):
    store = FakeStore()
    setattr(store, failure, True)
    tools, client, _ = make_tools(store=store)
    assert "error" in await place(tools) and not client.calls


@pytest.mark.asyncio
async def test_open_order_and_open_notional_limits():
    tools, client, _ = make_tools()
    for i in range(5):
        await place(tools, key=f"k-{i}", symbol=f"S{i}", notional="60")
    rejected = await place(tools, key="k-6", symbol="S6", notional="1")
    assert "error" in rejected and [c[0] for c in client.calls].count("POST") == 5


@pytest.mark.asyncio
async def test_symbol_open_notional_limit():
    tools, client, _ = make_tools()
    await place(tools, key="a", notional="100")
    await place(tools, key="b", notional="100")
    assert "error" in await place(tools, key="c", notional="1")
    assert [c[0] for c in client.calls].count("POST") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("symbols", [("AAPL", "aapl", "AaPl"), ("BTC/USD", "btc/usd", "BtC/uSd")])
async def test_case_variants_share_one_symbol_bucket(symbols):
    tools, client, _ = make_tools()
    tool = tools["safe_place_stock_order"] if "/" not in symbols[0] else tools["safe_place_crypto_order"]
    await tool(symbols[0], "buy", "s", "a", notional="100", limit_price="10")
    await tool(symbols[1], "buy", "s", "b", notional="100", limit_price="10")
    rejected = await tool(symbols[2], "buy", "s", "c", notional="10", limit_price="10")
    assert "error" in rejected
    assert [call[0] for call in client.calls].count("POST") == 2
    assert all(call[2]["json"]["symbol"] == symbols[0].upper() for call in client.calls)


@pytest.mark.asyncio
async def test_daily_limit_survives_cancelled_rows(monkeypatch):
    monkeypatch.setenv("ALPACA_SAFE_MAX_DAILY_SUBMITTED_NOTIONAL", "100")
    tools, client, store = make_tools()
    await place(tools, notional="100")
    key = next(iter(store.rows))
    await store.mark_cancelled(key)
    assert "error" in await place(tools, key="next", symbol="MSFT", notional="1")
    assert [c[0] for c in client.calls].count("POST") == 1


@pytest.mark.asyncio
async def test_absolute_daily_500_ceiling():
    tools, client, store = make_tools()
    for i in range(5):
        await place(tools, key=f"daily-{i}", symbol=f"D{i}", notional="100")
        await store.mark_cancelled(next(reversed(store.rows)))
    assert "error" in await place(tools, key="daily-6", symbol="D6", notional="1")
    assert [c[0] for c in client.calls].count("POST") == 5


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_oversubscribe(monkeypatch):
    monkeypatch.setenv("ALPACA_SAFE_MAX_OPEN_NOTIONAL", "100")
    tools, client, _ = make_tools()
    results = await asyncio.gather(
        *(place(tools, key=f"k{i}", symbol=f"S{i}", notional="60") for i in range(10))
    )
    assert sum("error" not in r for r in results) == 1
    assert [c[0] for c in client.calls].count("POST") == 1


@pytest.mark.asyncio
async def test_post_timeout_one_call_and_uncertain_reservation():
    tools, client, store = make_tools()
    client.post_timeout = True
    result = await place(tools)
    row = next(iter(store.rows.values()))
    assert (
        result["uncertain"] and row.status == "uncertain" and row.reserved_notional == Decimal(50)
    )
    assert [c[0] for c in client.calls] == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
async def test_uncertain_post_responses_do_not_release_reservation(status):
    tools, client, store = make_tools()
    client.post_status = status
    result = await place(tools)
    row = next(iter(store.rows.values()))
    assert result["uncertain"] and result["http_status"] == status
    assert row.status == "uncertain" and row.reserved_notional == Decimal(50)
    assert [c[0] for c in client.calls] == ["POST"]


@pytest.mark.asyncio
async def test_definitive_client_rejection_releases_reservation():
    tools, client, store = make_tools()
    client.post_status = 422
    result = await place(tools)
    row = next(iter(store.rows.values()))
    assert "error" in result and not result.get("uncertain", False)
    assert row.status == "rejected" and row.reserved_notional == Decimal(0)


@pytest.mark.asyncio
async def test_restart_reconciles_uncertain_owned_order_for_cancel():
    shared = {}
    first, client, _ = make_tools(store=FakeStore(shared))
    client.post_timeout = True
    timeout = await place(first)
    client_id = timeout["client_order_id"]
    restarted, client2, store2 = make_tools(store=FakeStore(shared))
    client2.get_payload = {"id": ORDER_1, "client_order_id": client_id}
    result = await restarted["safe_cancel_order"](ORDER_1, "strategy-1")
    assert "error" not in result
    assert [c[0] for c in client2.calls] == ["GET", "DELETE"]
    assert store2.rows[client_id].status == "cancelled"


async def owned_order():
    tools, client, store = make_tools()
    await place(tools)
    row = next(iter(store.rows.values()))
    client.calls.clear()
    client.get_payload = {"id": ORDER_1, "client_order_id": row.client_order_id}
    return tools, client, store, row


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id", ["manual", "safe-legacy", "safe-v2-forged"])
async def test_manual_legacy_and_forged_orders_cannot_cancel(client_id):
    tools, client, _ = make_tools()
    client.get_payload = {"id": ORDER_1, "client_order_id": client_id}
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_missing_and_corrupt_ownership_cannot_cancel():
    tools, client, store, row = await owned_order()
    store.rows.clear()
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]
    store.rows[row.client_order_id] = replace(row, idempotency_key="corrupt")
    client.calls.clear()
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_unavailable_ownership_store_cannot_cancel():
    tools, client, store, _ = await owned_order()
    store.unavailable = True
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_cross_strategy_and_principal_cannot_cancel():
    tools, client, store, row = await owned_order()
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "other")
    other_tools, other_client, _ = make_tools(store=store, principal="principal-b")
    other_client.get_payload = {"id": ORDER_1, "client_order_id": row.client_order_id}
    assert "error" in await other_tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert all(c[0] != "DELETE" for c in client.calls + other_client.calls)


@pytest.mark.asyncio
async def test_order_id_mismatch_cannot_cancel():
    tools, client, store, row = await owned_order()
    store.rows[row.client_order_id] = replace(row, order_id="223e4567-e89b-42d3-a456-426614174000")
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_alpaca_response_id_mismatch_cannot_cancel():
    tools, client, _, row = await owned_order()
    client.get_payload = {
        "id": "223e4567-e89b-42d3-a456-426614174000",
        "client_order_id": row.client_order_id,
    }
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_valid_owned_cancel_deletes_once():
    tools, client, store, row = await owned_order()
    result = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert "error" not in result and [c[0] for c in client.calls] == ["GET", "DELETE"]
    assert store.rows[row.client_order_id].status == "cancelled"


@pytest.mark.asyncio
async def test_concurrent_owned_cancels_issue_one_delete():
    tools, client, store, row = await owned_order()
    results = await asyncio.gather(
        tools["safe_cancel_order"](ORDER_1, "strategy-1"),
        tools["safe_cancel_order"](ORDER_1, "strategy-1"),
    )
    assert sum("error" not in result for result in results) == 1
    assert [call[0] for call in client.calls].count("DELETE") == 1
    assert store.rows[row.client_order_id].status == "cancelled"


@pytest.mark.asyncio
async def test_definitive_cancel_rejection_restores_submitted_state():
    tools, client, store, row = await owned_order()
    client.delete_status = 422
    result = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert result["http_status"] == 422
    assert store.rows[row.client_order_id].status == "submitted"
    assert [call[0] for call in client.calls].count("DELETE") == 1


@pytest.mark.asyncio
async def test_delete_timeout_once_and_retains_uncertain():
    tools, client, store, row = await owned_order()
    client.delete_timeout = True
    result = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert result["uncertain"] and [c[0] for c in client.calls] == ["GET", "DELETE"]
    assert store.rows[row.client_order_id].status == "cancel_uncertain"
    client.calls.clear()
    client.delete_timeout = False
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type", [httpx.ReadTimeout, httpx.ConnectError, httpx.WriteError, httpx.ProtocolError]
)
async def test_ambiguous_delete_errors_are_never_replayed(error_type):
    tools, client, store, row = await owned_order()
    client.delete_error = error_type
    first = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    client.delete_error = None
    second = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert first["uncertain"] and "error" in second
    assert [call[0] for call in client.calls].count("DELETE") == 1
    assert store.rows[row.client_order_id].status == "cancel_uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
async def test_ambiguous_delete_statuses_are_never_replayed(status):
    tools, client, store, row = await owned_order()
    client.delete_status = status
    first = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    client.delete_status = 204
    second = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert first["uncertain"] and first["http_status"] == status and "error" in second
    assert [call[0] for call in client.calls].count("DELETE") == 1
    assert store.rows[row.client_order_id].status == "cancel_uncertain"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second",
    [
        ({"qty": "5", "limit_price": "10"}, {"qty": "10", "limit_price": "5"}),
        ({"notional": "50", "limit_price": "10"}, {"notional": "50", "limit_price": "5"}),
        (
            {"notional": "50", "limit_price": "10", "time_in_force": "day"},
            {"notional": "50", "limit_price": "10", "time_in_force": "gtc"},
        ),
        ({"qty": "5", "limit_price": "10"}, {"notional": "50", "limit_price": "10"}),
    ],
)
async def test_changed_order_intent_replay_is_rejected(first, second):
    tools, client, _ = make_tools()
    await tools["safe_place_stock_order"]("AAPL", "buy", "s", "same-key", **first)
    result = await tools["safe_place_stock_order"]("aapl", "buy", "s", "same-key", **second)
    assert "error" in result
    assert [call[0] for call in client.calls].count("POST") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("paper", [None, "false", "0", "no", "unexpected"])
async def test_paper_configuration_fails_closed(monkeypatch, paper):
    if paper is None:
        monkeypatch.delenv("ALPACA_PAPER_TRADE", raising=False)
    else:
        monkeypatch.setenv("ALPACA_PAPER_TRADE", paper)
    tools, client, _ = make_tools()
    assert "error" in await place(tools)
    assert "error" in await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert not client.calls


@pytest.mark.parametrize(
    "name,value,field,hard",
    [
        ("ALPACA_SAFE_MAX_OPEN_ORDERS", "999", "max_open_orders", 5),
        ("ALPACA_SAFE_MAX_OPEN_NOTIONAL", "999999", "max_open_notional", Decimal(300)),
        (
            "ALPACA_SAFE_MAX_DAILY_SUBMITTED_NOTIONAL",
            "Infinity",
            "max_daily_submitted_notional",
            Decimal(500),
        ),
        (
            "ALPACA_SAFE_MAX_SYMBOL_OPEN_NOTIONAL",
            "999999",
            "max_symbol_open_notional",
            Decimal(200),
        ),
    ],
)
def test_environment_cannot_expand_hard_limits(monkeypatch, name, value, field, hard):
    monkeypatch.setenv(name, value)
    assert getattr(_limits(), field) == hard


@pytest.mark.asyncio
async def test_hosted_mode_ignores_safe_mode_false(monkeypatch):
    monkeypatch.setenv("ALPACA_SAFE_MODE", "false")
    monkeypatch.setenv("ALPACA_MCP_JWT_SIGNING_KEY", "independent-jwt-secret-not-production-1234")
    verifier = __import__(
        "fastmcp.server.auth.providers.jwt", fromlist=["StaticTokenVerifier"]
    ).StaticTokenVerifier({})
    names = {
        tool.name
        for tool in await build_server(
            hosted_mode=True,
            risk_store=FakeStore(),
            auth_provider=verifier,
            principal_provider=lambda: "principal-a",
        ).list_tools()
    }
    assert "safe_place_stock_order" in names and "place_stock_order" not in names
    assert {name for name in names if name.startswith("safe_")} == {
        "safe_place_stock_order", "safe_place_crypto_order", "safe_cancel_order",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "broker_status,target",
    [
        ("new", "submitted"),
        ("filled", "filled"),
        ("expired", "expired"),
        ("canceled", "cancelled"),
        ("rejected", "rejected"),
    ],
)
async def test_verified_reconciliation_state_machine(broker_status, target):
    tools, client, store = make_tools()
    await place(tools)
    operation = next(iter(store.rows.values()))
    client.calls.clear()
    client.get_payload = {
        "id": ORDER_1,
        "client_order_id": operation.client_order_id,
        "symbol": operation.symbol,
        "status": broker_status,
    }
    summary = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda row: _proof(row, "test-secret-not-production"),
        principal="principal-a",
    )
    updated = store.rows[operation.client_order_id]
    assert summary == {"checked": 1, "updated": 1, "quarantined": 0}
    assert updated.status == target and updated.broker_status == broker_status
    assert updated.reserved_notional == (Decimal(50) if target == "submitted" else Decimal(0))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,status",
    [
        ({}, 200),
        ({"id": ORDER_1, "client_order_id": "forged", "symbol": "AAPL", "status": "filled"}, 200),
        ({"id": ORDER_1, "client_order_id": "unused", "symbol": "MSFT", "status": "filled"}, 200),
        ({"id": ORDER_1, "client_order_id": "unused", "symbol": "AAPL", "status": "unknown"}, 200),
        ({}, 404),
        ({}, 503),
    ],
)
async def test_ambiguous_reconciliation_never_releases_risk(payload, status):
    tools, client, store = make_tools()
    await place(tools)
    operation = next(iter(store.rows.values()))
    if payload.get("client_order_id") == "unused":
        payload["client_order_id"] = operation.client_order_id
    client.calls.clear()
    client.get_payload = payload
    client.get_status = status
    summary = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda row: _proof(row, "test-secret-not-production"),
        principal="principal-a",
    )
    assert summary["updated"] == 0 and summary["quarantined"] == 1
    assert store.rows[operation.client_order_id].reserved_notional == Decimal(50)


@pytest.mark.asyncio
async def test_forged_hmac_and_cross_principal_reconciliation_are_quarantined():
    tools, client, store = make_tools()
    await place(tools)
    key = next(iter(store.rows))
    store.rows[key] = replace(store.rows[key], idempotency_key="forged")
    operation = store.rows[key]
    client.get_payload = {
        "id": ORDER_1,
        "client_order_id": key,
        "symbol": operation.symbol,
        "status": "filled",
    }
    forged = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda row: _proof(row, "test-secret-not-production"),
        principal="principal-a",
    )
    cross = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda row: True,
        principal="principal-b",
    )
    assert forged["updated"] == cross["updated"] == 0
    assert store.rows[key].reserved_notional == Decimal(50)


@pytest.mark.asyncio
async def test_terminal_state_replay_never_posts_again():
    tools, client, store = make_tools()
    await place(tools)
    key = next(iter(store.rows))
    store.rows[key] = replace(
        store.rows[key], status="filled", reserved_notional=Decimal(0), broker_status="filled"
    )
    client.calls.clear()
    replay = await place(tools)
    assert replay["idempotent_replay"] and replay["status"] == "filled"
    assert not client.calls


@pytest.mark.asyncio
async def test_reconciliation_rejects_mismatched_stored_order_id():
    tools, client, store = make_tools()
    await place(tools)
    key = next(iter(store.rows))
    store.rows[key] = replace(store.rows[key], order_id=ORDER_1, status="submitted")
    client.calls.clear()
    client.get_payload = {
        "id": "223e4567-e89b-42d3-a456-426614174000",
        "client_order_id": key,
        "symbol": "AAPL",
        "status": "filled",
    }
    summary = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda row: _proof(row, "test-secret-not-production"),
        principal="principal-a",
    )
    assert summary["updated"] == 0
    assert store.rows[key].status == "submitted" and store.rows[key].reserved_notional == 50


@pytest.mark.asyncio
async def test_concurrent_reconciliation_and_reservation_never_exceeds_limit():
    tools, client, store = make_tools()
    for index in range(5):
        await place(tools, key=f"existing-{index}", symbol=f"S{index}", notional="10")
    operation = next(iter(store.rows.values()))
    client.calls.clear()
    client.get_payload = {
        "id": ORDER_1,
        "client_order_id": operation.client_order_id,
        "symbol": operation.symbol,
        "status": "filled",
    }
    await asyncio.gather(
        reconcile_pending(
            client,
            store,
            ownership_proof=lambda row: _proof(row, "test-secret-not-production"),
            principal="principal-a",
        ),
        place(tools, key="concurrent-new", symbol="NEW", notional="10"),
    )
    assert sum(row.status in ACTIVE_STATUSES for row in store.rows.values()) <= 5
    assert [call[0] for call in client.calls].count("POST") <= 1


@pytest.mark.asyncio
async def test_active_cancel_uncertain_reconciliation_never_enables_second_delete():
    tools, client, store, row = await owned_order()
    store.rows[row.client_order_id] = replace(row, status="cancel_uncertain", uncertain=True)
    client.get_payload = {
        "id": ORDER_1,
        "client_order_id": row.client_order_id,
        "symbol": row.symbol,
        "status": "pending_cancel",
    }
    summary = await reconcile_pending(
        client,
        store,
        ownership_proof=lambda value: _proof(value, "test-secret-not-production"),
        principal="principal-a",
    )
    assert summary["updated"] == 0 and summary["quarantined"] == 1
    client.calls.clear()
    result = await tools["safe_cancel_order"](ORDER_1, "strategy-1")
    assert "error" in result and [call[0] for call in client.calls] == ["GET"]
