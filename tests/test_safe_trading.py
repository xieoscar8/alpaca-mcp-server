"""Pure-local mock tests for Safe Trading V1."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from alpaca_mcp_server.safe_overrides import register_safe_trading_tools
from alpaca_mcp_server.server import _safe_mode_enabled, build_server

ORDER_ID = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture(autouse=True)
def paper_mode(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")


class CaptureServer:
    def __init__(self):
        self.tools = {}

    def tool(self, **_kwargs):
        def decorator(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorator


class FakeClient:
    def __init__(self, responses=None, timeout_method=None):
        self.responses = list(responses or [])
        self.timeout_method = timeout_method
        self.calls = []

    async def _call(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == self.timeout_method:
            raise httpx.ReadTimeout("timeout")
        status, payload = self.responses.pop(0) if self.responses else (200, {})
        return httpx.Response(status, content=json.dumps(payload).encode())

    async def get(self, path, **kwargs): return await self._call("GET", path, **kwargs)
    async def post(self, path, **kwargs): return await self._call("POST", path, **kwargs)
    async def delete(self, path, **kwargs): return await self._call("DELETE", path, **kwargs)


def make_tools(client=None):
    server = CaptureServer()
    client = client or FakeClient()
    register_safe_trading_tools(server, client)
    return server.tools, client


def test_safe_mode_defaults_on_and_only_explicit_false_disables():
    with patch.dict("os.environ", {}, clear=True): assert _safe_mode_enabled()
    with patch.dict("os.environ", {"ALPACA_SAFE_MODE": "0"}, clear=True): assert _safe_mode_enabled()
    with patch.dict("os.environ", {"ALPACA_SAFE_MODE": "false"}, clear=True): assert not _safe_mode_enabled()


@pytest.mark.asyncio
async def test_safe_registry_hides_dangerous_writes_and_keeps_reads(monkeypatch):
    monkeypatch.delenv("ALPACA_SAFE_MODE", raising=False)
    names = {t.name for t in await build_server().list_tools()}
    assert {"safe_place_stock_order", "safe_place_crypto_order", "safe_cancel_order", "safe_close_position"} <= names
    assert {"get_account_info", "get_account_config", "get_orders", "get_open_position", "get_clock", "get_stock_bars"} <= names
    assert not ({"cancel_all_orders", "close_all_positions", "replace_order_by_id", "update_account_config",
                 "place_stock_order", "place_crypto_order", "place_option_order", "exercise_options_position",
                 "do_not_exercise_options_position", "create_watchlist", "create_locate"} & names)


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"side": "sell", "qty": "1", "limit_price": "10"},
    {"side": "buy", "qty": "1", "limit_price": "10", "type": "market"},
    {"side": "buy", "qty": "11", "limit_price": "10"},
    {"side": "buy", "qty": "NaN", "limit_price": "10"},
    {"side": "buy", "qty": "Infinity", "limit_price": "10"},
    {"side": "buy", "qty": "wat", "limit_price": "10"},
    {"side": "SELL", "qty": "1", "limit_price": "10"},
    {"side": " sell ", "qty": "1", "limit_price": "10"},
    {"side": "buy", "qty": "1", "limit_price": "10", "type": "MARKET"},
    {"side": "buy", "qty": "1", "limit_price": "10", "type": "stop"},
    {"side": "buy", "qty": "1", "limit_price": "10", "type": "stop_limit"},
    {"side": "buy", "qty": "1", "limit_price": "10", "type": "trailing_stop"},
    {"side": "buy", "qty": "1", "notional": "10", "limit_price": "10"},
    {"side": "buy", "limit_price": "10"},
    {"side": "buy", "qty": "0", "limit_price": "10"},
    {"side": "buy", "qty": "-1", "limit_price": "10"},
    {"side": "buy", "qty": "1", "limit_price": "0"},
    {"side": "buy", "qty": "1", "limit_price": "-1"},
    {"side": "buy", "qty": "1", "limit_price": "NaN"},
    {"side": "buy", "qty": " 1 ", "limit_price": "10"},
    {"side": "buy", "qty": "1", "limit_price": " 10 "},
    {"side": "buy", "qty": "1", "limit_price": "100.000001"},
    {"side": "buy", "qty": "1e100000", "limit_price": "1"},
])
async def test_stock_rejections_make_no_request(kwargs):
    tools, client = make_tools()
    result = await tools["safe_place_stock_order"]("AAPL", **kwargs)
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("time_in_force", ["ioc", "fok", "opg", "cls", "DAY", "GTC", " day ", "unknown"])
async def test_stock_time_in_force_rejections_never_post(time_in_force):
    tools, client = make_tools()
    result = await tools["safe_place_stock_order"](
        "AAPL", "buy", qty="1", limit_price="10", time_in_force=time_in_force
    )
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("time_in_force", ["day", "gtc"])
async def test_stock_allowed_time_in_force_posts(time_in_force):
    tools, client = make_tools(FakeClient([(200, {})]))
    result = await tools["safe_place_stock_order"](
        "AAPL", "buy", qty="1", limit_price="10", time_in_force=time_in_force
    )
    assert "error" not in result
    assert len(client.calls) == 1
    assert client.calls[0][2]["json"]["time_in_force"] == time_in_force


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", " ", " AAPL", "AAPL ", "BRK/B", "*", "AAPL,MSFT", "../AAPL", "AAPL\n"])
async def test_stock_symbol_rejections_never_post(symbol):
    tools, client = make_tools()
    result = await tools["safe_place_stock_order"](
        symbol, "buy", qty="1", limit_price="10"
    )
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"side": "sell", "notional": "10", "limit_price": "10"},
    {"side": "buy", "notional": "9.99", "limit_price": "10"},
    {"side": "buy", "notional": "100.01", "limit_price": "10"},
    {"side": "buy", "notional": "10", "limit_price": "10", "type": "market"},
    {"side": "buy", "notional": "10", "limit_price": "10", "type": "stop_limit"},
    {"side": "buy", "notional": "10", "limit_price": "10", "time_in_force": "day"},
    {"side": "buy", "notional": "10", "limit_price": "10", "time_in_force": "fok"},
    {"side": "buy", "qty": "1", "notional": "10", "limit_price": "10"},
    {"side": "buy", "limit_price": "10"},
    {"side": "buy", "notional": "NaN", "limit_price": "10"},
    {"side": "buy", "notional": "Infinity", "limit_price": "10"},
    {"side": "buy", "notional": "0", "limit_price": "10"},
    {"side": "buy", "notional": "-1", "limit_price": "10"},
    {"side": "buy", "notional": "bad", "limit_price": "10"},
])
async def test_crypto_rejections_make_no_request(kwargs):
    tools, client = make_tools()
    result = await tools["safe_place_crypto_order"]("BTC/USD", **kwargs)
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
async def test_standard_crypto_pair_is_allowed():
    tools, client = make_tools(FakeClient([(200, {})]))
    result = await tools["safe_place_crypto_order"](
        "BTC/USD", "buy", notional="10", limit_price="10"
    )
    assert "error" not in result and len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", " ", " BTC/USD", "BTC/USD ", "*", "BTC/USD,ETH/USD", "../BTC/USD", "BTC/USD/extra", "BTC//USD", "BTC\n/USD"])
async def test_crypto_symbol_rejections_never_post(symbol):
    tools, client = make_tools()
    result = await tools["safe_place_crypto_order"](
        symbol, "buy", notional="10", limit_price="10"
    )
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
async def test_generated_client_order_id_and_invalid_env_fails_safe(monkeypatch):
    monkeypatch.setenv("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "abc")
    tools, client = make_tools(FakeClient([(200, {"id": "ok"})]))
    ok = await tools["safe_place_stock_order"]("AAPL", "buy", qty="1", limit_price="100")
    assert client.calls[0][2]["json"]["client_order_id"].startswith("safe-")
    assert ok["max_order_notional"] == "100"
    rejected = await tools["safe_place_stock_order"]("AAPL", "buy", qty="1.01", limit_price="100")
    assert "error" in rejected and len(client.calls) == 1


@pytest.mark.asyncio
async def test_order_boundaries_ids_and_unsupported_fields():
    tools, client = make_tools(FakeClient([(200, {}), (200, {}), (200, {}), (200, {})]))
    first = await tools["safe_place_stock_order"]("AAPL", "buy", qty="1", limit_price="100")
    second = await tools["safe_place_stock_order"]("AAPL", "buy", notional="100", limit_price="1")
    await tools["safe_place_crypto_order"]("BTC/USD", "buy", notional="10", limit_price="1")
    await tools["safe_place_crypto_order"]("BTC/USD", "buy", notional="100", limit_price="1")
    ids = [call[2]["json"]["client_order_id"] for call in client.calls]
    assert len(set(ids)) == 4 and all(len(value) <= 128 for value in ids)
    assert first["estimated_notional"] == second["estimated_notional"] == "100"
    assert "order_class" not in tools["safe_place_stock_order"].__annotations__
    assert "advanced_instructions" not in tools["safe_place_stock_order"].__annotations__


@pytest.mark.asyncio
@pytest.mark.parametrize("client_id", ["", "   ", "x" * 129, " leading", "trailing ", "bad\nline"])
async def test_invalid_client_order_ids_never_post(client_id):
    tools, client = make_tools()
    result = await tools["safe_place_stock_order"](
        "AAPL", "buy", qty="1", limit_price="10", client_order_id=client_id
    )
    assert "error" in result and client.calls == []


@pytest.mark.asyncio
async def test_caller_client_order_id_passed_exactly():
    tools, client = make_tools(FakeClient([(200, {})]))
    await tools["safe_place_stock_order"](
        "AAPL", "buy", qty="1", limit_price="10", client_order_id="caller-id_1"
    )
    assert client.calls[0][2]["json"]["client_order_id"] == "caller-id_1"


@pytest.mark.asyncio
async def test_cancel_prechecks_then_cancels_exactly_one():
    tools, client = make_tools(FakeClient([(200, {"id": ORDER_ID}), (204, {})]))
    await tools["safe_cancel_order"](ORDER_ID)
    path = f"/v2/orders/{ORDER_ID}"
    assert [(m, p) for m, p, _ in client.calls] == [("GET", path), ("DELETE", path)]


@pytest.mark.asyncio
@pytest.mark.parametrize("order_id", ["", " ", "*", "all", "a,b", "a/b", "%2F", ".."])
async def test_cancel_identifier_attacks_make_no_request(order_id):
    tools, client = make_tools()
    assert "error" in await tools["safe_cancel_order"](order_id)
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload", [(404, {}), (401, {}), (403, {}), (200, {}), (200, {"id": "different"})])
async def test_cancel_failed_prechecks_never_delete(status, payload):
    tools, client = make_tools(FakeClient([(status, payload)]))
    assert "error" in await tools["safe_cancel_order"](ORDER_ID)
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
async def test_close_cap_and_single_endpoint():
    tools, client = make_tools(FakeClient([(200, {"market_value": "25.01"})]))
    assert "error" in await tools["safe_close_position"]("AAPL")
    assert [c[0] for c in client.calls] == ["GET"]
    tools, client = make_tools(FakeClient([(200, {"market_value": "-25"}), (200, {"id": "close"})]))
    await tools["safe_close_position"]("AAPL")
    assert [(m, p) for m, p, _ in client.calls] == [("GET", "/v2/positions/AAPL"), ("DELETE", "/v2/positions/AAPL")]


@pytest.mark.asyncio
@pytest.mark.parametrize("identifier", ["", " ", "*", "all", "ALL", "AAPL,MSFT", "../orders", "%2F"])
async def test_close_identifier_attacks_make_no_request(identifier):
    tools, client = make_tools()
    assert "error" in await tools["safe_close_position"](identifier)
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "bad", "NaN", "Infinity", "25.000001", "-25.000001"])
async def test_close_invalid_or_over_cap_values_never_delete(value):
    payload = {} if value is None else {"market_value": value}
    tools, client = make_tools(FakeClient([(200, payload)]))
    assert "error" in await tools["safe_close_position"]("AAPL")
    assert [c[0] for c in client.calls] == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["25", "-25"])
async def test_close_absolute_boundary_is_allowed(value):
    tools, client = make_tools(FakeClient([(200, {"market_value": value}), (200, {})]))
    await tools["safe_close_position"]("BTC/USD")
    path = "/v2/positions/BTC%2FUSD"
    assert [(m, p) for m, p, _ in client.calls] == [("GET", path), ("DELETE", path)]


@pytest.mark.asyncio
async def test_live_gate_rejects_all_writes(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "false")
    tools, client = make_tools()
    calls = [
        tools["safe_place_stock_order"]("AAPL", "buy", qty="1", limit_price="10"),
        tools["safe_place_crypto_order"]("BTC/USD", "buy", notional="10", limit_price="10"),
        tools["safe_cancel_order"](ORDER_ID), tools["safe_close_position"]("AAPL"),
    ]
    results = [await call for call in calls]
    assert all(r["error"]["message"] == "Safe Trading V1 write operations are Paper-only." for r in results)
    assert client.calls == []


@pytest.mark.asyncio
async def test_timeouts_never_retry_writes():
    tools, client = make_tools(FakeClient(timeout_method="POST"))
    result = await tools["safe_place_stock_order"]("AAPL", "buy", qty="1", limit_price="10", client_order_id="known")
    assert result["uncertain"] and len(client.calls) == 1
    tools, client = make_tools(FakeClient([(200, {"market_value": "10"})], timeout_method="DELETE"))
    result = await tools["safe_close_position"]("AAPL")
    assert result["uncertain"] and [c[0] for c in client.calls] == ["GET", "DELETE"]


@pytest.mark.asyncio
async def test_cancel_delete_timeout_is_uncertain_and_not_retried():
    tools, client = make_tools(FakeClient([(200, {"id": ORDER_ID})], timeout_method="DELETE"))
    result = await tools["safe_cancel_order"](ORDER_ID)
    assert result["uncertain"] and [c[0] for c in client.calls] == ["GET", "DELETE"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,args", [
    ("safe_cancel_order", (ORDER_ID,)), ("safe_close_position", ("AAPL",)),
])
async def test_get_timeout_never_reaches_delete(tool_name, args):
    tools, client = make_tools(FakeClient(timeout_method="GET"))
    result = await tools[tool_name](*args)
    assert "error" in result and [c[0] for c in client.calls] == ["GET"]


@pytest.mark.parametrize("raw,expected", [
    (None, True), ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("false", False),
])
def test_safe_mode_environment_matrix(monkeypatch, raw, expected):
    if raw is None: monkeypatch.delenv("ALPACA_SAFE_MODE", raising=False)
    else: monkeypatch.setenv("ALPACA_SAFE_MODE", raw)
    assert _safe_mode_enabled() is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,allowed", [
    (None, False), ("true", True), ("TRUE", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("no", False), ("unexpected", False),
])
async def test_paper_environment_matrix(monkeypatch, raw, allowed):
    if raw is None: monkeypatch.delenv("ALPACA_PAPER_TRADE", raising=False)
    else: monkeypatch.setenv("ALPACA_PAPER_TRADE", raw)
    tools, client = make_tools(FakeClient([(200, {})]))
    result = await tools["safe_place_stock_order"]("AAPL", "buy", qty="1", limit_price="10")
    assert ("error" not in result) is allowed
    assert len(client.calls) == (1 if allowed else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,value", [
    ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", ""), ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "abc"),
    ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "NaN"), ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "Infinity"),
    ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "0"), ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "-1"),
    ("ALPACA_SAFE_MAX_ORDER_NOTIONAL", "1e999999"),
    ("ALPACA_SAFE_MAX_CLOSE_MARKET_VALUE", "1e999999"),
    ("ALPACA_SAFE_MIN_CRYPTO_NOTIONAL", "0.00001"),
])
async def test_environment_limits_never_expand_permissions(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    tools, client = make_tools(FakeClient([(200, {"market_value": "25.01"})]))
    if name == "ALPACA_SAFE_MAX_CLOSE_MARKET_VALUE":
        result = await tools["safe_close_position"]("AAPL")
    elif name == "ALPACA_SAFE_MIN_CRYPTO_NOTIONAL":
        result = await tools["safe_place_crypto_order"]("BTC/USD", "buy", notional="9.99", limit_price="1")
    else:
        result = await tools["safe_place_stock_order"]("AAPL", "buy", notional="100.01", limit_price="1")
    assert "error" in result and all(call[0] != "POST" and call[0] != "DELETE" for call in client.calls)
