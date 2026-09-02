"""Blocker regressions: all broker traffic is in-memory."""

from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from alpaca_mcp_server.paper import PAPER_URL, paper_enabled
from alpaca_mcp_server.reconciliation import reconcile_pending
from alpaca_mcp_server.safe_overrides import _proof
from alpaca_mcp_server.server import build_server, _get_trading_base_url, _load_spec
from alpaca_mcp_server.tool_registry import TOOL_NAMES
from test_safe_trading import FakeClient, FakeStore, make_tools, owned_order, place


@pytest.fixture(autouse=True)
def authorized(monkeypatch):
    monkeypatch.setattr("alpaca_mcp_server.authentication.get_access_token",
                        lambda: SimpleNamespace(claims={"permissions": ["paper-trading"]}))


@pytest.mark.parametrize("value", ["true", "true ", " true", "TRUE", "false", "1", "yes", "", None])
async def test_paper_parser_and_actual_transport_host(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ALPACA_PAPER_TRADE", raising=False)
    else:
        monkeypatch.setenv("ALPACA_PAPER_TRADE", value)
    calls = []

    def broker(request):
        calls.append(request)
        assert request.url.host == "paper-api.alpaca.markets"
        return httpx.Response(503)

    async with httpx.AsyncClient(base_url=PAPER_URL, transport=httpx.MockTransport(broker)) as client:
        tools, _, store = make_tools(client=client)
        result = await tools["safe_place_crypto_order"](
            "BTC/USD", "buy", "s", "k", notional="20", limit_price="10")
    assert paper_enabled() is (value == "true")
    if value == "true":
        assert _get_trading_base_url() == PAPER_URL
        assert len(calls) == 1 and result["uncertain"]
    else:
        with pytest.raises(ValueError):
            _get_trading_base_url()
        assert "error" in result and not calls and not store.rows


@pytest.mark.parametrize("value", [None, "", "false", "true ", "TRUE", "1", "yes"])
async def test_invalid_hosted_paper_configuration_rejects_startup(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ALPACA_PAPER_TRADE", raising=False)
    else:
        monkeypatch.setenv("ALPACA_PAPER_TRADE", value)
    monkeypatch.setenv("ALPACA_SAFE_OWNERSHIP_SECRET", "synthetic-ownership-secret-for-tests-12345")
    with pytest.raises(ValueError, match="must be exactly true"):
        build_server(hosted_mode=True, risk_store=FakeStore(), auth_provider=StaticTokenVerifier({}))


@pytest.mark.parametrize("target,redirects", [
    ("https://api.alpaca.markets", False), (PAPER_URL + "/wrong", False),
    ("http://paper-api.alpaca.markets", False), (PAPER_URL, True),
])
async def test_unproven_transport_rejected(target, redirects):
    client = FakeClient()
    client.base_url = httpx.URL(target)
    client.follow_redirects = redirects
    tools, _, store = make_tools(client=client)
    assert "error" in await place(tools)
    assert not client.calls and not client.asset_calls and not store.rows


@pytest.mark.parametrize("asset", [None, [], {},
    {"class": "us_option"}, {"class": "crypto"},
    {"class": "us_equity", "asset_class": "us_option"},
    {"class": "us_equity", "symbol": "OTHER"},
    {"class": "us_equity", "tradable": False},
])
async def test_asset_rejection_precedes_reservation(asset):
    client = FakeClient()

    async def lookup(*args, **kwargs):
        payload = asset
        if isinstance(asset, dict):
            payload = {"symbol": "AAPL", "status": "active", "tradable": True, **asset}
        return httpx.Response(200, json=payload)

    client.get = lookup
    tools, _, store = make_tools(client=client)
    assert "error" in await place(tools)
    assert not client.calls and not store.rows


@pytest.mark.parametrize("failure", [302, 404, 408, 500, "timeout", "malformed"])
async def test_asset_failure_no_reservation(failure):
    client = FakeClient()

    async def lookup(*args, **kwargs):
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic")
        if failure == "malformed":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(failure)

    client.get = lookup
    tools, _, store = make_tools(client=client)
    assert "error" in await place(tools)
    assert not client.calls and not store.rows


async def test_equity_accepted_occ_rejected():
    tools, client, store = make_tools()
    assert "error" not in await place(tools)
    assert len(store.rows) == 1 and len(client.asset_calls) == 1
    before = len(client.calls)
    assert "error" in await place(tools, symbol="AAPL261218C00200000", key="option")
    assert len(client.calls) == before and len(store.rows) == 1


@pytest.mark.parametrize("bad", ["1e-1000000000", "1e999999999", "0." + "0" * 10000,
    "NaN", "Infinity", "-Infinity", "1.1234567890123", "１", [], {}, True, 1, 1.5])
async def test_bounded_decimals_no_side_effects(bad):
    tools, client, store = make_tools()
    assert "error" in await place(tools, notional=bad)
    assert not client.calls and not client.asset_calls and not store.rows


@pytest.mark.parametrize("field,bad", [("symbol", "A" * 33), ("symbol", "АAPL"),
    ("strategy", "s" * 65), ("key", "k" * 65), ("strategy", "策略")])
async def test_identifier_bounds(field, bad):
    tools, client, store = make_tools()
    assert "error" in await place(tools, **{field: bad})
    assert not client.calls and not store.rows


async def test_decimal_intent_canonical_and_exact():
    tools, client, store = make_tools()
    assert "error" not in await place(tools, notional="5.000e1")
    assert client.calls[0][2]["json"]["notional"] == "50"
    assert next(iter(store.rows.values())).requested_notional == Decimal(50)
    assert (await place(tools, notional="50"))["idempotent_replay"]


@pytest.mark.parametrize("response", ["pending_cancel", "unknown", "malformed", "timeout", 302, 404, 408, 500])
async def test_cancel_ack_restart_and_ambiguous_reconciliation(response):
    tools, client, store, row = await owned_order()
    result = await tools["safe_cancel_order"](row.order_id, row.strategy_id)
    assert result["status"] == "cancel_uncertain"
    restarted = FakeStore(store.rows)
    client.get_payload["status"] = response
    if response == "malformed":
        client.get_payload = {}
    elif response == "timeout":
        client.get_timeout = True
    elif isinstance(response, int):
        client.get_status = response
    await reconcile_pending(client, restarted, ownership_proof=lambda op: _proof(op, "test-secret-not-production"))
    current = restarted.rows[row.client_order_id]
    assert current.status == "cancel_uncertain" and current.reserved_notional == Decimal(50)
    assert "error" in await tools["safe_cancel_order"](row.order_id, row.strategy_id)
    assert sum(c[0] == "DELETE" for c in client.calls) == 1


@pytest.mark.parametrize("status", [200, 201, 301, 302, 307])
async def test_unexpected_delete_response_keeps_risk(status):
    tools, client, store, row = await owned_order()
    client.delete_status = status
    assert (await tools["safe_cancel_order"](row.order_id, row.strategy_id))["uncertain"]
    assert store.rows[row.client_order_id].reserved_notional == Decimal(50)


@pytest.mark.parametrize("uncertain", [False, True])
async def test_cancel_symbol_mismatch_even_when_uncertain(uncertain):
    from dataclasses import replace
    tools, client, store, row = await owned_order()
    if uncertain:
        store.rows[row.client_order_id] = replace(row, status="uncertain", order_id=None, uncertain=True)
    before = store.rows.copy()
    client.get_payload["symbol"] = "MSFT"
    assert "error" in await tools["safe_cancel_order"](row.order_id, row.strategy_id)
    assert store.rows == before and all(c[0] != "DELETE" for c in client.calls)


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("safe", ["true", "false"])
@pytest.mark.parametrize("toolsets", ["trading", "trading,stock-data,watchlists", ""])
async def test_registration_matrix(monkeypatch, hosted, safe, toolsets):
    monkeypatch.setenv("ALPACA_SAFE_MODE", safe)
    monkeypatch.setenv("ALPACA_TOOLSETS", toolsets)
    monkeypatch.setenv("ALPACA_SAFE_OWNERSHIP_SECRET", "synthetic-ownership-secret-for-tests-12345")
    server = build_server(hosted_mode=hosted, risk_store=FakeStore(),
                          auth_provider=StaticTokenVerifier({}))
    tools = await server.list_tools()
    names = {t.name for t in tools}
    raw_writes = {
        TOOL_NAMES.get(operation["operationId"], operation["operationId"])
        for path in _load_spec("trading-api")["paths"].values()
        for method, operation in path.items()
        if method.lower() in {"post", "put", "patch", "delete"}
        and isinstance(operation, dict) and "operationId" in operation
    }
    assert not names & raw_writes
    assert {t.name for t in tools if t.annotations and t.annotations.readOnlyHint is False} == {
        "safe_place_stock_order", "safe_place_crypto_order", "safe_cancel_order"}
    assert not names & {"safe_close_position", "place_stock_order", "place_option_order",
        "place_crypto_order", "cancel_all_orders", "close_all_positions", "close_position"}
