"""Per-request verified permissions, with no real broker or database traffic."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from alpaca_mcp_server import authentication as auth
from alpaca_mcp_server.readme_docs import README_DOC_TOOL_NAMES, register_readme_docs_tools
from alpaca_mcp_server.safe_overrides import register_safe_trading_tools
from test_authkit_resource_server import signed  # noqa: F401
from test_safe_trading import CaptureServer

MISSING = object()
INVALID = [MISSING, None, [], "paper-trading", {}, [1], [True], [None], [[]],
           ["paper-trading-admin"], ["Paper-Trading"], [" paper-trading"],
           ["paper-trading "], ["paper-trading", 1], ["paper-trading", " "],
           ["paper-trading\n"], ["paper-trader"]]
CALLS = [
    ("safe_place_stock_order", dict(symbol="AAPL", side="buy", strategy_id="s",
                                   idempotency_key="k", notional="50", limit_price="10")),
    ("safe_place_crypto_order", dict(symbol="BTC/USD", side="buy", strategy_id="s",
                                    idempotency_key="k", notional="50", limit_price="10")),
    ("safe_cancel_order", dict(order_id="123e4567-e89b-42d3-a456-426614174000",
                               strategy_id="s")),
]


def register(server):
    broker, store, principal = Mock(), Mock(), Mock(return_value="principal-a")
    register_safe_trading_tools(server, broker, store, principal_provider=principal,
                                ownership_secret="synthetic-test-only")
    return broker, store, principal


@pytest.mark.parametrize("permissions", INVALID)
@pytest.mark.parametrize("name,args", CALLS)
async def test_denied_before_any_broker_ledger_or_principal_access(monkeypatch, permissions, name, args):
    claims = {"role": "paper-trader"}
    if permissions is not MISSING:
        claims["permissions"] = permissions
    monkeypatch.setattr(auth, "get_access_token", lambda: SimpleNamespace(claims=claims))
    server = CaptureServer()
    broker, store, principal = register(server)
    result = await server.tools[name](**args)
    assert "requires paper-trading permission" in result["error"]["message"]
    assert not broker.mock_calls and not store.mock_calls and not principal.mock_calls


@pytest.mark.parametrize("failure", [False, True])
def test_missing_or_failed_context_denies(monkeypatch, failure):
    getter = Mock(side_effect=RuntimeError("unavailable")) if failure else Mock(return_value=None)
    monkeypatch.setattr(auth, "get_access_token", getter)
    assert not auth.has_paper_trading_permission()


def test_real_signed_http_permissions_are_not_cached(signed, monkeypatch):  # noqa: F811
    provider, sign, _ = signed
    monkeypatch.setenv("ALPACA_PAPER_TRADE", "false")
    mcp = FastMCP("permissions", auth=provider)
    broker, store, principal = register(mcp)
    docs = AsyncMock()
    docs.__aenter__.return_value = docs
    docs.call_tool.return_value = SimpleNamespace(
        data={"result": "read-only-ok"}, is_error=False,
    )
    register_readme_docs_tools(mcp, client_factory=lambda: docs)

    app = mcp.http_app(path="/mcp", stateless_http=True, json_response=True)
    with TestClient(app) as client:
        def call(method, params, claims):
            return client.post("/mcp", headers={
                "Authorization": "Bearer " + sign(claims),
                "Accept": "application/json, text/event-stream",
            }, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})

        names = {t["name"] for t in call("tools/list", {}, {}).json()["result"]["tools"]}
        assert names == {name for name, _ in CALLS} | README_DOC_TOOL_NAMES
        assert "read-only-ok" in call("tools/call", {
            "name": "search_alpaca_docs", "arguments": {"query": "orders"},
        }, {}).text
        for claims, message in [
            ({"sub": "member", "role": "paper-trader"}, "requires paper-trading permission"),
            ({"sub": "trader", "permissions": ["paper-trading"]}, "Paper-only"),
            ({"sub": "member", "permissions": []}, "requires paper-trading permission"),
            ({"sub": "trader"}, "requires paper-trading permission"),
        ]:
            for name, args in CALLS:
                r = call("tools/call", {"name": name, "arguments": args}, claims)
                assert r.status_code == 200 and message in r.text
        for permissions in INVALID:
            claims = {} if permissions is MISSING else {"permissions": permissions}
            for name, args in CALLS:
                r = call("tools/call", {"name": name, "arguments": args}, claims)
                assert "requires paper-trading permission" in r.text
        # Caller-controlled arguments cannot grant permission.
        name, args = CALLS[0]
        call("tools/call", {"name": name, "arguments": {**args, "permissions": ["paper-trading"]}}, {})
        assert not broker.mock_calls and not store.mock_calls and not principal.mock_calls


async def test_bad_signature_permissions_never_authenticate(signed, monkeypatch):  # noqa: F811
    provider, sign, _ = signed
    monkeypatch.setattr(provider.token_verifier, "_get_verification_key",
                        AsyncMock(side_effect=ValueError("invalid signature key")))
    assert await provider.verify_token(sign({"permissions": ["paper-trading"]})) is None
