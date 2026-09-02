"""Fixed-output diagnostics: synthetic claims only, no external requests."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastmcp.server.auth.auth import AccessToken
from starlette.testclient import TestClient

from alpaca_mcp_server import auth_diagnostics as diagnostic
from alpaca_mcp_server import authentication
from alpaca_mcp_server.security import DATA_KEY
from alpaca_mcp_server.server import build_server
from test_authkit_resource_server import signed  # noqa: F401

EMPTY = {
    "verified_context_present": False, "org_id_present": False,
    "role_present": False, "role_value": "absent", "permissions_present": False,
    "permissions_shape_valid": False, "paper_trading_present": False,
    "alpaca_role_present": False, "alpaca_role_value": "absent",
}
CANARY = "sensitive-synthetic-canary-never-echo"


def token(claims):
    return AccessToken.model_construct(token=CANARY, client_id=CANARY, scopes=[], claims=claims)


def assert_fixed(value):
    assert value.keys() == EMPTY.keys()
    assert value["role_value"] in {"member", "paper-trader", "other_or_invalid", "absent"}
    assert len(value) == 9
    assert value["alpaca_role_value"] in {"member", "paper-trader", "other_or_invalid", "absent"}
    assert all(type(v) is bool for k, v in value.items() if k not in {"role_value", "alpaca_role_value"})
    assert CANARY not in json.dumps(value)


@pytest.mark.parametrize("context", [None, object(), token(None), token([]), token("bad")])
def test_no_valid_context(monkeypatch, context):
    monkeypatch.setattr(diagnostic, "get_access_token", lambda: context)
    assert diagnostic.auth_claims_summary() == EMPTY


@pytest.mark.parametrize("claims,expected", [
    ({}, {}),
    ({"org_id": CANARY}, {"org_id_present": True}),
    ({"org_id": None}, {"org_id_present": True}),
    ({"role": "member"}, {"role_present": True, "role_value": "member"}),
    ({"role": "paper-trader"}, {"role_present": True, "role_value": "paper-trader"}),
    ({"role": CANARY}, {"role_present": True, "role_value": "other_or_invalid"}),
    ({"role": [CANARY]}, {"role_present": True, "role_value": "other_or_invalid"}),
    ({"role": None}, {"role_present": True, "role_value": "other_or_invalid"}),
    ({"role": "Paper-Trader"}, {"role_present": True, "role_value": "other_or_invalid"}),
    ({"role": " paper-trader"}, {"role_present": True, "role_value": "other_or_invalid"}),
])
def test_presence_and_role_allowlist(monkeypatch, claims, expected):
    claims = {"sub": CANARY, "email": CANARY, "sid": CANARY, "secret": CANARY, **claims}
    monkeypatch.setattr(diagnostic, "get_access_token", lambda: token(claims))
    result = diagnostic.auth_claims_summary()
    assert result == {**EMPTY, "verified_context_present": True, **expected}
    assert_fixed(result)


@pytest.mark.parametrize("permissions,valid,allowed", [
    (None, False, False), ([], False, False), ("paper-trading", False, False),
    ({"paper-trading": True}, False, False), ([False], False, False),
    ([1], False, False), ([None], False, False), ([[]], False, False),
    ([""], False, False), ([" paper-trading"], False, False),
    (["paper-trading "], False, False), (["paper-trading", 1], False, False),
    (["paper-trading", " "], False, False), ([CANARY], True, False),
    (["Paper-Trading"], True, False), (["paper-trading-admin"], True, False),
    (["paper-trading"], True, True), ([CANARY, "paper-trading"], True, True),
])
def test_permissions_match_production_contract(monkeypatch, permissions, valid, allowed):
    context = token({"permissions": permissions})
    monkeypatch.setattr(diagnostic, "get_access_token", lambda: context)
    monkeypatch.setattr(authentication, "get_access_token", lambda: context)
    result = diagnostic.auth_claims_summary()
    assert result == {**EMPTY, "verified_context_present": True, "permissions_present": True,
                      "permissions_shape_valid": valid, "paper_trading_present": allowed}
    assert result["paper_trading_present"] == authentication.has_paper_trading_permission()
    assert_fixed(result)


def test_exception_never_exposed(monkeypatch, caplog, capsys):
    monkeypatch.setattr(diagnostic, "get_access_token", Mock(side_effect=RuntimeError(CANARY)))
    assert diagnostic.auth_claims_summary() == EMPTY
    assert CANARY not in caplog.text
    captured = capsys.readouterr()
    assert CANARY not in captured.out + captured.err


def test_hosted_authenticated_context_and_zero_side_effects(signed, monkeypatch):  # noqa: F811
    provider, sign, _ = signed
    monkeypatch.setenv("ALPACA_TOOLSETS", "trading")
    monkeypatch.setenv("ALPACA_SAFE_OWNERSHIP_SECRET", "synthetic-ownership-test-only-123456789")
    store = Mock(open=AsyncMock(), close=AsyncMock(), list_reconcilable=AsyncMock(return_value=[]))
    network = AsyncMock(side_effect=AssertionError("external network prohibited"))
    monkeypatch.setattr("httpx.AsyncClient.send", network)
    server = build_server(hosted_mode=True, auth_provider=provider, risk_store=store)
    app = server.http_app(path="/mcp", stateless_http=True, json_response=True)
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "debug_auth_claims_summary", "arguments": {},
    }}
    with TestClient(app) as client:
        # Lifecycle initialization is separate from diagnostic invocation.
        store.reset_mock()
        getter = Mock(wraps=diagnostic.get_access_token)
        monkeypatch.setattr(diagnostic, "get_access_token", getter)
        assert client.post("/mcp", json=request).status_code == 401
        getter.assert_not_called()
        for claims, allowed in [
            ({}, False), ({"permissions": ["paper-trading"]}, True),
            ({"alpaca_role": "paper-trader"}, False),
            ({"alpaca_role": CANARY}, False), ({}, False),
        ]:
            headers = {"Authorization": "Bearer " + sign({
                "org_id": CANARY, "role": CANARY, "email": CANARY, **claims,
            }), "Accept": "application/json, text/event-stream"}
            response = client.post("/mcp", headers=headers, json=request)
            assert response.status_code == 200
            assert CANARY not in response.text
            result = response.json()["result"]["structuredContent"][DATA_KEY]
            assert_fixed(result)
            assert result["verified_context_present"]
            assert result["org_id_present"]
            assert result["paper_trading_present"] is allowed
            assert result["alpaca_role_present"] is ("alpaca_role" in claims)
            expected = "absent"
            if "alpaca_role" in claims:
                expected = "paper-trader" if claims["alpaca_role"] == "paper-trader" else "other_or_invalid"
            assert result["alpaca_role_value"] == expected
        listing = client.post("/mcp", headers=headers, json={
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        }).json()["result"]["tools"]
        safe = {t["name"] for t in listing if t["name"].startswith("safe_")}
        assert safe == {"safe_place_stock_order", "safe_place_crypto_order", "safe_cancel_order"}
        tool = next(t for t in listing if t["name"] == "debug_auth_claims_summary")
        assert tool["annotations"]["readOnlyHint"] is True
        assert tool["annotations"]["destructiveHint"] is False
        assert not store.mock_calls
        network.assert_not_called()


async def test_not_registered_on_unauthenticated_local_server(monkeypatch):
    monkeypatch.setenv("ALPACA_TOOLSETS", "trading")
    server = build_server()
    assert "debug_auth_claims_summary" not in {t.name for t in await server.list_tools()}


@pytest.mark.parametrize("value,expected", [
    ("paper-trader", "paper-trader"), ("member", "member"),
    (CANARY, "other_or_invalid"), ("", "other_or_invalid"),
    (" ", "other_or_invalid"), (" paper-trader", "other_or_invalid"),
    ("paper-trader ", "other_or_invalid"), ("paper-trader\n", "other_or_invalid"),
    ("Paper-Trader", "other_or_invalid"), ("PAPER-TRADER", "other_or_invalid"),
    ("Member", "other_or_invalid"), ([CANARY], "other_or_invalid"),
    ({"role": CANARY}, "other_or_invalid"), (1, "other_or_invalid"),
    (1.5, "other_or_invalid"), (True, "other_or_invalid"),
    (False, "other_or_invalid"), (None, "other_or_invalid"),
])
def test_alpaca_role_allowlist_and_capability(monkeypatch, caplog, capsys, value, expected):
    context = token({"alpaca_role": value, "sub": CANARY, "email": CANARY,
                     "sid": CANARY, "org_id": CANARY, "secret": CANARY})
    monkeypatch.setattr(diagnostic, "get_access_token", lambda: context)
    monkeypatch.setattr(authentication, "get_access_token", lambda: context)
    result = diagnostic.auth_claims_summary()
    assert result == {**EMPTY, "verified_context_present": True, "org_id_present": True,
                      "alpaca_role_present": True, "alpaca_role_value": expected}
    assert_fixed(result)
    assert authentication.has_paper_trading_permission() is (expected == "paper-trader")
    assert CANARY not in caplog.text
    captured = capsys.readouterr()
    assert CANARY not in captured.out + captured.err


def test_processing_exception_returns_entire_empty_summary(monkeypatch, caplog, capsys):
    class BrokenClaims(dict):
        def get(self, key, default=None):
            if key == "alpaca_role":
                raise RuntimeError(CANARY)
            return super().get(key, default)

    context = token(BrokenClaims(role="paper-trader", alpaca_role="paper-trader"))
    monkeypatch.setattr(diagnostic, "get_access_token", lambda: context)
    assert diagnostic.auth_claims_summary() == EMPTY
    assert CANARY not in caplog.text
    captured = capsys.readouterr()
    assert CANARY not in captured.out + captured.err
