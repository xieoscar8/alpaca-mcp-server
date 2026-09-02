"""Broker-verified, fail-closed reconciliation for Safe Trading V2."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from .risk_store import Operation, RiskStore, RiskStoreError

ACTIVE_BROKER_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "partially_filled",
    "pending_cancel",
    "pending_replace",
}
TERMINAL_BROKER_STATUSES = {
    "filled": "filled",
    "expired": "expired",
    "canceled": "cancelled",
    "cancelled": "cancelled",
    "rejected": "rejected",
}


async def reconcile_pending(
    client: httpx.AsyncClient,
    store: RiskStore,
    *,
    ownership_proof: Callable[[Operation], bool],
    principal: str | None = None,
) -> dict[str, int]:
    """Reconcile positively verified rows; ambiguous rows remain active."""
    summary = {"checked": 0, "updated": 0, "quarantined": 0}
    try:
        operations = await store.list_reconcilable(principal)
    except RiskStoreError:
        summary["quarantined"] += 1
        return summary

    for operation in operations:
        summary["checked"] += 1
        if (
            (principal is not None and operation.principal != principal)
            or not operation.strategy_id
            or not ownership_proof(operation)
        ):
            summary["quarantined"] += 1
            continue
        try:
            response = await client.get(
                "/v2/orders:by_client_order_id",
                params={"client_order_id": operation.client_order_id},
            )
        except httpx.RequestError:
            summary["quarantined"] += 1
            continue
        if response.status_code != 200:
            summary["quarantined"] += 1
            continue
        try:
            payload = response.json()
            order_id = payload["id"]
            client_order_id = payload["client_order_id"]
            symbol = payload["symbol"]
            broker_status = payload["status"]
        except (KeyError, TypeError, ValueError):
            summary["quarantined"] += 1
            continue
        if (
            not isinstance(order_id, str)
            or not isinstance(client_order_id, str)
            or not isinstance(symbol, str)
            or not isinstance(broker_status, str)
            or client_order_id != operation.client_order_id
            or symbol != operation.symbol
            or (operation.order_id is not None and operation.order_id != order_id)
        ):
            summary["quarantined"] += 1
            continue
        if broker_status in ACTIVE_BROKER_STATUSES:
            if operation.status == "cancel_uncertain":
                summary["quarantined"] += 1
                continue
            target = "submitted"
        else:
            target = TERMINAL_BROKER_STATUSES.get(broker_status)
        if target is None:
            summary["quarantined"] += 1
            continue
        try:
            await store.reconcile_verified(
                operation,
                order_id=order_id,
                broker_status=broker_status,
                target_status=target,
            )
        except RiskStoreError:
            summary["quarantined"] += 1
            continue
        summary["updated"] += 1
    return summary
