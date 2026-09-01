from __future__ import annotations

import asyncio
import os
from decimal import Decimal

import asyncpg
import pytest
import pytest_asyncio

from alpaca_mcp_server.risk_store import (
    PostgresRiskStore,
    RiskLimitExceeded,
    RiskLimits,
    RiskStoreError,
)

DSN = os.environ.get("TEST_POSTGRES_DSN", "")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="set TEST_POSTGRES_DSN to a disposable PostgreSQL database"),
]
ORDER = "123e4567-e89b-42d3-a456-426614174000"
LIMITS = RiskLimits(5, Decimal(300), Decimal(500), Decimal(200))


@pytest_asyncio.fixture
async def store():
    admin = await asyncpg.connect(DSN)
    await admin.execute("DROP TABLE IF EXISTS safe_v2_operations")
    await admin.close()
    value = PostgresRiskStore(DSN)
    await value.open()
    try:
        yield value
    finally:
        await value.close()


async def reserve(store, n, amount="10", symbol=None, limits=LIMITS, client_id=None):
    return await store.reserve(
        principal="principal-a",
        strategy_id="strategy-a",
        idempotency_key=f"key-{n}",
        client_order_id=client_id or f"safe-v2-{n:064x}",
        symbol=symbol or f"SYM{n}",
        asset_type="stock",
        requested_notional=Decimal(amount),
        limits=limits,
    )


@pytest.mark.asyncio
async def test_01_schema_creation_from_empty_database(store):
    connection = await asyncpg.connect(DSN)
    columns = await connection.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name='safe_v2_operations'"
    )
    await connection.close()
    assert {row["column_name"] for row in columns} >= {
        "principal",
        "strategy_id",
        "idempotency_key",
        "client_order_id",
        "requested_notional",
        "reserved_notional",
        "status",
        "uncertain",
    }


@pytest.mark.asyncio
async def test_02_initialization_is_idempotent(store):
    second = PostgresRiskStore(DSN)
    await second.open()
    await second.open()
    await second.close()


@pytest.mark.asyncio
async def test_03_unique_principal_strategy_idempotency(store):
    first = await reserve(store, 1)
    replay = await reserve(store, 1)
    assert first.created and not replay.created
    with pytest.raises(RiskStoreError):
        await store.reserve(
            principal="principal-a",
            strategy_id="strategy-a",
            idempotency_key="key-1",
            client_order_id=first.operation.client_order_id,
            symbol="CHANGED",
            asset_type="stock",
            requested_notional=Decimal(10),
            limits=LIMITS,
        )


@pytest.mark.asyncio
async def test_04_unique_client_order_id_and_rollback(store):
    shared = "safe-v2-" + "a" * 64
    await reserve(store, 1, client_id=shared)
    with pytest.raises(RiskStoreError):
        await reserve(store, 2, client_id=shared)
    connection = await asyncpg.connect(DSN)
    count = await connection.fetchval("SELECT COUNT(*) FROM safe_v2_operations")
    partial = await connection.fetchval(
        "SELECT COUNT(*) FROM safe_v2_operations WHERE idempotency_key='key-2'"
    )
    await connection.close()
    assert count == 1 and partial == 0


@pytest.mark.asyncio
async def test_05_transaction_scoped_advisory_lock(store):
    first = await asyncpg.connect(DSN)
    second = await asyncpg.connect(DSN)
    tx1 = first.transaction()
    await tx1.start()
    await first.execute("SELECT pg_advisory_xact_lock($1)", store._LOCK_ID)
    tx2 = second.transaction()
    await tx2.start()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            second.execute("SELECT pg_advisory_xact_lock($1)", store._LOCK_ID), 0.1
        )
    await tx2.rollback()
    await tx1.commit()
    async with second.transaction():
        await asyncio.wait_for(
            second.execute("SELECT pg_advisory_xact_lock($1)", store._LOCK_ID), 1
        )
    await first.close()
    await second.close()


async def concurrent_count(stores, count, amount, symbol, limits):
    results = await asyncio.gather(
        *(reserve(stores[n % len(stores)], n, amount, symbol, limits) for n in range(count)),
        return_exceptions=True,
    )
    return sum(not isinstance(value, Exception) for value in results), results


@pytest.mark.asyncio
async def test_06_concurrent_open_order_limit(store):
    ok, results = await concurrent_count([store], 12, "10", None, LIMITS)
    failures = [value for value in results if isinstance(value, Exception)]
    assert ok == 5 and all(isinstance(value, RiskLimitExceeded) for value in failures)


@pytest.mark.asyncio
async def test_07_concurrent_open_notional_limit(store):
    limits = RiskLimits(50, Decimal(300), Decimal(5000), Decimal(5000))
    ok, _ = await concurrent_count([store], 10, "100", None, limits)
    assert ok == 3


@pytest.mark.asyncio
async def test_08_concurrent_daily_limit(store):
    limits = RiskLimits(50, Decimal(5000), Decimal(500), Decimal(5000))
    ok, _ = await concurrent_count([store], 10, "100", None, limits)
    assert ok == 5


@pytest.mark.asyncio
async def test_09_concurrent_symbol_limit(store):
    limits = RiskLimits(50, Decimal(5000), Decimal(5000), Decimal(200))
    ok, _ = await concurrent_count([store], 10, "100", "AAPL", limits)
    assert ok == 2


@pytest.mark.asyncio
async def test_10_two_independent_pools_cannot_oversubscribe(store):
    second = PostgresRiskStore(DSN)
    await second.open()
    try:
        ok, _ = await concurrent_count([store, second], 20, "10", None, LIMITS)
        assert ok == 5
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_11_restart_preserves_reservation_and_ownership(store):
    created = await reserve(store, 1, "99.9900")
    await store.close()
    restarted = PostgresRiskStore(DSN)
    await restarted.open()
    row = await restarted.get_by_client_order_id(created.operation.client_order_id)
    replay = await reserve(restarted, 1, "99.9900")
    assert row == created.operation and not replay.created
    await restarted.close()


@pytest.mark.asyncio
async def test_12_uncertain_survives_restart_and_only_uncertain_reconciles(store):
    uncertain = await reserve(store, 1)
    reserved = await reserve(store, 2)
    await store.mark_uncertain(uncertain.operation.client_order_id)
    await store.close()
    restarted = PostgresRiskStore(DSN)
    await restarted.open()
    row = await restarted.get_by_client_order_id(uncertain.operation.client_order_id)
    assert row.status == "uncertain" and row.uncertain
    reconciled = await restarted.reconcile_uncertain_order(
        uncertain.operation.client_order_id, ORDER
    )
    assert reconciled.status == "submitted" and reconciled.order_id == ORDER
    with pytest.raises(RiskStoreError):
        await restarted.reconcile_uncertain_order(reserved.operation.client_order_id, ORDER)
    with pytest.raises(RiskStoreError):
        await restarted.reconcile_uncertain_order("safe-v2-" + "f" * 64, ORDER)
    await restarted.close()


@pytest.mark.asyncio
async def test_13_owned_cancellation_record_survives_restart(store):
    created = await reserve(store, 1)
    await store.mark_submitted(created.operation.client_order_id, ORDER)
    await store.close()
    restarted = PostgresRiskStore(DSN)
    await restarted.open()
    row = await restarted.get_by_client_order_id(created.operation.client_order_id)
    assert row.status == "submitted" and row.order_id == ORDER
    assert (row.principal, row.strategy_id) == ("principal-a", "strategy-a")
    await restarted.close()


@pytest.mark.asyncio
async def test_14_connection_loss_fails_closed(store):
    await store._pool.close()
    with pytest.raises(RiskStoreError):
        await reserve(store, 1)


@pytest.mark.asyncio
async def test_15_decimal_numeric_round_trip_is_exact(store):
    created = await reserve(store, 1, "99.123456789012345678901234567890")
    row = await store.get_by_client_order_id(created.operation.client_order_id)
    assert isinstance(row.requested_notional, Decimal)
    assert row.requested_notional == Decimal("99.123456789012345678901234567890")
    assert row.reserved_notional == row.requested_notional


@pytest.mark.asyncio
async def test_16_verified_terminal_reconciliation_releases_only_open_risk(store):
    created = await reserve(store, 1, "75")
    await store.mark_uncertain(created.operation.client_order_id)
    terminal = await store.reconcile_verified(
        created.operation,
        order_id=ORDER,
        broker_status="filled",
        target_status="filled",
    )
    assert terminal.status == "filled" and terminal.reserved_notional == 0
    assert terminal.broker_status == "filled" and terminal.reconciled_at is not None
    connection = await asyncpg.connect(DSN)
    daily = await connection.fetchval(
        "SELECT SUM(requested_notional) FROM safe_v2_operations WHERE submitted_at >= CURRENT_DATE"
    )
    await connection.close()
    assert daily == Decimal(75)
