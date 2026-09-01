"""Durable PostgreSQL ownership and cumulative-risk ledger for Safe Trading V2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

ACTIVE_STATUSES = ("reserved", "submitted", "uncertain", "cancel_uncertain")


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_open_orders: int
    max_open_notional: Decimal
    max_daily_submitted_notional: Decimal
    max_symbol_open_notional: Decimal


@dataclass(frozen=True, slots=True)
class Operation:
    principal: str
    strategy_id: str
    idempotency_key: str
    client_order_id: str
    order_id: str | None
    symbol: str
    asset_type: str
    requested_notional: Decimal
    reserved_notional: Decimal
    status: str
    uncertain: bool
    broker_status: str | None = None
    reconciled_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Reservation:
    operation: Operation
    created: bool


class RiskStoreError(RuntimeError):
    """Raised when durable risk state cannot be safely read or changed."""


class RiskLimitExceeded(RiskStoreError):
    """Raised when an atomic reservation would exceed a hard risk limit."""


class RiskStore(Protocol):
    async def open(self) -> None: ...
    async def close(self) -> None: ...
    async def reserve(
        self,
        *,
        principal: str,
        strategy_id: str,
        idempotency_key: str,
        client_order_id: str,
        symbol: str,
        asset_type: str,
        requested_notional: Decimal,
        limits: RiskLimits,
    ) -> Reservation: ...
    async def mark_submitted(self, client_order_id: str, order_id: str) -> Operation: ...
    async def mark_uncertain(self, client_order_id: str) -> None: ...
    async def mark_rejected(self, client_order_id: str) -> None: ...
    async def get_by_client_order_id(self, client_order_id: str) -> Operation | None: ...
    async def list_reconcilable(self, principal: str | None = None) -> list[Operation]: ...
    async def reconcile_uncertain_order(self, client_order_id: str, order_id: str) -> Operation: ...
    async def reconcile_verified(
        self,
        operation: Operation,
        *,
        order_id: str,
        broker_status: str,
        target_status: str,
    ) -> Operation: ...
    async def mark_cancelled(self, client_order_id: str) -> None: ...
    async def mark_cancel_uncertain(self, client_order_id: str) -> None: ...


class UnavailableRiskStore:
    """Fail-closed store used when PostgreSQL is not configured."""

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def _fail(self) -> None:
        raise RiskStoreError("Safe Trading V2 risk store is unavailable")

    async def reserve(self, **_kwargs) -> Reservation:
        self._fail()

    async def mark_submitted(self, *_args) -> Operation:
        self._fail()

    async def mark_uncertain(self, *_args) -> None:
        self._fail()

    async def mark_rejected(self, *_args) -> None:
        self._fail()

    async def get_by_client_order_id(self, *_args) -> Operation | None:
        self._fail()

    async def list_reconcilable(self, *_args) -> list[Operation]:
        self._fail()

    async def reconcile_uncertain_order(self, *_args) -> Operation:
        self._fail()

    async def reconcile_verified(self, *_args, **_kwargs) -> Operation:
        self._fail()

    async def mark_cancelled(self, *_args) -> None:
        self._fail()

    async def mark_cancel_uncertain(self, *_args) -> None:
        self._fail()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS safe_v2_operations (
    id BIGSERIAL PRIMARY KEY,
    principal TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    client_order_id VARCHAR(128) NOT NULL UNIQUE,
    alpaca_order_id UUID,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('stock', 'crypto')),
    requested_notional NUMERIC NOT NULL CHECK (requested_notional > 0),
    reserved_notional NUMERIC NOT NULL CHECK (reserved_notional >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('reserved', 'submitted', 'uncertain', 'rejected',
                   'cancelled', 'cancel_uncertain', 'filled', 'expired')
    ),
    uncertain BOOLEAN NOT NULL DEFAULT FALSE,
    submitted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (principal, strategy_id, idempotency_key)
);
ALTER TABLE safe_v2_operations
    ADD COLUMN IF NOT EXISTS broker_status TEXT,
    ADD COLUMN IF NOT EXISTS reconciled_at TIMESTAMPTZ;
ALTER TABLE safe_v2_operations
    DROP CONSTRAINT IF EXISTS safe_v2_operations_status_check;
ALTER TABLE safe_v2_operations
    ADD CONSTRAINT safe_v2_operations_status_check CHECK (
        status IN ('reserved', 'submitted', 'uncertain', 'rejected',
                   'cancelled', 'cancel_uncertain', 'filled', 'expired')
    );
CREATE INDEX IF NOT EXISTS safe_v2_operations_active_idx
    ON safe_v2_operations (status, symbol);
CREATE INDEX IF NOT EXISTS safe_v2_operations_submitted_idx
    ON safe_v2_operations (submitted_at);
"""


class PostgresRiskStore:
    """PostgreSQL ledger using a transaction-scoped global advisory lock."""

    _LOCK_ID = 8_172_026_002

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None

    async def open(self) -> None:
        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as connection, connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self._LOCK_ID)
                await connection.execute(SCHEMA_SQL)
        except Exception as exc:
            self._pool = None
            raise RiskStoreError("Unable to initialize Safe Trading V2 risk store") from exc

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self):
        if self._pool is None:
            raise RiskStoreError("Safe Trading V2 risk store is unavailable")
        return self._pool

    @staticmethod
    def _operation(row) -> Operation:
        return Operation(
            principal=row["principal"],
            strategy_id=row["strategy_id"],
            idempotency_key=row["idempotency_key"],
            client_order_id=row["client_order_id"],
            order_id=str(row["alpaca_order_id"]) if row["alpaca_order_id"] else None,
            symbol=row["symbol"],
            asset_type=row["asset_type"],
            requested_notional=Decimal(row["requested_notional"]),
            reserved_notional=Decimal(row["reserved_notional"]),
            status=row["status"],
            uncertain=row["uncertain"],
            broker_status=row["broker_status"],
            reconciled_at=row["reconciled_at"],
        )

    async def reserve(
        self,
        *,
        principal: str,
        strategy_id: str,
        idempotency_key: str,
        client_order_id: str,
        symbol: str,
        asset_type: str,
        requested_notional: Decimal,
        limits: RiskLimits,
    ) -> Reservation:
        pool = self._require_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self._LOCK_ID)
                existing = await connection.fetchrow(
                    """SELECT * FROM safe_v2_operations
                       WHERE principal=$1 AND strategy_id=$2 AND idempotency_key=$3""",
                    principal,
                    strategy_id,
                    idempotency_key,
                )
                if existing is not None:
                    operation = self._operation(existing)
                    if (
                        operation.client_order_id != client_order_id
                        or operation.symbol != symbol
                        or operation.asset_type != asset_type
                        or operation.requested_notional != requested_notional
                    ):
                        raise RiskStoreError("Idempotency key was reused with different inputs")
                    return Reservation(operation=operation, created=False)

                totals = await connection.fetchrow(
                    """SELECT
                         COUNT(*) FILTER (WHERE status = ANY($1::text[])) AS open_count,
                         COALESCE(SUM(reserved_notional) FILTER
                             (WHERE status = ANY($1::text[])), 0) AS open_notional,
                         COALESCE(SUM(reserved_notional) FILTER
                             (WHERE status = ANY($1::text[]) AND symbol=$2), 0)
                             AS symbol_notional,
                         COALESCE(SUM(requested_notional) FILTER
                             (WHERE submitted_at >= CURRENT_DATE
                              OR status='reserved'), 0) AS daily_notional
                       FROM safe_v2_operations""",
                    list(ACTIVE_STATUSES),
                    symbol,
                )
                if int(totals["open_count"]) + 1 > limits.max_open_orders:
                    raise RiskLimitExceeded("Maximum Safe V2 open orders exceeded")
                if Decimal(totals["open_notional"]) + requested_notional > limits.max_open_notional:
                    raise RiskLimitExceeded("Maximum Safe V2 open notional exceeded")
                if (
                    Decimal(totals["symbol_notional"]) + requested_notional
                    > limits.max_symbol_open_notional
                ):
                    raise RiskLimitExceeded("Maximum Safe V2 symbol open notional exceeded")
                if (
                    Decimal(totals["daily_notional"]) + requested_notional
                    > limits.max_daily_submitted_notional
                ):
                    raise RiskLimitExceeded("Maximum Safe V2 daily submitted notional exceeded")

                row = await connection.fetchrow(
                    """INSERT INTO safe_v2_operations
                       (principal, strategy_id, idempotency_key, client_order_id,
                        symbol, asset_type, requested_notional, reserved_notional, status)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$7,'reserved') RETURNING *""",
                    principal,
                    strategy_id,
                    idempotency_key,
                    client_order_id,
                    symbol,
                    asset_type,
                    requested_notional,
                )
                return Reservation(operation=self._operation(row), created=True)
        except (RiskStoreError, RiskLimitExceeded):
            raise
        except Exception as exc:
            raise RiskStoreError("Safe V2 risk reservation failed") from exc

    async def _update(self, client_order_id: str, sql: str, *args) -> Operation:
        pool = self._require_pool()
        try:
            row = await pool.fetchrow(sql, client_order_id, *args)
        except Exception as exc:
            raise RiskStoreError("Safe V2 risk state update failed") from exc
        if row is None:
            raise RiskStoreError("Safe V2 ownership record is missing")
        return self._operation(row)

    async def mark_submitted(self, client_order_id: str, order_id: str) -> Operation:
        return await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET alpaca_order_id=$2::uuid,
               status='submitted', uncertain=FALSE, submitted_at=NOW(), updated_at=NOW()
               WHERE client_order_id=$1 AND status='reserved' RETURNING *""",
            order_id,
        )

    async def mark_uncertain(self, client_order_id: str) -> None:
        await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET status='uncertain', uncertain=TRUE,
               submitted_at=COALESCE(submitted_at,NOW()), updated_at=NOW()
               WHERE client_order_id=$1 AND status='reserved' RETURNING *""",
        )

    async def mark_rejected(self, client_order_id: str) -> None:
        await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET status='rejected', reserved_notional=0,
               updated_at=NOW() WHERE client_order_id=$1 AND status='reserved' RETURNING *""",
        )

    async def get_by_client_order_id(self, client_order_id: str) -> Operation | None:
        pool = self._require_pool()
        try:
            row = await pool.fetchrow(
                "SELECT * FROM safe_v2_operations WHERE client_order_id=$1",
                client_order_id,
            )
        except Exception as exc:
            raise RiskStoreError("Safe V2 ownership lookup failed") from exc
        return self._operation(row) if row is not None else None

    async def list_reconcilable(self, principal: str | None = None) -> list[Operation]:
        pool = self._require_pool()
        try:
            rows = await pool.fetch(
                """SELECT * FROM safe_v2_operations
                   WHERE status = ANY($1::text[])
                     AND ($2::text IS NULL OR principal=$2)
                   ORDER BY id""",
                list(ACTIVE_STATUSES),
                principal,
            )
        except Exception as exc:
            raise RiskStoreError("Safe V2 reconciliation lookup failed") from exc
        return [self._operation(row) for row in rows]

    async def reconcile_uncertain_order(self, client_order_id: str, order_id: str) -> Operation:
        return await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET alpaca_order_id=$2::uuid,
               status='submitted', uncertain=FALSE, updated_at=NOW()
               WHERE client_order_id=$1 AND status='uncertain'
                 AND alpaca_order_id IS NULL RETURNING *""",
            order_id,
        )

    async def reconcile_verified(
        self,
        operation: Operation,
        *,
        order_id: str,
        broker_status: str,
        target_status: str,
    ) -> Operation:
        if target_status not in {"submitted", "filled", "expired", "cancelled", "rejected"}:
            raise RiskStoreError("Invalid Safe V2 reconciliation target")
        terminal = target_status in {"filled", "expired", "cancelled", "rejected"}
        pool = self._require_pool()
        try:
            async with pool.acquire() as connection, connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock($1)", self._LOCK_ID)
                row = await connection.fetchrow(
                    """UPDATE safe_v2_operations
                       SET alpaca_order_id=$6::uuid, status=$7, broker_status=$8,
                           reconciled_at=NOW(), uncertain=FALSE,
                           submitted_at=COALESCE(submitted_at,NOW()),
                           reserved_notional=CASE WHEN $9 THEN 0 ELSE reserved_notional END,
                           updated_at=NOW()
                       WHERE client_order_id=$1 AND principal=$2 AND strategy_id=$3
                         AND symbol=$4
                         AND (alpaca_order_id IS NULL OR alpaca_order_id=$5::uuid)
                         AND status = ANY($10::text[])
                       RETURNING *""",
                    operation.client_order_id,
                    operation.principal,
                    operation.strategy_id,
                    operation.symbol,
                    order_id,
                    order_id,
                    target_status,
                    broker_status,
                    terminal,
                    list(ACTIVE_STATUSES),
                )
        except Exception as exc:
            raise RiskStoreError("Safe V2 verified reconciliation failed") from exc
        if row is None:
            raise RiskStoreError("Safe V2 reconciliation state changed or mismatched")
        return self._operation(row)

    async def mark_cancelled(self, client_order_id: str) -> None:
        await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET status='cancelled', reserved_notional=0,
               uncertain=FALSE, updated_at=NOW()
               WHERE client_order_id=$1
                 AND status IN ('submitted', 'cancel_uncertain') RETURNING *""",
        )

    async def mark_cancel_uncertain(self, client_order_id: str) -> None:
        await self._update(
            client_order_id,
            """UPDATE safe_v2_operations SET status='cancel_uncertain', uncertain=TRUE,
               updated_at=NOW() WHERE client_order_id=$1
                 AND status='submitted' RETURNING *""",
        )
