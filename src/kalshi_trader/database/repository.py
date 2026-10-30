"""Persistence operations used by the engine and the dashboard."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import (
    ComponentHealthRow,
    EventRow,
    LiveViewRow,
    MarketRow,
    MarketSnapshotRow,
    ScoreSnapshotRow,
    SeriesRow,
    SignalRow,
    SystemEventRow,
    SystemStateRow,
)

# System-state keys (persisted so safety state survives restarts).
TRADING_ENABLED = "trading_enabled"
EMERGENCY_STOP = "emergency_stop"
PAUSE_REASON = "pause_reason"


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


class Repository:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---------------------------------------------------------- system state
    def get_state(self, key: str, default: str | None = None) -> str | None:
        with self.db.session() as s:
            row = s.get(SystemStateRow, key)
            return row.value if row else default

    def set_state(self, key: str, value: str) -> None:
        stmt = sqlite_insert(SystemStateRow).values(key=key, value=value, updated_ts=time.time())
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"value": value, "updated_ts": time.time()}
        )
        with self.db.session() as s:
            s.execute(stmt)

    def get_flag(self, key: str, default: bool = False) -> bool:
        value = self.get_state(key)
        return default if value is None else value == "true"

    def set_flag(self, key: str, value: bool) -> None:
        self.set_state(key, "true" if value else "false")

    def all_state(self) -> dict[str, str]:
        with self.db.session() as s:
            return {row.key: row.value for row in s.scalars(select(SystemStateRow))}

    # ------------------------------------------------------- events/health
    def record_event(
        self,
        level: str,
        component: str | None,
        message: str,
        details: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> None:
        self.record_events(
            [
                {
                    "ts": ts or time.time(),
                    "level": level,
                    "component": component,
                    "message": message,
                    "details": details,
                }
            ]
        )

    def record_events(self, events: Iterable[dict[str, Any]]) -> None:
        rows = [
            SystemEventRow(
                ts=float(e.get("ts") or time.time())
                if not isinstance(e.get("ts"), str)
                else time.time(),
                level=str(e.get("level", "info"))[:16],
                component=(str(e["component"])[:64] if e.get("component") else None),
                message=str(e.get("message", ""))[:2000],
                details_json=_json(e["details"]) if e.get("details") else None,
            )
            for e in events
        ]
        if rows:
            with self.db.session() as s:
                s.add_all(rows)

    def recent_events(self, limit: int = 100) -> list[SystemEventRow]:
        with self.db.session() as s:
            stmt = select(SystemEventRow).order_by(SystemEventRow.ts.desc()).limit(limit)
            return list(s.scalars(stmt))

    def prune_events(self, keep_seconds: float) -> None:
        with self.db.session() as s:
            s.execute(delete(SystemEventRow).where(SystemEventRow.ts < time.time() - keep_seconds))

    def upsert_health(
        self,
        component: str,
        status: str,
        *,
        ok: bool,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        values: dict[str, Any] = {
            "component": component,
            "status": status,
            "updated_ts": now,
            "details_json": _json(details) if details is not None else None,
        }
        if ok:
            values["last_ok_ts"] = now
        if error:
            values["last_error_ts"] = now
            values["last_error"] = error[:2000]
        stmt = sqlite_insert(ComponentHealthRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["component"], set_={k: v for k, v in values.items() if k != "component"}
        )
        with self.db.session() as s:
            s.execute(stmt)

    def all_health(self) -> list[ComponentHealthRow]:
        with self.db.session() as s:
            return list(
                s.scalars(select(ComponentHealthRow).order_by(ComponentHealthRow.component))
            )

    # ------------------------------------------------------ series/markets
    def upsert_series(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        now = time.time()
        with self.db.session() as s:
            for row in rows:
                stmt = sqlite_insert(SeriesRow).values(**row, updated_ts=now)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker"],
                    set_={**{k: v for k, v in row.items() if k != "ticker"}, "updated_ts": now},
                )
                s.execute(stmt)

    def upsert_events(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        now = time.time()
        with self.db.session() as s:
            for row in rows:
                stmt = sqlite_insert(EventRow).values(**row, first_seen_ts=now, updated_ts=now)
                # Never overwrite known metadata with NULLs from a sparser source.
                update = {k: v for k, v in row.items() if k != "event_ticker" and v is not None}
                stmt = stmt.on_conflict_do_update(
                    index_elements=["event_ticker"], set_={**update, "updated_ts": now}
                )
                s.execute(stmt)

    def upsert_markets(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        now = time.time()
        with self.db.session() as s:
            for row in rows:
                stmt = sqlite_insert(MarketRow).values(**row, first_seen_ts=now, updated_ts=now)
                update = {k: v for k, v in row.items() if k != "ticker" and v is not None}
                stmt = stmt.on_conflict_do_update(
                    index_elements=["ticker"], set_={**update, "updated_ts": now}
                )
                s.execute(stmt)

    def get_market(self, ticker: str) -> MarketRow | None:
        with self.db.session() as s:
            return s.get(MarketRow, ticker)

    def get_event(self, event_ticker: str) -> EventRow | None:
        with self.db.session() as s:
            return s.get(EventRow, event_ticker)

    def markets_for_events(self, event_tickers: Sequence[str]) -> list[MarketRow]:
        with self.db.session() as s:
            stmt = select(MarketRow).where(MarketRow.event_ticker.in_(list(event_tickers)))
            return list(s.scalars(stmt))

    def set_pre_match_prob(self, ticker: str, prob: float, ts: float) -> None:
        with self.db.session() as s:
            row = s.get(MarketRow, ticker)
            if row is not None:
                row.pre_match_prob = prob
                row.pre_match_prob_ts = ts

    # ------------------------------------------------------------ snapshots
    def insert_market_snapshots(self, rows: Sequence[dict[str, Any]]) -> None:
        if rows:
            with self.db.session() as s:
                s.execute(sqlite_insert(MarketSnapshotRow), list(rows))

    def insert_score_snapshots(self, rows: Sequence[dict[str, Any]]) -> None:
        if rows:
            with self.db.session() as s:
                s.execute(sqlite_insert(ScoreSnapshotRow), list(rows))

    def last_snapshot_before(self, ticker: str, ts: float) -> MarketSnapshotRow | None:
        with self.db.session() as s:
            stmt = (
                select(MarketSnapshotRow)
                .where(MarketSnapshotRow.ticker == ticker, MarketSnapshotRow.received_ts < ts)
                .order_by(MarketSnapshotRow.received_ts.desc())
                .limit(1)
            )
            return s.scalars(stmt).first()

    def snapshot_counts(self) -> dict[str, int]:
        with self.db.session() as s:
            return {
                "market_snapshots": s.scalar(select(func.count(MarketSnapshotRow.id))) or 0,
                "score_snapshots": s.scalar(select(func.count(ScoreSnapshotRow.id))) or 0,
                "markets": s.scalar(select(func.count(MarketRow.ticker))) or 0,
                "events": s.scalar(select(func.count(EventRow.event_ticker))) or 0,
            }

    # ------------------------------------------------------------ live view
    def upsert_live_views(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.db.session() as s:
            for row in rows:
                values = {
                    "market_ticker": row["market_ticker"],
                    "event_ticker": row["event_ticker"],
                    "updated_ts": row["updated_ts"],
                    "data_json": _json(row["data"]),
                }
                stmt = sqlite_insert(LiveViewRow).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["market_ticker"],
                    set_={k: v for k, v in values.items() if k != "market_ticker"},
                )
                s.execute(stmt)

    def delete_live_views(self, tickers: Sequence[str]) -> None:
        if tickers:
            with self.db.session() as s:
                s.execute(delete(LiveViewRow).where(LiveViewRow.market_ticker.in_(list(tickers))))

    def live_views(self) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(select(LiveViewRow).order_by(LiveViewRow.event_ticker))
            return [json.loads(r.data_json) | {"view_updated_ts": r.updated_ts} for r in rows]

    # -------------------------------------------------------------- signals
    def insert_signal(self, row: dict[str, Any]) -> int:
        with self.db.session() as s:
            signal = SignalRow(**row)
            s.add(signal)
            s.flush()
            return signal.id

    def recent_signals(self, limit: int = 200) -> list[SignalRow]:
        with self.db.session() as s:
            stmt = select(SignalRow).order_by(SignalRow.created_ts.desc()).limit(limit)
            return list(s.scalars(stmt))
