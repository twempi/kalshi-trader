"""Persists market and score observations for later replay/backtesting.

Only changed states are written (plus a periodic heartbeat), with the local receive time,
so a replay sees exactly what the system knew and when - no look-ahead.
"""

from __future__ import annotations

import json
import time
from typing import Any

from kalshi_trader.database.repository import Repository
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.tennis.providers.base import ProviderResult

log = get_logger("recorder")


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


class Recorder:
    def __init__(
        self, repo: Repository, depth_levels: int, book_heartbeat_s: float, score_heartbeat_s: float
    ) -> None:
        self._repo = repo
        self._depth = depth_levels
        self._book_heartbeat = book_heartbeat_s
        self._score_heartbeat = score_heartbeat_s
        self._last_book: dict[str, tuple[str, float]] = {}
        self._last_score: dict[str, tuple[str, float]] = {}
        self._market_rows: list[dict[str, Any]] = []
        self._score_rows: list[dict[str, Any]] = []
        self.market_rows_written = 0
        self.score_rows_written = 0

    def on_book(self, ticker: str, book: OrderBook, status: str | None = None) -> None:
        if not book.valid:
            return
        now = book.updated_at or time.time()
        signature = book.signature(self._depth)
        last = self._last_book.get(ticker)
        if last and last[0] == signature and now - last[1] < self._book_heartbeat:
            return
        self._last_book[ticker] = (signature, now)
        bid, ask = book.best_bid("yes"), book.best_ask("yes")
        self._market_rows.append(
            {
                "ticker": ticker,
                "received_ts": now,
                "source": book.source,
                "yes_bid": _f(bid[0]) if bid else None,
                "yes_bid_size": _f(bid[1]) if bid else None,
                "yes_ask": _f(ask[0]) if ask else None,
                "yes_ask_size": _f(ask[1]) if ask else None,
                "status": status,
                "depth_json": json.dumps(book.top_levels(self._depth), separators=(",", ":")),
            }
        )

    def on_score(self, event_ticker: str | None, result: ProviderResult) -> None:
        state = result.state
        fingerprint = state.fingerprint()
        last = self._last_score.get(state.match_id)
        if last and last[0] == fingerprint and state.received_ts - last[1] < self._score_heartbeat:
            return
        self._last_score[state.match_id] = (fingerprint, state.received_ts)
        self._score_rows.append(
            {
                "milestone_id": state.match_id,
                "event_ticker": event_ticker,
                "received_ts": state.received_ts,
                "provider": state.provider,
                "status": state.status.value,
                "state_hash": fingerprint,
                "state_json": json.dumps(state.to_dict(), default=str, separators=(",", ":")),
                "raw_json": json.dumps(result.raw, default=str, separators=(",", ":")),
            }
        )

    def flush(self) -> None:
        if self._market_rows:
            rows, self._market_rows = self._market_rows, []
            self._repo.insert_market_snapshots(rows)
            self.market_rows_written += len(rows)
        if self._score_rows:
            rows, self._score_rows = self._score_rows, []
            self._repo.insert_score_snapshots(rows)
            self.score_rows_written += len(rows)
