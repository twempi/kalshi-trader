"""Market data hub: one local order book per market, fed by WebSocket or REST.

When the WebSocket feed owns a market's book (snapshot received, sequence intact), REST
snapshots are *not* applied to it: mixing a REST snapshot into a delta stream would
double-count changes. If the WebSocket disconnects, detects a gap, or a delta shows the book
is out of sync, ownership is released and REST polling keeps the book fresh until a new
WebSocket snapshot arrives.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from kalshi_trader.api.schemas import OrderbookFp, parse_levels, parse_price, validate_ticker
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.orderbook import OrderBook

log = get_logger("market_data")

BookListener = Callable[[str, OrderBook], None]
LifecycleListener = Callable[[dict[str, Any]], None]


class MarketDataHub:
    def __init__(self) -> None:
        self.books: dict[str, OrderBook] = {}
        self._ws_owned: set[str] = set()
        self._book_listeners: list[BookListener] = []
        self._lifecycle_listeners: list[LifecycleListener] = []
        self.invalid_messages = 0
        self.resync_requests: set[str] = set()  # books that need a fresh WebSocket snapshot

    def add_book_listener(self, listener: BookListener) -> None:
        self._book_listeners.append(listener)

    def add_lifecycle_listener(self, listener: LifecycleListener) -> None:
        self._lifecycle_listeners.append(listener)

    def book(self, ticker: str) -> OrderBook:
        book = self.books.get(ticker)
        if book is None:
            book = self.books[ticker] = OrderBook(ticker)
        return book

    def ws_owns(self, ticker: str) -> bool:
        return ticker in self._ws_owned and self.book(ticker).valid

    def release_ws_ownership(self, tickers: list[str] | None = None) -> None:
        if tickers is None:
            self._ws_owned.clear()
        else:
            self._ws_owned.difference_update(tickers)

    def drop(self, ticker: str) -> None:
        self.books.pop(ticker, None)
        self._ws_owned.discard(ticker)

    def _notify(self, ticker: str, book: OrderBook) -> None:
        for listener in self._book_listeners:
            try:
                listener(ticker, book)
            except Exception as exc:  # a broken listener must not stop market data
                log.error("book_listener_failed", ticker=ticker, error=repr(exc)[:300])

    # ------------------------------------------------------------------ REST
    def apply_rest_snapshot(
        self, ticker: str, orderbook: OrderbookFp, received_at: float | None = None
    ) -> bool:
        if self.ws_owns(ticker):
            return False
        book = self.book(ticker)
        book.apply_snapshot(
            orderbook.yes_dollars,
            orderbook.no_dollars,
            source="rest",
            received_at=received_at or time.time(),
        )
        self._notify(ticker, book)
        return True

    # ------------------------------------------------------------- WebSocket
    def handle_ws_message(self, message: dict[str, Any]) -> None:
        mtype = message.get("type")
        msg = message.get("msg")
        if not isinstance(msg, dict):
            return
        try:
            if mtype == "orderbook_snapshot":
                self._ws_snapshot(msg)
            elif mtype == "orderbook_delta":
                self._ws_delta(msg)
            elif mtype in ("market_lifecycle_v2", "event_lifecycle"):
                for listener in self._lifecycle_listeners:
                    listener(msg | {"_type": mtype})
        except (ValueError, KeyError, TypeError) as exc:
            self.invalid_messages += 1
            log.warning("ws_message_rejected", type=mtype, error=str(exc)[:200])

    def _ws_snapshot(self, msg: dict[str, Any]) -> None:
        ticker = validate_ticker(msg["market_ticker"])
        book = self.book(ticker)
        book.apply_snapshot(
            parse_levels(msg.get("yes_dollars_fp")),
            parse_levels(msg.get("no_dollars_fp")),
            source="ws",
            received_at=time.time(),
            clock=msg.get("clock"),
        )
        self._ws_owned.add(ticker)
        self._notify(ticker, book)

    def _ws_delta(self, msg: dict[str, Any]) -> None:
        ticker = validate_ticker(msg["market_ticker"])
        book = self.books.get(ticker)
        if book is None or not book.valid or ticker not in self._ws_owned:
            return  # wait for a snapshot before applying deltas
        side = msg.get("side")
        if side not in ("yes", "no"):
            raise ValueError(f"bad delta side {side!r}")
        price = parse_price(msg.get("price_dollars"))
        delta = Decimal(str(msg.get("delta_fp")))
        if price is None or not delta.is_finite():
            raise ValueError("bad delta price/size")
        try:
            book.apply_delta(side, price, delta, received_at=time.time(), clock=msg.get("clock"))
        except ValueError:
            if not book.valid:  # out of sync: REST covers it until a new snapshot arrives
                self.release_ws_ownership([ticker])
                self.resync_requests.add(ticker)
            raise
        self._notify(ticker, book)

    def on_ws_gap(self, channel: str, tickers: list[str]) -> None:
        if channel == "orderbook_delta":
            for ticker in tickers:
                if ticker in self.books:
                    self.books[ticker].invalidate()
            self.release_ws_ownership(tickers)
