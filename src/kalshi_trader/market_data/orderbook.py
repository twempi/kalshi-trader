"""Local order book for a binary Kalshi market.

Kalshi publishes resting *bids* only (https://docs.kalshi.com/getting_started/orderbook_responses):
a YES bid at p is equivalent to a NO ask at 1 - p, and a NO bid at q is a YES ask at 1 - q.
So the executable YES asks are derived from the NO bids, and vice versa.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

Side = Literal["yes", "no"]
ONE = Decimal("1")
ZERO = Decimal("0")


def other_side(side: Side) -> Side:
    return "no" if side == "yes" else "yes"


@dataclass
class OrderBook:
    ticker: str
    yes_bids: dict[Decimal, Decimal] = field(default_factory=dict)
    no_bids: dict[Decimal, Decimal] = field(default_factory=dict)
    updated_at: float | None = None  # local wall-clock (epoch seconds) of last change
    source: str = ""
    valid: bool = False  # False until a snapshot is applied, and after a sequence gap
    clock: int | None = None

    # ------------------------------------------------------------- mutation
    def apply_snapshot(
        self,
        yes_levels: Iterable[tuple[Decimal, Decimal]],
        no_levels: Iterable[tuple[Decimal, Decimal]],
        *,
        source: str,
        received_at: float | None = None,
        clock: int | None = None,
    ) -> None:
        self.yes_bids = {p: s for p, s in yes_levels if ZERO < p < ONE and s > 0}
        self.no_bids = {p: s for p, s in no_levels if ZERO < p < ONE and s > 0}
        self.updated_at = received_at if received_at is not None else time.time()
        self.source = source
        self.clock = clock
        self.valid = True

    def apply_delta(
        self,
        side: Side,
        price: Decimal,
        delta: Decimal,
        *,
        received_at: float | None = None,
        clock: int | None = None,
    ) -> None:
        if not (ZERO < price < ONE):
            raise ValueError(f"delta price outside (0, 1): {price}")
        book = self.yes_bids if side == "yes" else self.no_bids
        new_size = book.get(price, ZERO) + delta
        if new_size < 0:  # removes more than is resting: an update was missed
            self.invalidate()
            raise ValueError(f"delta leaves {side} {price} negative: book out of sync")
        if new_size > 0:
            book[price] = new_size
        else:
            book.pop(price, None)
        self.updated_at = received_at if received_at is not None else time.time()
        if clock is not None:
            self.clock = clock

    def invalidate(self) -> None:
        self.valid = False

    # --------------------------------------------------------------- queries
    def bids(self, side: Side) -> list[tuple[Decimal, Decimal]]:
        """Resting bids to buy `side`, best (highest) first."""
        book = self.yes_bids if side == "yes" else self.no_bids
        return sorted(book.items(), key=lambda kv: kv[0], reverse=True)

    def asks(self, side: Side) -> list[tuple[Decimal, Decimal]]:
        """Executable asks to buy `side`, best (lowest) first, derived from the opposite bids."""
        opposite = self.no_bids if side == "yes" else self.yes_bids
        return sorted(((ONE - p, s) for p, s in opposite.items()), key=lambda kv: kv[0])

    def best_bid(self, side: Side) -> tuple[Decimal, Decimal] | None:
        levels = self.bids(side)
        return levels[0] if levels else None

    def best_ask(self, side: Side) -> tuple[Decimal, Decimal] | None:
        levels = self.asks(side)
        return levels[0] if levels else None

    def spread(self, side: Side = "yes") -> Decimal | None:
        bid, ask = self.best_bid(side), self.best_ask(side)
        if bid is None or ask is None:
            return None
        return ask[0] - bid[0]

    def mid(self, side: Side = "yes") -> Decimal | None:
        bid, ask = self.best_bid(side), self.best_ask(side)
        if bid is None or ask is None:
            return None
        return (bid[0] + ask[0]) / 2

    def ask_depth(self, side: Side, limit_price: Decimal) -> Decimal:
        """Contracts available to buy `side` at or below `limit_price`."""
        return sum((s for p, s in self.asks(side) if p <= limit_price), ZERO)

    def bid_depth(self, side: Side, limit_price: Decimal) -> Decimal:
        """Contracts that can be sold (`side`) at or above `limit_price`."""
        return sum((s for p, s in self.bids(side) if p >= limit_price), ZERO)

    def age_s(self, now: float | None = None) -> float | None:
        if self.updated_at is None:
            return None
        return max(0.0, (now if now is not None else time.time()) - self.updated_at)

    def top_levels(self, depth: int) -> dict[str, list[list[str]]]:
        return {
            "yes_bids": [[str(p), str(s)] for p, s in self.bids("yes")[:depth]],
            "no_bids": [[str(p), str(s)] for p, s in self.bids("no")[:depth]],
        }

    def signature(self, depth: int) -> str:
        """Stable digest of the top of book, used to skip recording unchanged snapshots."""
        payload = repr((self.bids("yes")[:depth], self.bids("no")[:depth]))
        return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()

    def copy(self) -> OrderBook:
        return OrderBook(
            ticker=self.ticker,
            yes_bids=dict(self.yes_bids),
            no_bids=dict(self.no_bids),
            updated_at=self.updated_at,
            source=self.source,
            valid=self.valid,
            clock=self.clock,
        )
