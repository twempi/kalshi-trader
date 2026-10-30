"""Strategy interface.

A strategy looks at one match (both markets, live score, model estimate) and returns zero
or more trade *signals*. Signals are only proposals: every signal goes through the central
RiskManager, which can reject or resize it, before the paper broker simulates execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch
from kalshi_trader.market_data.orderbook import OrderBook, Side
from kalshi_trader.models.estimator import ProbabilityEstimate
from kalshi_trader.tennis.state import MatchState


@dataclass(frozen=True)
class SideOption:
    """One way to back `player`: YES on their market, or NO on the opponent's market."""

    player: int
    market: TrackedMarket
    side: Side
    book: OrderBook


@dataclass
class MatchContext:
    now: float
    match: TrackedMatch
    markets: dict[str, TrackedMarket]
    books: dict[str, OrderBook]
    score: MatchState | None
    estimate: ProbabilityEstimate | None
    exchange_trading_active: bool

    def options_for(self, player: int) -> list[SideOption]:
        options = []
        own, other = self.match.market_tickers[player], self.match.market_tickers[1 - player]
        for ticker, side in ((own, "yes"), (other, "no")):
            if ticker and ticker in self.markets and ticker in self.books:
                options.append(SideOption(player, self.markets[ticker], side, self.books[ticker]))
        return options

    def score_age_s(self) -> float | None:
        return None if self.score is None else max(0.0, self.now - self.score.received_ts)


@dataclass(frozen=True)
class Signal:
    strategy: str
    market_ticker: str
    event_ticker: str
    side: Side
    player: int
    player_name: str
    best_ask: Decimal
    ask_size: Decimal
    limit_price: Decimal
    fee_multiplier: Decimal
    p_point: float | None  # model probability for the backed player
    p_low: float | None  # lower credible bound
    p_conservative: float | None  # latency-robust lower bound, used for EV and sizing
    market_prob: float | None  # market-implied (mid) for the backed side
    ev_per_contract: float | None  # conservative EV at the best ask, after fees
    requires_score: bool
    score_text: str
    reason: str


class Strategy(ABC):
    name: str = "abstract"
    requires_score: bool = True

    @abstractmethod
    def evaluate(self, ctx: MatchContext) -> list[Signal]: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "requires_score": self.requires_score}
