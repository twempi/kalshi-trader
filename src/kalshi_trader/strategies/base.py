"""Strategy interface and the market context strategies see.

A strategy turns a ``ScanContext`` (everything known at one instant: tracked markets, valid
order books, tennis match states and model estimates, related-market groups) into
``Opportunity`` proposals. Proposals that fail the strategy's own filters are returned too,
with status ``filtered``/``unsupported`` and a reason, so the system can always explain why
it is not trading. Strategies never size or execute trades: the shared evaluator, risk
manager and allocator in ``core.pipeline`` do that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_trader.config import FeeSettings, StrategySettingsBase
from kalshi_trader.core.opportunity import DecisionStatus, FairValue, Leg, Opportunity
from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch
from kalshi_trader.market_data.orderbook import OrderBook, Side
from kalshi_trader.models.estimator import ProbabilityEstimate
from kalshi_trader.tennis.state import MatchState


@dataclass(frozen=True)
class ScoreInfo:
    """How much the live score can be trusted right now.

    ``age_s`` = local age of the last response + the source's own age at receipt (CDN
    ``Age`` header, or the configured worst case when unknown). ``unchanged_s`` = time since
    the score last changed. ``mid_move`` = largest market-mid move since that change across
    the match's markets (a big move with no score change suggests the feed is behind).
    """

    age_s: float | None = None
    unchanged_s: float | None = None
    mid_move: Decimal | None = None
    issue: str | None = None


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
    score_info: ScoreInfo = field(default_factory=ScoreInfo)
    pre_match_p1: float | None = None

    def options_for(self, player: int) -> list[SideOption]:
        options = []
        own, other = self.match.market_tickers[player], self.match.market_tickers[1 - player]
        for ticker, side in ((own, "yes"), (other, "no")):
            if ticker and ticker in self.markets and ticker in self.books:
                options.append(SideOption(player, self.markets[ticker], side, self.books[ticker]))
        return options

    def score_age_s(self) -> float | None:
        if self.score_info.age_s is not None:
            return self.score_info.age_s
        return None if self.score is None else max(0.0, self.now - self.score.received_ts)

    @property
    def correlation_group(self) -> str:
        m = self.match
        return f"tennis:{m.tournament_name or m.competition or m.series_ticker}"


@dataclass
class RelatedGroup:
    """Markets of one Kalshi event considered together by the cross-market strategy."""

    event_ticker: str
    series_ticker: str
    title: str
    mutually_exclusive: bool | None
    markets: list[TrackedMarket]
    category: str = "generic"
    # Tennis pairs: True once a ball has been played (cancellation clause no longer applies).
    match_started: bool | None = None


@dataclass
class ScanContext:
    now: float
    markets: dict[str, TrackedMarket]
    books: dict[str, OrderBook]  # valid books only
    matches: list[MatchContext] = field(default_factory=list)
    groups: list[RelatedGroup] = field(default_factory=list)
    exchange_active: Callable[[TrackedMarket], bool] = lambda _m: True

    def book_age(self, ticker: str) -> float | None:
        book = self.books.get(ticker)
        return book.age_s(self.now) if book is not None else None


class Strategy:
    name: str = "abstract"
    version: str = "1"
    category: str = "tennis"
    requires_score: bool = True
    edge_gated: bool = True  # baselines set False: they are controls, not edge-seekers

    def __init__(self, settings: StrategySettingsBase, fees: FeeSettings) -> None:
        self.settings = settings
        self.fees = fees

    @property
    def mode(self) -> str:
        return self.settings.mode

    @property
    def config_hash(self) -> str:
        raw = json.dumps(self.settings.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    @property
    def version_tag(self) -> str:
        """Version + config hash: evidence only counts for the exact rules that earned it."""
        return f"{self.version}+{self.config_hash[:8]}"

    def scan(self, ctx: ScanContext) -> list[Opportunity]:
        out: list[Opportunity] = []
        for match_ctx in ctx.matches:
            out.extend(self.evaluate_match(match_ctx))
        return out

    def evaluate_match(self, ctx: MatchContext) -> list[Opportunity]:
        return []

    # --------------------------------------------------------------- helpers
    def directional(
        self,
        ctx: MatchContext,
        option: SideOption,
        *,
        fair: FairValue | None,
        limit_price: Decimal,
        reason: str,
        stage: str | None = None,
    ) -> Opportunity:
        book = option.book
        bid = book.best_bid(option.side)
        ask = book.best_ask(option.side)
        mid = book.mid(option.side)
        leg = Leg(
            market_ticker=option.market.ticker,
            event_ticker=ctx.match.event_ticker,
            side=option.side,
            fee_multiplier=option.market.fee_multiplier,
            price_ranges=list(option.market.price_ranges),
            best_bid=bid[0] if bid else None,
            best_ask=ask[0] if ask else None,
            ask_size=ask[1] if ask else None,
            limit_price=limit_price,
        )
        return Opportunity(
            strategy=self.name,
            strategy_version=self.version_tag,
            kind="directional",
            category="tennis",
            event_ticker=ctx.match.event_ticker,
            legs=[leg],
            observed_ts=ctx.now,
            title=f"{ctx.match.player_names[option.player]} ({ctx.match.title})",
            stage=stage,
            fair=fair,
            min_edge=self.settings.min_edge,
            edge_gated=self.edge_gated,
            requires_score=self.requires_score,
            correlation_group=ctx.correlation_group,
            provenance=_provenance(ctx, book),
            book_age_s=book.age_s(ctx.now),
            score_age_s=ctx.score_age_s(),
            player_name=ctx.match.player_names[option.player],
            score_text=ctx.score.score_text() if ctx.score else "",
            market_prob=float(mid) if mid is not None else None,
            reason=reason,
        )

    def note(
        self,
        ctx: MatchContext,
        status: DecisionStatus,
        reason: str,
        ticker: str | None = None,
        side: Side = "yes",
        stage: str | None = None,
    ) -> Opportunity:
        """A non-tradable record explaining why a match produced no candidate."""
        market = ticker or next((t for t in ctx.match.market_tickers if t), "") or ""
        return Opportunity(
            strategy=self.name,
            strategy_version=self.version_tag,
            kind="directional",
            category="tennis",
            event_ticker=ctx.match.event_ticker,
            legs=[Leg(market_ticker=market, event_ticker=ctx.match.event_ticker, side=side)],
            observed_ts=ctx.now,
            title=ctx.match.title,
            stage=stage,
            requires_score=self.requires_score,
            score_text=ctx.score.score_text() if ctx.score else "",
            status=status,
            reasons=[reason],
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "version_tag": self.version_tag,
            "mode": self.mode,
            "category": self.category,
            "requires_score": self.requires_score,
            "edge_gated": self.edge_gated,
            "params": {k: str(v) for k, v in self.settings.model_dump().items()},
        }


def _provenance(ctx: MatchContext, book: OrderBook) -> dict[str, Any]:
    score = ctx.score
    return {
        "book_source": book.source,
        "book_updated_ts": book.updated_at,
        "score_provider": score.provider if score else None,
        "score_received_ts": score.received_ts if score else None,
        "score_source_age_s": score.source_age_s if score else None,
        "score_unchanged_s": ctx.score_info.unchanged_s,
        "prior": ctx.estimate.prior.source if ctx.estimate else None,
    }
