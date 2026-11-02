"""Standardised opportunity representation shared by every strategy.

A strategy *proposes* opportunities; it never trades. Each opportunity then passes, in
order, through:

1. the strategy's own filters (failures are kept with status ``filtered`` and a reason, so
   the dashboard can show why nothing is trading);
2. the execution evaluator, which sizes the order, walks the real order book for that size,
   computes the exact fees and recomputes expected value at the *final* fill
   (``evaluated`` or ``rejected``);
3. the central risk manager and portfolio allocator (``approved`` or ``rejected``);
4. the paper broker (``filled`` / ``partial`` / ``unfilled``), or ``observed`` when the
   strategy runs in observe mode.

Prices are dollars per contract (``Decimal``). Probabilities are for the side being bought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from kalshi_trader.api.schemas import PriceRange
from kalshi_trader.market_data.orderbook import Side

ZERO = Decimal("0")
ONE = Decimal("1")


class DecisionStatus(StrEnum):
    UNSUPPORTED = "unsupported"  # no defensible probability/relationship for this market
    FILTERED = "filtered"  # failed a strategy filter
    CANDIDATE = "candidate"  # passed strategy filters, not yet evaluated
    REJECTED = "rejected"  # failed execution, risk or allocation checks
    APPROVED = "approved"  # cleared for (simulated) execution
    OBSERVED = "observed"  # would have been approved; strategy is in observe mode
    FILLED = "filled"
    PARTIAL = "partial"
    UNFILLED = "unfilled"

    @property
    def terminal_trade(self) -> bool:
        return self in (DecisionStatus.FILLED, DecisionStatus.PARTIAL)


OpportunityKind = Literal["directional", "basket"]


@dataclass(frozen=True)
class FairValue:
    """A probability estimate for the side being bought, with its uncertainty.

    ``p_low`` is the conservative value used for every trading decision; ``validated``
    says whether the source has passed out-of-sample calibration checks.
    """

    p: float
    p_low: float
    p_high: float
    source: str
    as_of_ts: float
    validated: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def uncertainty(self) -> float:
        return max(0.0, self.p_high - self.p_low)


@dataclass
class Leg:
    """One market/side of an opportunity (a single buy for directional trades)."""

    market_ticker: str
    event_ticker: str
    side: Side
    fee_multiplier: Decimal = ONE
    price_ranges: list[PriceRange] = field(default_factory=list)
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    ask_size: Decimal | None = None
    limit_price: Decimal | None = None
    # Filled by the evaluator.
    contracts: Decimal = ZERO
    avg_price: Decimal | None = None
    notional: Decimal = ZERO
    fee: Decimal = ZERO

    def to_dict(self) -> dict[str, Any]:
        return {
            "market": self.market_ticker,
            "side": self.side,
            "best_bid": _s(self.best_bid),
            "best_ask": _s(self.best_ask),
            "ask_size": _s(self.ask_size),
            "limit": _s(self.limit_price),
            "contracts": _s(self.contracts),
            "avg_price": _s(self.avg_price),
            "fee": _s(self.fee),
        }


def _s(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _f(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


@dataclass
class Opportunity:
    strategy: str
    strategy_version: str
    kind: OpportunityKind
    category: str  # "tennis" | "generic"
    event_ticker: str
    legs: list[Leg]
    observed_ts: float
    title: str = ""
    stage: str | None = None
    direction: Literal["buy"] = "buy"
    fair: FairValue | None = None
    # Baskets: payoff per unit that every outcome guarantees (verified relationship).
    guaranteed_payout: Decimal | None = None
    relationship: str | None = None
    # Strategy-level parameters for the evaluator.
    min_edge: Decimal = ZERO
    edge_gated: bool = True  # baselines are not gated on (model) edge
    requires_score: bool = False
    correlation_group: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    book_age_s: float | None = None
    score_age_s: float | None = None
    # Display/legacy fields.
    player_name: str = ""
    score_text: str = ""
    market_prob: float | None = None  # market-implied (mid) probability of the side bought
    reason: str = ""
    # Decision state.
    status: DecisionStatus = DecisionStatus.CANDIDATE
    reasons: list[str] = field(default_factory=list)
    # Evaluation results (at the final simulated fill).
    contracts: Decimal = ZERO  # per leg (baskets buy the same count of every leg)
    cost: Decimal = ZERO  # all legs: notional + fees (= cash debited)
    fees: Decimal = ZERO
    expected_profit: float | None = None  # point probability (or guaranteed payout)
    expected_profit_low: float | None = None  # conservative probability
    max_loss: Decimal = ZERO
    slippage: Decimal = ZERO  # extra paid versus the best ask, all legs
    size_multiplier: Decimal = ONE
    risk: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ identity
    @property
    def primary(self) -> Leg:
        return self.legs[0]

    @property
    def market_ticker(self) -> str:
        return self.primary.market_ticker

    @property
    def side(self) -> Side:
        return self.primary.side

    @property
    def opportunity_id(self) -> str:
        legs = "+".join(f"{leg.market_ticker}:{leg.side}" for leg in self.legs)
        return f"{self.strategy}|{legs}"[:160]

    # --------------------------------------------------------- transitions
    def reject(self, status: DecisionStatus, *reasons: str) -> Opportunity:
        self.status = status
        self.reasons.extend(r for r in reasons if r)
        return self

    @property
    def expected_roc(self) -> float | None:
        """Conservative expected profit per dollar of capital committed."""
        if self.expected_profit_low is None or self.cost <= 0:
            return None
        return self.expected_profit_low / float(self.cost)

    @property
    def avg_price(self) -> Decimal | None:
        return self.primary.avg_price

    def summary(self) -> dict[str, Any]:
        leg = self.primary
        return {
            "opportunity_id": self.opportunity_id,
            "strategy": self.strategy,
            "strategy_version": self.strategy_version,
            "kind": self.kind,
            "category": self.category,
            "stage": self.stage,
            "event_ticker": self.event_ticker,
            "market_ticker": self.market_ticker,
            "title": self.title,
            "side": self.side,
            "legs": [leg.to_dict() for leg in self.legs],
            "best_bid": _f(leg.best_bid),
            "best_ask": _f(leg.best_ask),
            "ask_size": _f(leg.ask_size),
            "fair_value": self.fair.p if self.fair else None,
            "fair_low": self.fair.p_low if self.fair else None,
            "fair_source": self.fair.source if self.fair else None,
            "market_prob": self.market_prob,
            "contracts": _f(self.contracts),
            "limit_price": _f(leg.limit_price),
            "avg_price": _f(leg.avg_price),
            "fees": _f(self.fees),
            "cost": _f(self.cost),
            "expected_profit": self.expected_profit,
            "expected_profit_low": self.expected_profit_low,
            "expected_roc": self.expected_roc,
            "max_loss": _f(self.max_loss),
            "book_age_s": self.book_age_s,
            "score_age_s": self.score_age_s,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "reason": self.reason,
            "score_text": self.score_text,
            "relationship": self.relationship,
            "provenance": self.provenance,
        }
