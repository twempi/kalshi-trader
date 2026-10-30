"""Expected value of buying a contract at *executable* prices.

EV per contract = P(win) * $1 - (average fill price + fees per contract). The fill price
comes from walking the real ask side of the book for the evaluated size, never from the
displayed mid or last trade. Two EVs are reported: with the model's point probability and
with its conservative (latency-robust lower-bound) probability; only the latter is used
for trading decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.fills import FillResult, simulate_buy
from kalshi_trader.market_data.orderbook import OrderBook, Side

ZERO = Decimal("0")
ONE = Decimal("1")
TICK = Decimal("0.01")


@dataclass(frozen=True)
class EVResult:
    side: Side
    contracts: Decimal
    fill: FillResult
    cost_per_contract: Decimal | None  # average price + fee, per contract
    ev_point: float | None
    ev_conservative: float | None

    @property
    def fillable(self) -> bool:
        return self.fill.filled > 0

    @property
    def roi_conservative(self) -> float | None:
        if self.ev_conservative is None or not self.cost_per_contract:
            return None
        return self.ev_conservative / float(self.cost_per_contract)


def evaluate_buy(
    book: OrderBook,
    side: Side,
    p_point: float | None,
    p_conservative: float | None,
    contracts: Decimal,
    *,
    fee_rate: Decimal,
    fee_multiplier: Decimal = ONE,
    liquidity_fraction: Decimal = ONE,
    limit_price: Decimal = ONE,
) -> EVResult:
    fill = simulate_buy(
        book.asks(side),
        limit_price,
        contracts,
        fee_rate=fee_rate,
        fee_multiplier=fee_multiplier,
        liquidity_fraction=liquidity_fraction,
    )
    if fill.filled <= 0:
        return EVResult(side, contracts, fill, None, None, None)
    cost = (fill.notional + fill.fee) / fill.filled
    ev_point = None if p_point is None else p_point - float(cost)
    ev_cons = None if p_conservative is None else p_conservative - float(cost)
    return EVResult(side, contracts, fill, cost, ev_point, ev_cons)


def max_price_for_edge(
    p_conservative: float,
    min_edge: Decimal,
    *,
    fee_rate: Decimal,
    fee_multiplier: Decimal = ONE,
    contracts_hint: Decimal = Decimal("10"),
    ceiling: Decimal = Decimal("0.99"),
    tick: Decimal = TICK,
) -> Decimal | None:
    """Highest price on the tick grid whose conservative EV (after fees) >= min_edge."""
    price = (ceiling / tick).to_integral_value() * tick
    while price > ZERO:
        fee = fee_per_contract_estimate(price, contracts_hint, fee_rate, fee_multiplier)
        if Decimal(str(p_conservative)) - price - fee >= min_edge:
            return price
        price -= tick
    return None
