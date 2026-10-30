"""Simulated immediate-or-cancel execution against a real order book snapshot.

Conservative assumptions: whole contracts only, only a fraction of displayed depth is
assumed available (other traders react to the same prices), the order never trades
through its limit price, and anything not immediately fillable is cancelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from kalshi_trader.api.schemas import PriceRange
from kalshi_trader.execution.fees import (
    buy_cash_debit,
    order_trade_fee,
    sell_cash_credit,
)

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True)
class Fill:
    price: Decimal
    contracts: Decimal


@dataclass(frozen=True)
class FillResult:
    requested: Decimal
    fills: tuple[Fill, ...]
    notional: Decimal
    fee: Decimal
    cash_change: Decimal  # negative for buys, positive for sells

    @property
    def filled(self) -> Decimal:
        return sum((f.contracts for f in self.fills), ZERO)

    @property
    def avg_price(self) -> Decimal | None:
        filled = self.filled
        return (self.notional / filled) if filled > 0 else None

    @property
    def status(self) -> str:
        filled = self.filled
        if filled <= 0:
            return "unfilled"
        return "filled" if filled >= self.requested else "partial"


def _whole(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_FLOOR)


def snap_price_down(price: Decimal, price_ranges: Sequence[PriceRange] | None) -> Decimal:
    """Snap a price down onto the market's valid tick grid (``price_ranges``)."""
    if not price_ranges:
        step, start = Decimal("0.01"), ZERO
    else:
        band = next((r for r in price_ranges if r.start <= price <= r.end), None)
        if band is None:
            band = price_ranges[-1] if price > price_ranges[-1].end else price_ranges[0]
            price = min(max(price, band.start), band.end)
        step, start = band.step, band.start
    if step <= 0:
        return price
    return start + ((price - start) / step).to_integral_value(rounding=ROUND_FLOOR) * step


def simulate_buy(
    asks: Sequence[tuple[Decimal, Decimal]],
    limit_price: Decimal,
    max_contracts: Decimal,
    *,
    fee_rate: Decimal,
    fee_multiplier: Decimal = ONE,
    liquidity_fraction: Decimal = ONE,
    max_cash: Decimal | None = None,
) -> FillResult:
    """Walk ascending asks up to `limit_price` buying at most `max_contracts`."""
    requested = _whole(max_contracts)
    remaining = requested
    fills: list[Fill] = []
    for price, size in sorted(asks, key=lambda kv: kv[0]):
        if remaining <= 0 or price > limit_price:
            break
        available = _whole(size * liquidity_fraction)
        take = min(available, remaining)
        if take > 0:
            fills.append(Fill(price, take))
            remaining -= take

    def cost(levels: list[Fill]) -> tuple[Decimal, Decimal, Decimal]:
        notional = sum((f.price * f.contracts for f in levels), ZERO)
        trade_fee = order_trade_fee(
            [(f.price, f.contracts) for f in levels], fee_rate, fee_multiplier
        )
        debit, fee = buy_cash_debit(notional, trade_fee) if notional > 0 else (ZERO, ZERO)
        return notional, fee, debit

    notional, fee, debit = cost(fills)
    if max_cash is not None:
        # Drop contracts from the most expensive level until the order is affordable.
        while fills and debit > max_cash:
            last = fills[-1]
            if last.contracts > 1:
                fills[-1] = Fill(last.price, last.contracts - 1)
            else:
                fills.pop()
            notional, fee, debit = cost(fills)
    return FillResult(requested, tuple(fills), notional, fee, -debit)


def simulate_sell(
    bids: Sequence[tuple[Decimal, Decimal]],
    limit_price: Decimal,
    contracts: Decimal,
    *,
    fee_rate: Decimal,
    fee_multiplier: Decimal = ONE,
    liquidity_fraction: Decimal = ONE,
) -> FillResult:
    """Walk descending bids down to `limit_price` selling at most `contracts`."""
    requested = _whole(contracts)
    remaining = requested
    fills: list[Fill] = []
    for price, size in sorted(bids, key=lambda kv: kv[0], reverse=True):
        if remaining <= 0 or price < limit_price:
            break
        take = min(_whole(size * liquidity_fraction), remaining)
        if take > 0:
            fills.append(Fill(price, take))
            remaining -= take
    proceeds = sum((f.price * f.contracts for f in fills), ZERO)
    trade_fee = order_trade_fee([(f.price, f.contracts) for f in fills], fee_rate, fee_multiplier)
    credit, fee = sell_cash_credit(proceeds, trade_fee) if proceeds > 0 else (ZERO, ZERO)
    return FillResult(requested, tuple(fills), proceeds, fee, credit)
