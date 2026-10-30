"""Kalshi trading fees.

General (taker) fee per the Kalshi fee schedule: ``rate * multiplier * C * P * (1 - P)``
with rate 0.07 (maker fees, where charged, use 0.0175). Per
https://docs.kalshi.com/getting_started/fee_rounding the trade fee is rounded *up* to
$0.000001, then the balance change is aligned to the member's precision ($0.01 for
non-direct members) with a rounding fee. The order-level accumulator means the total
converges to what a single equivalent fill would cost, so we compute fees per order.
Settlement of yes/no markets carries no fee.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

CENT = Decimal("0.01")
MICRO = Decimal("0.000001")
ONE = Decimal("1")
ZERO = Decimal("0")


def ceil_to(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_CEILING)


def floor_to(value: Decimal, quantum: Decimal) -> Decimal:
    return value.quantize(quantum, rounding=ROUND_FLOOR)


def raw_trade_fee(
    price: Decimal, contracts: Decimal, rate: Decimal, multiplier: Decimal = ONE
) -> Decimal:
    if not (ZERO <= price <= ONE):
        raise ValueError(f"price outside [0, 1]: {price}")
    if contracts < 0:
        raise ValueError("contracts must be non-negative")
    return rate * multiplier * contracts * price * (ONE - price)


def order_trade_fee(
    fills: Iterable[tuple[Decimal, Decimal]], rate: Decimal, multiplier: Decimal = ONE
) -> Decimal:
    """Model trade fee for all fills of one order, rounded up to $0.000001."""
    raw = sum((raw_trade_fee(p, c, rate, multiplier) for p, c in fills), ZERO)
    return ceil_to(raw, MICRO) if raw > 0 else ZERO


def buy_cash_debit(
    notional: Decimal, trade_fee: Decimal, precision: Decimal = CENT
) -> tuple[Decimal, Decimal]:
    """(cash debited, effective fee) for a buy. The debit is aligned to `precision`."""
    debit = ceil_to(notional + trade_fee, precision)
    return debit, debit - notional


def sell_cash_credit(
    proceeds: Decimal, trade_fee: Decimal, precision: Decimal = CENT
) -> tuple[Decimal, Decimal]:
    """(cash credited, effective fee) for a sell. The credit is aligned to `precision`."""
    credit = floor_to(proceeds - trade_fee, precision)
    if credit < 0:
        credit = ZERO
    return credit, proceeds - credit


def fee_per_contract_estimate(
    price: Decimal, contracts: Decimal, rate: Decimal, multiplier: Decimal = ONE
) -> Decimal:
    """All-in fee per contract for a single-price order of `contracts` (includes rounding)."""
    if contracts <= 0:
        return ZERO
    trade_fee = order_trade_fee([(price, contracts)], rate, multiplier)
    _, fee = buy_cash_debit(price * contracts, trade_fee)
    return fee / contracts
