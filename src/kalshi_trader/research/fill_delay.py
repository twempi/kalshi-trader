"""Would the paper fills have survived the delay of a real order?

A paper trade fills instantly against the book the bot saw. A real order reaches Kalshi a
moment later, and in thin, fast in-play markets the ask can be gone by then. This re-prices
every settled directional trade of one ledger against the *recorded* order book ``delay``
seconds after the trade (same contracts, same limit price, same liquidity haircut and fees)
and settles it the same way. A trade the delayed book cannot fill at its limit is simply
missed (P&L 0), as a real limit order would be.

Delay 0 re-prices against the decision-time book; it should reproduce the recorded P&L
closely, which checks the reconstruction itself.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal
from typing import Any

from sqlalchemy import select

from kalshi_trader.backtesting.replay import _book_from
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import (
    MarketRow,
    MarketSnapshotRow,
    PaperOrderRow,
    PaperTradeRow,
)
from kalshi_trader.execution.fills import simulate_buy

DELAYS_S = (0.0, 1.0, 2.0, 5.0)
CENT = Decimal("0.01")
ONE = Decimal("1")


def fill_delay_report(
    db: Database,
    account_id: int,
    taker_rate: Decimal,
    liquidity_fraction: Decimal,
    delays: tuple[float, ...] = DELAYS_S,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with db.session() as s:
        trades = [
            t
            for t in s.scalars(
                select(PaperTradeRow)
                .where(PaperTradeRow.account_id == account_id, PaperTradeRow.status == "settled")
                .order_by(PaperTradeRow.opened_ts)
            )
            # baskets and early exits are not single fills held to settlement
            if t.group_id is None and t.exit_contracts == 0 and t.settlement_value is not None
        ]
        limits = {
            o.trade_id: o.limit_price
            for o in s.scalars(
                select(PaperOrderRow).where(
                    PaperOrderRow.account_id == account_id, PaperOrderRow.action == "buy"
                )
            )
        }
        for t in trades:
            limit = limits.get(t.id)
            if limit is None:
                continue
            market = s.get(MarketRow, t.market_ticker)
            multiplier = market.fee_multiplier if market is not None else ONE
            value = t.settlement_value
            assert value is not None
            per_contract = value if t.side == "yes" else ONE - value
            by_delay: dict[str, Any] = {}
            for delay in delays:
                snap = s.scalars(
                    select(MarketSnapshotRow)
                    .where(
                        MarketSnapshotRow.ticker == t.market_ticker,
                        MarketSnapshotRow.received_ts <= t.opened_ts + delay,
                    )
                    .order_by(MarketSnapshotRow.received_ts.desc())
                    .limit(1)
                ).first()
                if snap is None or not snap.depth_json:
                    continue
                fill = simulate_buy(
                    _book_from(snap).asks(t.side),  # type: ignore[arg-type]
                    limit,
                    t.entry_contracts,
                    fee_rate=taker_rate,
                    fee_multiplier=multiplier,
                    liquidity_fraction=liquidity_fraction,
                )
                payout = (fill.filled * per_contract).quantize(CENT, rounding=ROUND_FLOOR)
                by_delay[str(delay)] = {
                    "filled": float(fill.filled),
                    "avg_price": float(fill.avg_price) if fill.avg_price is not None else None,
                    "pnl": float(payout + fill.cash_change),
                }
            rows.append(
                {
                    "trade_id": t.id,
                    "market": t.market_ticker,
                    "side": t.side,
                    "opened_ts": t.opened_ts,
                    "stage": t.stage,
                    "contracts": float(t.entry_contracts),
                    "entry_price": float(t.entry_price),
                    "limit_price": float(limit),
                    "pnl": float(t.realized_pnl or 0),
                    "by_delay": by_delay,
                }
            )
    return {
        "note": __doc__.strip().split("\n\n")[0] if __doc__ else "",
        "trades_checked": len(rows),
        "summary": {str(d): _summary(rows, str(d)) for d in delays},
        "trades": rows,
    }


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    have = [r for r in rows if key in r["by_delay"]]
    full = partial = missed = 0
    price_moves = []
    missed_pnl = 0.0
    for r in have:
        d = r["by_delay"][key]
        if d["filled"] >= r["contracts"]:
            full += 1
        elif d["filled"] > 0:
            partial += 1
        else:
            missed += 1
            missed_pnl += r["pnl"]
        if d["avg_price"] is not None:
            price_moves.append(d["avg_price"] - r["entry_price"])
    actual = sum(r["pnl"] for r in have)
    delayed = sum(r["by_delay"][key]["pnl"] for r in have)
    return {
        "trades": len(have),
        "no_recorded_book": len(rows) - len(have),
        "full_fills": full,
        "partial_fills": partial,
        "missed": missed,
        "missed_trades_actual_pnl": round(missed_pnl, 2),
        "pnl_actual": round(actual, 2),
        "pnl_delayed": round(delayed, 2),
        "pnl_retained": round(delayed / actual, 3) if actual else None,
        "avg_price_change_cents": round(100 * sum(price_moves) / len(price_moves), 2)
        if price_moves
        else None,
    }
