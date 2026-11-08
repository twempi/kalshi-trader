"""Compare early selling with holding the exact same purchases to settlement.

Only completed trades whose market outcome is known enter either total. A position sold
early is still pending for this comparison until its market settles. This avoids comparing
quickly realised wins with unfinished positions. The comparison isolates exit decisions;
it does not model the extra trades made possible by releasing capital sooner, or prove
that the simulated fills were executable.
"""

from __future__ import annotations

from decimal import ROUND_FLOOR, Decimal
from typing import Any

from sqlalchemy import select

from kalshi_trader.backtesting.metrics import TradeOutcome
from kalshi_trader.backtesting.stats import cluster_bootstrap_mean_ci
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import MarketRow, PaperTradeRow

CENT = Decimal("0.01")
ONE = Decimal("1")


def exit_comparison_report(
    db: Database, account_id: int | None, strategy_version: str
) -> dict[str, Any]:
    """Current-version directional trades in one ledger, paired with their held outcome."""
    rows: list[dict[str, Any]] = []
    differences: list[TradeOutcome] = []
    pending = open_trades = 0
    with db.session() as s:
        trades = s.execute(
            select(PaperTradeRow, MarketRow)
            .outerjoin(MarketRow, MarketRow.ticker == PaperTradeRow.market_ticker)
            .where(
                PaperTradeRow.account_id == account_id,
                PaperTradeRow.strategy_version == strategy_version,
                PaperTradeRow.group_id.is_(None),
            )
            .order_by(PaperTradeRow.opened_ts, PaperTradeRow.id)
        )
        for trade, market in trades:
            value = _settlement_value(trade, market)
            if (
                value is None
                or trade.status not in ("closed", "settled")
                or trade.realized_pnl is None
            ):
                pending += 1
                open_trades += trade.status == "open"
                continue
            per_contract = value if trade.side == "yes" else ONE - value
            held_payout = (trade.entry_contracts * per_contract).quantize(
                CENT, rounding=ROUND_FLOOR
            )
            cost = trade.entry_cost + trade.entry_fee
            held_pnl = held_payout - cost
            lift = trade.realized_pnl - held_pnl
            rows.append(
                {
                    "trade_id": trade.id,
                    "market": trade.market_ticker,
                    "side": trade.side,
                    "entry_contracts": float(trade.entry_contracts),
                    "exit_contracts": float(trade.exit_contracts),
                    "actual_pnl": float(trade.realized_pnl),
                    "hold_pnl": float(held_pnl),
                    "difference": float(lift),
                    "exit_fees": float(trade.exit_fee),
                }
            )
            differences.append(
                TradeOutcome(
                    opened_ts=trade.opened_ts,
                    closed_ts=trade.closed_ts or trade.opened_ts,
                    strategy=trade.strategy,
                    entry_price=float(trade.entry_price),
                    contracts=float(trade.entry_contracts),
                    cost=float(cost),
                    pnl=float(lift),
                    fees=float(trade.exit_fee),
                    event=trade.event_ticker,
                )
            )
    return {
        "strategy_version": strategy_version,
        "note": "Same purchases and quantities, after fees. Both totals exclude pending "
        "markets, including positions already sold. Positive difference favours early selling. "
        "Paper fills only; this does not test execution delays or capital reuse.",
        "compared_trades": len(rows),
        "compared_events": len({t.event or f"trade-{i}" for i, t in enumerate(differences)}),
        "early_exit_trades": sum(r["exit_contracts"] > 0 for r in rows),
        "pending_trades": pending,
        "open_trades": open_trades,
        "actual_pnl": round(sum(r["actual_pnl"] for r in rows), 2),
        "hold_pnl": round(sum(r["hold_pnl"] for r in rows), 2),
        "difference": round(sum(r["difference"] for r in rows), 2),
        "exit_fees": round(sum(r["exit_fees"] for r in rows), 2),
        "mean_difference_ci95": cluster_bootstrap_mean_ci(differences),
        "trades": rows,
    }


def _settlement_value(trade: PaperTradeRow, market: MarketRow | None) -> Decimal | None:
    if trade.status == "settled":
        value, result = trade.settlement_value, trade.settlement_result
    elif market is not None and market.status in ("settled", "finalized"):
        value, result = market.settlement_value, market.result
    else:
        return None
    if value is None:
        value = {"yes": ONE, "no": Decimal("0")}.get(result or "")
    return value if value is not None and 0 <= value <= ONE else None
