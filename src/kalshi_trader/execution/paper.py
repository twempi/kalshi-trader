"""Paper-trading broker: simulated execution and persistent account state.

No real orders are ever sent. Fills are simulated against the live Kalshi order book
captured at decision time (see ``fills.py``). Each order carries a deterministic
``client_order_id`` with a UNIQUE constraint, so a restart or a repeated signal can never
create a duplicate position. Account, order and trade changes commit atomically.

Accounts: id 1 (``main``) is the paper account the central allocator trades. Other
accounts (``shadow:<strategy>``) are standalone per-strategy ledgers used to compare
strategies on identical observations; they never share cash with the main account.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from sqlalchemy import String, cast, func, select
from sqlalchemy.exc import IntegrityError

from kalshi_trader.config import FeeSettings, PaperSettings
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import (
    EquitySnapshotRow,
    PaperAccountRow,
    PaperOrderRow,
    PaperTradeRow,
)
from kalshi_trader.execution.fills import FillResult, simulate_buy, simulate_sell
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.orderbook import OrderBook, Side
from kalshi_trader.risk.manager import AccountSnapshot, OpenExposure

log = get_logger("paper")
ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")
ACCOUNT_ID = 1  # the main paper account
MAIN_ACCOUNT = "main"


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    strategy: str
    market_ticker: str
    event_ticker: str
    side: Side
    limit_price: Decimal
    contracts: Decimal
    max_cash: Decimal
    fee_multiplier: Decimal = ONE
    match_title: str = ""
    player_name: str = ""
    score_text: str = ""
    model_prob: float | None = None
    model_prob_low: float | None = None
    robust_prob: float | None = None
    market_prob: float | None = None
    ev_per_contract: float | None = None
    data_source: str = "live_paper"
    strategy_version: str | None = None
    group_id: str | None = None
    opportunity_id: str | None = None
    best_price: Decimal | None = None
    expected_pnl: float | None = None
    expected_pnl_point: float | None = None
    max_loss: Decimal | None = None
    category: str | None = None
    stage: str | None = None
    correlation_group: str | None = None
    # Extra multiplier on displayed depth for this order (basket legs after the first).
    liquidity_multiplier: Decimal = ONE


@dataclass(frozen=True)
class ExecutionReport:
    status: str  # filled | partial | unfilled | duplicate
    order_id: int | None = None
    trade_id: int | None = None
    fill: FillResult | None = None


def _scaled(value: float | None, share: Decimal) -> float | None:
    return None if value is None else value * float(share)


def _fills_json(fill: FillResult) -> str:
    return json.dumps([[str(f.price), str(f.contracts)] for f in fill.fills])


class PaperBroker:
    def __init__(
        self,
        db: Database,
        paper: PaperSettings,
        fees: FeeSettings,
        clock: Callable[[], float] = time.time,
        account_name: str = MAIN_ACCOUNT,
    ) -> None:
        self.db = db
        self.paper = paper
        self.fees = fees
        self.clock = clock
        self.account_name = account_name
        self.account_id = self.ensure_account()

    # --------------------------------------------------------------- account
    def ensure_account(self) -> int:
        """Return this broker's account id, creating the account if needed."""
        with self.db.session() as s:
            if self.account_name == MAIN_ACCOUNT:
                row = s.get(PaperAccountRow, ACCOUNT_ID)
                if row is not None and row.name is None:
                    row.name = MAIN_ACCOUNT
                account_id = ACCOUNT_ID
            else:
                row = s.scalars(
                    select(PaperAccountRow).where(PaperAccountRow.name == self.account_name)
                ).first()
                if row is not None:
                    return row.id
                highest = s.scalar(select(func.max(PaperAccountRow.id))) or ACCOUNT_ID
                account_id = max(highest + 1, 100)  # 2..99 reserved
            if row is None:
                now = self.clock()
                start = self.paper.starting_balance
                s.add(
                    PaperAccountRow(
                        id=account_id,
                        name=self.account_name,
                        starting_balance=start,
                        cash=start,
                        realized_pnl=ZERO,
                        fees_paid=ZERO,
                        peak_equity=start,
                        created_ts=now,
                        updated_ts=now,
                    )
                )
                log.debug(
                    "paper_account_created",
                    account=self.account_name,
                    starting_balance=str(start),
                )
            return account_id

    def account(self) -> PaperAccountRow:
        with self.db.session() as s:
            row = s.get(PaperAccountRow, self.account_id)
            assert row is not None
            return row

    def open_trades(self) -> list[PaperTradeRow]:
        with self.db.session() as s:
            stmt = select(PaperTradeRow).where(
                PaperTradeRow.status == "open", PaperTradeRow.account_id == self.account_id
            )
            return list(s.scalars(stmt.order_by(PaperTradeRow.opened_ts)))

    def all_trades(self) -> list[PaperTradeRow]:
        with self.db.session() as s:
            stmt = select(PaperTradeRow).where(PaperTradeRow.account_id == self.account_id)
            return list(s.scalars(stmt.order_by(PaperTradeRow.opened_ts)))

    def trade_counts(self, market_ticker: str, event_ticker: str) -> tuple[int, int, bool]:
        """(positions ever opened in market, in event, has open position in event).

        Legs of one multi-leg basket count as a single position.
        """
        key = func.coalesce(PaperTradeRow.group_id, cast(PaperTradeRow.id, String))
        mine = PaperTradeRow.account_id == self.account_id
        with self.db.session() as s:
            in_market = (
                s.scalar(
                    select(func.count(func.distinct(key))).where(
                        mine, PaperTradeRow.market_ticker == market_ticker
                    )
                )
                or 0
            )
            in_event = (
                s.scalar(
                    select(func.count(func.distinct(key))).where(
                        mine, PaperTradeRow.event_ticker == event_ticker
                    )
                )
                or 0
            )
            open_in_event = (
                s.scalar(
                    select(func.count(PaperTradeRow.id)).where(
                        mine,
                        PaperTradeRow.event_ticker == event_ticker,
                        PaperTradeRow.status == "open",
                    )
                )
                or 0
            ) > 0
        return in_market, in_event, open_in_event

    def order_attempts(self, market_ticker: str, strategy: str) -> int:
        with self.db.session() as s:
            return (
                s.scalar(
                    select(func.count(PaperOrderRow.id)).where(
                        PaperOrderRow.account_id == self.account_id,
                        PaperOrderRow.market_ticker == market_ticker,
                        PaperOrderRow.strategy == strategy,
                    )
                )
                or 0
            )

    def snapshot(
        self,
        marks: dict[tuple[str, str], Decimal],
        now: float | None = None,
        day_start_ts: float | None = None,
    ) -> AccountSnapshot:
        """Account state with open positions marked at `marks[(ticker, side)]` (best bid).

        Exposure per open position is its remaining maximum loss: the recorded ex-ante max
        loss (or entry cost) scaled by the share of contracts still held.
        """
        account = self.account()
        open_value = ZERO
        open_cost = ZERO
        exposures = []
        trades = self.open_trades()
        for trade in trades:
            mark = marks.get((trade.market_ticker, trade.side))
            if mark is None:
                mark = trade.entry_price  # no bid seen: fall back to cost
            open_value += trade.contracts * mark
            cost = trade.entry_cost + trade.entry_fee
            open_cost += cost
            held = trade.contracts / trade.entry_contracts if trade.entry_contracts else ONE
            base = trade.max_loss if trade.max_loss is not None else cost
            exposures.append(
                OpenExposure(
                    strategy=trade.strategy,
                    event_ticker=trade.event_ticker or "",
                    market_ticker=trade.market_ticker,
                    max_loss=base * held,
                    correlation_group=trade.correlation_group,
                    group_id=trade.group_id,
                )
            )
        equity = account.cash + open_value
        day_start_equity = equity
        if day_start_ts is not None:
            with self.db.session() as s:
                first = s.scalars(
                    select(EquitySnapshotRow)
                    .where(
                        EquitySnapshotRow.ts >= day_start_ts,
                        EquitySnapshotRow.account_id == self.account_id,
                    )
                    .order_by(EquitySnapshotRow.ts)
                    .limit(1)
                ).first()
                if first is not None:
                    day_start_equity = first.equity
        return AccountSnapshot(
            cash=account.cash,
            equity=equity,
            peak_equity=max(account.peak_equity, equity),
            day_start_equity=day_start_equity,
            open_positions=len({t.group_id or t.id for t in trades}),
            open_cost=open_cost,
            exposures=tuple(exposures),
        )

    def record_equity(self, snap: AccountSnapshot, now: float | None = None) -> None:
        ts = now if now is not None else self.clock()
        with self.db.session() as s:
            account = s.get(PaperAccountRow, self.account_id)
            assert account is not None
            if snap.equity > account.peak_equity:
                account.peak_equity = snap.equity
            account.updated_ts = ts
            s.add(
                EquitySnapshotRow(
                    ts=ts,
                    cash=snap.cash,
                    open_value=snap.equity - snap.cash,
                    equity=snap.equity,
                    realized_pnl=account.realized_pnl,
                    open_positions=snap.open_positions,
                    account_id=self.account_id,
                )
            )

    # -------------------------------------------------------------- execution
    def execute_buy(self, intent: OrderIntent, book: OrderBook) -> ExecutionReport:
        now = self.clock()
        try:
            with self.db.session() as s:
                existing = s.scalars(
                    select(PaperOrderRow).where(
                        PaperOrderRow.client_order_id == intent.client_order_id
                    )
                ).first()
                if existing is not None:
                    return ExecutionReport("duplicate", existing.id, existing.trade_id)
                account = s.get(PaperAccountRow, self.account_id)
                assert account is not None
                max_cash = min(intent.max_cash, account.cash)
                fill = simulate_buy(
                    book.asks(intent.side),
                    intent.limit_price,
                    intent.contracts,
                    fee_rate=self.fees.taker_rate,
                    fee_multiplier=intent.fee_multiplier,
                    liquidity_fraction=self.paper.displayed_liquidity_fraction
                    * intent.liquidity_multiplier,
                    max_cash=max_cash,
                )
                order = PaperOrderRow(
                    client_order_id=intent.client_order_id,
                    created_ts=now,
                    strategy=intent.strategy,
                    market_ticker=intent.market_ticker,
                    event_ticker=intent.event_ticker,
                    side=intent.side,
                    action="buy",
                    limit_price=intent.limit_price,
                    requested_contracts=fill.requested,
                    filled_contracts=fill.filled,
                    avg_price=fill.avg_price,
                    notional=fill.notional,
                    fee=fill.fee,
                    status=fill.status,
                    fills_json=_fills_json(fill),
                    data_source=intent.data_source,
                    account_id=self.account_id,
                    group_id=intent.group_id,
                    opportunity_id=intent.opportunity_id,
                    best_price_at_decision=intent.best_price,
                )
                trade_id = None
                if fill.filled > 0:
                    avg = fill.avg_price or ZERO
                    # Ex-ante numbers were computed for the intended size; scale them to
                    # what actually filled so expected-vs-realised stays comparable.
                    share = fill.filled / fill.requested if fill.requested else ONE
                    debit = -fill.cash_change
                    trade = PaperTradeRow(
                        strategy=intent.strategy,
                        market_ticker=intent.market_ticker,
                        event_ticker=intent.event_ticker,
                        match_title=intent.match_title,
                        player_name=intent.player_name,
                        side=intent.side,
                        opened_ts=now,
                        contracts=fill.filled,
                        entry_contracts=fill.filled,
                        entry_price=avg,
                        entry_cost=fill.notional,
                        entry_fee=fill.fee,
                        score_at_entry=intent.score_text[:128],
                        model_prob=intent.model_prob,
                        model_prob_low=intent.model_prob_low,
                        robust_prob=intent.robust_prob,
                        market_prob=intent.market_prob,
                        ev_per_contract=intent.ev_per_contract,
                        status="open",
                        data_source=intent.data_source,
                        account_id=self.account_id,
                        strategy_version=intent.strategy_version,
                        group_id=intent.group_id,
                        opportunity_id=intent.opportunity_id,
                        category=intent.category,
                        stage=intent.stage,
                        correlation_group=intent.correlation_group,
                        expected_pnl=_scaled(intent.expected_pnl, share),
                        expected_pnl_point=_scaled(intent.expected_pnl_point, share),
                        max_loss=(
                            min(intent.max_loss * share, debit)
                            if intent.max_loss is not None
                            else debit
                        ),
                        slippage=(
                            fill.notional - intent.best_price * fill.filled
                            if intent.best_price is not None
                            else None
                        ),
                    )
                    s.add(trade)
                    s.flush()
                    trade_id = trade.id
                    order.trade_id = trade.id
                    account.cash += fill.cash_change
                    account.fees_paid += fill.fee
                    account.updated_ts = now
                s.add(order)
                s.flush()
                order_id = order.id
        except IntegrityError:
            return ExecutionReport("duplicate")
        log.info(
            "paper_order",
            client_order_id=intent.client_order_id,
            status=fill.status,
            market=intent.market_ticker,
            side=intent.side,
            filled=str(fill.filled),
            avg_price=str(fill.avg_price),
            fee=str(fill.fee),
        )
        return ExecutionReport(fill.status, order_id, trade_id, fill)

    def execute_sell(
        self,
        trade_id: int,
        book: OrderBook,
        limit_price: Decimal,
        client_order_id: str,
        fee_multiplier: Decimal = ONE,
        contracts: Decimal | None = None,
    ) -> ExecutionReport:
        """Sell held contracts into the bid (IOC): optional early exits and basket unwinds.
        ``contracts`` limits the quantity (default: everything still held)."""
        now = self.clock()
        try:
            with self.db.session() as s:
                if s.scalars(
                    select(PaperOrderRow).where(PaperOrderRow.client_order_id == client_order_id)
                ).first():
                    return ExecutionReport("duplicate")
                trade = s.get(PaperTradeRow, trade_id)
                account = s.get(PaperAccountRow, self.account_id)
                if (
                    trade is None
                    or trade.status != "open"
                    or account is None
                    or trade.account_id != self.account_id
                ):
                    return ExecutionReport("unfilled")
                quantity = trade.contracts if contracts is None else min(contracts, trade.contracts)
                fill = simulate_sell(
                    book.bids(trade.side),
                    limit_price,
                    quantity,
                    fee_rate=self.fees.taker_rate,
                    fee_multiplier=fee_multiplier,
                    liquidity_fraction=self.paper.displayed_liquidity_fraction,
                )
                order = PaperOrderRow(
                    client_order_id=client_order_id,
                    created_ts=now,
                    strategy=trade.strategy,
                    market_ticker=trade.market_ticker,
                    event_ticker=trade.event_ticker,
                    side=trade.side,
                    action="sell",
                    limit_price=limit_price,
                    requested_contracts=fill.requested,
                    filled_contracts=fill.filled,
                    avg_price=fill.avg_price,
                    notional=fill.notional,
                    fee=fill.fee,
                    status=fill.status,
                    fills_json=_fills_json(fill),
                    trade_id=trade.id,
                    data_source=trade.data_source,
                    account_id=self.account_id,
                    group_id=trade.group_id,
                )
                if fill.filled > 0:
                    trade.contracts -= fill.filled
                    trade.exit_contracts += fill.filled
                    trade.exit_proceeds += fill.notional
                    trade.exit_fee += fill.fee
                    account.cash += fill.cash_change
                    account.fees_paid += fill.fee
                    if trade.contracts <= 0:
                        trade.status = "closed"
                        trade.closed_ts = now
                        trade.realized_pnl = (
                            trade.exit_proceeds
                            - trade.exit_fee
                            - trade.entry_cost
                            - trade.entry_fee
                        )
                        account.realized_pnl += trade.realized_pnl
                s.add(order)
        except IntegrityError:
            return ExecutionReport("duplicate")
        return ExecutionReport(fill.status, None, trade_id, fill)

    def mark_unhedged(self, trade_id: int) -> None:
        """A basket leg that could not be hedged or unwound is a plain directional position:
        its maximum loss is its full cost."""
        with self.db.session() as s:
            trade = s.get(PaperTradeRow, trade_id)
            if trade is not None and trade.account_id == self.account_id:
                trade.max_loss = trade.entry_cost + trade.entry_fee

    # ------------------------------------------------------------- settlement
    def settle_market(
        self,
        market_ticker: str,
        result: str,
        settlement_value: Decimal | None,
        ts: float | None = None,
    ) -> list[PaperTradeRow]:
        """Settle open trades. `settlement_value` is the payout per YES contract."""
        yes_value = settlement_value
        if yes_value is None:
            yes_value = {"yes": ONE, "no": ZERO}.get(result)
        if yes_value is None or not (ZERO <= yes_value <= ONE):
            log.warning(
                "unsettleable_result",
                market=market_ticker,
                result=result,
                settlement_value=str(settlement_value),
            )
            return []
        now = ts if ts is not None else self.clock()
        settled = []
        with self.db.session() as s:
            account = s.get(PaperAccountRow, self.account_id)
            assert account is not None
            trades = s.scalars(
                select(PaperTradeRow).where(
                    PaperTradeRow.market_ticker == market_ticker,
                    PaperTradeRow.status == "open",
                    PaperTradeRow.account_id == self.account_id,
                )
            )
            for trade in trades:
                per_contract = yes_value if trade.side == "yes" else ONE - yes_value
                payout = (trade.contracts * per_contract).quantize(CENT, rounding=ROUND_FLOOR)
                trade.payout = payout
                trade.realized_pnl = (
                    payout
                    + trade.exit_proceeds
                    - trade.exit_fee
                    - trade.entry_cost
                    - trade.entry_fee
                )
                trade.status = "settled"
                trade.settlement_result = result or "value"
                trade.settlement_value = yes_value
                trade.closed_ts = now
                account.cash += payout
                account.realized_pnl += trade.realized_pnl
                account.updated_ts = now
                settled.append(trade)
        for trade in settled:
            log.info(
                "paper_trade_settled",
                market=market_ticker,
                side=trade.side,
                result=result,
                pnl=str(trade.realized_pnl),
            )
        return settled
