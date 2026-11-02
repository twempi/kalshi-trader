"""TradingCore: the one decision path shared by the live engine and the replay backtest.

Per evaluation cycle (``run_cycle``)::

    strategies.scan(ctx) ──> opportunities (candidates + filtered/unsupported notes)
        │
        ├─ shadow ledgers: each strategy alone, identical risk rules, its own account
        │
        └─ main account: candidates of strategies in paper mode (and not paused/rejected)
             1. market checks per leg (state, freshness, spread, liquidity, trade counts)
             2. portfolio checks (safety switches, open positions, loss limits)
             3. risk budget (max loss allowed by every exposure cap)
             4. execution evaluator (size, walk book, exact fees, EV at the final fill)
             5. allocator ranking (conservative expected return on capital at risk)
             6. for each in rank order: re-snapshot account, re-check, re-evaluate on the
                current book, then simulate execution (paper broker)
        observe-mode candidates go through 1-5 against the main account and end as
        ``observed`` (what would have happened) - they never touch main-account cash.

The clock is injected, so the replay drives the exact same code with historical time.
Nothing here can place a real order: execution is the paper broker only.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal
from typing import Any

from kalshi_trader.backtesting.metrics import outcomes_from_trades
from kalshi_trader.config import Settings
from kalshi_trader.core.allocator import rank
from kalshi_trader.core.diagnostics import Diagnostics
from kalshi_trader.core.evaluator import ExecutionEvaluator, SizingBudget
from kalshi_trader.core.opportunity import DecisionStatus, Leg, Opportunity
from kalshi_trader.core.registry import StrategyRegistry
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.database.tables import PaperTradeRow
from kalshi_trader.execution.paper import ExecutionReport, OrderIntent, PaperBroker
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.risk.manager import AccountSnapshot, MarketConditions, RiskManager
from kalshi_trader.strategies.base import ScanContext, ScoreInfo, Strategy
from kalshi_trader.strategies.market_making import MarketMakingSimulator

log = get_logger("pipeline")
ZERO = Decimal("0")
ONE = Decimal("1")
CENT = Decimal("0.01")
UNWIND_LIMIT = Decimal("0.01")  # sell excess basket legs into any bid


def _client_id(*parts: object) -> str:
    raw = ":".join(str(p) for p in parts)
    if len(raw) <= 96:
        return raw
    digest = hashlib.blake2b(raw.encode(), digest_size=10).hexdigest()
    return f"{raw[:74]}:{digest}"


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class TradingCore:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        repo: Repository,
        strategies: list[Strategy],
        *,
        clock: Callable[[], float] = time.time,
        data_kind: str = "live",
        market_maker: MarketMakingSimulator | None = None,
        persist_opportunities: bool = True,
        record_only: bool = False,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.strategies = strategies
        self.by_name = {s.name: s for s in strategies}
        self.clock = clock
        self.data_kind = data_kind
        self.record_only = record_only
        self.mm = market_maker
        self.broker = PaperBroker(db, settings.paper, settings.fees, clock=clock)
        self.shadows: dict[str, PaperBroker] = {}
        if settings.strategies.shadow_ledgers:
            for s in strategies:
                self.shadows[s.name] = PaperBroker(
                    db, settings.paper, settings.fees, clock=clock, account_name=f"shadow:{s.name}"
                )
        self.risk = RiskManager(settings.risk, repo, settings.fees.taker_rate)
        self.evaluator = ExecutionEvaluator(
            settings.fees.taker_rate, settings.paper.displayed_liquidity_fraction
        )
        self.registry = StrategyRegistry(repo, settings.evidence)
        self.registry.sync(
            [(s.name, s.version, s.config_hash, s.mode) for s in strategies], now=clock()
        )
        self.diagnostics = Diagnostics(
            repo, data_kind=data_kind, persist_opportunities=persist_opportunities
        )
        self._last_attempt: dict[tuple[int, str], float] = {}
        self.opportunities: list[Opportunity] = []

    # ------------------------------------------------------------ accounts
    def brokers(self) -> list[PaperBroker]:
        return [self.broker, *self.shadows.values()]

    @staticmethod
    def marks(books: dict[str, OrderBook]) -> dict[tuple[str, str], Decimal]:
        marks: dict[tuple[str, str], Decimal] = {}
        for ticker, book in books.items():
            if not book.valid:
                continue
            for side in ("yes", "no"):
                bid = book.best_bid(side)  # type: ignore[arg-type]
                if bid is not None:
                    marks[(ticker, side)] = bid[0]
        return marks

    def account_snapshot(
        self, broker: PaperBroker, books: dict[str, OrderBook], now: float
    ) -> AccountSnapshot:
        return broker.snapshot(self.marks(books), day_start_ts=RiskManager.utc_day_start(now))

    def open_market_tickers(self) -> set[str]:
        return {t.market_ticker for b in self.brokers() for t in b.open_trades()}

    # --------------------------------------------------------------- cycle
    def run_cycle(self, ctx: ScanContext) -> list[Opportunity]:
        opps: list[Opportunity] = []
        for strategy in self.strategies:
            try:
                opps.extend(strategy.scan(ctx))
            except Exception as exc:  # one broken strategy must not stop the others
                message = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.diagnostics.record_error(strategy.name, message)
                log.error("strategy_scan_failed", strategy=strategy.name, error=message)
        if self.mm is not None:
            try:
                self.mm.on_cycle(ctx)
            except Exception as exc:
                self.diagnostics.record_error("market_making", repr(exc)[:200])
                log.error("market_making_failed", error=repr(exc)[:200])

        info = {m.match.event_ticker: m.score_info for m in ctx.matches}
        candidates = [o for o in opps if o.status == DecisionStatus.CANDIDATE]
        if not self.record_only:
            for name, shadow in self.shadows.items():
                mine = [copy.deepcopy(o) for o in candidates if o.strategy == name]
                if mine:
                    self._allocate(shadow, mine, ctx, info, execute=True, main=False)

        main: list[Opportunity] = []
        observe: list[Opportunity] = []
        for opp in candidates:
            strategy = self.by_name.get(opp.strategy)
            allowed, why = self.registry.allows_main(
                opp.strategy, strategy.mode if strategy else "off"
            )
            if allowed:
                main.append(opp)
            else:
                opp.provenance["main_account"] = why
                observe.append(opp)
        self._allocate(self.broker, main, ctx, info, execute=not self.record_only, main=True)
        self._allocate(self.broker, observe, ctx, info, execute=False, main=True)
        self.diagnostics.record(opps, ctx.now, {s.name: s.mode for s in self.strategies})
        self.opportunities = opps
        return opps

    # ---------------------------------------------------------- allocation
    def _conditions(
        self,
        opp: Opportunity,
        leg: Leg,
        broker: PaperBroker,
        ctx: ScanContext,
        info: dict[str, ScoreInfo],
    ) -> MarketConditions:
        market = ctx.markets.get(leg.market_ticker)
        book = ctx.books.get(leg.market_ticker)
        in_market, in_event, open_in_event = broker.trade_counts(
            leg.market_ticker, opp.event_ticker
        )
        score = info.get(opp.event_ticker, ScoreInfo())
        limit = leg.limit_price if leg.limit_price is not None else Decimal("0.99")
        return MarketConditions(
            market_tradable=market is not None and market.tradable,
            exchange_trading_active=market is not None and ctx.exchange_active(market),
            book_age_s=book.age_s(ctx.now) if book is not None else None,
            score_age_s=score.age_s,
            spread=book.spread(leg.side) if book is not None else None,
            depth_at_limit=book.ask_depth(leg.side, limit) if book is not None else ZERO,
            trades_in_market=in_market,
            trades_in_event=in_event,
            has_open_position_in_event=open_in_event,
            score_unchanged_s=score.unchanged_s,
            mid_move_since_score_change=score.mid_move,
            score_issue=score.issue,
        )

    def _evaluate(
        self,
        opp: Opportunity,
        broker: PaperBroker,
        snap: AccountSnapshot,
        portfolio_reasons: list[str],
        ctx: ScanContext,
        info: dict[str, ScoreInfo],
        main: bool,
    ) -> None:
        reasons = list(portfolio_reasons)
        for leg in opp.legs:
            reasons.extend(
                self.risk.check_market(opp, self._conditions(opp, leg, broker, ctx, info))
            )
        if reasons:
            opp.reject(DecisionStatus.REJECTED, *_dedupe(reasons))
            return
        budget = self.risk.budget(opp, snap)
        if budget.reasons:
            opp.reject(DecisionStatus.REJECTED, *budget.reasons)
            return
        multiplier = (
            self.registry.size_multiplier(
                opp.strategy, self.settings.risk.unvalidated_size_multiplier
            )
            if main
            else ONE
        )
        self.evaluator.evaluate(
            opp,
            ctx.books,
            SizingBudget(
                max_loss=budget.max_loss,
                equity=max(ZERO, snap.equity),
                kelly_fraction=self.settings.risk.kelly_fraction,
                size_multiplier=multiplier,
            ),
        )
        if opp.status == DecisionStatus.CANDIDATE:
            opp.risk = {
                "risk_budget": str(budget.max_loss),
                "binding_limit": budget.binding,
                "size_multiplier": str(multiplier),
                "account": broker.account_name,
            }

    def _allocate(
        self,
        broker: PaperBroker,
        cands: list[Opportunity],
        ctx: ScanContext,
        info: dict[str, ScoreInfo],
        *,
        execute: bool,
        main: bool,
    ) -> None:
        if not cands:
            return
        proposals = {id(o): copy.deepcopy(o) for o in cands}
        snap = self.account_snapshot(broker, ctx.books, ctx.now)
        port = self.risk.check_portfolio(snap, escalate=main, safety=main)
        for opp in cands:
            self._evaluate(opp, broker, snap, port, ctx, info, main)
        cooldown = self.settings.strategies.entry_cooldown_s
        for opp in rank(o for o in cands if o.status == DecisionStatus.CANDIDATE):
            if not execute:
                opp.status = DecisionStatus.OBSERVED
                continue
            key = (broker.account_id, opp.opportunity_id)
            if ctx.now - self._last_attempt.get(key, float("-inf")) < cooldown:
                opp.reject(DecisionStatus.REJECTED, "entry cooldown")
                continue
            # Re-check and re-size against the portfolio as it is *now* (earlier
            # allocations in this cycle may have used capital or event headroom).
            fresh = copy.deepcopy(proposals[id(opp)])
            snap = self.account_snapshot(broker, ctx.books, ctx.now)
            port = self.risk.check_portfolio(snap, escalate=main, safety=main)
            self._evaluate(fresh, broker, snap, port, ctx, info, main)
            if fresh.status != DecisionStatus.CANDIDATE:
                opp.__dict__.update(fresh.__dict__)
                continue
            fresh.status = DecisionStatus.APPROVED
            self._last_attempt[key] = ctx.now
            self._execute(fresh, broker, ctx)
            opp.__dict__.update(fresh.__dict__)

    # ----------------------------------------------------------- execution
    def _source(self, broker: PaperBroker) -> str:
        shadow = broker is not self.broker
        if self.data_kind == "live":
            return "shadow" if shadow else "live_paper"
        return f"{self.data_kind}_shadow" if shadow else self.data_kind

    def _intent(
        self,
        opp: Opportunity,
        leg: Leg,
        broker: PaperBroker,
        contracts: Decimal,
        max_cash: Decimal,
        group_id: str | None,
        max_loss: Decimal,
    ) -> OrderIntent:
        attempts = broker.order_attempts(leg.market_ticker, opp.strategy)
        fair = opp.fair
        share = contracts / opp.contracts if opp.contracts else ONE
        return OrderIntent(
            client_order_id=_client_id(
                broker.account_id, opp.strategy, leg.market_ticker, leg.side, attempts + 1
            ),
            strategy=opp.strategy,
            market_ticker=leg.market_ticker,
            event_ticker=opp.event_ticker,
            side=leg.side,
            limit_price=leg.limit_price if leg.limit_price is not None else Decimal("0.99"),
            contracts=contracts,
            max_cash=max_cash,
            fee_multiplier=leg.fee_multiplier,
            match_title=opp.title[:256],
            player_name=opp.player_name,
            score_text=opp.score_text,
            model_prob=fair.p if fair else None,
            model_prob_low=fair.p_low if fair else None,
            robust_prob=fair.p_low if fair else None,
            market_prob=opp.market_prob,
            ev_per_contract=(
                opp.expected_profit_low / float(opp.contracts)
                if opp.expected_profit_low is not None and opp.contracts
                else None
            ),
            data_source=self._source(broker),
            strategy_version=opp.strategy_version,
            group_id=group_id,
            opportunity_id=opp.opportunity_id,
            best_price=leg.best_ask,
            expected_pnl=(
                opp.expected_profit_low * float(share)
                if opp.expected_profit_low is not None and group_id is None
                else None
            ),
            expected_pnl_point=(
                opp.expected_profit * float(share)
                if opp.expected_profit is not None and group_id is None
                else None
            ),
            max_loss=max_loss,
            category=opp.category,
            stage=opp.stage,
            correlation_group=opp.correlation_group,
        )

    def _execute(self, opp: Opportunity, broker: PaperBroker, ctx: ScanContext) -> None:
        if opp.kind == "basket":
            self._execute_basket(opp, broker, ctx)
            return
        leg = opp.primary
        book = ctx.books[leg.market_ticker]
        intent = self._intent(opp, leg, broker, opp.contracts, opp.cost, None, opp.max_loss)
        report = broker.execute_buy(intent, book)
        self._apply_report(opp, report)
        if (
            report.status in ("filled", "partial")
            and report.fill is not None
            and broker is self.broker
        ):
            self.repo.record_event(
                "info",
                "paper",
                f"PAPER {report.status}: {opp.strategy} bought {report.fill.filled} "
                f"{leg.side.upper()} {leg.market_ticker} @ {report.fill.avg_price} "
                f"(fee {report.fill.fee}) | {opp.reason}"[:2000],
                ts=ctx.now,
            )

    @staticmethod
    def _apply_report(opp: Opportunity, report: ExecutionReport) -> None:
        if report.status == "duplicate":
            opp.reject(DecisionStatus.REJECTED, "duplicate order id (already executed)")
        elif report.status == "filled":
            opp.status = DecisionStatus.FILLED
        elif report.status == "partial":
            opp.status = DecisionStatus.PARTIAL
            filled = report.fill.filled if report.fill else ZERO
            opp.reasons.append(f"partial fill {filled}/{opp.contracts}")
        else:
            opp.status = DecisionStatus.UNFILLED
            opp.reasons.append("no fill within limit")

    def _execute_basket(self, opp: Opportunity, broker: PaperBroker, ctx: ScanContext) -> None:
        """Legs are bought one at a time, least liquid first (most likely to fail first,
        which minimises unhedged exposure). Later legs only buy what earlier legs filled and
        may only take ``paper.basket_later_leg_liquidity_fraction`` of their displayed depth
        (a deliberate stress: real books move between legs). If legs end up unequal, the
        excess is sold back into the bids immediately; contracts that cannot be sold remain
        as directional positions carrying their full cost as maximum loss."""
        group_id = _client_id(broker.account_id, "basket", opp.opportunity_id, int(ctx.now * 1000))
        group_id = hashlib.blake2b(group_id.encode(), digest_size=12).hexdigest()
        legs = sorted(
            opp.legs, key=lambda leg: self.evaluator.fillable(ctx.books[leg.market_ticker], leg)
        )
        target = opp.contracts
        done: list[tuple[Leg, int | None, Decimal]] = []
        for leg in legs:
            if target <= 0:
                break
            book = ctx.books[leg.market_ticker]
            leg_cost = leg.notional + leg.fee
            share = leg_cost / opp.cost if opp.cost else ZERO
            max_cash = (leg_cost * target / opp.contracts).quantize(
                CENT, rounding=ROUND_CEILING
            ) + CENT
            intent = self._intent(
                opp, leg, broker, target, max_cash, group_id, opp.max_loss * share
            )
            if done:  # later legs: the book may have moved while earlier legs executed
                intent = replace(
                    intent,
                    liquidity_multiplier=self.settings.paper.basket_later_leg_liquidity_fraction,
                )
            report = broker.execute_buy(intent, book)
            filled = report.fill.filled if report.fill is not None else ZERO
            done.append((leg, report.trade_id, filled))
            target = min(target, filled)
        hedged = min((f for _, _, f in done), default=ZERO) if len(done) == len(opp.legs) else ZERO
        unwound = []
        for leg, trade_id, filled in done:
            excess = filled - hedged
            if excess <= 0 or trade_id is None:
                continue
            book = ctx.books[leg.market_ticker]
            report = broker.execute_sell(
                trade_id,
                book,
                UNWIND_LIMIT,
                _client_id(broker.account_id, "unwind", trade_id, int(ctx.now * 1000)),
                leg.fee_multiplier,
                contracts=excess,
            )
            sold = report.fill.filled if report.fill is not None else ZERO
            unwound.append(f"unwound {sold}/{excess} {leg.side} {leg.market_ticker}")
            if sold < excess:
                broker.mark_unhedged(trade_id)
        if hedged >= opp.contracts:
            opp.status = DecisionStatus.FILLED
        elif hedged > 0:
            opp.status = DecisionStatus.PARTIAL
            opp.reasons.append(f"legging: hedged {hedged}/{opp.contracts}")
        else:
            opp.status = DecisionStatus.UNFILLED
            opp.reasons.append("legging: basket could not be completed")
        opp.reasons.extend(unwound)
        if broker is self.broker and hedged > 0:
            self.repo.record_event(
                "info",
                "paper",
                f"PAPER basket {opp.status.value}: {opp.strategy} {hedged} units "
                f"{opp.relationship} | cost {opp.cost}"[:2000],
                ts=ctx.now,
            )

    # ----------------------------------------------------------- settlement
    def settle(
        self, ticker: str, result: str, value: Decimal | None, ts: float
    ) -> list[PaperTradeRow]:
        settled = []
        for broker in self.brokers():
            settled.extend(broker.settle_market(ticker, result, value, ts=ts))
        if self.mm is not None:
            yes_value = value if value is not None else {"yes": ONE, "no": ZERO}.get(result)
            if yes_value is not None:
                self.mm.settle(ticker, yes_value)
        return settled

    def record_equity(self, books: dict[str, OrderBook], now: float) -> AccountSnapshot:
        main_snap = self.account_snapshot(self.broker, books, now)
        self.broker.record_equity(main_snap, now)
        self.risk.check_account_limits(main_snap)
        for shadow in self.shadows.values():
            shadow.record_equity(self.account_snapshot(shadow, books, now), now)
        return main_snap

    def refresh_evidence(self, now: float) -> dict[str, Any]:
        """Re-evaluate every strategy's forward evidence from its shadow ledger."""
        out = {}
        start = float(self.settings.paper.starting_balance)
        for strategy in self.strategies:
            shadow = self.shadows.get(strategy.name)
            if shadow is None:
                continue
            rows = [t for t in shadow.all_trades() if t.strategy_version == strategy.version_tag]
            outcomes = outcomes_from_trades(rows)
            out[strategy.name] = self.registry.update_evidence(
                strategy.name, strategy.mode, outcomes, start, now=now
            )
        return out
