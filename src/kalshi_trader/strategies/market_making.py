"""Strategy D: market making / spread capture - RESEARCH SIMULATION ONLY.

This never touches the paper account. It answers one question with recorded/live books:
*if we had rested one-tick-improved bids on both sides of wide markets, would the fills we
plausibly got have made money after maker fees and adverse selection?*

Model (deliberately pessimistic):

* **Hypothetical quotes.** When a market's spread is at least ``min_spread`` and the price is
  inside ``[min_price, max_price]``, we post a YES bid one tick above the best YES bid and a
  NO bid one tick above the best NO bid (i.e. a YES ask one tick below the best YES ask),
  ``quote_size`` contracts each. Quotes are cancelled and replaced every cycle, so they never
  rest at stale prices. Quoting stops on the side that would push net inventory beyond
  ``max_inventory``.
* **Fill rule ``trade_through``.** A resting YES bid at p is counted as filled only if the
  best YES ask later drops *strictly below* p (some seller crossed our level, which would
  have executed against us first because we were alone at the improved price). Fills from
  sellers hitting our bid exactly at p are *not* counted, because the public order book
  cannot show them and queue position is unknowable. This understates fill rates and
  overweights fills that happen because the market moved against us - so it is a
  conservative estimate of maker P&L, not an optimistic one.
* **Fill size** = min(quote size, contracts displayed at the crossing price).
* **Fees.** The maker fee (``fees.maker_rate`` x multiplier x C x P x (1 - P), rounded up) is
  always charged, although many series charge makers nothing.
* **Adverse selection** is measured as the mid move against each fill after
  ``adverse_horizon_s``.
* **Settlement** values inventory at the official settlement value.

Displayed spreads are hypothetical quoting opportunities; only rows in
``mm_simulated_fills`` are simulated fills, and neither is a real or paper trade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_trader.config import FeeSettings, MarketMakingStrategySettings
from kalshi_trader.database.repository import Repository
from kalshi_trader.execution.fees import order_trade_fee
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.orderbook import Side
from kalshi_trader.strategies.base import ScanContext

log = get_logger("market_making")
ZERO = Decimal("0")
ONE = Decimal("1")
TICK = Decimal("0.01")
MM_SUMMARY_KEY = "mm_summary"


@dataclass
class _Fill:
    ts: float
    side: Side
    price: Decimal
    contracts: Decimal
    fee: Decimal
    mid_at_fill: float | None
    mid_after: float | None = None
    row_id: int | None = None


@dataclass
class _MarketState:
    ticker: str
    event_ticker: str
    quotes: dict[str, Decimal] = field(default_factory=dict)  # side -> resting bid price
    yes: Decimal = ZERO  # contracts held
    no: Decimal = ZERO
    cash: Decimal = ZERO  # negative = spent
    fees: Decimal = ZERO
    fills: list[_Fill] = field(default_factory=list)
    settled: bool = False
    realized: Decimal | None = None

    @property
    def net(self) -> Decimal:
        return self.yes - self.no


class MarketMakingSimulator:
    name = "market_making"
    version = "1"

    def __init__(
        self,
        settings: MarketMakingStrategySettings,
        fees: FeeSettings,
        repo: Repository | None = None,
        data_kind: str = "live",
    ) -> None:
        self.settings = settings
        self.fees = fees
        self.repo = repo
        self.data_kind = data_kind
        self.markets: dict[str, _MarketState] = {}
        self.quote_cycles = 0  # (market, cycle) pairs with at least one hypothetical quote
        self.replacements = 0
        self.skipped: dict[str, int] = {}

    @property
    def mode(self) -> str:
        return self.settings.mode

    # ---------------------------------------------------------------- cycle
    def on_cycle(self, ctx: ScanContext) -> None:
        s = self.settings
        for ticker, market in ctx.markets.items():
            book = ctx.books.get(ticker)
            state = self.markets.get(ticker)
            if book is None or not market.tradable:
                if state is not None:
                    state.quotes.clear()
                continue
            age = book.age_s(ctx.now)
            if age is None or age > s.max_book_age_s:
                self._skip("book stale")
                if state is not None:
                    state.quotes.clear()  # never let a quote rest on data we cannot see
                continue
            if state is None:
                state = self.markets[ticker] = _MarketState(ticker, market.event_ticker)
            mid = book.mid("yes")
            # 1) resolve fills of the quotes that rested since the last cycle
            for side, price in list(state.quotes.items()):
                ask = book.best_ask(side)  # type: ignore[arg-type]
                if ask is not None and ask[0] < price:
                    self._fill(state, side, price, min(s.quote_size, ask[1]), ctx.now, mid)  # type: ignore[arg-type]
            # 2) adverse-selection follow-up
            for f in state.fills:
                if (
                    f.mid_after is None
                    and ctx.now - f.ts >= s.adverse_horizon_s
                    and mid is not None
                ):
                    f.mid_after = float(mid)
                    self._update_after(f)
            # 3) cancel and replace
            previous = dict(state.quotes)
            state.quotes.clear()
            spread = book.spread("yes")
            if spread is None or spread < s.min_spread or mid is None:
                self._skip("spread below minimum" if spread is not None else "one-sided book")
                continue
            if not (s.min_price <= mid <= s.max_price):
                self._skip("price outside band")
                continue
            for side in ("yes", "no"):
                bid = book.best_bid(side)  # type: ignore[arg-type]
                ask = book.best_ask(side)  # type: ignore[arg-type]
                if bid is None or ask is None:
                    continue
                price = bid[0] + TICK
                if price >= ask[0]:
                    continue
                signed = 1 if side == "yes" else -1
                if signed * state.net + s.quote_size > s.max_inventory:
                    self._skip("inventory limit")
                    continue
                state.quotes[side] = price
            if state.quotes:
                self.quote_cycles += 1
                if previous and previous != state.quotes:
                    self.replacements += 1

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _fill(
        self,
        state: _MarketState,
        side: Side,
        price: Decimal,
        contracts: Decimal,
        ts: float,
        mid: Decimal | None,
    ) -> None:
        contracts = contracts.to_integral_value()
        if contracts <= 0:
            return
        fee = order_trade_fee([(price, contracts)], self.fees.maker_rate)
        if side == "yes":
            state.yes += contracts
        else:
            state.no += contracts
        state.cash -= price * contracts + fee
        state.fees += fee
        fill = _Fill(ts, side, price, contracts, fee, float(mid) if mid is not None else None)
        state.fills.append(fill)
        state.quotes.pop(side, None)
        if self.repo is not None:
            from kalshi_trader.database.tables import SimulatedQuoteFillRow

            with self.repo.db.session() as s:
                row = SimulatedQuoteFillRow(
                    ts=ts,
                    market_ticker=state.ticker,
                    event_ticker=state.event_ticker,
                    side=side,
                    price=price,
                    contracts=contracts,
                    fee=fee,
                    fill_rule="trade_through",
                    mid_at_fill=fill.mid_at_fill,
                    data_kind=self.data_kind,
                )
                s.add(row)
                s.flush()
                fill.row_id = row.id

    def _update_after(self, fill: _Fill) -> None:
        if self.repo is None or fill.row_id is None:
            return
        from kalshi_trader.database.tables import SimulatedQuoteFillRow

        with self.repo.db.session() as s:
            row = s.get(SimulatedQuoteFillRow, fill.row_id)
            if row is not None:
                row.mid_after = fill.mid_after

    # ----------------------------------------------------------- settlement
    def settle(self, ticker: str, yes_value: Decimal) -> None:
        state = self.markets.get(ticker)
        if state is None or state.settled:
            return
        state.quotes.clear()
        payout = state.yes * yes_value + state.no * (ONE - yes_value)
        state.realized = state.cash + payout
        state.settled = True

    # -------------------------------------------------------------- summary
    def summary(self, marks: dict[str, Decimal] | None = None) -> dict[str, Any]:
        fills = [f for st in self.markets.values() for f in st.fills]
        adverse = []
        for f in fills:
            if f.mid_at_fill is None or f.mid_after is None:
                continue
            move = f.mid_after - f.mid_at_fill
            adverse.append(-move if f.side == "yes" else move)  # positive = against us
        realized = sum(
            (st.realized for st in self.markets.values() if st.realized is not None), ZERO
        )
        unrealized = ZERO
        for st in self.markets.values():
            if st.settled or not (st.yes or st.no):
                continue
            mark = (marks or {}).get(st.ticker)
            if mark is not None:
                unrealized += st.cash + st.yes * mark + st.no * (ONE - mark)
        return {
            "mode": "research simulation (no paper or real orders)",
            "fill_rule": "trade_through (pessimistic; queue position unknown)",
            "markets_seen": len(self.markets),
            "hypothetical_quote_cycles": self.quote_cycles,
            "quote_replacements": self.replacements,
            "simulated_fills": len(fills),
            "simulated_contracts": float(sum((f.contracts for f in fills), ZERO)),
            "maker_fees": float(sum((st.fees for st in self.markets.values()), ZERO)),
            "realized_pnl_settled_markets": float(realized),
            "unrealized_pnl_at_mid": float(unrealized),
            "settled_markets": sum(1 for st in self.markets.values() if st.settled),
            "avg_adverse_move_after_fill": (sum(adverse) / len(adverse)) if adverse else None,
            "adverse_measurements": len(adverse),
            "open_inventory": {
                st.ticker: float(st.net)
                for st in self.markets.values()
                if st.net and not st.settled
            },
            "skipped": dict(self.skipped),
        }

    def persist_summary(self, marks: dict[str, Decimal] | None = None) -> None:
        if self.repo is not None:
            self.repo.set_state(MM_SUMMARY_KEY, json.dumps(self.summary(marks), default=str))
