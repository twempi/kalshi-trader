"""Execution-aware evaluation: size the order, walk the book, charge exact fees, re-check EV.

This replaces the earlier practice of computing expected value for a hypothetical
10-contract order. For each opportunity the evaluator finds the *largest* order that

* fits the risk budget (maximum loss allowed for this trade, from the risk manager);
* fits the fractional-Kelly stake (model strategies; recomputed at the final average cost);
* is fillable from displayed depth (times ``displayed_liquidity_fraction``) within the
  strategy's limit price;
* still clears the strategy's minimum net edge **after** the walked average price and the
  exact Kalshi fee for that order size (fees round up per order, so small orders pay more
  per contract).

If no size satisfies all of these the opportunity is rejected with the binding reason.

Directional (single YES/NO buy), with conservative win probability q and point p::

    expected_profit_low = n * q - (notional + fees)
    expected_profit     = n * p - (notional + fees)
    max_loss            = notional + fees      (the contract can settle at $0)

Baskets (multi-leg, every leg bought n times) with a verified minimum payout G per unit::

    guaranteed_profit = n * G - sum_legs(notional + fees)
    max_loss          = max(0, -guaranteed_profit, legging_risk)

where ``legging_risk`` is the cost of every leg except the cheapest one: legs execute one
after another, so the realistic worst case is that the final leg cannot be bought after the
others filled and the unhedged legs settle at zero. This is why baskets are never treated
as risk-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

from kalshi_trader.core.opportunity import DecisionStatus, Leg, Opportunity
from kalshi_trader.execution.fills import FillResult, simulate_buy
from kalshi_trader.market_data.orderbook import OrderBook

ZERO = Decimal("0")
ONE = Decimal("1")
MAX_LIMIT = Decimal("0.99")


def kelly_fraction_binary(p_win: float, cost: Decimal) -> Decimal:
    """Full-Kelly fraction of bankroll for a $1 binary bought at all-in `cost` per contract.

    f* = (q - c) / (1 - c): the bet loses `c` with probability 1-q and wins `1-c` with
    probability q. Depends only on the estimated edge, never on past results.
    """
    if cost <= 0 or cost >= 1:
        return ZERO
    edge = Decimal(str(p_win)) - cost
    return max(ZERO, edge / (ONE - cost))


@dataclass(frozen=True)
class SizingBudget:
    max_loss: Decimal  # dollars this trade may lose (from the risk manager / allocator)
    equity: Decimal  # for fractional Kelly
    kelly_fraction: Decimal
    size_multiplier: Decimal = ONE  # evidence-based scaling (unvalidated strategies < 1)
    max_contracts: Decimal | None = None


class ExecutionEvaluator:
    def __init__(
        self,
        taker_rate: Decimal,
        liquidity_fraction: Decimal = ONE,
        max_search_steps: int = 2000,
    ) -> None:
        self.taker_rate = taker_rate
        self.liquidity_fraction = liquidity_fraction
        self.max_search_steps = max_search_steps

    # ----------------------------------------------------------------- core
    def fill(self, book: OrderBook, leg: Leg, contracts: Decimal) -> FillResult:
        limit = leg.limit_price if leg.limit_price is not None else MAX_LIMIT
        return simulate_buy(
            book.asks(leg.side),
            limit,
            contracts,
            fee_rate=self.taker_rate,
            fee_multiplier=leg.fee_multiplier,
            liquidity_fraction=self.liquidity_fraction,
        )

    def fillable(self, book: OrderBook, leg: Leg) -> Decimal:
        """Contracts available within the leg's limit after the liquidity haircut."""
        limit = leg.limit_price if leg.limit_price is not None else MAX_LIMIT
        total = ZERO
        for price, size in book.asks(leg.side):
            if price > limit:
                break
            total += (size * self.liquidity_fraction).to_integral_value(rounding=ROUND_FLOOR)
        return total

    @staticmethod
    def _largest(n_max: int, ok: Callable[[int], bool], steps: int) -> int:
        """Largest n in [1, n_max] with ok(n). Binary search (ok is monotone up to fee
        rounding at very small n), then a short linear scan below to absorb rounding."""
        if n_max < 1:
            return 0
        lo, hi = 0, n_max
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        if lo >= 1:
            return lo
        for n in range(min(n_max, steps), 0, -1):  # rounding can make tiny sizes fail
            if ok(n):
                return n
        return 0

    # ---------------------------------------------------------- directional
    def evaluate(
        self, opp: Opportunity, books: Mapping[str, OrderBook], budget: SizingBudget
    ) -> Opportunity:
        if opp.kind == "basket":
            return self._evaluate_basket(opp, books, budget)
        return self._evaluate_directional(opp, books, budget)

    def _evaluate_directional(
        self, opp: Opportunity, books: Mapping[str, OrderBook], budget: SizingBudget
    ) -> Opportunity:
        leg = opp.primary
        book = books.get(leg.market_ticker)
        if book is None or not book.valid:
            return opp.reject(DecisionStatus.REJECTED, "no valid order book at evaluation")
        best = book.best_ask(leg.side)
        if best is None:
            return opp.reject(DecisionStatus.REJECTED, "no executable ask")
        depth = int(self.fillable(book, leg))
        if budget.max_contracts is not None:
            depth = min(depth, int(budget.max_contracts))
        if depth < 1:
            return opp.reject(DecisionStatus.REJECTED, "no fillable depth within limit price")

        fair = opp.fair
        q = fair.p_low if fair else None
        min_edge = opp.min_edge if opp.edge_gated else None
        cache: dict[int, FillResult] = {}

        def result(n: int) -> FillResult:
            if n not in cache:
                cache[n] = self.fill(book, leg, Decimal(n))
            return cache[n]

        def debit(n: int) -> Decimal:
            return -result(n).cash_change

        def stake_cap(n: int) -> Decimal:
            cap = budget.max_loss * budget.size_multiplier
            if q is not None and opp.edge_gated:
                per = debit(n) / n
                kelly = kelly_fraction_binary(q, per) * budget.kelly_fraction
                cap = min(cap, budget.equity * kelly * budget.size_multiplier)
            return cap

        def ok(n: int) -> bool:
            r = result(n)
            if r.filled < n:
                return False
            if debit(n) > stake_cap(n):
                return False
            if min_edge is not None and q is not None:
                return Decimal(str(q)) * n - debit(n) >= min_edge * n
            return True

        n = self._largest(depth, ok, self.max_search_steps)
        if n < 1:
            return opp.reject(DecisionStatus.REJECTED, self._why_zero(opp, book, leg, budget, q))

        r = result(n)
        contracts = Decimal(n)
        leg.contracts, leg.avg_price, leg.notional, leg.fee = (
            contracts,
            r.avg_price,
            r.notional,
            r.fee,
        )
        opp.contracts = contracts
        opp.cost = debit(n)
        opp.fees = opp.cost - r.notional  # includes cent-rounding of the debit
        opp.slippage = r.notional - best[0] * contracts
        opp.max_loss = opp.cost
        opp.size_multiplier = budget.size_multiplier
        if fair is not None:
            opp.expected_profit = float(Decimal(str(fair.p)) * contracts - opp.cost)
            opp.expected_profit_low = float(Decimal(str(fair.p_low)) * contracts - opp.cost)
        elif opp.market_prob is not None:  # baselines: EV if the market mid is fair
            ev = float(Decimal(str(opp.market_prob)) * contracts - opp.cost)
            opp.expected_profit = opp.expected_profit_low = ev
        return opp

    def _why_zero(
        self,
        opp: Opportunity,
        book: OrderBook,
        leg: Leg,
        budget: SizingBudget,
        q: float | None,
    ) -> str:
        one = self.fill(book, leg, ONE)
        if one.filled < 1:
            return "no fillable depth within limit price"
        cost1 = -one.cash_change
        if q is not None and opp.edge_gated and Decimal(str(q)) - cost1 < opp.min_edge:
            return (
                f"net edge vanishes after fill+fees (q={q:.3f}, all-in {cost1} "
                f"< min edge {opp.min_edge})"
            )
        if q is not None and opp.edge_gated and kelly_fraction_binary(q, cost1) <= 0:
            return "no Kelly edge at executable cost"
        return f"risk budget ${budget.max_loss * budget.size_multiplier:.2f} below one contract"

    # --------------------------------------------------------------- basket
    def _evaluate_basket(
        self, opp: Opportunity, books: Mapping[str, OrderBook], budget: SizingBudget
    ) -> Opportunity:
        payout = opp.guaranteed_payout
        if payout is None:
            return opp.reject(DecisionStatus.REJECTED, "basket without verified payout")
        legs_books: list[tuple[Leg, OrderBook]] = []
        for leg in opp.legs:
            book = books.get(leg.market_ticker)
            if book is None or not book.valid or book.best_ask(leg.side) is None:
                return opp.reject(
                    DecisionStatus.REJECTED, f"leg {leg.market_ticker} {leg.side}: no ask"
                )
            legs_books.append((leg, book))
        depth = min(int(self.fillable(b, leg)) for leg, b in legs_books)
        if budget.max_contracts is not None:
            depth = min(depth, int(budget.max_contracts))
        if depth < 1:
            return opp.reject(DecisionStatus.REJECTED, "a leg has no fillable depth")

        cache: dict[int, list[FillResult]] = {}

        def results(n: int) -> list[FillResult]:
            if n not in cache:
                cache[n] = [self.fill(b, leg, Decimal(n)) for leg, b in legs_books]
            return cache[n]

        def costs(n: int) -> list[Decimal]:
            return [-r.cash_change for r in results(n)]

        def max_loss(n: int) -> Decimal:
            c = costs(n)
            guaranteed = payout * n - sum(c, ZERO)
            legging = sum(c, ZERO) - min(c)
            return max(ZERO, -guaranteed, legging)

        def ok(n: int) -> bool:
            if any(r.filled < n for r in results(n)):
                return False
            if payout * n - sum(costs(n), ZERO) < opp.min_edge * n:
                return False
            return max_loss(n) <= budget.max_loss * budget.size_multiplier

        n = self._largest(depth, ok, self.max_search_steps)
        if n < 1:
            one = results(1)
            if all(r.filled >= 1 for r in one):
                edge = payout - sum(costs(1), ZERO)
                if edge < opp.min_edge:
                    return opp.reject(
                        DecisionStatus.REJECTED,
                        f"basket edge {edge} per unit after fees < min {opp.min_edge}",
                    )
                return opp.reject(DecisionStatus.REJECTED, "legging risk exceeds risk budget")
            return opp.reject(DecisionStatus.REJECTED, "a leg has no fillable depth")

        contracts = Decimal(n)
        total_cost = ZERO
        notional = ZERO
        best_notional = ZERO
        for (leg, book), r in zip(legs_books, results(n), strict=True):
            leg.contracts, leg.avg_price, leg.notional, leg.fee = (
                contracts,
                r.avg_price,
                r.notional,
                r.fee,
            )
            total_cost += -r.cash_change
            notional += r.notional
            best = book.best_ask(leg.side)
            best_notional += (best[0] if best else ZERO) * contracts
        opp.contracts = contracts
        opp.cost = total_cost
        opp.fees = total_cost - notional
        opp.slippage = notional - best_notional
        guaranteed = float(payout * contracts - total_cost)
        opp.expected_profit = opp.expected_profit_low = guaranteed
        opp.max_loss = max_loss(n)
        opp.size_multiplier = budget.size_multiplier
        return opp
