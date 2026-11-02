"""Strategy B: baskets of related contracts whose combined payoff is fixed by settlement rules.

Only relationships that can be verified from documented Kalshi fields are used. Similar
titles are never treated as evidence. Supported relationships (G = guaranteed minimum
payout per basket unit):

``tennis_pair``
    A tennis match event with exactly two markets for two distinct competitors
    (``custom_strike.tennis_competitor``). Once a ball has been played, the rules settle
    exactly one market YES (a retirement counts as a win for the opponent), so
    YES_A + YES_B pays G = 1, and so does NO_A + NO_B. Before the first ball, a cancellation
    settles both at a "fair price", so the pair is only considered once the live score
    shows the match in progress.

``mutually_exclusive_no``
    An event flagged ``mutually_exclusive`` (Kalshi ``MECNET``: *at most one* market can
    resolve YES). Buying NO on every one of its N markets pays at least G = N - 1. The flag
    does not imply exhaustiveness, so the YES basket is not derived from it.

``exhaustive_yes``
    Bracket markets of one event (``strike_type`` less/less_or_equal, between,
    greater/greater_or_equal with ``floor_strike``/``cap_strike``) whose ranges cover every
    real value. Buying YES on all pays at least G = 1. Strict bounds (``less``,
    ``greater``) are treated as exclusive and ``between`` as inclusive; any gap - including a
    gap that would only be harmless if the outcome were an integer - rejects the basket,
    because the settlement granularity is not verifiable from the API.

``ladder``
    Two threshold markets of the same event on the same side of the strike (both
    ``greater*`` or both ``less*``) with different strikes. For floors a < b,
    YES(> a) + NO(> b) pays at least G = 1 for every outcome (similarly for caps).

Residual risks that no relationship removes, and which are reported, not hidden: legs
execute one after another (legging risk, bounded in the evaluator), books change between
legs, and a voided market or a fair-price settlement can pay amounts other than 0/1. An
apparent mispricing must also *persist* for ``min_persistence_scans`` scans spanning
``min_persistence_s`` seconds; single-snapshot crossings are usually stale data.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from kalshi_trader.config import CrossMarketStrategySettings, FeeSettings
from kalshi_trader.core.opportunity import DecisionStatus, Leg, Opportunity
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.market_data.discovery import TrackedMarket
from kalshi_trader.market_data.orderbook import Side
from kalshi_trader.strategies.base import RelatedGroup, ScanContext, Strategy

ONE = Decimal("1")
ZERO = Decimal("0")
MAX_LIMIT = Decimal("0.99")
GREATER = ("greater", "greater_or_equal")
LESS = ("less", "less_or_equal")


@dataclass(frozen=True)
class Relationship:
    kind: str
    legs: tuple[tuple[TrackedMarket, Side], ...]
    guaranteed_payout: Decimal
    note: str


# ------------------------------------------------------- relationship discovery
def tennis_pair(group: RelatedGroup) -> list[Relationship] | str:
    markets = group.markets
    if len(markets) != 2:
        return f"tennis event has {len(markets)} markets, expected 2"
    if (
        not all(m.competitor_id for m in markets)
        or markets[0].competitor_id == markets[1].competitor_id
    ):
        return "competitor ids missing or identical"
    if not group.match_started:
        return "match not started: cancellation would settle both legs at a fair price"
    a, b = markets
    note = "exactly one player wins once a ball is played (retirement = opponent wins)"
    return [
        Relationship("tennis_pair", ((a, "yes"), (b, "yes")), ONE, note),
        Relationship("tennis_pair", ((a, "no"), (b, "no")), ONE, note),
    ]


def mutually_exclusive_no(group: RelatedGroup, max_legs: int) -> list[Relationship] | str:
    if group.mutually_exclusive is not True:
        return "event not flagged mutually exclusive"
    n = len(group.markets)
    if n < 2 or n > max_legs:
        return f"{n} markets outside [2, {max_legs}]"
    legs = tuple((m, "no") for m in group.markets)
    return [
        Relationship(
            "mutually_exclusive_no",
            legs,  # type: ignore[arg-type]
            Decimal(n - 1),
            "MECNET: at most one market resolves YES, so at least N-1 NO legs pay",
        )
    ]


def _bounds(m: TrackedMarket) -> tuple[float, bool, float, bool] | None:
    """(low, low_inclusive, high, high_inclusive) of the YES region, or None if unknown."""
    inf = float("inf")
    if m.strike_type == "less" and m.cap_strike is not None:
        return (-inf, False, m.cap_strike, False)
    if m.strike_type == "less_or_equal" and m.cap_strike is not None:
        return (-inf, False, m.cap_strike, True)
    if m.strike_type == "greater" and m.floor_strike is not None:
        return (m.floor_strike, False, inf, False)
    if m.strike_type == "greater_or_equal" and m.floor_strike is not None:
        return (m.floor_strike, True, inf, False)
    if m.strike_type == "between" and m.floor_strike is not None and m.cap_strike is not None:
        return (m.floor_strike, True, m.cap_strike, True)
    return None


def covers_real_line(markets: list[TrackedMarket]) -> str | None:
    """None if the YES regions cover every real number, else the first gap found."""
    bounds = []
    for m in markets:
        b = _bounds(m)
        if b is None:
            return f"{m.ticker}: strike type {m.strike_type!r} is not a verifiable range"
        bounds.append(b)
    bounds.sort(key=lambda b: (b[0], not b[1]))
    if bounds[0][0] != float("-inf"):
        return "no market covers the lowest values"
    reach, reach_inclusive = bounds[0][2], bounds[0][3]
    for low, low_inc, high, high_inc in bounds[1:]:
        if low > reach or (low == reach and not (low_inc or reach_inclusive)):
            return f"gap at {reach}"
        if high > reach or (high == reach and high_inc):
            reach, reach_inclusive = high, high_inc
    if reach != float("inf"):
        return "no market covers the highest values"
    return None


def exhaustive_yes(group: RelatedGroup, max_legs: int) -> list[Relationship] | str:
    n = len(group.markets)
    if n < 2 or n > max_legs:
        return f"{n} markets outside [2, {max_legs}]"
    gap = covers_real_line(group.markets)
    if gap is not None:
        return f"exhaustiveness not verified: {gap}"
    legs = tuple((m, "yes") for m in group.markets)
    return [
        Relationship(
            "exhaustive_yes",
            legs,  # type: ignore[arg-type]
            ONE,
            "bracket ranges cover every value, so at least one YES leg pays",
        )
    ]


def ladders(group: RelatedGroup) -> list[Relationship]:
    out: list[Relationship] = []
    greater = sorted(
        (m for m in group.markets if m.strike_type in GREATER and m.floor_strike is not None),
        key=lambda m: m.floor_strike or 0.0,
    )
    for i, low in enumerate(greater):
        for high in greater[i + 1 :]:
            if (high.floor_strike or 0) > (low.floor_strike or 0):
                out.append(
                    Relationship(
                        "ladder",
                        ((low, "yes"), (high, "no")),
                        ONE,
                        f"YES(>{low.floor_strike}) + NO(>{high.floor_strike}) pays >= 1",
                    )
                )
    less = sorted(
        (m for m in group.markets if m.strike_type in LESS and m.cap_strike is not None),
        key=lambda m: m.cap_strike or 0.0,
    )
    for i, low in enumerate(less):
        for high in less[i + 1 :]:
            if (high.cap_strike or 0) > (low.cap_strike or 0):
                out.append(
                    Relationship(
                        "ladder",
                        ((high, "yes"), (low, "no")),
                        ONE,
                        f"YES(<{high.cap_strike}) + NO(<{low.cap_strike}) pays >= 1",
                    )
                )
    return out


# ----------------------------------------------------------------- strategy
class CrossMarketStrategy(Strategy):
    name = "cross_market"
    version = "1"
    category = "multi"
    requires_score = False

    def __init__(self, settings: CrossMarketStrategySettings, fees: FeeSettings) -> None:
        super().__init__(settings, fees)
        self.settings: CrossMarketStrategySettings = settings
        # basket id -> (first seen ts, consecutive qualifying scans)
        self._persistence: dict[str, tuple[float, int]] = {}

    def groups(self, ctx: ScanContext) -> list[RelatedGroup]:
        groups = list(ctx.groups)
        if self.settings.include_tennis_pairs:
            for m in ctx.matches:
                markets = [ctx.markets[t] for t in m.match.market_tickers if t and t in ctx.markets]
                groups.append(
                    RelatedGroup(
                        event_ticker=m.match.event_ticker,
                        series_ticker=m.match.series_ticker,
                        title=m.match.title,
                        mutually_exclusive=None,
                        markets=markets,
                        category="tennis",
                        match_started=bool(m.score and m.score.is_live),
                    )
                )
        return groups

    def relationships(self, group: RelatedGroup) -> tuple[list[Relationship], list[str]]:
        found: list[Relationship] = []
        why: list[str] = []
        if group.category == "tennis":
            result = tennis_pair(group)
            if isinstance(result, str):
                why.append(result)
            else:
                found.extend(result)
            return found, why
        for discover in (mutually_exclusive_no, exhaustive_yes):
            result = discover(group, self.settings.max_legs)
            if isinstance(result, str):
                why.append(result)
            else:
                found.extend(result)
        found.extend(ladders(group))
        return found, why

    def _unit_cost(self, ctx: ScanContext, rel: Relationship) -> tuple[Decimal | None, str]:
        """All-in cost of one basket unit at the best asks (pre-screen only)."""
        total = ZERO
        for market, side in rel.legs:
            if not market.tradable:
                return None, f"{market.ticker} not active"
            book = ctx.books.get(market.ticker)
            if book is None:
                return None, f"{market.ticker}: no valid book"
            age = book.age_s(ctx.now)
            if age is None or age > self.settings.max_book_age_s:
                return None, f"{market.ticker}: book stale"
            ask = book.best_ask(side)
            if ask is None:
                return None, f"{market.ticker} {side}: no ask"
            fee = fee_per_contract_estimate(
                ask[0], ONE, self.fees.taker_rate, market.fee_multiplier
            )
            total += ask[0] + fee
        return total, ""

    def scan(self, ctx: ScanContext) -> list[Opportunity]:
        s = self.settings
        out: list[Opportunity] = []
        seen: set[str] = set()
        for group in self.groups(ctx):
            if not group.markets:
                continue
            rels, why = self.relationships(group)
            if not rels:
                out.append(self._note(ctx, group, DecisionStatus.UNSUPPORTED, "; ".join(why)))
                continue
            best: tuple[Decimal, Relationship] | None = None
            reasons: list[str] = []
            for rel in rels:
                cost, problem = self._unit_cost(ctx, rel)
                if cost is None:
                    reasons.append(problem)
                    continue
                edge = rel.guaranteed_payout - cost
                if best is None or edge > best[0]:
                    best = (edge, rel)
            if best is None:
                out.append(self._note(ctx, group, DecisionStatus.FILTERED, reasons[0]))
                continue
            edge, rel = best
            key = self._key(rel)
            if edge < s.min_edge:
                self._persistence.pop(key, None)
                out.append(
                    self._note(
                        ctx,
                        group,
                        DecisionStatus.FILTERED,
                        f"best {rel.kind} basket edge {edge:+.4f}/unit at 1 unit < {s.min_edge}",
                    )
                )
                continue
            seen.add(key)
            first, count = self._persistence.get(key, (ctx.now, 0))
            count += 1
            self._persistence[key] = (first, count)
            opp = self._basket(ctx, group, rel, edge)
            if count < s.min_persistence_scans or ctx.now - first < s.min_persistence_s:
                opp.reject(
                    DecisionStatus.FILTERED,
                    f"awaiting persistence ({count}/{s.min_persistence_scans} scans, "
                    f"{ctx.now - first:.1f}/{s.min_persistence_s}s)",
                )
            out.append(opp)
        for key in list(self._persistence):
            if key not in seen:
                del self._persistence[key]
        return out

    @staticmethod
    def _key(rel: Relationship) -> str:
        return "+".join(f"{m.ticker}:{side}" for m, side in rel.legs)

    def _basket(
        self, ctx: ScanContext, group: RelatedGroup, rel: Relationship, edge: Decimal
    ) -> Opportunity:
        legs = []
        for market, side in rel.legs:
            book = ctx.books[market.ticker]
            bid, ask = book.best_bid(side), book.best_ask(side)
            legs.append(
                Leg(
                    market_ticker=market.ticker,
                    event_ticker=group.event_ticker,
                    side=side,
                    fee_multiplier=market.fee_multiplier,
                    price_ranges=list(market.price_ranges),
                    best_bid=bid[0] if bid else None,
                    best_ask=ask[0] if ask else None,
                    ask_size=ask[1] if ask else None,
                    limit_price=MAX_LIMIT,
                )
            )
        ages = [ctx.book_age(m.ticker) for m, _ in rel.legs]
        return Opportunity(
            strategy=self.name,
            strategy_version=self.version_tag,
            kind="basket",
            category=group.category,
            event_ticker=group.event_ticker,
            legs=legs,
            observed_ts=ctx.now,
            title=group.title,
            guaranteed_payout=rel.guaranteed_payout,
            relationship=f"{rel.kind}: {rel.note}",
            min_edge=self.settings.min_edge,
            edge_gated=True,
            requires_score=False,
            correlation_group=f"{group.category}:{group.series_ticker}",
            book_age_s=max((a for a in ages if a is not None), default=None),
            provenance={
                "relationship": rel.kind,
                "unit_edge_at_best": str(edge),
                "residual_risks": "legging; book changes between legs; void/fair-price settlement",
            },
            reason=f"{rel.kind}: unit cost {rel.guaranteed_payout - edge} vs payout "
            f">= {rel.guaranteed_payout}",
        )

    def _note(
        self, ctx: ScanContext, group: RelatedGroup, status: DecisionStatus, reason: str
    ) -> Opportunity:
        market = group.markets[0]
        return Opportunity(
            strategy=self.name,
            strategy_version=self.version_tag,
            kind="basket",
            category=group.category,
            event_ticker=group.event_ticker,
            legs=[Leg(market.ticker, group.event_ticker, "yes")],
            observed_ts=ctx.now,
            title=group.title,
            status=status,
            reasons=[reason],
        )
