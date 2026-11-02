"""Baseline 2: the original late-match favourite strategy, rules preserved unchanged.

Hypothesis under test: near the end of a match, the contract on the player who is about to
win is sometimes priced below their true win probability. This strategy does NOT assume
that is true. It proposes a trade only when *all* of these hold:

* the live score is valid, not delayed, and the match format is known;
* the backed player needs few games to win (``max_games_from_victory``), optionally only in
  the deciding set;
* the model's point probability is high (``min_model_prob``);
* the executable ask is inside a price band and the spread is narrow;
* the conservative probability (lower credible bound after assuming the score feed may be
  ``latency_points`` behind) beats the all-in cost by ``min_edge`` per contract.

It is kept as a fixed baseline so changes in the general tennis-model strategy can be
compared against it fairly. The final edge check now happens at the actual simulated fill
size in the shared evaluator (previously it used a hypothetical 10-contract order).
"""

from __future__ import annotations

from decimal import Decimal

from kalshi_trader.config import FeeSettings, LateMatchStrategySettings
from kalshi_trader.core.opportunity import DecisionStatus, FairValue, Opportunity
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.fills import snap_price_down
from kalshi_trader.models.ev import max_price_for_edge
from kalshi_trader.strategies.base import MatchContext, Strategy

# Per-contract fee estimate used only to set the limit-price cap; the evaluator re-checks
# the edge with the exact fee for the real order size.
LIMIT_FEE_HINT = Decimal("100")
MODEL_SOURCE = "tennis_markov"


class LateMatchFavoriteStrategy(Strategy):
    name = "late_match"
    version = "2"
    requires_score = True

    def __init__(self, settings: LateMatchStrategySettings, fees: FeeSettings) -> None:
        super().__init__(settings, fees)
        self.settings: LateMatchStrategySettings = settings

    def evaluate_match(self, ctx: MatchContext) -> list[Opportunity]:
        s = self.settings
        score, estimate = ctx.score, ctx.estimate
        if score is None or not score.is_live:
            return []  # not live: nothing to say for an in-play strategy
        if estimate is None or not score.fmt.known:
            return [self.note(ctx, DecisionStatus.FILTERED, "no model estimate / unknown format")]
        if score.source_delayed:
            return [self.note(ctx, DecisionStatus.FILTERED, "score source flagged delayed")]
        state = score.score
        if state is None:
            return []
        leader = 0 if estimate.p1_point >= 0.5 else 1
        prob = estimate.for_player(leader)
        if prob.point < s.min_model_prob:
            return [
                self.note(
                    ctx,
                    DecisionStatus.FILTERED,
                    f"leader model prob {prob.point:.3f} < {s.min_model_prob}",
                )
            ]
        if s.require_deciding_set and not state.is_final_set:
            return [self.note(ctx, DecisionStatus.FILTERED, "not the deciding set")]
        games_left = state.games_from_victory(leader)
        if games_left is None or games_left > s.max_games_from_victory:
            return [
                self.note(
                    ctx,
                    DecisionStatus.FILTERED,
                    f"leader {games_left} games from victory > {s.max_games_from_victory}",
                )
            ]

        fair = FairValue(
            p=prob.point,
            p_low=prob.robust_low,
            p_high=prob.point if leader == 0 else 1 - estimate.p1_low,
            source=MODEL_SOURCE,
            as_of_ts=score.received_ts,
        )
        best: Opportunity | None = None
        best_ev = float("-inf")
        why: list[str] = []
        for option in ctx.options_for(leader):
            label = f"{option.side.upper()} {option.market.ticker}"
            if not option.market.tradable:
                why.append(f"{label}: market not active")
                continue
            ask = option.book.best_ask(option.side)
            spread = option.book.spread(option.side)
            if ask is None or spread is None or spread > s.max_spread:
                why.append(f"{label}: spread {spread} > {s.max_spread}")
                continue
            price, _size = ask
            if not (s.min_price <= price <= s.max_price):
                why.append(f"{label}: ask {price} outside [{s.min_price}, {s.max_price}]")
                continue
            mult = option.market.fee_multiplier
            fee_pc = fee_per_contract_estimate(price, LIMIT_FEE_HINT, self.fees.taker_rate, mult)
            ev = prob.robust_low - float(price + fee_pc)
            if ev < float(s.min_edge):
                why.append(
                    f"{label}: robust p {prob.robust_low:.3f} - all-in {price + fee_pc} < edge"
                )
                continue
            cap = max_price_for_edge(
                prob.robust_low,
                s.min_edge,
                fee_rate=self.fees.taker_rate,
                fee_multiplier=mult,
                contracts_hint=LIMIT_FEE_HINT,
                ceiling=min(Decimal("0.99"), price + s.limit_slippage),
            )
            if cap is None or cap < price:
                why.append(f"{label}: no limit price keeps the edge")
                continue
            limit = snap_price_down(min(price + s.limit_slippage, cap), option.market.price_ranges)
            if limit < price:
                why.append(f"{label}: limit below best ask after tick snapping")
                continue
            opp = self.directional(
                ctx,
                option,
                fair=fair,
                limit_price=limit,
                stage="late",
                reason=f"{games_left} game(s) from victory; robust p={prob.robust_low:.3f} "
                f"vs ask {price}",
            )
            if ev > best_ev:
                best, best_ev = opp, ev
        if best is None:
            reason = "; ".join(why) or "no executable option for the leader"
            return [self.note(ctx, DecisionStatus.FILTERED, reason)]
        return [best]
