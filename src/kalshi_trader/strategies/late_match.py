"""Late-match favourite strategy.

Hypothesis under test: near the end of a match, the contract on the player who is about to
win is sometimes priced below their true win probability. This strategy does NOT assume
that is true. It only proposes a trade when *all* of these hold:

* the live score is valid, fresh and the match format is known;
* the backed player needs few games to win (``max_games_from_victory``), optionally only in
  the deciding set;
* the model's point probability is high (``min_model_prob``);
* the executable ask is inside a price band, the spread is narrow;
* the conservative probability (lower credible bound after assuming the score feed may be
  ``latency_points`` behind the market) beats the all-in cost by ``min_edge`` per contract.

The IOC limit price is capped so every contract filled still clears ``min_edge``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kalshi_trader.config import FeeSettings, LateMatchStrategySettings
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.fills import snap_price_down
from kalshi_trader.models.ev import max_price_for_edge
from kalshi_trader.strategies.base import MatchContext, Signal, Strategy

SIZE_HINT = Decimal("10")


class LateMatchFavoriteStrategy(Strategy):
    name = "late_match"
    requires_score = True

    def __init__(self, settings: LateMatchStrategySettings, fees: FeeSettings) -> None:
        self.settings = settings
        self.fees = fees

    def evaluate(self, ctx: MatchContext) -> list[Signal]:
        s = self.settings
        score, estimate = ctx.score, ctx.estimate
        if score is None or estimate is None or not score.is_live or not score.fmt.known:
            return []
        if score.source_delayed:
            return []
        state = score.score
        if state is None:
            return []
        leader = 0 if estimate.p1_point >= 0.5 else 1
        prob = estimate.for_player(leader)
        if prob.point < s.min_model_prob:
            return []
        if s.require_deciding_set and not state.is_final_set:
            return []
        games_left = state.games_from_victory(leader)
        if games_left is None or games_left > s.max_games_from_victory:
            return []

        best: Signal | None = None
        for option in ctx.options_for(leader):
            if not option.market.tradable:
                continue
            ask = option.book.best_ask(option.side)
            spread = option.book.spread(option.side)
            if ask is None or spread is None or spread > s.max_spread:
                continue
            price, size = ask
            if not (s.min_price <= price <= s.max_price):
                continue
            mult = option.market.fee_multiplier
            fee_pc = fee_per_contract_estimate(price, SIZE_HINT, self.fees.taker_rate, mult)
            ev = prob.robust_low - float(price + fee_pc)
            if ev < float(s.min_edge):
                continue
            cap = max_price_for_edge(
                prob.robust_low,
                s.min_edge,
                fee_rate=self.fees.taker_rate,
                fee_multiplier=mult,
                contracts_hint=SIZE_HINT,
                ceiling=min(Decimal("0.99"), price + s.limit_slippage),
            )
            if cap is None or cap < price:
                continue
            limit = snap_price_down(min(price + s.limit_slippage, cap), option.market.price_ranges)
            if limit < price:
                continue
            mid = option.book.mid(option.side)
            signal = Signal(
                strategy=self.name,
                market_ticker=option.market.ticker,
                event_ticker=ctx.match.event_ticker,
                side=option.side,
                player=leader,
                player_name=ctx.match.player_names[leader],
                best_ask=price,
                ask_size=size,
                limit_price=limit,
                fee_multiplier=mult,
                p_point=prob.point,
                p_low=prob.low,
                p_conservative=prob.robust_low,
                market_prob=float(mid) if mid is not None else None,
                ev_per_contract=ev,
                requires_score=True,
                score_text=score.score_text(),
                reason=f"{games_left} game(s) from victory; robust p={prob.robust_low:.3f} "
                f"vs ask {price}",
            )
            if best is None or (signal.ev_per_contract or 0) > (best.ev_per_contract or 0):
                best = signal
        return [best] if best else []

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_score": True,
            "params": {k: str(v) for k, v in self.settings.model_dump().items()},
        }
