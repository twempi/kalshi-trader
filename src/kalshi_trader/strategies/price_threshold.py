"""Score-agnostic baseline: back the favourite whenever its ask sits inside a price band.

This is a *control* strategy. It tests the pure "high-priced contracts are underpriced"
(favourite-longshot) hypothesis without any model; the late-match strategy should beat it
before its model is credited with anything. Sizing is flat (no probability estimate).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from kalshi_trader.config import FeeSettings, PriceThresholdStrategySettings
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.fills import snap_price_down
from kalshi_trader.strategies.base import MatchContext, Signal, Strategy
from kalshi_trader.tennis.state import MatchStatus

SIZE_HINT = Decimal("10")


class PriceThresholdStrategy(Strategy):
    name = "price_threshold"
    requires_score = False

    def __init__(self, settings: PriceThresholdStrategySettings, fees: FeeSettings) -> None:
        self.settings = settings
        self.fees = fees

    def evaluate(self, ctx: MatchContext) -> list[Signal]:
        s = self.settings
        if s.require_live_match:
            if ctx.score is not None:
                if ctx.score.status != MatchStatus.LIVE:
                    return []
            elif (ctx.match.scheduled_start_ts or ctx.now) > ctx.now:
                return []
        best: Signal | None = None
        for player in (0, 1):
            for option in ctx.options_for(player):
                if not option.market.tradable:
                    continue
                ask = option.book.best_ask(option.side)
                spread = option.book.spread(option.side)
                if ask is None or spread is None or spread > s.max_spread:
                    continue
                price, size = ask
                if not (s.min_price <= price <= s.max_price):
                    continue
                limit = snap_price_down(price + s.limit_slippage, option.market.price_ranges)
                fee_pc = fee_per_contract_estimate(
                    price, SIZE_HINT, self.fees.taker_rate, option.market.fee_multiplier
                )
                mid = option.book.mid(option.side)
                signal = Signal(
                    strategy=self.name,
                    market_ticker=option.market.ticker,
                    event_ticker=ctx.match.event_ticker,
                    side=option.side,
                    player=player,
                    player_name=ctx.match.player_names[player],
                    best_ask=price,
                    ask_size=size,
                    limit_price=max(limit, price),
                    fee_multiplier=option.market.fee_multiplier,
                    p_point=None,
                    p_low=None,
                    p_conservative=None,
                    market_prob=float(mid) if mid is not None else None,
                    # Break-even EV relative to the market's own mid (no model).
                    ev_per_contract=(float(mid) - float(price + fee_pc)) if mid else None,
                    requires_score=False,
                    score_text=ctx.score.score_text() if ctx.score else "",
                    reason=f"ask {price} inside [{s.min_price}, {s.max_price}]",
                )
                if best is None or signal.best_ask > best.best_ask:
                    best = signal
        return [best] if best else []

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "requires_score": False,
            "params": {k: str(v) for k, v in self.settings.model_dump().items()},
        }
