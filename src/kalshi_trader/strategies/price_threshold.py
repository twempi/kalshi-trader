"""Baseline 1: buy the favourite whenever its executable ask sits inside a price band.

A *control* strategy for the favourite-longshot hypothesis ("high-priced contracts are
underpriced"). It uses no model, so it is not gated on estimated edge; its expected value
is reported relative to the market mid (i.e. assuming the market is fair, it loses the
spread and fees). Any model strategy should beat it before the model is credited.
"""

from __future__ import annotations

from kalshi_trader.config import FeeSettings, PriceThresholdStrategySettings
from kalshi_trader.core.opportunity import DecisionStatus, Opportunity
from kalshi_trader.execution.fills import snap_price_down
from kalshi_trader.strategies.base import MatchContext, Strategy
from kalshi_trader.tennis.state import MatchStatus


class PriceThresholdStrategy(Strategy):
    name = "price_threshold"
    version = "2"
    requires_score = False
    edge_gated = False

    def __init__(self, settings: PriceThresholdStrategySettings, fees: FeeSettings) -> None:
        super().__init__(settings, fees)
        self.settings: PriceThresholdStrategySettings = settings

    def evaluate_match(self, ctx: MatchContext) -> list[Opportunity]:
        s = self.settings
        if s.require_live_match:
            if ctx.score is not None:
                if ctx.score.status != MatchStatus.LIVE:
                    return []
            elif (ctx.match.scheduled_start_ts or ctx.now) > ctx.now:
                return []
        best: Opportunity | None = None
        for player in (0, 1):
            for option in ctx.options_for(player):
                if not option.market.tradable:
                    continue
                ask = option.book.best_ask(option.side)
                spread = option.book.spread(option.side)
                if ask is None or spread is None or spread > s.max_spread:
                    continue
                price, _size = ask
                if not (s.min_price <= price <= s.max_price):
                    continue
                limit = snap_price_down(price + s.limit_slippage, option.market.price_ranges)
                opp = self.directional(
                    ctx,
                    option,
                    fair=None,
                    limit_price=max(limit, price),
                    reason=f"ask {price} inside [{s.min_price}, {s.max_price}]",
                )
                if best is None or price > (best.primary.best_ask or 0):
                    best = opp
        if best is None:
            return [
                self.note(
                    ctx,
                    DecisionStatus.FILTERED,
                    f"no ask in [{s.min_price}, {s.max_price}] with spread <= {s.max_spread}",
                )
            ]
        return [best]
