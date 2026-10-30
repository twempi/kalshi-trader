"""Trading strategies. Register new strategies in ``build_strategies``."""

from kalshi_trader.config import Settings
from kalshi_trader.strategies.base import MatchContext, Signal, Strategy
from kalshi_trader.strategies.late_match import LateMatchFavoriteStrategy
from kalshi_trader.strategies.price_threshold import PriceThresholdStrategy

__all__ = ["MatchContext", "Signal", "Strategy", "build_strategies"]


def build_strategies(settings: Settings) -> list[Strategy]:
    strategies: list[Strategy] = []
    if settings.strategies.late_match.enabled:
        strategies.append(LateMatchFavoriteStrategy(settings.strategies.late_match, settings.fees))
    if settings.strategies.price_threshold.enabled:
        strategies.append(
            PriceThresholdStrategy(settings.strategies.price_threshold, settings.fees)
        )
    return strategies
