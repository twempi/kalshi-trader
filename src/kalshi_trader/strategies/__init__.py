"""Trading strategies. Register new strategies in ``build_strategies``.

| name              | role                                   | default mode |
|-------------------|----------------------------------------|--------------|
| tennis_model      | A/C: statistical mispricing (Markov)   | paper        |
| cross_market      | B: verified multi-leg relationships    | paper        |
| late_match        | baseline 2: original late-match rules  | observe      |
| price_threshold   | baseline 1: favourite price band       | observe      |
| market_making     | D: research simulation (separate)      | observe      |

Baseline 3 (market-implied probability) is a benchmark in ``research.model_eval``, not a
trading strategy.
"""

from kalshi_trader.config import Settings
from kalshi_trader.strategies.base import MatchContext, ScanContext, Strategy
from kalshi_trader.strategies.cross_market import CrossMarketStrategy
from kalshi_trader.strategies.late_match import LateMatchFavoriteStrategy
from kalshi_trader.strategies.price_threshold import PriceThresholdStrategy
from kalshi_trader.strategies.stat_mispricing import TennisModelStrategy

__all__ = ["MatchContext", "ScanContext", "Strategy", "build_strategies"]


def build_strategies(settings: Settings) -> list[Strategy]:
    cfg, fees = settings.strategies, settings.fees
    candidates: list[Strategy] = [
        TennisModelStrategy(cfg.tennis_model, fees),
        CrossMarketStrategy(cfg.cross_market, fees),
        LateMatchFavoriteStrategy(cfg.late_match, fees),
        PriceThresholdStrategy(cfg.price_threshold, fees),
    ]
    return [s for s in candidates if s.mode != "off"]
