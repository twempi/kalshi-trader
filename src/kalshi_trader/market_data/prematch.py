"""Recover a match's pre-match market price from Kalshi's public 1-minute candles.

Used when a match is first seen after it started (e.g. the engine was restarted). Only
candles that closed at least ``gap_s`` before the earliest known scheduled start are used,
so in-play information cannot leak into the strength prior.
"""

from __future__ import annotations

from decimal import Decimal

from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient
from kalshi_trader.logging_setup import get_logger

log = get_logger("prematch")

MAX_PRE_MATCH_SPREAD = Decimal("0.10")


async def pre_match_mid_from_candles(
    rest: KalshiRestClient,
    series_ticker: str,
    ticker: str,
    scheduled_start_ts: float,
    gap_s: float = 600.0,
    lookback_s: float = 6 * 3600.0,
) -> tuple[float, float] | None:
    """(mid, candle end ts) of the last quoted candle ending >= gap_s before the start."""
    end = int(scheduled_start_ts - gap_s)
    try:
        candles = await rest.get_candlesticks(series_ticker, ticker, int(end - lookback_s), end, 1)
    except KalshiAPIError as exc:
        log.warning("pre_match_candles_failed", ticker=ticker, error=str(exc)[:200])
        return None
    for candle in sorted(candles, key=lambda c: c.end_period_ts, reverse=True):
        if candle.end_period_ts > end:
            continue
        bid = candle.yes_bid.close_dollars if candle.yes_bid else None
        ask = candle.yes_ask.close_dollars if candle.yes_ask else None
        if bid is None or ask is None or not (0 < bid < ask < 1):
            continue
        if ask - bid > MAX_PRE_MATCH_SPREAD:
            continue
        return float((bid + ask) / 2), float(candle.end_period_ts)
    return None
