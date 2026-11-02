"""Discovery of open events in arbitrary (non-tennis) series, for cross-market research.

Only series explicitly listed in ``strategies.cross_market.series_tickers`` are scanned.
Each event is fetched with its nested markets (``GET /events?with_nested_markets=true``),
and the fields needed to verify settlement relationships are kept: the event's
``mutually_exclusive`` flag and each market's ``strike_type``/``floor_strike``/``cap_strike``.
No probability model exists for these markets; they are used only by relationship-based
strategies (and reported as unsupported by model-based ones).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient
from kalshi_trader.database.repository import Repository
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.discovery import TrackedMarket
from kalshi_trader.strategies.base import RelatedGroup

log = get_logger("discovery.generic")


def _ts(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt else None


@dataclass
class GenericDiscoveryResult:
    markets: dict[str, TrackedMarket]
    groups: dict[str, RelatedGroup]


class GenericDiscovery:
    def __init__(self, rest: KalshiRestClient, repo: Repository, series: list[str]) -> None:
        self._rest = rest
        self._repo = repo
        self._series = series
        self._fee_multipliers: dict[str, Decimal] = {}

    async def _fee_multiplier(self, series: str) -> Decimal:
        if series not in self._fee_multipliers:
            try:
                info = await self._rest.get_series(series)
                self._fee_multipliers[series] = info.fee_multiplier
                self._repo.upsert_series(
                    [
                        {
                            "ticker": info.ticker,
                            "title": info.title,
                            "fee_type": info.fee_type,
                            "fee_multiplier": info.fee_multiplier,
                        }
                    ]
                )
            except KalshiAPIError as exc:
                log.warning("series_lookup_failed", series=series, error=str(exc)[:200])
                self._fee_multipliers[series] = Decimal("1")
        return self._fee_multipliers[series]

    async def refresh(self) -> GenericDiscoveryResult:
        markets: dict[str, TrackedMarket] = {}
        groups: dict[str, RelatedGroup] = {}
        event_rows, market_rows = [], []
        for series in self._series:
            try:
                events = await self._rest.get_events(series_ticker=series, status="open")
            except KalshiAPIError as exc:
                log.warning("generic_events_failed", series=series, error=str(exc)[:200])
                continue
            series_mult = await self._fee_multiplier(series)
            for event in events:
                mult = (
                    Decimal(str(event.fee_multiplier_override))
                    if event.fee_multiplier_override is not None
                    else series_mult
                )
                tracked = []
                for m in event.markets or []:
                    if m.market_type != "binary":
                        continue
                    tm = TrackedMarket(
                        ticker=m.ticker,
                        event_ticker=event.event_ticker,
                        series_ticker=series,
                        player_name=m.yes_sub_title,
                        competitor_id=None,
                        status=m.status,
                        result=m.result,
                        settlement_value=m.settlement_value_dollars,
                        fee_multiplier=mult,
                        price_ranges=list(m.price_ranges),
                        exchange_index=m.exchange_index,
                        close_ts=_ts(m.close_time),
                        occurrence_ts=_ts(m.occurrence_datetime),
                        title=m.title,
                        strike_type=m.strike_type,
                        floor_strike=m.floor_strike,
                        cap_strike=m.cap_strike,
                        category="generic",
                    )
                    tracked.append(tm)
                    markets[m.ticker] = tm
                    market_rows.append(
                        {
                            "ticker": m.ticker,
                            "event_ticker": event.event_ticker,
                            "series_ticker": series,
                            "player_name": m.yes_sub_title,
                            "status": m.status,
                            "result": m.result,
                            "open_ts": _ts(m.open_time),
                            "close_ts": _ts(m.close_time),
                            "exchange_index": m.exchange_index,
                            "price_ranges_json": json.dumps(
                                [r.model_dump(mode="json") for r in m.price_ranges]
                            ),
                            "fee_multiplier": mult,
                            "title": m.title,
                            "strike_type": m.strike_type,
                            "floor_strike": m.floor_strike,
                            "cap_strike": m.cap_strike,
                        }
                    )
                groups[event.event_ticker] = RelatedGroup(
                    event_ticker=event.event_ticker,
                    series_ticker=series,
                    title=event.title,
                    mutually_exclusive=event.mutually_exclusive,
                    markets=tracked,
                    category="generic",
                )
                event_rows.append(
                    {
                        "event_ticker": event.event_ticker,
                        "series_ticker": series,
                        "title": event.title,
                        "sub_title": event.sub_title,
                        "mutually_exclusive": event.mutually_exclusive,
                        "category": "generic",
                    }
                )
        self._repo.upsert_events(event_rows)
        self._repo.upsert_markets(market_rows)
        log.info("generic_discovery_complete", events=len(groups), markets=len(markets))
        return GenericDiscoveryResult(markets, groups)
