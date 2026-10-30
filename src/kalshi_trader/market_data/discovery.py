"""Discovery of live/upcoming tennis match-winner markets on Kalshi.

Each tennis match is a Kalshi *event* with two binary markets (one per player). Events are
linked to Kalshi *milestones* (type ``tennis_tournament_singles``) whose ids key the live
score feed and whose ``first/second_competitor_id`` fix player order. A market's
``custom_strike.tennis_competitor`` identifies which player it pays out on.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient
from kalshi_trader.api.schemas import Market, Milestone, PriceRange
from kalshi_trader.config import DiscoverySettings
from kalshi_trader.database.repository import Repository
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.tennis.providers.base import MatchRef
from kalshi_trader.tennis.state import MatchFormat, infer_format

log = get_logger("discovery")

TENNIS_MILESTONE_TYPE = "tennis_tournament_singles"
SERIES_TOUR_KEYS = {
    "KXATPMATCH": "ATP",
    "KXWTAMATCH": "WTA",
    "KXATPCHALLENGERMATCH": "ATP_CHALLENGER",
    "KXCHALLENGERMATCH": "ATP_CHALLENGER",
    "KXWTACHALLENGERMATCH": "WTA_CHALLENGER",
    "KXITFMATCH": "ITF_MEN",
    "KXITFWMATCH": "ITF_WOMEN",
}


def tour_key_for(series_ticker: str, tour: str | None, gender: str | None) -> str:
    if series_ticker in SERIES_TOUR_KEYS:
        return SERIES_TOUR_KEYS[series_ticker]
    tour_l, gender_l = (tour or "").lower(), (gender or "").lower()
    if "challenger" in tour_l:
        return "WTA_CHALLENGER" if gender_l == "women" else "ATP_CHALLENGER"
    if "itf" in tour_l:
        return "ITF_WOMEN" if gender_l == "women" else "ITF_MEN"
    if "wta" in tour_l or gender_l == "women":
        return "WTA"
    return "ATP"


def _ts(dt: datetime | None) -> float | None:
    return dt.timestamp() if dt else None


@dataclass
class TrackedMarket:
    ticker: str
    event_ticker: str
    series_ticker: str
    player_name: str
    competitor_id: str | None
    status: str
    result: str = ""
    settlement_value: Decimal | None = None
    fee_multiplier: Decimal = Decimal("1")
    price_ranges: list[PriceRange] = field(default_factory=list)
    exchange_index: int | None = None
    close_ts: float | None = None
    occurrence_ts: float | None = None
    pre_match_prob: float | None = None
    in_open_list: bool = True
    last_status_check: float = 0.0

    @property
    def tradable(self) -> bool:
        return self.status == "active"


@dataclass
class TrackedMatch:
    event_ticker: str
    series_ticker: str
    title: str
    competition: str | None
    milestone_id: str | None
    tour: str | None
    gender: str | None
    round_name: str | None
    tournament_name: str | None
    best_of: int | None
    scheduled_start_ts: float | None
    player_ids: tuple[str | None, str | None]
    market_tickers: tuple[str | None, str | None]
    player_names: tuple[str, str]
    tour_key: str

    def format_hint(self) -> MatchFormat:
        return infer_format(
            best_of=self.best_of,
            gender=self.gender,
            tournament_name=self.tournament_name,
            last_set_scoring_type=None,
        )

    def match_ref(self) -> MatchRef | None:
        if not self.milestone_id:
            return None
        return MatchRef(
            match_id=self.milestone_id,
            event_ticker=self.event_ticker,
            player_ids=self.player_ids,
            tournament_name=self.tournament_name,
            gender=self.gender,
            best_of=self.best_of,
        )

    def player_index(self, ticker: str) -> int | None:
        if ticker == self.market_tickers[0]:
            return 0
        if ticker == self.market_tickers[1]:
            return 1
        return None


@dataclass
class DiscoveryResult:
    matches: dict[str, TrackedMatch]
    markets: dict[str, TrackedMarket]


class TennisDiscovery:
    def __init__(
        self, rest: KalshiRestClient, repo: Repository, settings: DiscoverySettings
    ) -> None:
        self._rest = rest
        self._repo = repo
        self._settings = settings
        self._fee_multipliers: dict[str, Decimal] = {}
        self._series_loaded_at = 0.0
        self._event_meta: dict[str, tuple[str, str | None]] = {}  # ticker -> (title, competition)

    async def _load_series(self) -> None:
        if time.time() - self._series_loaded_at < 3600 and self._fee_multipliers:
            return
        try:
            series = await self._rest.get_series_list(category="Sports", tags="Tennis")
        except KalshiAPIError as exc:
            log.warning("series_load_failed", error=str(exc)[:200])
            return
        wanted = set(self._settings.series_tickers)
        rows = []
        for s in series:
            if s.ticker in wanted:
                self._fee_multipliers[s.ticker] = s.fee_multiplier
                rows.append(
                    {
                        "ticker": s.ticker,
                        "title": s.title,
                        "fee_type": s.fee_type,
                        "fee_multiplier": s.fee_multiplier,
                    }
                )
        self._repo.upsert_series(rows)
        self._series_loaded_at = time.time()

    async def _milestones_by_event(self, now: datetime) -> dict[str, Milestone]:
        since = now - timedelta(hours=self._settings.lookback_hours + 24)
        milestones = await self._rest.get_milestones(
            milestone_type=TENNIS_MILESTONE_TYPE,
            minimum_start_date=since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        by_event: dict[str, Milestone] = {}
        for m in milestones:
            main = m.details.get("main_game_event_ticker")
            for ticker in ([main] if main else []) + list(m.primary_event_tickers):
                if ticker and ticker not in by_event:
                    by_event[ticker] = m
        return by_event

    async def _event_metadata(self, event_ticker: str) -> tuple[str, str | None]:
        cached = self._event_meta.get(event_ticker)
        if cached is not None:
            return cached
        row = self._repo.get_event(event_ticker)
        if row is not None and row.competition:
            meta = (row.title, row.competition)
        else:
            try:
                event = await self._rest.get_event(event_ticker, with_nested_markets=False)
                meta = (event.title, event.competition)
            except KalshiAPIError as exc:
                log.warning(
                    "event_metadata_failed", event_ticker=event_ticker, error=str(exc)[:200]
                )
                return event_ticker, None
        self._event_meta[event_ticker] = meta
        return meta

    async def refresh(self) -> DiscoveryResult:
        now_dt = datetime.now(UTC)
        now = now_dt.timestamp()
        await self._load_series()
        window_start = now - self._settings.lookback_hours * 3600
        window_end = now + self._settings.lookahead_hours * 3600

        by_event: dict[str, list[Market]] = {}
        for series in self._settings.series_tickers:
            try:
                markets = await self._rest.get_markets(series_ticker=series, status="open")
            except KalshiAPIError as exc:
                log.warning("market_list_failed", series=series, error=str(exc)[:200])
                continue
            for m in markets:
                if m.market_type != "binary":
                    continue
                start = _ts(m.occurrence_datetime) or _ts(m.expected_expiration_time)
                if start is not None and not (window_start <= start <= window_end):
                    continue
                by_event.setdefault(m.event_ticker, []).append(m)

        try:
            milestones = await self._milestones_by_event(now_dt)
        except KalshiAPIError as exc:
            log.warning("milestones_failed", error=str(exc)[:200])
            milestones = {}

        matches: dict[str, TrackedMatch] = {}
        tracked: dict[str, TrackedMarket] = {}
        event_rows, market_rows = [], []
        for event_ticker, markets in by_event.items():
            series = event_ticker.split("-", 1)[0]
            ms = milestones.get(event_ticker)
            details = ms.details if ms else {}
            title, competition = await self._event_metadata(event_ticker)
            by_competitor = {m.competitor_id: m for m in markets if m.competitor_id}
            first, second = details.get("first_competitor_id"), details.get("second_competitor_id")
            if first in by_competitor and second in by_competitor:
                ids: tuple[str | None, str | None] = (first, second)
            else:  # no usable milestone: fall back to ticker order
                ordered = [m.competitor_id for m in sorted(markets, key=lambda m: m.ticker)]
                ids = (ordered[0] if ordered else None, ordered[1] if len(ordered) > 1 else None)
            tickers = tuple(by_competitor[c].ticker if c in by_competitor else None for c in ids)
            names = tuple(
                by_competitor[c].yes_sub_title if c in by_competitor else "?" for c in ids
            )
            best_of_raw = details.get("best_of")
            best_of = int(best_of_raw) if str(best_of_raw or "").isdigit() else None
            # Earliest known start: milestone and market times can disagree by ~30 min.
            candidates = [_ts(m.occurrence_datetime) for m in markets]
            candidates.append(_ts(ms.start_date) if ms else None)
            known = [ts for ts in candidates if ts is not None]
            start_ts = min(known) if known else now
            match = TrackedMatch(
                event_ticker=event_ticker,
                series_ticker=series,
                title=title,
                competition=competition,
                milestone_id=ms.id if ms else None,
                tour=details.get("tour"),
                gender=details.get("gender"),
                round_name=details.get("round"),
                tournament_name=details.get("tournament_name"),
                best_of=best_of,
                scheduled_start_ts=start_ts,
                player_ids=(ids[0], ids[1]),
                market_tickers=(tickers[0], tickers[1]),
                player_names=(names[0], names[1]),
                tour_key=tour_key_for(series, details.get("tour"), details.get("gender")),
            )
            matches[event_ticker] = match
            fee_mult = self._fee_multipliers.get(series, Decimal("1"))
            for m in markets:
                tracked[m.ticker] = TrackedMarket(
                    ticker=m.ticker,
                    event_ticker=event_ticker,
                    series_ticker=series,
                    player_name=m.yes_sub_title,
                    competitor_id=m.competitor_id,
                    status=m.status,
                    result=m.result,
                    settlement_value=m.settlement_value_dollars,
                    fee_multiplier=fee_mult,
                    price_ranges=list(m.price_ranges),
                    exchange_index=m.exchange_index,
                    close_ts=_ts(m.close_time),
                    occurrence_ts=_ts(m.occurrence_datetime),
                )
                market_rows.append(
                    {
                        "ticker": m.ticker,
                        "event_ticker": event_ticker,
                        "series_ticker": series,
                        "player_name": m.yes_sub_title,
                        "competitor_id": m.competitor_id,
                        "status": m.status,
                        "result": m.result,
                        "open_ts": _ts(m.open_time),
                        "close_ts": _ts(m.close_time),
                        "expected_expiration_ts": _ts(m.expected_expiration_time),
                        "occurrence_ts": _ts(m.occurrence_datetime),
                        "exchange_index": m.exchange_index,
                        "price_ranges_json": json.dumps(
                            [r.model_dump(mode="json") for r in m.price_ranges]
                        ),
                        "fee_multiplier": fee_mult,
                    }
                )
            event_rows.append(
                {
                    "event_ticker": event_ticker,
                    "series_ticker": series,
                    "title": title,
                    "competition": competition,
                    "milestone_id": match.milestone_id,
                    "tour": match.tour,
                    "gender": match.gender,
                    "round_name": match.round_name,
                    "tournament_name": match.tournament_name,
                    "best_of": best_of,
                    "scheduled_start_ts": start_ts,
                    "first_competitor_id": ids[0],
                    "second_competitor_id": ids[1],
                }
            )
        self._repo.upsert_events(event_rows)
        self._repo.upsert_markets(market_rows)
        log.info(
            "discovery_complete",
            matches=len(matches),
            markets=len(tracked),
            with_milestone=sum(1 for m in matches.values() if m.milestone_id),
        )
        return DiscoveryResult(matches=matches, markets=tracked)
