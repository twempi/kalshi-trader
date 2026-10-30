"""Read-only async client for the Kalshi Trade API v2 REST endpoints.

Safety: this client can only issue GET requests. There are deliberately no methods for
order placement, amendment or cancellation, and ``_request`` refuses any other HTTP verb.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.api.ratelimit import AsyncTokenBucket
from kalshi_trader.api.schemas import (
    Candlestick,
    Event,
    ExchangeStatus,
    HistoricalCutoff,
    LiveData,
    Market,
    Milestone,
    OrderbookFp,
    Series,
    Trade,
    validate_ticker,
)
from kalshi_trader.config import KalshiSettings
from kalshi_trader.logging_setup import get_logger

log = get_logger("kalshi.rest")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_PAGES = 200


class KalshiAPIError(Exception):
    def __init__(self, status: int | None, message: str, path: str) -> None:
        super().__init__(f"Kalshi API error {status} on {path}: {message}")
        self.status = status
        self.path = path


class ReadOnlyViolation(RuntimeError):
    """Raised if anything tries to send a non-GET request through this client."""


@dataclass
class RestHealth:
    requests: int = 0
    failures: int = 0
    retries: int = 0
    rate_limited: int = 0
    consecutive_failures: int = 0
    last_success_at: float | None = None
    last_error_at: float | None = None
    last_error: str | None = None
    latency_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        recent = self.latency_ms[-100:]
        return {
            "requests": self.requests,
            "failures": self.failures,
            "retries": self.retries,
            "rate_limited": self.rate_limited,
            "consecutive_failures": self.consecutive_failures,
            "last_success_at": self.last_success_at,
            "last_error_at": self.last_error_at,
            "last_error": self.last_error,
            "avg_latency_ms": round(sum(recent) / len(recent), 1) if recent else None,
        }


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class KalshiRestClient:
    def __init__(
        self,
        settings: KalshiSettings,
        signer: KalshiSigner | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 30.0,
    ) -> None:
        self._settings = settings
        self._signer = signer
        self._base_url = settings.rest_url
        self._api_prefix = urlsplit(self._base_url).path.rstrip("/")  # e.g. /trade-api/v2
        self._bucket = AsyncTokenBucket(settings.max_requests_per_second, float(settings.burst))
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self.health = RestHealth()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(settings.request_timeout_s),
            headers={
                "Accept": "application/json",
                "User-Agent": "kalshi-trader/0.1 (research; paper-trading only)",
            },
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self) -> KalshiRestClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def authenticated(self) -> bool:
        return self._signer is not None

    # ------------------------------------------------------------------ core
    async def _request(
        self, method: str, path: str, params: Any = None, *, auth: bool = False
    ) -> dict[str, Any]:
        if method.upper() != "GET":
            raise ReadOnlyViolation(
                f"Refusing {method} {path}: this application never modifies exchange state"
            )
        if auth and self._signer is None:
            raise KalshiAPIError(None, "endpoint requires API credentials", path)

        attempt = 0
        while True:
            await self._bucket.acquire()
            headers = self._signer.headers("GET", self._api_prefix + path) if self._signer else {}
            started = time.monotonic()
            self.health.requests += 1
            try:
                response = await self._client.get(path, params=params, headers=headers)
            except httpx.TransportError as exc:
                error: Exception = exc
                retryable = True
                status = None
            else:
                status = response.status_code
                if status == 200:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        error, retryable = (
                            KalshiAPIError(status, f"invalid JSON: {exc}", path),
                            True,
                        )
                    else:
                        if not isinstance(payload, dict):
                            error = KalshiAPIError(status, "unexpected JSON shape", path)
                            retryable = False
                        else:
                            self._record_success(started)
                            self._bucket.recover()
                            return payload
                else:
                    if status == 429:
                        self.health.rate_limited += 1
                        self._bucket.throttle()
                    retryable = status in RETRYABLE_STATUS
                    error = KalshiAPIError(status, response.text[:300], path)

            self._record_failure(error)
            if not retryable or attempt >= self._settings.max_retries:
                log.warning(
                    "kalshi_request_failed",
                    path=path,
                    status=status,
                    attempt=attempt,
                    error=str(error)[:300],
                )
                if isinstance(error, KalshiAPIError):
                    raise error
                raise KalshiAPIError(status, f"{type(error).__name__}: {error}", path) from error
            delay = min(self._backoff_max_s, self._backoff_base_s * (2**attempt))
            delay *= random.uniform(0.5, 1.5)
            attempt += 1
            self.health.retries += 1
            log.info(
                "kalshi_request_retry",
                path=path,
                status=status,
                attempt=attempt,
                delay_s=round(delay, 2),
            )
            await asyncio.sleep(delay)

    def _record_success(self, started: float) -> None:
        self.health.consecutive_failures = 0
        self.health.last_success_at = time.time()
        self.health.latency_ms.append((time.monotonic() - started) * 1000)
        if len(self.health.latency_ms) > 500:
            del self.health.latency_ms[:250]

    def _record_failure(self, error: Exception) -> None:
        self.health.failures += 1
        self.health.consecutive_failures += 1
        self.health.last_error_at = time.time()
        self.health.last_error = f"{type(error).__name__}: {str(error)[:200]}"

    async def _paginate(
        self, path: str, params: dict[str, Any], key: str, *, auth: bool = False
    ) -> AsyncIterator[dict[str, Any]]:
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            page_params = {k: v for k, v in params.items() if v is not None}
            if cursor:
                page_params["cursor"] = cursor
            payload = await self._request("GET", path, page_params, auth=auth)
            for item in payload.get(key) or []:
                yield item
            cursor = payload.get("cursor") or None
            if not cursor:
                return
        log.warning("pagination_limit_reached", path=path, max_pages=MAX_PAGES)

    @staticmethod
    def _parse_many(model: type, items: Iterable[dict[str, Any]], kind: str) -> list[Any]:
        parsed = []
        for item in items:
            try:
                parsed.append(model.model_validate(item))
            except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
                log.warning(
                    "invalid_api_record_skipped",
                    kind=kind,
                    error=str(exc)[:300],
                    ticker=item.get("ticker") if isinstance(item, dict) else None,
                )
        return parsed

    # ------------------------------------------------------------- exchange
    async def get_exchange_status(self) -> ExchangeStatus:
        return ExchangeStatus.model_validate(await self._request("GET", "/exchange/status"))

    async def get_exchange_schedule(self) -> dict[str, Any]:
        return await self._request("GET", "/exchange/schedule")

    # --------------------------------------------------------------- series
    async def get_series_list(
        self, category: str | None = None, tags: str | None = None
    ) -> list[Series]:
        params = {k: v for k, v in {"category": category, "tags": tags}.items() if v}
        payload = await self._request("GET", "/series", params)
        return self._parse_many(Series, payload.get("series") or [], "series")

    async def get_series(self, series_ticker: str) -> Series:
        validate_ticker(series_ticker)
        payload = await self._request("GET", f"/series/{quote(series_ticker)}")
        return Series.model_validate(payload["series"])

    # -------------------------------------------------------------- markets
    async def get_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        tickers: Sequence[str] | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        min_settled_ts: int | None = None,
        limit: int = 1000,
    ) -> list[Market]:
        params = {
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
            "status": status,
            "min_close_ts": min_close_ts,
            "max_close_ts": max_close_ts,
            "min_settled_ts": min_settled_ts,
            "limit": limit,
        }
        if not tickers:
            return await self._get_all_markets(params)
        results: list[Market] = []
        for chunk in _chunks([validate_ticker(t) for t in tickers], 50):
            results.extend(await self._get_all_markets({**params, "tickers": ",".join(chunk)}))
        return results

    async def _get_all_markets(self, params: dict[str, Any]) -> list[Market]:
        items = [item async for item in self._paginate("/markets", params, "markets")]
        return self._parse_many(Market, items, "market")

    async def get_market(self, ticker: str) -> Market:
        validate_ticker(ticker)
        payload = await self._request("GET", f"/markets/{quote(ticker)}")
        return Market.model_validate(payload["market"])

    async def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> Event:
        validate_ticker(event_ticker)
        payload = await self._request(
            "GET",
            f"/events/{quote(event_ticker)}",
            {"with_nested_markets": str(with_nested_markets).lower()},
        )
        event = Event.model_validate(payload["event"])
        if event.markets is None and payload.get("markets"):
            event.markets = self._parse_many(Market, payload["markets"], "market")
        return event

    async def get_orderbook(self, ticker: str, depth: int = 0) -> OrderbookFp:
        validate_ticker(ticker)
        params = {"depth": depth} if depth else None
        payload = await self._request("GET", f"/markets/{quote(ticker)}/orderbook", params)
        return OrderbookFp.model_validate(payload.get("orderbook_fp") or {})

    async def get_orderbooks(self, tickers: Sequence[str]) -> dict[str, OrderbookFp]:
        books: dict[str, OrderbookFp] = {}
        for chunk in _chunks([validate_ticker(t) for t in tickers], 100):
            payload = await self._request(
                "GET", "/markets/orderbooks", [("tickers", t) for t in chunk]
            )
            for item in payload.get("orderbooks") or []:
                ticker = item.get("ticker")
                if ticker not in chunk:
                    log.warning("unexpected_orderbook_ticker", ticker=str(ticker)[:100])
                    continue
                try:
                    books[ticker] = OrderbookFp.model_validate(item.get("orderbook_fp") or {})
                except ValueError as exc:
                    log.warning("invalid_orderbook_skipped", ticker=ticker, error=str(exc)[:300])
        return books

    async def get_trades(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        historical: bool = False,
    ) -> list[Trade]:
        if ticker:
            validate_ticker(ticker)
        path = "/historical/trades" if historical else "/markets/trades"
        params = {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts, "limit": 1000}
        items = [item async for item in self._paginate(path, params, "trades")]
        return self._parse_many(Trade, items, "trade")

    async def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
        historical: bool = False,
    ) -> list[Candlestick]:
        validate_ticker(series_ticker)
        validate_ticker(ticker)
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60 or 1440 minutes")
        if historical:
            path = f"/historical/markets/{quote(ticker)}/candlesticks"
        else:
            path = f"/series/{quote(series_ticker)}/markets/{quote(ticker)}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        payload = await self._request("GET", path, params)
        return self._parse_many(Candlestick, payload.get("candlesticks") or [], "candlestick")

    # ------------------------------------------------------------ historical
    async def get_historical_cutoff(self) -> HistoricalCutoff:
        return HistoricalCutoff.model_validate(await self._request("GET", "/historical/cutoff"))

    async def get_historical_markets(
        self, *, series_ticker: str | None = None, event_ticker: str | None = None
    ) -> list[Market]:
        params = {"series_ticker": series_ticker, "event_ticker": event_ticker, "limit": 1000}
        items = [item async for item in self._paginate("/historical/markets", params, "markets")]
        return self._parse_many(Market, items, "market")

    # ------------------------------------------------------- milestones/live
    async def get_milestones(
        self,
        *,
        milestone_type: str | None = None,
        category: str | None = None,
        minimum_start_date: str | None = None,
        related_event_ticker: str | None = None,
    ) -> list[Milestone]:
        params = {
            "limit": 500,
            "type": milestone_type,
            "category": category,
            "minimum_start_date": minimum_start_date,
            "related_event_ticker": related_event_ticker,
        }
        items = [item async for item in self._paginate("/milestones", params, "milestones")]
        return self._parse_many(Milestone, items, "milestone")

    async def get_live_data_batch(self, milestone_ids: Sequence[str]) -> list[LiveData]:
        results: list[LiveData] = []
        for chunk in _chunks(list(milestone_ids), 100):
            payload = await self._request(
                "GET", "/live_data/batch", [("milestone_ids", mid) for mid in chunk]
            )
            results.extend(self._parse_many(LiveData, payload.get("live_datas") or [], "live"))
        return results

    # --------------------------------------------------------- authenticated
    async def get_api_keys(self) -> list[dict[str, Any]]:
        """Requires credentials. Used only to warn if the configured key can trade."""
        payload = await self._request("GET", "/api_keys", auth=True)
        return list(payload.get("api_keys") or [])

    async def get_balance(self) -> dict[str, Any]:
        """Requires credentials. Read-only; used to verify authentication works."""
        return await self._request("GET", "/portfolio/balance", auth=True)
