"""Main asyncio engine: discovery, data feeds, scores, and the shared trading core.

Data flow::

    tennis discovery ─┐                       ┌─> WS orderbook subscriptions / REST polling
    generic discovery ┴─> tracked markets ────┤
    score provider ──> TennisTracker (freshness, regression guard, pre-match prior, model)
    every second: ScanContext ──> TradingCore.run_cycle (strategies -> evaluator -> risk ->
                  allocator -> PaperBroker; shadow ledgers; market-making research sim)
    settlement monitor ──> TradingCore.settle (main + shadow ledgers + MM sim)
    recorder ──> SQLite (market + score observations for replay)

The engine never places real orders: the REST client is GET-only, the WebSocket client only
manages subscriptions, and execution is the paper broker.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.api.rest import KalshiRestClient
from kalshi_trader.api.schemas import ExchangeStatus
from kalshi_trader.api.ws import KalshiWebSocketClient
from kalshi_trader.config import Settings
from kalshi_trader.core.opportunity import DecisionStatus, Opportunity
from kalshi_trader.core.pipeline import TradingCore
from kalshi_trader.core.tennis_tracker import TennisTracker
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.logging_setup import drain_recent_events, get_logger
from kalshi_trader.market_data.discovery import TennisDiscovery, TrackedMarket, TrackedMatch
from kalshi_trader.market_data.feeds import MarketDataHub
from kalshi_trader.market_data.generic import GenericDiscovery
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.market_data.prematch import pre_match_mid_from_candles
from kalshi_trader.market_data.recorder import Recorder
from kalshi_trader.models.calibration import calibrate
from kalshi_trader.models.estimator import (
    PlayerProbability,
    ProbabilityEstimate,
    WinProbabilityEstimator,
)
from kalshi_trader.strategies import ScanContext, build_strategies
from kalshi_trader.strategies.base import RelatedGroup
from kalshi_trader.strategies.market_making import MarketMakingSimulator
from kalshi_trader.strategies.stat_mispricing import match_stage
from kalshi_trader.tennis.providers.base import ProviderResult, ScoreProvider
from kalshi_trader.tennis.providers.kalshi_live import KalshiLiveDataProvider
from kalshi_trader.tennis.state import MatchStatus

log = get_logger("engine")

MAX_BACKFILLS_PER_CYCLE = 4
FINALIZED_GRACE_S = 600.0
EVIDENCE_REFRESH_S = 600.0
OPPORTUNITY_RETENTION_S = 14 * 86400.0
STATUS_PRIORITY = {
    DecisionStatus.FILLED: 0,
    DecisionStatus.PARTIAL: 1,
    DecisionStatus.APPROVED: 2,
    DecisionStatus.OBSERVED: 3,
    DecisionStatus.UNFILLED: 4,
    DecisionStatus.REJECTED: 5,
    DecisionStatus.CANDIDATE: 6,
    DecisionStatus.FILTERED: 7,
    DecisionStatus.UNSUPPORTED: 8,
}


def load_signer(settings: Settings) -> KalshiSigner | None:
    k = settings.kalshi
    if not k.has_credentials:
        return None
    assert k.api_key_id is not None and k.private_key_path is not None
    try:
        return KalshiSigner.from_file(k.api_key_id.get_secret_value(), k.private_key_path)
    except (OSError, ValueError, TypeError) as exc:
        log.error("kalshi_key_load_failed", path=str(k.private_key_path), error=str(exc)[:200])
        return None


@dataclass
class ComponentHealth:
    status: str = "starting"
    last_ok: float | None = None
    last_error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def ok(self, **details: Any) -> None:
        self.status, self.last_ok = "ok", time.time()
        self.details = dict(details)

    def fail(self, error: str, status: str = "error") -> None:
        self.status, self.last_error = status, error[:500]


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        record_only: bool = False,
        db: Database | None = None,
        rest: KalshiRestClient | None = None,
        provider: ScoreProvider | None = None,
    ) -> None:
        self.settings = settings
        self.record_only = record_only
        self.db = db or Database(settings.db_path)
        self.repo = Repository(self.db)
        self.signer = load_signer(settings)
        self.rest = rest or KalshiRestClient(settings.kalshi, self.signer)
        self.hub = MarketDataHub()
        md = settings.market_data
        self.recorder = Recorder(
            self.repo,
            md.recorded_depth_levels,
            md.snapshot_heartbeat_s,
            settings.tennis.snapshot_heartbeat_s,
            md.snapshot_min_interval_s,
        )
        self.discovery = TennisDiscovery(self.rest, self.repo, settings.discovery)
        cm = settings.strategies.cross_market
        self.generic_discovery = (
            GenericDiscovery(self.rest, self.repo, cm.series_tickers)
            if cm.enabled and cm.series_tickers
            else None
        )
        if provider is not None:
            self.provider: ScoreProvider | None = provider
        elif settings.tennis.provider == "kalshi_live_data":
            self.provider = KalshiLiveDataProvider(self.rest)
        else:
            self.provider = None

        self.markets: dict[str, TrackedMarket] = {}
        self.groups: dict[str, RelatedGroup] = {}
        self.estimator = WinProbabilityEstimator(settings.model)
        self.tracker = TennisTracker(
            self.estimator, settings.risk, self.markets, prediction_sink=self._record_prediction
        )
        self.strategies = build_strategies(settings)
        mm_settings = settings.strategies.market_making
        self.market_maker = (
            MarketMakingSimulator(mm_settings, settings.fees, self.repo)
            if mm_settings.enabled
            else None
        )
        self.core = TradingCore(
            settings,
            self.db,
            self.repo,
            self.strategies,
            data_kind="live",
            market_maker=self.market_maker,
            record_only=record_only,
        )
        self.broker = self.core.broker
        self.risk = self.core.risk
        self.ws: KalshiWebSocketClient | None = None
        if self.signer is not None and settings.kalshi.use_websocket:
            self.ws = KalshiWebSocketClient(
                settings.kalshi.websocket_url,
                self.signer,
                self.hub.handle_ws_message,
                self._on_ws_gap,
            )

        self.scores: dict[str, ProviderResult] = {}  # by event ticker (latest accepted)
        self._finalized_at: dict[str, float] = {}
        self._backfill_done: set[str] = set()
        self._last_evidence = 0.0
        self._last_generic = 0.0
        self.exchange_status: ExchangeStatus | None = None
        self.health: dict[str, ComponentHealth] = {
            name: ComponentHealth()
            for name in (
                "engine",
                "discovery",
                "kalshi_rest",
                "kalshi_ws",
                "tennis_scores",
                "exchange",
                "recorder",
                "strategies",
                "settlement",
            )
        }
        if self.ws is None:
            self.health["kalshi_ws"].status = "disabled"
            self.health["kalshi_ws"].details["reason"] = (
                "no API credentials (REST polling used)"
                if self.signer is None
                else "disabled in settings"
            )
        if self.provider is None:
            self.health["tennis_scores"].status = "disabled"
        self._stop = asyncio.Event()
        self.hub.add_book_listener(self._on_book)
        self.hub.add_lifecycle_listener(self._on_lifecycle)

    # Compatibility accessors (dashboard/CLI/tests).
    @property
    def matches(self) -> dict[str, TrackedMatch]:
        return self.tracker.matches

    # ============================================================ lifecycle
    def request_stop(self) -> None:
        self._stop.set()

    def _task_factories(self) -> dict[str, Callable[[], Awaitable[None]]]:
        s = self.settings
        factories: dict[str, Callable[[], Awaitable[None]]] = {
            "discovery": lambda: self._loop(
                "discovery", self._discovery_step, s.discovery.refresh_interval_s
            ),
            "exchange": lambda: self._loop("exchange", self._exchange_step, 30.0),
            "kalshi_rest": lambda: self._loop(
                "kalshi_rest", self._rest_books_step, s.market_data.rest_poll_interval_s
            ),
            "tennis_scores": lambda: self._loop(
                "tennis_scores", self._scores_step, s.tennis.poll_interval_s
            ),
            "settlement": lambda: self._loop(
                "settlement", self._status_step, s.market_data.status_refresh_interval_s
            ),
            "strategies": lambda: self._loop(
                "strategies", self._evaluate_step, s.strategies.evaluation_interval_s
            ),
            "housekeeping": lambda: self._loop("housekeeping", self._housekeeping_step, 2.0),
            "equity": lambda: self._loop("equity", self._equity_step, 60.0),
        }
        if self.ws is not None:
            ws = self.ws
            factories["kalshi_ws"] = lambda: ws.run(self._stop)
        return factories

    async def run(self, install_signal_handlers: bool = True) -> None:
        if install_signal_handlers:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(sig, self.request_stop)
        self.repo.record_event(
            "info",
            "engine",
            f"engine started (record_only={self.record_only}, "
            f"websocket={'on' if self.ws else 'off'}, "
            f"strategies={[f'{s.name}:{s.mode}' for s in self.strategies]})",
        )
        log.info(
            "engine_starting",
            record_only=self.record_only,
            websocket=bool(self.ws),
            strategies={s.name: s.mode for s in self.strategies},
        )
        self.health["engine"].ok(started_at=time.time(), record_only=self.record_only)
        factories = self._task_factories()
        tasks = {asyncio.create_task(factory()): name for name, factory in factories.items()}
        restarts: dict[str, int] = {}
        try:
            # Supervision: a background task that ends unexpectedly is logged and restarted
            # (each loop already survives ordinary step errors).
            while not self._stop.is_set():
                done, _ = await asyncio.wait(
                    set(tasks), timeout=1.0, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    name = tasks.pop(task)
                    if self._stop.is_set():
                        continue
                    exc = task.exception() if not task.cancelled() else None
                    restarts[name] = restarts.get(name, 0) + 1
                    message = f"task {name} ended unexpectedly ({exc!r}); restarting"
                    log.error("task_restarted", task=name, error=repr(exc)[:300])
                    self.repo.record_event("error", "engine", message[:500])
                    self.health["engine"].details["restarts"] = dict(restarts)
                    await asyncio.sleep(min(30.0, 2.0 ** min(restarts[name], 5)))
                    tasks[asyncio.create_task(factories[name]())] = name
        finally:
            log.info("engine_stopping")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown()

    async def _shutdown(self) -> None:
        with contextlib.suppress(Exception):
            self.recorder.flush()
        with contextlib.suppress(Exception):
            self.core.diagnostics.flush()
        with contextlib.suppress(Exception):
            await self._equity_step()
        self.health["engine"].status = "stopped"
        with contextlib.suppress(Exception):
            self._persist_health()
            self.repo.record_event("info", "engine", "engine stopped cleanly")
        await self.rest.close()
        log.info("engine_stopped")

    async def _loop(self, name: str, step: Any, interval: float) -> None:
        """Run `step` forever; errors are logged, counted and never kill the loop."""
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                await step()
                if name == "strategies":
                    self.risk.record_success()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {str(exc)[:300]}"
                log.error("loop_step_failed", loop=name, error=message)
                if name in self.health:
                    self.health[name].fail(message)
                if name in ("strategies", "settlement"):
                    self.risk.record_error(name, message)
            elapsed = time.monotonic() - started
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=max(0.05, interval - elapsed))

    # ============================================================ discovery
    async def _discovery_step(self) -> None:
        result = await self.discovery.refresh()
        now = time.time()
        for ticker, market in result.markets.items():
            existing = self.markets.get(ticker)
            if existing is None:
                row = self.repo.get_market(ticker)
                if row is not None and row.pre_match_prob is not None:
                    market.pre_match_prob = row.pre_match_prob
                    market.pre_match_prob_ts = row.pre_match_prob_ts
                self.markets[ticker] = market
            else:
                existing.status = market.status
                existing.fee_multiplier = market.fee_multiplier
                existing.price_ranges = market.price_ranges
                existing.close_ts = market.close_ts
                existing.in_open_list = True
        for ticker, market in self.markets.items():
            if market.category == "tennis" and ticker not in result.markets:
                market.in_open_list = False
        self.tracker.matches.update(result.matches)
        if self.generic_discovery is not None and (
            now - self._last_generic >= self.settings.strategies.cross_market.refresh_interval_s
        ):
            generic = await self.generic_discovery.refresh()
            self._last_generic = now
            for ticker, market in generic.markets.items():
                existing = self.markets.get(ticker)
                if existing is None:
                    self.markets[ticker] = market
                else:
                    existing.status = market.status
                    existing.in_open_list = True
            for ticker, market in self.markets.items():
                if market.category == "generic" and ticker not in generic.markets:
                    market.in_open_list = False
            self.groups = generic.groups
        if self.ws is not None:
            wanted = [t for t, m in self.markets.items() if m.status in ("active", "inactive")]
            await self.ws.set_markets("orderbook_delta", wanted)
            await self.ws.set_markets("market_lifecycle_v2", None)
        self.health["discovery"].ok(
            matches=len(self.tracker.matches),
            markets=len(self.markets),
            generic_events=len(self.groups),
            last_run=now,
        )

    async def _exchange_step(self) -> None:
        self.exchange_status = await self.rest.get_exchange_status()
        st = self.exchange_status
        self.health["exchange"].ok(
            exchange_active=st.exchange_active, trading_active=st.trading_active
        )
        if not (st.exchange_active and st.trading_active):
            self.health["exchange"].status = "paused"

    def _trading_active(self, market: TrackedMarket) -> bool:
        if self.exchange_status is None:
            return False
        return self.exchange_status.trading_active_for(market.exchange_index)

    # ========================================================== market data
    async def _rest_books_step(self) -> None:
        ws_ok = self.ws is not None and self.ws.health.connected
        if self.ws is not None:
            self.health["kalshi_ws"].status = "ok" if ws_ok else "reconnecting"
            self.health["kalshi_ws"].details.update(self.ws.health.snapshot())
            if not ws_ok:
                self.hub.release_ws_ownership()
        if self.hub.resync_requests:
            resync = sorted(self.hub.resync_requests)
            self.hub.resync_requests.clear()
            if self.ws is not None:
                await self.ws.request_snapshot("orderbook_delta", resync)
        tickers = [
            t
            for t, m in self.markets.items()
            if m.status in ("active", "inactive") and not self.hub.ws_owns(t)
        ]
        if tickers:
            books = await self.rest.get_orderbooks(tickers)
            received = time.time()
            for ticker, orderbook in books.items():
                self.hub.apply_rest_snapshot(ticker, orderbook, received)
        self.health["kalshi_rest"].ok(polled=len(tickers), **self.rest.health.snapshot())

    def _on_book(self, ticker: str, book: OrderBook) -> None:
        market = self.markets.get(ticker)
        self.recorder.on_book(ticker, book, market.status if market else None)
        self.tracker.on_book(ticker, book)

    async def _on_ws_gap(self, channel: str, tickers: list[str]) -> None:
        """Missed WebSocket message(s): distrust the affected books (REST polling covers them
        meanwhile) and ask for fresh snapshots, which hand them back to the WebSocket."""
        self.hub.on_ws_gap(channel, tickers)
        if self.ws is not None and channel == "orderbook_delta":
            await self.ws.request_snapshot(channel, tickers)

    def _on_lifecycle(self, msg: dict[str, Any]) -> None:
        ticker = msg.get("market_ticker")
        if isinstance(ticker, str) and ticker in self.markets:
            self.markets[ticker].last_status_check = 0.0  # re-check status promptly
            log.info("market_lifecycle_event", market=ticker, event_type=msg.get("event_type"))

    # =============================================================== scores
    async def _scores_step(self) -> None:
        if self.provider is None:
            return
        now = time.time()
        refs = []
        for match in self.tracker.matches.values():
            ref = match.match_ref()
            start = match.scheduled_start_ts or now
            if ref is None or start - now > 3600:
                continue
            current = self.scores.get(match.event_ticker)
            if current and current.state.status.is_over and now - current.state.received_ts > 900:
                continue
            refs.append(ref)
        if not refs:
            self.health["tennis_scores"].ok(tracked=0)
            return
        results = await self.provider.fetch(refs)
        by_match = {r.match_id: r for r in refs}
        live = rejected = 0
        for match_id, result in results.items():
            event = by_match[match_id].event_ticker
            self.recorder.on_score(event, result)
            if not self.tracker.on_score(event, result.state):
                rejected += 1
                continue
            self.scores[event] = result
            if result.state.status == MatchStatus.LIVE:
                live += 1
                for ticker, prob, ts in self.tracker.capture_pre_match(event):
                    self.repo.set_pre_match_prob(ticker, prob, ts, known_ts=now)
                    log.info("pre_match_price_captured", market=ticker, prob=prob)
        backfills = 0
        for match_id, result in results.items():
            event = by_match[match_id].event_ticker
            if result.state.status == MatchStatus.LIVE and event not in self._backfill_done:
                if backfills >= MAX_BACKFILLS_PER_CYCLE:
                    break
                backfills += 1
                await self._backfill_pre_match(event)
        self.health["tennis_scores"].ok(
            requested=len(refs),
            received=len(results),
            live=live,
            out_of_order_rejected=rejected,
            provider=self.provider.name,
        )

    async def _backfill_pre_match(self, event_ticker: str) -> None:
        """Matches first seen in progress: recover the pre-match price from public candles.
        The candle must predate the first live score by the guard interval, so a match that
        started earlier than scheduled cannot leak in-play prices into the prior."""
        self._backfill_done.add(event_ticker)
        match = self.tracker.matches.get(event_ticker)
        if match is None or match.scheduled_start_ts is None:
            return
        for ticker in match.market_tickers:
            market = self.markets.get(ticker or "")
            if market is None or market.pre_match_prob is not None:
                continue
            found = await pre_match_mid_from_candles(
                self.rest, market.series_ticker, market.ticker, match.scheduled_start_ts
            )
            if found is None:
                continue
            prob, ts = found
            if not self.tracker.prior_usable(event_ticker, ts):
                log.info("pre_match_backfill_too_recent", market=market.ticker, candle_ts=ts)
                continue
            market.pre_match_prob, market.pre_match_prob_ts = prob, ts
            self.repo.set_pre_match_prob(market.ticker, prob, ts, known_ts=time.time())
            log.info("pre_match_price_backfilled", market=market.ticker, prob=prob)

    def _record_prediction(
        self, match: TrackedMatch, estimate: ProbabilityEstimate, now: float
    ) -> None:
        state = self.tracker.scores.get(match.event_ticker)
        info = self.tracker.score_info(match.event_ticker, now)
        rows = []
        for index, ticker in enumerate(match.market_tickers):
            if not ticker:
                continue
            book = self.hub.books.get(ticker)
            valid = book is not None and book.valid
            bid = book.best_bid("yes") if valid and book else None
            ask = book.best_ask("yes") if valid and book else None
            prob = estimate.for_player(index)
            rows.append(
                {
                    "ts": now,
                    "model": "tennis_markov",
                    "event_ticker": match.event_ticker,
                    "market_ticker": ticker,
                    "tour": match.tour_key,
                    "stage": match_stage(
                        state, self.settings.strategies.tennis_model.late_games_from_victory
                    ),
                    "score_text": state.score_text()[:128] if state else None,
                    "p_point": prob.point,
                    "p_low": prob.low,
                    "p_robust_low": prob.robust_low,
                    "prior_source": estimate.prior.source,
                    "yes_bid": float(bid[0]) if bid else None,
                    "yes_ask": float(ask[0]) if ask else None,
                    "book_age_s": book.age_s(now) if valid and book else None,
                    "score_age_s": info.age_s,
                }
            )
        try:
            self.repo.insert_model_predictions(rows)
        except Exception as exc:  # recording must never block trading decisions
            log.warning("prediction_record_failed", error=repr(exc)[:200])

    # ========================================================== settlement
    async def _status_step(self) -> None:
        now = time.time()
        open_markets = self.core.open_market_tickers()
        due = []
        for ticker, market in self.markets.items():
            needs = (not market.in_open_list) or ticker in open_markets
            if needs and now - market.last_status_check >= 25:
                due.append(ticker)
        # Positions in markets no longer tracked (e.g. after a restart) still need settling.
        due.extend(t for t in open_markets if t not in self.markets)
        # An early sale releases exposure, but its final outcome is still needed to
        # compare that sale with holding the same purchase. Closed trades stay closed.
        due.extend(self.repo.unresolved_exit_markets())
        if not due:
            self.health["settlement"].ok(open_markets=len(open_markets), checked=0)
            return
        fetched = await self.rest.get_markets(tickers=sorted(set(due)))
        rows = []
        settled = 0
        for m in fetched:
            tracked = self.markets.get(m.ticker)
            if tracked is not None:
                tracked.status, tracked.result = m.status, m.result
                tracked.settlement_value = m.settlement_value_dollars
                tracked.last_status_check = now
            rows.append(
                {
                    "ticker": m.ticker,
                    "event_ticker": m.event_ticker,
                    "series_ticker": m.event_ticker.split("-", 1)[0],
                    "status": m.status,
                    "result": m.result,
                    "settlement_value": m.settlement_value_dollars,
                    "settlement_ts": m.settlement_ts.timestamp() if m.settlement_ts else None,
                }
            )
            if m.is_settled and m.result:
                if m.ticker in open_markets:
                    trades = self.core.settle(m.ticker, m.result, m.settlement_value_dollars, now)
                    settled += len(trades)
                    self.repo.record_event(
                        "info",
                        "paper",
                        f"settled {m.ticker}: result={m.result} "
                        f"({len(trades)} position(s) across ledgers)",
                    )
                elif self.market_maker is not None:
                    value = m.settlement_value_dollars
                    if value is None:
                        value = {"yes": Decimal(1), "no": Decimal(0)}.get(m.result)
                    if value is not None:
                        self.market_maker.settle(m.ticker, value)
                self._finalized_at.setdefault(m.ticker, now)
        self.repo.upsert_markets(rows)
        self._drop_finished(now)
        self.health["settlement"].ok(
            open_markets=len(open_markets), checked=len(fetched), settled_positions=settled
        )

    def _drop_finished(self, now: float) -> None:
        for ticker, finalized in list(self._finalized_at.items()):
            if now - finalized < FINALIZED_GRACE_S:
                continue
            market = self.markets.pop(ticker, None)
            self._finalized_at.pop(ticker, None)
            self.hub.drop(ticker)
            self.tracker.mid_history.pop(ticker, None)
            self.repo.delete_live_views([ticker])
            if market is not None and not any(
                m.event_ticker == market.event_ticker for m in self.markets.values()
            ):
                self.tracker.forget(market.event_ticker)
                self.scores.pop(market.event_ticker, None)
                self.groups.pop(market.event_ticker, None)

    # ============================================================ strategy
    def _valid_books(self) -> dict[str, OrderBook]:
        return {t: b for t, b in self.hub.books.items() if b.valid}

    async def build_context(self, now: float) -> ScanContext:
        books = self._valid_books()
        matches = []
        for match in list(self.tracker.matches.values()):
            if self.tracker.estimate_key(match) is not None and (
                self.tracker.cached_estimate(match) is None
            ):
                await asyncio.to_thread(self.tracker.estimate, match, now)
            matches.append(
                self.tracker.match_context(
                    match,
                    now,
                    books,
                    self.exchange_status is not None and self.exchange_status.trading_active,
                )
            )
        return ScanContext(
            now=now,
            markets=self.markets,
            books=books,
            matches=matches,
            groups=list(self.groups.values()),
            exchange_active=self._trading_active,
        )

    async def _evaluate_step(self) -> None:
        now = time.time()
        ctx = await self.build_context(now)
        opps = self.core.run_cycle(ctx)
        best: dict[str, Opportunity] = {}
        for opp in opps:
            for leg in opp.legs:
                current = best.get(leg.market_ticker)
                if current is None or STATUS_PRIORITY[opp.status] < STATUS_PRIORITY[current.status]:
                    best[leg.market_ticker] = opp
        views = []
        for match_ctx in ctx.matches:
            views.extend(self._views_for(match_ctx, best, now))
        self.repo.upsert_live_views(views)
        if self.settings.paper.enable_exits and not self.record_only:
            self._check_exits()
        counts: dict[str, int] = {}
        for opp in opps:
            counts[opp.status.value] = counts.get(opp.status.value, 0) + 1
        self.health["strategies"].ok(
            matches=len(ctx.matches), opportunities=len(opps), statuses=counts
        )

    def _check_exits(self) -> None:
        target = self.settings.paper.take_profit_bid
        for trade in self.broker.open_trades():
            book = self.hub.books.get(trade.market_ticker)
            market = self.markets.get(trade.market_ticker)
            if book is None or not book.valid or market is None or not market.tradable:
                continue
            bid = book.best_bid(trade.side)  # type: ignore[arg-type]
            if bid is not None and bid[0] >= target:
                self.broker.execute_sell(
                    trade.id,
                    book,
                    target,
                    f"exit:{trade.id}:{int(time.time() // 30)}",
                    market.fee_multiplier,
                )

    def _traded_probability(
        self, raw: PlayerProbability | None, mid: Decimal | None
    ) -> PlayerProbability | None:
        """The tennis_model strategy's view: its calibration applied to the raw model."""
        cal = self.settings.strategies.tennis_model.calibration
        if raw is None or cal.is_identity or (cal.market_weight and mid is None):
            return raw
        market = float(mid) if mid is not None else None
        return PlayerProbability(
            calibrate(raw.point, cal, market),
            calibrate(raw.low, cal, market),
            calibrate(raw.robust_low, cal, market),
        )

    def _views_for(
        self, ctx: Any, best: dict[str, Opportunity], now: float
    ) -> list[dict[str, Any]]:
        views = []
        match = ctx.match
        score = ctx.score
        estimate = ctx.estimate
        info = ctx.score_info
        state = score.score if score is not None else None
        for index, ticker in enumerate(match.market_tickers):
            market = self.markets.get(ticker or "")
            if market is None:
                continue
            book = self.hub.books.get(market.ticker)
            valid = book is not None and book.valid
            bid = book.best_bid("yes") if valid and book else None
            ask = book.best_ask("yes") if valid and book else None
            mid = book.mid("yes") if valid and book else None
            raw = estimate.for_player(index) if estimate else None
            prob = self._traded_probability(raw, mid)
            opp = best.get(market.ticker)
            views.append(
                {
                    "market_ticker": market.ticker,
                    "event_ticker": match.event_ticker,
                    "updated_ts": now,
                    "data": {
                        "market_ticker": market.ticker,
                        "event_ticker": match.event_ticker,
                        "match_title": match.title,
                        "competition": match.competition,
                        "tour": match.tour_key,
                        "round": match.round_name,
                        "player_name": market.player_name,
                        "player_index": index,
                        "opponent_name": match.player_names[1 - index],
                        "scheduled_start_ts": match.scheduled_start_ts,
                        "market_status": market.status,
                        "match_status": score.status.value if score else None,
                        "stage": match_stage(
                            score, self.settings.strategies.tennis_model.late_games_from_victory
                        ),
                        "score_text": score.score_text() if score else None,
                        "score_valid": bool(score and score.valid),
                        "score_issues": [info.issue]
                        if info.issue
                        else (list(score.issues) if score else []),
                        "score_age_s": info.age_s,
                        "score_source_age_s": score.source_age_s if score else None,
                        "score_last_change_s": info.unchanged_s,
                        "score_quiet": bool(
                            info.unchanged_s is not None
                            and info.unchanged_s > self.settings.risk.max_score_unchanged_s
                        ),
                        "mid_move_since_score_change": float(info.mid_move)
                        if info.mid_move is not None
                        else None,
                        "format": score.fmt.label() if score else match.format_hint().label(),
                        "is_final_set": bool(state and state.is_final_set),
                        "serving": (state.server == index) if state else None,
                        "games_from_victory": state.games_from_victory(index) if state else None,
                        "has_match_point": bool(state and state.has_match_point(index)),
                        "yes_bid": float(bid[0]) if bid else None,
                        "yes_bid_size": float(bid[1]) if bid else None,
                        "yes_ask": float(ask[0]) if ask else None,
                        "yes_ask_size": float(ask[1]) if ask else None,
                        "spread": float(ask[0] - bid[0]) if bid and ask else None,
                        "ask_depth_3c": float(book.ask_depth("yes", ask[0] + Decimal("0.03")))
                        if ask and book
                        else None,
                        "book_age_s": book.age_s(now) if valid and book else None,
                        "book_source": book.source if valid and book else None,
                        "market_prob": float(mid) if mid is not None else None,
                        "model_prob": prob.point if prob else None,
                        "model_prob_low": prob.low if prob else None,
                        "model_prob_robust": prob.robust_low if prob else None,
                        "model_prob_raw": raw.point if raw else None,
                        "model_calibrated": prob is not raw,
                        "model_std": estimate.p1_std if estimate else None,
                        "prior_source": estimate.prior.source if estimate else None,
                        "pre_match_prob": market.pre_match_prob,
                        # Edge of buying YES at the best ask with the conservative / point
                        # probability, before fees (fees depend on the final order size).
                        "ev_conservative": (prob.robust_low - float(ask[0]))
                        if prob and ask
                        else None,
                        "ev_point": (prob.point - float(ask[0])) if prob and ask else None,
                        "opportunity": opp is not None
                        and opp.status
                        in (
                            DecisionStatus.FILLED,
                            DecisionStatus.PARTIAL,
                            DecisionStatus.APPROVED,
                            DecisionStatus.OBSERVED,
                        ),
                        "opportunity_status": opp.status.value if opp else None,
                        "opportunity_side": opp.side if opp else None,
                        "opportunity_strategy": opp.strategy if opp else None,
                        "opportunity_reason": (
                            (opp.reasons[0] if opp.reasons else opp.reason) if opp else None
                        ),
                    },
                }
            )
        return views

    # ========================================================= housekeeping
    async def _housekeeping_step(self) -> None:
        self.recorder.flush()
        retention = self.settings.market_data.snapshot_retention_days
        if retention is not None:
            self.repo.prune_market_snapshots(time.time() - retention * 86400.0)
        self.core.diagnostics.flush()
        self.health["recorder"].ok(
            market_rows=self.recorder.market_rows_written,
            score_rows=self.recorder.score_rows_written,
        )
        events = drain_recent_events()
        if events:
            self.repo.record_events(events)
        self._persist_health()

    def _persist_health(self) -> None:
        self.health["strategies"].details["diagnostics"] = self.core.diagnostics.snapshot()
        for name, h in self.health.items():
            self.repo.upsert_health(
                name,
                h.status,
                ok=h.status == "ok",
                error=h.last_error,
                details={**h.details, "last_ok": h.last_ok},
            )

    async def _equity_step(self) -> None:
        now = time.time()
        books = self._valid_books()
        self.core.record_equity(books, now)
        if self.market_maker is not None:
            marks = {t: m for t, b in books.items() if (m := b.mid("yes")) is not None}
            self.market_maker.persist_summary(marks)
        if now - self._last_evidence >= EVIDENCE_REFRESH_S:
            self._last_evidence = now
            self.core.refresh_evidence(now)
            self.repo.prune_opportunities(OPPORTUNITY_RETENTION_S)
