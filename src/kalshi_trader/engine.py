"""Main asyncio engine: market discovery, data feeds, scoring, strategies and paper trading.

Data flow::

    discovery ──> tracked matches/markets ──> WS orderbook subscriptions / REST polling
    score provider ──> normalized MatchState ──> estimator (cached per score change)
    every second: per match -> strategies -> Signal -> RiskManager -> PaperBroker
    settlement monitor -> market results -> PaperBroker.settle_market
    recorder -> SQLite (market + score observations for replay)

The engine never places real orders: the REST client is GET-only and the WebSocket client
only manages subscriptions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.api.rest import KalshiRestClient
from kalshi_trader.api.schemas import ExchangeStatus
from kalshi_trader.api.ws import KalshiWebSocketClient
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.paper import OrderIntent, PaperBroker
from kalshi_trader.logging_setup import drain_recent_events, get_logger
from kalshi_trader.market_data.discovery import TennisDiscovery, TrackedMarket, TrackedMatch
from kalshi_trader.market_data.feeds import MarketDataHub
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.market_data.prematch import pre_match_mid_from_candles
from kalshi_trader.market_data.recorder import Recorder
from kalshi_trader.models.estimator import ProbabilityEstimate, WinProbabilityEstimator
from kalshi_trader.risk.manager import AccountSnapshot, MarketConditions, RiskManager
from kalshi_trader.strategies import MatchContext, Signal, build_strategies
from kalshi_trader.tennis.providers.base import ProviderResult, ScoreProvider
from kalshi_trader.tennis.providers.kalshi_live import KalshiLiveDataProvider
from kalshi_trader.tennis.state import MatchStatus

log = get_logger("engine")

PRE_MATCH_GUARD_S = 60.0  # pre-match price must predate the first "live" score by this much
MAX_BACKFILLS_PER_CYCLE = 4
SCORE_QUIET_LIMIT_S = 300.0  # a live match whose score has not changed for this long is suspect
ENTRY_COOLDOWN_S = 30.0
FINALIZED_GRACE_S = 600.0
EV_SIZE_HINT = Decimal("10")


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
        )
        self.discovery = TennisDiscovery(self.rest, self.repo, settings.discovery)
        if provider is not None:
            self.provider: ScoreProvider | None = provider
        elif settings.tennis.provider == "kalshi_live_data":
            self.provider = KalshiLiveDataProvider(self.rest)
        else:
            self.provider = None
        self.estimator = WinProbabilityEstimator(settings.model)
        self.strategies = [] if record_only else build_strategies(settings)
        self.broker = PaperBroker(self.db, settings.paper, settings.fees)
        self.risk = RiskManager(settings.risk, self.repo, settings.fees.taker_rate)
        self.ws: KalshiWebSocketClient | None = None
        if self.signer is not None and settings.kalshi.use_websocket:
            self.ws = KalshiWebSocketClient(
                settings.kalshi.websocket_url,
                self.signer,
                self.hub.handle_ws_message,
                self._on_ws_gap,
            )

        self.matches: dict[str, TrackedMatch] = {}
        self.markets: dict[str, TrackedMarket] = {}
        self.scores: dict[str, ProviderResult] = {}  # by event ticker
        self._score_changed: dict[str, tuple[str, float]] = {}
        self._first_live: dict[str, float] = {}
        self._estimates: dict[str, tuple[tuple[Any, ...], ProbabilityEstimate]] = {}
        self._mid_history: dict[str, deque[tuple[float, float]]] = {}
        self._finalized_at: dict[str, float] = {}
        self._backfill_done: set[str] = set()
        self._last_entry_attempt: dict[str, float] = {}
        self._last_signal_key: dict[tuple[str, str], tuple[str, float]] = {}
        self._opportunities: dict[str, Signal] = {}
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

    # ============================================================ lifecycle
    def request_stop(self) -> None:
        self._stop.set()

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
            f"websocket={'on' if self.ws else 'off'})",
        )
        log.info(
            "engine_starting",
            record_only=self.record_only,
            websocket=bool(self.ws),
            strategies=[s.name for s in self.strategies],
        )
        self.health["engine"].ok(started_at=time.time(), record_only=self.record_only)
        tasks = [
            asyncio.create_task(
                self._loop(
                    "discovery", self._discovery_step, self.settings.discovery.refresh_interval_s
                )
            ),
            asyncio.create_task(self._loop("exchange", self._exchange_step, 30.0)),
            asyncio.create_task(
                self._loop(
                    "kalshi_rest",
                    self._rest_books_step,
                    self.settings.market_data.rest_poll_interval_s,
                )
            ),
            asyncio.create_task(
                self._loop("tennis_scores", self._scores_step, self.settings.tennis.poll_interval_s)
            ),
            asyncio.create_task(
                self._loop(
                    "settlement",
                    self._status_step,
                    self.settings.market_data.status_refresh_interval_s,
                )
            ),
            asyncio.create_task(
                self._loop(
                    "strategies",
                    self._evaluate_step,
                    self.settings.strategies.evaluation_interval_s,
                )
            ),
            asyncio.create_task(self._loop("housekeeping", self._housekeeping_step, 2.0)),
            asyncio.create_task(self._loop("equity", self._equity_step, 60.0)),
        ]
        if self.ws is not None:
            tasks.append(asyncio.create_task(self.ws.run(self._stop)))
        try:
            await self._stop.wait()
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
                self.markets[ticker] = market
            else:
                existing.status = market.status
                existing.fee_multiplier = market.fee_multiplier
                existing.price_ranges = market.price_ranges
                existing.close_ts = market.close_ts
                existing.in_open_list = True
        for ticker, market in self.markets.items():
            if ticker not in result.markets:
                market.in_open_list = False
        self.matches.update(result.matches)
        if self.ws is not None:
            wanted = [t for t, m in self.markets.items() if m.status in ("active", "inactive")]
            await self.ws.set_markets("orderbook_delta", wanted)
            await self.ws.set_markets("market_lifecycle_v2", None)
        self.health["discovery"].ok(
            matches=len(self.matches), markets=len(self.markets), last_run=now
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
        mid = book.mid("yes")
        if mid is not None:
            history = self._mid_history.setdefault(ticker, deque(maxlen=3000))
            value = float(mid)
            if not history or history[-1][1] != value:
                history.append((book.updated_at or time.time(), value))

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
        for match in self.matches.values():
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
        live = 0
        for match_id, result in results.items():
            ref = by_match[match_id]
            self.scores[ref.event_ticker] = result
            self.recorder.on_score(ref.event_ticker, result)
            fingerprint = result.state.fingerprint()
            previous = self._score_changed.get(ref.event_ticker)
            if previous is None or previous[0] != fingerprint:
                self._score_changed[ref.event_ticker] = (fingerprint, result.state.received_ts)
            if result.state.status == MatchStatus.LIVE:
                live += 1
                if ref.event_ticker not in self._first_live:
                    self._first_live[ref.event_ticker] = result.state.received_ts
                    self._capture_pre_match(ref.event_ticker, result.state.received_ts)
        backfills = 0
        for match_id, result in results.items():
            event = by_match[match_id].event_ticker
            if result.state.status == MatchStatus.LIVE and event not in self._backfill_done:
                if backfills >= MAX_BACKFILLS_PER_CYCLE:
                    break
                backfills += 1
                await self._backfill_pre_match(event)
        self.health["tennis_scores"].ok(
            requested=len(refs), received=len(results), live=live, provider=self.provider.name
        )

    def _capture_pre_match(self, event_ticker: str, first_live_ts: float) -> None:
        match = self.matches.get(event_ticker)
        if match is None:
            return
        for ticker in match.market_tickers:
            market = self.markets.get(ticker or "")
            if market is None or market.pre_match_prob is not None:
                continue
            history = self._mid_history.get(market.ticker) or ()
            before = [(ts, mid) for ts, mid in history if ts <= first_live_ts - PRE_MATCH_GUARD_S]
            if before:
                ts, mid = before[-1]
                market.pre_match_prob = mid
                self.repo.set_pre_match_prob(market.ticker, mid, ts)
                log.info("pre_match_price_captured", market=market.ticker, prob=mid)

    async def _backfill_pre_match(self, event_ticker: str) -> None:
        """Matches first seen in progress: recover the pre-match price from public candles."""
        self._backfill_done.add(event_ticker)
        match = self.matches.get(event_ticker)
        if match is None or match.scheduled_start_ts is None:
            return
        for ticker in match.market_tickers:
            market = self.markets.get(ticker or "")
            if market is None or market.pre_match_prob is not None:
                continue
            found = await pre_match_mid_from_candles(
                self.rest, market.series_ticker, market.ticker, match.scheduled_start_ts
            )
            if found is not None:
                market.pre_match_prob, ts = found
                self.repo.set_pre_match_prob(market.ticker, market.pre_match_prob, ts)
                log.info(
                    "pre_match_price_backfilled", market=market.ticker, prob=market.pre_match_prob
                )

    def _pre_match_p1(self, match: TrackedMatch) -> float | None:
        values = []
        for index, ticker in enumerate(match.market_tickers):
            market = self.markets.get(ticker or "")
            if market is not None and market.pre_match_prob is not None:
                values.append(market.pre_match_prob if index == 0 else 1 - market.pre_match_prob)
        return sum(values) / len(values) if values else None

    # ========================================================== settlement
    async def _status_step(self) -> None:
        now = time.time()
        open_markets = {t.market_ticker for t in self.broker.open_trades()}
        due = []
        for ticker, market in self.markets.items():
            needs = (not market.in_open_list) or ticker in open_markets
            if needs and now - market.last_status_check >= 25:
                due.append(ticker)
        # Positions in markets no longer tracked (e.g. after a restart) still need settling.
        due.extend(t for t in open_markets if t not in self.markets)
        if not due:
            return
        fetched = await self.rest.get_markets(tickers=sorted(set(due)))
        rows = []
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
                    self.broker.settle_market(m.ticker, m.result, m.settlement_value_dollars)
                    self.repo.record_event(
                        "info", "paper", f"settled {m.ticker}: result={m.result}"
                    )
                self._finalized_at.setdefault(m.ticker, now)
        self.repo.upsert_markets(rows)
        self._drop_finished(now)

    def _drop_finished(self, now: float) -> None:
        for ticker, finalized in list(self._finalized_at.items()):
            if now - finalized < FINALIZED_GRACE_S:
                continue
            market = self.markets.pop(ticker, None)
            self._finalized_at.pop(ticker, None)
            self.hub.drop(ticker)
            self._mid_history.pop(ticker, None)
            self.repo.delete_live_views([ticker])
            if market is not None and not any(
                m.event_ticker == market.event_ticker for m in self.markets.values()
            ):
                self.matches.pop(market.event_ticker, None)
                self.scores.pop(market.event_ticker, None)
                self._estimates.pop(market.event_ticker, None)

    # ============================================================ strategy
    async def _estimate(
        self, match: TrackedMatch, result: ProviderResult | None
    ) -> ProbabilityEstimate | None:
        if result is None or not result.state.is_live or result.state.score is None:
            return None
        prior_p1 = self._pre_match_p1(match)
        key = (result.state.fingerprint(), prior_p1)
        cached = self._estimates.get(match.event_ticker)
        if cached is not None and cached[0] == key:
            return cached[1]
        state = result.state
        prior = self.estimator.build_prior(match.tour_key, state.fmt, prior_p1)
        estimate = await asyncio.to_thread(
            self.estimator.estimate,
            state.score,
            prior,
            state.serve_stats,
            f"{match.event_ticker}:{key[0]}",
        )
        self._estimates[match.event_ticker] = (key, estimate)
        return estimate

    def _account_snapshot(self) -> AccountSnapshot:
        marks: dict[tuple[str, str], Decimal] = {}
        for trade in self.broker.open_trades():
            book = self.hub.books.get(trade.market_ticker)
            bid = book.best_bid(trade.side) if book is not None and book.valid else None
            if bid is not None:
                marks[(trade.market_ticker, trade.side)] = bid[0]
        return self.broker.snapshot(marks, day_start_ts=RiskManager.utc_day_start())

    async def _evaluate_step(self) -> None:
        now = time.time()
        views = []
        self._opportunities = {}
        for match in list(self.matches.values()):
            result = self.scores.get(match.event_ticker)
            estimate = await self._estimate(match, result)
            score_state = result.state if result else None
            changed = self._score_changed.get(match.event_ticker)
            quiet = bool(
                score_state
                and score_state.is_live
                and changed
                and now - changed[1] > SCORE_QUIET_LIMIT_S
            )
            ctx = MatchContext(
                now=now,
                match=match,
                markets={t: m for t in match.market_tickers if t and (m := self.markets.get(t))},
                books={
                    t: b
                    for t in match.market_tickers
                    if t and (b := self.hub.books.get(t)) and b.valid
                },
                score=None if quiet else score_state,
                estimate=None if quiet else estimate,
                exchange_trading_active=self.exchange_status is not None
                and self.exchange_status.trading_active,
            )
            signals: list[Signal] = []
            for strategy in self.strategies:
                signals.extend(strategy.evaluate(ctx))
            for sig in signals:
                self._opportunities[sig.market_ticker] = sig
                await self._handle_signal(sig, ctx)
            views.extend(self._views_for(match, ctx, estimate, score_state, quiet, now))
        self.repo.upsert_live_views(views)
        if self.settings.paper.enable_exits and not self.record_only:
            self._check_exits()
        self.health["strategies"].ok(matches=len(self.matches), signals=len(self._opportunities))

    async def _handle_signal(self, sig: Signal, ctx: MatchContext) -> None:
        market = self.markets.get(sig.market_ticker)
        book = self.hub.books.get(sig.market_ticker)
        if market is None or book is None:
            return
        in_market, in_event, open_in_event = self.broker.trade_counts(
            sig.market_ticker, sig.event_ticker
        )
        account = self._account_snapshot()
        conditions = MarketConditions(
            market_tradable=market.tradable,
            exchange_trading_active=self._trading_active(market),
            book_age_s=book.age_s(ctx.now),
            score_age_s=ctx.score_age_s(),
            spread=book.spread(sig.side),
            depth_at_limit=book.ask_depth(sig.side, sig.limit_price),
            trades_in_market=in_market,
            trades_in_event=in_event,
            has_open_position_in_event=open_in_event,
        )
        decision = self.risk.check(sig, account, conditions)
        self._record_signal(sig, decision.approved, decision.reasons)
        if not decision.approved or self.record_only:
            return
        if ctx.now - self._last_entry_attempt.get(sig.market_ticker, 0.0) < ENTRY_COOLDOWN_S:
            return
        self._last_entry_attempt[sig.market_ticker] = ctx.now
        attempts = self.broker.order_attempts(sig.market_ticker, sig.strategy)
        intent = OrderIntent(
            client_order_id=f"{sig.strategy}:{sig.market_ticker}:{sig.side}:{attempts + 1}",
            strategy=sig.strategy,
            market_ticker=sig.market_ticker,
            event_ticker=sig.event_ticker,
            side=sig.side,
            limit_price=sig.limit_price,
            contracts=decision.contracts,
            max_cash=decision.max_cash,
            fee_multiplier=sig.fee_multiplier,
            match_title=ctx.match.title,
            player_name=sig.player_name,
            score_text=sig.score_text,
            model_prob=sig.p_point,
            model_prob_low=sig.p_low,
            robust_prob=sig.p_conservative,
            market_prob=sig.market_prob,
            ev_per_contract=sig.ev_per_contract,
        )
        report = self.broker.execute_buy(intent, book)
        if report.status in ("filled", "partial") and report.fill is not None:
            self.repo.record_event(
                "info",
                "paper",
                f"PAPER {report.status}: {sig.strategy} bought {report.fill.filled} "
                f"{sig.side.upper()} {sig.market_ticker} @ {report.fill.avg_price} "
                f"(fee {report.fill.fee}) | {sig.score_text}",
            )

    def _record_signal(self, sig: Signal, approved: bool, reasons: list[str]) -> None:
        key = (sig.strategy, sig.market_ticker)
        digest = f"{approved}|{'|'.join(sorted(reasons))}|{sig.best_ask}"
        last = self._last_signal_key.get(key)
        now = time.time()
        if last and last[0] == digest and now - last[1] < 60:
            return
        self._last_signal_key[key] = (digest, now)
        self.repo.insert_signal(
            {
                "created_ts": now,
                "strategy": sig.strategy,
                "market_ticker": sig.market_ticker,
                "event_ticker": sig.event_ticker,
                "side": sig.side,
                "ask_price": sig.best_ask,
                "model_prob": sig.p_point,
                "model_prob_low": sig.p_low,
                "robust_prob": sig.p_conservative,
                "market_prob": sig.market_prob,
                "ev_per_contract": sig.ev_per_contract,
                "approved": approved,
                "reasons_json": json.dumps(reasons),
                "score_text": sig.score_text[:128],
            }
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

    def _views_for(
        self,
        match: TrackedMatch,
        ctx: MatchContext,
        estimate: ProbabilityEstimate | None,
        score: Any,
        quiet: bool,
        now: float,
    ) -> list[dict[str, Any]]:
        views = []
        fee_rate = self.settings.fees.taker_rate
        changed = self._score_changed.get(match.event_ticker)
        state = score.score if score is not None else None
        for index, ticker in enumerate(match.market_tickers):
            market = self.markets.get(ticker or "")
            if market is None:
                continue
            book = self.hub.books.get(market.ticker)
            valid = book is not None and book.valid
            bid = book.best_bid("yes") if valid else None
            ask = book.best_ask("yes") if valid else None
            mid = book.mid("yes") if valid else None
            prob = estimate.for_player(index) if estimate else None
            ev_cons = ev_point = None
            if ask is not None and prob is not None:
                fee_pc = fee_per_contract_estimate(
                    ask[0], EV_SIZE_HINT, fee_rate, market.fee_multiplier
                )
                cost = float(ask[0] + fee_pc)
                ev_cons, ev_point = prob.robust_low - cost, prob.point - cost
            sig = self._opportunities.get(market.ticker)
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
                        "score_text": score.score_text() if score else None,
                        "score_valid": bool(score and score.valid),
                        "score_issues": list(score.issues) if score else [],
                        "score_age_s": (now - score.received_ts) if score else None,
                        "score_last_change_s": (now - changed[1]) if changed else None,
                        "score_quiet": quiet,
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
                        "model_std": estimate.p1_std if estimate else None,
                        "prior_source": estimate.prior.source if estimate else None,
                        "pre_match_prob": market.pre_match_prob,
                        "ev_conservative": ev_cons,
                        "ev_point": ev_point,
                        "opportunity": sig is not None,
                        "opportunity_side": sig.side if sig else None,
                        "opportunity_strategy": sig.strategy if sig else None,
                        "opportunity_reason": sig.reason if sig else None,
                    },
                }
            )
        return views

    # ========================================================= housekeeping
    async def _housekeeping_step(self) -> None:
        self.recorder.flush()
        self.health["recorder"].ok(
            market_rows=self.recorder.market_rows_written,
            score_rows=self.recorder.score_rows_written,
        )
        events = drain_recent_events()
        if events:
            self.repo.record_events(events)
        self._persist_health()

    def _persist_health(self) -> None:
        for name, h in self.health.items():
            self.repo.upsert_health(
                name,
                h.status,
                ok=h.status == "ok",
                error=h.last_error,
                details={**h.details, "last_ok": h.last_ok},
            )

    async def _equity_step(self) -> None:
        snap = self._account_snapshot()
        self.broker.record_equity(snap)
        self.risk.check_account_limits(snap)
