"""Tennis match state shared by the live engine and the replay (identical rules in both).

Responsibilities:

* keep the latest score per match, **rejecting out-of-order observations** (a CDN-cached
  response showing fewer points played than one already seen is stale and ignored);
* track when the score last *changed* and the market mids at that moment, to detect a
  frozen or lagging feed (``ScoreInfo``);
* compute the score age as local age + the source's own age at receipt (HTTP ``Age`` when
  known, otherwise the configured worst case ``risk.score_source_staleness_s``);
* capture the pre-match price strictly from observations made at least
  ``PRE_MATCH_GUARD_S`` before the first live score, and reject backfilled priors that are
  not old enough (they could contain in-play information if the match started early);
* cache the probability estimate per score state and optionally record model predictions
  next to the executable market (forward data for calibration).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from decimal import Decimal

from kalshi_trader.config import RiskSettings
from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import ProbabilityEstimate, WinProbabilityEstimator
from kalshi_trader.strategies.base import MatchContext, ScoreInfo
from kalshi_trader.tennis.state import MatchState, MatchStatus

PRE_MATCH_GUARD_S = 60.0  # a pre-match price must predate the first live score by this much
REGRESSION_ISSUE = "score went backwards: stale cached response ignored"

PredictionSink = Callable[[TrackedMatch, ProbabilityEstimate, float], None]


class TennisTracker:
    def __init__(
        self,
        estimator: WinProbabilityEstimator,
        risk: RiskSettings,
        markets: dict[str, TrackedMarket],
        prediction_sink: PredictionSink | None = None,
    ) -> None:
        self.estimator = estimator
        self.risk = risk
        self.markets = markets  # shared with the owner (engine/replay)
        self.matches: dict[str, TrackedMatch] = {}
        self.scores: dict[str, MatchState] = {}
        self.changed: dict[str, tuple[str, float]] = {}
        self.first_live: dict[str, float] = {}
        self.mid_history: dict[str, deque[tuple[float, float]]] = {}
        self.mid_at_change: dict[str, dict[str, float]] = {}
        self.issues: dict[str, str] = {}
        self.regressions = 0
        self._estimates: dict[str, tuple[tuple[object, ...], ProbabilityEstimate]] = {}
        self.prediction_sink = prediction_sink

    # ------------------------------------------------------------ inputs
    def on_book(self, ticker: str, book: OrderBook) -> None:
        mid = book.mid("yes") if book.valid else None
        if mid is None:
            return
        history = self.mid_history.setdefault(ticker, deque(maxlen=3000))
        value = float(mid)
        if not history or history[-1][1] != value:
            history.append((book.updated_at or 0.0, value))

    def _current_mid(self, ticker: str) -> float | None:
        history = self.mid_history.get(ticker)
        return history[-1][1] if history else None

    def on_score(self, event: str, state: MatchState) -> bool:
        """Apply a new observation. Returns False if it was rejected as out of order."""
        previous = self.scores.get(event)
        if (
            previous is not None
            and previous.score is not None
            and state.score is not None
            and previous.status == MatchStatus.LIVE
            and state.status == MatchStatus.LIVE
            and state.score.progress() < previous.score.progress()
        ):
            self.issues[event] = REGRESSION_ISSUE
            self.regressions += 1
            return False
        self.issues.pop(event, None)
        self.scores[event] = state
        fingerprint = state.fingerprint()
        last = self.changed.get(event)
        if last is None or last[0] != fingerprint:
            self.changed[event] = (fingerprint, state.received_ts)
            match = self.matches.get(event)
            if match is not None:
                self.mid_at_change[event] = {
                    t: mid for t in match.market_tickers if t and (mid := self._current_mid(t))
                }
        if state.status == MatchStatus.LIVE and event not in self.first_live:
            self.first_live[event] = state.received_ts
        return True

    # ------------------------------------------------------------ pre-match prior
    def capture_pre_match(self, event: str) -> list[tuple[str, float, float]]:
        """Set pre-match prices from mids observed before the first live score."""
        first_live = self.first_live.get(event)
        match = self.matches.get(event)
        captured = []
        if first_live is None or match is None:
            return captured
        for ticker in match.market_tickers:
            market = self.markets.get(ticker or "")
            if market is None or market.pre_match_prob is not None:
                continue
            history = self.mid_history.get(market.ticker) or ()
            before = [(ts, mid) for ts, mid in history if ts <= first_live - PRE_MATCH_GUARD_S]
            if before:
                ts, mid = before[-1]
                market.pre_match_prob, market.pre_match_prob_ts = mid, ts
                captured.append((market.ticker, mid, ts))
        return captured

    def prior_usable(self, event: str, prob_ts: float | None) -> bool:
        """A stored/backfilled prior is usable only if it predates the first live score."""
        first_live = self.first_live.get(event)
        if prob_ts is None:
            return False
        return first_live is None or prob_ts <= first_live - PRE_MATCH_GUARD_S

    def pre_match_p1(self, match: TrackedMatch) -> float | None:
        values = []
        for index, ticker in enumerate(match.market_tickers):
            market = self.markets.get(ticker or "")
            if market is None or market.pre_match_prob is None:
                continue
            if market.pre_match_prob_ts is not None and not self.prior_usable(
                match.event_ticker, market.pre_match_prob_ts
            ):
                continue
            values.append(market.pre_match_prob if index == 0 else 1 - market.pre_match_prob)
        return sum(values) / len(values) if values else None

    # ------------------------------------------------------------ model
    def estimate_key(self, match: TrackedMatch) -> tuple[object, ...] | None:
        state = self.scores.get(match.event_ticker)
        if state is None or not state.is_live or state.score is None:
            return None
        return (state.fingerprint(), self.pre_match_p1(match))

    def cached_estimate(self, match: TrackedMatch) -> ProbabilityEstimate | None:
        key = self.estimate_key(match)
        cached = self._estimates.get(match.event_ticker)
        if key is None or cached is None or cached[0] != key:
            return None
        return cached[1]

    def estimate(self, match: TrackedMatch, now: float | None = None) -> ProbabilityEstimate | None:
        """Compute (or reuse) the estimate for the current score. CPU-bound: the live engine
        runs it in a worker thread."""
        key = self.estimate_key(match)
        if key is None:
            return None
        cached = self._estimates.get(match.event_ticker)
        if cached is not None and cached[0] == key:
            return cached[1]
        state = self.scores[match.event_ticker]
        assert state.score is not None
        prior = self.estimator.build_prior(match.tour_key, state.fmt, key[1])  # type: ignore[arg-type]
        estimate = self.estimator.estimate(
            state.score, prior, state.serve_stats, f"{match.event_ticker}:{key[0]}"
        )
        self._estimates[match.event_ticker] = (key, estimate)
        if self.prediction_sink is not None:
            self.prediction_sink(match, estimate, now if now is not None else state.received_ts)
        return estimate

    # ------------------------------------------------------------ freshness
    def score_info(self, event: str, now: float) -> ScoreInfo:
        state = self.scores.get(event)
        issue = self.issues.get(event)
        if state is None:
            return ScoreInfo(issue=issue)
        source_age = (
            state.source_age_s
            if state.source_age_s is not None
            else self.risk.score_source_staleness_s
        )
        age = max(0.0, now - state.received_ts) + source_age
        unchanged = None
        changed = self.changed.get(event)
        if changed is not None and state.status == MatchStatus.LIVE:
            unchanged = max(0.0, now - changed[1])
        moves = []
        for ticker, mid0 in self.mid_at_change.get(event, {}).items():
            mid = self._current_mid(ticker)
            if mid is not None:
                moves.append(abs(mid - mid0))
        if not issue and not state.valid and state.issues:
            issue = "; ".join(state.issues)[:200]
        return ScoreInfo(
            age_s=age,
            unchanged_s=unchanged,
            mid_move=Decimal(str(round(max(moves), 4))) if moves else None,
            issue=issue,
        )

    def match_context(
        self,
        match: TrackedMatch,
        now: float,
        books: dict[str, OrderBook],
        exchange_trading_active: bool,
    ) -> MatchContext:
        tickers = [t for t in match.market_tickers if t]
        return MatchContext(
            now=now,
            match=match,
            markets={t: m for t in tickers if (m := self.markets.get(t))},
            books={t: b for t in tickers if (b := books.get(t)) is not None and b.valid},
            score=self.scores.get(match.event_ticker),
            estimate=self.cached_estimate(match),
            exchange_trading_active=exchange_trading_active,
            score_info=self.score_info(match.event_ticker, now),
            pre_match_p1=self.pre_match_p1(match),
        )

    def forget(self, event: str) -> None:
        for store in (
            self.matches,
            self.scores,
            self.changed,
            self.first_live,
            self.mid_at_change,
            self.issues,
            self._estimates,
        ):
            store.pop(event, None)
