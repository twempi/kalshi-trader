"""Event-driven replay of *recorded* observations through the live decision path.

What makes this faithful to live trading:

* **One chronological event stream** - order-book snapshots, score snapshots and market
  settlements are merged and processed strictly in time order. A position settles at the
  moment its market's settlement became known (``settlement_ts``, else ``close_ts``), so
  cash, equity, open positions, exposure and every risk limit reflect the simulated state
  *at that time* (previously all settlement happened after the last observation).
* **The live code path** - the same ``TennisTracker`` (score regression guard, freshness,
  pre-match prior) and ``TradingCore`` (strategies, evaluator, risk, allocator, paper broker,
  shadow ledgers) run against an in-memory database with a simulated clock: account marks
  from current books, UTC-day loss limits, drawdown stop, cooldowns, duplicate prevention.
* **Market lifecycle** - each market's status comes from the recorded snapshot status, and a
  market is never tradable after its close time.
* **Data freshness like live** - the recorder stores a book/score only when it changes (plus
  a heartbeat). A recorded value is treated as continuously confirmed until the next record
  of the same series if the gap is within the heartbeat (+ slack); otherwise it ages from
  its last record, exactly as stale data would in live trading.
* **No look-ahead in priors** - pre-match priors are captured from replayed books before the
  first live score; a stored prior is used only if its observation time predates the first
  live score by the guard interval (it was public before the match).

Known limits (reported with every run): market metadata (fee multiplier, tick structure)
is the latest recorded value; fills are simulated against recorded top-N depth only; and the
recorded data covers only the periods the engine was running.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import product
from typing import Any

from sqlalchemy import func, select

from kalshi_trader.api.schemas import PriceRange
from kalshi_trader.backtesting.metrics import (
    TradeOutcome,
    by_strategy,
    compute_metrics,
    outcomes_from_trades,
    win_rate_by_bucket,
)
from kalshi_trader.backtesting.stats import significance_report
from kalshi_trader.config import Settings
from kalshi_trader.core.pipeline import TradingCore
from kalshi_trader.core.tennis_tracker import TennisTracker
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.database.tables import (
    BacktestRunRow,
    EventRow,
    MarketRow,
    MarketSnapshotRow,
    PaperTradeRow,
    ScoreSnapshotRow,
)
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch, tour_key_for
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import ProbabilityEstimate, WinProbabilityEstimator
from kalshi_trader.research.model_eval import (
    Prediction,
    evaluate_predictions,
    one_market_per_event,
)
from kalshi_trader.research.splits import EventSplit, split_events
from kalshi_trader.strategies import ScanContext, Strategy, build_strategies
from kalshi_trader.strategies.base import RelatedGroup
from kalshi_trader.strategies.market_making import MarketMakingSimulator
from kalshi_trader.strategies.stat_mispricing import match_stage
from kalshi_trader.tennis.providers.base import MatchRef
from kalshi_trader.tennis.providers.kalshi_live import parse_kalshi_live
from kalshi_trader.tennis.state import MatchStatus

log = get_logger("backtest.replay")
DATA_SOURCE = "recorded_live_observations (real Kalshi books + Kalshi live scores)"
CONTINUITY_SLACK_S = 15.0
EQUITY_INTERVAL_S = 3600.0
ONE = Decimal("1")
ZERO = Decimal("0")


# ------------------------------------------------------------------ loading
@dataclass
class RecordedData:
    matches: dict[str, TrackedMatch]
    markets: dict[str, TrackedMarket]
    groups: dict[str, RelatedGroup]
    refs: dict[str, MatchRef]
    books: list[MarketSnapshotRow]
    scores: list[ScoreSnapshotRow]
    stored_priors: dict[str, tuple[float, float | None]]  # ticker -> (prob, observed ts)
    settlements: dict[str, tuple[float, str, Decimal | None]]  # ticker -> (ts, result, value)

    def first_seen(self) -> dict[str, float]:
        seen: dict[str, float] = {}
        for row in self.books:
            market = self.markets.get(row.ticker)
            if market is not None:
                seen[market.event_ticker] = min(
                    seen.get(market.event_ticker, row.received_ts), row.received_ts
                )
        return seen

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(f"{len(self.books)}:{len(self.scores)}".encode())
        for rows in (self.books[:1], self.books[-1:], self.scores[:1], self.scores[-1:]):
            for r in rows:
                h.update(f"{r.id}:{r.received_ts}".encode())
        for ticker, (ts, result, value) in sorted(self.settlements.items()):
            h.update(f"{ticker}:{ts}:{result}:{value}".encode())
        return h.hexdigest()


def load_recorded(db: Database) -> RecordedData:
    with db.session() as s:
        events = {e.event_ticker: e for e in s.scalars(select(EventRow))}
        market_rows = list(s.scalars(select(MarketRow)))
        books = list(s.scalars(select(MarketSnapshotRow).order_by(MarketSnapshotRow.received_ts)))
        scores = list(s.scalars(select(ScoreSnapshotRow).order_by(ScoreSnapshotRow.received_ts)))
    markets: dict[str, TrackedMarket] = {}
    priors: dict[str, tuple[float, float | None]] = {}
    settlements: dict[str, tuple[float, str, Decimal | None]] = {}
    for m in market_rows:
        ranges = [PriceRange.model_validate(r) for r in json.loads(m.price_ranges_json or "[]")]
        event = events.get(m.event_ticker)
        markets[m.ticker] = TrackedMarket(
            ticker=m.ticker,
            event_ticker=m.event_ticker,
            series_ticker=m.series_ticker,
            player_name=m.player_name,
            competitor_id=m.competitor_id,
            status="unopened",  # set from recorded snapshots as the replay proceeds
            fee_multiplier=m.fee_multiplier,
            price_ranges=ranges,
            exchange_index=m.exchange_index,
            close_ts=m.close_ts,
            occurrence_ts=m.occurrence_ts,
            title=m.title or "",
            strike_type=m.strike_type,
            floor_strike=m.floor_strike,
            cap_strike=m.cap_strike,
            category=(event.category if event and event.category else "tennis"),
        )
        if m.pre_match_prob is not None:
            priors[m.ticker] = (m.pre_match_prob, m.pre_match_prob_ts)
        if m.result in ("yes", "no"):
            ts = m.settlement_ts or m.close_ts
            if ts is not None:
                settlements[m.ticker] = (ts, m.result, m.settlement_value)
    matches: dict[str, TrackedMatch] = {}
    groups: dict[str, RelatedGroup] = {}
    refs: dict[str, MatchRef] = {}
    for ticker, e in events.items():
        in_event = [m for m in markets.values() if m.event_ticker == ticker]
        if e.category == "generic":
            groups[ticker] = RelatedGroup(
                event_ticker=ticker,
                series_ticker=e.series_ticker,
                title=e.title,
                mutually_exclusive=e.mutually_exclusive,
                markets=in_event,
                category="generic",
            )
            continue
        by_comp = {m.competitor_id: m for m in in_event}
        ids = (e.first_competitor_id, e.second_competitor_id)
        matches[ticker] = TrackedMatch(
            event_ticker=ticker,
            series_ticker=e.series_ticker,
            title=e.title,
            competition=e.competition,
            milestone_id=e.milestone_id,
            tour=e.tour,
            gender=e.gender,
            round_name=e.round_name,
            tournament_name=e.tournament_name,
            best_of=e.best_of,
            scheduled_start_ts=e.scheduled_start_ts,
            player_ids=ids,
            market_tickers=tuple(by_comp[c].ticker if c in by_comp else None for c in ids),
            player_names=tuple(by_comp[c].player_name if c in by_comp else "?" for c in ids),
            tour_key=tour_key_for(e.series_ticker, e.tour, e.gender),
        )
        if e.milestone_id:
            refs[e.milestone_id] = MatchRef(
                match_id=e.milestone_id,
                event_ticker=ticker,
                player_ids=ids,
                tournament_name=e.tournament_name,
                gender=e.gender,
                best_of=e.best_of,
            )
    books = [b for b in books if b.ticker in markets and b.depth_json]
    scores = [s for s in scores if s.milestone_id in refs and s.raw_json and s.event_ticker]
    return RecordedData(matches, markets, groups, refs, books, scores, priors, settlements)


def _continuity(rows: list[Any], key: str, heartbeat: float) -> dict[int, float]:
    """For each row id: until when its value is known to have stayed current."""
    out: dict[int, float] = {}
    last: dict[str, Any] = {}
    for row in rows:
        k = getattr(row, key)
        prev = last.get(k)
        if prev is not None:
            gap = row.received_ts - prev.received_ts
            out[prev.id] = (
                row.received_ts if gap <= heartbeat + CONTINUITY_SLACK_S else prev.received_ts
            )
        last[k] = row
    for row in last.values():
        out[row.id] = row.received_ts
    return out


# ------------------------------------------------------------------- replay
@dataclass
class ReplayResult:
    outcomes: list[TradeOutcome]  # main account
    shadow_outcomes: dict[str, list[TradeOutcome]]
    info: dict[str, Any]
    ledger: list[dict[str, Any]]
    predictions: list[Prediction] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    market_making: dict[str, Any] | None = None


def _trade_dict(t: PaperTradeRow) -> dict[str, Any]:
    return {
        "account_id": t.account_id,
        "strategy": t.strategy,
        "strategy_version": t.strategy_version,
        "market": t.market_ticker,
        "event": t.event_ticker,
        "side": t.side,
        "opened_ts": t.opened_ts,
        "closed_ts": t.closed_ts,
        "contracts": str(t.entry_contracts),
        "entry_price": str(t.entry_price),
        "entry_fee": str(t.entry_fee),
        "status": t.status,
        "pnl": str(t.realized_pnl) if t.realized_pnl is not None else None,
        "expected_pnl": t.expected_pnl,
        "group_id": t.group_id,
        "stage": t.stage,
    }


def replay(
    settings: Settings,
    data: RecordedData | Database,
    strategies: list[Strategy] | None = None,
    *,
    events: frozenset[str] | set[str] | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    score_delay_s: float = 0.0,
    collect_predictions: bool = True,
    market_making: bool = False,
) -> ReplayResult:
    """Replay recorded observations (optionally only `events`) in chronological order.

    ``score_delay_s`` delays every score observation (stale-score sensitivity analysis).
    """
    if isinstance(data, Database):
        data = load_recorded(data)
    sim_db = Database(":memory:")
    repo = Repository(sim_db)
    clock = {"now": 0.0}
    strategies = strategies if strategies is not None else build_strategies(settings)
    markets = {t: _copy_market(m) for t, m in data.markets.items()}
    predictions: list[Prediction] = []
    outcomes_by_ticker = {t: 1 if r == "yes" else 0 for t, (_, r, _) in data.settlements.items()}
    late_games = settings.strategies.tennis_model.late_games_from_victory

    def sink(match: TrackedMatch, est: ProbabilityEstimate, now: float) -> None:
        if not collect_predictions:
            return
        state = tracker.scores.get(match.event_ticker)
        info = tracker.score_info(match.event_ticker, now)
        for index, ticker in enumerate(match.market_tickers):
            if ticker is None or ticker not in outcomes_by_ticker:
                continue
            book = books.get(ticker)
            bid = book.best_bid("yes") if book is not None else None
            ask = book.best_ask("yes") if book is not None else None
            prob = est.for_player(index)
            predictions.append(
                Prediction(
                    ts=now,
                    event=match.event_ticker,
                    market=ticker,
                    p_model=prob.point,
                    p_model_low=prob.robust_low,
                    outcome=outcomes_by_ticker[ticker],
                    yes_bid=float(bid[0]) if bid else None,
                    yes_ask=float(ask[0]) if ask else None,
                    stage=match_stage(state, late_games),
                    tour=match.tour_key,
                    score_age_s=info.age_s,
                    prior_source=est.prior.source,
                )
            )

    tracker = TennisTracker(
        WinProbabilityEstimator(settings.model), settings.risk, markets, prediction_sink=sink
    )
    tracker.matches.update({e: m for e, m in data.matches.items() if events is None or e in events})
    mm = (
        MarketMakingSimulator(settings.strategies.market_making, settings.fees, None, "replay")
        if market_making
        else None
    )
    core = TradingCore(
        settings,
        sim_db,
        repo,
        strategies,
        clock=lambda: clock["now"],
        data_kind="replay",
        market_maker=mm,
        persist_opportunities=False,
    )
    md, ts_settings = settings.market_data, settings.tennis
    book_until = _continuity(data.books, "ticker", md.snapshot_heartbeat_s)
    score_until = _continuity(data.scores, "milestone_id", ts_settings.snapshot_heartbeat_s)

    # (ts, order, kind, payload): settlements first at equal times (they were known first).
    stream: list[tuple[float, int, str, Any]] = []
    for row in data.books:
        market = markets[row.ticker]
        if events is None or market.event_ticker in events:
            stream.append((row.received_ts, 1, "book", row))
    for row in data.scores:
        if events is None or row.event_ticker in events:
            stream.append((row.received_ts + score_delay_s, 2, "score", row))
    for ticker, (ts, result, value) in data.settlements.items():
        if events is None or markets[ticker].event_ticker in events:
            stream.append((ts, 0, "settle", (ticker, result, value)))
    stream.sort(key=lambda item: (item[0], item[1]))

    books: dict[str, OrderBook] = {}
    book_rows: dict[str, MarketSnapshotRow] = {}
    score_rows: dict[str, ScoreSnapshotRow] = {}
    settled: set[str] = set()
    last_equity = None
    cycles = rejected_scores = 0
    first_live_done: set[str] = set()

    def refresh_freshness(now: float) -> None:
        # Books/scores confirmed by a later heartbeat count as freshly polled until then.
        for ticker, row in book_rows.items():
            book = books[ticker]
            confirmed = min(now, book_until.get(row.id, row.received_ts))
            book.updated_at = max(row.received_ts, confirmed - md.rest_poll_interval_s)
        for event, row in score_rows.items():
            state = tracker.scores.get(event)
            if state is None:
                continue
            confirmed = min(now, score_until.get(row.id, row.received_ts) + score_delay_s)
            received = max(row.received_ts + score_delay_s, confirmed - ts_settings.poll_interval_s)
            if received != state.received_ts:
                tracker.scores[event] = _with_received(state, received)

    for ts, _, kind, payload in stream:
        if (start_ts and ts < start_ts) or (end_ts and ts > end_ts):
            continue
        clock["now"] = ts
        if (
            last_equity is None
            or ts - last_equity >= EQUITY_INTERVAL_S
            or _new_day(last_equity, ts)
        ):
            core.record_equity(books, ts)
            last_equity = ts
        if kind == "settle":
            ticker, result, value = payload
            core.settle(ticker, result, value, ts)
            markets[ticker].status = "finalized"
            settled.add(ticker)
            continue
        if kind == "book":
            row: MarketSnapshotRow = payload
            book = _book_from(row)
            books[row.ticker] = book
            book_rows[row.ticker] = row
            market = markets[row.ticker]
            if row.ticker not in settled:
                # Older recordings did not store status: a quoted market is assumed active.
                market.status = row.status or "active"
            tracker.on_book(row.ticker, book)
            event = market.event_ticker
        else:
            row_s: ScoreSnapshotRow = payload
            ref = data.refs[row_s.milestone_id]
            state = parse_kalshi_live(json.loads(row_s.raw_json or "{}"), ref, ts)
            event = row_s.event_ticker or ref.event_ticker
            if not tracker.on_score(event, state):
                rejected_scores += 1
                continue
            score_rows[event] = row_s
            if state.status == MatchStatus.LIVE and event not in first_live_done:
                first_live_done.add(event)
                tracker.capture_pre_match(event)
                _apply_stored_priors(tracker, data, markets, event)
        for market in markets.values():  # lifecycle: nothing trades after close
            if market.close_ts is not None and ts >= market.close_ts and market.status == "active":
                market.status = "closed"
        refresh_freshness(ts)
        ctx = _context(tracker, markets, books, data, event, ts)
        if ctx is None:
            continue
        cycles += 1
        core.run_cycle(ctx)

    end = clock["now"]
    core.record_equity(books, end)
    unsettled = [t for t in core.broker.open_trades()]
    with sim_db.session() as s:
        all_trades = list(s.scalars(select(PaperTradeRow).order_by(PaperTradeRow.opened_ts)))
    main_rows = [t for t in all_trades if t.account_id == core.broker.account_id]
    shadow_outcomes = {
        name: outcomes_from_trades([t for t in all_trades if t.account_id == b.account_id])
        for name, b in core.shadows.items()
    }
    info = {
        "observations": len(stream),
        "evaluation_cycles": cycles,
        "events_replayed": len(
            {m.event_ticker for m in markets.values() if events is None or m.event_ticker in events}
        ),
        "matches_with_scores": len(score_rows),
        "out_of_order_scores_rejected": rejected_scores,
        "settlements_applied": len(settled),
        "unsettled_open_positions_main": len(unsettled),
        "final_cash_main": str(core.broker.account().cash),
        "score_delay_s": score_delay_s,
    }
    return ReplayResult(
        outcomes=outcomes_from_trades(main_rows),
        shadow_outcomes=shadow_outcomes,
        info=info,
        ledger=[_trade_dict(t) for t in all_trades],
        predictions=predictions,
        diagnostics=core.diagnostics.snapshot(),
        market_making=mm.summary() if mm is not None else None,
    )


def _copy_market(m: TrackedMarket) -> TrackedMarket:
    return replace(m, price_ranges=list(m.price_ranges))


def _with_received(state: Any, received: float) -> Any:
    return replace(state, received_ts=received)


def _new_day(previous: float, now: float) -> bool:
    return int(previous // 86400) != int(now // 86400)


def _apply_stored_priors(
    tracker: TennisTracker, data: RecordedData, markets: dict[str, TrackedMarket], event: str
) -> None:
    match = tracker.matches.get(event)
    if match is None:
        return
    for ticker in match.market_tickers:
        market = markets.get(ticker or "")
        stored = data.stored_priors.get(ticker or "")
        if market is None or market.pre_match_prob is not None or stored is None:
            continue
        prob, observed = stored
        if tracker.prior_usable(event, observed):
            market.pre_match_prob, market.pre_match_prob_ts = prob, observed


def _context(
    tracker: TennisTracker,
    markets: dict[str, TrackedMarket],
    books: dict[str, OrderBook],
    data: RecordedData,
    event: str,
    now: float,
) -> ScanContext | None:
    valid = {t: b for t, b in books.items() if b.valid}
    match = tracker.matches.get(event)
    if match is not None:
        tracker.estimate(match, now)
        return ScanContext(
            now=now,
            markets=markets,
            books=valid,
            matches=[tracker.match_context(match, now, valid, True)],
        )
    group = data.groups.get(event)
    if group is not None:
        group.markets = [markets[m.ticker] for m in group.markets if m.ticker in markets]
        return ScanContext(now=now, markets=markets, books=valid, groups=[group])
    return None


def _book_from(row: MarketSnapshotRow) -> OrderBook:
    depth = json.loads(row.depth_json or "{}")
    book = OrderBook(row.ticker)
    book.apply_snapshot(
        [(Decimal(p), Decimal(s)) for p, s in depth.get("yes_bids", [])],
        [(Decimal(p), Decimal(s)) for p, s in depth.get("no_bids", [])],
        source="replay",
        received_at=row.received_ts,
    )
    return book


# ------------------------------------------------------------------ reports
def _grid() -> Iterator[dict[str, Any]]:
    for min_edge, latency in product(("0.02", "0.03", "0.05"), (0, 1, 2)):
        yield {"min_edge": min_edge, "latency_points": latency}


def _code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
        )
        return (out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def _config_hash(settings: Settings) -> str:
    raw = settings.model_dump_json(exclude={"kalshi", "db_path", "log_dir", "dashboard"})
    return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()


def summarize(result: ReplayResult, start_balance: float) -> dict[str, Any]:
    shadow = {
        name: {
            "metrics": compute_metrics(outs, start_balance),
            "significance": significance_report(outs),
        }
        for name, outs in result.shadow_outcomes.items()
    }
    preds = one_market_per_event(result.predictions)
    return {
        "main_account": compute_metrics(result.outcomes, start_balance),
        "main_by_strategy": by_strategy(result.outcomes, start_balance),
        "main_significance": significance_report(result.outcomes),
        "shadow_ledgers": shadow,
        "calibration_by_entry_price": win_rate_by_bucket(result.outcomes),
        "model_evaluation": evaluate_predictions(preds) if preds else None,
        "info": result.info,
        "diagnostics": result.diagnostics,
        "market_making": result.market_making,
    }


def run_replay_backtest(
    settings: Settings,
    test_fraction: float = 0.3,
    validation_fraction: float = 0.2,
    grid: bool = False,
    embargo_s: float = 3600.0,
    evaluate_test: bool = True,
    market_making: bool = False,
) -> dict[str, Any]:
    source = Database(settings.db_path)
    data = load_recorded(source)
    if not data.scores and not data.books:
        return {
            "data_source": DATA_SOURCE,
            "error": "no recorded data yet - run `kalshi-trader run` to collect observations; "
            "a replay needs recorded matches with final results",
        }
    split: EventSplit = split_events(
        data.first_seen(),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        embargo_s=embargo_s,
    )
    start_balance = float(settings.paper.starting_balance)
    started = time.time()
    config_hash = _config_hash(settings)

    with source.session() as s:
        prior_test_runs = (
            s.scalar(
                select(func.count(BacktestRunRow.id)).where(
                    BacktestRunRow.kind == "replay_recorded",
                    BacktestRunRow.input_fingerprint == data.fingerprint(),
                    BacktestRunRow.notes.like("%test evaluated%"),
                )
            )
            or 0
        )

    ledgers: dict[str, list[dict[str, Any]]] = {}

    def run(cfg: Settings, group: frozenset[str], name: str = "") -> dict[str, Any]:
        result = replay(cfg, data, events=group, market_making=market_making)
        if name:
            ledgers[name] = result.ledger
        return summarize(result, start_balance)

    report: dict[str, Any] = {
        "data_source": DATA_SOURCE,
        "split": split.summary(),
        "settled_markets_available": len(data.settlements),
        "limitations": [
            "Market metadata (fee multiplier, tick size) is the latest recorded value.",
            "Fills are simulated on recorded top-of-book depth (configured levels) only.",
            "Only periods when the engine was recording are covered.",
            "Positions in markets without a recorded result stay open and are excluded from "
            "realised P&L (counted in info.unsettled_open_positions_main).",
        ],
        "train": run(settings, split.train),
        "validation": run(settings, split.validation) if split.validation else None,
    }
    selected = None
    if grid:
        rows = []
        for params in _grid():
            cfg = settings.model_copy(deep=True)
            cfg.strategies.tennis_model.min_edge = Decimal(params["min_edge"])
            cfg.model.latency_points = params["latency_points"]
            val = run(cfg, split.validation) if split.validation else None
            rows.append(
                {
                    "params": params,
                    "train": run(cfg, split.train)["shadow_ledgers"],
                    "validation": val["shadow_ledgers"] if val else None,
                }
            )
        report["grid"] = rows

        def score(r: dict[str, Any]) -> float:
            v = (r["validation"] or {}).get("tennis_model", {}).get("metrics", {})
            return (
                float(v.get("total_pnl") or float("-inf"))
                if v.get("trades", 0) >= 20
                else float("-inf")
            )

        best = max(rows, key=score) if rows else None
        if best is not None and score(best) > float("-inf"):
            selected = best["params"]
        report["selected_on_validation"] = {
            "params": selected,
            "note": f"{len(rows)} configurations compared on the validation split only "
            "(>= 20 validation trades required); the test split was not used for selection",
        }
    if evaluate_test:
        cfg = settings
        if selected is not None:
            cfg = settings.model_copy(deep=True)
            cfg.strategies.tennis_model.min_edge = Decimal(selected["min_edge"])
            cfg.model.latency_points = selected["latency_points"]
        report["test"] = run(cfg, split.test, "test")
        report["test_access"] = {
            "previous_test_evaluations_on_this_data": prior_test_runs,
            "warning": (
                "the test split has been evaluated before on this exact data; treat these "
                "results as increasingly optimistic"
                if prior_test_runs
                else None
            ),
        }
    report["runtime_s"] = round(time.time() - started, 1)
    ledger = ledgers.get("test", [])
    with source.session() as s:
        s.add(
            BacktestRunRow(
                created_ts=time.time(),
                kind="replay_recorded",
                data_source=DATA_SOURCE,
                params_json=json.dumps(
                    {
                        "test_fraction": test_fraction,
                        "validation_fraction": validation_fraction,
                        "embargo_s": embargo_s,
                        "grid": grid,
                        "selected": selected,
                        "config_hash": config_hash,
                    }
                ),
                metrics_json=json.dumps(
                    {
                        "train": report["train"]["shadow_ledgers"],
                        "test": report.get("test", {}).get("shadow_ledgers"),
                        "test_main": report.get("test", {}).get("main_account"),
                    },
                    default=str,
                ),
                notes="test evaluated" if evaluate_test else "train/validation only",
                config_json=settings.model_dump_json(exclude={"kalshi"}),
                universe_json=json.dumps(
                    {
                        "train": sorted(split.train),
                        "validation": sorted(split.validation),
                        "test": sorted(split.test),
                    }
                ),
                input_fingerprint=data.fingerprint(),
                code_version=_code_version(),
                ledger_json=json.dumps(ledger, default=str),
            )
        )
    return report
