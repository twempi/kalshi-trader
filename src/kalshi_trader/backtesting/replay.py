"""Chronological replay of *recorded* observations through the real strategy stack.

Uses the market snapshots and live-score snapshots this application recorded, each with
its local receive time, so the strategy only ever sees what the live system saw at that
moment (no look-ahead). The same strategy, estimator, risk manager and paper broker run
against an in-memory account with a simulated clock. Markets whose final result was not
recorded are excluded from P&L and reported.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Any

from sqlalchemy import select

from kalshi_trader.api.schemas import PriceRange
from kalshi_trader.backtesting.metrics import TradeOutcome, compute_metrics, win_rate_by_bucket
from kalshi_trader.backtesting.stats import significance_report
from kalshi_trader.config import Settings
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
from kalshi_trader.execution.paper import OrderIntent, PaperBroker
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch, tour_key_for
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import ProbabilityEstimate, WinProbabilityEstimator
from kalshi_trader.risk.manager import MarketConditions, RiskManager
from kalshi_trader.strategies import MatchContext, Strategy, build_strategies
from kalshi_trader.tennis.providers.base import MatchRef
from kalshi_trader.tennis.providers.kalshi_live import parse_kalshi_live
from kalshi_trader.tennis.state import MatchState

log = get_logger("backtest.replay")
DATA_SOURCE = "recorded_live_observations (real Kalshi books + Kalshi live scores)"


@dataclass(frozen=True)
class _Obs:
    ts: float
    kind: str  # "book" | "score"
    event_ticker: str
    payload: Any


def _load(
    db: Database,
) -> tuple[dict[str, TrackedMatch], dict[str, TrackedMarket], dict[str, MatchRef], list[_Obs]]:
    with db.session() as s:
        events = {e.event_ticker: e for e in s.scalars(select(EventRow))}
        markets = list(s.scalars(select(MarketRow)))
        books = list(s.scalars(select(MarketSnapshotRow).order_by(MarketSnapshotRow.received_ts)))
        scores = list(s.scalars(select(ScoreSnapshotRow).order_by(ScoreSnapshotRow.received_ts)))
    tracked: dict[str, TrackedMarket] = {}
    for m in markets:
        ranges = [PriceRange.model_validate(r) for r in json.loads(m.price_ranges_json or "[]")]
        tracked[m.ticker] = TrackedMarket(
            ticker=m.ticker,
            event_ticker=m.event_ticker,
            series_ticker=m.series_ticker,
            player_name=m.player_name,
            competitor_id=m.competitor_id,
            status="active",
            result=m.result,
            settlement_value=m.settlement_value,
            fee_multiplier=m.fee_multiplier,
            price_ranges=ranges,
            exchange_index=m.exchange_index,
            close_ts=m.close_ts,
            occurrence_ts=m.occurrence_ts,
            pre_match_prob=m.pre_match_prob,
        )
    matches: dict[str, TrackedMatch] = {}
    refs: dict[str, MatchRef] = {}
    for ticker, e in events.items():
        by_comp = {m.competitor_id: m for m in tracked.values() if m.event_ticker == ticker}
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
    obs: list[_Obs] = []
    for b in books:
        market = tracked.get(b.ticker)
        if market is not None and b.depth_json:
            obs.append(_Obs(b.received_ts, "book", market.event_ticker, b))
    for sc in scores:
        if sc.event_ticker and sc.milestone_id in refs and sc.raw_json:
            obs.append(_Obs(sc.received_ts, "score", sc.event_ticker, sc))
    obs.sort(key=lambda o: o.ts)
    return matches, tracked, refs, obs


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


def replay(
    settings: Settings,
    source_db: Database,
    strategies: list[Strategy] | None = None,
    start_ts: float | None = None,
    end_ts: float | None = None,
) -> tuple[list[TradeOutcome], dict[str, Any]]:
    matches, markets, refs, observations = _load(source_db)
    sim_db = Database(":memory:")
    repo = Repository(sim_db)
    clock = {"now": 0.0}
    broker = PaperBroker(sim_db, settings.paper, settings.fees, clock=lambda: clock["now"])
    risk = RiskManager(settings.risk, repo, settings.fees.taker_rate)
    estimator = WinProbabilityEstimator(settings.model)
    strategies = strategies if strategies is not None else build_strategies(settings)
    books: dict[str, OrderBook] = {}
    scores: dict[str, MatchState] = {}
    estimates: dict[str, tuple[str, ProbabilityEstimate]] = {}
    evaluated = 0

    for ob in observations:
        if (start_ts and ob.ts < start_ts) or (end_ts and ob.ts > end_ts):
            continue
        clock["now"] = ob.ts
        if ob.kind == "book":
            books[ob.payload.ticker] = _book_from(ob.payload)
        else:
            ref = refs[ob.payload.milestone_id]
            scores[ob.event_ticker] = parse_kalshi_live(json.loads(ob.payload.raw_json), ref, ob.ts)
        match = matches.get(ob.event_ticker)
        state = scores.get(ob.event_ticker)
        if match is None or state is None or not state.is_live or state.score is None:
            continue
        fp = state.fingerprint()
        cached = estimates.get(match.event_ticker)
        if cached is None or cached[0] != fp:
            pm = [
                m.pre_match_prob if i == 0 else 1 - m.pre_match_prob
                for i, t in enumerate(match.market_tickers)
                if t and (m := markets.get(t)) and m.pre_match_prob is not None
            ]
            prior = estimator.build_prior(
                match.tour_key, state.fmt, sum(pm) / len(pm) if pm else None
            )
            cached = (
                fp,
                estimator.estimate(
                    state.score, prior, state.serve_stats, f"{match.event_ticker}:{fp}"
                ),
            )
            estimates[match.event_ticker] = cached
        ctx = MatchContext(
            now=ob.ts,
            match=match,
            markets={t: markets[t] for t in match.market_tickers if t in markets},
            books={t: books[t] for t in match.market_tickers if t in books},
            score=state,
            estimate=cached[1],
            exchange_trading_active=True,
        )
        evaluated += 1
        for strategy in strategies:
            for sig in strategy.evaluate(ctx):
                book = books[sig.market_ticker]
                in_market, in_event, open_in_event = broker.trade_counts(
                    sig.market_ticker, sig.event_ticker
                )
                account = broker.snapshot({}, day_start_ts=None)
                decision = risk.check(
                    sig,
                    account,
                    MarketConditions(
                        market_tradable=True,
                        exchange_trading_active=True,
                        book_age_s=book.age_s(ob.ts),
                        score_age_s=ctx.score_age_s(),
                        spread=book.spread(sig.side),
                        depth_at_limit=book.ask_depth(sig.side, sig.limit_price),
                        trades_in_market=in_market,
                        trades_in_event=in_event,
                        has_open_position_in_event=open_in_event,
                    ),
                )
                if not decision.approved:
                    continue
                attempts = broker.order_attempts(sig.market_ticker, sig.strategy)
                order_id = f"{sig.strategy}:{sig.market_ticker}:{sig.side}:{attempts + 1}"
                broker.execute_buy(
                    OrderIntent(
                        client_order_id=order_id,
                        strategy=sig.strategy,
                        market_ticker=sig.market_ticker,
                        event_ticker=sig.event_ticker,
                        side=sig.side,
                        limit_price=sig.limit_price,
                        contracts=decision.contracts,
                        max_cash=decision.max_cash,
                        fee_multiplier=sig.fee_multiplier,
                        match_title=match.title,
                        player_name=sig.player_name,
                        score_text=sig.score_text,
                        model_prob=sig.p_point,
                        model_prob_low=sig.p_low,
                        robust_prob=sig.p_conservative,
                        market_prob=sig.market_prob,
                        ev_per_contract=sig.ev_per_contract,
                        data_source="replay",
                    ),
                    book,
                )

    unsettled = 0
    for trade in broker.open_trades():
        market = markets.get(trade.market_ticker)
        if market is not None and market.result in ("yes", "no"):
            broker.settle_market(
                trade.market_ticker,
                market.result,
                market.settlement_value,
                ts=market.close_ts or clock["now"],
            )
        else:
            unsettled += 1
    with sim_db.session() as s:
        closed = list(
            s.scalars(select(PaperTradeRow).where(PaperTradeRow.status.in_(["settled", "closed"])))
        )
    outcomes = [
        TradeOutcome(
            opened_ts=t.opened_ts,
            closed_ts=t.closed_ts or t.opened_ts,
            strategy=t.strategy,
            entry_price=float(t.entry_price),
            contracts=float(t.entry_contracts),
            cost=float(t.entry_cost + t.entry_fee),
            pnl=float(t.realized_pnl or 0),
            fees=float(t.entry_fee + t.exit_fee),
            model_prob=t.robust_prob,
            event=t.event_ticker,
        )
        for t in closed
    ]
    info = {
        "observations": len(observations),
        "evaluations": evaluated,
        "matches_with_scores": len({o.event_ticker for o in observations if o.kind == "score"}),
        "unsettled_trades_excluded": unsettled,
    }
    return outcomes, info


def _grid() -> Iterator[dict[str, Any]]:
    for min_edge, games, latency in product(("0.01", "0.02", "0.04"), (1, 2, 3), (0, 1, 2)):
        yield {"min_edge": min_edge, "max_games_from_victory": games, "latency_points": latency}


def run_replay_backtest(
    settings: Settings, test_fraction: float = 0.3, grid: bool = False
) -> dict[str, Any]:
    source = Database(settings.db_path)
    with source.session() as s:
        first = s.scalars(
            select(ScoreSnapshotRow.received_ts).order_by(ScoreSnapshotRow.received_ts).limit(1)
        ).first()
        last = s.scalars(
            select(ScoreSnapshotRow.received_ts)
            .order_by(ScoreSnapshotRow.received_ts.desc())
            .limit(1)
        ).first()
    if first is None or last is None:
        return {
            "data_source": DATA_SOURCE,
            "error": "no recorded score data yet - run `kalshi-trader run` to collect "
            "observations; a replay needs recorded matches with final results",
        }
    boundary = first + (last - first) * (1 - test_fraction)
    start_balance = float(settings.paper.starting_balance)

    def evaluate(cfg: Settings) -> dict[str, Any]:
        train, info_train = replay(cfg, source, end_ts=boundary)
        test, info_test = replay(cfg, source, start_ts=boundary)
        return {
            "train": compute_metrics(train, start_balance),
            "test": compute_metrics(test, start_balance),
            "test_significance": significance_report(test),
            "calibration_all": win_rate_by_bucket(train + test),
            "info": {"train": info_train, "test": info_test},
        }

    started = time.time()
    report: dict[str, Any] = {
        "data_source": DATA_SOURCE,
        "period": {"start": first, "end": last, "oos_boundary": boundary},
        "baseline_config": evaluate(settings),
    }
    if grid:
        rows = []
        for params in _grid():
            cfg = settings.model_copy(deep=True)
            cfg.strategies.late_match.min_edge = Decimal(params["min_edge"])
            cfg.strategies.late_match.max_games_from_victory = params["max_games_from_victory"]
            cfg.model.latency_points = params["latency_points"]
            rows.append({"params": params, **evaluate(cfg)})
        eligible = [r for r in rows if r["train"].get("trades", 0) >= 20]
        best = max(eligible, key=lambda r: r["train"]["total_pnl"]) if eligible else None
        report["grid"] = [
            {"params": r["params"], "train": r["train"], "test": r["test"]} for r in rows
        ]
        report["selected_in_sample"] = {
            "params": best["params"] if best else None,
            "out_of_sample": best["test"] if best else None,
            "note": "chosen on the training window only; 27 configurations compared, so "
            "in-sample results are optimistic",
        }
    report["runtime_s"] = round(time.time() - started, 1)
    with source.session() as s:
        base = report["baseline_config"]
        s.add(
            BacktestRunRow(
                created_ts=time.time(),
                kind="replay_recorded",
                data_source=DATA_SOURCE,
                params_json=json.dumps({"test_fraction": test_fraction, "grid": grid}),
                metrics_json=json.dumps(
                    {
                        "train": base["train"],
                        "test": base["test"],
                        "significance_test": base["test_significance"],
                    },
                    default=str,
                ),
            )
        )
    return report
