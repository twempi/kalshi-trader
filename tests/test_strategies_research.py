"""Strategies A-D, tennis tracker freshness, splits, model evaluation, replay accounting.

SYNTHETIC test data only; none of these numbers are market results.
"""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal as D

import httpx
import pytest
from sqlalchemy import text

from kalshi_trader.api.rest import KalshiRestClient
from kalshi_trader.backtesting.replay import replay
from kalshi_trader.config import (
    CrossMarketStrategySettings,
    FeeSettings,
    KalshiSettings,
    MarketMakingStrategySettings,
    RiskSettings,
    Settings,
    TennisModelStrategySettings,
)
from kalshi_trader.core.opportunity import DecisionStatus
from kalshi_trader.core.tennis_tracker import REGRESSION_ISSUE, TennisTracker
from kalshi_trader.database.db import Database, add_missing_columns
from kalshi_trader.database.tables import EventRow, MarketRow, MarketSnapshotRow, ScoreSnapshotRow
from kalshi_trader.market_data.discovery import TrackedMarket
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import WinProbabilityEstimator
from kalshi_trader.research.model_eval import (
    Prediction,
    evaluate_predictions,
    one_market_per_event,
)
from kalshi_trader.research.splits import split_events
from kalshi_trader.strategies.base import RelatedGroup, ScanContext
from kalshi_trader.strategies.cross_market import (
    CrossMarketStrategy,
    covers_real_line,
    ladders,
    mutually_exclusive_no,
)
from kalshi_trader.strategies.market_making import MarketMakingSimulator
from kalshi_trader.strategies.stat_mispricing import TennisModelStrategy, match_stage
from kalshi_trader.tennis.state import MatchFormat, MatchState, MatchStatus, ScoreState
from tests.test_trading import make_context

FEES = FeeSettings()


# ------------------------------------------------------------ tennis model
def live_state(sets=(1, 0), games=(5, 3), points=(3, 0), completed=((6, 4),)) -> MatchState:
    fmt = MatchFormat(best_of=3)
    return MatchState(
        provider="t",
        match_id="m",
        player_ids=("a", "b"),
        status=MatchStatus.LIVE,
        received_ts=1000.0,
        fmt=fmt,
        score=ScoreState(fmt=fmt, sets=sets, games=games, points=points, completed_sets=completed),
    )


def test_match_stages() -> None:
    assert match_stage(None, 3) == "prematch"
    assert match_stage(live_state(sets=(0, 0), games=(2, 1), completed=()), 3) == "early"
    assert match_stage(live_state(), 3) == "late"  # 1 game from victory
    assert match_stage(live_state(games=(1, 1), points=(0, 0)), 3) == "mid"


def test_tennis_model_requires_independent_prior() -> None:
    ctx = make_context()
    ctx.estimate = replace(ctx.estimate, prior=replace(ctx.estimate.prior, source="tour_average"))
    strat = TennisModelStrategy(TennisModelStrategySettings(), FEES)
    (note,) = strat.evaluate_match(ctx)
    assert note.status == DecisionStatus.UNSUPPORTED and "pre-match price prior" in note.reasons[0]


def test_tennis_model_candidate_and_stage_filter() -> None:
    strat = TennisModelStrategy(TennisModelStrategySettings(max_price=D("0.95")), FEES)
    (opp,) = strat.evaluate_match(make_context())
    assert opp.status == DecisionStatus.CANDIDATE and opp.stage == "late"
    assert opp.fair.source == "tennis_markov" and opp.fair.p_low == 0.985
    assert opp.requires_score and opp.correlation_group == "tennis:ATP Test"
    only_early = TennisModelStrategy(TennisModelStrategySettings(stages=["early"]), FEES)
    (note,) = only_early.evaluate_match(make_context())
    assert note.status == DecisionStatus.FILTERED and "stage late disabled" in note.reasons[0]


def test_tennis_model_explains_insufficient_edge() -> None:
    strat = TennisModelStrategy(TennisModelStrategySettings(), FEES)
    (note,) = strat.evaluate_match(make_context(p_robust=0.93))
    assert note.status == DecisionStatus.FILTERED
    assert "best conservative edge" in note.reasons[0]


def test_tennis_model_marks_prematch_unsupported_when_enabled() -> None:
    strat = TennisModelStrategy(TennisModelStrategySettings(stages=["prematch", "late"]), FEES)
    ctx = make_context(status=MatchStatus.SCHEDULED)
    (note,) = strat.evaluate_match(ctx)
    assert note.status == DecisionStatus.UNSUPPORTED and "pre-match" in note.reasons[0]


# ------------------------------------------------------------ cross market
def gm(ticker: str, strike: str | None, floor=None, cap=None, competitor=None) -> TrackedMarket:
    return TrackedMarket(
        ticker=ticker,
        event_ticker="E",
        series_ticker="S",
        player_name=ticker,
        competitor_id=competitor,
        status="active",
        strike_type=strike,
        floor_strike=floor,
        cap_strike=cap,
        category="generic",
    )


def test_bracket_coverage_requires_no_gaps() -> None:
    full = [
        gm("L", "less", cap=58),
        gm("B1", "between", floor=58, cap=59),
        gm("B2", "between", floor=59, cap=61),
        gm("G", "greater", floor=61),
    ]
    assert covers_real_line(full) is None
    gap = [
        gm("L", "less", cap=58),
        gm("B", "between", floor=59, cap=60),
        gm("G", "greater", floor=60),
    ]
    assert "gap at 58" in covers_real_line(gap)  # integer outcomes are not assumed
    strict_both = [gm("L", "less", cap=60), gm("G", "greater", floor=60)]
    assert "gap at 60" in covers_real_line(strict_both)  # x == 60 pays nothing
    assert (
        covers_real_line([gm("L", "less_or_equal", cap=60), gm("G", "greater", floor=60)]) is None
    )
    assert "not a verifiable range" in covers_real_line([gm("C", "custom")])


def test_mutually_exclusive_no_basket_and_ladders() -> None:
    group = RelatedGroup("E", "S", "t", True, [gm("A", None), gm("B", None), gm("C", None)])
    (rel,) = mutually_exclusive_no(group, 12)
    assert rel.guaranteed_payout == D("2") and {s for _, s in rel.legs} == {"no"}
    assert "not flagged" in mutually_exclusive_no(replace(group, mutually_exclusive=False), 12)
    lad = ladders(
        RelatedGroup(
            "E", "S", "t", None, [gm("G1", "greater", floor=100), gm("G2", "greater", floor=105)]
        )
    )
    assert len(lad) == 1
    (low, side_low), (high, side_high) = lad[0].legs
    assert (low.ticker, side_low, high.ticker, side_high) == ("G1", "yes", "G2", "no")


def tennis_group_ctx(ask_a: str, ask_b: str, started: bool, now: float = 1000.0) -> ScanContext:
    markets = {
        "T-A": replace(gm("T-A", "structured", competitor="a"), category="tennis"),
        "T-B": replace(gm("T-B", "structured", competitor="b"), category="tennis"),
    }
    books = {}
    for t, ask in (("T-A", ask_a), ("T-B", ask_b)):
        b = OrderBook(t)
        b.apply_snapshot(
            [(D("0.30"), D("50"))], [(1 - D(ask), D("50"))], source="t", received_at=now
        )
        books[t] = b
    group = RelatedGroup("E", "S", "A vs B", None, list(markets.values()), "tennis", started)
    return ScanContext(now=now, markets=markets, books=books, groups=[group])


def test_tennis_pair_needs_started_match_and_persistence() -> None:
    strat = CrossMarketStrategy(CrossMarketStrategySettings(include_tennis_pairs=False), FEES)
    (note,) = strat.scan(tennis_group_ctx("0.45", "0.50", started=False))
    assert note.status == DecisionStatus.UNSUPPORTED and "not started" in note.reasons[0]
    (first,) = strat.scan(tennis_group_ctx("0.45", "0.50", started=True, now=1000.0))
    assert first.status == DecisionStatus.FILTERED and "persistence" in first.reasons[0]
    (second,) = strat.scan(tennis_group_ctx("0.45", "0.50", started=True, now=1003.0))
    assert second.status == DecisionStatus.CANDIDATE and second.kind == "basket"
    assert second.guaranteed_payout == D("1") and len(second.legs) == 2
    (gone,) = strat.scan(tennis_group_ctx("0.49", "0.50", started=True, now=1004.0))
    assert gone.status == DecisionStatus.FILTERED and "edge" in gone.reasons[0]


# ----------------------------------------------------------- market making
def mm_ctx(yes_bid: str, yes_ask: str, now: float, size: str = "30") -> ScanContext:
    m = gm("M", None)
    b = OrderBook("M")
    b.apply_snapshot(
        [(D(yes_bid), D(size))], [(1 - D(yes_ask), D(size))], source="t", received_at=now
    )
    return ScanContext(now=now, markets={"M": m}, books={"M": b})


def test_market_making_fills_only_on_trade_through() -> None:
    sim = MarketMakingSimulator(MarketMakingStrategySettings(), FEES)
    sim.on_cycle(mm_ctx("0.40", "0.50", 0.0))  # quotes: YES bid .41, NO bid .51 (YES ask .49)
    assert sim.markets["M"].quotes == {"yes": D("0.41"), "no": D("0.51")}
    sim.on_cycle(mm_ctx("0.30", "0.41", 1.0))  # ask only touches our 0.41 bid: NOT a fill
    assert sim.markets["M"].fills == []
    assert sim.markets["M"].quotes["yes"] == D("0.31")  # cancelled and re-quoted
    sim.on_cycle(mm_ctx("0.20", "0.30", 2.0))  # ask trades through 0.31 -> filled
    (fill,) = sim.markets["M"].fills
    assert fill.side == "yes" and fill.price == D("0.31") and fill.contracts == D("10")
    sim.on_cycle(mm_ctx("0.20", "0.21", 3.0))  # spread too narrow: quotes withdrawn
    assert sim.markets["M"].quotes == {}
    sim.settle("M", D("0"))  # the market kept moving against the filled bid
    summary = sim.summary()
    assert summary["simulated_fills"] == 1 and summary["realized_pnl_settled_markets"] < 0
    assert summary["maker_fees"] > 0  # maker fee is always charged (conservative)


def test_market_making_inventory_limit_and_research_only() -> None:
    sim = MarketMakingSimulator(MarketMakingStrategySettings(max_inventory=D("10")), FEES)
    state_ctx = mm_ctx("0.40", "0.50", 0.0)
    sim.on_cycle(state_ctx)
    sim.markets["M"].yes = D("10")  # already long the limit
    sim.on_cycle(mm_ctx("0.40", "0.50", 1.0))
    assert "yes" not in sim.markets["M"].quotes and "no" in sim.markets["M"].quotes
    with pytest.raises(ValueError, match="observe"):
        MarketMakingStrategySettings(mode="paper")


# ------------------------------------------------------------ tracker
def tracker() -> TennisTracker:
    return TennisTracker(
        WinProbabilityEstimator(Settings(_env_file=None).model), RiskSettings(), {}
    )


def test_tracker_rejects_out_of_order_scores() -> None:
    t = tracker()
    assert t.on_score("E", live_state(points=(3, 0)))
    older = replace(live_state(points=(1, 0)), received_ts=1005.0)
    assert not t.on_score("E", older)  # a stale cache edge served an older score
    assert t.scores["E"].score.points == (3, 0)
    assert t.score_info("E", 1006.0).issue == REGRESSION_ISSUE
    assert t.on_score("E", replace(live_state(points=(3, 1)), received_ts=1010.0))
    assert t.score_info("E", 1011.0).issue is None


def test_score_age_includes_source_age() -> None:
    t = tracker()
    t.on_score("E", replace(live_state(), source_age_s=7.0))
    assert t.score_info("E", 1003.0).age_s == pytest.approx(10.0)  # 3s local + 7s cache
    t2 = tracker()
    t2.on_score("E", live_state())  # unknown cache age: configured worst case (10s)
    assert t2.score_info("E", 1003.0).age_s == pytest.approx(13.0)


def test_mid_move_since_score_change_and_prior_guard() -> None:
    from kalshi_trader.market_data.discovery import TrackedMatch

    t = tracker()
    t.matches["E"] = TrackedMatch(
        "E",
        "S",
        "t",
        None,
        None,
        None,
        None,
        None,
        None,
        3,
        0.0,
        ("a", "b"),
        ("A", "B"),
        ("A", "B"),
        "ATP",
    )
    b = OrderBook("A")
    b.apply_snapshot([(D("0.60"), D("1"))], [(D("0.38"), D("1"))], source="t", received_at=990.0)
    t.on_book("A", b)
    t.on_score("E", live_state())
    b2 = OrderBook("A")
    b2.apply_snapshot([(D("0.75"), D("1"))], [(D("0.23"), D("1"))], source="t", received_at=1001.0)
    t.on_book("A", b2)
    assert t.score_info("E", 1002.0).mid_move == D("0.15")
    assert t.prior_usable("E", 900.0) and not t.prior_usable("E", 990.0)  # guard 60s


# --------------------------------------------------------------- splits
def test_event_splits_never_leak_and_respect_time() -> None:
    first_seen = {f"E{i}": float(i * 1000) for i in range(20)}
    split = split_events(first_seen, validation_fraction=0.2, test_fraction=0.3, embargo_s=1500)
    groups = [split.train, split.validation, split.test, split.dropped_by_embargo]
    assert sum(len(g) for g in groups) == 20
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not (a & b)
    assert max(first_seen[e] for e in split.train) < min(first_seen[e] for e in split.validation)
    assert max(first_seen[e] for e in split.validation) < min(first_seen[e] for e in split.test)
    assert split.dropped_by_embargo == {"E10", "E11", "E14", "E15"}


# ----------------------------------------------------------- model eval
def test_model_evaluation_is_event_weighted() -> None:
    preds = [
        # event A: model confident and right, market 0.5 - observed 3 times
        *[Prediction(float(i), "A", "A-1", 0.9, 1, 0.49, 0.51) for i in range(3)],
        # event B: model wrong, observed once
        Prediction(10.0, "B", "B-1", 0.9, 0, 0.49, 0.51),
    ]
    report = evaluate_predictions(preds)
    overall = report["overall"]
    # Each event weighs 1: Brier_model = (0.01 + 0.81) / 2; market = 0.25.
    assert overall["brier_model"] == pytest.approx(0.41)
    assert overall["brier_market"] == pytest.approx(0.25)
    assert not report["sufficient"] and "only 2 independent events" in report["note"]
    two_markets = [*preds, Prediction(0.0, "A", "A-2", 0.1, 0, 0.49, 0.51)]
    assert {p.market for p in one_market_per_event(two_markets)} == {"A-1", "B-1"}


# ------------------------------------------------------------ migration
def test_old_database_gets_new_columns(tmp_path) -> None:
    path = tmp_path / "old.db"
    db = Database(path)
    with db.engine.begin() as conn:  # simulate a database created before these columns
        conn.execute(text("DROP TABLE paper_trades"))
        conn.execute(
            text(
                "CREATE TABLE paper_trades (id INTEGER PRIMARY KEY, strategy VARCHAR(64), "
                "market_ticker VARCHAR(128), side VARCHAR(4), status VARCHAR(16))"
            )
        )
        conn.execute(text("INSERT INTO paper_trades VALUES (1, 's', 'm', 'yes', 'open')"))
    added = add_missing_columns(db.engine)
    assert "paper_trades.account_id" in added and "paper_trades.expected_pnl" in added
    with db.engine.connect() as conn:
        assert conn.execute(text("SELECT account_id FROM paper_trades")).scalar() == 1


# ------------------------------------------------------------ REST Age
async def test_live_data_records_http_cache_age() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"live_datas": [{"type": "tennis_tournament_singles", "milestone_id": "m1"}]}
        return httpx.Response(200, json=body, headers={"Age": "7"})

    client = KalshiRestClient(
        KalshiSettings(max_requests_per_second=1000, burst=1000),
        transport=httpx.MockTransport(handler),
    )
    (item,) = await client.get_live_data_batch(["m1"])
    assert item.http_cache_age_s == 7.0
    await client.close()


# ------------------------------------------------- replay capital recycling
def live_details(points: tuple[int, int], a: str, b: str) -> dict:
    return {
        "competitor1_id": a,
        "competitor2_id": b,
        "status": "live",
        "server": a,
        "winner": "",
        "advantage": "",
        "last_set_scoring_type": "StandardTiebreak",
        "competitor1_overall_score": 1,
        "competitor2_overall_score": 0,
        "competitor1_round_scores": [
            {"outcome": "winner", "score": 6},
            {"outcome": "ongoing", "score": 5},
        ],
        "competitor2_round_scores": [
            {"outcome": "loser", "score": 4},
            {"outcome": "ongoing", "score": 3},
        ],
        "competitor1_current_round_score": points[0],
        "competitor2_current_round_score": points[1],
    }


def add_match(s, event: str, t0: float, close: float) -> None:
    a, b = f"{event}-ca", f"{event}-cb"
    s.add(
        EventRow(
            event_ticker=event,
            series_ticker="KXATPMATCH",
            title=event,
            milestone_id=f"ms-{event}",
            tour="ATP",
            gender="men",
            best_of=3,
            tournament_name="ATP Replay",
            scheduled_start_ts=t0,
            first_competitor_id=a,
            second_competitor_id=b,
            first_seen_ts=0.0,
            updated_ts=0.0,
        )
    )
    for suffix, comp, result in (("A", a, "yes"), ("B", b, "no")):
        s.add(
            MarketRow(
                ticker=f"{event}-{suffix}",
                event_ticker=event,
                series_ticker="KXATPMATCH",
                player_name=suffix,
                competitor_id=comp,
                status="finalized",
                result=result,
                settlement_value=D("1") if result == "yes" else D("0"),
                close_ts=close,
                first_seen_ts=0.0,
                updated_ts=0.0,
                price_ranges_json='[{"start":"0","end":"1","step":"0.01"}]',
            )
        )
    s.add(
        MarketSnapshotRow(
            ticker=f"{event}-A",
            received_ts=t0,
            source="ws",
            status="active",
            depth_json=json.dumps({"yes_bids": [["0.90", "40"]], "no_bids": [["0.08", "40"]]}),
        )
    )
    s.add(
        ScoreSnapshotRow(
            milestone_id=f"ms-{event}",
            event_ticker=event,
            received_ts=t0 + 5,
            provider="kalshi_live_data",
            status="live",
            state_hash="h",
            state_json="{}",
            raw_json=json.dumps(live_details((40, 0), a, b)),
        )
    )


def test_replay_settles_chronologically_and_recycles_capital(tmp_path, settings: Settings) -> None:
    """$10 bankroll, one position at a time. Match 1 settles at t=500, before match 2's
    opportunity at t=605. Chronological settlement frees the cash and the position slot;
    settling only at the end of the replay (the old behaviour) would block match 2."""
    settings.model.uncertainty_samples = 32
    settings.model.prior_draws = 1000
    settings.strategies.late_match.mode = "paper"
    settings.paper.starting_balance = D("10")
    settings.paper.displayed_liquidity_fraction = D("1")
    r = settings.risk
    r.max_fraction_per_trade = r.max_aggregate_exposure_fraction = D("1")
    r.max_event_exposure_fraction = r.max_strategy_exposure_fraction = D("1")
    r.max_correlated_exposure_fraction = r.unvalidated_size_multiplier = D("1")
    r.kelly_fraction = D("1")
    r.max_open_positions = 1
    r.max_daily_loss = D("1000")
    db = Database(tmp_path / "rec.db")
    with db.session() as s:
        add_match(s, "KXATPMATCH-M1", 100.0, 500.0)
        add_match(s, "KXATPMATCH-M2", 600.0, 900.0)
    result = replay(settings, db)
    main = sorted(result.outcomes, key=lambda o: o.opened_ts)
    assert [o.event for o in main] == ["KXATPMATCH-M1", "KXATPMATCH-M2"]
    assert main[0].closed_ts == 500.0 and main[1].opened_ts == 605.0
    assert main[1].cost > 5  # sized from cash that only exists after the first settlement
    assert result.info["settlements_applied"] == 4
