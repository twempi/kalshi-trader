"""SYNTHETIC checks for the first-set-loss experiment, not performance evidence."""

from dataclasses import replace
from decimal import Decimal as D

import pytest
from fastapi.testclient import TestClient

from kalshi_trader.config import FeeSettings, Settings, TennisModelStrategySettings
from kalshi_trader.core.opportunity import DecisionStatus
from kalshi_trader.core.pipeline import TradingCore
from kalshi_trader.dashboard.app import create_app
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.strategies import build_strategies
from kalshi_trader.strategies.base import MatchContext, ScanContext, ScoreInfo
from kalshi_trader.strategies.stat_mispricing import TennisModelStrategy
from tests.test_trading import make_context


def comeback_context(favourite: int = 0) -> MatchContext:
    ctx = make_context()
    ctx.pre_match_p1 = 0.75 if favourite == 0 else 0.25
    ctx.score = replace(
        ctx.score,
        score=replace(
            ctx.score.score,
            sets=(0, 1) if favourite == 0 else (1, 0),
            completed_sets=((2, 6),) if favourite == 0 else ((6, 2),),
            games=(0, 0),
            points=(0, 0),
        ),
    )
    ctx.estimate = replace(
        ctx.estimate,
        p1_point=0.6 if favourite == 0 else 0.4,
        p1_low=0.52 if favourite == 0 else 0.32,
        p1_high=0.68 if favourite == 0 else 0.48,
        p1_robust_low=0.5 if favourite == 0 else 0.3,
        p1_robust_high=0.7 if favourite == 0 else 0.5,
    )
    for player, ticker in enumerate(ctx.match.market_tickers):
        bid, opposite_bid = ("0.38", "0.60") if player == favourite else ("0.58", "0.40")
        ctx.books[ticker].apply_snapshot(
            [(D(bid), D("100"))],
            [(D(opposite_bid), D("100"))],
            source="test",
            received_at=1000,
        )
    ctx.score_info = ScoreInfo(age_s=1, unchanged_s=1, mid_move=D("0"))
    return ctx


def comeback_strategy() -> TennisModelStrategy:
    return TennisModelStrategy(
        TennisModelStrategySettings(
            mode="observe", entry_pattern="first_set_comeback", exit_edge=D("0.03")
        ),
        FeeSettings(),
        name="tennis_model_comeback",
    )


@pytest.mark.parametrize("favourite", [0, 1])
def test_backs_original_favourite_after_losing_set_one(favourite: int) -> None:
    ctx = comeback_context(favourite)
    (opp,) = comeback_strategy().evaluate_match(ctx)
    assert opp.status == DecisionStatus.CANDIDATE
    assert opp.market_ticker == ctx.match.market_tickers[favourite] and opp.side == "yes"
    assert opp.provenance["entry_pattern"] == "first_set_comeback"
    assert opp.provenance["pre_match_p1"] == ctx.pre_match_p1
    assert opp.reason.startswith("first-set comeback:")


def test_can_back_favourite_through_no_on_opponents_contract() -> None:
    ctx = comeback_context(favourite=1)
    ctx.books["MKT-B"].apply_snapshot(
        [(D("0.48"), D("100"))],
        [(D("0.50"), D("100"))],
        source="test",
        received_at=1000,
    )
    (opp,) = comeback_strategy().evaluate_match(ctx)
    assert opp.status == DecisionStatus.CANDIDATE
    assert opp.market_ticker == "MKT-A" and opp.side == "no"  # backs player B at 42c


@pytest.mark.parametrize("sets", [(0, 0), (1, 0), (1, 1)])
def test_no_entry_in_first_set_after_favourite_won_or_in_deciding_set(sets) -> None:
    ctx = comeback_context()
    ctx.score = replace(ctx.score, score=replace(ctx.score.score, sets=sets))
    (note,) = comeback_strategy().evaluate_match(ctx)
    assert (
        note.status == DecisionStatus.FILTERED
        and "entries are limited to set two" in note.reasons[0]
    )


@pytest.mark.parametrize("pre_match", [None, 0, 1, float("nan")])
def test_requires_a_valid_pre_match_probability(pre_match) -> None:
    ctx = comeback_context()
    ctx.pre_match_p1 = pre_match
    (note,) = comeback_strategy().evaluate_match(ctx)
    assert note.status == DecisionStatus.UNSUPPORTED and "pre-match probability" in note.reasons[0]


def test_skips_evenly_matched_players_and_best_of_five() -> None:
    ctx = comeback_context()
    ctx.pre_match_p1 = 0.5
    assert "no unique pre-match favourite" in comeback_strategy().evaluate_match(ctx)[0].reasons[0]
    ctx.pre_match_p1 = 0.75
    ctx.score = replace(
        ctx.score, score=replace(ctx.score.score, fmt=replace(ctx.score.fmt, best_of=5))
    )
    assert "best-of-three" in comeback_strategy().evaluate_match(ctx)[0].reasons[0]


def test_pattern_never_buys_opponent_or_overrides_the_fee_and_edge_gate() -> None:
    ctx = comeback_context()
    ctx.estimate = replace(ctx.estimate, p1_robust_low=0.43)
    assert comeback_strategy().evaluate_match(ctx)[0].status == DecisionStatus.FILTERED
    # The current model now favours B, but the original favourite was A. This experiment
    # must not turn into a bet on B merely because B offers a larger model edge.
    ctx.estimate = replace(ctx.estimate, p1_point=0.1, p1_robust_low=0.05, p1_robust_high=0.2)
    baseline = TennisModelStrategy(TennisModelStrategySettings(), FeeSettings())
    assert baseline.evaluate_match(ctx)[0].status == DecisionStatus.CANDIDATE
    assert comeback_strategy().evaluate_match(ctx)[0].status == DecisionStatus.FILTERED


def test_variant_trades_only_its_shadow_and_can_exit_after_set_two(settings: Settings) -> None:
    settings.strategies.tennis_model_variants = {
        "comeback": {"entry_pattern": "first_set_comeback", "exit_edge": "0.03", "mode": "paper"}
    }
    variant = next(s for s in build_strategies(settings) if s.name == "tennis_model_comeback")
    assert variant.mode == "observe"
    db = Database(":memory:")
    core = TradingCore(settings, db, Repository(db), [variant], clock=lambda: 1001)
    ctx = comeback_context()
    scan = ScanContext(now=1001, markets=ctx.markets, books=ctx.books, matches=[ctx])
    core.run_cycle(scan)
    shadow = core.shadows[variant.name]
    assert len(shadow.open_trades()) == 1
    assert core.broker.account().cash == D("1000") and not core.broker.all_trades()

    ctx.score = replace(ctx.score, score=replace(ctx.score.score, sets=(1, 1)))
    ctx.estimate = replace(ctx.estimate, p1_point=0.4)
    ctx.books["MKT-A"].apply_snapshot(
        [(D("0.65"), D("100"))],
        [(D("0.33"), D("100"))],
        source="test",
        received_at=1001,
    )
    # The entry filter no longer matches, but it must not prevent selling an existing bet.
    core.run_cycle(scan)
    (trade,) = shadow.all_trades()
    assert trade.status == "closed" and trade.exit_contracts == trade.entry_contracts
    assert trade.realized_pnl > 0 and trade.exit_fee > 0
    assert not core.broker.all_trades() and core.broker.account().cash == D("1000")
    core.refresh_evidence(1001)
    client = TestClient(create_app(settings, db))
    reported = next(
        s for s in client.get("/api/strategies").json()["strategies"] if s["name"] == variant.name
    )
    assert reported["mode"] == "observe" and reported["evidence"]["trades"] == 1
    assert not reported["evidence"]["validated"]
    (comparison,) = client.get("/api/research/early-exits").json()["strategies"]
    assert comparison["strategy"] == variant.name and comparison["pending_trades"] == 1
    assert client.get("/api/trades").json()["trades"] == []
