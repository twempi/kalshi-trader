"""Risk limits, position sizing, paper execution, settlement, persistence, strategies."""

from __future__ import annotations

from decimal import Decimal as D

import pytest

from kalshi_trader.config import FeeSettings, LateMatchStrategySettings, PaperSettings, RiskSettings
from kalshi_trader.core.opportunity import DecisionStatus, FairValue, Leg, Opportunity
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import EMERGENCY_STOP, TRADING_ENABLED, Repository
from kalshi_trader.execution.paper import OrderIntent, PaperBroker
from kalshi_trader.market_data.discovery import TrackedMarket, TrackedMatch
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import ProbabilityEstimate, StrengthPrior
from kalshi_trader.risk.manager import (
    AccountSnapshot,
    MarketConditions,
    OpenExposure,
    RiskManager,
    kelly_fraction_binary,
)
from kalshi_trader.strategies.base import MatchContext
from kalshi_trader.strategies.late_match import LateMatchFavoriteStrategy
from kalshi_trader.tennis.state import MatchFormat, MatchState, MatchStatus, ScoreState

FEES = FeeSettings()


def opp(**overrides) -> Opportunity:
    leg = Leg(
        market_ticker="MKT-A",
        event_ticker="EV-1",
        side="yes",
        best_ask=D("0.90"),
        ask_size=D("100"),
        limit_price=D("0.90"),
    )
    base = dict(
        strategy="late_match",
        strategy_version="2+test",
        kind="directional",
        category="tennis",
        event_ticker="EV-1",
        legs=[leg],
        observed_ts=1000.0,
        fair=FairValue(0.97, 0.95, 0.98, "test", 1000.0),
        min_edge=D("0.02"),
        requires_score=True,
        correlation_group="tennis:ATP Test",
    )
    base.update(overrides)
    return Opportunity(**base)


def account(**overrides) -> AccountSnapshot:
    base = dict(
        cash=D("1000"),
        equity=D("1000"),
        peak_equity=D("1000"),
        day_start_equity=D("1000"),
        open_positions=0,
    )
    base.update(overrides)
    return AccountSnapshot(**base)


def conditions(**overrides) -> MarketConditions:
    base = dict(
        market_tradable=True,
        exchange_trading_active=True,
        book_age_s=1.0,
        score_age_s=12.0,
        spread=D("0.01"),
        depth_at_limit=D("100"),
        trades_in_market=0,
        trades_in_event=0,
        has_open_position_in_event=False,
    )
    base.update(overrides)
    return MarketConditions(**base)


@pytest.fixture
def risk(memory_db: Database) -> RiskManager:
    return RiskManager(RiskSettings(), Repository(memory_db), FEES.taker_rate)


# ----------------------------------------------------------------------- risk
def test_clean_opportunity_passes_with_per_trade_budget(risk: RiskManager) -> None:
    assert risk.check_market(opp(), conditions()) == []
    assert risk.check_portfolio(account()) == []
    budget = risk.budget(opp(), account())
    # 2% of $1000 equity = $20 is the tightest cap (event cap is 3% = $30)
    assert budget.max_loss == D("20") and budget.binding == "per-trade fraction"
    # the per-trade limit scales with equity unless a fixed dollar ceiling is configured
    assert risk.budget(opp(), account(equity=D("2000"), peak_equity=D("2000"))).max_loss == D("40")
    risk.settings.max_position_cost = D("25")
    capped = risk.budget(opp(), account(equity=D("2000"), peak_equity=D("2000")))
    assert capped.max_loss == D("25") and capped.binding == "per-trade cap"


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"book_age_s": 30.0}, "order book stale"),
        ({"score_age_s": 99.0}, "score data stale"),
        ({"score_age_s": None}, "score data stale"),
        ({"score_unchanged_s": 900.0}, "score unchanged"),
        ({"mid_move_since_score_change": D("0.20")}, "score feed likely behind"),
        ({"score_issue": "score went backwards"}, "score feed"),
        ({"spread": D("0.10")}, "spread"),
        ({"depth_at_limit": D("3")}, "liquidity"),
        ({"trades_in_market": 1}, "max trades for this market"),
        ({"trades_in_event": 1}, "max trades for this event"),
        ({"has_open_position_in_event": True}, "already exposed"),
        ({"market_tradable": False}, "market not active"),
        ({"exchange_trading_active": False}, "exchange trading inactive"),
    ],
)
def test_rejects_each_market_limit(risk: RiskManager, override: dict, reason: str) -> None:
    reasons = risk.check_market(opp(), conditions(**override))
    assert any(reason in r for r in reasons), reasons


def test_score_checks_only_apply_to_score_strategies(risk: RiskManager) -> None:
    assert risk.check_market(opp(requires_score=False), conditions(score_age_s=None)) == []


def test_rejects_portfolio_limits_and_escalates(risk: RiskManager) -> None:
    assert risk.check_portfolio(account(open_positions=5))
    daily = risk.check_portfolio(account(equity=D("940"), cash=D("940")))
    assert any("daily loss" in r for r in daily)
    assert not risk.emergency_stopped()
    dd = risk.check_portfolio(
        account(equity=D("800"), cash=D("800"), day_start_equity=D("800"), peak_equity=D("1000"))
    )
    assert any("drawdown" in r for r in dd)
    assert risk.emergency_stopped()  # persisted, requires an operator to clear
    assert "emergency stop active" in risk.check_portfolio(account())
    risk.clear_emergency_stop()
    assert risk.check_portfolio(account()) == []


def test_shadow_checks_ignore_loss_limits(risk: RiskManager) -> None:
    # A shadow ledger past the drawdown and daily-loss limits keeps collecting evidence.
    dd = account(equity=D("800"), cash=D("800"), peak_equity=D("1000"))
    assert risk.check_portfolio(dd, escalate=False, safety=False) == []
    assert not risk.emergency_stopped()


def test_exposure_caps_shrink_the_budget(risk: RiskManager) -> None:
    held = (
        OpenExposure("late_match", "EV-1", "MKT-X", D("25"), "tennis:ATP Test"),
        OpenExposure("tennis_model", "EV-2", "MKT-Y", D("20"), "tennis:Other"),
    )
    # event EV-1 already has $25 of max loss: 3% x 1000 - 25 = $5 left
    budget = risk.budget(opp(), account(exposures=held))
    assert budget.max_loss == D("5") and budget.binding == "event exposure"
    # correlated group already at $85 > 8% x 1000: no headroom at all
    crowded = (*held, OpenExposure("tennis_model", "EV-4", "MKT-Z", D("60"), "tennis:ATP Test"))
    blocked = risk.budget(opp(event_ticker="EV-3"), account(exposures=crowded))
    assert blocked.max_loss == 0 and "correlated exposure" in blocked.reasons[0]
    snap = account(exposures=crowded)
    # unrelated tournament: per-trade cap binds again
    free = risk.budget(opp(event_ticker="EV-9", correlation_group="tennis:Other"), snap)
    assert free.max_loss == D("20")


def test_aggregate_exposure_cap(risk: RiskManager) -> None:
    held = tuple(
        OpenExposure("s", f"EX-{i}", f"M-{i}", D("29"), f"g{i}") for i in range(10)
    )  # $290 of $300 aggregate cap used
    budget = risk.budget(opp(correlation_group="new"), account(exposures=held))
    assert budget.max_loss == D("10") and budget.binding == "aggregate exposure"


def test_emergency_stop_and_pause_persist_across_restart(tmp_path) -> None:
    path = tmp_path / "state.db"
    first = RiskManager(RiskSettings(), Repository(Database(path)), FEES.taker_rate)
    first.emergency_stop("test")
    first.pause("maintenance")
    second = RiskManager(RiskSettings(), Repository(Database(path)), FEES.taker_rate)
    assert second.emergency_stopped() and not second.trading_enabled()


def test_repeated_errors_pause_trading(risk: RiskManager) -> None:
    for _ in range(4):
        risk.record_error("strategies", "boom")
    assert risk.trading_enabled()
    risk.record_error("strategies", "boom")
    assert not risk.trading_enabled()
    assert "paper trading paused" in risk.check_portfolio(account())
    risk.resume()
    assert risk.trading_enabled()


def test_kelly_and_no_martingale(risk: RiskManager) -> None:
    assert kelly_fraction_binary(0.95, D("0.90")) == pytest.approx(D("0.5"))
    assert kelly_fraction_binary(0.85, D("0.90")) == 0
    big = risk.budget(opp(), account())
    after_loss = risk.budget(opp(), account(equity=D("900"), cash=D("900")))
    assert after_loss.max_loss < big.max_loss  # budgets shrink after losses, never grow


# ---------------------------------------------------------------------- paper
def book_with_asks() -> OrderBook:
    book = OrderBook("MKT-A")
    book.apply_snapshot(
        [(D("0.88"), D("50"))], [(D("0.10"), D("20")), (D("0.09"), D("40"))], source="test"
    )
    return book  # YES asks: 0.90 x20, 0.91 x40


def intent(**overrides) -> OrderIntent:
    base = dict(
        client_order_id="late_match:MKT-A:yes:1",
        strategy="late_match",
        market_ticker="MKT-A",
        event_ticker="EV-1",
        side="yes",
        limit_price=D("0.91"),
        contracts=D("30"),
        max_cash=D("100"),
        match_title="A vs B",
        player_name="A",
        score_text="6-4 5-3 (40-0)",
        model_prob=0.98,
        robust_prob=0.95,
        ev_per_contract=0.04,
    )
    base.update(overrides)
    return OrderIntent(**base)


def test_paper_buy_settle_and_account(memory_db: Database) -> None:
    broker = PaperBroker(memory_db, PaperSettings(displayed_liquidity_fraction=D("1")), FEES)
    report = broker.execute_buy(intent(), book_with_asks())
    assert report.status == "filled" and report.fill.filled == D("30")
    acct = broker.account()
    debit = report.fill.notional + report.fill.fee
    assert acct.cash == D("1000") - debit and acct.fees_paid == report.fill.fee
    trade = broker.open_trades()[0]
    assert trade.entry_price == report.fill.avg_price and trade.score_at_entry
    settled = broker.settle_market("MKT-A", "yes", D("1"))
    assert len(settled) == 1 and settled[0].realized_pnl == D("30") - debit
    acct = broker.account()
    assert acct.cash == D("1000") - debit + D("30") and acct.realized_pnl == D("30") - debit
    assert broker.open_trades() == []
    assert broker.settle_market("MKT-A", "yes", D("1")) == []  # idempotent


def test_paper_loss_and_fair_value_settlement(memory_db: Database) -> None:
    broker = PaperBroker(memory_db, PaperSettings(displayed_liquidity_fraction=D("1")), FEES)
    broker.execute_buy(intent(), book_with_asks())
    lost = broker.settle_market("MKT-A", "no", None)[0]
    assert lost.payout == 0 and lost.realized_pnl < 0
    broker.execute_buy(
        intent(client_order_id="x:2", market_ticker="MKT-B", side="no", event_ticker="EV-2"),
        book_with_asks(),
    )
    void = broker.settle_market("MKT-B", "", D("0.37"))[0]  # walkover fair-price settlement
    assert void.payout == (void.contracts * (1 - D("0.37"))).quantize(D("0.01"))


def test_duplicate_orders_prevented_across_restart(tmp_path) -> None:
    path = tmp_path / "paper.db"
    broker = PaperBroker(Database(path), PaperSettings(), FEES)
    assert broker.execute_buy(intent(), book_with_asks()).status in ("filled", "partial")
    restarted = PaperBroker(Database(path), PaperSettings(), FEES)
    again = restarted.execute_buy(intent(), book_with_asks())
    assert again.status == "duplicate"
    assert len(restarted.open_trades()) == 1
    assert restarted.account().cash == broker.account().cash


def test_unfilled_and_partial_orders(memory_db: Database) -> None:
    broker = PaperBroker(memory_db, PaperSettings(displayed_liquidity_fraction=D("0.5")), FEES)
    unfilled = broker.execute_buy(intent(limit_price=D("0.85")), book_with_asks())
    assert unfilled.status == "unfilled" and broker.open_trades() == []
    partial = broker.execute_buy(
        intent(client_order_id="p:1", contracts=D("100")), book_with_asks()
    )
    assert partial.status == "partial" and partial.fill.filled == D("30")  # 50% of 20+40


def test_early_exit_realises_pnl(memory_db: Database) -> None:
    broker = PaperBroker(memory_db, PaperSettings(displayed_liquidity_fraction=D("1")), FEES)
    report = broker.execute_buy(intent(contracts=D("10")), book_with_asks())
    exit_book = OrderBook("MKT-A")
    exit_book.apply_snapshot([(D("0.99"), D("100"))], [], source="test")
    out = broker.execute_sell(report.trade_id, exit_book, D("0.99"), "exit:1")
    assert out.status == "filled"
    with memory_db.session() as s:
        from kalshi_trader.database.tables import PaperTradeRow

        trade = s.get(PaperTradeRow, report.trade_id)
        assert trade.status == "closed" and trade.realized_pnl is not None
        assert (
            trade.realized_pnl
            == trade.exit_proceeds - trade.exit_fee - trade.entry_cost - trade.entry_fee
        )


def test_mark_to_bid_snapshot(memory_db: Database) -> None:
    broker = PaperBroker(memory_db, PaperSettings(displayed_liquidity_fraction=D("1")), FEES)
    broker.execute_buy(intent(contracts=D("10")), book_with_asks())
    snap = broker.snapshot({("MKT-A", "yes"): D("0.95")})
    assert snap.open_positions == 1 and snap.equity == snap.cash + D("9.50")


# ------------------------------------------------------------------- strategy
def make_context(
    points=(3, 0),
    games=(5, 3),
    ask_no_bid=D("0.08"),
    p_robust=0.985,
    status=MatchStatus.LIVE,
    fmt_known=True,
    delayed=False,
) -> MatchContext:
    fmt = MatchFormat(best_of=3, known=fmt_known)
    state = ScoreState(
        fmt=fmt, sets=(1, 0), games=games, points=points, server=0, completed_sets=((6, 4),)
    )
    score = MatchState(
        provider="test",
        match_id="m",
        player_ids=("a", "b"),
        status=status,
        received_ts=1000.0,
        fmt=fmt,
        score=state,
        source_delayed=delayed,
    )
    estimate = ProbabilityEstimate(
        p1_point=0.995,
        p1_mean=0.993,
        p1_std=0.005,
        p1_low=0.99,
        p1_high=0.998,
        p1_robust_low=p_robust,
        p1_robust_high=0.999,
        expected_remaining_games=1.0,
        serve_p1=0.66,
        serve_p2=0.62,
        prior=StrengthPrior(0.64, 0.02, 0.025, 0.03, "pre_match_price", 0.6),
        latency_points=1,
        samples=64,
        credible_quantile=0.1,
        effective_sample_size=900.0,
    )
    match = TrackedMatch(
        event_ticker="EV-1",
        series_ticker="KXATPMATCH",
        title="A vs B",
        competition="ATP Test",
        milestone_id="m",
        tour="ATP",
        gender="men",
        round_name="R16",
        tournament_name="ATP Test",
        best_of=3,
        scheduled_start_ts=0.0,
        player_ids=("a", "b"),
        market_tickers=("MKT-A", "MKT-B"),
        player_names=("A", "B"),
        tour_key="ATP",
    )
    markets = {
        t: TrackedMarket(
            ticker=t,
            event_ticker="EV-1",
            series_ticker="KXATPMATCH",
            player_name=n,
            competitor_id=c,
            status="active",
        )
        for t, n, c in (("MKT-A", "A", "a"), ("MKT-B", "B", "b"))
    }
    book_a = OrderBook("MKT-A")
    book_a.apply_snapshot([(D("0.90"), D("40"))], [(ask_no_bid, D("40"))], source="t")
    book_b = OrderBook("MKT-B")
    book_b.apply_snapshot([(D("0.05"), D("40"))], [(D("0.90"), D("40"))], source="t")
    return MatchContext(
        now=1001.0,
        match=match,
        markets=markets,
        books={"MKT-A": book_a, "MKT-B": book_b},
        score=score,
        estimate=estimate,
        exchange_trading_active=True,
    )


def test_late_match_signals_when_edge_survives_conservative_bound() -> None:
    strat = LateMatchFavoriteStrategy(LateMatchStrategySettings(), FEES)
    opps = strat.evaluate_match(make_context())
    assert len(opps) == 1
    o = opps[0]
    assert o.status == DecisionStatus.CANDIDATE
    assert o.market_ticker == "MKT-A" and o.side == "yes" and o.primary.best_ask == D("0.92")
    assert o.primary.limit_price >= o.primary.best_ask
    assert o.fair is not None and o.fair.p_low == 0.985
    # limit is capped so every fill still clears the minimum edge
    assert D(str(o.fair.p_low)) - o.primary.limit_price >= D("0.02")


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"p_robust": 0.93}, "robust p"),  # conservative probability too low for the price
        ({"ask_no_bid": D("0.01")}, "YES MKT-A: spread"),  # ask 0.99: spread too wide
        ({"fmt_known": False}, "unknown format"),
        ({"delayed": True}, "delayed"),
        ({"games": (2, 1), "points": (0, 0)}, "games from victory"),  # not close to victory
    ],
)
def test_late_match_explains_why_it_stays_silent(kwargs: dict, reason: str) -> None:
    strat = LateMatchFavoriteStrategy(LateMatchStrategySettings(), FEES)
    opps = strat.evaluate_match(make_context(**kwargs))
    assert len(opps) == 1 and opps[0].status == DecisionStatus.FILTERED
    assert reason in opps[0].reasons[0]


def test_late_match_ignores_matches_not_in_play() -> None:
    strat = LateMatchFavoriteStrategy(LateMatchStrategySettings(), FEES)
    assert strat.evaluate_match(make_context(status=MatchStatus.SUSPENDED)) == []


def test_flags_persist_in_repository(memory_db: Database) -> None:
    repo = Repository(memory_db)
    assert repo.get_flag(TRADING_ENABLED, default=True)
    repo.set_flag(EMERGENCY_STOP, True)
    assert repo.get_flag(EMERGENCY_STOP)
