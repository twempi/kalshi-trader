"""Probability model, estimator, fees, fill simulation and expected value."""

from __future__ import annotations

import random
from decimal import Decimal as D

import pytest

from kalshi_trader.api.schemas import PriceRange
from kalshi_trader.config import ModelSettings
from kalshi_trader.execution.fees import (
    buy_cash_debit,
    fee_per_contract_estimate,
    order_trade_fee,
    raw_trade_fee,
    sell_cash_credit,
)
from kalshi_trader.execution.fills import simulate_buy, simulate_sell, snap_price_down
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.models.estimator import WinProbabilityEstimator, adjust_for_retirement
from kalshi_trader.models.ev import evaluate_buy, max_price_for_edge
from kalshi_trader.models.markov import (
    MarkovMatchModel,
    implied_serve_difference,
    pre_match_probability,
)
from kalshi_trader.tennis.state import MatchFormat, ScoreState, ServeStats

BO3 = MatchFormat(best_of=3)
FORMATS = [
    BO3,
    MatchFormat(best_of=5, final_set_tiebreak_points=10),
    MatchFormat(final_set="advantage"),
    MatchFormat(final_set="match_tiebreak", final_set_tiebreak_points=10),
]


# --------------------------------------------------------------------- Markov
def test_known_hold_probability() -> None:
    assert MarkovMatchModel((0.6, 0.6), BO3).hold_prob(0) == pytest.approx(0.73565, abs=1e-4)


@pytest.mark.parametrize("fmt", FORMATS)
def test_symmetric_players_are_even(fmt: MatchFormat) -> None:
    assert pre_match_probability((0.62, 0.62), fmt) == pytest.approx(0.5, abs=1e-9)


def test_longer_matches_favour_the_stronger_player() -> None:
    bo3 = pre_match_probability((0.65, 0.62), BO3)
    bo5 = pre_match_probability((0.65, 0.62), MatchFormat(best_of=5))
    assert 0.5 < bo3 < bo5 < 1


def _random_state(rng: random.Random, fmt: MatchFormat) -> ScoreState:
    state = ScoreState(fmt=fmt, server=rng.randint(0, 1))
    for _ in range(rng.randint(0, 220)):
        nxt = state.apply_point(rng.randint(0, 1))
        if nxt.winner is not None:
            break
        state = nxt
    return state


@pytest.mark.parametrize("fmt", FORMATS)
def test_win_probability_is_a_martingale(fmt: MatchFormat) -> None:
    """P(now) must equal the probability-weighted average after the next point. This
    checks every transition, deuce closed form, tiebreak rotation and set hand-over."""
    rng = random.Random(1234)
    model = MarkovMatchModel((0.66, 0.59), fmt)
    for _ in range(150):
        state = _random_state(rng, fmt)
        p_now = model.win_probability(state)[0]
        server = state.server
        p0_point = model.p[0] if server == 0 else 1 - model.p[1]
        p_next = (
            p0_point * model.win_probability(state.apply_point(0))[0]
            + (1 - p0_point) * model.win_probability(state.apply_point(1))[0]
        )
        assert p_now == pytest.approx(p_next, abs=1e-9), state


def test_finished_and_near_finished_states() -> None:
    model = MarkovMatchModel((0.64, 0.64), BO3)
    done = ScoreState(fmt=BO3, sets=(2, 0))
    assert model.win_probability(done) == (1.0, 0.0)
    serving_for_match = ScoreState(fmt=BO3, sets=(1, 0), games=(5, 3), points=(3, 0), server=0)
    p, games_left = model.win_probability(serving_for_match)
    assert 0.99 < p < 1 and games_left < 2


def test_implied_serve_difference_round_trips() -> None:
    for target in (0.25, 0.5, 0.8, 0.95):
        d = implied_serve_difference(target, 0.64, BO3)
        p = (0.64 + d / 2, 0.64 - d / 2)
        avg = 0.5 * (pre_match_probability(p, BO3, 0) + pre_match_probability(p, BO3, 1))
        assert avg == pytest.approx(target, abs=1e-4)


# ------------------------------------------------------------------ estimator
def test_retirement_adjustment_pulls_towards_half() -> None:
    assert adjust_for_retirement(0.99, 0, 0.001) == 0.99
    adjusted = adjust_for_retirement(0.99, 10, 0.001)
    assert 0.5 < adjusted < 0.99
    assert adjust_for_retirement(0.5, 10, 0.01) == pytest.approx(0.5)


def test_estimator_is_reproducible_and_ordered() -> None:
    est = WinProbabilityEstimator(ModelSettings(uncertainty_samples=64, prior_draws=1000))
    state = ScoreState(fmt=BO3, sets=(1, 0), games=(4, 2), points=(1, 1), server=0)
    prior = est.build_prior("ATP", BO3, 0.70)
    a = est.estimate(state, prior, (ServeStats(40, 15), ServeStats(30, 25)), "k")
    b = est.estimate(state, prior, (ServeStats(40, 15), ServeStats(30, 25)), "k")
    assert a == b
    assert a.p1_robust_low <= a.p1_low <= a.p1_point <= a.p1_high <= a.p1_robust_high
    p2 = a.for_player(1)
    assert p2.point == pytest.approx(1 - a.p1_point) and p2.robust_low <= p2.point


def test_price_prior_and_live_stats_move_the_estimate() -> None:
    est = WinProbabilityEstimator(ModelSettings(uncertainty_samples=64, prior_draws=2000))
    state = ScoreState(fmt=BO3)
    neutral = est.estimate(state, est.build_prior("ATP", BO3, None), seed_key="a")
    assert neutral.p1_point == pytest.approx(0.5, abs=0.02)
    favoured = est.estimate(state, est.build_prior("ATP", BO3, 0.8), seed_key="a")
    assert favoured.p1_point == pytest.approx(0.8, abs=0.03)
    dominant_serving = est.estimate(
        state, est.build_prior("ATP", BO3, None), (ServeStats(55, 10), ServeStats(30, 35)), "a"
    )
    assert dominant_serving.serve_p1 > neutral.serve_p1
    assert dominant_serving.p1_point > 0.6
    # A price-anchored prior is tighter than a tour-average one.
    assert favoured.prior.delta_sd < neutral.prior.delta_sd


def test_estimator_ignores_implausible_stats() -> None:
    est = WinProbabilityEstimator(ModelSettings(uncertainty_samples=32, prior_draws=1000))
    prior = est.build_prior("WTA", BO3, None)
    clean = est.estimate(ScoreState(fmt=BO3), prior, seed_key="x")
    junk = est.estimate(ScoreState(fmt=BO3), prior, (ServeStats(5000, 1), None), seed_key="x")
    assert junk.p1_point == pytest.approx(clean.p1_point)


# ----------------------------------------------------------------------- fees
def test_fee_formula_examples() -> None:
    rate = D("0.07")
    assert raw_trade_fee(D("0.50"), D("100"), rate) == D("1.75")
    assert order_trade_fee([(D("0.50"), D("100"))], rate) == D("1.750000")
    # 1 contract at 93c: 0.07*0.93*0.07 = 0.004557 -> rounds up to a full cent
    debit, fee = buy_cash_debit(D("0.93"), order_trade_fee([(D("0.93"), D("1"))], rate))
    assert (debit, fee) == (D("0.94"), D("0.01"))
    assert fee_per_contract_estimate(D("0.93"), D("10"), rate) == D("0.005")
    # multiplier and maker rate
    assert order_trade_fee([(D("0.50"), D("100"))], D("0.0175")) == D("0.437500")
    assert order_trade_fee([(D("0.50"), D("10"))], rate, D("0")) == D("0")
    with pytest.raises(ValueError):
        raw_trade_fee(D("1.20"), D("1"), rate)


def test_sub_cent_notional_is_aligned_to_cents() -> None:
    # 3 contracts at 0.955 = 2.865 notional (sub-cent tick near the edge)
    trade_fee = order_trade_fee([(D("0.955"), D("3"))], D("0.07"))
    debit, fee = buy_cash_debit(D("2.865"), trade_fee)
    assert debit == D("2.88") and fee == debit - D("2.865")
    credit, sell_fee = sell_cash_credit(D("2.865"), trade_fee)
    assert credit == D("2.85") and sell_fee == D("0.015")


# ---------------------------------------------------------------------- fills
ASKS = [(D("0.93"), D("10")), (D("0.94"), D("20")), (D("0.96"), D("100"))]


def test_buy_walks_book_and_respects_limit() -> None:
    fill = simulate_buy(ASKS, D("0.94"), D("50"), fee_rate=D("0.07"))
    assert fill.filled == D("30") and fill.status == "partial"
    assert fill.notional == D("9.30") + D("18.80")
    assert fill.avg_price == (D("9.30") + D("18.80")) / 30
    assert fill.cash_change == -(fill.notional + fill.fee)


def test_buy_liquidity_haircut_whole_contracts_and_cash_cap() -> None:
    fill = simulate_buy(ASKS, D("0.99"), D("15"), fee_rate=D("0.07"), liquidity_fraction=D("0.5"))
    assert [(f.price, f.contracts) for f in fill.fills] == [
        (D("0.93"), D("5")),
        (D("0.94"), D("10")),
    ]
    capped = simulate_buy(ASKS, D("0.99"), D("15"), fee_rate=D("0.07"), max_cash=D("5.00"))
    assert -capped.cash_change <= D("5.00") and capped.filled == D("5")
    none = simulate_buy(ASKS, D("0.90"), D("5"), fee_rate=D("0.07"))
    assert none.status == "unfilled" and none.cash_change == 0


def test_sell_walks_bids_down_to_limit() -> None:
    bids = [(D("0.99"), D("4")), (D("0.97"), D("10"))]
    fill = simulate_sell(bids, D("0.98"), D("10"), fee_rate=D("0.07"))
    assert fill.filled == D("4") and fill.cash_change > 0


def test_snap_price_to_tick_grid() -> None:
    tapered = [
        PriceRange(start=D("0"), end=D("0.10"), step=D("0.001")),
        PriceRange(start=D("0.10"), end=D("0.90"), step=D("0.01")),
        PriceRange(start=D("0.90"), end=D("1"), step=D("0.001")),
    ]
    assert snap_price_down(D("0.9567"), tapered) == D("0.956")
    assert snap_price_down(D("0.4567"), tapered) == D("0.45")
    assert snap_price_down(D("0.4567"), None) == D("0.45")


# ------------------------------------------------------------------------- EV
def test_ev_uses_executable_vwap_and_fees() -> None:
    book = OrderBook("T")
    book.apply_snapshot(
        [(D("0.90"), D("50"))], [(D("0.07"), D("10")), (D("0.06"), D("10"))], source="test"
    )
    result = evaluate_buy(book, "yes", 0.97, 0.95, D("20"), fee_rate=D("0.07"))
    # 10 @ 0.93 + 10 @ 0.94, fee ceil(0.07*(10*.93*.07 + 10*.94*.06)) = 0.09
    assert result.fill.avg_price == D("0.935")
    assert result.cost_per_contract == D("0.9395")
    assert result.ev_point == pytest.approx(0.97 - 0.9395)
    assert result.ev_conservative == pytest.approx(0.95 - 0.9395)


def test_max_price_for_edge() -> None:
    cap = max_price_for_edge(0.96, D("0.02"), fee_rate=D("0.07"))
    assert cap is not None
    fee = fee_per_contract_estimate(cap, D("10"), D("0.07"))
    assert D("0.96") - cap - fee >= D("0.02")
    above = cap + D("0.01")
    assert D("0.96") - above - fee_per_contract_estimate(above, D("10"), D("0.07")) < D("0.02")
    assert max_price_for_edge(0.01, D("0.02"), fee_rate=D("0.07")) is None
