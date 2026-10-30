"""Backtest metrics, significance tests and look-ahead protection (synthetic data only)."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from decimal import Decimal as D

import pytest

from kalshi_trader.api.schemas import Candlestick, Market
from kalshi_trader.backtesting.historical import MarketHistory, simulate_band
from kalshi_trader.backtesting.metrics import (
    TradeOutcome,
    compute_metrics,
    equity_curve,
    max_drawdown,
    win_rate_by_bucket,
)
from kalshi_trader.backtesting.replay import replay
from kalshi_trader.backtesting.stats import (
    calibration_null_pvalue,
    significance_report,
    wilson_interval,
)
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import EventRow, MarketRow, MarketSnapshotRow, ScoreSnapshotRow


def outcome(pnl: float, ts: float, price: float = 0.9, event: str | None = None) -> TradeOutcome:
    return TradeOutcome(
        opened_ts=ts,
        closed_ts=ts + 1,
        strategy="s",
        entry_price=price,
        contracts=10,
        cost=price * 10 + 0.05,
        pnl=pnl,
        fees=0.05,
        event=event,
    )


def test_metrics_on_synthetic_sequence() -> None:
    trades = [outcome(1.0, 1), outcome(-9.0, 2), outcome(1.0, 3), outcome(2.0, 4)]
    m = compute_metrics(trades, 100.0)
    assert m["trades"] == 4 and m["wins"] == 3 and m["losses"] == 1
    assert m["total_pnl"] == pytest.approx(-5.0)
    assert m["profit_factor"] == pytest.approx(4.0 / 9.0)
    assert m["max_drawdown"] == pytest.approx(9.0)
    assert m["win_rate"] == pytest.approx(0.75)
    curve = equity_curve(trades, 100.0)
    assert curve[-1][1] == pytest.approx(95.0)
    assert max_drawdown(curve, 100.0)[1] == pytest.approx(9.0 / 101.0)
    buckets = win_rate_by_bucket(trades)
    assert buckets and buckets[0]["trades"] == 4
    assert compute_metrics([], 100.0)["trades"] == 0


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_calibrated_null_is_not_flagged_but_real_edge_is() -> None:
    rng = random.Random(3)
    fair = []
    for i in range(300):  # outcomes drawn exactly at the entry price: no edge
        price = rng.uniform(0.85, 0.97)
        won = rng.random() < price
        pnl = (10 if won else 0) - price * 10 - 0.05
        fair.append(
            TradeOutcome(i, i + 1, "s", price, 10, price * 10 + 0.05, pnl, 0.05, event=f"e{i}")
        )
    p_fair = calibration_null_pvalue(fair, n_sims=4000)
    assert p_fair > 0.01
    edge = [
        TradeOutcome(i, i + 1, "s", 0.9, 10, 9.05, 0.95, 0.05, event=f"e{i}") for i in range(300)
    ]  # wins every time at 90c: wildly unlikely if calibrated
    assert calibration_null_pvalue(edge, n_sims=4000) < 0.001
    assert "far too few" in significance_report(edge[:10])["verdict"]
    worse = [
        TradeOutcome(
            i, i + 1, "s", 0.9, 10, 9.05, -9.05 if i % 4 == 0 else 0.95, 0.05, event=f"e{i}"
        )
        for i in range(300)
    ]  # wins only 75% at 90c: far below what the price implies
    assert "worse than fair" in significance_report(worse)["verdict"]


# ---------------------------------------------------- historical (no look-ahead)
T0 = 1_800_000_000


def candle(
    minute: int, ask_close: str, ask_open: str | None = None, bid_close: str | None = None
) -> Candlestick:
    bid = bid_close or str(D(ask_close) - D("0.01"))
    return Candlestick.model_validate(
        {
            "end_period_ts": T0 + 60 * minute,
            "yes_ask": {"open_dollars": ask_open or ask_close, "close_dollars": ask_close},
            "yes_bid": {"close_dollars": bid},
        }
    )


def history(ticker: str, event: str, result: str, candles: list[Candlestick]) -> MarketHistory:
    m = Market.model_validate(
        {
            "ticker": ticker,
            "event_ticker": event,
            "market_type": "binary",
            "status": "finalized",
            "result": result,
            "close_time": datetime.fromtimestamp(T0 + 3540, UTC).isoformat(),
            "settlement_ts": datetime.fromtimestamp(T0 + 3600, UTC).isoformat(),
        }
    )
    return MarketHistory(m, candles, "KXTEST", start_ts=T0)


def test_historical_fill_uses_next_candle_not_signal_candle() -> None:
    h = history(
        "KXTEST-E1-A",
        "KXTEST-E1",
        "yes",
        [
            candle(-5, "0.95"),  # before scheduled start: ignored
            candle(1, "0.80"),
            candle(2, "0.91"),
            candle(3, "0.93", ask_open="0.92"),
        ],
    )
    trades = simulate_band([h], D("0.90"), D("0.97"), fee_rate=D("0.07"))
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == pytest.approx(0.92)  # next candle's opening ask
    assert t.opened_ts == T0 + 120  # decided after minute 2 closed
    assert t.pnl == pytest.approx(10 - 9.20 - 0.06)


def test_historical_skips_when_price_moves_away_during_latency() -> None:
    h = history(
        "KXTEST-E2-A",
        "KXTEST-E2",
        "no",
        [
            candle(1, "0.91"),
            candle(2, "0.99", ask_open="0.99"),
            candle(3, "0.94"),
            candle(4, "0.95", ask_open="0.95"),
        ],
    )
    trades = simulate_band([h], D("0.90"), D("0.97"), fee_rate=D("0.07"))
    assert len(trades) == 1 and trades[0].entry_price == pytest.approx(0.95)
    assert trades[0].pnl < 0  # the favourite lost: no hindsight filtering


def test_historical_never_fills_after_market_close() -> None:
    # close_time is T0+3540 (minute 59). Stale in-band quotes after the close (when the winner
    # is already known) must not be treated as fillable.
    h = history(
        "KXTEST-E4-A",
        "KXTEST-E4",
        "yes",
        [candle(1, "0.99"), candle(59, "0.95"), candle(60, "0.95"), candle(61, "0.95")],
    )
    assert simulate_band([h], D("0.90"), D("0.97"), fee_rate=D("0.07")) == []


def test_historical_window_starts_at_milestone_start_not_occurrence_time() -> None:
    h = history("KXTEST-E5-A", "KXTEST-E5", "yes", [candle(-3, "0.92"), candle(-2, "0.92")])
    early = MarketHistory(h.market, h.candles, "KXTEST", start_ts=T0 - 600)
    assert simulate_band([h], D("0.90"), D("0.97"), fee_rate=D("0.07")) == []
    assert len(simulate_band([early], D("0.90"), D("0.97"), fee_rate=D("0.07"))) == 1


def test_historical_one_trade_per_event() -> None:
    a = history("KXTEST-E3-A", "KXTEST-E3", "yes", [candle(1, "0.92"), candle(2, "0.92")])
    b = history("KXTEST-E3-B", "KXTEST-E3", "no", [candle(5, "0.92"), candle(6, "0.92")])
    trades = simulate_band([a, b], D("0.90"), D("0.97"), fee_rate=D("0.07"))
    assert len(trades) == 1 and trades[0].event == "KXTEST-E3"


# ------------------------------------------------------- recorded-data replay
def live_details(points: tuple[int, int]) -> dict:
    return {
        "competitor1_id": "comp-a",
        "competitor2_id": "comp-b",
        "status": "live",
        "match_status": "2nd_set",
        "server": "comp-a",
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


def book_row(ts: float, yes_bid: str, no_bid: str) -> MarketSnapshotRow:
    return MarketSnapshotRow(
        ticker="KXATPMATCH-R-A",
        received_ts=ts,
        source="ws",
        depth_json=json.dumps({"yes_bids": [[yes_bid, "40"]], "no_bids": [[no_bid, "40"]]}),
    )


def test_replay_runs_strategy_without_look_ahead(tmp_path, settings: Settings) -> None:
    settings.model.uncertainty_samples = 32
    settings.model.prior_draws = 1000
    db = Database(tmp_path / "recorded.db")
    with db.session() as s:
        s.add(
            EventRow(
                event_ticker="KXATPMATCH-R",
                series_ticker="KXATPMATCH",
                title="A vs B",
                milestone_id="ms-r",
                tour="ATP",
                gender="men",
                best_of=3,
                tournament_name="ATP Replay",
                scheduled_start_ts=0.0,
                first_competitor_id="comp-a",
                second_competitor_id="comp-b",
                first_seen_ts=0.0,
                updated_ts=0.0,
            )
        )
        for ticker, name, comp, result in (
            ("KXATPMATCH-R-A", "A", "comp-a", "yes"),
            ("KXATPMATCH-R-B", "B", "comp-b", "no"),
        ):
            s.add(
                MarketRow(
                    ticker=ticker,
                    event_ticker="KXATPMATCH-R",
                    series_ticker="KXATPMATCH",
                    player_name=name,
                    competitor_id=comp,
                    status="finalized",
                    result=result,
                    settlement_value=D("1") if result == "yes" else D("0"),
                    close_ts=500.0,
                    first_seen_ts=0.0,
                    updated_ts=0.0,
                    price_ranges_json='[{"start":"0","end":"1","step":"0.01"}]',
                )
            )
        s.add(book_row(100.0, "0.90", "0.08"))  # ask 0.92 known at t=100
        s.add(
            ScoreSnapshotRow(
                milestone_id="ms-r",
                event_ticker="KXATPMATCH-R",
                received_ts=105.0,
                provider="kalshi_live_data",
                status="live",
                state_hash="h1",
                state_json="{}",
                raw_json=json.dumps(live_details((40, 0))),
            )
        )
        s.add(book_row(110.0, "0.78", "0.20"))  # a cheaper ask appears only later
    trades, info = replay(settings, db)
    assert info["unsettled_trades_excluded"] == 0
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == pytest.approx(0.92)  # not the later 0.80
    assert t.opened_ts == 105.0 and t.pnl > 0
