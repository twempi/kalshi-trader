"""Probability calibration: the runtime map, the strategy using it, the fit and its report.

All data here is synthetic, built in code; none of it is a market result.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal as D
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kalshi_trader.config import (
    FeeSettings,
    ProbabilityCalibrationSettings,
    Settings,
    TennisModelStrategySettings,
)
from kalshi_trader.core.opportunity import DecisionStatus
from kalshi_trader.dashboard.app import create_app
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.database.tables import MarketRow
from kalshi_trader.engine import TradingEngine
from kalshi_trader.models.calibration import calibrate, logit
from kalshi_trader.models.estimator import PlayerProbability
from kalshi_trader.research.calibration import calibration_report, decision_check, fit_logistic
from kalshi_trader.research.model_eval import Prediction
from kalshi_trader.strategies.stat_mispricing import TennisModelStrategy, market_probability
from tests.test_trading import make_context

FEES = FeeSettings()


# ------------------------------------------------------------ runtime map
def test_calibration_map_identity_symmetry_and_anchor() -> None:
    raw = ProbabilityCalibrationSettings()
    assert raw.is_identity and calibrate(0.37, raw) == 0.37
    steeper = ProbabilityCalibrationSettings(slope=1.5)
    p = calibrate(0.7, steeper)
    assert p > 0.7 and calibrate(0.3, steeper) == pytest.approx(1 - p)  # players still sum to 1
    assert calibrate(0.5, steeper) == pytest.approx(0.5)
    anchored = ProbabilityCalibrationSettings(slope=0.5, market_weight=0.5)
    assert logit(calibrate(0.8, anchored, 0.6)) == pytest.approx(
        0.5 * logit(0.8) + 0.5 * logit(0.6)
    )
    with pytest.raises(ValueError, match="market probability"):
        calibrate(0.8, anchored)


def test_strategy_applies_calibration_to_conservative_probability() -> None:
    ctx = make_context()  # model: p = 0.995, conservative 0.985; MKT-A YES 0.90 / 0.92
    assert market_probability(ctx, 0) == pytest.approx(0.91)
    assert market_probability(ctx, 1) == pytest.approx(0.075)  # MKT-B YES 0.05 / 0.10

    timid = TennisModelStrategySettings(calibration=ProbabilityCalibrationSettings(slope=0.5))
    (note,) = TennisModelStrategy(timid, FEES).evaluate_match(ctx)
    assert note.status == DecisionStatus.FILTERED  # 0.985 shrinks to ~0.89 < the 0.92 ask

    cal = ProbabilityCalibrationSettings(slope=0.5, market_weight=0.6)
    (opp,) = TennisModelStrategy(TennisModelStrategySettings(calibration=cal), FEES).evaluate_match(
        make_context()
    )
    assert opp.status == DecisionStatus.CANDIDATE
    assert opp.fair.source == "tennis_markov+calibrated"
    assert opp.fair.p_low == pytest.approx(calibrate(0.985, cal, 0.91))
    assert opp.fair.p == pytest.approx(calibrate(0.995, cal, 0.91))
    assert opp.fair.details["p_low_model"] == 0.985 and opp.fair.details["p_market"] == 0.91


def test_market_anchored_calibration_needs_a_two_sided_book() -> None:
    ctx = make_context()
    ctx.books = {}
    cal = ProbabilityCalibrationSettings(market_weight=0.5)
    (note,) = TennisModelStrategy(
        TennisModelStrategySettings(calibration=cal), FEES
    ).evaluate_match(ctx)
    assert note.status == DecisionStatus.UNSUPPORTED and "two-sided book" in note.reasons[0]


def test_calibration_changes_the_strategy_version() -> None:
    raw = TennisModelStrategy(TennisModelStrategySettings(), FEES)
    cal = TennisModelStrategySettings(calibration=ProbabilityCalibrationSettings(slope=1.2))
    assert TennisModelStrategy(cal, FEES).version_tag != raw.version_tag


def test_live_view_shows_the_probability_the_strategy_trades_on(settings: Settings) -> None:
    raw = PlayerProbability(0.7, 0.66, 0.62)
    engine = SimpleNamespace(settings=settings)
    assert TradingEngine._traded_probability(engine, raw, D("0.65")) is raw  # no calibration
    cal = ProbabilityCalibrationSettings(slope=1.5, market_weight=0.3)
    settings.strategies.tennis_model.calibration = cal
    shown = TradingEngine._traded_probability(engine, raw, D("0.65"))
    assert shown.robust_low == pytest.approx(calibrate(0.62, cal, 0.65))
    assert shown.point == pytest.approx(calibrate(0.7, cal, 0.65))
    assert TradingEngine._traded_probability(engine, raw, None) is raw  # needs the mid


# ------------------------------------------------------------ fitting
def test_fit_logistic_recovers_coefficients() -> None:
    rng = np.random.default_rng(3)
    x1 = rng.normal(0, 1.5, 20000)
    x2 = 0.6 * x1 + rng.normal(0, 1.0, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(1.3 * x1 + 0.4 * x2)))).astype(float)
    beta, se = fit_logistic(np.column_stack([x1, x2]), y)
    assert beta == pytest.approx([1.3, 0.4], abs=4 * se.max())
    assert (se > 0).all() and se.max() < 0.1


def synthetic_predictions(
    events: int, per_event: int = 4, timid: float = 0.5, seed: int = 11
) -> list[Prediction]:
    """Matches whose true probability is fixed; the model is too timid (logit scaled by
    `timid`), the market is the truth plus a little noise."""
    rng = np.random.default_rng(seed)
    out = []
    for e in range(events):
        truth = 1 / (1 + math.exp(-rng.normal(0, 1.2)))
        y = int(rng.random() < truth)
        for k in range(per_event):
            z = logit(truth)
            p_model = 1 / (1 + math.exp(-(timid * z + rng.normal(0, 0.1))))
            mid = min(0.94, max(0.06, 1 / (1 + math.exp(-(z + rng.normal(0, 0.1))))))
            out.append(
                Prediction(
                    ts=e * 10_000.0 + k,
                    event=f"E{e:04d}",
                    market=f"E{e:04d}-A",
                    p_model=p_model,
                    outcome=y,
                    yes_bid=round(mid - 0.01, 4),
                    yes_ask=round(mid + 0.01, 4),
                    p_model_low=max(0.0, p_model - 0.02),
                    stage="mid",
                    tour="ATP",
                    prior_source="pre_match_price",
                )
            )
    return out


def test_calibration_report_finds_and_checks_the_fix() -> None:
    preds = synthetic_predictions(3000)
    report = calibration_report(preds, TennisModelStrategySettings(), FEES, embargo_s=0.0)
    json.dumps(report, allow_nan=False)  # the dashboard serves this as strict JSON
    assert report["sufficient"] and report["test"]["events"] == 900
    slope = report["candidates"]["model_slope"]
    assert slope["params"]["slope"] == pytest.approx(1 / 0.5, rel=0.2)
    assert slope["vs_raw_model"]["ci95"][1] < 0  # held-out log loss improves on the raw model
    assert report["candidates"]["market_anchored"]["params"]["market_weight"] > 0.5
    rec = report["recommendation"]
    assert rec["candidate"] in ("model_slope", "market_anchored")
    assert "[strategies.tennis_model.calibration]" in rec["toml"]
    assert set(report["decision_check"]) >= {"note", "raw_model", "model_slope"}
    assert report["by_stage"]["mid"]["events"] == 900


def test_calibration_report_filters_and_needs_enough_events() -> None:
    preds = synthetic_predictions(20)
    report = calibration_report(preds, TennisModelStrategySettings(), FEES)
    json.dumps(report, allow_nan=False)
    assert not report["sufficient"] and report["recommendation"]["candidate"] is None
    late_only = TennisModelStrategySettings(stages=["late"])
    assert calibration_report(preds, late_only, FEES)["observations"] == 0


def test_decision_check_takes_first_qualifying_entry_per_event() -> None:
    def pred(ts: float, event: str, low: float, ask: float, y: int) -> Prediction:
        return Prediction(ts, event, f"{event}-A", low + 0.01, y, ask - 0.02, ask, low, "mid")

    preds = [
        pred(1, "A", 0.50, 0.55, 1),  # no edge yet
        pred(2, "A", 0.70, 0.60, 1),  # first entry: 1 - 0.60 - fee
        pred(3, "A", 0.90, 0.50, 1),  # later, ignored (one trade per event)
        pred(1, "B", 0.30, 0.10, 0),  # below the 20c minimum price
    ]
    strategy = TennisModelStrategySettings(min_price=D("0.20"))
    check = decision_check(preds, strategy, FEES, 1.0, 0.0)
    assert check["entries"] == 1 and check["avg_price"] == pytest.approx(0.60)
    fee = 0.07 * 0.60 * 0.40
    assert check["pnl_per_contract_after_fees"] == pytest.approx(1 - 0.60 - fee)


# ------------------------------------------------------------ dashboard
def seed_predictions(db: Database, preds: list[Prediction]) -> None:
    repo = Repository(db)
    with db.session() as s:
        for event in sorted({p.event for p in preds}):
            outcome = next(p.outcome for p in preds if p.event == event)
            s.add(
                MarketRow(
                    ticker=f"{event}-A",
                    event_ticker=event,
                    series_ticker="KXATPMATCH",
                    result="yes" if outcome else "no",
                    first_seen_ts=0.0,
                    updated_ts=0.0,
                )
            )
    repo.insert_model_predictions(
        [
            {
                "ts": p.ts,
                "model": "tennis_markov",
                "event_ticker": p.event,
                "market_ticker": p.market,
                "tour": p.tour,
                "stage": p.stage,
                "p_point": p.p_model,
                "p_robust_low": p.p_model_low,
                "prior_source": p.prior_source,
                "yes_bid": p.yes_bid,
                "yes_ask": p.yes_ask,
            }
            for p in preds
        ]
    )


def test_dashboard_calibration_and_exports(settings: Settings) -> None:
    db = Database(settings.db_path)
    seed_predictions(db, synthetic_predictions(120))
    client = TestClient(create_app(settings, db))
    report = client.get("/api/research/calibration").json()
    assert report["sufficient"] and report["events"] == 120
    export = client.get("/api/export/predictions.csv")
    lines = export.text.strip().splitlines()
    assert lines[0].startswith("ts,event,market") and len(lines) == 1 + 120 * 4
    assert client.get("/api/trades?ledger=main").json()["ledger"] == "main"
    assert client.get("/api/trades?ledger=bogus").status_code == 404
    assert (
        client.get("/api/export/trades.csv?ledger=tennis_model").status_code == 404
    )  # no ledger yet
    assert D(lines[1].split(",")[6]) > 0  # p_model column
