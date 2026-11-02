"""Sensitivity of the tennis model's accuracy to its assumptions.

Each variant replays the recorded score/book observations with one assumption changed and
scores the resulting predictions (one market per event, event-weighted) against outcomes
and against the market mid. Variants:

* serve-strength priors shifted by +/-0.02 (tour-average serve-point win probability);
* pre-match strength uncertainty doubled / halved;
* retirement hazard off / tripled;
* latency guard 0 or 2 points (affects the conservative probability only);
* stale scores: every score observation delayed by 15 / 30 / 60 seconds.

With few settled events these numbers are noisy; the report carries the event count and
the insufficiency flag from ``model_eval`` so they are not over-read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kalshi_trader.backtesting.replay import RecordedData, load_recorded, replay
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.research.model_eval import evaluate_predictions, one_market_per_event

Variant = Callable[[Settings], None]


def _shift_serve(delta: float) -> Variant:
    def apply(s: Settings) -> None:
        s.model.serve_win_prior = {k: v + delta for k, v in s.model.serve_win_prior.items()}

    return apply


def _scale(attr: str, factor: float) -> Variant:
    def apply(s: Settings) -> None:
        setattr(s.model, attr, getattr(s.model, attr) * factor)

    return apply


def _set(attr: str, value: Any) -> Variant:
    def apply(s: Settings) -> None:
        setattr(s.model, attr, value)

    return apply


MODEL_VARIANTS: dict[str, Variant] = {
    "baseline": lambda s: None,
    "serve_prior_-0.02": _shift_serve(-0.02),
    "serve_prior_+0.02": _shift_serve(0.02),
    "prior_strength_sd_x2": _scale("strength_diff_sd_with_price", 2.0),
    "prior_strength_sd_x0.5": _scale("strength_diff_sd_with_price", 0.5),
    "no_retirement_hazard": _set("retirement_hazard_per_game", 0.0),
    "retirement_hazard_x3": _scale("retirement_hazard_per_game", 3.0),
    "latency_guard_0": _set("latency_points", 0),
    "latency_guard_2": _set("latency_points", 2),
}
SCORE_DELAYS_S = (15.0, 30.0, 60.0)


def _evaluate(settings: Settings, data: RecordedData, delay: float = 0.0) -> dict[str, Any]:
    result = replay(settings, data, strategies=[], score_delay_s=delay)
    preds = one_market_per_event(result.predictions)
    report = evaluate_predictions(preds)
    paired = report.get("paired_brier_vs_market") or {}
    return {
        "events": report["events"],
        "observations": report["observations"],
        "sufficient": report["sufficient"],
        **{k: v for k, v in report["overall"].items() if k.startswith(("brier", "logloss"))},
        "paired_brier_minus_market": paired.get("mean_difference"),
        "paired_ci95": paired.get("ci95"),
        "executable_check": report["executable_check"],
    }


def run_sensitivity(settings: Settings, db: Database | None = None) -> dict[str, Any]:
    data = load_recorded(db or Database(settings.db_path))
    out: dict[str, Any] = {"model_variants": {}, "score_delay": {}}
    for name, apply in MODEL_VARIANTS.items():
        cfg = settings.model_copy(deep=True)
        apply(cfg)
        out["model_variants"][name] = _evaluate(cfg, data)
    for delay in SCORE_DELAYS_S:
        out["score_delay"][f"{delay:.0f}s"] = _evaluate(settings, data, delay)
    base = out["model_variants"]["baseline"]
    out["note"] = (
        f"{base['events']} settled events with predictions; "
        + ("enough for a first look" if base["sufficient"] else "too few for conclusions")
        + ". Lower Brier/log loss is better; compare each variant with 'baseline'."
    )
    return out
