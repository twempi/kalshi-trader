"""Model evaluation against outcomes and against the market (Baseline 3).

Inputs are ``Prediction`` records: a model probability for a YES outcome, the executable
YES bid/ask at the same instant, and the market's eventual result. Sources:

* the live ``model_predictions`` table (forward data recorded by the engine) joined with
  final market results, and
* the replay (same tracker/estimator code on recorded observations).

Statistics, each reported for the model and for the **market-implied probability** (YES
mid) so the model is always compared with the obvious benchmark:

* Brier score and log loss, with a skill score 1 - Brier_model / Brier_market;
* reliability (calibration) table by probability bin;
* breakdowns by match stage, tour and market-probability band;
* paired per-event difference in Brier score with a cluster-bootstrap CI;
* an *executable* check: when the model's conservative probability exceeded the YES ask
  (or the NO ask for the other side), what did buying at that ask actually return?

Every observation weighs the same, and CIs resample whole events because observations of one
match are strongly dependent. Observations are taken at score changes, which are stopping
times, so for a calibrated forecaster the observation-weighted error in any probability bin
has mean zero. Giving each *event* a total weight of 1 instead (weight 1/n for an event
with n observations) is biased, because n depends on the future: matches the favourite
wins quickly have few observations, so each counts more, and a perfectly calibrated
forecaster then looks about 3 points too timid (underdogs "win" less, favourites more).
Results with fewer than ``MIN_EVENTS`` events are labelled insufficient.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from sqlalchemy import select

from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import MarketRow, ModelPredictionRow

MIN_EVENTS = 30
EPS = 1e-6
PROB_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


@dataclass(frozen=True)
class Prediction:
    ts: float
    event: str
    market: str
    p_model: float
    outcome: int  # 1 if the market's YES won
    yes_bid: float | None = None
    yes_ask: float | None = None
    p_model_low: float | None = None
    stage: str | None = None
    tour: str | None = None
    score_age_s: float | None = None
    prior_source: str | None = None

    @property
    def p_market(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None or self.yes_ask < self.yes_bid:
            return None
        return (self.yes_bid + self.yes_ask) / 2


def _band(p: float | None) -> str:
    if p is None:
        return "no market"
    for lo, hi in pairwise(PROB_BINS):
        if lo <= p < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "?"


def _logloss(p: float, y: int) -> float:
    p = min(1 - EPS, max(EPS, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _scores(preds: Sequence[Prediction]) -> dict[str, Any]:
    """Brier/log loss for model and market on the observations that have both."""
    both = [p for p in preds if p.p_market is not None]
    if not both:
        return {"observations": len(preds), "events": len({p.event for p in preds})}

    def mean(f: Callable[[Prediction], float]) -> float:
        return sum(f(p) for p in both) / len(both)

    brier_m = mean(lambda p: (p.p_model - p.outcome) ** 2)
    brier_k = mean(lambda p: ((p.p_market or 0) - p.outcome) ** 2)
    return {
        "observations": len(both),
        "events": len({p.event for p in both}),
        "brier_model": brier_m,
        "brier_market": brier_k,
        "brier_skill_vs_market": (1 - brier_m / brier_k) if brier_k > 0 else None,
        "logloss_model": mean(lambda p: _logloss(p.p_model, p.outcome)),
        "logloss_market": mean(lambda p: _logloss(p.p_market or 0.5, p.outcome)),
    }


def calibration_table(
    preds: Sequence[Prediction], prob: Callable[[Prediction], float | None]
) -> list[dict[str, Any]]:
    rows = []
    usable = [(p, prob(p)) for p in preds]
    usable = [(p, q) for p, q in usable if q is not None]
    if not usable:
        return rows
    for lo, hi in pairwise(PROB_BINS):
        idx = [i for i, (_, q) in enumerate(usable) if lo <= q < hi]  # type: ignore[operator]
        if not idx:
            continue
        rows.append(
            {
                "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "observations": len(idx),
                "events": len({usable[i][0].event for i in idx}),
                "mean_predicted": sum(usable[i][1] or 0 for i in idx) / len(idx),
                "observed_frequency": sum(usable[i][0].outcome for i in idx) / len(idx),
            }
        )
    return rows


def paired_brier_difference(
    preds: Sequence[Prediction], n_boot: int = 2000, seed: int = 5
) -> dict[str, Any] | None:
    """Mean over observations of (Brier_model - Brier_market), with a CI that resamples
    whole events; negative favours the model."""
    per_event: dict[str, list[float]] = defaultdict(list)
    for p in preds:
        if p.p_market is None:
            continue
        per_event[p.event].append((p.p_model - p.outcome) ** 2 - (p.p_market - p.outcome) ** 2)
    if len(per_event) < 2:
        return None
    mean, ci = cluster_mean_ci(list(per_event.values()), n_boot=n_boot, seed=seed)
    return {
        "events": len(per_event),
        "mean_difference": mean,
        "ci95": ci,
        "interpretation": "negative = model more accurate than the market mid",
    }


def cluster_mean_ci(
    groups: Sequence[Sequence[float]], n_boot: int = 2000, seed: int = 5
) -> tuple[float, list[float]]:
    """Mean of all values, and a 95% CI from resampling whole groups (events)."""
    sums = np.array([sum(g) for g in groups], dtype=float)
    counts = np.array([len(g) for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(sums.sum() / counts.sum()), [
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    ]


def executable_check(preds: Sequence[Prediction], min_edge: float = 0.0) -> dict[str, Any]:
    """First observation per (event, side) where the conservative model probability beat
    the executable ask by ``min_edge``; P&L per contract of buying there (before fees)."""
    taken: dict[tuple[str, str], tuple[float, int]] = {}
    for p in sorted(preds, key=lambda p: p.ts):
        q = p.p_model_low if p.p_model_low is not None else p.p_model
        if p.yes_ask is not None and q - p.yes_ask >= min_edge:
            taken.setdefault((p.event, f"{p.market}:yes"), (p.yes_ask, p.outcome))
        if p.yes_bid is not None:
            no_ask = 1 - p.yes_bid
            q_no = 1 - p.p_model  # conservative NO probability is not stored; use point
            if q_no - no_ask >= min_edge:
                taken.setdefault((p.event, f"{p.market}:no"), (no_ask, 1 - p.outcome))
    pnl = [y - price for price, y in taken.values()]
    return {
        "entries": len(pnl),
        "events": len({k[0] for k in taken}),
        "mean_pnl_per_contract_before_fees": (sum(pnl) / len(pnl)) if pnl else None,
        "win_rate": (sum(1 for x in pnl if x > 0) / len(pnl)) if pnl else None,
    }


def evaluate_predictions(preds: Sequence[Prediction]) -> dict[str, Any]:
    preds = list(preds)
    events = len({p.event for p in preds})
    report: dict[str, Any] = {
        "events": events,
        "observations": len(preds),
        "sufficient": events >= MIN_EVENTS,
        "note": (
            f"only {events} independent events; at least {MIN_EVENTS} are needed before any "
            "calibration or model-vs-market conclusion"
            if events < MIN_EVENTS
            else "every observation weighs the same; CIs resample whole events"
        ),
        "overall": _scores(preds),
        "calibration_model": calibration_table(preds, lambda p: p.p_model),
        "calibration_market": calibration_table(preds, lambda p: p.p_market),
        "paired_brier_vs_market": paired_brier_difference(preds),
        "executable_check": executable_check(preds),
    }
    for name, key in (
        ("by_stage", lambda p: p.stage or "?"),
        ("by_tour", lambda p: p.tour or "?"),
        ("by_market_band", lambda p: _band(p.p_market)),
    ):
        groups: dict[str, list[Prediction]] = defaultdict(list)
        for p in preds:
            groups[key(p)].append(p)
        report[name] = {k: _scores(v) for k, v in sorted(groups.items())}
    return report


def predictions_from_db(db: Database, model: str = "tennis_markov") -> list[Prediction]:
    """Forward predictions recorded live, joined with final results (settled markets only)."""
    with db.session() as s:
        results = {
            m.ticker: m.result
            for m in s.scalars(select(MarketRow).where(MarketRow.result.in_(["yes", "no"])))
        }
        rows = list(
            s.scalars(
                select(ModelPredictionRow)
                .where(ModelPredictionRow.model == model)
                .order_by(ModelPredictionRow.ts)
            )
        )
    out = []
    for r in rows:
        result = results.get(r.market_ticker)
        if result is None:
            continue
        out.append(
            Prediction(
                ts=r.ts,
                event=r.event_ticker,
                market=r.market_ticker,
                p_model=r.p_point,
                p_model_low=r.p_robust_low,
                outcome=1 if result == "yes" else 0,
                yes_bid=r.yes_bid,
                yes_ask=r.yes_ask,
                stage=r.stage,
                tour=r.tour,
                score_age_s=r.score_age_s,
                prior_source=r.prior_source,
            )
        )
    return out


def one_market_per_event(preds: Iterable[Prediction]) -> list[Prediction]:
    """Keep only the first market of each event: the two markets of a tennis match are
    complements, so scoring both double-counts the same information."""
    first: dict[str, str] = {}
    out = []
    for p in sorted(preds, key=lambda p: (p.event, p.market)):
        market = first.setdefault(p.event, p.market)
        if p.market == market:
            out.append(p)
    return sorted(out, key=lambda p: p.ts)
