"""Fitting the tennis model's probability calibration (``ProbabilityCalibrationSettings``).

Data: forward predictions recorded live for settled markets, filtered to what the
tennis_model strategy could trade on. That means its match stages and tours, the pre-match
price prior when the strategy requires one, a two-sided book no wider than its
``max_spread``, and a mid inside the band where either player's contract can be bought.
Fits use one market per event, because the two markets of a match are complements. Every
observation weighs the same, and statistics resample whole events (see ``model_eval``).

Candidates are logistic regressions without an intercept, so the two players'
probabilities still sum to one:

* ``raw_model``: the model as it is (slope 1);
* ``model_slope``: logit(p) = slope * logit(p_model);
* ``market_anchored``: logit(p) = slope * logit(p_model) + market_weight * logit(p_market);
* ``market_slope``: logit(p) = market_weight * logit(p_market). This one is a benchmark,
  not a trading candidate: it never disagrees with the market. The model is worth trading
  only if adding it to the market improves held-out accuracy.

Each candidate is fitted by maximum likelihood on the earlier events and scored on later,
held-out events, using a chronological split with an embargo as in the replay. A candidate
is recommended only if it beats the raw model on held-out log loss with a CI that excludes
zero. Its parameters are then refitted on all events.

The decision check replays the strategy's entry rule on the held-out events. It looks for
the first observation per event where the calibrated conservative probability beats the
YES ask plus the taker fee by ``min_edge``, with the ask inside the strategy's price band.
It buys one contract there and holds it to settlement. This is a rough check: observations
exist only at score changes, and it ignores depth, sizing and the risk limits.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kalshi_trader.config import FeeSettings, TennisModelStrategySettings
from kalshi_trader.models.calibration import P_CLIP
from kalshi_trader.research.model_eval import (
    MIN_EVENTS,
    Prediction,
    cluster_mean_ci,
    one_market_per_event,
)
from kalshi_trader.research.splits import split_events

# candidate -> regressors, from "model" = logit(p_model) and "market" = logit(p_market)
CANDIDATES: dict[str, tuple[str, ...]] = {
    "model_slope": ("model",),
    "market_anchored": ("model", "market"),
    "market_slope": ("market",),
}
TRADING_CANDIDATES = ("model_slope", "market_anchored")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(z / 2.0))


def _logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1.0 - p))


def apply_params(
    slope: float, market_weight: float, p_model: np.ndarray, p_market: np.ndarray
) -> np.ndarray:
    """Vectorised ``models.calibration.calibrate``."""
    if slope == 1.0 and market_weight == 0.0:
        return p_model
    z = slope * _logit(p_model)
    if market_weight:
        z = z + market_weight * _logit(p_market)
    return _sigmoid(z)


def fit_logistic(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood logistic regression without an intercept (Newton steps with step
    halving). Returns coefficients and cluster-robust standard errors (clusters = groups)."""
    beta = np.zeros(X.shape[1])

    def loglik(b: np.ndarray) -> float:
        z = X @ b
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    current = loglik(beta)
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        hessian = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (y - p))
        scale = 1.0
        value = loglik(beta + step)
        while value < current and scale > 1e-6:
            scale /= 2
            value = loglik(beta + scale * step)
        if value < current:
            break  # converged as far as floating point allows
        beta, improvement, current = beta + scale * step, value - current, value
        if np.max(np.abs(scale * step)) < 1e-9 or improvement < 1e-10:
            break
    p = _sigmoid(X @ beta)
    bread = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    scores = X * (y - p)[:, None]
    if groups is not None:
        _, inverse = np.unique(groups, return_inverse=True)
        summed = np.zeros((int(inverse.max()) + 1, X.shape[1]))
        np.add.at(summed, inverse, scores)
        scores = summed
    cov = bread @ (scores.T @ scores) @ bread
    return beta, np.sqrt(np.diag(cov))


@dataclass
class _Sample:
    preds: list[Prediction]
    event: np.ndarray
    y: np.ndarray
    p_model: np.ndarray
    p_market: np.ndarray

    @classmethod
    def of(cls, preds: Sequence[Prediction]) -> _Sample:
        preds = list(preds)
        return cls(
            preds,
            np.array([p.event for p in preds], dtype=object),
            np.array([p.outcome for p in preds], dtype=float),
            np.array([p.p_model for p in preds], dtype=float),
            np.array([p.p_market for p in preds], dtype=float),
        )

    def design(self, regressors: tuple[str, ...]) -> np.ndarray:
        cols = {"model": _logit(self.p_model), "market": _logit(self.p_market)}
        return np.column_stack([cols[r] for r in regressors])

    def __len__(self) -> int:
        return len(self.preds)


def eligible(
    preds: Sequence[Prediction], strategy: TennisModelStrategySettings
) -> list[Prediction]:
    """Observations the strategy could have traded on (module docstring)."""
    low = float(min(strategy.min_price, 1 - strategy.max_price))
    high = float(max(strategy.max_price, 1 - strategy.min_price))
    out = []
    for p in preds:
        mid = p.p_market
        if mid is None or p.yes_bid is None or p.yes_ask is None:
            continue
        if p.yes_ask - p.yes_bid > float(strategy.max_spread) + 1e-9 or not low <= mid <= high:
            continue
        if p.stage is not None and p.stage not in strategy.stages:
            continue
        if strategy.tours and p.tour is not None and p.tour not in strategy.tours:
            continue
        if strategy.require_pre_match_prior and p.prior_source not in (None, "pre_match_price"):
            continue
        out.append(p)
    return out


def _params(name: str, beta: np.ndarray) -> dict[str, float]:
    values = dict(zip(CANDIDATES[name], (float(b) for b in beta), strict=True))
    return {"slope": values.get("model", 0.0), "market_weight": values.get("market", 0.0)}


def _per_event(sample: _Sample, values: np.ndarray) -> list[list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for event, value in zip(sample.event, values, strict=True):
        groups[event].append(float(value))
    return list(groups.values())


def _compare(sample: _Sample, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Held-out log loss of `a` minus `b` per observation; negative favours `a`."""
    mean, ci = cluster_mean_ci(_per_event(sample, _logloss(a, sample.y) - _logloss(b, sample.y)))
    return {"mean_logloss_difference": mean, "ci95": ci}


def _scores(sample: _Sample, p: np.ndarray) -> dict[str, float]:
    return {
        "logloss": float(_logloss(p, sample.y).mean()),
        "brier": float(((p - sample.y) ** 2).mean()),
    }


def decision_check(
    preds: Sequence[Prediction],
    strategy: TennisModelStrategySettings,
    fees: FeeSettings,
    slope: float,
    market_weight: float,
) -> dict[str, Any]:
    """One contract at the first qualifying observation per event (module docstring)."""
    rows = sorted(
        (p for p in preds if p.p_model_low is not None and p.yes_ask is not None),
        key=lambda p: p.ts,
    )
    if not rows:
        return {"entries": 0}
    ask = np.array([p.yes_ask for p in rows], dtype=float)
    q = apply_params(
        slope,
        market_weight,
        np.array([p.p_model_low for p in rows], dtype=float),
        np.array([p.p_market if p.p_market is not None else 0.5 for p in rows], dtype=float),
    )
    fee = float(fees.taker_rate) * ask * (1.0 - ask)
    ok = (
        (ask >= float(strategy.min_price))
        & (ask <= float(strategy.max_price))
        & (q - ask - fee >= float(strategy.min_edge))
    )
    taken: dict[str, float] = {}
    prices = []
    for i in np.flatnonzero(ok):
        event = rows[i].event
        if event not in taken:
            taken[event] = rows[i].outcome - ask[i] - fee[i]
            prices.append(ask[i])
    pnl = np.array(list(taken.values()))
    if len(pnl) == 0:
        return {"entries": 0}
    mean, ci = cluster_mean_ci([[x] for x in pnl])
    return {
        "entries": len(pnl),
        "win_rate": float((pnl > 0).mean()),
        "avg_price": float(np.mean(prices)),
        "pnl_per_contract_after_fees": mean,
        "ci95": ci,
        "total_pnl_one_contract_each": float(pnl.sum()),
        "roi_on_cost": float(pnl.sum() / np.sum(prices)),
    }


def _toml(params: dict[str, float]) -> str:
    return (
        "[strategies.tennis_model.calibration]\n"
        f"slope = {params['slope']:.3f}\n"
        f"market_weight = {params['market_weight']:.3f}\n"
    )


def calibration_report(
    preds: Sequence[Prediction],
    strategy: TennisModelStrategySettings,
    fees: FeeSettings,
    *,
    test_fraction: float = 0.3,
    embargo_s: float = 3600.0,
) -> dict[str, Any]:
    usable = eligible(preds, strategy)
    first_seen: dict[str, float] = {}
    for p in usable:
        first_seen[p.event] = min(p.ts, first_seen.get(p.event, p.ts))
    split = split_events(
        first_seen, validation_fraction=0.0, test_fraction=test_fraction, embargo_s=embargo_s
    )
    one = one_market_per_event(usable)
    train = _Sample.of([p for p in one if p.event in split.train])
    test = _Sample.of([p for p in one if p.event in split.test])
    report: dict[str, Any] = {
        "data_source": "forward predictions recorded live, settled markets only",
        "filters": {
            "stages": list(strategy.stages),
            "tours": list(strategy.tours) or "all",
            "pre_match_prior_required": strategy.require_pre_match_prior,
            "max_spread": float(strategy.max_spread),
            "note": "one market per event; every observation weighs the same; "
            "CIs resample whole events",
        },
        "observations": len(one),
        "events": len(first_seen),
        "split": {
            **split.summary(),
            "boundaries": [b if math.isfinite(b) else None for b in split.boundaries],
        },
        "train": {"events": len(set(train.event)), "observations": len(train)},
        "test": {"events": len(set(test.event)), "observations": len(test)},
        "configured": {
            "slope": strategy.calibration.slope,
            "market_weight": strategy.calibration.market_weight,
        },
    }
    enough = len(set(train.event)) >= MIN_EVENTS and len(set(test.event)) >= MIN_EVENTS
    report["sufficient"] = enough
    if not enough:
        report["recommendation"] = {
            "candidate": None,
            "reason": f"need at least {MIN_EVENTS} training and {MIN_EVENTS} held-out events",
        }
        return report

    raw_test = test.p_model
    candidates: dict[str, Any] = {
        "raw_model": {
            "params": {"slope": 1.0, "market_weight": 0.0},
            "test": _scores(test, raw_test),
        }
    }
    predicted: dict[str, np.ndarray] = {"raw_model": raw_test}
    for name, regressors in CANDIDATES.items():
        beta, se = fit_logistic(train.design(regressors), train.y, train.event)
        params = _params(name, beta)
        p_hat = apply_params(params["slope"], params["market_weight"], test.p_model, test.p_market)
        predicted[name] = p_hat
        candidates[name] = {
            "params": params,
            "standard_errors": dict(zip(regressors, (float(s) for s in se), strict=True)),
            "test": _scores(test, p_hat),
            "vs_raw_model": _compare(test, p_hat, raw_test),
            "vs_market_mid": _compare(test, p_hat, test.p_market),
        }
    candidates["raw_model"]["vs_market_mid"] = _compare(test, raw_test, test.p_market)
    report["market_mid"] = {"test": _scores(test, test.p_market)}
    report["candidates"] = candidates

    # Does the model add anything the market price does not already contain?
    adds = _compare(test, predicted["market_anchored"], predicted["market_slope"])
    adds["verdict"] = (
        "the model adds information beyond the market price"
        if adds["ci95"][1] < 0
        else "no evidence the model adds information beyond the market price"
    )
    report["model_adds_information"] = adds

    held_out = [p for p in usable if p.event in split.test]  # both markets of each match
    checks = {"raw_model": decision_check(held_out, strategy, fees, 1.0, 0.0)}
    for name in TRADING_CANDIDATES:
        params = candidates[name]["params"]
        checks[name] = decision_check(
            held_out, strategy, fees, params["slope"], params["market_weight"]
        )
    if not strategy.calibration.is_identity:
        cal = strategy.calibration
        checks["configured"] = decision_check(
            held_out, strategy, fees, cal.slope, cal.market_weight
        )
    report["decision_check"] = {
        "note": "held-out events only; one contract at the first observation per event that "
        "clears the strategy's entry rule, held to settlement, after the taker fee",
        **checks,
    }

    by_stage: dict[str, Any] = {}
    for stage in sorted({p.stage or "?" for p in test.preds}):
        mask = np.array([(p.stage or "?") == stage for p in test.preds])
        sub = _Sample(
            [p for p, m in zip(test.preds, mask, strict=True) if m],
            test.event[mask],
            test.y[mask],
            test.p_model[mask],
            test.p_market[mask],
        )
        by_stage[stage] = {
            "events": len(set(sub.event)),
            "observations": len(sub),
            **{
                f"logloss_{name}": float(_logloss(values[mask], sub.y).mean())
                for name, values in (*predicted.items(), ("market_mid", test.p_market))
            },
        }
    report["by_stage"] = by_stage

    best = min(TRADING_CANDIDATES, key=lambda n: candidates[n]["test"]["logloss"])
    gain = candidates[best]["vs_raw_model"]
    if gain["ci95"][1] >= 0:
        report["recommendation"] = {
            "candidate": None,
            "reason": f"best candidate {best} does not beat the raw model on held-out log "
            f"loss (CI {gain['ci95'][0]:+.4f} to {gain['ci95'][1]:+.4f})",
        }
        return report
    everything = _Sample.of(one)
    regressors = CANDIDATES[best]
    beta, se = fit_logistic(everything.design(regressors), everything.y, everything.event)
    final = {k: round(v, 3) for k, v in _params(best, beta).items()}
    report["recommendation"] = {
        "candidate": best,
        "reason": f"{best} beats the raw model on held-out log loss by "
        f"{-gain['mean_logloss_difference']:.4f} (CI {-gain['ci95'][1]:.4f} to "
        f"{-gain['ci95'][0]:.4f}); parameters refitted on all {len(set(everything.event))} "
        "events",
        "settings": final,
        "standard_errors": dict(zip(regressors, (round(float(s), 4) for s in se), strict=True)),
        "toml": _toml(final),
    }
    if best == "market_anchored" and adds["ci95"][1] >= 0:
        report["recommendation"]["warning"] = (
            "the model does not add information beyond the market price on held-out events, "
            "so trades on this calibration would mostly bet on the market's own pricing errors"
        )
    return report
