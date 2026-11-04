"""Recalibration of model win probabilities (``ProbabilityCalibrationSettings``).

    logit(p) = slope * logit(p_model) + market_weight * logit(p_market)

slope > 1 spreads the model's probabilities away from 50% (a model that is too timid),
slope < 1 pulls them in. market_weight > 0 anchors the estimate to the live market price,
so the model has to disagree with the market by more before a trade clears its edge. The
map is monotone in p_model, so it is applied to the point estimate and to the conservative
bounds alike and a lower bound stays a lower bound. Fitting lives in
``research.calibration``.
"""

from __future__ import annotations

import math

from kalshi_trader.config import ProbabilityCalibrationSettings

P_CLIP = 1e-4


def logit(p: float) -> float:
    p = min(1.0 - P_CLIP, max(P_CLIP, p))
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def calibrate(
    p_model: float, cal: ProbabilityCalibrationSettings, p_market: float | None = None
) -> float:
    if cal.is_identity:
        return p_model
    z = cal.slope * logit(p_model)
    if cal.market_weight:
        if p_market is None:
            raise ValueError("market-anchored calibration needs the market probability")
        z += cal.market_weight * logit(p_market)
    return sigmoid(z)
