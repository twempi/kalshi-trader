"""Statistical win-probability estimation with explicit uncertainty.

Three quantities are kept deliberately separate:

1. Market-implied probability: from Kalshi prices (computed elsewhere, never used here
   except optionally as a *pre-match* strength prior).
2. Model probability: the Markov model evaluated at the posterior-mean serve probabilities,
   adjusted for in-match retirement risk.
3. Model uncertainty: the distribution of the model probability over posterior draws of
   the serve probabilities. Its ``credible_quantile`` lower bound (and the latency-robust
   variant) is what trading decisions use.

Parameterisation. Player serve-point win probabilities are p1 = mu + delta/2 and
p2 = mu - delta/2, where ``mu`` is the match's overall serve dominance (surface,
conditions; it barely changes who wins) and ``delta`` is the strength difference (it drives
the result). Priors are independent normals:

* mu ~ N(tour average, ``serve_mean_sd``)
* delta ~ N(delta0, sd), where delta0 makes the Markov pre-match win probability equal the
  last pre-match market price (Klaassen & Magnus-style calibration) and sd is
  ``strength_diff_sd_with_price``; without a pre-match price delta0 = 0 and the wider
  ``strength_diff_sd_no_price`` is used.

Live service points won/lost update the prior by sampling-importance-resampling with the
binomial likelihood, so small in-match samples move the estimate only modestly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kalshi_trader.config import ModelSettings
from kalshi_trader.models.markov import MarkovMatchModel, implied_serve_difference
from kalshi_trader.tennis.state import MatchFormat, ScoreState, ServeStats

MAX_PLAUSIBLE_SERVICE_POINTS = 400
P_MIN, P_MAX = 0.30, 0.95


@dataclass(frozen=True)
class StrengthPrior:
    base: float  # prior mean of mu (tour-average serve-point win probability)
    delta_mean: float
    delta_sd: float
    mu_sd: float
    source: str  # "pre_match_price" | "tour_average"
    pre_match_prob: float | None = None

    @property
    def p1_mean(self) -> float:
        return self.base + self.delta_mean / 2

    @property
    def p2_mean(self) -> float:
        return self.base - self.delta_mean / 2


@dataclass(frozen=True)
class PlayerProbability:
    point: float  # model probability
    low: float  # lower credible bound (parameter uncertainty)
    robust_low: float  # lower bound after the latency guard (used for EV)


@dataclass(frozen=True)
class ProbabilityEstimate:
    p1_point: float
    p1_mean: float
    p1_std: float
    p1_low: float
    p1_high: float
    p1_robust_low: float
    p1_robust_high: float
    expected_remaining_games: float
    serve_p1: float
    serve_p2: float
    prior: StrengthPrior
    latency_points: int
    samples: int
    credible_quantile: float
    effective_sample_size: float

    def for_player(self, index: int) -> PlayerProbability:
        if index == 0:
            return PlayerProbability(self.p1_point, self.p1_low, self.p1_robust_low)
        return PlayerProbability(1.0 - self.p1_point, 1.0 - self.p1_high, 1.0 - self.p1_robust_high)


def adjust_for_retirement(p: float, remaining_games: float, hazard_per_game: float) -> float:
    """Mix in the chance that either player retires before the match ends.

    With equal per-game hazards h for both players, P(someone retires) = 1 - exp(-2 h G);
    each player is equally likely to be the one who retires, and the opponent then wins.
    """
    if hazard_per_game <= 0 or remaining_games <= 0:
        return p
    r = 1.0 - math.exp(-2.0 * hazard_per_game * remaining_games)
    return (1.0 - r) * p + r / 2.0


@lru_cache(maxsize=4096)
def _cached_serve_difference(target: float, base: float, fmt: MatchFormat) -> float:
    return implied_serve_difference(target, base, fmt)


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


def _valid(stats: ServeStats | None) -> ServeStats | None:
    if stats is None or not (0 <= stats.total <= MAX_PLAUSIBLE_SERVICE_POINTS):
        return None
    return stats


class WinProbabilityEstimator:
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    def tour_base(self, tour_key: str) -> float:
        priors = self.settings.serve_win_prior
        return priors.get(tour_key) or priors.get("ATP") or 0.62

    def build_prior(
        self, tour_key: str, fmt: MatchFormat, pre_match_prob_p1: float | None
    ) -> StrengthPrior:
        cfg = self.settings
        base = self.tour_base(tour_key)
        if (
            cfg.use_pre_match_price_prior
            and pre_match_prob_p1 is not None
            and 0.0 < pre_match_prob_p1 < 1.0
        ):
            target = round(pre_match_prob_p1, 3)  # rounding keeps the cache effective
            delta = _cached_serve_difference(target, base, fmt)
            return StrengthPrior(
                base,
                delta,
                cfg.strength_diff_sd_with_price,
                cfg.serve_mean_sd,
                "pre_match_price",
                pre_match_prob_p1,
            )
        return StrengthPrior(
            base, 0.0, cfg.strength_diff_sd_no_price, cfg.serve_mean_sd, "tour_average", None
        )

    def posterior_draws(
        self,
        prior: StrengthPrior,
        serve_stats: tuple[ServeStats | None, ServeStats | None],
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """(p1 draws, p2 draws, posterior mean p1, posterior mean p2, effective sample size)."""
        m = max(500, self.settings.prior_draws)
        mu = rng.normal(prior.base, prior.mu_sd, m)
        delta = rng.normal(prior.delta_mean, prior.delta_sd, m)
        p1 = np.clip(mu + delta / 2, P_MIN, P_MAX)
        p2 = np.clip(mu - delta / 2, P_MIN, P_MAX)
        log_w = np.zeros(m)
        for stats, p in ((_valid(serve_stats[0]), p1), (_valid(serve_stats[1]), p2)):
            if stats is not None and stats.total > 0:
                log_w += stats.points_won * np.log(p) + stats.points_lost * np.log1p(-p)
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        ess = float(1.0 / np.sum(w * w))
        n = max(16, self.settings.uncertainty_samples)
        idx = rng.choice(m, size=n, replace=True, p=w)
        return p1[idx], p2[idx], float(np.sum(w * p1)), float(np.sum(w * p2)), ess

    def estimate(
        self,
        state: ScoreState,
        prior: StrengthPrior,
        serve_stats: tuple[ServeStats | None, ServeStats | None] = (None, None),
        seed_key: str = "",
    ) -> ProbabilityEstimate:
        cfg = self.settings
        rng = np.random.default_rng(_seed(seed_key or repr(state)))
        s1, s2, m1, m2, ess = self.posterior_draws(prior, serve_stats, rng)
        hazard = cfg.retirement_hazard_per_game

        point_model = MarkovMatchModel((m1, m2), state.fmt)
        p_point_raw, remaining = point_model.win_probability(state)
        p_point = adjust_for_retirement(p_point_raw, remaining, hazard)

        reachable = state.reachable_states(max(0, cfg.latency_points))
        values = np.empty((len(s1), len(reachable)))
        for i in range(len(s1)):
            model = MarkovMatchModel((float(s1[i]), float(s2[i])), state.fmt)
            for j, st in enumerate(reachable):
                p, games = model.win_probability(st)
                values[i, j] = adjust_for_retirement(p, games, hazard)

        current = reachable.index(state)
        q = cfg.credible_quantile
        col = values[:, current]
        lows = np.quantile(values, q, axis=0)
        highs = np.quantile(values, 1.0 - q, axis=0)
        return ProbabilityEstimate(
            p1_point=float(p_point),
            p1_mean=float(col.mean()),
            p1_std=float(col.std()),
            p1_low=float(lows[current]),
            p1_high=float(highs[current]),
            p1_robust_low=float(lows.min()),
            p1_robust_high=float(highs.max()),
            expected_remaining_games=float(remaining),
            serve_p1=m1,
            serve_p2=m2,
            prior=prior,
            latency_points=cfg.latency_points,
            samples=len(s1),
            credible_quantile=q,
            effective_sample_size=ess,
        )
