"""Could the observed results be luck? Significance tests for binary-contract strategies.

The key null hypothesis is *market calibration*: each contract bought at all-in price c
wins with probability equal to its pre-fee price p. Under that null the expected P&L is
negative (fees), so a strategy must beat it by a margin that is unlikely under random
variation. Tests are deliberately simple and transparent:

* Monte Carlo p-value of total P&L under the calibration null (exact for independent bets).
* Cluster bootstrap CI of mean P&L per trade (trades in the same match are not independent).
* Wilson interval of the win rate vs the break-even win rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from kalshi_trader.backtesting.metrics import TradeOutcome


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_null_pvalue(
    trades: Sequence[TradeOutcome], n_sims: int = 20000, seed: int = 7
) -> float | None:
    """P(total P&L >= observed) if every contract won with probability = its entry price."""
    if not trades:
        return None
    rng = np.random.default_rng(seed)
    price = np.array([t.entry_price for t in trades])
    contracts = np.array([t.contracts for t in trades])
    fees = np.array([t.fees for t in trades])
    observed = sum(t.pnl for t in trades)
    fixed = float((price * contracts + fees).sum())
    exceed = 0
    chunk = max(1, min(n_sims, 2_000_000 // max(1, len(trades))))  # bound memory use
    done = 0
    while done < n_sims:
        size = min(chunk, n_sims - done)
        wins = rng.random((size, len(trades))) < price
        sim_pnl = wins.astype(float) @ contracts - fixed
        exceed += int(np.sum(sim_pnl >= observed - 1e-9))
        done += size
    return float((exceed + 1) / (n_sims + 1))


def cluster_bootstrap_mean_ci(
    trades: Sequence[TradeOutcome], n_boot: int = 5000, alpha: float = 0.05, seed: int = 11
) -> tuple[float, float] | None:
    """Percentile CI for mean P&L per trade, resampling whole matches (events)."""
    if len(trades) < 2:
        return None
    clusters: dict[str, list[float]] = {}
    for i, t in enumerate(trades):
        clusters.setdefault(t.event or f"trade-{i}", []).append(t.pnl)
    groups = list(clusters.values())
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def significance_report(trades: Sequence[TradeOutcome]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "verdict": "no trades - nothing to evaluate"}
    wins = sum(1 for t in trades if t.won)
    lo, hi = wilson_interval(wins, n)
    breakeven = sum(t.cost / t.contracts for t in trades if t.contracts) / n
    pvalue = calibration_null_pvalue(trades)
    ci = cluster_bootstrap_mean_ci(trades)
    return {
        "trades": n,
        "win_rate": wins / n,
        "win_rate_ci95": [lo, hi],
        "breakeven_win_rate": breakeven,
        "calibration_null_pvalue": pvalue,
        "mean_pnl_ci95_cluster_bootstrap": list(ci) if ci else None,
        "verdict": verdict(n, pvalue, ci),
    }


def verdict(n: int, pvalue: float | None, ci: Sequence[float] | None) -> str:
    """Plain-language reading of the calibration-null p-value (one-sided: P(P&L >= observed)
    under fair prices, fees included) and the bootstrap CI of mean P&L per trade."""
    if n < 30:
        return f"only {n} trades: far too few to distinguish skill from luck; keep collecting data"
    if pvalue is not None and pvalue < 0.01 and ci is not None and ci[0] > 0:
        return (
            "results are unlikely under the market-calibration null (p<0.01) and the "
            "bootstrap CI of mean P&L excludes zero; still validate out-of-sample"
        )
    if pvalue is not None and pvalue < 0.05:
        return (
            "weak evidence against market calibration (p<0.05); not conclusive given "
            "multiple testing - validate out-of-sample"
        )
    if pvalue is not None and pvalue > 0.99:
        return (
            "significantly worse than fair prices (p>0.99): these contracts won less often "
            "than their prices implied, losing more than fees alone explain"
        )
    if pvalue is not None and pvalue > 0.95:
        return (
            "weak evidence that these contracts won less often than their prices implied (p>0.95)"
        )
    if ci is not None and ci[1] < 0:
        return (
            "win rate consistent with fair prices, but the strategy loses money after fees "
            "(bootstrap CI of mean P&L is below zero)"
        )
    return "results are consistent with random variation around fair market prices"
