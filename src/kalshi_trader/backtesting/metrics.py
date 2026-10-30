"""Performance metrics shared by live paper trading analytics and backtests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any


@dataclass(frozen=True)
class TradeOutcome:
    opened_ts: float
    closed_ts: float
    strategy: str
    entry_price: float  # average price per contract (excluding fees)
    contracts: float
    cost: float  # entry notional + entry fee (capital at risk)
    pnl: float  # realised P&L after all fees
    fees: float
    model_prob: float | None = None
    event: str | None = None

    @property
    def won(self) -> bool:
        return self.pnl > 0


def equity_curve(
    trades: Sequence[TradeOutcome], starting_balance: float
) -> list[tuple[float, float]]:
    """Realised equity after each trade closes (chronological)."""
    curve = []
    equity = starting_balance
    for trade in sorted(trades, key=lambda t: t.closed_ts):
        equity += trade.pnl
        curve.append((trade.closed_ts, equity))
    return curve


def max_drawdown(
    curve: Sequence[tuple[float, float]], starting_balance: float
) -> tuple[float, float]:
    """(max drawdown in dollars, as a fraction of the running peak)."""
    peak = starting_balance
    worst_abs = worst_frac = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        dd = peak - equity
        if dd > worst_abs:
            worst_abs = dd
        if peak > 0:
            worst_frac = max(worst_frac, dd / peak)
    return worst_abs, worst_frac


def drawdown_series(
    curve: Sequence[tuple[float, float]], starting_balance: float
) -> list[tuple[float, float]]:
    peak = starting_balance
    series = []
    for ts, equity in curve:
        peak = max(peak, equity)
        series.append((ts, (equity - peak) / peak if peak > 0 else 0.0))
    return series


def peak_exposure(trades: Sequence[TradeOutcome]) -> float:
    """Largest total capital simultaneously at risk."""
    events = []
    for t in trades:
        events.append((t.opened_ts, t.cost))
        events.append((t.closed_ts, -t.cost))
    events.sort(key=lambda e: (e[0], e[1]))  # releases before new entries at equal times
    exposure = worst = 0.0
    for _, delta in events:
        exposure += delta
        worst = max(worst, exposure)
    return worst


def compute_metrics(trades: Sequence[TradeOutcome], starting_balance: float) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "note": "no completed trades"}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = sum(pnls)
    fees = sum(t.fees for t in trades)
    deployed = sum(t.cost for t in trades)
    curve = equity_curve(trades, starting_balance)
    dd_abs, dd_frac = max_drawdown(curve, starting_balance)
    gross_win, gross_loss = sum(wins), -sum(losses)
    mean = total / n
    std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
    return {
        "trades": n,
        "total_pnl": round(total, 4),
        "total_fees": round(fees, 4),
        "pnl_before_fees": round(total + fees, 4),
        "return_on_bankroll": total / starting_balance if starting_balance else None,
        "roi_on_capital_deployed": total / deployed if deployed else None,
        "win_rate": len(wins) / n,
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": gross_win / len(wins) if wins else None,
        "avg_loss": -gross_loss / len(losses) if losses else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "ev_per_trade": mean,
        "pnl_std_per_trade": std,
        "max_drawdown": round(dd_abs, 4),
        "max_drawdown_fraction": dd_frac,
        "avg_stake_fraction": (deployed / n) / starting_balance if starting_balance else None,
        "peak_exposure_fraction": peak_exposure(trades) / starting_balance
        if starting_balance
        else None,
        "avg_entry_price": sum(t.entry_price for t in trades) / n,
        "final_equity": round(starting_balance + total, 4),
    }


def win_rate_by_bucket(
    trades: Sequence[TradeOutcome],
    edges: Sequence[float] = (0.0, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.0),
) -> list[dict[str, Any]]:
    """Empirical win rate vs average entry price, per entry-price bucket (calibration)."""
    rows = []
    for lo, hi in pairwise(edges):
        bucket = [t for t in trades if lo <= t.entry_price < hi]
        if not bucket:
            continue
        wins = sum(1 for t in bucket if t.won)
        rows.append(
            {
                "bucket": f"{lo:.2f}-{hi:.2f}",
                "trades": len(bucket),
                "win_rate": wins / len(bucket),
                "avg_entry_price": sum(t.entry_price for t in bucket) / len(bucket),
                "pnl": sum(t.pnl for t in bucket),
            }
        )
    return rows


def by_strategy(trades: Sequence[TradeOutcome], starting_balance: float) -> dict[str, Any]:
    names = sorted({t.strategy for t in trades})
    return {
        name: compute_metrics([t for t in trades if t.strategy == name], starting_balance)
        for name in names
    }


def pnl_histogram(trades: Sequence[TradeOutcome], bins: int = 12) -> list[dict[str, float]]:
    if not trades:
        return []
    values = [t.pnl for t in trades]
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": c} for i, c in enumerate(counts)
    ]
