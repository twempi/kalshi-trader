"""Strategy lifecycle: evidence status, operator pause, and what each status allows.

Statuses:

``experimental``      new, or its rules (version + config hash) changed; no evidence yet
``collecting_data``   observe mode, accumulating forward shadow-ledger results
``ready_for_paper``   forward evidence meets every predefined requirement; an operator may
                      switch it to ``mode = "paper"`` (never automatic)
``paper_trading``     ``mode = "paper"``: eligible for the main account. Until validated it
                      is sized at ``risk.unvalidated_size_multiplier``
``paused``            paused by an operator (persists across restarts)
``rejected``          forward evidence is significantly negative; main-account entries are
                      blocked even in paper mode until its rules change

Evidence = settled trades in the strategy's *shadow ledger* made under the strategy's
current version tag. The shadow ledger trades every approved opportunity of that strategy
from the moment the rules took effect, so it is genuinely out-of-sample: nothing was tuned
on it. Backtests are reported separately and never change a status on their own, so a
strategy cannot be enabled because it looked good in historical optimisation.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any

from sqlalchemy import select

from kalshi_trader.backtesting.metrics import TradeOutcome, compute_metrics
from kalshi_trader.backtesting.stats import significance_report
from kalshi_trader.config import EvidenceSettings
from kalshi_trader.database.repository import Repository
from kalshi_trader.database.tables import StrategyStateRow, StrategyStatusHistoryRow

STATUSES = (
    "experimental",
    "collecting_data",
    "ready_for_paper",
    "paper_trading",
    "paused",
    "rejected",
)
ONE = Decimal("1")


def evaluate_evidence(
    trades: list[TradeOutcome], evidence: EvidenceSettings, starting_balance: float
) -> dict[str, Any]:
    """Apply the predefined evidence requirements to a list of settled trades."""
    n = len(trades)
    events = len({t.event for t in trades if t.event})
    metrics = compute_metrics(trades, starting_balance)
    sig = significance_report(trades)
    pvalue = sig.get("calibration_null_pvalue")
    ci = sig.get("mean_pnl_ci95_cluster_bootstrap")
    enough = n >= evidence.min_out_of_sample_trades and events >= evidence.min_out_of_sample_events
    failures = []
    if n < evidence.min_out_of_sample_trades:
        failures.append(f"{n}/{evidence.min_out_of_sample_trades} trades")
    if events < evidence.min_out_of_sample_events:
        failures.append(f"{events}/{evidence.min_out_of_sample_events} independent events")
    if pvalue is None or pvalue > evidence.max_calibration_pvalue:
        failures.append(f"calibration-null p={pvalue} > {evidence.max_calibration_pvalue}")
    if evidence.require_positive_ci and not (ci and ci[0] > 0):
        failures.append("bootstrap CI of mean P&L does not exclude zero")
    dd = metrics.get("max_drawdown_fraction") or 0.0
    if dd > evidence.max_drawdown_fraction:
        failures.append(f"drawdown {dd:.1%} > {evidence.max_drawdown_fraction:.0%}")
    negative = enough and ((pvalue is not None and pvalue > 0.95) or (ci is not None and ci[1] < 0))
    return {
        "trades": n,
        "events": events,
        "validated": not failures,
        "negative": negative,
        "failures": failures,
        "metrics": metrics,
        "significance": sig,
    }


class StrategyRegistry:
    def __init__(self, repo: Repository, evidence: EvidenceSettings) -> None:
        self.repo = repo
        self.evidence = evidence
        self._cache: dict[str, StrategyStateRow] = {}

    def _history(self, s: Any, row: StrategyStateRow, reason: str, now: float) -> None:
        s.add(
            StrategyStatusHistoryRow(
                ts=now,
                name=row.name,
                version=row.version,
                config_hash=row.config_hash,
                status=row.status,
                reason=reason,
            )
        )

    def sync(self, strategies: list[tuple[str, str, str, str]], now: float | None = None) -> None:
        """Register (name, version, config_hash, mode) tuples; a rules change resets status."""
        now = now if now is not None else time.time()
        with self.repo.db.session() as s:
            for name, version, config_hash, mode in strategies:
                row = s.get(StrategyStateRow, name)
                initial = "paper_trading" if mode == "paper" else "experimental"
                if row is None:
                    row = StrategyStateRow(
                        name=name,
                        version=version,
                        config_hash=config_hash,
                        status=initial,
                        status_reason="registered",
                        paused=False,
                        updated_ts=now,
                    )
                    s.add(row)
                    self._history(s, row, "registered", now)
                elif row.version != version or row.config_hash != config_hash:
                    row.version, row.config_hash = version, config_hash
                    row.status = initial
                    row.status_reason = "rules changed: earlier evidence no longer applies"
                    row.evidence_json = None
                    row.updated_ts = now
                    self._history(s, row, row.status_reason, now)
                elif mode == "paper" and row.status in ("experimental", "collecting_data"):
                    row.status, row.status_reason, row.updated_ts = (
                        "paper_trading",
                        "mode set to paper",
                        now,
                    )
                    self._history(s, row, row.status_reason, now)
                elif mode != "paper" and row.status == "paper_trading":
                    row.status, row.status_reason, row.updated_ts = (
                        "collecting_data",
                        "mode set to observe",
                        now,
                    )
                    self._history(s, row, row.status_reason, now)
        self.reload()

    def reload(self) -> None:
        with self.repo.db.session() as s:
            self._cache = {r.name: r for r in s.scalars(select(StrategyStateRow))}

    def state(self, name: str) -> StrategyStateRow | None:
        return self._cache.get(name)

    def all(self) -> list[StrategyStateRow]:
        return list(self._cache.values())

    def set_paused(
        self,
        name: str,
        paused: bool,
        reason: str = "",
        mode: str = "observe",
        now: float | None = None,
    ) -> bool:
        now = now if now is not None else time.time()
        with self.repo.db.session() as s:
            row = s.get(StrategyStateRow, name)
            if row is None:
                return False
            row.paused = paused
            resumed = "paper_trading" if mode == "paper" else "collecting_data"
            row.status = "paused" if paused else resumed
            row.status_reason = reason or ("paused by operator" if paused else "resumed")
            row.updated_ts = now
            self._history(s, row, row.status_reason, now)
        self.reload()
        return True

    def validated(self, name: str) -> bool:
        row = self._cache.get(name)
        if row is None or not row.evidence_json:
            return False
        return bool(json.loads(row.evidence_json).get("validated"))

    def allows_main(self, name: str, mode: str) -> tuple[bool, str]:
        if mode != "paper":
            return False, "observe mode (shadow ledger only)"
        row = self._cache.get(name)
        if row is None:
            return False, "strategy not registered"
        if row.paused:
            return False, "strategy paused by operator"
        if row.status == "rejected":
            return False, f"strategy rejected: {row.status_reason}"
        return True, ""

    def size_multiplier(self, name: str, unvalidated: Decimal) -> Decimal:
        return ONE if self.validated(name) else unvalidated

    def update_evidence(
        self,
        name: str,
        mode: str,
        trades: list[TradeOutcome],
        starting_balance: float,
        now: float | None = None,
    ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        result = evaluate_evidence(trades, self.evidence, starting_balance)
        with self.repo.db.session() as s:
            row = s.get(StrategyStateRow, name)
            if row is None:
                return result
            row.evidence_json = json.dumps(
                {k: result[k] for k in ("trades", "events", "validated", "negative", "failures")}
                | {"metrics": result["metrics"], "verdict": result["significance"].get("verdict")},
                default=str,
            )
            if not row.paused:
                if result["negative"]:
                    status, reason = "rejected", "forward evidence significantly negative"
                elif result["validated"]:
                    status = "paper_trading" if mode == "paper" else "ready_for_paper"
                    reason = "forward evidence meets requirements"
                elif mode == "paper":
                    status, reason = (
                        "paper_trading",
                        "unvalidated: " + "; ".join(result["failures"][:3]),
                    )
                elif result["trades"] > 0:
                    status, reason = "collecting_data", "; ".join(result["failures"][:3])
                else:
                    status, reason = row.status, row.status_reason
                    if status not in ("experimental", "collecting_data"):
                        status, reason = "experimental", "no settled shadow trades yet"
                if status != row.status:
                    row.status = status
                    self._history(s, row, reason, now)
                row.status_reason = reason
            row.updated_ts = now
        self.reload()
        return result
