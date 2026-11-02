"""Why is (or isn't) the system trading? Counts every opportunity outcome per strategy.

* ``last_cycle``: status counts from the most recent evaluation cycle.
* ``totals``: counts by (strategy, status, normalised first reason) since start.
* Persisted daily aggregates (``strategy_diagnostics``) and a throttled log of individual
  opportunities (``opportunities``): a row is written when an opportunity's status/reasons
  change, otherwise at most every ``persist_interval_s`` (actionable) or ``quiet_interval_s``
  (filtered/unsupported).

Reasons are normalised (numbers replaced by ``#``) so "order book stale (12.3s)" and
"order book stale (14.0s)" count as the same cause.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kalshi_trader.core.opportunity import DecisionStatus, Opportunity
from kalshi_trader.database.repository import Repository
from kalshi_trader.database.tables import DiagnosticCountRow, OpportunityRow

NUMBER = re.compile(r"[-+]?\$?\d+(?:\.\d+)?%?")
TICKER = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9.]+)+\b")


def reason_key(reason: str) -> str:
    """Group equivalent reasons: market tickers -> <market>, numbers -> #."""
    return NUMBER.sub("#", TICKER.sub("<market>", reason)).strip()[:160] or "-"


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")


class Diagnostics:
    def __init__(
        self,
        repo: Repository | None,
        data_kind: str = "live",
        persist_interval_s: float = 60.0,
        persist_opportunities: bool = True,
        quiet_interval_s: float = 600.0,
    ) -> None:
        self.repo = repo
        self.data_kind = data_kind
        self.persist_interval_s = persist_interval_s
        self.quiet_interval_s = quiet_interval_s
        self.persist_opportunities = persist_opportunities and repo is not None
        self.last_cycle: dict[str, Counter[str]] = {}
        self.totals: Counter[tuple[str, str, str]] = Counter()
        self._pending: Counter[tuple[str, str, str, str]] = Counter()
        self._written: dict[str, tuple[str, float]] = {}
        self._rows: list[dict[str, Any]] = []
        self.latest: dict[str, dict[str, Any]] = {}  # opportunity id -> summary (live view)
        self.errors: Counter[str] = Counter()

    def record(self, opps: list[Opportunity], now: float, modes: dict[str, str]) -> None:
        cycle: dict[str, Counter[str]] = {}
        latest: dict[str, dict[str, Any]] = {}
        for opp in opps:
            status = opp.status.value
            cycle.setdefault(opp.strategy, Counter())[status] += 1
            reason = reason_key(opp.reasons[0]) if opp.reasons else "-"
            self.totals[(opp.strategy, status, reason)] += 1
            self._pending[(_day(now), opp.strategy, status, reason)] += 1
            summary = opp.summary() | {"mode": modes.get(opp.strategy, ""), "ts": now}
            latest[opp.opportunity_id] = summary
            if self.persist_opportunities:
                self._maybe_write(opp, summary, now)
        self.last_cycle = cycle
        self.latest = latest

    def record_error(self, strategy: str, error: str) -> None:
        self.errors[f"{strategy}: {reason_key(error)[:120]}"] += 1

    def _maybe_write(self, opp: Opportunity, summary: dict[str, Any], now: float) -> None:
        # Non-actionable records (filtered/unsupported) are keyed on the normalised reason
        # only and rewritten at most every `quiet_interval_s`, so price ticks on markets that
        # never qualify do not flood the table. Actionable ones also track the best ask.
        quiet = opp.status in (DecisionStatus.FILTERED, DecisionStatus.UNSUPPORTED)
        reasons = "|".join(reason_key(r) for r in opp.reasons)
        digest = f"{opp.status.value}|{reasons}" + ("" if quiet else f"|{summary['best_ask']}")
        interval = self.quiet_interval_s if quiet else self.persist_interval_s
        last = self._written.get(opp.opportunity_id)
        if last and last[0] == digest and now - last[1] < interval:
            return
        self._written[opp.opportunity_id] = (digest, now)
        self._rows.append(
            {
                "ts": now,
                "opportunity_id": opp.opportunity_id,
                "strategy": opp.strategy,
                "strategy_version": opp.strategy_version,
                "mode": summary["mode"],
                "kind": opp.kind,
                "category": opp.category,
                "stage": opp.stage,
                "event_ticker": opp.event_ticker,
                "market_ticker": opp.market_ticker,
                "title": opp.title[:256],
                "side": opp.side,
                "legs_json": json.dumps(summary["legs"]) if opp.kind == "basket" else None,
                "best_bid": summary["best_bid"],
                "best_ask": summary["best_ask"],
                "ask_size": summary["ask_size"],
                "fair_value": summary["fair_value"],
                "fair_low": summary["fair_low"],
                "contracts": summary["contracts"],
                "limit_price": summary["limit_price"],
                "avg_price": summary["avg_price"],
                "fees": summary["fees"],
                "cost": summary["cost"],
                "expected_profit": opp.expected_profit,
                "expected_profit_low": opp.expected_profit_low,
                "expected_roc": opp.expected_roc,
                "max_loss": summary["max_loss"],
                "book_age_s": opp.book_age_s,
                "score_age_s": opp.score_age_s,
                "provenance_json": json.dumps(opp.provenance, default=str)[:4000],
                "status": opp.status.value,
                "reasons_json": json.dumps(opp.reasons)[:4000],
                "data_kind": self.data_kind,
            }
        )

    def flush(self) -> None:
        if self.repo is None:
            self._pending.clear()
            return
        rows, self._rows = self._rows, []
        pending, self._pending = self._pending, Counter()
        now = time.time()
        with self.repo.db.session() as s:
            if rows:
                s.execute(sqlite_insert(OpportunityRow), rows)
            for (day, strategy, status, reason), count in pending.items():
                stmt = sqlite_insert(DiagnosticCountRow).values(
                    day=day,
                    strategy=strategy,
                    status=status,
                    reason=reason,
                    count=count,
                    updated_ts=now,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["day", "strategy", "status", "reason"],
                    set_={"count": DiagnosticCountRow.count + count, "updated_ts": now},
                )
                s.execute(stmt)
        if len(self._written) > 50_000:
            cutoff = now - 3600
            self._written = {k: v for k, v in self._written.items() if v[1] >= cutoff}

    def snapshot(self) -> dict[str, Any]:
        by_strategy: dict[str, dict[str, Any]] = {}
        for (strategy, status, reason), count in self.totals.items():
            entry = by_strategy.setdefault(strategy, {"statuses": Counter(), "reasons": Counter()})
            entry["statuses"][status] += count
            if status not in ("filled", "partial", "observed", "approved"):
                entry["reasons"][reason] += count
        return {
            "last_cycle": {k: dict(v) for k, v in self.last_cycle.items()},
            "since_start": {
                k: {
                    "statuses": dict(v["statuses"]),
                    "top_reasons": v["reasons"].most_common(8),
                }
                for k, v in by_strategy.items()
            },
            "errors": dict(self.errors),
        }
