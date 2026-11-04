"""Local web dashboard (FastAPI + static vanilla-JS frontend, no external CDNs).

All data is read from the SQLite database the engine writes, so the dashboard also works
when the engine is stopped (showing the last known state). Control endpoints only flip
persisted local flags (pause/resume/emergency stop) - they never touch the exchange.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from kalshi_trader.backtesting.metrics import (
    TradeOutcome,
    by_strategy,
    compute_metrics,
    drawdown_series,
    outcomes_from_trades,
    pnl_histogram,
    win_rate_by_bucket,
)
from kalshi_trader.backtesting.stats import significance_report
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import (
    EMERGENCY_STOP,
    PAUSE_REASON,
    TRADING_ENABLED,
    Repository,
)
from kalshi_trader.database.tables import (
    BacktestRunRow,
    DiagnosticCountRow,
    EquitySnapshotRow,
    OpportunityRow,
    PaperAccountRow,
    PaperTradeRow,
    SimulatedQuoteFillRow,
    StrategyStateRow,
)

STATIC_DIR = Path(__file__).parent / "static"
ENGINE_STALE_S = 15.0
CALIBRATION_CACHE_S = 600.0  # the calibration fit reads every recorded prediction
PREDICTION_FIELDS = (
    "ts",
    "event",
    "market",
    "tour",
    "stage",
    "prior_source",
    "p_model",
    "p_model_low",
    "yes_bid",
    "yes_ask",
    "score_age_s",
    "outcome",
)
DATA_LABEL = (
    "SIMULATED paper trading on real, live Kalshi order books. "
    "No real orders are placed. Model probabilities are estimates."
)
MAIN_ACCOUNT_ID = 1
# Data kinds shown in the UI so simulated, replayed and synthetic data are never confused.
DATA_KINDS = {
    "live_paper": "simulated fill on a real live order book (main paper account)",
    "shadow": "simulated fill on a real live order book (per-strategy shadow ledger)",
    "replay": "simulated fill on recorded order books (backtest)",
    "synthetic": "synthetic test data (never a market result)",
    "research_sim": "hypothetical maker fills (market-making research simulation)",
}
STRATEGY_NAMES = ("tennis_model", "cross_market", "late_match", "price_threshold", "market_making")


def _f(value: Decimal | float | None) -> float | None:
    return None if value is None else float(value)


TRADE_FIELDS = tuple(c.name for c in PaperTradeRow.__table__.columns)


def trade_to_dict(t: PaperTradeRow) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in TRADE_FIELDS:
        value = getattr(t, name)
        out[name] = float(value) if isinstance(value, Decimal) else value
    return out


def outcome_from_row(t: PaperTradeRow) -> TradeOutcome:
    return outcomes_from_trades([t])[0]


def _modes(settings: Settings) -> dict[str, str]:
    cfg = settings.strategies
    return {name: getattr(cfg, name).mode for name in STRATEGY_NAMES}


def create_app(settings: Settings, db: Database | None = None) -> FastAPI:
    database = db or Database(settings.db_path)
    repo = Repository(database)
    app = FastAPI(title="Kalshi Trader (paper)", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        if request.headers.get("tailscale-funnel-request"):
            # Tailscale Funnel means the public internet; this dashboard has no login.
            response = JSONResponse({"detail": "not available over Tailscale Funnel"}, 403)
        else:
            response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def closed_trades(account_id: int = MAIN_ACCOUNT_ID) -> list[PaperTradeRow]:
        with database.session() as s:
            stmt = select(PaperTradeRow).where(
                PaperTradeRow.status.in_(["settled", "closed"]),
                PaperTradeRow.account_id == account_id,
            )
            return list(s.scalars(stmt.order_by(PaperTradeRow.closed_ts)))

    def open_trades(account_id: int = MAIN_ACCOUNT_ID) -> list[PaperTradeRow]:
        with database.session() as s:
            stmt = select(PaperTradeRow).where(
                PaperTradeRow.status == "open", PaperTradeRow.account_id == account_id
            )
            return list(s.scalars(stmt.order_by(PaperTradeRow.opened_ts)))

    def accounts() -> dict[str, PaperAccountRow]:
        with database.session() as s:
            return {(a.name or "main"): a for a in s.scalars(select(PaperAccountRow))}

    def ledger_account_id(ledger: str | None) -> int:
        """``main`` (the default) or a strategy name, for that strategy's shadow ledger."""
        if not ledger or ledger == "main":
            return MAIN_ACCOUNT_ID
        account = accounts().get(f"shadow:{ledger}") if ledger in STRATEGY_NAMES else None
        if account is None:
            raise HTTPException(404, "unknown ledger: use main or a strategy name")
        return account.id

    def marks() -> dict[tuple[str, str], float]:
        result = {}
        for view in repo.live_views():
            ticker = view["market_ticker"]
            if view.get("yes_bid") is not None:
                result[(ticker, "yes")] = view["yes_bid"]
            if view.get("yes_ask") is not None:
                result[(ticker, "no")] = round(1 - view["yes_ask"], 4)
        return result

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        main = EquitySnapshotRow.account_id == MAIN_ACCOUNT_ID
        with database.session() as s:
            account = s.get(PaperAccountRow, MAIN_ACCOUNT_ID)
            last_equity = s.scalars(
                select(EquitySnapshotRow).where(main).order_by(EquitySnapshotRow.ts.desc()).limit(1)
            ).first()
            day_start = time.time() - (time.time() % 86400)
            first_today = s.scalars(
                select(EquitySnapshotRow)
                .where(main, EquitySnapshotRow.ts >= day_start)
                .order_by(EquitySnapshotRow.ts)
                .limit(1)
            ).first()
            strategies = list(s.scalars(select(StrategyStateRow)))
        current_marks = marks()
        opens = open_trades()
        open_value = sum(
            (
                current_marks.get((t.market_ticker, t.side), float(t.entry_price))
                * float(t.contracts)
            )
            for t in opens
        )
        closed = closed_trades()
        wins = sum(1 for t in closed if (t.realized_pnl or 0) > 0)
        state = repo.all_state()
        health = {h.component: h for h in repo.all_health()}
        engine = health.get("engine")
        running = bool(
            engine and engine.status == "ok" and time.time() - engine.updated_ts < ENGINE_STALE_S
        )
        cash = _f(account.cash) if account else None
        equity = (cash or 0.0) + open_value if account else None
        day_ref = _f(first_today.equity) if first_today else None
        open_cost = sum(float(t.entry_cost + t.entry_fee) for t in opens)
        peak = max(_f(account.peak_equity) or 0.0, equity or 0.0) if account else None
        modes = _modes(settings)
        components = list(health.values())
        unhealthy = [
            h.component
            for h in components
            if h.status not in ("ok", "disabled", "stopped", "starting")
            or (h.status == "ok" and time.time() - h.updated_ts > 60)
        ]
        return {
            "label": DATA_LABEL,
            "starting_balance": _f(account.starting_balance) if account else None,
            "cash": cash,
            "open_value": open_value,
            "equity": equity,
            "realized_pnl": _f(account.realized_pnl) if account else None,
            "total_pnl": (equity - float(account.starting_balance)) if account else None,
            "fees_paid": _f(account.fees_paid) if account else None,
            "daily_pnl": (equity - day_ref)
            if (equity is not None and day_ref is not None)
            else None,
            "last_equity_snapshot_ts": last_equity.ts if last_equity else None,
            "open_positions": len(opens),
            "completed_trades": len(closed),
            "win_rate": wins / len(closed) if closed else None,
            "engine_running": running,
            "engine_last_heartbeat": engine.updated_ts if engine else None,
            "trading_enabled": state.get(TRADING_ENABLED, "true") == "true",
            "emergency_stop": state.get(EMERGENCY_STOP, "false") == "true",
            "pause_reason": state.get(PAUSE_REASON) or None,
            "record_only": bool(
                engine
                and engine.details_json
                and json.loads(engine.details_json).get("record_only")
            ),
            "unrealized_pnl": (open_value - open_cost) if account else None,
            "open_cost": open_cost,
            "available_cash": cash,
            "peak_equity": peak,
            "drawdown_fraction": ((peak - equity) / peak) if peak and equity is not None else None,
            "strategies": [
                {
                    "name": r.name,
                    "mode": modes.get(r.name, "off"),
                    "status": r.status,
                    "paused": r.paused,
                }
                for r in strategies
            ],
            "active_paper_strategies": sum(
                1
                for r in strategies
                if modes.get(r.name) == "paper" and not r.paused and r.status != "rejected"
            ),
            "unhealthy_components": unhealthy,
        }

    @app.get("/api/markets")
    def markets() -> dict[str, Any]:
        return {"label": DATA_LABEL, "markets": repo.live_views(), "now": time.time()}

    @app.get("/api/positions")
    def positions() -> dict[str, Any]:
        current_marks = marks()
        rows = []
        for t in open_trades():
            row = trade_to_dict(t)
            mark = current_marks.get((t.market_ticker, t.side))
            row["mark"] = mark
            row["unrealized_pnl"] = (
                mark * float(t.contracts) - float(t.entry_cost + t.entry_fee)
                if mark is not None
                else None
            )
            rows.append(row)
        return {"positions": rows}

    @app.get("/api/trades")
    def trades(limit: int = 500, ledger: str | None = None) -> dict[str, Any]:
        account_id = ledger_account_id(ledger)
        rows = [trade_to_dict(t) for t in closed_trades(account_id)][-max(1, min(limit, 5000)) :]
        return {"ledger": ledger or "main", "trades": list(reversed(rows))}

    def opportunity_rows(
        max_age_s: float, limit: int, **filters: str | None
    ) -> list[dict[str, Any]]:
        since = time.time() - max_age_s
        with database.session() as s:
            stmt = select(OpportunityRow).where(OpportunityRow.ts >= since)
            for column in ("strategy", "status", "category"):
                value = filters.get(column)
                if value:
                    stmt = stmt.where(getattr(OpportunityRow, column) == value)
            rows = list(s.scalars(stmt.order_by(OpportunityRow.ts.desc()).limit(20000)))
        latest: dict[str, OpportunityRow] = {}
        for r in rows:  # newest first: keep each opportunity's latest state
            latest.setdefault(r.opportunity_id, r)
        out = []
        for r in list(latest.values())[:limit]:
            out.append(
                {
                    "ts": r.ts,
                    "age_s": time.time() - r.ts,
                    "opportunity_id": r.opportunity_id,
                    "strategy": r.strategy,
                    "strategy_version": r.strategy_version,
                    "mode": r.mode,
                    "kind": r.kind,
                    "category": r.category,
                    "stage": r.stage,
                    "event_ticker": r.event_ticker,
                    "market": r.market_ticker,
                    "title": r.title,
                    "side": r.side,
                    "legs": json.loads(r.legs_json) if r.legs_json else None,
                    "best_bid": r.best_bid,
                    "best_ask": r.best_ask,
                    "ask_size": r.ask_size,
                    "fair_value": r.fair_value,
                    "fair_low": r.fair_low,
                    "contracts": r.contracts,
                    "limit_price": r.limit_price,
                    "avg_price": r.avg_price,
                    "fees": r.fees,
                    "cost": r.cost,
                    "expected_profit": r.expected_profit,
                    "expected_profit_low": r.expected_profit_low,
                    "expected_roc": r.expected_roc,
                    "max_loss": r.max_loss,
                    "book_age_s": r.book_age_s,
                    "score_age_s": r.score_age_s,
                    "status": r.status,
                    "reasons": json.loads(r.reasons_json or "[]"),
                    "data_kind": r.data_kind,
                }
            )
        return out

    @app.get("/api/opportunities")
    def opportunities(
        strategy: str | None = None,
        status: str | None = None,
        category: str | None = None,
        max_age_s: float = 900,
        limit: int = 500,
    ) -> dict[str, Any]:
        rows = opportunity_rows(
            max(10.0, min(max_age_s, 7 * 86400.0)),
            max(1, min(limit, 5000)),
            strategy=strategy,
            status=status,
            category=category,
        )
        return {
            "label": DATA_LABEL,
            "data_kinds": DATA_KINDS,
            "opportunities": rows,
            "now": time.time(),
        }

    @app.get("/api/signals")
    def signals(limit: int = 100) -> dict[str, Any]:
        """Backward-compatible view: recent opportunities that reached the risk stage."""
        out = []
        for r in opportunity_rows(86400.0, max(1, min(limit, 1000))):
            if r["status"] in ("filtered", "unsupported"):
                continue
            out.append(
                {
                    "ts": r["ts"],
                    "strategy": r["strategy"],
                    "market": r["market"],
                    "side": r["side"],
                    "ask": r["best_ask"],
                    "model_prob": r["fair_value"],
                    "robust_prob": r["fair_low"],
                    "market_prob": None,
                    "ev": (r["expected_profit_low"] / r["contracts"])
                    if r["expected_profit_low"] is not None and r["contracts"]
                    else None,
                    "approved": r["status"] in ("approved", "observed", "filled", "partial"),
                    "status": r["status"],
                    "reasons": r["reasons"],
                    "score": None,
                }
            )
        return {"signals": out}

    @app.get("/api/strategies")
    def strategies() -> dict[str, Any]:
        modes = _modes(settings)
        start = float(settings.paper.starting_balance)
        accts = accounts()
        with database.session() as s:
            states = {r.name: r for r in s.scalars(select(StrategyStateRow))}
            day = time.strftime("%Y-%m-%d", time.gmtime())
            diag = list(s.scalars(select(DiagnosticCountRow).where(DiagnosticCountRow.day == day)))
        main_closed = outcomes_from_trades(closed_trades())
        main_open = open_trades()
        out = []
        for name in STRATEGY_NAMES:
            row = states.get(name)
            shadow = accts.get(f"shadow:{name}")
            shadow_outcomes = (
                outcomes_from_trades(closed_trades(shadow.id)) if shadow is not None else []
            )
            mine = [o for o in main_closed if o.strategy == name]
            counts: dict[str, int] = {}
            for d in diag:
                if d.strategy == name:
                    counts[d.status] = counts.get(d.status, 0) + d.count
            out.append(
                {
                    "name": name,
                    "mode": modes[name],
                    "status": row.status if row else "unregistered",
                    "status_reason": row.status_reason if row else "",
                    "paused": bool(row and row.paused),
                    "version": f"{row.version}+{row.config_hash[:8]}" if row else None,
                    "evidence": json.loads(row.evidence_json)
                    if row and row.evidence_json
                    else None,
                    "main": compute_metrics(mine, start),
                    "main_open_positions": sum(1 for t in main_open if t.strategy == name),
                    "main_capital_in_use": sum(
                        float(t.entry_cost + t.entry_fee) for t in main_open if t.strategy == name
                    ),
                    "shadow": compute_metrics(shadow_outcomes, start),
                    "shadow_cash": _f(shadow.cash) if shadow else None,
                    "today_counts": counts,
                }
            )
        return {"label": DATA_LABEL, "strategies": out}

    @app.get("/api/diagnostics")
    def diagnostics(day: str | None = None) -> dict[str, Any]:
        day = day or time.strftime("%Y-%m-%d", time.gmtime())
        with database.session() as s:
            rows = list(s.scalars(select(DiagnosticCountRow).where(DiagnosticCountRow.day == day)))
        engine = {h.component: h for h in repo.all_health()}.get("strategies")
        live = {}
        if engine and engine.details_json:
            live = (json.loads(engine.details_json) or {}).get("diagnostics") or {}
        by: dict[str, dict[str, Any]] = {}
        for r in rows:
            entry = by.setdefault(r.strategy, {"statuses": {}, "reasons": []})
            entry["statuses"][r.status] = entry["statuses"].get(r.status, 0) + r.count
            entry["reasons"].append({"status": r.status, "reason": r.reason, "count": r.count})
        for entry in by.values():
            entry["reasons"].sort(key=lambda x: -x["count"])
            entry["reasons"] = entry["reasons"][:25]
        return {
            "day": day,
            "by_strategy": by,
            "last_cycle": live.get("last_cycle", {}),
            "errors": live.get("errors", {}),
        }

    @app.get("/api/exposure")
    def exposure() -> dict[str, Any]:
        risk = settings.risk
        account = accounts().get("main")
        opens = open_trades()
        equity = (
            float(account.cash) + sum(float(t.entry_cost + t.entry_fee) for t in opens)
            if account
            else 0.0
        )

        def bucket(key: Any) -> dict[str, float]:
            out: dict[str, float] = {}
            for t in opens:
                held = float(t.contracts / t.entry_contracts) if t.entry_contracts else 1.0
                loss = float(t.max_loss if t.max_loss is not None else t.entry_cost + t.entry_fee)
                k = key(t) or "-"
                out[k] = out.get(k, 0.0) + loss * held
            return out

        return {
            "equity_at_cost": equity,
            "by_strategy": bucket(lambda t: t.strategy),
            "by_event": bucket(lambda t: t.event_ticker),
            "by_correlation_group": bucket(lambda t: t.correlation_group),
            "total_max_loss": sum(bucket(lambda t: "all").values()),
            "caps": {
                "aggregate": float(risk.max_aggregate_exposure_fraction) * equity,
                "per_event": float(risk.max_event_exposure_fraction) * equity,
                "per_strategy": float(risk.max_strategy_exposure_fraction) * equity,
                "per_correlation_group": float(risk.max_correlated_exposure_fraction) * equity,
                "per_trade": min(
                    float(risk.max_fraction_per_trade) * equity, float(risk.max_position_cost)
                ),
                "max_open_positions": risk.max_open_positions,
            },
            "note": "Exposure = maximum possible loss of open positions. Loss limits stop new "
            "entries; they do not cap losses on positions already open.",
        }

    @app.get("/api/research")
    def research() -> dict[str, Any]:
        from kalshi_trader.research.model_eval import (
            evaluate_predictions,
            one_market_per_event,
            predictions_from_db,
        )

        state = repo.all_state()
        with database.session() as s:
            fills = list(
                s.scalars(
                    select(SimulatedQuoteFillRow)
                    .order_by(SimulatedQuoteFillRow.ts.desc())
                    .limit(50)
                )
            )
        preds = one_market_per_event(predictions_from_db(database))
        return {
            "model_evaluation": {
                "data_source": "forward predictions recorded live, settled markets only",
                "benchmark": "market-implied probability (YES mid)",
                **evaluate_predictions(preds),
            },
            "market_making": {
                "data_kind": DATA_KINDS["research_sim"],
                "summary": json.loads(state["mm_summary"]) if state.get("mm_summary") else None,
                "recent_fills": [
                    {
                        "ts": f.ts,
                        "market": f.market_ticker,
                        "side": f.side,
                        "price": _f(f.price),
                        "contracts": _f(f.contracts),
                        "fee": _f(f.fee),
                        "mid_at_fill": f.mid_at_fill,
                        "mid_after": f.mid_after,
                    }
                    for f in fills
                ],
            },
        }

    calibration_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    @app.get("/api/research/calibration")
    def research_calibration() -> dict[str, Any]:
        """Fit and held-out check of the tennis model's probability calibration."""
        from kalshi_trader.research.calibration import calibration_report
        from kalshi_trader.research.model_eval import predictions_from_db

        cached = calibration_cache.get("report")
        if cached is not None and time.time() - cached[0] < CALIBRATION_CACHE_S:
            return cached[1]
        report = calibration_report(
            predictions_from_db(database), settings.strategies.tennis_model, settings.fees
        )
        report["computed_ts"] = time.time()
        calibration_cache["report"] = (time.time(), report)
        return report

    @app.get("/api/analytics")
    def analytics() -> dict[str, Any]:
        with database.session() as s:
            account = s.get(PaperAccountRow, MAIN_ACCOUNT_ID)
            snaps = list(
                s.scalars(
                    select(EquitySnapshotRow)
                    .where(EquitySnapshotRow.account_id == MAIN_ACCOUNT_ID)
                    .order_by(EquitySnapshotRow.ts)
                )
            )
        start = float(account.starting_balance) if account else settings.paper.starting_balance
        outcomes = outcomes_from_trades(closed_trades())
        mtm = [(e.ts, float(e.equity)) for e in snaps]
        if len(mtm) > 2000:  # thin the curve for the browser
            step = len(mtm) // 2000 + 1
            mtm = [*mtm[::step], mtm[-1]]
        cumulative, total = [], 0.0
        for o in sorted(outcomes, key=lambda o: o.closed_ts):
            total += o.pnl
            cumulative.append((o.closed_ts, total))
        return {
            "label": DATA_LABEL,
            "equity_curve": mtm,
            "drawdown": drawdown_series(mtm, float(start)),
            "cumulative_pnl": cumulative,
            "metrics": compute_metrics(outcomes, float(start)),
            "by_strategy": by_strategy(outcomes, float(start)),
            "win_rate_by_entry_price": win_rate_by_bucket(outcomes),
            "pnl_distribution": pnl_histogram(outcomes),
            "significance": significance_report(outcomes),
        }

    @app.get("/api/backtests")
    def backtests(limit: int = 20) -> dict[str, Any]:
        with database.session() as s:
            rows = s.scalars(
                select(BacktestRunRow)
                .order_by(BacktestRunRow.created_ts.desc())
                .limit(max(1, min(limit, 200)))
            )
            return {
                "runs": [
                    {
                        "id": r.id,
                        "created_ts": r.created_ts,
                        "kind": r.kind,
                        "data_source": r.data_source,
                        "params": json.loads(r.params_json),
                        "metrics": json.loads(r.metrics_json),
                        "notes": r.notes,
                    }
                    for r in rows
                ]
            }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        now = time.time()
        components = []
        for h in repo.all_health():
            details = json.loads(h.details_json) if h.details_json else {}
            components.append(
                {
                    "component": h.component,
                    "status": h.status,
                    "updated_ts": h.updated_ts,
                    "age_s": now - h.updated_ts,
                    "last_ok_ts": h.last_ok_ts,
                    "last_error_ts": h.last_error_ts,
                    "last_error": h.last_error,
                    "details": details,
                }
            )
        events = [
            {"ts": e.ts, "level": e.level, "component": e.component, "message": e.message}
            for e in repo.recent_events(150)
        ]
        with database.session() as s:
            open_count = (
                s.scalar(
                    select(func.count(PaperTradeRow.id)).where(
                        PaperTradeRow.status == "open",
                        PaperTradeRow.account_id == MAIN_ACCOUNT_ID,
                    )
                )
                or 0
            )
        return {
            "components": components,
            "events": events,
            "now": now,
            "counts": {**repo.snapshot_counts(), "open_trades": open_count},
        }

    @app.get("/api/export/trades.csv")
    def export_trades(ledger: str | None = None) -> StreamingResponse:
        account_id = ledger_account_id(ledger)
        buffer = io.StringIO()
        rows = [trade_to_dict(t) for t in closed_trades(account_id) + open_trades(account_id)]
        writer = csv.DictWriter(buffer, fieldnames=list(TRADE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
        buffer.seek(0)
        name = "paper_trades.csv" if account_id == MAIN_ACCOUNT_ID else f"shadow_{ledger}.csv"
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={name}"},
        )

    @app.get("/api/export/predictions.csv")
    def export_predictions() -> StreamingResponse:
        """Settled forward predictions with their outcomes (raw model, both markets)."""
        from kalshi_trader.research.model_eval import predictions_from_db

        preds = predictions_from_db(database)

        def chunks() -> Any:
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(PREDICTION_FIELDS)
            for start in range(0, len(preds), 2000):
                for p in preds[start : start + 2000]:
                    writer.writerow([getattr(p, f) for f in PREDICTION_FIELDS])
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate()
            yield buffer.getvalue()

        return StreamingResponse(
            chunks(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=model_predictions.csv"},
        )

    control_networks = [ipaddress.ip_network(str(n)) for n in settings.dashboard.control_networks]

    def control_allowed(client: str) -> bool:
        """This machine, or a configured network. Behind `tailscale serve` the client is the
        visitor's tailnet address, taken from X-Forwarded-For (trusted only from localhost)."""
        try:
            address = ipaddress.ip_address(client)
        except ValueError:
            return False
        return address.is_loopback or any(address in network for network in control_networks)

    @app.post("/api/control/strategy/{name}/{action}")
    def strategy_control(name: str, action: str, request: Request) -> JSONResponse:
        if not control_allowed(request.client.host if request.client else ""):
            raise HTTPException(status_code=403, detail="control not allowed from this address")
        if request.headers.get("x-kalshi-trader") != "1":
            raise HTTPException(status_code=403, detail="missing control header")
        if name not in STRATEGY_NAMES or action not in ("pause", "resume"):
            raise HTTPException(status_code=404, detail="unknown strategy or action")
        from kalshi_trader.core.registry import StrategyRegistry

        registry = StrategyRegistry(repo, settings.evidence)
        if not registry.set_paused(
            name, action == "pause", f"{action}d from dashboard", mode=_modes(settings)[name]
        ):
            raise HTTPException(status_code=409, detail="strategy not registered yet")
        repo.record_event("warning", "dashboard", f"operator action: {action} strategy {name}")
        return JSONResponse({"ok": True, "strategy": name, "action": action})

    @app.post("/api/control/{action}")
    def control(action: str, request: Request) -> JSONResponse:
        if not control_allowed(request.client.host if request.client else ""):
            raise HTTPException(status_code=403, detail="control not allowed from this address")
        if request.headers.get("x-kalshi-trader") != "1":
            raise HTTPException(status_code=403, detail="missing control header")
        if action == "pause":
            repo.set_flag(TRADING_ENABLED, False)
            repo.set_state(PAUSE_REASON, "paused from dashboard")
        elif action == "resume":
            repo.set_flag(TRADING_ENABLED, True)
            repo.set_state(PAUSE_REASON, "")
        elif action == "emergency_stop":
            repo.set_flag(EMERGENCY_STOP, True)
            repo.set_state(PAUSE_REASON, "EMERGENCY STOP from dashboard")
        elif action == "clear_emergency_stop":
            repo.set_flag(EMERGENCY_STOP, False)
            repo.set_state(PAUSE_REASON, "")
        else:
            raise HTTPException(status_code=404, detail="unknown action")
        repo.record_event("warning", "dashboard", f"operator action: {action}")
        return JSONResponse({"ok": True, "action": action})

    return app
