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
    EquitySnapshotRow,
    PaperAccountRow,
    PaperTradeRow,
)

STATIC_DIR = Path(__file__).parent / "static"
ENGINE_STALE_S = 15.0
DATA_LABEL = (
    "SIMULATED paper trading on real, live Kalshi order books. "
    "No real orders are placed. Model probabilities are estimates."
)


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
    return TradeOutcome(
        opened_ts=t.opened_ts,
        closed_ts=t.closed_ts or t.opened_ts,
        strategy=t.strategy,
        entry_price=float(t.entry_price),
        contracts=float(t.entry_contracts),
        cost=float(t.entry_cost + t.entry_fee),
        pnl=float(t.realized_pnl or 0),
        fees=float(t.entry_fee + t.exit_fee),
        model_prob=t.robust_prob,
        event=t.event_ticker,
    )


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

    def closed_trades() -> list[PaperTradeRow]:
        with database.session() as s:
            stmt = select(PaperTradeRow).where(PaperTradeRow.status.in_(["settled", "closed"]))
            return list(s.scalars(stmt.order_by(PaperTradeRow.closed_ts)))

    def open_trades() -> list[PaperTradeRow]:
        with database.session() as s:
            stmt = select(PaperTradeRow).where(PaperTradeRow.status == "open")
            return list(s.scalars(stmt.order_by(PaperTradeRow.opened_ts)))

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
        with database.session() as s:
            account = s.get(PaperAccountRow, 1)
            last_equity = s.scalars(
                select(EquitySnapshotRow).order_by(EquitySnapshotRow.ts.desc()).limit(1)
            ).first()
            day_start = time.time() - (time.time() % 86400)
            first_today = s.scalars(
                select(EquitySnapshotRow)
                .where(EquitySnapshotRow.ts >= day_start)
                .order_by(EquitySnapshotRow.ts)
                .limit(1)
            ).first()
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
    def trades(limit: int = 500) -> dict[str, Any]:
        rows = [trade_to_dict(t) for t in closed_trades()][-max(1, min(limit, 5000)) :]
        return {"trades": list(reversed(rows))}

    @app.get("/api/signals")
    def signals(limit: int = 100) -> dict[str, Any]:
        out = []
        for s in repo.recent_signals(max(1, min(limit, 1000))):
            out.append(
                {
                    "ts": s.created_ts,
                    "strategy": s.strategy,
                    "market": s.market_ticker,
                    "side": s.side,
                    "ask": _f(s.ask_price),
                    "model_prob": s.model_prob,
                    "robust_prob": s.robust_prob,
                    "market_prob": s.market_prob,
                    "ev": s.ev_per_contract,
                    "approved": s.approved,
                    "reasons": json.loads(s.reasons_json or "[]"),
                    "score": s.score_text,
                }
            )
        return {"signals": out}

    @app.get("/api/analytics")
    def analytics() -> dict[str, Any]:
        with database.session() as s:
            account = s.get(PaperAccountRow, 1)
            snaps = list(s.scalars(select(EquitySnapshotRow).order_by(EquitySnapshotRow.ts)))
        start = float(account.starting_balance) if account else settings.paper.starting_balance
        outcomes = [outcome_from_row(t) for t in closed_trades()]
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
                s.scalar(select(func.count(PaperTradeRow.id)).where(PaperTradeRow.status == "open"))
                or 0
            )
        return {
            "components": components,
            "events": events,
            "now": now,
            "counts": {**repo.snapshot_counts(), "open_trades": open_count},
        }

    @app.get("/api/export/trades.csv")
    def export_trades() -> StreamingResponse:
        buffer = io.StringIO()
        rows = [trade_to_dict(t) for t in closed_trades() + open_trades()]
        writer = csv.DictWriter(buffer, fieldnames=list(TRADE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=paper_trades.csv"},
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
