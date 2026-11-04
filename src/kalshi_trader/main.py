"""Command-line interface: ``kalshi-trader <command>`` (see ``--help``)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_trader.config import Settings, load_settings
from kalshi_trader.logging_setup import configure_logging, get_logger

log = get_logger("cli")


# ---------------------------------------------------------------- run/serve
def _uvicorn_options(settings: Settings) -> dict[str, Any]:
    # X-Forwarded-For is trusted only from this machine: that is where a local proxy such as
    # `tailscale serve` connects from, and it reports the real visitor's address there.
    return {
        "host": settings.dashboard.host,
        "port": settings.dashboard.port,
        "log_level": "warning",
        "access_log": False,
        "proxy_headers": True,
        "forwarded_allow_ips": "127.0.0.1,::1",
    }


def _embedded_server(settings: Settings) -> Any:
    import uvicorn

    from kalshi_trader.dashboard.app import create_app

    class EmbeddedServer(uvicorn.Server):
        """uvicorn server that leaves signal handling to the engine."""

        @contextlib.contextmanager
        def capture_signals(self):  # type: ignore[override]
            yield

    return EmbeddedServer(uvicorn.Config(create_app(settings), **_uvicorn_options(settings)))


async def _run(settings: Settings, record_only: bool, dashboard: bool) -> None:
    from kalshi_trader.engine import TradingEngine

    engine = TradingEngine(settings, record_only=record_only)
    server = _embedded_server(settings) if dashboard else None
    server_task = asyncio.create_task(server.serve()) if server else None
    if server:
        log.info(
            "dashboard_listening", url=f"http://{settings.dashboard.host}:{settings.dashboard.port}"
        )
    try:
        await engine.run()
    finally:
        if server and server_task:
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server_task, timeout=10)


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    asyncio.run(_run(settings, record_only=args.record_only, dashboard=not args.no_dashboard))
    return 0


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from kalshi_trader.dashboard.app import create_app

    print(f"Dashboard: http://{settings.dashboard.host}:{settings.dashboard.port}")
    uvicorn.run(create_app(settings), **_uvicorn_options(settings))
    return 0


# --------------------------------------------------------------------- scan
async def _scan(settings: Settings, live_only: bool) -> list[dict[str, Any]]:
    from kalshi_trader.api.rest import KalshiRestClient
    from kalshi_trader.database.db import Database
    from kalshi_trader.database.repository import Repository
    from kalshi_trader.engine import load_signer
    from kalshi_trader.market_data.discovery import TennisDiscovery
    from kalshi_trader.market_data.prematch import pre_match_mid_from_candles
    from kalshi_trader.models.estimator import WinProbabilityEstimator
    from kalshi_trader.tennis.providers.kalshi_live import KalshiLiveDataProvider

    repo = Repository(Database(":memory:"))  # one-shot scan: no persistence side effects
    async with KalshiRestClient(settings.kalshi, load_signer(settings)) as rest:
        result = await TennisDiscovery(rest, repo, settings.discovery).refresh()
        books = await rest.get_orderbooks(list(result.markets))
        refs = [r for m in result.matches.values() if (r := m.match_ref())]
        scores = await KalshiLiveDataProvider(rest).fetch(refs) if refs else {}
        pre_match: dict[str, float] = {}  # event -> P(player 1) from pre-start candles
        for match in result.matches.values():
            score = scores.get(match.milestone_id or "")
            if not (score and score.state.is_live and match.scheduled_start_ts):
                continue
            values = []
            for index, ticker in enumerate(match.market_tickers):
                if not ticker:
                    continue
                found = await pre_match_mid_from_candles(
                    rest, match.series_ticker, ticker, match.scheduled_start_ts
                )
                if found is not None:
                    values.append(found[0] if index == 0 else 1 - found[0])
            if values:
                pre_match[match.event_ticker] = sum(values) / len(values)
    estimator = WinProbabilityEstimator(settings.model)
    rows = []
    for match in sorted(result.matches.values(), key=lambda m: m.scheduled_start_ts or 0):
        score = scores.get(match.milestone_id or "")
        state = score.state if score else None
        if live_only and not (state and state.is_live):
            continue
        estimate = None
        if state and state.is_live and state.score:
            prior = estimator.build_prior(
                match.tour_key, state.fmt, pre_match.get(match.event_ticker)
            )
            estimate = estimator.estimate(
                state.score, prior, state.serve_stats, f"scan:{match.event_ticker}"
            )
        for index, ticker in enumerate(match.market_tickers):
            if not ticker or ticker not in result.markets:
                continue
            ob = books.get(ticker)
            yes_bid = ob.yes_dollars[-1][0] if ob and ob.yes_dollars else None
            yes_ask = (Decimal(1) - ob.no_dollars[-1][0]) if ob and ob.no_dollars else None
            prob = estimate.for_player(index) if estimate else None
            rows.append(
                {
                    "match": match.title,
                    "competition": match.competition or match.tour_key,
                    "player": match.player_names[index],
                    "ticker": ticker,
                    "status": state.status.value if state else "?",
                    "score": state.score_text() if state else "",
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "model": prob.point if prob else None,
                    "robust_low": prob.robust_low if prob else None,
                    "prior": (
                        ("pre-match" if match.event_ticker in pre_match else "tour avg")
                        if estimate
                        else ""
                    ),
                }
            )
    return rows


def cmd_scan(settings: Settings, args: argparse.Namespace) -> int:
    rows = asyncio.run(_scan(settings, args.live_only))
    if args.json:
        print(json.dumps(rows, default=str, indent=1))
        return 0
    fmt = "{:<34} {:<22} {:<10} {:<28} {:>6} {:>6} {:>7} {:>7}  {}"
    print(
        "REAL Kalshi prices and scores | model/robust = statistical estimates | prior = "
        "pre-match market price when found, else tour average (wider uncertainty)"
    )
    print(
        fmt.format(
            "player (YES)",
            "competition",
            "status",
            "score",
            "bid",
            "ask",
            "model",
            "robust",
            "prior",
        )
    )
    for r in rows:
        print(
            fmt.format(
                r["player"][:34],
                str(r["competition"])[:22],
                r["status"][:10],
                r["score"][:28],
                str(r["yes_bid"] or "-"),
                str(r["yes_ask"] or "-"),
                f"{r['model']:.3f}" if r["model"] is not None else "-",
                f"{r['robust_low']:.3f}" if r["robust_low"] is not None else "-",
                r["prior"],
            )
        )
    print(f"\n{len(rows)} markets")
    return 0


# ------------------------------------------------------------ status/control
def _repo(settings: Settings) -> Any:
    from kalshi_trader.database.db import Database
    from kalshi_trader.database.repository import Repository

    return Repository(Database(settings.db_path))


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    from kalshi_trader.config import FeeSettings
    from kalshi_trader.execution.paper import PaperBroker

    repo = _repo(settings)
    broker = PaperBroker(repo.db, settings.paper, FeeSettings())
    account = broker.account()
    state = repo.all_state()
    print("PAPER ACCOUNT (simulated)")
    print(f"  starting balance : ${account.starting_balance}")
    print(f"  cash             : ${account.cash}")
    print(f"  realised P&L     : ${account.realized_pnl}")
    print(f"  fees paid        : ${account.fees_paid}")
    print(f"  open positions   : {len(broker.open_trades())}")
    print("SAFETY STATE")
    print(f"  trading enabled  : {state.get('trading_enabled', 'true')}")
    print(f"  emergency stop   : {state.get('emergency_stop', 'false')}")
    print(f"  pause reason     : {state.get('pause_reason') or '-'}")
    print("COMPONENTS")
    now = time.time()
    for h in repo.all_health():
        print(
            f"  {h.component:<14} {h.status:<12} updated {now - h.updated_ts:7.0f}s ago"
            + (f"  last error: {h.last_error[:80]}" if h.last_error else "")
        )
    print("STRATEGIES (mode from config, status from evidence/operator)")
    from kalshi_trader.core.registry import StrategyRegistry

    registry = StrategyRegistry(repo, settings.evidence)
    registry.reload()
    for name, mode in _strategy_modes(settings).items():
        row = registry.state(name)
        status = row.status if row else "not yet registered (start the engine)"
        reason = f" - {row.status_reason}" if row and row.status_reason else ""
        print(f"  {name:<16} mode={mode:<8} status={status}{reason}"[:160])
    print("DATA", repo.snapshot_counts())
    return 0


def _strategy_modes(settings: Settings) -> dict[str, str]:
    cfg = settings.strategies
    return {
        "tennis_model": cfg.tennis_model.mode,
        "cross_market": cfg.cross_market.mode,
        "late_match": cfg.late_match.mode,
        "price_threshold": cfg.price_threshold.mode,
        "market_making": cfg.market_making.mode,
    }


def cmd_strategies(settings: Settings, args: argparse.Namespace) -> int:
    from kalshi_trader.core.registry import StrategyRegistry

    repo = _repo(settings)
    registry = StrategyRegistry(repo, settings.evidence)
    registry.reload()
    modes = _strategy_modes(settings)
    if args.action in ("pause", "resume"):
        if not args.name or args.name not in modes:
            print(f"usage: kalshi-trader strategies {args.action} <{'|'.join(modes)}>")
            return 1
        ok = registry.set_paused(
            args.name, args.action == "pause", f"{args.action}d from CLI", mode=modes[args.name]
        )
        if not ok:
            print(f"{args.name} is not registered yet; start the engine once first")
            return 1
        repo.record_event("warning", "cli", f"strategy {args.name} {args.action}d")
        print(f"ok: {args.name} {args.action}d (persists across restarts)")
        return 0
    rows = []
    for name, mode in modes.items():
        row = registry.state(name)
        evidence = json.loads(row.evidence_json) if row and row.evidence_json else {}
        rows.append(
            {
                "strategy": name,
                "mode": mode,
                "status": row.status if row else "unregistered",
                "reason": row.status_reason if row else "",
                "version": f"{row.version}+{row.config_hash[:8]}" if row else "",
                "evidence_trades": evidence.get("trades"),
                "validated": evidence.get("validated"),
            }
        )
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    print("Modes come from config (strategies.<name>.mode = off|observe|paper).")
    for r in rows:
        print(
            f"{r['strategy']:<16} mode={r['mode']:<8} status={r['status']:<16} "
            f"trades={r['evidence_trades']} validated={r['validated']}  {r['reason']}"[:200]
        )
    return 0


def cmd_diagnostics(settings: Settings, args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from kalshi_trader.database.tables import DiagnosticCountRow

    repo = _repo(settings)
    with repo.db.session() as s:
        stmt = select(DiagnosticCountRow).order_by(DiagnosticCountRow.day.desc())
        rows = [r for r in s.scalars(stmt) if not args.day or r.day == args.day]
    if not rows:
        print("no diagnostics recorded yet (the engine writes them while running)")
        return 0
    day = args.day or rows[0].day
    print(f"Opportunity outcomes on {day} (UTC). Reasons are normalised (numbers -> #).")
    by_strategy: dict[str, list[Any]] = {}
    for r in rows:
        if r.day == day:
            by_strategy.setdefault(r.strategy, []).append(r)
    for strategy, items in sorted(by_strategy.items()):
        totals: dict[str, int] = {}
        for r in items:
            totals[r.status] = totals.get(r.status, 0) + r.count
        print(f"\n{strategy}: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
        for r in sorted(items, key=lambda r: -r.count)[: args.top]:
            print(f"  {r.count:>8}  {r.status:<11} {r.reason}"[:200])
    return 0


def cmd_research(settings: Settings, args: argparse.Namespace) -> int:
    from kalshi_trader.database.db import Database

    db = Database(settings.db_path)
    if args.kind == "calibrate":
        from kalshi_trader.research.calibration import calibration_report
        from kalshi_trader.research.model_eval import predictions_from_db

        if args.source == "replay":
            from kalshi_trader.backtesting.replay import replay

            preds = replay(settings, db, strategies=[]).predictions
        else:
            preds = predictions_from_db(db)
        report = calibration_report(
            preds,
            settings.strategies.tennis_model,
            settings.fees,
            test_fraction=args.test_fraction,
            embargo_s=args.embargo_hours * 3600,
        )
        print(json.dumps(report, indent=1, default=str))
        toml = (report.get("recommendation") or {}).get("toml")
        print("\n" + (toml or "No calibration recommended: " + report["recommendation"]["reason"]))
        return 0
    if args.kind == "model-eval":
        from kalshi_trader.research.model_eval import (
            evaluate_predictions,
            one_market_per_event,
            predictions_from_db,
        )

        if args.source == "live":
            preds = predictions_from_db(db)
            source = "forward predictions recorded live (model_predictions table)"
        else:
            from kalshi_trader.backtesting.replay import replay

            preds = replay(settings, db, strategies=[]).predictions
            source = "replay of recorded observations"
        report = {
            "data_source": source,
            "benchmark": "market-implied probability (YES mid) = Baseline 3",
            **evaluate_predictions(one_market_per_event(preds)),
        }
    else:
        from kalshi_trader.research.sensitivity import run_sensitivity

        report = {"data_source": "replay of recorded observations", **run_sensitivity(settings, db)}
    print(json.dumps(report, indent=1, default=str))
    return 0


def cmd_trading(settings: Settings, args: argparse.Namespace) -> int:
    from kalshi_trader.risk.manager import RiskManager

    repo = _repo(settings)
    risk = RiskManager(settings.risk, repo, settings.fees.taker_rate)
    if args.action == "pause":
        risk.pause("paused from CLI")
    elif args.action == "resume":
        if risk.emergency_stopped():
            print("Emergency stop is active; clear it first with: kalshi-trader trading clear-stop")
            return 1
        risk.resume()
    elif args.action == "stop":
        risk.emergency_stop("emergency stop from CLI")
    elif args.action == "clear-stop":
        risk.clear_emergency_stop()
    print(f"ok: {args.action}")
    return 0


async def _check_auth(settings: Settings) -> int:
    from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient
    from kalshi_trader.engine import load_signer

    signer = load_signer(settings)
    if signer is None:
        print(
            "No usable credentials (set KT_KALSHI__API_KEY_ID and KT_KALSHI__PRIVATE_KEY_PATH)."
            " Public market data works without them; only the WebSocket feed needs them."
        )
        return 1
    async with KalshiRestClient(settings.kalshi, signer) as rest:
        try:
            await rest.get_balance()
            print(f"Authentication OK ({signer.key_type} key, {settings.kalshi.environment}).")
            keys = await rest.get_api_keys()
        except KalshiAPIError as exc:
            print(f"Authentication failed: {exc}")
            return 1
    key_id = settings.kalshi.api_key_id.get_secret_value() if settings.kalshi.api_key_id else ""
    for key in keys:
        if key.get("api_key_id") == key_id:
            scopes = key.get("scopes") or []
            print(f"Key scopes: {scopes}")
            if any(str(s).startswith("write") for s in scopes):
                print(
                    "WARNING: this key can trade. This application never uses write access; "
                    "a key with only the 'read' scope is recommended."
                )
    return 0


def cmd_check_auth(settings: Settings, args: argparse.Namespace) -> int:
    return asyncio.run(_check_auth(settings))


# ----------------------------------------------------------- backtests/export
def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    if args.kind == "historical":
        from kalshi_trader.backtesting.historical import run_historical_backtest

        report = asyncio.run(
            run_historical_backtest(
                settings,
                days=args.days,
                series=args.series or None,
                min_price=Decimal(str(args.min_price)),
                max_price=Decimal(str(args.max_price)),
                max_events=args.max_events,
                test_fraction=args.test_fraction,
                cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            )
        )
    else:
        from kalshi_trader.backtesting.replay import run_replay_backtest

        report = run_replay_backtest(
            settings,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            grid=args.grid,
            embargo_s=args.embargo_hours * 3600,
            evaluate_test=not args.no_test,
            market_making=args.market_making,
        )
    print(json.dumps(report, indent=1, default=str))
    return 0


def cmd_export(settings: Settings, args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from kalshi_trader.dashboard.app import TRADE_FIELDS, trade_to_dict
    from kalshi_trader.database.tables import PaperTradeRow

    repo = _repo(settings)
    with repo.db.session() as s:
        trades = list(s.scalars(select(PaperTradeRow).order_by(PaperTradeRow.opened_ts)))
    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRADE_FIELDS))
        writer.writeheader()
        writer.writerows(trade_to_dict(t) for t in trades)
    print(f"wrote {len(trades)} simulated trades to {out}")
    return 0


def cmd_reset_account(settings: Settings, args: argparse.Namespace) -> int:
    if not args.yes:
        print(
            "This deletes all simulated orders, trades and equity history and restores the "
            "starting balance. Re-run with --yes to confirm."
        )
        return 1
    from sqlalchemy import delete

    from kalshi_trader.database.tables import (
        EquitySnapshotRow,
        PaperAccountRow,
        PaperOrderRow,
        PaperTradeRow,
        SignalRow,
    )
    from kalshi_trader.execution.paper import PaperBroker

    repo = _repo(settings)
    with repo.db.session() as s:
        for table in (PaperOrderRow, PaperTradeRow, EquitySnapshotRow, SignalRow, PaperAccountRow):
            s.execute(delete(table))
    PaperBroker(repo.db, settings.paper, settings.fees)
    repo.record_event("warning", "cli", "paper account reset")
    print(f"paper account reset to ${settings.paper.starting_balance}")
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kalshi-trader",
        description="Kalshi multi-strategy research and PAPER trading (never places real orders).",
    )
    parser.add_argument("--log-level", default=None, help="override KT_LOG_LEVEL")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the engine (data recording + paper trading + dashboard)")
    p.add_argument(
        "--record-only",
        action="store_true",
        help="record data and evaluate strategies, but never simulate any trade (not even "
        "shadow ledgers)",
    )
    p.add_argument("--no-dashboard", action="store_true", help="do not serve the dashboard")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("dashboard", help="serve the dashboard only (reads the database)")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("scan", help="one-shot scan of live/upcoming tennis markets")
    p.add_argument("--live-only", action="store_true", help="only matches in progress")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("status", help="show paper account, safety state and component health")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("trading", help="pause/resume paper trading or toggle the emergency stop")
    p.add_argument("action", choices=["pause", "resume", "stop", "clear-stop"])
    p.set_defaults(func=cmd_trading)

    p = sub.add_parser("check-auth", help="verify Kalshi API credentials (read-only call)")
    p.set_defaults(func=cmd_check_auth)

    p = sub.add_parser("backtest", help="evaluate strategies on historical/recorded data")
    p.add_argument("kind", choices=["historical", "replay"])
    p.add_argument("--days", type=int, default=30, help="historical: look-back window in days")
    p.add_argument("--series", nargs="*", help="historical: series tickers (default: config)")
    p.add_argument("--min-price", type=float, default=0.90, help="historical: band low (ask)")
    p.add_argument("--max-price", type=float, default=0.97, help="historical: band high (ask)")
    p.add_argument(
        "--max-events", type=int, default=300, help="historical: cap on matches downloaded"
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.3, help="chronological out-of-sample share"
    )
    p.add_argument("--cache-dir", default=None, help="historical: cache downloaded candles here")
    p.add_argument("--grid", action="store_true", help="replay: sweep a parameter grid")
    p.add_argument(
        "--validation-fraction", type=float, default=0.2, help="replay: validation share"
    )
    p.add_argument(
        "--embargo-hours",
        type=float,
        default=1.0,
        help="replay: drop events first seen within this long after a split boundary",
    )
    p.add_argument(
        "--no-test",
        action="store_true",
        help="replay: evaluate train/validation only (keep the test split untouched)",
    )
    p.add_argument(
        "--market-making", action="store_true", help="replay: include the MM research sim"
    )
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("strategies", help="list strategies, or pause/resume one")
    p.add_argument("action", nargs="?", default="list", choices=["list", "pause", "resume"])
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser("diagnostics", help="why strategies are (not) trading, by reason")
    p.add_argument("--day", default=None, help="UTC day YYYY-MM-DD (default: latest)")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser(
        "research", help="model evaluation, probability calibration and sensitivity analysis"
    )
    p.add_argument("kind", choices=["model-eval", "calibrate", "sensitivity"])
    p.add_argument(
        "--source",
        choices=["live", "replay"],
        default=None,
        help="forward predictions recorded live, or a replay of recorded observations "
        "(default: replay for model-eval, live for calibrate)",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.3, help="calibrate: held-out share of events"
    )
    p.add_argument(
        "--embargo-hours",
        type=float,
        default=1.0,
        help="calibrate: drop events first seen within this long after the split boundary",
    )
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("export", help="export simulated trades to CSV")
    p.add_argument("--out", default="paper_trades.csv")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("reset-account", help="reset the paper account (destructive)")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_reset_account)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    configure_logging(
        args.log_level or settings.log_level, settings.log_dir if args.command == "run" else None
    )
    return int(args.func(settings, args) or 0)


if __name__ == "__main__":
    sys.exit(main())
