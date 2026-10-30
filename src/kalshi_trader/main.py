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
def _embedded_server(settings: Settings) -> Any:
    import uvicorn

    from kalshi_trader.dashboard.app import create_app

    class EmbeddedServer(uvicorn.Server):
        """uvicorn server that leaves signal handling to the engine."""

        @contextlib.contextmanager
        def capture_signals(self):  # type: ignore[override]
            yield

    config = uvicorn.Config(
        create_app(settings),
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="warning",
        access_log=False,
    )
    return EmbeddedServer(config)


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
    uvicorn.run(
        create_app(settings),
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="warning",
        access_log=False,
    )
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
    print("DATA", repo.snapshot_counts())
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

        report = run_replay_backtest(settings, test_fraction=args.test_fraction, grid=args.grid)
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
        description="Kalshi tennis market research and PAPER trading (never places real orders).",
    )
    parser.add_argument("--log-level", default=None, help="override KT_LOG_LEVEL")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run the engine (data recording + paper trading + dashboard)")
    p.add_argument(
        "--record-only",
        action="store_true",
        help="record data and show the dashboard, but never simulate trades",
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
    p.set_defaults(func=cmd_backtest)

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
