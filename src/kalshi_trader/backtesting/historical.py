"""Historical, price-only backtest on real Kalshi 1-minute candlesticks.

What it tests: "buy the favourite once its executable YES ask enters a price band"
(no score information - Kalshi does not publish historical scores). It answers whether
high-priced tennis contracts have historically been *underpriced after fees*, i.e. the
favourite-longshot side of the motivating hypothesis.

Look-ahead protection:
* decisions use only candles that have closed (``end_period_ts <= t``);
* the fill uses the *next* candle's opening ask (one-minute latency), and is skipped if
  that ask is outside the band + slippage;
* settlement uses the market's final result only to compute P&L;
* the in-play window starts at the milestone ``start_date`` and ends at market close;
* at most one trade per match (event), the first qualifying minute after the match start.

Limitations (reported with every run): minute granularity, no depth data (a fixed small
size is assumed fillable at the best ask), no score context, survivorship of markets that
Kalshi lists as binary match-winner markets in the configured series.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient
from kalshi_trader.api.schemas import Candlestick, Market
from kalshi_trader.backtesting.metrics import (
    TradeOutcome,
    compute_metrics,
    win_rate_by_bucket,
)
from kalshi_trader.backtesting.stats import significance_report
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import BacktestRunRow
from kalshi_trader.execution.fees import buy_cash_debit, order_trade_fee
from kalshi_trader.logging_setup import get_logger

log = get_logger("backtest.historical")

DATA_SOURCE = "kalshi_historical_candlesticks_1m (real prices, no score data)"
BANDS = [
    (Decimal("0.80"), Decimal("0.85")),
    (Decimal("0.85"), Decimal("0.90")),
    (Decimal("0.90"), Decimal("0.93")),
    (Decimal("0.93"), Decimal("0.95")),
    (Decimal("0.95"), Decimal("0.97")),
    (Decimal("0.97"), Decimal("0.99")),
]


@dataclass(frozen=True)
class MarketHistory:
    market: Market
    candles: list[Candlestick]
    series: str
    start_ts: float  # match start from the Kalshi milestone (never its end/outcome time)


def _settled_ts(m: Market) -> float | None:
    for dt in (m.settlement_ts, m.close_time):
        if dt is not None:
            return dt.timestamp()
    return None


def _trading_end_ts(m: Market) -> float | None:
    """When trading stopped: tennis markets close as soon as a winner is declared, which can
    be minutes before settlement. Nothing after this instant is a fillable price."""
    for dt in (m.close_time, m.settlement_ts):
        if dt is not None:
            return dt.timestamp()
    return None


async def fetch_histories(
    rest: KalshiRestClient,
    series_list: list[str],
    days: int,
    max_events: int,
    cache_dir: Path | None,
) -> tuple[list[MarketHistory], dict[str, int]]:
    now = datetime.now(UTC)
    since = (now - timedelta(days=days)).timestamp()
    cutoff = await rest.get_historical_cutoff()
    cutoff_ts = cutoff.market_settled_ts.timestamp() if cutoff.market_settled_ts else 0.0
    candidates: list[tuple[Market, str, bool]] = []
    for series in series_list:
        try:
            live = await rest.get_markets(
                series_ticker=series, status="settled", min_settled_ts=int(since)
            )
        except KalshiAPIError as exc:
            log.warning("settled_markets_failed", series=series, error=str(exc)[:200])
            live = []
        archived: list[Market] = []
        if since < cutoff_ts:
            try:
                archived = await rest.get_historical_markets(series_ticker=series)
            except KalshiAPIError as exc:
                log.warning("historical_markets_failed", series=series, error=str(exc)[:200])
        for m, is_hist in [(m, False) for m in live] + [(m, True) for m in archived]:
            ts = _settled_ts(m)
            if m.market_type == "binary" and m.result in ("yes", "no") and ts and ts >= since:
                candidates.append((m, series, is_hist))
    # Keep both sides of each match together. If capped, take an evenly spaced (not
    # "most recent") subset of events.
    events: dict[str, list[tuple[Market, str, bool]]] = {}
    for m, s, h in {m.ticker: (m, s, h) for m, s, h in candidates}.values():
        events.setdefault(m.event_ticker, []).append((m, s, h))
    ordered = sorted(events.items(), key=lambda kv: min(_settled_ts(m) or 0 for m, _, _ in kv[1]))
    if len(ordered) > max_events:
        step = len(ordered) / max_events
        ordered = [ordered[int(i * step)] for i in range(max_events)]
    stats = {
        "settled_events_in_window": len(events),
        "events_selected": len(ordered),
        "events_without_start_time": 0,
        "markets_without_candles": 0,
    }
    histories = []
    for i, (event, group) in enumerate(ordered):
        start_ts = await _match_start(rest, event, cache_dir)
        if start_ts is None:
            stats["events_without_start_time"] += 1
            continue
        for m, series, is_hist in group:
            historical = is_hist or (_settled_ts(m) or 0) < cutoff_ts
            candles = await _candles(rest, m, series, start_ts, historical, cache_dir)
            if candles:
                histories.append(MarketHistory(m, candles, series, start_ts))
            else:
                stats["markets_without_candles"] += 1
        if (i + 1) % 50 == 0:
            log.info("historical_download_progress", done=i + 1, total=len(ordered))
    return histories, stats


async def _match_start(rest: KalshiRestClient, event: str, cache_dir: Path | None) -> float | None:
    """Match start time from the event's tennis milestone.

    Market ``occurrence_datetime`` is not usable: for tennis it is the *expected expiration*
    (an estimate of when the match ends). Only the milestone's ``start_date`` is used; its
    ``end_date`` is the outcome time and must never drive a decision.
    """
    cache = cache_dir / f"{event}.start.json" if cache_dir else None
    if cache and cache.is_file():
        return json.loads(cache.read_text())["start_ts"]
    try:
        milestones = await rest.get_milestones(related_event_ticker=event)
    except KalshiAPIError as exc:
        log.warning("milestone_failed", event=event, error=str(exc)[:200])
        return None
    starts = [
        m.start_date.timestamp()
        for m in milestones
        if m.start_date is not None and m.type.startswith("tennis")
    ]
    start_ts = min(starts) if starts else None
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"start_ts": start_ts}))
    return start_ts


async def _candles(
    rest: KalshiRestClient,
    m: Market,
    series: str,
    start_ts: float,
    historical: bool,
    cache_dir: Path | None,
) -> list[Candlestick]:
    cache = cache_dir / f"{m.ticker}.json" if cache_dir else None
    if cache and cache.is_file():
        return [Candlestick.model_validate(c) for c in json.loads(cache.read_text())]
    end = _settled_ts(m)
    start = int(start_ts - 3600)
    if end is None or int(end) + 60 <= start:
        return []
    try:
        candles = await rest.get_candlesticks(
            series, m.ticker, start, int(end) + 60, 1, historical=historical
        )
    except KalshiAPIError as exc:
        log.warning("candles_failed", ticker=m.ticker, error=str(exc)[:200])
        return []
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([c.model_dump(mode="json") for c in candles]))
    return candles


def simulate_band(
    histories: list[MarketHistory],
    lo: Decimal,
    hi: Decimal,
    *,
    fee_rate: Decimal,
    contracts: Decimal = Decimal("10"),
    slippage: Decimal = Decimal("0.00"),
    max_spread: Decimal = Decimal("0.03"),
) -> list[TradeOutcome]:
    """At most one entry per event: first minute (after scheduled start) whose closed
    candle shows a YES ask inside [lo, hi]; filled at the next candle's opening ask."""
    by_event: dict[str, list[MarketHistory]] = {}
    for h in histories:
        by_event.setdefault(h.market.event_ticker, []).append(h)
    trades: list[TradeOutcome] = []
    for event, group in by_event.items():
        best: tuple[int, MarketHistory, Decimal] | None = None
        for h in group:
            fill = _first_fill(h, lo, hi, slippage, max_spread)
            if fill is not None and (best is None or fill[0] < best[0]):
                best = (fill[0], h, fill[1])
        if best is None:
            continue
        ts, h, price = best
        fee = order_trade_fee([(price, contracts)], fee_rate)
        debit, eff_fee = buy_cash_debit(price * contracts, fee)
        payout = contracts if h.market.result == "yes" else Decimal(0)
        trades.append(
            TradeOutcome(
                opened_ts=float(ts),
                closed_ts=_settled_ts(h.market) or float(ts),
                strategy=f"price_band_{lo}_{hi}",
                entry_price=float(price),
                contracts=float(contracts),
                cost=float(debit),
                pnl=float(payout - debit),
                fees=float(eff_fee),
                event=event,
            )
        )
    return trades


def _first_fill(
    h: MarketHistory, lo: Decimal, hi: Decimal, slippage: Decimal, max_spread: Decimal
) -> tuple[int, Decimal] | None:
    candles = sorted(h.candles, key=lambda c: c.end_period_ts)
    trading_end = _trading_end_ts(h.market) or float("inf")
    for i, c in enumerate(candles[:-1]):
        if c.end_period_ts < h.start_ts or c.end_period_ts >= trading_end:
            continue
        ask = c.yes_ask.close_dollars if c.yes_ask else None
        bid = c.yes_bid.close_dollars if c.yes_bid else None
        if ask is None or bid is None or ask >= Decimal("0.999") or not (lo <= ask <= hi):
            continue
        if ask - bid > max_spread:
            continue
        nxt = candles[i + 1]
        fill = (nxt.yes_ask.open_dollars or nxt.yes_ask.close_dollars) if nxt.yes_ask else None
        if fill is None or fill > hi + slippage or fill < lo or fill >= Decimal("0.999"):
            continue  # price moved away during the one-minute latency: no fill
        if nxt.end_period_ts >= trading_end:
            return None
        return nxt.end_period_ts - 60, fill
    return None


def _split(
    trades: list[TradeOutcome], test_fraction: float, boundary: float
) -> tuple[list[TradeOutcome], list[TradeOutcome]]:
    train = [t for t in trades if t.opened_ts < boundary]
    test = [t for t in trades if t.opened_ts >= boundary]
    return train, test


async def run_historical_backtest(
    settings: Settings,
    *,
    days: int,
    series: list[str] | None,
    min_price: Decimal,
    max_price: Decimal,
    max_events: int,
    test_fraction: float,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    series_list = series or settings.discovery.series_tickers
    started = time.time()
    async with KalshiRestClient(settings.kalshi) as rest:
        histories, coverage = await fetch_histories(rest, series_list, days, max_events, cache_dir)
    if not histories:
        return {
            "data_source": DATA_SOURCE,
            "error": "no settled markets with candles found",
            "coverage": coverage,
        }
    starts = sorted(h.start_ts for h in histories)
    boundary = (
        starts[int(len(starts) * (1 - test_fraction))] if 0 < test_fraction < 1 else float("inf")
    )
    fee_rate = settings.fees.taker_rate
    start_balance = float(settings.paper.starting_balance)

    requested = simulate_band(histories, min_price, max_price, fee_rate=fee_rate)
    train, test = _split(requested, test_fraction, boundary)

    sweep = []
    for lo, hi in BANDS:
        band_trades = simulate_band(histories, lo, hi, fee_rate=fee_rate)
        tr, te = _split(band_trades, test_fraction, boundary)
        sweep.append(
            {
                "band": f"{lo}-{hi}",
                "train": compute_metrics(tr, start_balance),
                "test": compute_metrics(te, start_balance),
                "test_significance": significance_report(te),
            }
        )
    eligible = [s for s in sweep if s["train"].get("trades", 0) >= 20]
    chosen = max(eligible, key=lambda s: s["train"]["total_pnl"]) if eligible else None

    report = {
        "data_source": DATA_SOURCE,
        "markets_with_candles": len(histories),
        "events": len({h.market.event_ticker for h in histories}),
        "coverage": coverage,
        "window_days": days,
        "series": series_list,
        "assumptions": {
            "contracts_per_trade": 10,
            "fill": "next 1-min candle opening YES ask",
            "fees": f"taker {fee_rate} * C * P * (1-P), rounded up to the cent",
            "entries": "one per match, first qualifying minute after the match start",
            "match_start": "Kalshi milestone start_date (market occurrence_datetime is the "
            "expected expiration, not the start)",
            "trading_end": "market close_time (closes when a winner is declared)",
            "max_spread": "0.03",
        },
        "out_of_sample_split": {
            "test_fraction": test_fraction,
            "boundary": datetime.fromtimestamp(boundary, UTC).isoformat()
            if boundary != float("inf")
            else None,
        },
        "requested_band": {
            "band": f"{min_price}-{max_price}",
            "all": compute_metrics(requested, start_balance),
            "train": compute_metrics(train, start_balance),
            "test": compute_metrics(test, start_balance),
            "calibration_all": win_rate_by_bucket(requested),
            "significance_all": significance_report(requested),
            "significance_test": significance_report(test),
        },
        "band_sweep": sweep,
        "selected_in_sample": {
            "band": chosen["band"] if chosen else None,
            "out_of_sample_metrics": chosen["test"] if chosen else None,
            "note": "Band chosen on the training period only, then evaluated once on the "
            "later test period. Six bands were compared, so treat in-sample p-values "
            "as optimistic (multiple comparisons).",
        },
        "limitations": [
            "No score context: this is not the late-match model strategy, only its price-only "
            "baseline. Score-aware evaluation needs recorded data (backtest replay).",
            "Minute candles: intra-minute price paths and queue position are unknown.",
            "No order-book depth: assumes 10 contracts fillable at the best ask.",
        ],
        "runtime_s": round(time.time() - started, 1),
    }
    db = Database(settings.db_path)
    with db.session() as s:
        s.add(
            BacktestRunRow(
                created_ts=time.time(),
                kind="historical_price_band",
                data_source=DATA_SOURCE,
                params_json=json.dumps(
                    {
                        "days": days,
                        "series": series_list,
                        "band": f"{min_price}-{max_price}",
                        "max_events": max_events,
                        "test_fraction": test_fraction,
                    }
                ),
                metrics_json=json.dumps(
                    {
                        "all": report["requested_band"]["all"],
                        "train": report["requested_band"]["train"],
                        "test": report["requested_band"]["test"],
                        "significance_test": report["requested_band"]["significance_test"],
                    },
                    default=str,
                ),
                notes=report["selected_in_sample"]["note"],
            )
        )
    return report
