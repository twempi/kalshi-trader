"""End-to-end engine cycle against a fake exchange (synthetic data), plus dashboard API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D
from typing import Any

import pytest
from fastapi.testclient import TestClient

from kalshi_trader.api.rest import RestHealth
from kalshi_trader.api.schemas import (
    Event,
    ExchangeStatus,
    Market,
    Milestone,
    OrderbookFp,
    Series,
)
from kalshi_trader.api.ws import WsHealth
from kalshi_trader.config import Settings
from kalshi_trader.dashboard.app import create_app
from kalshi_trader.engine import TradingEngine
from kalshi_trader.tennis.providers.base import ProviderResult, ScoreProvider
from kalshi_trader.tennis.state import MatchFormat, MatchState, MatchStatus, ScoreState

NOW = datetime.now(UTC)


def market(ticker: str, name: str, competitor: str, **extra: Any) -> Market:
    return Market.model_validate(
        {
            "ticker": ticker,
            "event_ticker": "KXATPMATCH-TEST",
            "market_type": "binary",
            "status": "active",
            "yes_sub_title": name,
            "occurrence_datetime": (NOW - timedelta(hours=1)).isoformat(),
            "custom_strike": {"tennis_competitor": competitor},
            "price_ranges": [{"start": "0", "end": "1", "step": "0.01"}],
            "exchange_index": 3,
            **extra,
        }
    )


class FakeRest:
    """Implements the subset of KalshiRestClient the engine uses. Synthetic data only."""

    def __init__(self) -> None:
        self.health = RestHealth()
        self.final: dict[str, tuple[str, str]] = {}
        self.markets = [
            market("KXATPMATCH-TEST-A", "Player A", "comp-a"),
            market("KXATPMATCH-TEST-B", "Player B", "comp-b"),
        ]

    async def get_series_list(self, category=None, tags=None):
        return [Series(ticker="KXATPMATCH", fee_type="quadratic", fee_multiplier=D("1"))]

    async def get_markets(self, *, series_ticker=None, status=None, tickers=None, **_):
        if tickers:
            out = []
            for m in self.markets:
                if m.ticker in tickers:
                    if m.ticker in self.final:
                        st, res = self.final[m.ticker]
                        m = m.model_copy(
                            update={
                                "status": st,
                                "result": res,
                                "settlement_value_dollars": D("1") if res == "yes" else D("0"),
                            }
                        )
                    out.append(m)
            return out
        return list(self.markets) if series_ticker == "KXATPMATCH" else []

    async def get_milestones(self, **_):
        return [
            Milestone(
                id="ms-1",
                type="tennis_tournament_singles",
                start_date=NOW - timedelta(hours=1),
                details={
                    "first_competitor_id": "comp-a",
                    "second_competitor_id": "comp-b",
                    "main_game_event_ticker": "KXATPMATCH-TEST",
                    "best_of": "3",
                    "tour": "ATP",
                    "gender": "men",
                    "tournament_name": "ATP Test Open",
                },
            )
        ]

    async def get_event(self, ticker, with_nested_markets=True):
        return Event(
            event_ticker=ticker,
            title="Player A vs Player B",
            product_metadata={"competition": "ATP Test Open"},
        )

    async def get_exchange_status(self):
        return ExchangeStatus(exchange_active=True, trading_active=True)

    async def get_orderbooks(self, tickers):
        books = {}
        for t in tickers:
            if t.endswith("-A"):  # YES ask 0.92 x 40, YES bid 0.90 x 40
                books[t] = OrderbookFp.model_validate(
                    {"yes_dollars": [["0.90", "40"]], "no_dollars": [["0.08", "40"]]}
                )
            else:
                books[t] = OrderbookFp.model_validate(
                    {"yes_dollars": [["0.07", "40"]], "no_dollars": [["0.90", "40"]]}
                )
        return books

    async def get_candlesticks(self, *args, **kwargs):
        return []

    async def close(self) -> None:
        return None


class FakeProvider(ScoreProvider):
    name = "fake"

    async def fetch(self, refs):
        fmt = MatchFormat(best_of=3)
        state = ScoreState(
            fmt=fmt, sets=(1, 0), games=(5, 3), points=(3, 0), server=0, completed_sets=((6, 4),)
        )
        out = {}
        for ref in refs:
            out[ref.match_id] = ProviderResult(
                MatchState(
                    provider="fake",
                    match_id=ref.match_id,
                    player_ids=ref.player_ids,
                    status=MatchStatus.LIVE,
                    received_ts=datetime.now(UTC).timestamp(),
                    fmt=fmt,
                    score=state,
                ),
                raw={"synthetic": True},
            )
        return out


@pytest.fixture
def engine(settings: Settings) -> TradingEngine:
    settings.model.uncertainty_samples = 32
    settings.model.prior_draws = 1000
    return TradingEngine(settings, rest=FakeRest(), provider=FakeProvider())


async def test_full_cycle_trades_and_settles(engine: TradingEngine) -> None:
    await engine._discovery_step()
    assert "KXATPMATCH-TEST" in engine.matches
    match = engine.matches["KXATPMATCH-TEST"]
    assert match.market_tickers == ("KXATPMATCH-TEST-A", "KXATPMATCH-TEST-B")
    await engine._exchange_step()
    await engine._rest_books_step()
    await engine._scores_step()
    await engine._evaluate_step()

    trades = engine.broker.open_trades()
    assert len(trades) == 1, engine.repo.recent_signals()
    trade = trades[0]
    assert trade.market_ticker == "KXATPMATCH-TEST-A" and trade.side == "yes"
    assert trade.entry_price == D("0.92") and trade.contracts == D("20")  # 50% of 40 shown
    assert trade.data_source == "live_paper"

    await engine._evaluate_step()  # same opportunity again: no duplicate entry
    assert len(engine.broker.open_trades()) == 1

    views = engine.repo.live_views()
    assert {v["market_ticker"] for v in views} == {"KXATPMATCH-TEST-A", "KXATPMATCH-TEST-B"}
    a = next(v for v in views if v["market_ticker"].endswith("-A"))
    assert a["opportunity"] and a["model_prob"] > 0.95 and a["has_match_point"]

    engine.rest.final = {
        "KXATPMATCH-TEST-A": ("finalized", "yes"),
        "KXATPMATCH-TEST-B": ("finalized", "no"),
    }
    engine.markets["KXATPMATCH-TEST-A"].in_open_list = False
    await engine._status_step()
    assert engine.broker.open_trades() == []
    account = engine.broker.account()
    assert account.realized_pnl == D("20") - (D("18.40") + trade.entry_fee)
    await engine._equity_step()
    await engine._housekeeping_step()
    assert any(h.component == "kalshi_rest" for h in engine.repo.all_health())


async def test_engine_respects_pause_and_record_only(settings: Settings) -> None:
    settings.model.uncertainty_samples = 32
    settings.model.prior_draws = 1000
    eng = TradingEngine(settings, rest=FakeRest(), provider=FakeProvider())
    eng.risk.pause("test pause")
    for step in (
        eng._discovery_step,
        eng._exchange_step,
        eng._rest_books_step,
        eng._scores_step,
        eng._evaluate_step,
    ):
        await step()
    assert eng.broker.open_trades() == []
    rejected = eng.repo.recent_signals()
    assert rejected and not rejected[0].approved
    rec = TradingEngine(settings, record_only=True, rest=FakeRest(), provider=FakeProvider())
    assert rec.strategies == []


async def test_ws_gap_invalidates_books_and_requests_snapshots(engine: TradingEngine) -> None:
    class FakeWs:
        def __init__(self) -> None:
            self.requests: list[tuple[str, list[str]]] = []

        async def request_snapshot(self, channel: str, tickers: list[str]) -> None:
            self.requests.append((channel, tickers))

    ws = FakeWs()
    engine.ws = ws  # type: ignore[assignment]
    snapshot = {
        "type": "orderbook_snapshot",
        "sid": 1,
        "seq": 1,
        "msg": {"market_ticker": "KXATPMATCH-TEST-A", "yes_dollars_fp": [["0.90", "10"]]},
    }
    engine.hub.handle_ws_message(snapshot)
    assert engine.hub.ws_owns("KXATPMATCH-TEST-A")
    await engine._on_ws_gap("orderbook_delta", ["KXATPMATCH-TEST-A"])
    assert not engine.hub.books["KXATPMATCH-TEST-A"].valid
    assert not engine.hub.ws_owns("KXATPMATCH-TEST-A")  # REST polling takes over meanwhile
    assert ws.requests == [("orderbook_delta", ["KXATPMATCH-TEST-A"])]
    engine.hub.handle_ws_message(snapshot)  # the requested snapshot arrives
    assert engine.hub.ws_owns("KXATPMATCH-TEST-A") and engine.hub.books["KXATPMATCH-TEST-A"].valid

    ws.health = WsHealth()  # type: ignore[attr-defined]
    engine.hub.resync_requests.add("KXATPMATCH-TEST-B")  # e.g. a delta showed it out of sync
    await engine._rest_books_step()
    assert ws.requests[-1] == ("orderbook_delta", ["KXATPMATCH-TEST-B"])
    assert not engine.hub.resync_requests


async def test_engine_run_starts_and_stops_gracefully(engine: TradingEngine) -> None:
    task = asyncio.create_task(engine.run(install_signal_handlers=False))
    await asyncio.sleep(1.5)
    engine.request_stop()
    await asyncio.wait_for(task, timeout=10)
    health = {h.component: h.status for h in engine.repo.all_health()}
    assert health["engine"] == "stopped" and health["discovery"] == "ok"
    assert engine.recorder.market_rows_written > 0


def test_dashboard_endpoints(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    assert client.get("/").status_code == 200
    for path in (
        "/api/overview",
        "/api/markets",
        "/api/positions",
        "/api/trades",
        "/api/signals",
        "/api/analytics",
        "/api/health",
        "/api/backtests",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
    assert "SIMULATED" in client.get("/api/overview").json()["label"]
    csv = client.get("/api/export/trades.csv")
    assert csv.status_code == 200 and csv.text.startswith("id,strategy,market_ticker")


def test_dashboard_controls_require_header(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    assert client.post("/api/control/pause").status_code == 403
    ok = client.post("/api/control/pause", headers={"X-Kalshi-Trader": "1"})
    assert ok.status_code == 200
    assert client.get("/api/overview").json()["trading_enabled"] is False
    client.post("/api/control/emergency_stop", headers={"X-Kalshi-Trader": "1"})
    assert client.get("/api/overview").json()["emergency_stop"] is True
    assert client.post("/api/control/nope", headers={"X-Kalshi-Trader": "1"}).status_code == 404
