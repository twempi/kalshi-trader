"""Opt-in smoke tests against the real, public Kalshi API (read-only GET requests).

Skipped by default. Run with ``uv run pytest -m live``. They check that the live API still
matches the parsers; they assert nothing about prices, which change constantly. Tennis is
seasonal, so the score checks pass vacuously when no match is in progress.
"""

from __future__ import annotations

import pytest

from kalshi_trader.api.rest import KalshiRestClient
from kalshi_trader.api.schemas import validate_ticker
from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.market_data.discovery import TennisDiscovery
from kalshi_trader.tennis.providers.kalshi_live import KalshiLiveDataProvider

pytestmark = pytest.mark.live


async def test_exchange_status_and_tennis_series(settings: Settings) -> None:
    async with KalshiRestClient(settings.kalshi) as rest:
        status = await rest.get_exchange_status()
        assert isinstance(status.trading_active, bool)
        series = await rest.get_series_list(category="Sports", tags="Tennis")
    assert {"KXATPMATCH", "KXWTAMATCH"} <= {s.ticker for s in series}


async def test_discovery_books_and_scores_parse(settings: Settings) -> None:
    async with KalshiRestClient(settings.kalshi) as rest:
        discovery = TennisDiscovery(rest, Repository(Database(":memory:")), settings.discovery)
        found = await discovery.refresh()
        for ticker in found.markets:
            validate_ticker(ticker)
        tickers = list(found.markets)[:20]
        books = await rest.get_orderbooks(tickers) if tickers else {}
        assert set(books) <= set(tickers)
        for book in books.values():
            for level in [*book.yes_dollars, *book.no_dollars]:
                assert 0 < level[0] < 1 and level[1] >= 0
        refs = [r for m in found.matches.values() if (r := m.match_ref()) is not None]
        results = await KalshiLiveDataProvider(rest).fetch(refs[:50])
    for result in results.values():
        state = result.state
        if state.is_live and state.valid:
            assert state.score is not None
            assert all(0 <= g <= 30 for g in state.score.games)
