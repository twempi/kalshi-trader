"""API response parsing, identifier validation and request signing."""

from __future__ import annotations

import base64
from decimal import Decimal

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from pydantic import ValidationError

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.api.schemas import (
    Candlestick,
    Event,
    ExchangeStatus,
    Market,
    OrderbookFp,
    validate_ticker,
)
from tests.conftest import load_fixture


def test_parses_real_market_list() -> None:
    markets = [Market.model_validate(m) for m in load_fixture("markets_atp.json")["markets"]]
    assert len(markets) == 4
    m = markets[0]
    assert m.ticker == "KXATPMATCH-26OCT30ZVEFER-ZVE"
    assert m.competitor_id == "dc4002ad-fb32-4f36-b59f-7c7af1927c57"
    assert m.yes_sub_title == "Alexander Zverev"
    assert isinstance(m.yes_bid_dollars, Decimal) and Decimal(0) <= m.yes_bid_dollars <= 1
    assert m.price_ranges and m.price_ranges[0].step == Decimal("0.0100")
    assert m.is_active and not m.is_settled


def test_parses_real_event_orderbook_candles_status() -> None:
    event = Event.model_validate(load_fixture("event.json")["event"])
    assert event.competition and event.series_ticker == "KXATPMATCH"
    book = OrderbookFp.model_validate(load_fixture("orderbook.json")["orderbook_fp"])
    assert book.yes_dollars and book.no_dollars
    assert all(Decimal(0) < p < 1 and s > 0 for p, s in book.yes_dollars + book.no_dollars)
    candles = [
        Candlestick.model_validate(c) for c in load_fixture("candlesticks.json")["candlesticks"]
    ]
    assert candles and candles[0].yes_ask is not None
    status = ExchangeStatus.model_validate(load_fixture("exchange_status.json"))
    assert isinstance(status.trading_active_for(3), bool)


def test_orderbook_rejects_out_of_range_prices() -> None:
    with pytest.raises(ValidationError):
        OrderbookFp.model_validate({"yes_dollars": [["1.5000", "10.00"]], "no_dollars": []})
    with pytest.raises(ValidationError):
        OrderbookFp.model_validate({"yes_dollars": [["0.5000", "-3"]], "no_dollars": []})
    with pytest.raises(ValidationError):
        OrderbookFp.model_validate({"yes_dollars": [["abc", "1"]], "no_dollars": []})
    empty = OrderbookFp.model_validate({"yes_dollars": None, "no_dollars": None})
    assert empty.yes_dollars == [] and empty.no_dollars == []


def test_market_rejects_bad_prices_and_tickers() -> None:
    base = load_fixture("markets_atp.json")["markets"][0]
    with pytest.raises(ValidationError):
        Market.model_validate({**base, "yes_ask_dollars": "1.2000"})
    with pytest.raises(ValidationError):
        Market.model_validate({**base, "ticker": "../../portfolio/orders"})


@pytest.mark.parametrize("ticker", ["KXATPMATCH-26OCT30ZVEFER-ZVE", "FED-23DEC-T3.00", "A_B-1"])
def test_valid_tickers(ticker: str) -> None:
    assert validate_ticker(ticker) == ticker


@pytest.mark.parametrize("ticker", ["", "a/b", "../x", "x?y=1", "a b", "-leading", "x" * 201])
def test_invalid_tickers(ticker: str) -> None:
    with pytest.raises(ValueError):
        validate_ticker(ticker)


def test_go_zero_time_is_treated_as_unset() -> None:
    event = Event.model_validate(
        {"event_ticker": "KXATPMATCH-X", "last_updated_ts": "0001-01-01T00:00:00Z"}
    )
    assert event.event_ticker == "KXATPMATCH-X"


def test_ed25519_signature_verifies_and_strips_query() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    signer = KalshiSigner("key-id", key)
    headers = signer.headers("get", "/trade-api/v2/portfolio/balance?limit=5", timestamp_ms=1700)
    assert headers["KALSHI-ACCESS-KEY"] == "key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700"
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    key.public_key().verify(sig, b"1700GET/trade-api/v2/portfolio/balance")
    assert "key-id" not in repr(signer)


def test_rsa_pss_signature_verifies() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer = KalshiSigner("k", key)
    headers = signer.headers("GET", "/trade-api/ws/v2", timestamp_ms=42)
    key.public_key().verify(
        base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"]),
        b"42GET/trade-api/ws/v2",
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    assert signer.key_type == "rsa"
