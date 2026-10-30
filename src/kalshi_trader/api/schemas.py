"""Pydantic models for the subset of Kalshi Trade API v2 responses this project uses.

Field names follow the official OpenAPI spec (https://docs.kalshi.com/openapi.yaml).
Prices are fixed-point dollar strings (``*_dollars``) and counts are fixed-point strings
(``*_fp``); both are parsed into ``Decimal``. Unknown fields are ignored so additive API
changes do not break parsing, while values used in trading maths are range-checked.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
ONE = Decimal("1")
ZERO = Decimal("0")


def validate_ticker(value: str) -> str:
    if not isinstance(value, str) or not TICKER_RE.match(value):
        raise ValueError(f"invalid Kalshi ticker: {value!r}")
    return value


def parse_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite decimal: {value!r}")
    return result


def parse_price(value: Any) -> Decimal | None:
    price = parse_decimal(value)
    if price is not None and not (ZERO <= price <= ONE):
        raise ValueError(f"price outside [0, 1]: {price}")
    return price


def parse_count(value: Any) -> Decimal | None:
    count = parse_decimal(value)
    if count is not None and count < 0:
        raise ValueError(f"negative contract count: {count}")
    return count


Price = Annotated[Decimal | None, BeforeValidator(parse_price)]
Count = Annotated[Decimal | None, BeforeValidator(parse_count)]
Ticker = Annotated[str, BeforeValidator(validate_ticker)]


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.startswith("0001-01-01"):  # Go zero time used by the API for "unset"
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


OptionalDatetime = Annotated[datetime | None, BeforeValidator(_parse_dt)]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class PriceRange(ApiModel):
    start: Decimal
    end: Decimal
    step: Decimal


class Market(ApiModel):
    ticker: Ticker
    event_ticker: Ticker
    market_type: str = "binary"
    title: str = ""
    yes_sub_title: str = ""
    no_sub_title: str = ""
    status: str = ""
    open_time: OptionalDatetime = None
    close_time: OptionalDatetime = None
    expected_expiration_time: OptionalDatetime = None
    occurrence_datetime: OptionalDatetime = None
    updated_time: OptionalDatetime = None
    yes_bid_dollars: Price = None
    yes_ask_dollars: Price = None
    yes_bid_size_fp: Count = None
    yes_ask_size_fp: Count = None
    no_bid_dollars: Price = None
    no_ask_dollars: Price = None
    last_price_dollars: Price = None
    volume_fp: Count = None
    volume_24h_fp: Count = None
    open_interest_fp: Count = None
    result: str = ""
    settlement_value_dollars: Price = None
    settlement_ts: OptionalDatetime = None
    can_close_early: bool | None = None
    custom_strike: dict[str, Any] | None = None
    rules_primary: str = ""
    price_level_structure: str | None = None
    price_ranges: list[PriceRange] = Field(default_factory=list)
    exchange_index: int | None = None

    @property
    def competitor_id(self) -> str | None:
        if self.custom_strike:
            value = self.custom_strike.get("tennis_competitor")
            return str(value) if value else None
        return None

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_settled(self) -> bool:
        return self.status in ("finalized", "settled")

    @property
    def is_determined(self) -> bool:
        return self.status in ("determined", "amended", "finalized", "settled") and bool(
            self.result
        )


class Event(ApiModel):
    event_ticker: Ticker
    series_ticker: str = ""
    title: str = ""
    sub_title: str = ""
    category: str = ""
    mutually_exclusive: bool | None = None
    product_metadata: dict[str, Any] | None = None
    markets: list[Market] | None = None

    @property
    def competition(self) -> str | None:
        if self.product_metadata:
            value = self.product_metadata.get("competition")
            return str(value) if value else None
        return None


class Series(ApiModel):
    ticker: Ticker
    title: str = ""
    category: str = ""
    tags: list[str] | None = None
    frequency: str = ""
    fee_type: str = ""
    fee_multiplier: Decimal = Decimal("1")

    @field_validator("fee_multiplier", mode="before")
    @classmethod
    def _fee_multiplier(cls, value: Any) -> Decimal:
        parsed = parse_decimal(value)
        return parsed if parsed is not None and parsed >= 0 else Decimal("1")


class Milestone(ApiModel):
    id: str
    category: str = ""
    type: str = ""
    title: str = ""
    start_date: OptionalDatetime = None
    end_date: OptionalDatetime = None
    details: dict[str, Any] = Field(default_factory=dict)
    primary_event_tickers: list[str] = Field(default_factory=list)
    related_event_tickers: list[str] = Field(default_factory=list)
    source_id: str | None = None
    last_updated_ts: OptionalDatetime = None


class LiveData(ApiModel):
    type: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    milestone_id: str


def parse_levels(value: Any) -> list[tuple[Decimal, Decimal]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("orderbook side must be a list")
    levels: list[tuple[Decimal, Decimal]] = []
    for item in value:
        if not isinstance(item, list | tuple) or len(item) < 2:
            raise ValueError(f"malformed orderbook level: {item!r}")
        price = parse_price(item[0])
        count = parse_count(item[1])
        if price is None or count is None:
            raise ValueError(f"malformed orderbook level: {item!r}")
        if ZERO < price < ONE and count > 0:
            levels.append((price, count))
    return levels


Levels = Annotated[list[tuple[Decimal, Decimal]], BeforeValidator(parse_levels)]


class OrderbookFp(ApiModel):
    """Resting bids only: a YES bid at p is a NO ask at 1-p and vice versa."""

    yes_dollars: Levels = Field(default_factory=list)
    no_dollars: Levels = Field(default_factory=list)


class MarketOrderbook(ApiModel):
    ticker: Ticker
    orderbook_fp: OrderbookFp


class Trade(ApiModel):
    trade_id: str
    ticker: Ticker
    count_fp: Count = None
    yes_price_dollars: Price = None
    no_price_dollars: Price = None
    taker_outcome_side: str | None = None
    created_time: OptionalDatetime = None
    is_block_trade: bool = False


class OHLC(ApiModel):
    open_dollars: Price = None
    high_dollars: Price = None
    low_dollars: Price = None
    close_dollars: Price = None


class Candlestick(ApiModel):
    end_period_ts: int
    yes_bid: OHLC | None = None
    yes_ask: OHLC | None = None
    price: OHLC | None = None
    volume_fp: Count = None
    open_interest_fp: Count = None


class ExchangeIndexStatus(ApiModel):
    exchange_index: int
    exchange_active: bool = True
    trading_active: bool = True
    description: str = ""


class ExchangeStatus(ApiModel):
    exchange_active: bool
    trading_active: bool
    exchange_estimated_resume_time: OptionalDatetime = None
    exchange_index_statuses: list[ExchangeIndexStatus] = Field(default_factory=list)

    def trading_active_for(self, exchange_index: int | None) -> bool:
        if not (self.exchange_active and self.trading_active) and exchange_index in (None, 0):
            return False
        for status in self.exchange_index_statuses:
            if status.exchange_index == exchange_index:
                return status.exchange_active and status.trading_active
        return self.exchange_active and self.trading_active


class HistoricalCutoff(ApiModel):
    market_settled_ts: OptionalDatetime = None
    trades_created_ts: OptionalDatetime = None
    orders_updated_ts: OptionalDatetime = None
