"""SQLite schema (SQLAlchemy 2.0 ORM).

Conventions:
* Timestamps are UTC epoch seconds (REAL) in ``*_ts`` columns.
* Accounting values (cash, cost, fees, P&L, fill prices) use ``DecimalText`` so they are
  stored exactly; high-volume observations (market/score snapshots) use REAL for speed.
* Every paper trade carries ``data_source`` so simulated results can never be confused
  with anything else (``live_paper`` = simulated fills against real live Kalshi books).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class DecimalText(TypeDecorator[Decimal]):
    impl = String(48)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        return None if value is None else str(Decimal(value))

    def process_result_value(self, value: Any, dialect: Any) -> Decimal | None:
        return None if value is None else Decimal(value)


class Base(DeclarativeBase):
    pass


class SeriesRow(Base):
    __tablename__ = "series"
    ticker: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    fee_type: Mapped[str] = mapped_column(String(64), default="")
    fee_multiplier: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("1"))
    updated_ts: Mapped[float] = mapped_column(Float)


class EventRow(Base):
    __tablename__ = "events"
    event_ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    series_ticker: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(256), default="")
    sub_title: Mapped[str] = mapped_column(String(256), default="")
    competition: Mapped[str | None] = mapped_column(String(128))
    milestone_id: Mapped[str | None] = mapped_column(String(64), index=True)
    tour: Mapped[str | None] = mapped_column(String(64))
    gender: Mapped[str | None] = mapped_column(String(16))
    round_name: Mapped[str | None] = mapped_column(String(64))
    tournament_name: Mapped[str | None] = mapped_column(String(128))
    best_of: Mapped[int | None] = mapped_column(Integer)
    scheduled_start_ts: Mapped[float | None] = mapped_column(Float)
    first_competitor_id: Mapped[str | None] = mapped_column(String(64))
    second_competitor_id: Mapped[str | None] = mapped_column(String(64))
    # Generic (non-tennis) events: Kalshi's MECNET flag ("at most one market resolves yes").
    mutually_exclusive: Mapped[bool | None] = mapped_column(Boolean)
    category: Mapped[str | None] = mapped_column(String(64))
    first_seen_ts: Mapped[float] = mapped_column(Float)
    updated_ts: Mapped[float] = mapped_column(Float)


class MarketRow(Base):
    __tablename__ = "markets"
    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_ticker: Mapped[str] = mapped_column(String(128), index=True)
    series_ticker: Mapped[str] = mapped_column(String(64), index=True)
    player_name: Mapped[str] = mapped_column(String(128), default="")
    competitor_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="")
    result: Mapped[str] = mapped_column(String(16), default="")
    settlement_value: Mapped[Decimal | None] = mapped_column(DecimalText)
    settlement_ts: Mapped[float | None] = mapped_column(Float)
    open_ts: Mapped[float | None] = mapped_column(Float)
    close_ts: Mapped[float | None] = mapped_column(Float)
    expected_expiration_ts: Mapped[float | None] = mapped_column(Float)
    occurrence_ts: Mapped[float | None] = mapped_column(Float)
    exchange_index: Mapped[int | None] = mapped_column(Integer)
    price_ranges_json: Mapped[str | None] = mapped_column(Text)
    fee_multiplier: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("1"))
    # Last market-implied probability observed before the match started (strength prior).
    pre_match_prob: Mapped[float | None] = mapped_column(Float)
    pre_match_prob_ts: Mapped[float | None] = mapped_column(Float)
    # When this application *learned* the pre-match price (live capture or candle backfill).
    # Replay must not use a prior before this moment unless it was capturable at the time.
    pre_match_prob_known_ts: Mapped[float | None] = mapped_column(Float)
    title: Mapped[str | None] = mapped_column(String(256))
    strike_type: Mapped[str | None] = mapped_column(String(32))
    floor_strike: Mapped[float | None] = mapped_column(Float)
    cap_strike: Mapped[float | None] = mapped_column(Float)
    first_seen_ts: Mapped[float] = mapped_column(Float)
    updated_ts: Mapped[float] = mapped_column(Float)


class MarketSnapshotRow(Base):
    __tablename__ = "market_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128))
    received_ts: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(8))
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_bid_size: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    yes_ask_size: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(32))
    depth_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_market_snapshots_ticker_ts", "ticker", "received_ts"),
        Index("ix_market_snapshots_ts", "received_ts"),
    )


class ScoreSnapshotRow(Base):
    __tablename__ = "score_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    milestone_id: Mapped[str] = mapped_column(String(64))
    event_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    received_ts: Mapped[float] = mapped_column(Float)
    provider: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    state_hash: Mapped[str] = mapped_column(String(32))
    state_json: Mapped[str] = mapped_column(Text)
    raw_json: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("ix_score_snapshots_milestone_ts", "milestone_id", "received_ts"),
        Index("ix_score_snapshots_ts", "received_ts"),
    )


class SignalRow(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_ts: Mapped[float] = mapped_column(Float, index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    market_ticker: Mapped[str] = mapped_column(String(128), index=True)
    event_ticker: Mapped[str | None] = mapped_column(String(128))
    side: Mapped[str] = mapped_column(String(4))
    ask_price: Mapped[Decimal | None] = mapped_column(DecimalText)
    model_prob: Mapped[float | None] = mapped_column(Float)
    model_prob_low: Mapped[float | None] = mapped_column(Float)
    robust_prob: Mapped[float | None] = mapped_column(Float)
    market_prob: Mapped[float | None] = mapped_column(Float)
    ev_per_contract: Mapped[float | None] = mapped_column(Float)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    reasons_json: Mapped[str | None] = mapped_column(Text)
    score_text: Mapped[str | None] = mapped_column(String(128))


class PaperOrderRow(Base):
    __tablename__ = "paper_orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    client_order_id: Mapped[str] = mapped_column(String(96), unique=True)
    created_ts: Mapped[float] = mapped_column(Float, index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    market_ticker: Mapped[str] = mapped_column(String(128), index=True)
    event_ticker: Mapped[str | None] = mapped_column(String(128))
    side: Mapped[str] = mapped_column(String(4))
    action: Mapped[str] = mapped_column(String(8))
    limit_price: Mapped[Decimal] = mapped_column(DecimalText)
    requested_contracts: Mapped[Decimal] = mapped_column(DecimalText)
    filled_contracts: Mapped[Decimal] = mapped_column(DecimalText)
    avg_price: Mapped[Decimal | None] = mapped_column(DecimalText)
    notional: Mapped[Decimal] = mapped_column(DecimalText)
    fee: Mapped[Decimal] = mapped_column(DecimalText)
    status: Mapped[str] = mapped_column(String(16))
    fills_json: Mapped[str | None] = mapped_column(Text)
    trade_id: Mapped[int | None] = mapped_column(ForeignKey("paper_trades.id"))
    data_source: Mapped[str] = mapped_column(String(32), default="live_paper")
    # 1 = the main paper account; other ids are per-strategy shadow ledgers.
    account_id: Mapped[int] = mapped_column(Integer, default=1, index=True)
    group_id: Mapped[str | None] = mapped_column(String(128))  # multi-leg baskets
    opportunity_id: Mapped[str | None] = mapped_column(String(160))
    best_price_at_decision: Mapped[Decimal | None] = mapped_column(DecimalText)


class PaperTradeRow(Base):
    __tablename__ = "paper_trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    market_ticker: Mapped[str] = mapped_column(String(128), index=True)
    event_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    match_title: Mapped[str] = mapped_column(String(256), default="")
    player_name: Mapped[str] = mapped_column(String(128), default="")
    side: Mapped[str] = mapped_column(String(4))
    opened_ts: Mapped[float] = mapped_column(Float, index=True)
    contracts: Mapped[Decimal] = mapped_column(DecimalText)  # contracts still held
    entry_contracts: Mapped[Decimal] = mapped_column(DecimalText)
    entry_price: Mapped[Decimal] = mapped_column(DecimalText)
    entry_cost: Mapped[Decimal] = mapped_column(DecimalText)
    entry_fee: Mapped[Decimal] = mapped_column(DecimalText)
    exit_contracts: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    exit_proceeds: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    exit_fee: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    score_at_entry: Mapped[str | None] = mapped_column(String(128))
    model_prob: Mapped[float | None] = mapped_column(Float)
    model_prob_low: Mapped[float | None] = mapped_column(Float)
    robust_prob: Mapped[float | None] = mapped_column(Float)
    market_prob: Mapped[float | None] = mapped_column(Float)
    ev_per_contract: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    settlement_result: Mapped[str | None] = mapped_column(String(16))
    settlement_value: Mapped[Decimal | None] = mapped_column(DecimalText)
    payout: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    realized_pnl: Mapped[Decimal | None] = mapped_column(DecimalText)
    closed_ts: Mapped[float | None] = mapped_column(Float)
    data_source: Mapped[str] = mapped_column(String(32), default="live_paper")
    account_id: Mapped[int] = mapped_column(Integer, default=1, index=True)
    strategy_version: Mapped[str | None] = mapped_column(String(32))
    group_id: Mapped[str | None] = mapped_column(String(128))
    opportunity_id: Mapped[str | None] = mapped_column(String(160))
    category: Mapped[str | None] = mapped_column(String(32))
    stage: Mapped[str | None] = mapped_column(String(32))
    correlation_group: Mapped[str | None] = mapped_column(String(160))
    # Ex-ante expectations at the *actual* simulated fill, for expected-vs-realised analysis.
    expected_pnl: Mapped[float | None] = mapped_column(Float)  # conservative probability
    expected_pnl_point: Mapped[float | None] = mapped_column(Float)  # point probability
    max_loss: Mapped[Decimal | None] = mapped_column(DecimalText)
    slippage: Mapped[Decimal | None] = mapped_column(DecimalText)  # vs best ask, dollars


class PaperAccountRow(Base):
    __tablename__ = "paper_account"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(96))  # "main" or "shadow:<strategy>"
    starting_balance: Mapped[Decimal] = mapped_column(DecimalText)
    cash: Mapped[Decimal] = mapped_column(DecimalText)
    realized_pnl: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    fees_paid: Mapped[Decimal] = mapped_column(DecimalText, default=Decimal("0"))
    peak_equity: Mapped[Decimal] = mapped_column(DecimalText)
    created_ts: Mapped[float] = mapped_column(Float)
    updated_ts: Mapped[float] = mapped_column(Float)


class EquitySnapshotRow(Base):
    __tablename__ = "equity_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    cash: Mapped[Decimal] = mapped_column(DecimalText)
    open_value: Mapped[Decimal] = mapped_column(DecimalText)
    equity: Mapped[Decimal] = mapped_column(DecimalText)
    realized_pnl: Mapped[Decimal] = mapped_column(DecimalText)
    open_positions: Mapped[int] = mapped_column(Integer)
    account_id: Mapped[int] = mapped_column(Integer, default=1, index=True)


class SystemStateRow(Base):
    __tablename__ = "system_state"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_ts: Mapped[float] = mapped_column(Float)


class SystemEventRow(Base):
    __tablename__ = "system_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    level: Mapped[str] = mapped_column(String(16))
    component: Mapped[str | None] = mapped_column(String(64))
    message: Mapped[str] = mapped_column(Text)
    details_json: Mapped[str | None] = mapped_column(Text)


class ComponentHealthRow(Base):
    __tablename__ = "component_health"
    component: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16))
    updated_ts: Mapped[float] = mapped_column(Float)
    last_ok_ts: Mapped[float | None] = mapped_column(Float)
    last_error_ts: Mapped[float | None] = mapped_column(Float)
    last_error: Mapped[str | None] = mapped_column(Text)
    details_json: Mapped[str | None] = mapped_column(Text)


class LiveViewRow(Base):
    """Latest per-market view written by the engine and read by the dashboard."""

    __tablename__ = "live_view"
    market_ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_ticker: Mapped[str] = mapped_column(String(128), index=True)
    updated_ts: Mapped[float] = mapped_column(Float, index=True)
    data_json: Mapped[str] = mapped_column(Text)


class BacktestRunRow(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_ts: Mapped[float] = mapped_column(Float)
    kind: Mapped[str] = mapped_column(String(32))
    data_source: Mapped[str] = mapped_column(String(64))
    params_json: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[str] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    # Reproducibility: full config, market universe, input fingerprint, code version, ledger.
    config_json: Mapped[str | None] = mapped_column(Text)
    universe_json: Mapped[str | None] = mapped_column(Text)
    input_fingerprint: Mapped[str | None] = mapped_column(String(64))
    code_version: Mapped[str | None] = mapped_column(String(64))
    ledger_json: Mapped[str | None] = mapped_column(Text)


class OpportunityRow(Base):
    """Every evaluated opportunity (throttled), including why it was rejected.

    ``data_kind`` separates real observations (``live``) from ``replay`` and ``synthetic``.
    """

    __tablename__ = "opportunities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    opportunity_id: Mapped[str] = mapped_column(String(160), index=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    strategy_version: Mapped[str] = mapped_column(String(32), default="")
    mode: Mapped[str] = mapped_column(String(16), default="")
    kind: Mapped[str] = mapped_column(String(16), default="directional")
    category: Mapped[str] = mapped_column(String(32), default="")
    stage: Mapped[str | None] = mapped_column(String(32))
    event_ticker: Mapped[str | None] = mapped_column(String(128), index=True)
    market_ticker: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(256), default="")
    side: Mapped[str] = mapped_column(String(8))
    legs_json: Mapped[str | None] = mapped_column(Text)
    best_bid: Mapped[float | None] = mapped_column(Float)
    best_ask: Mapped[float | None] = mapped_column(Float)
    ask_size: Mapped[float | None] = mapped_column(Float)
    fair_value: Mapped[float | None] = mapped_column(Float)
    fair_low: Mapped[float | None] = mapped_column(Float)
    contracts: Mapped[float | None] = mapped_column(Float)
    limit_price: Mapped[float | None] = mapped_column(Float)
    avg_price: Mapped[float | None] = mapped_column(Float)
    fees: Mapped[float | None] = mapped_column(Float)
    cost: Mapped[float | None] = mapped_column(Float)
    expected_profit: Mapped[float | None] = mapped_column(Float)
    expected_profit_low: Mapped[float | None] = mapped_column(Float)
    expected_roc: Mapped[float | None] = mapped_column(Float)
    max_loss: Mapped[float | None] = mapped_column(Float)
    book_age_s: Mapped[float | None] = mapped_column(Float)
    score_age_s: Mapped[float | None] = mapped_column(Float)
    provenance_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), index=True)
    reasons_json: Mapped[str | None] = mapped_column(Text)
    data_kind: Mapped[str] = mapped_column(String(16), default="live")


class StrategyStateRow(Base):
    """Persistent per-strategy lifecycle state (evidence status, operator pause)."""

    __tablename__ = "strategy_state"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), default="")
    config_hash: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(24), default="experimental")
    status_reason: Mapped[str] = mapped_column(Text, default="")
    paused: Mapped[bool] = mapped_column(Boolean, default=False)
    evidence_json: Mapped[str | None] = mapped_column(Text)
    updated_ts: Mapped[float] = mapped_column(Float)


class StrategyStatusHistoryRow(Base):
    __tablename__ = "strategy_status_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32), default="")
    config_hash: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(24))
    reason: Mapped[str] = mapped_column(Text, default="")


class DiagnosticCountRow(Base):
    """Daily per-strategy counts of candidate outcomes and rejection reasons."""

    __tablename__ = "strategy_diagnostics"
    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # UTC YYYY-MM-DD
    strategy: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), primary_key=True)
    reason: Mapped[str] = mapped_column(String(160), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    updated_ts: Mapped[float] = mapped_column(Float)


class ModelPredictionRow(Base):
    """Model probability next to the executable market at the same instant (forward data
    for calibration and model-vs-market evaluation). One row per score change per market."""

    __tablename__ = "model_predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    model: Mapped[str] = mapped_column(String(48))
    event_ticker: Mapped[str] = mapped_column(String(128), index=True)
    market_ticker: Mapped[str] = mapped_column(String(128), index=True)
    tour: Mapped[str | None] = mapped_column(String(32))
    stage: Mapped[str | None] = mapped_column(String(32))
    score_text: Mapped[str | None] = mapped_column(String(128))
    p_point: Mapped[float] = mapped_column(Float)
    p_low: Mapped[float | None] = mapped_column(Float)
    p_robust_low: Mapped[float | None] = mapped_column(Float)
    prior_source: Mapped[str | None] = mapped_column(String(32))
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    book_age_s: Mapped[float | None] = mapped_column(Float)
    score_age_s: Mapped[float | None] = mapped_column(Float)


class SimulatedQuoteFillRow(Base):
    """Market-making RESEARCH simulation: hypothetical maker fills (never account cash)."""

    __tablename__ = "mm_simulated_fills"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    market_ticker: Mapped[str] = mapped_column(String(128), index=True)
    event_ticker: Mapped[str | None] = mapped_column(String(128))
    side: Mapped[str] = mapped_column(String(4))  # side bought by our resting bid
    price: Mapped[Decimal] = mapped_column(DecimalText)
    contracts: Mapped[Decimal] = mapped_column(DecimalText)
    fee: Mapped[Decimal] = mapped_column(DecimalText)
    fill_rule: Mapped[str] = mapped_column(String(32))
    mid_at_fill: Mapped[float | None] = mapped_column(Float)
    mid_after: Mapped[float | None] = mapped_column(Float)  # adverse-selection measurement
    settled_value: Mapped[Decimal | None] = mapped_column(DecimalText)
    pnl: Mapped[Decimal | None] = mapped_column(DecimalText)
    data_kind: Mapped[str] = mapped_column(String(16), default="live")
