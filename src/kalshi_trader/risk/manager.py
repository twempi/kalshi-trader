"""Centralised risk management. Every opportunity must pass ``RiskManager`` before any
simulated execution, for the main paper account and for shadow ledgers alike.

Two kinds of output:

* **Rejections** (``check_market`` / ``check_portfolio``): data quality, market state,
  safety switches and hard portfolio limits.
* **A risk budget** (``budget``): the maximum dollars this trade may *lose*, i.e. the
  smallest remaining headroom among the per-trade cap, the aggregate exposure cap and the
  per-event, per-strategy and correlation-group caps. Exposure is measured as maximum
  possible loss (for a long binary contract, its full cost), not as probability of winning.
  The execution evaluator sizes the order inside this budget.

Sizing depends only on current equity, open exposure and estimated edge - never on past
wins or losses - so there is no martingale or loss-chasing: after a loss equity is lower
and the next budget is smaller.

The daily-loss and drawdown limits block *new* entries. They are not a loss ceiling:
positions already open can still lose up to their full cost.

Safety state (emergency stop, pause + reason) is persisted so it survives restarts.
Drawdown breaches trigger the emergency stop, which only an operator can clear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal

from kalshi_trader.config import RiskSettings
from kalshi_trader.core.evaluator import kelly_fraction_binary
from kalshi_trader.core.opportunity import Opportunity
from kalshi_trader.database.repository import (
    EMERGENCY_STOP,
    PAUSE_REASON,
    TRADING_ENABLED,
    Repository,
)
from kalshi_trader.logging_setup import get_logger

__all__ = [
    "AccountSnapshot",
    "MarketConditions",
    "OpenExposure",
    "RiskBudget",
    "RiskManager",
    "kelly_fraction_binary",
]

log = get_logger("risk")
ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True)
class OpenExposure:
    strategy: str
    event_ticker: str
    market_ticker: str
    max_loss: Decimal
    correlation_group: str | None = None
    group_id: str | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    cash: Decimal
    equity: Decimal
    peak_equity: Decimal
    day_start_equity: Decimal
    open_positions: int
    open_cost: Decimal = ZERO
    exposures: tuple[OpenExposure, ...] = ()

    @property
    def daily_pnl(self) -> Decimal:
        return self.equity - self.day_start_equity

    @property
    def drawdown_fraction(self) -> Decimal:
        if self.peak_equity <= 0:
            return ZERO
        return max(ZERO, (self.peak_equity - self.equity) / self.peak_equity)

    def exposure(
        self,
        *,
        event: str | None = None,
        strategy: str | None = None,
        group: str | None = None,
    ) -> Decimal:
        total = ZERO
        for e in self.exposures:
            if event is not None and e.event_ticker != event:
                continue
            if strategy is not None and e.strategy != strategy:
                continue
            if group is not None and e.correlation_group != group:
                continue
            total += e.max_loss
        return total


@dataclass(frozen=True)
class MarketConditions:
    market_tradable: bool
    exchange_trading_active: bool
    book_age_s: float | None
    score_age_s: float | None
    spread: Decimal | None
    depth_at_limit: Decimal
    trades_in_market: int
    trades_in_event: int
    has_open_position_in_event: bool
    score_unchanged_s: float | None = None
    mid_move_since_score_change: Decimal | None = None
    score_issue: str | None = None


@dataclass
class RiskBudget:
    max_loss: Decimal
    binding: str
    reasons: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, settings: RiskSettings, repo: Repository, fee_rate: Decimal) -> None:
        self.settings = settings
        self.repo = repo
        self.fee_rate = fee_rate
        self._consecutive_errors = 0

    # ---------------------------------------------------------- safety state
    def trading_enabled(self) -> bool:
        return self.repo.get_flag(TRADING_ENABLED, default=True)

    def emergency_stopped(self) -> bool:
        return self.repo.get_flag(EMERGENCY_STOP, default=False)

    def pause(self, reason: str) -> None:
        self.repo.set_flag(TRADING_ENABLED, False)
        self.repo.set_state(PAUSE_REASON, reason)
        self.repo.record_event("warning", "risk", f"paper trading paused: {reason}")
        log.warning("trading_paused", reason=reason)

    def resume(self) -> None:
        self.repo.set_flag(TRADING_ENABLED, True)
        self.repo.set_state(PAUSE_REASON, "")
        self._consecutive_errors = 0
        self.repo.record_event("info", "risk", "paper trading resumed")

    def emergency_stop(self, reason: str) -> None:
        self.repo.set_flag(EMERGENCY_STOP, True)
        self.repo.set_state(PAUSE_REASON, f"EMERGENCY STOP: {reason}")
        self.repo.record_event("error", "risk", f"emergency stop: {reason}")
        log.error("emergency_stop", reason=reason)

    def clear_emergency_stop(self) -> None:
        self.repo.set_flag(EMERGENCY_STOP, False)
        self.repo.set_state(PAUSE_REASON, "")
        self.repo.record_event("warning", "risk", "emergency stop cleared by operator")

    def record_error(self, component: str, error: str) -> None:
        self._consecutive_errors += 1
        if (
            self._consecutive_errors >= self.settings.max_consecutive_errors
            and self.trading_enabled()
        ):
            self.pause(
                f"{self._consecutive_errors} consecutive errors (last in {component}: "
                f"{error[:120]})"
            )

    def record_success(self) -> None:
        self._consecutive_errors = 0

    def check_account_limits(self, account: AccountSnapshot, escalate: bool = True) -> list[str]:
        """Portfolio-level loss checks; a drawdown breach escalates to the emergency stop."""
        problems = []
        s = self.settings
        if account.drawdown_fraction >= s.max_drawdown_fraction:
            problems.append(
                f"drawdown {account.drawdown_fraction:.1%} >= {s.max_drawdown_fraction:.0%}"
            )
            if escalate and not self.emergency_stopped():
                self.emergency_stop(problems[-1])
        if account.daily_pnl <= -s.max_daily_loss:
            problems.append(f"daily loss {account.daily_pnl} exceeds -{s.max_daily_loss}")
        return problems

    # ---------------------------------------------------------------- checks
    def check_market(self, opp: Opportunity, market: MarketConditions) -> list[str]:
        """Market-state and data-quality checks for one opportunity (or one basket leg)."""
        s = self.settings
        reasons: list[str] = []
        if not market.exchange_trading_active:
            reasons.append("exchange trading inactive")
        if not market.market_tradable:
            reasons.append("market not active")
        if market.book_age_s is None or market.book_age_s > s.max_market_data_age_s:
            age = "none" if market.book_age_s is None else f"{market.book_age_s:.1f}s"
            reasons.append(f"order book stale ({age})")
        if opp.requires_score:
            age = market.score_age_s
            if age is None or age > s.max_score_data_age_s:
                reasons.append(
                    f"score data stale ({'none' if age is None else f'{age:.0f}s'} incl. "
                    f"{s.score_source_staleness_s:.0f}s source cache)"
                )
            if (
                market.score_unchanged_s is not None
                and market.score_unchanged_s > s.max_score_unchanged_s
            ):
                reasons.append(f"score unchanged for {market.score_unchanged_s:.0f}s (frozen?)")
            if (
                market.mid_move_since_score_change is not None
                and market.mid_move_since_score_change > s.max_mid_move_since_score_change
            ):
                reasons.append(
                    f"market moved {market.mid_move_since_score_change} since the last score "
                    "change (score feed likely behind)"
                )
            if market.score_issue:
                reasons.append(f"score feed: {market.score_issue}")
        if opp.kind == "directional":
            if market.spread is None or market.spread > s.max_spread:
                reasons.append(f"spread {market.spread} > {s.max_spread}")
            if market.depth_at_limit < s.min_liquidity_contracts:
                reasons.append(f"liquidity {market.depth_at_limit} < {s.min_liquidity_contracts}")
        if market.trades_in_market >= s.max_trades_per_market:
            reasons.append("max trades for this market reached")
        if market.trades_in_event >= s.max_trades_per_event:
            reasons.append("max trades for this event reached")
        if market.has_open_position_in_event:
            reasons.append("already exposed to this event")
        return reasons

    def check_portfolio(
        self, account: AccountSnapshot, *, escalate: bool = True, safety: bool = True
    ) -> list[str]:
        s = self.settings
        reasons: list[str] = []
        if safety:
            if self.emergency_stopped():
                reasons.append("emergency stop active")
            if not self.trading_enabled():
                reasons.append("paper trading paused")
        if account.open_positions >= s.max_open_positions:
            reasons.append(f"max open positions ({s.max_open_positions})")
        reasons.extend(self.check_account_limits(account, escalate=escalate))
        return reasons

    def budget(self, opp: Opportunity, account: AccountSnapshot) -> RiskBudget:
        """Maximum loss this trade may add, given every exposure limit."""
        s = self.settings
        equity = max(ZERO, account.equity)
        caps: list[tuple[str, Decimal]] = [
            ("per-trade fraction", equity * s.max_fraction_per_trade),
            ("per-trade cap", s.max_position_cost),
            ("cash", account.cash),
            (
                "aggregate exposure",
                equity * s.max_aggregate_exposure_fraction - account.exposure(),
            ),
            (
                "event exposure",
                equity * s.max_event_exposure_fraction - account.exposure(event=opp.event_ticker),
            ),
            (
                "strategy exposure",
                equity * s.max_strategy_exposure_fraction - account.exposure(strategy=opp.strategy),
            ),
        ]
        if opp.correlation_group:
            caps.append(
                (
                    "correlated exposure",
                    equity * s.max_correlated_exposure_fraction
                    - account.exposure(group=opp.correlation_group),
                )
            )
        binding, value = min(caps, key=lambda kv: kv[1])
        value = max(ZERO, value)
        reasons = [] if value > 0 else [f"no risk headroom ({binding})"]
        return RiskBudget(value, binding, reasons)

    @staticmethod
    def utc_day_start(ts: float | None = None) -> float:
        now = ts if ts is not None else time.time()
        return now - (now % 86400)
