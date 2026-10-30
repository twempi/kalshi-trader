"""Centralised risk management. Every strategy signal must pass ``RiskManager.check``.

Sizing is fractional Kelly on the *conservative* probability, capped by a bankroll
fraction, an absolute position cap and available cash. It depends only on current equity
and estimated edge - never on past wins or losses - so there is no martingale or
loss-chasing behaviour: after a loss, equity is lower and the next size is smaller.

Safety state (emergency stop, pause + reason) is persisted in the database so it survives
restarts. Drawdown breaches trigger the emergency stop, which only a human can clear.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal

from kalshi_trader.config import RiskSettings
from kalshi_trader.database.repository import (
    EMERGENCY_STOP,
    PAUSE_REASON,
    TRADING_ENABLED,
    Repository,
)
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.strategies.base import Signal

log = get_logger("risk")
ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True)
class AccountSnapshot:
    cash: Decimal
    equity: Decimal
    peak_equity: Decimal
    day_start_equity: Decimal
    open_positions: int
    open_cost: Decimal = ZERO

    @property
    def daily_pnl(self) -> Decimal:
        return self.equity - self.day_start_equity

    @property
    def drawdown_fraction(self) -> Decimal:
        if self.peak_equity <= 0:
            return ZERO
        return max(ZERO, (self.peak_equity - self.equity) / self.peak_equity)


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


@dataclass
class RiskDecision:
    approved: bool
    contracts: Decimal = ZERO
    max_cash: Decimal = ZERO
    reasons: list[str] = field(default_factory=list)


def kelly_fraction_binary(p_win: float, cost: Decimal) -> Decimal:
    """Full-Kelly fraction of bankroll to stake on a $1 binary bought at all-in `cost`."""
    if cost <= 0 or cost >= 1:
        return ZERO
    edge = Decimal(str(p_win)) - cost
    return max(ZERO, edge / (ONE - cost))


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

    def check_account_limits(self, account: AccountSnapshot) -> list[str]:
        """Portfolio-level checks; may escalate to pause/emergency stop."""
        problems = []
        s = self.settings
        if account.drawdown_fraction >= s.max_drawdown_fraction:
            problems.append(
                f"drawdown {account.drawdown_fraction:.1%} >= {s.max_drawdown_fraction:.0%}"
            )
            if not self.emergency_stopped():
                self.emergency_stop(problems[-1])
        if account.daily_pnl <= -s.max_daily_loss:
            problems.append(f"daily loss {account.daily_pnl} exceeds -{s.max_daily_loss}")
        return problems

    # ----------------------------------------------------------------- check
    def size(self, signal: Signal, account: AccountSnapshot) -> tuple[Decimal, Decimal]:
        """(contracts, max cash) for a signal, before liquidity limits."""
        s = self.settings
        price = signal.limit_price
        fee_pc = fee_per_contract_estimate(
            price, Decimal("10"), self.fee_rate, signal.fee_multiplier
        )
        cost = price + fee_pc
        equity = max(ZERO, account.equity)
        cap_cash = min(equity * s.max_fraction_per_trade, s.max_position_cost, account.cash)
        if signal.p_conservative is not None:
            stake = equity * kelly_fraction_binary(signal.p_conservative, cost) * s.kelly_fraction
            stake = min(stake, cap_cash)
        else:  # model-free strategies get a flat stake at the caps
            stake = cap_cash
        if cost <= 0 or stake <= 0:
            return ZERO, ZERO
        contracts = (stake / cost).to_integral_value(rounding=ROUND_FLOOR)
        return contracts, stake

    def check(
        self, signal: Signal, account: AccountSnapshot, market: MarketConditions
    ) -> RiskDecision:
        s = self.settings
        reasons: list[str] = []
        if self.emergency_stopped():
            reasons.append("emergency stop active")
        if not self.trading_enabled():
            reasons.append("paper trading paused")
        if not market.exchange_trading_active:
            reasons.append("exchange trading inactive")
        if not market.market_tradable:
            reasons.append("market not active")
        if market.book_age_s is None or market.book_age_s > s.max_market_data_age_s:
            reasons.append(f"order book stale ({market.book_age_s})")
        if signal.requires_score and (
            market.score_age_s is None or market.score_age_s > s.max_score_data_age_s
        ):
            reasons.append(f"score data stale ({market.score_age_s})")
        if market.spread is None or market.spread > s.max_spread:
            reasons.append(f"spread {market.spread} > {s.max_spread}")
        if market.depth_at_limit < s.min_liquidity_contracts:
            reasons.append(f"liquidity {market.depth_at_limit} < {s.min_liquidity_contracts}")
        if market.trades_in_market >= s.max_trades_per_market:
            reasons.append("max trades for this market reached")
        if market.trades_in_event >= s.max_trades_per_event:
            reasons.append("max trades for this match reached")
        if market.has_open_position_in_event:
            reasons.append("already exposed to this match")
        if account.open_positions >= s.max_open_positions:
            reasons.append(f"max open positions ({s.max_open_positions})")
        reasons.extend(self.check_account_limits(account))
        if signal.p_conservative is not None and signal.p_conservative <= float(signal.limit_price):
            reasons.append("no conservative edge at limit price")

        contracts, max_cash = self.size(signal, account)
        if contracts < 1:
            reasons.append("position size rounds to zero")
        if reasons:
            return RiskDecision(False, ZERO, ZERO, reasons)
        return RiskDecision(True, contracts, max_cash, [])

    @staticmethod
    def utc_day_start(ts: float | None = None) -> float:
        now = ts if ts is not None else time.time()
        return now - (now % 86400)
