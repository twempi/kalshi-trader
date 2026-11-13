"""Application configuration.

Precedence (highest first): explicit kwargs > environment variables > ``.env`` >
TOML config file (``config/settings.toml`` or ``$KT_CONFIG_PATH``) > defaults.

Nested values can be overridden from the environment with ``__`` as delimiter,
e.g. ``KT_RISK__MAX_OPEN_POSITIONS=3``. Secrets (API key id / private key) are
only ever read from the environment or ``.env``, never from the TOML file.
"""

from __future__ import annotations

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    IPvAnyNetwork,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"

KALSHI_ENVIRONMENTS: dict[str, dict[str, str]] = {
    "prod": {
        "rest": "https://external-api.kalshi.com/trade-api/v2",
        "ws": "wss://external-api-ws.kalshi.com/trade-api/ws/v2",
    },
    "demo": {
        "rest": "https://external-api.demo.kalshi.co/trade-api/v2",
        "ws": "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2",
    },
}


def _to_decimal(value: Any) -> Decimal:
    # Route floats through str() so 0.07 becomes Decimal("0.07"), not a binary approximation.
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


Money = Annotated[Decimal, BeforeValidator(_to_decimal)]


class KalshiSettings(BaseModel):
    environment: Literal["prod", "demo"] = "prod"
    rest_base_url: str | None = None
    ws_url: str | None = None
    # Credentials are optional: all market data used here is public over REST.
    # They are required only for the WebSocket feed (Kalshi requires an authenticated
    # handshake even for public channels). A key with only the `read` scope is sufficient.
    api_key_id: SecretStr | None = None
    private_key_path: Path | None = None
    request_timeout_s: float = 15.0
    max_requests_per_second: float = 8.0
    burst: int = 10
    max_retries: int = 5
    use_websocket: bool = True

    @property
    def rest_url(self) -> str:
        return (self.rest_base_url or KALSHI_ENVIRONMENTS[self.environment]["rest"]).rstrip("/")

    @property
    def websocket_url(self) -> str:
        return self.ws_url or KALSHI_ENVIRONMENTS[self.environment]["ws"]

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.api_key_id and self.api_key_id.get_secret_value() and self.private_key_path
        )


class DiscoverySettings(BaseModel):
    # Singles match-winner series. Verified against GET /series?category=Sports&tags=Tennis.
    series_tickers: list[str] = Field(
        default_factory=lambda: [
            "KXATPMATCH",
            "KXWTAMATCH",
            "KXATPCHALLENGERMATCH",
            "KXWTACHALLENGERMATCH",
            "KXITFMATCH",
            "KXITFWMATCH",
        ]
    )
    refresh_interval_s: float = 60.0
    # Track matches scheduled to start within this window around "now".
    lookahead_hours: float = 6.0
    lookback_hours: float = 18.0


class MarketDataSettings(BaseModel):
    rest_poll_interval_s: float = 3.0
    orderbook_depth: int = 10
    recorded_depth_levels: int = 5
    # Record an unchanged snapshot at least this often (keeps a heartbeat in the data).
    snapshot_heartbeat_s: float = 60.0
    # Write a market's changed book at most this often; the latest state is still written
    # once the interval passes. 0 = every change.
    snapshot_min_interval_s: float = Field(default=0.0, ge=0.0)
    # Delete recorded order books older than this many days (freed space is reused, so the
    # database stops growing). None = keep everything. Score snapshots and trades are kept.
    snapshot_retention_days: float | None = Field(default=None, gt=0)
    status_refresh_interval_s: float = 30.0


class TennisSettings(BaseModel):
    provider: Literal["kalshi_live_data", "none"] = "kalshi_live_data"
    poll_interval_s: float = 5.0
    snapshot_heartbeat_s: float = 120.0


class ModelSettings(BaseModel):
    # Long-run average share of service points won (approximate, by tour level).
    serve_win_prior: dict[str, float] = Field(
        default_factory=lambda: {
            "ATP": 0.640,
            "WTA": 0.560,
            "ATP_CHALLENGER": 0.620,
            "WTA_CHALLENGER": 0.555,
            "ITF_MEN": 0.600,
            "ITF_WOMEN": 0.545,
        }
    )
    # Prior sd of the match's average serve-point win probability (mu).
    serve_mean_sd: float = 0.03
    # Prior sd of the serve-probability difference (delta) with / without a pre-match price.
    strength_diff_sd_with_price: float = 0.025
    strength_diff_sd_no_price: float = 0.06
    # Prior draws for sampling-importance-resampling of the live serve statistics.
    prior_draws: int = 4000
    use_pre_match_price_prior: bool = True
    # Per-player, per-game probability of an in-match retirement.
    retirement_hazard_per_game: float = 0.0003
    uncertainty_samples: int = 200
    # Lower quantile of the model's probability distribution used for conservative EV.
    credible_quantile: float = 0.10
    # Assume the score feed may lag the market by this many points (latency guard).
    latency_points: int = 1


StrategyMode = Literal["off", "observe", "paper"]


class StrategySettingsBase(BaseModel):
    """Common strategy switches.

    ``mode``: ``off`` (not loaded), ``observe`` (scanned, evaluated and recorded - including
    a per-strategy *shadow* ledger - but never allocated main paper capital) or ``paper``
    (eligible for the main paper account through the central allocator). Nothing here can
    place a real order. The legacy ``enabled = false`` key maps to ``mode = "off"``.
    """

    mode: StrategyMode = "observe"
    min_edge: Money = Decimal("0.02")  # $ per contract after fees, conservative probability
    max_spread: Money = Decimal("0.04")
    limit_slippage: Money = Decimal("0.01")

    @model_validator(mode="before")
    @classmethod
    def _legacy_enabled(cls, data: Any) -> Any:
        if isinstance(data, dict) and "enabled" in data:
            data = dict(data)
            enabled = data.pop("enabled")
            if not enabled:
                data["mode"] = "off"
        return data

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


class LateMatchStrategySettings(StrategySettingsBase):
    """Baseline 2: the original late-match favourite rules, preserved unchanged."""

    mode: StrategyMode = "observe"
    min_price: Money = Decimal("0.80")
    max_price: Money = Decimal("0.97")
    min_model_prob: float = 0.90
    min_edge: Money = Decimal("0.02")
    max_games_from_victory: int = 3
    require_deciding_set: bool = False
    max_spread: Money = Decimal("0.04")
    limit_slippage: Money = Decimal("0.01")


class PriceThresholdStrategySettings(StrategySettingsBase):
    """Baseline 1: buy the favourite whenever its ask is inside a band (no model)."""

    mode: StrategyMode = "observe"
    min_price: Money = Decimal("0.90")
    max_price: Money = Decimal("0.97")
    max_spread: Money = Decimal("0.03")
    limit_slippage: Money = Decimal("0.00")
    require_live_match: bool = True


TennisStage = Literal["prematch", "early", "mid", "late"]


class ProbabilityCalibrationSettings(BaseModel):
    """Recalibration of the tennis model's win probability before it is traded on:

        logit(p) = slope * logit(p_model) + market_weight * logit(p_market)

    ``p_market`` is the live YES mid of the backed player's market. slope = 1 and
    market_weight = 0 is the raw model. The form has no intercept, so the two players'
    probabilities still sum to one. Values come from ``kalshi-trader research calibrate``
    (or the dashboard's Research tab), fitted on settled forward predictions and checked on
    later, held-out matches; do not set them by hand.
    """

    slope: float = Field(default=1.0, ge=0.0, le=5.0)
    market_weight: float = Field(default=0.0, ge=0.0, le=2.0)

    @property
    def is_identity(self) -> bool:
        return self.slope == 1.0 and self.market_weight == 0.0


class TennisModelStrategySettings(StrategySettingsBase):
    """Strategy A/C: statistical mispricing with the tennis Markov model, any match stage.

    Stages: ``early`` = first set, ``late`` = either player within
    ``late_games_from_victory`` games of winning, ``mid`` = otherwise. ``prematch`` is
    accepted but always reported unsupported: the only pre-match prior available is the
    market price itself, which cannot be evidence that the market is mispriced.
    """

    mode: StrategyMode = "paper"
    stages: list[TennisStage] = Field(default_factory=lambda: ["early", "mid", "late"])
    tours: list[str] = Field(default_factory=list)  # empty = all tours
    min_price: Money = Decimal("0.05")
    max_price: Money = Decimal("0.95")
    min_edge: Money = Decimal("0.03")
    max_spread: Money = Decimal("0.04")
    limit_slippage: Money = Decimal("0.02")
    late_games_from_victory: int = 3
    # Without a pre-match price the prior is a tour average (50/50 strengths), which would
    # flag nearly every underdog as cheap. Require the anchored prior by default.
    require_pre_match_prior: bool = True
    min_effective_sample_size: float = 50.0
    calibration: ProbabilityCalibrationSettings = Field(
        default_factory=ProbabilityCalibrationSettings
    )
    # None = hold to settlement. Otherwise sell into the bid once the bid after the taker fee
    # exceeds the model's point probability for the held side by at least this much.
    exit_edge: Money | None = None
    # Optional structural cohort filter, not a fitted probability adjustment. None preserves
    # existing versions (config hashes exclude None). The comeback filter only backs the
    # pre-match favourite while it is down one set in set two of a best-of-three match.
    entry_pattern: Literal["first_set_comeback"] | None = None
    # None = no cap. Otherwise skip an option when the model's point probability exceeds its
    # ask by more than this: a large gap against a liquid market is more often model error.
    max_model_gap: Money | None = None


class CrossMarketStrategySettings(StrategySettingsBase):
    """Strategy B: baskets whose combined payoff is fixed by verifiable settlement rules."""

    mode: StrategyMode = "paper"
    min_edge: Money = Decimal("0.01")  # $ per basket unit after all fees
    include_tennis_pairs: bool = True
    # Extra (non-tennis) series to scan for related markets, e.g. ["KXHIGHNY"].
    series_tickers: list[str] = Field(default_factory=list)
    refresh_interval_s: float = 300.0
    max_legs: int = 12
    max_book_age_s: float = 5.0
    # An apparent mispricing must persist this long / this many scans before it is eligible
    # (one-snapshot crossings are usually stale or racing books).
    min_persistence_s: float = 2.0
    min_persistence_scans: int = 2
    max_spread: Money = Decimal("1")  # spreads are already priced into executable asks
    max_units: int = 50


class MarketMakingStrategySettings(StrategySettingsBase):
    """Strategy D: RESEARCH ONLY. Hypothetical quotes and conservative simulated maker fills.

    Fill rule ``trade_through``: a resting bid fills only once the opposite best ask
    trades *through* it (strictly better than our price), so queue position cannot be
    assumed. This is pessimistic about fill rates and inherently captures adverse selection.
    """

    mode: StrategyMode = "observe"
    min_spread: Money = Decimal("0.04")
    quote_size: Money = Decimal("10")
    max_inventory: Money = Decimal("50")  # net contracts per market
    min_price: Money = Decimal("0.10")
    max_price: Money = Decimal("0.90")
    adverse_horizon_s: float = 60.0
    max_book_age_s: float = 10.0

    @model_validator(mode="after")
    def _research_only(self) -> MarketMakingStrategySettings:
        if self.mode == "paper":
            raise ValueError(
                "market_making supports only 'off' or 'observe': maker fills cannot be "
                "simulated credibly enough to allocate paper capital yet"
            )
        return self


class StrategiesSettings(BaseModel):
    tennis_model: TennisModelStrategySettings = Field(default_factory=TennisModelStrategySettings)
    late_match: LateMatchStrategySettings = Field(default_factory=LateMatchStrategySettings)
    price_threshold: PriceThresholdStrategySettings = Field(
        default_factory=PriceThresholdStrategySettings
    )
    cross_market: CrossMarketStrategySettings = Field(default_factory=CrossMarketStrategySettings)
    market_making: MarketMakingStrategySettings = Field(
        default_factory=MarketMakingStrategySettings
    )
    # Observe-only copies of tennis_model with some settings overridden, keyed by a short
    # name; each runs as strategy ``tennis_model_<key>`` with its own shadow ledger.
    tennis_model_variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evaluation_interval_s: float = 1.0
    # Every strategy that is not "off" also trades a standalone shadow ledger so all
    # strategies can be compared on identical observations with identical risk rules.
    shadow_ledgers: bool = True
    entry_cooldown_s: float = 30.0

    @model_validator(mode="after")
    def _check_variants(self) -> StrategiesSettings:
        for key in self.tennis_model_variants:
            if not re.fullmatch(r"[a-z0-9_]{1,40}", key):
                raise ValueError(f"tennis_model variant name {key!r}: use a-z, 0-9 and _")
        self.tennis_variants()  # surfaces bad override values at load time
        return self

    def tennis_variants(self) -> dict[str, TennisModelStrategySettings]:
        """``tennis_model_<key>`` -> tennis_model's settings with the overrides applied.
        Variants never get main-account capital: their mode is always ``observe``."""
        base = self.tennis_model.model_dump()
        return {
            f"tennis_model_{key}": TennisModelStrategySettings.model_validate(
                {**base, **overrides, "mode": "observe"}
            )
            for key, overrides in self.tennis_model_variants.items()
        }


class RiskSettings(BaseModel):
    max_fraction_per_trade: Money = Decimal("0.02")  # max loss of one trade / equity
    max_position_cost: Money | None = None  # optional fixed $ ceiling; None = % of equity only
    # Daily loss / drawdown limits stop NEW entries. They are not a loss ceiling: positions
    # already open can still lose their full cost.
    max_daily_loss: Money = Decimal("50")
    max_drawdown_fraction: Money = Decimal("0.15")
    max_open_positions: int = 5
    max_aggregate_exposure_fraction: Money = Decimal("0.30")  # open max-loss / equity
    max_event_exposure_fraction: Money = Decimal("0.03")
    max_strategy_exposure_fraction: Money = Decimal("0.15")
    # Correlation group: tournament/competition for tennis, series for other markets.
    max_correlated_exposure_fraction: Money = Decimal("0.08")
    min_liquidity_contracts: Money = Decimal("10")
    max_spread: Money = Decimal("0.05")
    max_market_data_age_s: float = 10.0
    max_score_data_age_s: float = 30.0
    # Kalshi's live-data responses are CDN-cached (max-age 10 s): a locally fresh response
    # can describe a score this old. Added to every score age.
    score_source_staleness_s: float = 10.0
    # A live score that has not changed for this long is treated as possibly frozen.
    max_score_unchanged_s: float = 300.0
    # If the market mid moved this much since the last score change, the score is suspect.
    max_mid_move_since_score_change: Money = Decimal("0.10")
    max_trades_per_market: int = 1
    max_trades_per_event: int = 1
    # Fractional Kelly multiplier; sizing never depends on past losses (no martingale).
    kelly_fraction: Money = Decimal("0.10")
    # Strategies without validated out-of-sample evidence get this share of normal size.
    unvalidated_size_multiplier: Money = Decimal("0.25")
    max_consecutive_errors: int = 5


class EvidenceSettings(BaseModel):
    """Predefined evidence a strategy needs before it is classified as validated."""

    min_out_of_sample_trades: int = 50
    min_out_of_sample_events: int = 30
    max_calibration_pvalue: float = 0.05
    require_positive_ci: bool = True  # cluster-bootstrap CI of mean P&L excludes zero
    max_drawdown_fraction: float = 0.20


class FeeSettings(BaseModel):
    # Kalshi general trading fee: ceil_to_cent(rate * multiplier * C * P * (1 - P)).
    taker_rate: Money = Decimal("0.07")
    maker_rate: Money = Decimal("0.0175")


class PaperSettings(BaseModel):
    starting_balance: Money = Decimal("1000")
    # Only this share of displayed depth is assumed fillable (others compete for it).
    displayed_liquidity_fraction: Money = Decimal("0.5")
    # Multi-leg baskets: legs after the first may only take this share of the (already
    # haircut) displayed depth, because books move while earlier legs execute. Within one
    # book snapshot a later leg could otherwise never fail, hiding legging risk.
    basket_later_leg_liquidity_fraction: Money = Decimal("0.5")
    enable_exits: bool = False
    take_profit_bid: Money = Decimal("0.99")


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    refresh_interval_ms: int = 2000
    # Networks besides this machine allowed to use pause/resume/emergency stop (e.g. the
    # tailnet when the dashboard is shared with `tailscale serve`).
    control_networks: list[IPvAnyNetwork] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KT_",
        env_nested_delimiter="__",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_path: Path = PROJECT_ROOT / "data" / "kalshi_trader.db"
    log_dir: Path = PROJECT_ROOT / "data" / "logs"
    log_level: str = "INFO"

    kalshi: KalshiSettings = Field(default_factory=KalshiSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    tennis: TennisSettings = Field(default_factory=TennisSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    strategies: StrategiesSettings = Field(default_factory=StrategiesSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    evidence: EvidenceSettings = Field(default_factory=EvidenceSettings)
    fees: FeeSettings = Field(default_factory=FeeSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        if not self.db_path.is_absolute():
            self.db_path = PROJECT_ROOT / self.db_path
        if not self.log_dir.is_absolute():
            self.log_dir = PROJECT_ROOT / self.log_dir
        key_path = self.kalshi.private_key_path
        if key_path is not None:
            key_path = key_path.expanduser()
            self.kalshi.private_key_path = (
                key_path if key_path.is_absolute() else (PROJECT_ROOT / key_path)
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_path = Path(os.environ.get("KT_CONFIG_PATH", DEFAULT_CONFIG_PATH))
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if config_path.is_file():
            sources.append(_TomlWithoutSecrets(settings_cls, toml_file=config_path))
        return tuple(sources)


class _TomlWithoutSecrets(TomlConfigSettingsSource):
    """TOML source that refuses credentials so secrets never live in tracked config files."""

    def __call__(self) -> dict[str, Any]:
        data = super().__call__()
        kalshi = data.get("kalshi")
        if isinstance(kalshi, dict):
            for secret_key in ("api_key_id", "private_key_path"):
                if secret_key in kalshi:
                    raise ValueError(
                        f"kalshi.{secret_key} must be set via environment/.env, not the TOML file"
                    )
        return data


def load_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
