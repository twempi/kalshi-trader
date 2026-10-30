"""Application configuration.

Precedence (highest first): explicit kwargs > environment variables > ``.env`` >
TOML config file (``config/settings.toml`` or ``$KT_CONFIG_PATH``) > defaults.

Nested values can be overridden from the environment with ``__`` as delimiter,
e.g. ``KT_RISK__MAX_OPEN_POSITIONS=3``. Secrets (API key id / private key) are
only ever read from the environment or ``.env``, never from the TOML file.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, model_validator
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


class LateMatchStrategySettings(BaseModel):
    enabled: bool = True
    min_price: Money = Decimal("0.80")
    max_price: Money = Decimal("0.97")
    min_model_prob: float = 0.90
    # Minimum EV per contract (dollars) after fees, using the conservative probability.
    min_edge: Money = Decimal("0.02")
    max_games_from_victory: int = 3
    require_deciding_set: bool = False
    max_spread: Money = Decimal("0.04")
    # Extra price tolerance over the best ask for the simulated IOC limit order.
    limit_slippage: Money = Decimal("0.01")


class PriceThresholdStrategySettings(BaseModel):
    """Score-agnostic baseline: buy the favourite whenever its ask is inside a band."""

    enabled: bool = False
    min_price: Money = Decimal("0.90")
    max_price: Money = Decimal("0.97")
    max_spread: Money = Decimal("0.03")
    limit_slippage: Money = Decimal("0.00")
    require_live_match: bool = True


class StrategiesSettings(BaseModel):
    late_match: LateMatchStrategySettings = Field(default_factory=LateMatchStrategySettings)
    price_threshold: PriceThresholdStrategySettings = Field(
        default_factory=PriceThresholdStrategySettings
    )
    evaluation_interval_s: float = 1.0


class RiskSettings(BaseModel):
    max_fraction_per_trade: Money = Decimal("0.02")
    max_position_cost: Money = Decimal("25")
    max_daily_loss: Money = Decimal("50")
    max_drawdown_fraction: Money = Decimal("0.15")
    max_open_positions: int = 5
    min_liquidity_contracts: Money = Decimal("10")
    max_spread: Money = Decimal("0.05")
    max_market_data_age_s: float = 10.0
    max_score_data_age_s: float = 30.0
    max_trades_per_market: int = 1
    max_trades_per_event: int = 1
    # Fractional Kelly multiplier; sizing never depends on past losses (no martingale).
    kelly_fraction: Money = Decimal("0.10")
    max_consecutive_errors: int = 5


class FeeSettings(BaseModel):
    # Kalshi general trading fee: ceil_to_cent(rate * multiplier * C * P * (1 - P)).
    taker_rate: Money = Decimal("0.07")
    maker_rate: Money = Decimal("0.0175")


class PaperSettings(BaseModel):
    starting_balance: Money = Decimal("1000")
    # Only this share of displayed depth is assumed fillable (others compete for it).
    displayed_liquidity_fraction: Money = Decimal("0.5")
    enable_exits: bool = False
    take_profit_bid: Money = Decimal("0.99")


class DashboardSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    refresh_interval_ms: int = 2000


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
