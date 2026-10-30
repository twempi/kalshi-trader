"""Configuration precedence and secret-handling rules."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kalshi_trader.config import DEFAULT_CONFIG_PATH, Settings


def test_defaults_are_conservative() -> None:
    s = Settings(_env_file=None)
    assert s.paper.starting_balance == Decimal("1000")
    assert s.risk.max_fraction_per_trade == Decimal("0.02")
    assert s.fees.taker_rate == Decimal("0.07") and str(s.fees.taker_rate) == "0.07"
    assert not s.kalshi.has_credentials
    assert s.kalshi.rest_url.startswith("https://external-api.kalshi.com")


def test_env_overrides_nested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KT_RISK__MAX_OPEN_POSITIONS", "2")
    monkeypatch.setenv("KT_PAPER__STARTING_BALANCE", "250.50")
    monkeypatch.setenv("KT_KALSHI__ENVIRONMENT", "demo")
    s = Settings(_env_file=None)
    assert s.risk.max_open_positions == 2
    assert s.paper.starting_balance == Decimal("250.50")
    assert "demo" in s.kalshi.rest_url and "demo" in s.kalshi.websocket_url


def test_toml_values_load_and_secrets_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "good.toml"
    good.write_text("[risk]\nmax_daily_loss = 12.5\n[strategies.late_match]\nmin_edge = 0.03\n")
    monkeypatch.setenv("KT_CONFIG_PATH", str(good))
    s = Settings(_env_file=None)
    assert s.risk.max_daily_loss == Decimal("12.5")
    assert s.strategies.late_match.min_edge == Decimal("0.03")
    bad = tmp_path / "bad.toml"
    bad.write_text('[kalshi]\napi_key_id = "should-not-be-here"\n')
    monkeypatch.setenv("KT_CONFIG_PATH", str(bad))
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_shipped_config_file_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KT_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))
    s = Settings(_env_file=None)
    assert s.strategies.late_match.enabled and not s.strategies.price_threshold.enabled
    assert "KXATPMATCH" in s.discovery.series_tickers
    assert [str(n) for n in s.dashboard.control_networks] == [
        "100.64.0.0/10",
        "fd7a:115c:a1e0::/48",
    ]
    assert s.dashboard.host == "127.0.0.1"  # never bound to the network directly


def test_secret_values_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KT_KALSHI__API_KEY_ID", "super-secret-id")
    s = Settings(_env_file=None)
    assert "super-secret-id" not in repr(s) and "super-secret-id" not in str(s.model_dump())


def test_logging_redacts_sensitive_keys() -> None:
    from kalshi_trader.logging_setup import redact_processor

    event = redact_processor(
        None,
        "info",
        {"event": "x", "KALSHI-ACCESS-SIGNATURE": "abc", "headers": {"api_key": "k", "ok": 1}},
    )
    assert event["KALSHI-ACCESS-SIGNATURE"] == "***REDACTED***"
    assert event["headers"] == {"api_key": "***REDACTED***", "ok": 1}
