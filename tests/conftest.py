"""Shared fixtures.

Files in tests/fixtures/ are REAL Kalshi API responses captured on 2026-10-29 and used only
to test parsing. Everything else in the tests is synthetic test data, built in code and
clearly unrelated to any trading result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from kalshi_trader.config import Settings
from kalshi_trader.database.db import Database

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never read the developer's .env or config/settings.toml during tests."""
    for key in list(os.environ):
        if key.startswith("KT_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KT_CONFIG_PATH", str(tmp_path / "no-such-config.toml"))


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, db_path=tmp_path / "test.db", log_dir=tmp_path / "logs")


@pytest.fixture
def memory_db() -> Database:
    return Database(":memory:")
