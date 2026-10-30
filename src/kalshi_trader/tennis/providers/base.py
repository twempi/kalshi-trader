"""Score-provider interface. Implement this to add another live tennis data source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kalshi_trader.tennis.state import MatchState


@dataclass(frozen=True)
class MatchRef:
    """Identifies a match for a provider and fixes our player order (player 1 / player 2)."""

    match_id: str  # provider-specific id (Kalshi milestone id for the Kalshi provider)
    event_ticker: str
    player_ids: tuple[str | None, str | None]
    tournament_name: str | None = None
    gender: str | None = None
    best_of: int | str | None = None


@dataclass(frozen=True)
class ProviderResult:
    state: MatchState
    raw: dict[str, Any] = field(default_factory=dict)


class ScoreProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    async def fetch(self, refs: Sequence[MatchRef]) -> dict[str, ProviderResult]:
        """Return the latest state for each requested match id that the provider knows."""

    async def close(self) -> None:  # pragma: no cover - default no-op
        return None
