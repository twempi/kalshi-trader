"""Normalized tennis match model shared by all score providers.

``ScoreState`` is a complete, provider-independent scoring state (player indices 0/1) with a
point-by-point state machine. It is what the probability model consumes. ``MatchState``
wraps it with status, identities, statistics and validation results from a provider.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Literal

FinalSetRule = Literal["tiebreak", "advantage", "match_tiebreak"]

GRAND_SLAM_KEYWORDS = ("australian open", "french open", "roland garros", "wimbledon", "us open")


class MatchStatus(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    SUSPENDED = "suspended"
    FINISHED = "finished"
    RETIRED = "retired"
    WALKOVER = "walkover"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_over(self) -> bool:
        return self in (
            MatchStatus.FINISHED,
            MatchStatus.RETIRED,
            MatchStatus.WALKOVER,
            MatchStatus.CANCELLED,
        )


@dataclass(frozen=True)
class MatchFormat:
    best_of: int = 3
    games_per_set: int = 6
    tiebreak_points: int = 7
    final_set: FinalSetRule = "tiebreak"
    final_set_tiebreak_points: int = 7
    # False when the format could not be determined reliably; strategies should skip.
    known: bool = True

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    def label(self) -> str:
        final = {
            "tiebreak": f"final-set TB{self.final_set_tiebreak_points}",
            "advantage": "advantage final set",
            "match_tiebreak": f"match TB{self.final_set_tiebreak_points}",
        }[self.final_set]
        return f"Bo{self.best_of}, {final}" + ("" if self.known else " (format uncertain)")


def is_grand_slam(tournament_name: str | None) -> bool:
    name = (tournament_name or "").lower()
    return any(keyword in name for keyword in GRAND_SLAM_KEYWORDS)


def infer_format(
    *,
    best_of: int | str | None,
    gender: str | None,
    tournament_name: str | None,
    last_set_scoring_type: str | None,
) -> MatchFormat:
    slam = is_grand_slam(tournament_name)
    known = True
    parsed_best_of: int | None = None
    if best_of not in (None, ""):
        try:
            parsed_best_of = int(str(best_of))
        except ValueError:
            parsed_best_of = None
            known = False
    if parsed_best_of not in (3, 5):
        if parsed_best_of is not None:
            known = False
        parsed_best_of = 5 if slam and (gender or "").lower() == "men" else 3

    final_set: FinalSetRule = "tiebreak"
    final_points = 7
    scoring = (last_set_scoring_type or "").lower()
    if not scoring:
        if slam:
            final_points = 10  # all four majors play a 10-point tiebreak at 6-6 since 2022
    elif scoring == "standardtiebreak":
        final_points = 7
    elif "match" in scoring or "super" in scoring or "champions" in scoring:
        final_set, final_points = "match_tiebreak", 10
    elif "advantage" in scoring:
        final_set = "advantage"
    elif "10" in scoring or "ten" in scoring:
        final_points = 10
    else:
        known = False
    return MatchFormat(
        best_of=parsed_best_of,
        final_set=final_set,
        final_set_tiebreak_points=final_points,
        known=known,
    )


def tiebreak_server(first_server: int, points_played: int) -> int:
    """Server of point number `points_played` (0-based) in a tiebreak.

    The first server serves one point, then players alternate every two points.
    """
    return first_server if ((points_played + 1) // 2) % 2 == 0 else 1 - first_server


@dataclass(frozen=True)
class ScoreState:
    """Scoring state. Indices: 0 = player 1, 1 = player 2.

    ``points`` are raw point counts in the current game (deuce is normalised to 3-3 and
    advantage to 4-3) or tiebreak point counts when ``in_tiebreak``. ``server`` is the
    player serving the *next* point.
    """

    fmt: MatchFormat
    sets: tuple[int, int] = (0, 0)
    games: tuple[int, int] = (0, 0)
    points: tuple[int, int] = (0, 0)
    server: int = 0
    in_tiebreak: bool = False
    tiebreak_first_server: int | None = None
    completed_sets: tuple[tuple[int, int], ...] = ()

    # ------------------------------------------------------------ properties
    @property
    def winner(self) -> int | None:
        k = self.fmt.sets_to_win
        if self.sets[0] >= k:
            return 0
        if self.sets[1] >= k:
            return 1
        return None

    @property
    def is_final_set(self) -> bool:
        k = self.fmt.sets_to_win
        return self.sets[0] == k - 1 and self.sets[1] == k - 1

    @property
    def tiebreak_target(self) -> int:
        return self.fmt.final_set_tiebreak_points if self.is_final_set else self.fmt.tiebreak_points

    def set_uses_tiebreak(self) -> bool:
        return not (self.is_final_set and self.fmt.final_set == "advantage")

    # --------------------------------------------------------- transitions
    def apply_point(self, point_winner: int) -> ScoreState:
        if point_winner not in (0, 1):
            raise ValueError("point winner must be 0 or 1")
        if self.winner is not None:
            return self
        if self.in_tiebreak:
            return self._tiebreak_point(point_winner)
        return self._game_point(point_winner)

    def _game_point(self, w: int) -> ScoreState:
        pts = [self.points[0], self.points[1]]
        pts[w] += 1
        if pts[w] >= 4 and pts[w] - pts[1 - w] >= 2:
            return self._game_won(w)
        if pts[0] >= 3 and pts[1] >= 3:  # normalise deuce / advantage
            lead = pts[0] - pts[1]
            pts = [3, 3] if lead == 0 else ([4, 3] if lead > 0 else [3, 4])
        return replace(self, points=(pts[0], pts[1]))

    def _game_won(self, w: int) -> ScoreState:
        games = [self.games[0], self.games[1]]
        games[w] += 1
        next_server = 1 - self.server
        n = self.fmt.games_per_set
        if games[w] >= n and games[w] - games[1 - w] >= 2:
            return self._set_won(w, (games[0], games[1]), next_server)
        if games[0] == n and games[1] == n and self.set_uses_tiebreak():
            return replace(
                self,
                games=(games[0], games[1]),
                points=(0, 0),
                server=next_server,
                in_tiebreak=True,
                tiebreak_first_server=next_server,
            )
        return replace(self, games=(games[0], games[1]), points=(0, 0), server=next_server)

    def _tiebreak_point(self, w: int) -> ScoreState:
        pts = [self.points[0], self.points[1]]
        pts[w] += 1
        first = (
            self.tiebreak_first_server if self.tiebreak_first_server is not None else self.server
        )
        target = self.tiebreak_target
        if pts[w] >= target and pts[w] - pts[1 - w] >= 2:
            games = [self.games[0], self.games[1]]
            games[w] += 1
            # The player who received first in the tiebreak serves first in the next set.
            return self._set_won(w, (games[0], games[1]), 1 - first)
        return replace(
            self, points=(pts[0], pts[1]), server=tiebreak_server(first, pts[0] + pts[1])
        )

    def _set_won(self, w: int, final_games: tuple[int, int], next_server: int) -> ScoreState:
        sets = [self.sets[0], self.sets[1]]
        sets[w] += 1
        state = ScoreState(
            fmt=self.fmt,
            sets=(sets[0], sets[1]),
            games=(0, 0),
            points=(0, 0),
            server=next_server,
            completed_sets=(*self.completed_sets, final_games),
        )
        if state.winner is None and state.is_final_set and self.fmt.final_set == "match_tiebreak":
            # Deciding set replaced by a single tiebreak (e.g. first to 10).
            state = replace(state, in_tiebreak=True, tiebreak_first_server=next_server)
        return state

    # -------------------------------------------------------------- helpers
    def reachable_states(self, max_points: int) -> list[ScoreState]:
        """All distinct states reachable within `max_points` further points (incl. self)."""
        frontier = [self]
        seen = {self}
        for _ in range(max_points):
            nxt = []
            for state in frontier:
                for w in (0, 1):
                    child = state.apply_point(w)
                    if child not in seen:
                        seen.add(child)
                        nxt.append(child)
            frontier = nxt
        return list(seen)

    def points_from_victory(self, player: int, limit: int = 40) -> int | None:
        """Minimum number of consecutive points `player` needs to win the match."""
        state = self
        for n in range(limit + 1):
            if state.winner == player:
                return n
            if state.winner is not None:
                return None
            state = state.apply_point(player)
        return None

    def games_from_victory(self, player: int, limit: int = 30) -> int | None:
        """Minimum number of games (a tiebreak counts as one) `player` needs to win."""
        state = self
        for n in range(limit + 1):
            if state.winner == player:
                return n
            if state.winner is not None:
                return None
            start = (state.sets, state.games, state.in_tiebreak)
            # Win points until the game (or tiebreak) changes hands.
            for _ in range(30):
                state = state.apply_point(player)
                if (state.sets, state.games, state.in_tiebreak) != start:
                    break
        return None

    def has_match_point(self, player: int) -> bool:
        return self.winner is None and self.apply_point(player).winner == player

    def display_points(self) -> tuple[str, str]:
        if self.in_tiebreak:
            return str(self.points[0]), str(self.points[1])
        labels = ("0", "15", "30", "40")
        a, b = self.points
        if a >= 3 and b >= 3:
            if a == b:
                return "40", "40"
            return ("AD", "40") if a > b else ("40", "AD")
        return labels[min(a, 3)], labels[min(b, 3)]

    def score_text(self) -> str:
        sets = " ".join(f"{a}-{b}" for a, b in self.completed_sets)
        if self.winner is not None:
            return sets
        pa, pb = self.display_points()
        cur = f"{self.games[0]}-{self.games[1]}"
        tb = " TB" if self.in_tiebreak else ""
        serve = "P1" if self.server == 0 else "P2"
        return f"{sets + ' ' if sets else ''}{cur}{tb} ({pa}-{pb}) {serve} srv".strip()


@dataclass(frozen=True)
class ServeStats:
    points_won: int
    points_lost: int

    @property
    def total(self) -> int:
        return self.points_won + self.points_lost


@dataclass(frozen=True)
class MatchState:
    provider: str
    match_id: str
    player_ids: tuple[str | None, str | None]
    status: MatchStatus
    received_ts: float
    fmt: MatchFormat = field(default_factory=MatchFormat)
    score: ScoreState | None = None
    winner: int | None = None
    serve_stats: tuple[ServeStats | None, ServeStats | None] = (None, None)
    source_delayed: bool = False
    valid: bool = True
    issues: tuple[str, ...] = ()

    @property
    def is_live(self) -> bool:
        return self.status == MatchStatus.LIVE and self.score is not None and self.valid

    def score_text(self) -> str:
        if self.score is None:
            return self.status.value
        text = self.score.score_text()
        if self.status != MatchStatus.LIVE:
            text = f"{text} [{self.status.value}]"
        return text

    def fingerprint(self) -> str:
        """Hash of everything except the receive time, to detect changes cheaply."""
        payload = self.to_dict()
        payload.pop("received_ts", None)
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)
