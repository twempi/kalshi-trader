"""Tennis scores from Kalshi's own Live Data API (``GET /live_data/batch``).

Free and public, keyed by the same milestone ids that link to the markets, so no
fragile name matching is needed. Observed payload (type ``tennis_tournament_singles``):
per-set ``competitorN_round_scores`` (outcome winner/loser/ongoing, games, optional
tiebreak_score), ``competitorN_current_round_score`` (0/15/30/40 in a game, point counts in
a tiebreak), ``advantage``, ``server`` (competitor id), ``status``/``match_status``,
``winner`` and per-player serve statistics. Responses are CDN-cached (max-age 10 s) and
Kalshi documents that the feed can lag, so consumers must treat it as delayed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from kalshi_trader.api.rest import KalshiRestClient
from kalshi_trader.logging_setup import get_logger
from kalshi_trader.tennis.providers.base import MatchRef, ProviderResult, ScoreProvider
from kalshi_trader.tennis.state import (
    MatchFormat,
    MatchState,
    MatchStatus,
    ScoreState,
    ServeStats,
    infer_format,
    tiebreak_server,
)

log = get_logger("tennis.kalshi_live")

TENNIS_LIVE_TYPE = "tennis_tournament_singles"
GAME_POINTS = {0: 0, 15: 1, 30: 2, 40: 3}
ADVANTAGE_TOKENS = {"A", "AD", "ADV", "50"}


def _status(details: dict[str, Any], has_winner: bool) -> MatchStatus:
    status = str(details.get("status") or "").lower()
    match_status = str(details.get("match_status") or "").lower()
    stoppage = str(details.get("stoppage_status") or "").lower()
    if status == "cancelled" or match_status in ("cancelled", "canceled"):
        return MatchStatus.CANCELLED
    if "walkover" in match_status:
        return MatchStatus.WALKOVER
    if "retired" in match_status or "retire" in status:
        return MatchStatus.RETIRED
    if (
        status in ("closed", "ended", "finished", "complete", "completed")
        or match_status == "ended"
    ):
        return MatchStatus.FINISHED if has_winner else MatchStatus.UNKNOWN
    if (
        status in ("interrupted", "suspended")
        or match_status in ("suspended", "interrupted")
        or stoppage in ("suspended", "interrupted")
    ):
        return MatchStatus.SUSPENDED
    if status in ("live", "started", "inprogress", "in_progress"):
        return MatchStatus.LIVE
    if status in ("not_started", "scheduled", "") and not has_winner:
        return MatchStatus.SCHEDULED
    return MatchStatus.UNKNOWN


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _serve_stats(stats: Any) -> ServeStats | None:
    if not isinstance(stats, dict):
        return None
    won, lost = _int(stats.get("service_points_won")), _int(stats.get("service_points_lost"))
    if won is None or lost is None or won < 0 or lost < 0:
        return None
    return ServeStats(won, lost)


def parse_kalshi_live(details: dict[str, Any], ref: MatchRef, received_ts: float) -> MatchState:
    issues: list[str] = []
    c1, c2 = details.get("competitor1_id") or None, details.get("competitor2_id") or None
    p1, p2 = ref.player_ids
    if p1 is None and p2 is None:
        p1, p2 = c1, c2
    if (c1, c2) == (p1, p2):
        swap = False
    elif (c1, c2) == (p2, p1):
        swap = True
    else:
        return MatchState(
            provider="kalshi_live_data",
            match_id=ref.match_id,
            player_ids=(p1, p2),
            status=MatchStatus.UNKNOWN,
            received_ts=received_ts,
            valid=False,
            issues=("competitor ids do not match the market's players",),
        )

    def ordered(a: Any, b: Any) -> tuple[Any, Any]:
        return (b, a) if swap else (a, b)

    def index_of(competitor_id: Any) -> int | None:
        if not competitor_id:
            return None
        if competitor_id == p1:
            return 0
        if competitor_id == p2:
            return 1
        return None

    winner_raw = details.get("winner") or None
    winner = index_of(winner_raw)
    if winner_raw and winner is None:
        issues.append("winner id not recognised")
    status = _status(details, has_winner=winner is not None)
    fmt = infer_format(
        best_of=ref.best_of,
        gender=ref.gender,
        tournament_name=ref.tournament_name,
        last_set_scoring_type=details.get("last_set_scoring_type"),
    )
    stats = ordered(
        _serve_stats(details.get("competitor1_statistics")),
        _serve_stats(details.get("competitor2_statistics")),
    )
    delayed = bool(details.get("live_data_delayed"))

    base = dict(
        provider="kalshi_live_data",
        match_id=ref.match_id,
        player_ids=(p1, p2),
        received_ts=received_ts,
        fmt=fmt,
        winner=winner,
        serve_stats=stats,
        source_delayed=delayed,
    )

    if status in (MatchStatus.SCHEDULED, MatchStatus.CANCELLED):
        return MatchState(status=status, issues=tuple(issues), **base)

    score, score_issues = _parse_score(details, fmt, ordered, index_of, status)
    issues.extend(score_issues)
    valid = not score_issues and not issues
    if status == MatchStatus.LIVE and score is not None and score.winner is not None:
        issues.append("feed says live but set count shows a finished match")
        valid = False
    return MatchState(status=status, score=score, valid=valid, issues=tuple(issues), **base)


def _parse_score(
    details: dict[str, Any], fmt: MatchFormat, ordered: Any, index_of: Any, status: MatchStatus
) -> tuple[ScoreState | None, list[str]]:
    issues: list[str] = []
    r1 = details.get("competitor1_round_scores") or []
    r2 = details.get("competitor2_round_scores") or []
    if not isinstance(r1, list) or not isinstance(r2, list) or len(r1) != len(r2):
        return None, ["set score arrays malformed"]
    rounds_a, rounds_b = ordered(r1, r2)

    completed: list[tuple[int, int]] = []
    sets = [0, 0]
    current = (0, 0)
    for i, (ra, rb) in enumerate(zip(rounds_a, rounds_b, strict=True)):
        if not isinstance(ra, dict) or not isinstance(rb, dict):
            return None, ["set entry malformed"]
        ga, gb = _int(ra.get("score")), _int(rb.get("score"))
        if ga is None or gb is None or not (0 <= ga <= 50 and 0 <= gb <= 50):
            return None, [f"set {i + 1} games invalid"]
        oa, ob = str(ra.get("outcome") or ""), str(rb.get("outcome") or "")
        if oa == "ongoing" or ob == "ongoing":
            if i != len(rounds_a) - 1:
                issues.append("ongoing set is not the last set")
            current = (ga, gb)
            continue
        if {oa, ob} != {"winner", "loser"}:
            issues.append(f"set {i + 1} outcome unclear ({oa}/{ob})")
            continue
        w = 0 if oa == "winner" else 1
        if (ga > gb) != (w == 0) and fmt.final_set != "match_tiebreak":
            issues.append(f"set {i + 1} winner inconsistent with games")
        sets[w] += 1
        completed.append((ga, gb))

    overall = ordered(
        _int(details.get("competitor1_overall_score")),
        _int(details.get("competitor2_overall_score")),
    )
    if None not in overall and tuple(overall) != (sets[0], sets[1]) and status == MatchStatus.LIVE:
        issues.append(f"sets won {tuple(sets)} disagree with overall score {tuple(overall)}")

    in_final = sets[0] == fmt.sets_to_win - 1 and sets[1] == fmt.sets_to_win - 1
    match_tb = in_final and fmt.final_set == "match_tiebreak"
    n = fmt.games_per_set
    in_tiebreak = match_tb or (
        current == (n, n) and not (in_final and fmt.final_set == "advantage")
    )

    raw_a, raw_b = ordered(
        details.get("competitor1_current_round_score"),
        details.get("competitor2_current_round_score"),
    )
    points: tuple[int, int] | None = None
    if in_tiebreak:
        pa, pb = _int(raw_a), _int(raw_b)
        if pa is None or pb is None or pa < 0 or pb < 0 or pa > 60 or pb > 60:
            issues.append("tiebreak points invalid")
        else:
            points = (pa, pb)
    else:
        points = _game_points(raw_a, raw_b, details.get("advantage"), index_of, issues)

    server = index_of(details.get("server"))
    if status == MatchStatus.LIVE and server is None:
        issues.append("server unknown")
    if points is None:
        return None, issues

    tb_first: int | None = None
    if in_tiebreak and server is not None:
        n_played = points[0] + points[1]
        tb_first = next(f for f in (0, 1) if tiebreak_server(f, n_played) == server)

    state = ScoreState(
        fmt=fmt,
        sets=(sets[0], sets[1]),
        games=current,
        points=points,
        server=server if server is not None else 0,
        in_tiebreak=in_tiebreak,
        tiebreak_first_server=tb_first,
        completed_sets=tuple(completed),
    )
    return state, issues


def _game_points(
    raw_a: Any, raw_b: Any, advantage: Any, index_of: Any, issues: list[str]
) -> tuple[int, int] | None:
    def one(raw: Any) -> int | str | None:
        if isinstance(raw, str) and raw.strip().upper() in ADVANTAGE_TOKENS:
            return "A"
        value = _int(raw)
        if value is None:
            return None
        if value == 50:
            return "A"
        return GAME_POINTS.get(value)

    a, b = one(raw_a), one(raw_b)
    if a is None or b is None:
        issues.append(f"unrecognised point score {raw_a!r}-{raw_b!r}")
        return None
    if a == "A" or b == "A":
        return (4, 3) if a == "A" else (3, 4)
    pa, pb = int(a), int(b)
    if (pa, pb) == (3, 3) and advantage:
        holder = index_of(advantage)
        if holder is None:
            issues.append(f"advantage value not recognised: {str(advantage)[:40]!r}")
            return None
        return (4, 3) if holder == 0 else (3, 4)
    return pa, pb


class KalshiLiveDataProvider(ScoreProvider):
    name = "kalshi_live_data"

    def __init__(self, rest: KalshiRestClient) -> None:
        self._rest = rest

    async def fetch(self, refs: Sequence[MatchRef]) -> dict[str, ProviderResult]:
        if not refs:
            return {}
        by_id = {ref.match_id: ref for ref in refs}
        live = await self._rest.get_live_data_batch(list(by_id))
        received = time.time()
        results: dict[str, ProviderResult] = {}
        for item in live:
            ref = by_id.get(item.milestone_id)
            if ref is None:
                continue
            if item.type != TENNIS_LIVE_TYPE:
                log.warning(
                    "unexpected_live_data_type", milestone_id=item.milestone_id, type=item.type
                )
                continue
            try:
                state = parse_kalshi_live(item.details, ref, received)
            except Exception as exc:  # untrusted input: never let one record kill the loop
                log.warning(
                    "live_data_parse_failed", milestone_id=item.milestone_id, error=repr(exc)[:300]
                )
                continue
            results[ref.match_id] = ProviderResult(state=state, raw=item.details)
        return results
