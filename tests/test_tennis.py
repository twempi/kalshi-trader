"""Tennis scoring state machine, format inference and the Kalshi live-data parser."""

from __future__ import annotations

from typing import Any

import pytest

from kalshi_trader.tennis.providers.base import MatchRef
from kalshi_trader.tennis.providers.kalshi_live import parse_kalshi_live
from kalshi_trader.tennis.state import (
    MatchFormat,
    MatchStatus,
    ScoreState,
    infer_format,
    tiebreak_server,
)
from tests.conftest import load_fixture

BO3 = MatchFormat(best_of=3)


def play(state: ScoreState, winners: str) -> ScoreState:
    for w in winners:
        state = state.apply_point(int(w))
    return state


def test_game_deuce_and_advantage() -> None:
    s = play(ScoreState(fmt=BO3), "000111")  # 40-40
    assert s.points == (3, 3) and s.display_points() == ("40", "40")
    s = s.apply_point(0)
    assert s.points == (4, 3) and s.display_points() == ("AD", "40")
    s = s.apply_point(1)
    assert s.points == (3, 3)
    s = play(s, "00")
    assert s.games == (1, 0) and s.points == (0, 0) and s.server == 1


def test_set_and_match_transitions() -> None:
    s = ScoreState(fmt=BO3)
    for _ in range(6):
        s = play(s, "0000")  # player 0 wins every point
    assert s.sets == (1, 0) and s.completed_sets == ((6, 0),) and s.games == (0, 0)
    for _ in range(6):
        s = play(s, "0000")
    assert s.winner == 0 and s.apply_point(1) is s


def test_tiebreak_serve_rotation_and_next_set_server() -> None:
    assert [tiebreak_server(0, n) for n in range(7)] == [0, 1, 1, 0, 0, 1, 1]
    s = ScoreState(fmt=BO3, games=(6, 5), server=1)
    s = play(s, "1111")  # player 1 holds -> 6-6 tiebreak, player 0 serves first
    assert s.in_tiebreak and s.tiebreak_first_server == 0 and s.server == 0
    s = s.apply_point(1)
    assert s.points == (0, 1) and s.server == 1
    s = play(s, "0000000")  # 7-1
    assert s.sets == (1, 0) and s.completed_sets == ((7, 6),)
    assert s.server == 1  # the player who received first in the TB serves the next set


def test_final_set_variants() -> None:
    adv = ScoreState(fmt=MatchFormat(final_set="advantage"), sets=(1, 1), games=(6, 6))
    assert not adv.in_tiebreak
    adv2 = play(adv, "0000")
    assert adv2.games == (7, 6) and adv2.winner is None
    tb10 = ScoreState(
        fmt=MatchFormat(final_set_tiebreak_points=10), sets=(1, 1), games=(6, 5), server=1
    )
    tb10 = play(tb10, "1111")
    assert tb10.in_tiebreak and tb10.tiebreak_target == 10
    tb10 = play(tb10, "0" * 7)
    assert tb10.winner is None and tb10.points[0] == 7
    mtb = ScoreState(
        fmt=MatchFormat(final_set="match_tiebreak", final_set_tiebreak_points=10),
        sets=(1, 0),
        games=(0, 5),
        server=1,
    )
    mtb = play(mtb, "1111")  # player 2 takes set 2 6-0 -> deciding match tiebreak
    assert mtb.sets == (1, 1) and mtb.in_tiebreak and mtb.tiebreak_target == 10
    mtb = play(mtb, "0" * 10)
    assert mtb.winner == 0 and mtb.completed_sets[-1] == (1, 0)


def test_reachable_states_and_victory_distance() -> None:
    s = ScoreState(fmt=BO3, sets=(1, 0), games=(5, 3), points=(3, 0), server=0)
    assert s.has_match_point(0) and not s.has_match_point(1)
    assert s.points_from_victory(0) == 1
    assert s.games_from_victory(0) == 1
    assert len(s.reachable_states(1)) == 3
    assert ScoreState(fmt=BO3).games_from_victory(0) == 12


def test_infer_format() -> None:
    assert infer_format(
        best_of=None, gender="men", tournament_name="Wimbledon", last_set_scoring_type=None
    ) == MatchFormat(best_of=5, final_set_tiebreak_points=10)
    f = infer_format(
        best_of="3",
        gender="women",
        tournament_name="WTA Tokyo",
        last_set_scoring_type="StandardTiebreak",
    )
    assert f.best_of == 3 and f.final_set == "tiebreak" and f.known
    assert (
        infer_format(
            best_of=None, gender=None, tournament_name=None, last_set_scoring_type="MatchTiebreak"
        ).final_set
        == "match_tiebreak"
    )
    assert not infer_format(
        best_of=None, gender=None, tournament_name=None, last_set_scoring_type="SomethingNew"
    ).known
    assert not infer_format(
        best_of="7", gender=None, tournament_name=None, last_set_scoring_type=None
    ).known


# ------------------------------------------------- parser on REAL captured payloads
def samples() -> dict[str, dict[str, Any]]:
    return {s["kind"]: s for s in load_fixture("live_data_samples.json")["samples"]}


def ref_for(sample: dict[str, Any], swap: bool = False) -> MatchRef:
    d = sample["milestone"]["details"]
    ids = (d.get("first_competitor_id"), d.get("second_competitor_id"))
    return MatchRef(
        match_id=sample["milestone"]["id"],
        event_ticker="EV",
        player_ids=(ids[1], ids[0]) if swap else ids,
        tournament_name=d.get("tournament_name"),
        gender=d.get("gender"),
        best_of=d.get("best_of"),
    )


def test_parse_live_sample() -> None:
    s = samples()["live"]
    state = parse_kalshi_live(s["live_data"]["details"], ref_for(s), 1.0)
    assert state.status == MatchStatus.LIVE and state.valid and state.is_live
    assert state.score is not None and state.score.games == (5, 5)
    assert state.score.points == (3, 3) and state.score.server == 0


def test_parse_started_and_tiebreak_samples() -> None:
    started = samples()["started"]
    state = parse_kalshi_live(started["live_data"]["details"], ref_for(started), 1.0)
    assert state.status == MatchStatus.LIVE and state.score.points == (3, 2)
    tb = samples()["tiebreak"]
    state = parse_kalshi_live(tb["live_data"]["details"], ref_for(tb), 1.0)
    assert state.score.in_tiebreak and state.score.points == (1, 4)
    assert state.score.server == 0
    # first server inferred from rotation: point 5 (0-based) is served by player 0
    assert tiebreak_server(state.score.tiebreak_first_server, 5) == 0


def test_parse_swapped_player_order() -> None:
    s = samples()["live"]
    state = parse_kalshi_live(s["live_data"]["details"], ref_for(s, swap=True), 1.0)
    assert state.valid and state.score.server == 1 and state.score.games == (5, 5)


def test_parse_finished_retired_scheduled() -> None:
    fin = samples()["finished"]
    state = parse_kalshi_live(fin["live_data"]["details"], ref_for(fin), 1.0)
    assert state.status == MatchStatus.FINISHED and state.winner == 1
    assert state.score.completed_sets == ((6, 7), (1, 6)) and state.score.winner == 1
    ret = samples()["retired"]
    state = parse_kalshi_live(ret["live_data"]["details"], ref_for(ret), 1.0)
    assert state.status == MatchStatus.RETIRED and state.winner == 1 and state.status.is_over
    sched = samples()["scheduled"]
    state = parse_kalshi_live(sched["live_data"]["details"], ref_for(sched), 1.0)
    assert state.status == MatchStatus.SCHEDULED and state.score is None


def test_parser_flags_untrusted_inconsistencies() -> None:
    s = samples()["live"]
    details = dict(s["live_data"]["details"])
    bad_points = {**details, "competitor1_current_round_score": 25}
    assert not parse_kalshi_live(bad_points, ref_for(s), 1.0).valid
    wrong_ids = MatchRef(match_id="m", event_ticker="EV", player_ids=("x", "y"))
    assert not parse_kalshi_live(details, wrong_ids, 1.0).valid
    no_server = {**details, "server": ""}
    assert not parse_kalshi_live(no_server, ref_for(s), 1.0).valid
    adv = {**details, "advantage": details["competitor2_id"]}
    assert parse_kalshi_live(adv, ref_for(s), 1.0).score.points == (3, 4)
    adv_bad = {**details, "advantage": "someone-else"}
    assert not parse_kalshi_live(adv_bad, ref_for(s), 1.0).valid
    garbage_sets = {**details, "competitor1_round_scores": [{"score": "x", "outcome": "ongoing"}]}
    assert not parse_kalshi_live(garbage_sets, ref_for(s), 1.0).valid


@pytest.mark.parametrize("raw,expected", [(50, (4, 3)), ("A", (4, 3)), ("AD", (4, 3))])
def test_parser_accepts_advantage_tokens(raw: Any, expected: tuple[int, int]) -> None:
    s = samples()["live"]
    details = {
        **s["live_data"]["details"],
        "competitor1_current_round_score": raw,
        "competitor2_current_round_score": 40,
    }
    assert parse_kalshi_live(details, ref_for(s), 1.0).score.points == expected
