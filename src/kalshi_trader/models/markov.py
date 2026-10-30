"""Exact Markov-chain tennis win probabilities.

Classic hierarchical model (Newton & Keller 2005; O'Malley 2008; Barnett & Clarke 2005):
points are independent, and each player wins a point on their own serve with a fixed
probability ``p[i]``. Probabilities are computed exactly (no simulation) by recursion over
game -> tiebreak -> set -> match states, with closed forms for deuce and for extended
tiebreaks / advantage sets so the recursion is finite.

The model also returns the expected number of remaining games, used to scale an
in-match retirement hazard. It ignores momentum, fatigue and in-match strength changes;
those limitations are why the estimator layers parameter uncertainty on top.
"""

from __future__ import annotations

from kalshi_trader.tennis.state import MatchFormat, ScoreState, tiebreak_server

EPS = 1e-4


def _clamp(p: float) -> float:
    return min(1.0 - EPS, max(EPS, p))


class MarkovMatchModel:
    def __init__(self, p_serve: tuple[float, float], fmt: MatchFormat) -> None:
        self.p = (_clamp(p_serve[0]), _clamp(p_serve[1]))
        self.fmt = fmt
        self._game: dict[tuple[int, int, int], float] = {}
        self._tb: dict[tuple[int, int, int, int], float] = {}
        self._set: dict[tuple[int, int, int, bool], tuple[dict[tuple[int, int], float], float]] = {}
        self._match: dict[tuple[int, int, int], tuple[float, float]] = {}

    # ----------------------------------------------------------------- game
    def game_prob(self, server: int, a: int = 0, b: int = 0) -> float:
        """P(server wins the game) from server points `a`, receiver points `b`."""
        key = (server, a, b)
        cached = self._game.get(key)
        if cached is not None:
            return cached
        p = self.p[server]
        q = 1.0 - p
        if a >= 4 and a - b >= 2:
            result = 1.0
        elif b >= 4 and b - a >= 2:
            result = 0.0
        elif a >= 3 and b >= 3:
            deuce = p * p / (p * p + q * q)
            if a == b:
                result = deuce
            elif a > b:
                result = p + q * deuce
            else:
                result = p * deuce
        else:
            result = p * self.game_prob(server, a + 1, b) + q * self.game_prob(server, a, b + 1)
        self._game[key] = result
        return result

    def hold_prob(self, server: int) -> float:
        return self.game_prob(server, 0, 0)

    # ------------------------------------------------------------- tiebreak
    def _point_prob_p0(self, server: int) -> float:
        """P(player 0 wins a point) given who serves it."""
        return self.p[0] if server == 0 else 1.0 - self.p[1]

    def tiebreak_prob(self, a: int, b: int, first_server: int, target: int) -> float:
        """P(player 0 wins the tiebreak) from points (a, b)."""
        key = (a, b, first_server, target)
        cached = self._tb.get(key)
        if cached is not None:
            return cached
        if a >= target and a - b >= 2:
            result = 1.0
        elif b >= target and b - a >= 2:
            result = 0.0
        elif a == b and a >= target - 1:
            # From a tie each pair of points has one serve per player.
            x = self._point_prob_p0(0)
            y = self._point_prob_p0(1)
            result = x * y / (x * y + (1.0 - x) * (1.0 - y))
        else:
            server = tiebreak_server(first_server, a + b)
            pw = self._point_prob_p0(server)
            result = pw * self.tiebreak_prob(a + 1, b, first_server, target) + (
                1.0 - pw
            ) * self.tiebreak_prob(a, b + 1, first_server, target)
        self._tb[key] = result
        return result

    # ------------------------------------------------------------------ set
    def set_outcomes(
        self, g0: int, g1: int, server: int, final: bool
    ) -> tuple[dict[tuple[int, int], float], float]:
        """Distribution over (set winner, first server of next set) and expected games.

        `server` serves the next game of the set.
        """
        key = (g0, g1, server, final)
        cached = self._set.get(key)
        if cached is not None:
            return cached
        n = self.fmt.games_per_set
        advantage = final and self.fmt.final_set == "advantage"
        result: tuple[dict[tuple[int, int], float], float]
        if g0 >= n and g0 - g1 >= 2:
            result = ({(0, server): 1.0}, 0.0)
        elif g1 >= n and g1 - g0 >= 2:
            result = ({(1, server): 1.0}, 0.0)
        elif g0 == n and g1 == n and not advantage:
            target = self.fmt.final_set_tiebreak_points if final else self.fmt.tiebreak_points
            p0 = self.tiebreak_prob(0, 0, server, target)
            result = ({(0, 1 - server): p0, (1, 1 - server): 1.0 - p0}, 1.0)
        elif advantage and g0 == g1 and g0 >= n - 1:
            hold0 = self.hold_prob(0)
            break0 = 1.0 - self.hold_prob(1)
            both0 = hold0 * break0
            both1 = (1.0 - hold0) * (1.0 - break0)
            p0 = both0 / (both0 + both1)
            # An even number of games is played from a tie, so the server is unchanged.
            result = ({(0, server): p0, (1, server): 1.0 - p0}, 2.0 / (both0 + both1))
        else:
            hold = self.hold_prob(server)
            won = (g0 + 1, g1) if server == 0 else (g0, g1 + 1)
            lost = (g0, g1 + 1) if server == 0 else (g0 + 1, g1)
            dist_w, e_w = self.set_outcomes(won[0], won[1], 1 - server, final)
            dist_l, e_l = self.set_outcomes(lost[0], lost[1], 1 - server, final)
            dist: dict[tuple[int, int], float] = {}
            for k, v in dist_w.items():
                dist[k] = dist.get(k, 0.0) + hold * v
            for k, v in dist_l.items():
                dist[k] = dist.get(k, 0.0) + (1.0 - hold) * v
            result = (dist, 1.0 + hold * e_w + (1.0 - hold) * e_l)
        self._set[key] = result
        return result

    # ---------------------------------------------------------------- match
    def match_from_sets(self, s0: int, s1: int, server: int) -> tuple[float, float]:
        """(P(player 0 wins match), expected remaining games) at the start of a set."""
        key = (s0, s1, server)
        cached = self._match.get(key)
        if cached is not None:
            return cached
        k = self.fmt.sets_to_win
        if s0 >= k:
            result = (1.0, 0.0)
        elif s1 >= k:
            result = (0.0, 0.0)
        else:
            final = s0 == k - 1 and s1 == k - 1
            if final and self.fmt.final_set == "match_tiebreak":
                p0 = self.tiebreak_prob(0, 0, server, self.fmt.final_set_tiebreak_points)
                result = (p0, 1.0)
            else:
                dist, e_set = self.set_outcomes(0, 0, server, final)
                result = self._combine(dist, e_set, s0, s1)
        self._match[key] = result
        return result

    def _combine(
        self, dist: dict[tuple[int, int], float], e_now: float, s0: int, s1: int
    ) -> tuple[float, float]:
        p0 = 0.0
        e = e_now
        for (w, next_server), prob in dist.items():
            if prob == 0.0:
                continue
            m_p, m_e = self.match_from_sets(s0 + (w == 0), s1 + (w == 1), next_server)
            p0 += prob * m_p
            e += prob * m_e
        return p0, e

    def win_probability(self, state: ScoreState) -> tuple[float, float]:
        """(P(player 0 wins the match), expected remaining games) from any state."""
        if state.winner is not None:
            return (1.0 if state.winner == 0 else 0.0), 0.0
        s0, s1 = state.sets
        final = state.is_final_set
        if state.in_tiebreak:
            first = state.tiebreak_first_server
            if first is None:
                first = state.server
            a, b = state.points
            p_tb = self.tiebreak_prob(a, b, first, state.tiebreak_target)
            if final and self.fmt.final_set == "match_tiebreak":
                return p_tb, 0.5
            dist = {(0, 1 - first): p_tb, (1, 1 - first): 1.0 - p_tb}
            return self._combine(dist, 0.5, s0, s1)

        server = state.server
        a, b = state.points[server], state.points[1 - server]
        hold = self.game_prob(server, a, b)
        g0, g1 = state.games
        won = (g0 + 1, g1) if server == 0 else (g0, g1 + 1)
        lost = (g0, g1 + 1) if server == 0 else (g0 + 1, g1)
        dist_w, e_w = self.set_outcomes(won[0], won[1], 1 - server, final)
        dist_l, e_l = self.set_outcomes(lost[0], lost[1], 1 - server, final)
        dist: dict[tuple[int, int], float] = {}
        for k, v in dist_w.items():
            dist[k] = dist.get(k, 0.0) + hold * v
        for k, v in dist_l.items():
            dist[k] = dist.get(k, 0.0) + (1.0 - hold) * v
        return self._combine(dist, 0.5 + hold * e_w + (1.0 - hold) * e_l, s0, s1)


def pre_match_probability(
    p_serve: tuple[float, float], fmt: MatchFormat, first_server: int = 0
) -> float:
    model = MarkovMatchModel(p_serve, fmt)
    return model.match_from_sets(0, 0, first_server)[0]


def implied_serve_difference(
    target_prob: float, base: float, fmt: MatchFormat, tol: float = 1e-6
) -> float:
    """Find d such that a match with serve probs (base + d/2, base - d/2) has P(p1 wins)
    equal to `target_prob` (averaged over who serves first). Bisection; monotone in d."""
    target = min(0.995, max(0.005, target_prob))

    def prob(d: float) -> float:
        p = (base + d / 2, base - d / 2)
        return 0.5 * (pre_match_probability(p, fmt, 0) + pre_match_probability(p, fmt, 1))

    lo, hi = -0.5, 0.5
    lo = max(lo, -2 * (1 - base) + 0.02, -2 * base + 0.02)
    hi = min(hi, 2 * (1 - base) - 0.02, 2 * base - 0.02)
    for _ in range(60):
        mid = (lo + hi) / 2
        if prob(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2
