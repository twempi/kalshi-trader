"""Central capital allocation: which approved opportunities get capital, in what order.

Objective (explicit, no ad-hoc weighted scores)::

    maximise    sum_i  E_low[profit_i]          (conservative expected profit, after fees)
    subject to  sum_i  max_loss_i  <=  available risk capital
                per-trade, per-event, per-strategy, correlation-group and aggregate caps
                (all expressed as maximum possible loss, see ``RiskManager.budget``)
                liquidity (fills walked on the real book), data freshness, safety switches

With one binding capital constraint this is a knapsack problem; its LP relaxation is solved
exactly by taking items in decreasing order of value per unit of capital (Dantzig's greedy
rule). So candidates are ranked by **conservative expected return on capital at risk**::

    rank_score = E_low[profit] / max_loss

and allocated greedily: each one is re-sized against the *current* portfolio (after the
earlier allocations in the same cycle were executed) and re-evaluated on the current book
immediately before simulated execution. With several caps binding at once the greedy rule
is a heuristic, not an exact optimum; it never violates a cap because every cap is
re-checked before each allocation.

Uncertainty and evidence enter through the inputs, not through extra weights: E_low uses
the model's lower credible bound (wider uncertainty => smaller E_low), and strategies
without validated out-of-sample evidence get a reduced risk budget
(``risk.unvalidated_size_multiplier``). Baseline strategies that are not edge-gated have no
model expectation; they are ranked after every edge-gated candidate, by their EV at the
market mid (usually negative - they exist for comparison).
"""

from __future__ import annotations

from collections.abc import Iterable

from kalshi_trader.core.opportunity import Opportunity


def rank_score(opp: Opportunity) -> float:
    if opp.expected_profit_low is None or opp.max_loss <= 0:
        # A fully hedged basket with no possible loss is the best use of capital.
        return float("inf") if (opp.expected_profit_low or 0) > 0 else float("-inf")
    return opp.expected_profit_low / float(opp.max_loss)


def rank(opps: Iterable[Opportunity]) -> list[Opportunity]:
    """Edge-gated candidates first (by rank_score, then absolute E_low), then baselines."""

    def key(opp: Opportunity) -> tuple[int, float, float]:
        return (
            0 if opp.edge_gated else 1,
            -rank_score(opp),
            -(opp.expected_profit_low or 0.0),
        )

    return sorted(opps, key=key)
