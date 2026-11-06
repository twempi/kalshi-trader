"""Strategy A (statistical mispricing), instantiated for tennis as Strategy C.

Idea: buy a contract when a *defensible, independent* probability model says it is worth
more than its full executable cost, by enough to cover fees, execution costs and the
model's own uncertainty. The decision always uses the model's conservative probability
``q`` (a lower credible bound), never its point estimate:

    expected net profit (conservative) = n * q - (walked notional + exact fees)  >= n * min_edge

For a NO contract the backed player's probability is used (NO on player A's market pays
exactly when player B wins), so complementary probabilities are handled by construction.

Fair values come from pluggable ``FairValueModel`` objects. A market for which no model
can produce an independent estimate is reported as ``unsupported`` - the strategy never
invents a probability, and never uses the current market price as both the fair value and
the evidence of mispricing. Today exactly one model is implemented:

* ``tennis_markov``: the exact Markov-chain match model with posterior uncertainty over
  serve strengths. Its only market input is the *pre-match* price used to anchor relative
  strength; the in-play information (score, server, live serve statistics) is independent of
  the live market price, which is what the strategy compares against.

The model's probabilities can be recalibrated before use (``calibration``, fitted on settled
forward predictions by ``research.calibration``). With a ``market_weight`` the estimate is
shrunk toward the live mid: the market is then a prior, and the evidence of mispricing is
still the model's disagreement with it, which must be large enough to survive the shrinkage.

Stages: ``prematch`` is unsupported (the only pre-match strength information available is
the market price itself), ``early`` = first set, ``late`` = a player within
``late_games_from_victory`` games of winning, ``mid`` = otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from itertools import pairwise

from kalshi_trader.config import (
    FeeSettings,
    ProbabilityCalibrationSettings,
    TennisModelStrategySettings,
)
from kalshi_trader.core.opportunity import DecisionStatus, FairValue, Leg, Opportunity
from kalshi_trader.execution.fees import fee_per_contract_estimate
from kalshi_trader.execution.fills import snap_price_down
from kalshi_trader.market_data.orderbook import OrderBook, Side
from kalshi_trader.models.calibration import calibrate
from kalshi_trader.models.ev import max_price_for_edge
from kalshi_trader.strategies.base import MatchContext, ScanContext, Strategy
from kalshi_trader.tennis.state import MatchState, MatchStatus

LIMIT_FEE_HINT = Decimal("100")


def match_stage(score: MatchState | None, late_games: int) -> str:
    """prematch | early | mid | late (documented in the module docstring)."""
    if score is None or score.status != MatchStatus.LIVE or score.score is None:
        return "prematch"
    state = score.score
    if state.sets == (0, 0):
        return "early"
    distances = [d for d in (state.games_from_victory(0), state.games_from_victory(1)) if d]
    if distances and min(distances) <= late_games:
        return "late"
    return "mid"


def market_probability(ctx: MatchContext, player: int) -> float | None:
    """The market's probability that ``player`` wins: their YES mid, else 1 - the opponent's."""
    for index in (player, 1 - player):
        ticker = ctx.match.market_tickers[index]
        book = ctx.books.get(ticker) if ticker else None
        mid = book.mid("yes") if book is not None else None
        if mid is not None:
            return float(mid) if index == player else 1.0 - float(mid)
    return None


def probability_band(p: float) -> str:
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0001)
    for lo, hi in pairwise(edges):
        if lo <= p < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    return "unknown"


class FairValueModel(ABC):
    name: str = "abstract"

    @abstractmethod
    def unsupported_reason(self, ctx: MatchContext) -> str | None:
        """None if a defensible estimate exists for this context, else why not."""

    @abstractmethod
    def fair_value(self, ctx: MatchContext, player: int) -> FairValue: ...


class TennisMarkovFairValue(FairValueModel):
    name = "tennis_markov"

    def __init__(
        self,
        require_pre_match_prior: bool,
        min_ess: float,
        calibration: ProbabilityCalibrationSettings | None = None,
    ) -> None:
        self.require_pre_match_prior = require_pre_match_prior
        self.min_ess = min_ess
        self.calibration = calibration or ProbabilityCalibrationSettings()

    def unsupported_reason(self, ctx: MatchContext) -> str | None:
        score, estimate = ctx.score, ctx.estimate
        if score is None or not score.is_live:
            return "match not live with a valid score"
        if score.source_delayed:
            return "score source flagged delayed"
        if not score.fmt.known:
            return "match format unknown"
        if estimate is None:
            return "no model estimate"
        if self.require_pre_match_prior and estimate.prior.source != "pre_match_price":
            return "no pre-match price prior (tour-average prior is not specific enough)"
        if estimate.effective_sample_size < self.min_ess:
            return f"posterior degenerate (ESS {estimate.effective_sample_size:.0f})"
        if self.calibration.market_weight and market_probability(ctx, 0) is None:
            return "no two-sided book for the market-anchored calibration"
        return None

    def fair_value(self, ctx: MatchContext, player: int) -> FairValue:
        est = ctx.estimate
        assert est is not None and ctx.score is not None
        prob = est.for_player(player)
        p_high = est.p1_robust_high if player == 0 else 1.0 - est.p1_robust_low
        details: dict[str, object] = {
            "prior": est.prior.source,
            "std": round(est.p1_std, 4),
            "latency_points": est.latency_points,
            "credible_quantile": est.credible_quantile,
        }
        cal = self.calibration
        if cal.is_identity:
            return FairValue(
                p=prob.point,
                p_low=prob.robust_low,
                p_high=p_high,
                source=self.name,
                as_of_ts=ctx.score.received_ts,
                details=details,
            )
        market = market_probability(ctx, player) if cal.market_weight else None
        details.update(
            {
                "p_model": round(prob.point, 4),
                "p_low_model": round(prob.robust_low, 4),
                "p_market": None if market is None else round(market, 4),
                "calibration": {"slope": cal.slope, "market_weight": cal.market_weight},
            }
        )
        return FairValue(
            p=calibrate(prob.point, cal, market),
            p_low=calibrate(prob.robust_low, cal, market),
            p_high=calibrate(p_high, cal, market),
            source=f"{self.name}+calibrated",
            as_of_ts=ctx.score.received_ts,
            details=details,
        )


class TennisModelStrategy(Strategy):
    """Statistical mispricing on tennis match-winner markets, any configured match stage."""

    name = "tennis_model"
    version = "1"
    requires_score = True

    def __init__(
        self, settings: TennisModelStrategySettings, fees: FeeSettings, name: str = "tennis_model"
    ) -> None:
        super().__init__(settings, fees)
        self.name = name  # variants run under their own name (own ledger and evidence)
        self.settings: TennisModelStrategySettings = settings
        self.model: FairValueModel = TennisMarkovFairValue(
            settings.require_pre_match_prior,
            settings.min_effective_sample_size,
            settings.calibration,
        )

    def scan(self, ctx: ScanContext) -> list[Opportunity]:
        out = super().scan(ctx)
        # Other categories are visible to the system (cross-market discovery) but have no
        # independent probability model: say so instead of guessing.
        for group in ctx.groups:
            if group.category != "tennis" and group.markets:
                market = group.markets[0]
                out.append(
                    Opportunity(
                        strategy=self.name,
                        strategy_version=self.version_tag,
                        kind="directional",
                        category=group.category,
                        event_ticker=group.event_ticker,
                        legs=[Leg(market.ticker, group.event_ticker, "yes")],
                        observed_ts=ctx.now,
                        title=group.title,
                        status=DecisionStatus.UNSUPPORTED,
                        reasons=[
                            f"no independent probability model for series {group.series_ticker}"
                        ],
                    )
                )
        return out

    def evaluate_match(self, ctx: MatchContext) -> list[Opportunity]:
        s = self.settings
        stage = match_stage(ctx.score, s.late_games_from_victory)
        if stage == "prematch":
            if "prematch" in s.stages and ctx.score is not None:
                return [
                    self.note(
                        ctx,
                        DecisionStatus.UNSUPPORTED,
                        "pre-match: no independent probability source (only the market price)",
                        stage=stage,
                    )
                ]
            return []
        if stage not in s.stages:
            return [self.note(ctx, DecisionStatus.FILTERED, f"stage {stage} disabled", stage=stage)]
        if s.tours and ctx.match.tour_key not in s.tours:
            return [
                self.note(
                    ctx, DecisionStatus.FILTERED, f"tour {ctx.match.tour_key} disabled", stage=stage
                )
            ]
        why = self.model.unsupported_reason(ctx)
        if why is not None:
            return [self.note(ctx, DecisionStatus.UNSUPPORTED, why, stage=stage)]

        best: Opportunity | None = None
        best_ev = float("-inf")
        why: list[str] = []
        closest: tuple[float, str] | None = None  # best sub-threshold edge, for diagnostics
        for player in (0, 1):
            fair = self.model.fair_value(ctx, player)
            for option in ctx.options_for(player):
                label = f"{option.side.upper()} {option.market.ticker}"
                if not option.market.tradable:
                    why.append(f"{label}: market not active")
                    continue
                ask = option.book.best_ask(option.side)
                spread = option.book.spread(option.side)
                if ask is None or spread is None or spread > s.max_spread:
                    why.append(f"{label}: spread {spread} > {s.max_spread}")
                    continue
                price = ask[0]
                if not (s.min_price <= price <= s.max_price):
                    why.append(f"{label}: ask {price} outside [{s.min_price}, {s.max_price}]")
                    continue
                mult = option.market.fee_multiplier
                fee_pc = fee_per_contract_estimate(
                    price, LIMIT_FEE_HINT, self.fees.taker_rate, mult
                )
                ev = fair.p_low - float(price + fee_pc)
                if ev < float(s.min_edge):
                    if closest is None or ev > closest[0]:
                        closest = (
                            ev,
                            f"best conservative edge {ev:+.3f} < {s.min_edge} "
                            f"({label}: q={fair.p_low:.3f}, ask {price})",
                        )
                    continue
                cap = max_price_for_edge(
                    fair.p_low,
                    s.min_edge,
                    fee_rate=self.fees.taker_rate,
                    fee_multiplier=mult,
                    contracts_hint=LIMIT_FEE_HINT,
                    ceiling=min(Decimal("0.99"), price + s.limit_slippage),
                )
                if cap is None or cap < price:
                    why.append(f"{label}: no limit price keeps the edge")
                    continue
                limit = snap_price_down(
                    min(price + s.limit_slippage, cap), option.market.price_ranges
                )
                if limit < price:
                    continue
                opp = self.directional(
                    ctx,
                    option,
                    fair=fair,
                    limit_price=limit,
                    stage=stage,
                    reason=f"{stage}: q={fair.p_low:.3f} (p={fair.p:.3f}) vs ask {price}, "
                    f"band {probability_band(fair.p)}",
                )
                opp.provenance["probability_band"] = probability_band(fair.p)
                opp.provenance["tour"] = ctx.match.tour_key
                if best is None or ev > best_ev:
                    best, best_ev = opp, ev
        if best is None:
            reason = closest[1] if closest else ("; ".join(why) or "no executable option")
            return [self.note(ctx, DecisionStatus.FILTERED, reason, stage=stage)]
        return [best]

    def exit_limit(
        self, ctx: MatchContext, ticker: str, side: Side, book: OrderBook, fee_multiplier: Decimal
    ) -> tuple[Decimal, str] | None:
        """Sell when the market now pays more than the model thinks the position is worth:
        bid - taker fee >= p + exit_edge, with p the point probability of the held side."""
        edge = self.settings.exit_edge
        tickers = ctx.match.market_tickers
        if edge is None or ticker not in tickers or self.model.unsupported_reason(ctx):
            return None
        bid = book.best_bid(side)
        if bid is None:
            return None
        index = tickers.index(ticker)
        fair = self.model.fair_value(ctx, index if side == "yes" else 1 - index)
        fee = fee_per_contract_estimate(
            bid[0], LIMIT_FEE_HINT, self.fees.taker_rate, fee_multiplier
        )
        if float(bid[0] - fee) - fair.p < float(edge):
            return None
        return bid[0], f"edge flipped: bid {bid[0]} (fee {fee}) vs model p={fair.p:.3f}"
