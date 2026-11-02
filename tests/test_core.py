"""Shared core: execution-aware evaluation, allocation, pipeline, registry, diagnostics.

All data here is SYNTHETIC test data, built to make every financial number checkable by
hand. Nothing in this file is a trading result.
"""

from __future__ import annotations

import math
from decimal import Decimal as D

import pytest

from kalshi_trader.backtesting.metrics import TradeOutcome
from kalshi_trader.config import Settings, StrategySettingsBase
from kalshi_trader.core.allocator import rank, rank_score
from kalshi_trader.core.diagnostics import reason_key
from kalshi_trader.core.evaluator import ExecutionEvaluator, SizingBudget
from kalshi_trader.core.opportunity import DecisionStatus, FairValue, Leg, Opportunity
from kalshi_trader.core.pipeline import TradingCore
from kalshi_trader.core.registry import StrategyRegistry, evaluate_evidence
from kalshi_trader.database.db import Database
from kalshi_trader.database.repository import Repository
from kalshi_trader.market_data.discovery import TrackedMarket
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.strategies.base import ScanContext, Strategy

RATE = D("0.07")


# ------------------------------------------------- independent fee reference
def ref_debit(levels: list[tuple[str, int]]) -> D:
    """Kalshi taker fee re-implemented from the published formula, independently of
    execution.fees: fee = ceil_to_1e-6(sum 0.07*C*P*(1-P)); debit = ceil_to_cent(notional+fee).
    """
    notional = sum(D(p) * c for p, c in levels)
    raw = sum(RATE * c * D(p) * (1 - D(p)) for p, c in levels)
    fee = D(math.ceil(raw * 1_000_000)) / 1_000_000
    return D(math.ceil((notional + fee) * 100)) / 100


def book(ticker: str, yes_asks: list[tuple[str, str]], yes_bids=(("0.49", "100"),)) -> OrderBook:
    """YES asks are given directly; they are stored as NO bids at 1 - price."""
    b = OrderBook(ticker)
    b.apply_snapshot(
        [(D(p), D(s)) for p, s in yes_bids],
        [(1 - D(p), D(s)) for p, s in yes_asks],
        source="test",
        received_at=1000.0,
    )
    return b


def directional(
    p: float = 0.62, q: float = 0.60, min_edge: str = "0.03", limit: str = "0.60", **kw
) -> Opportunity:
    leg = Leg(
        market_ticker=kw.pop("ticker", "M-A"),
        event_ticker=kw.get("event_ticker", "EV-1"),
        side="yes",
        best_ask=D("0.50"),
        limit_price=D(limit),
    )
    base = dict(
        strategy="stub",
        strategy_version="1+t",
        kind="directional",
        category="test",
        event_ticker="EV-1",
        legs=[leg],
        observed_ts=1000.0,
        fair=FairValue(p, q, 0.65, "test", 1000.0),
        min_edge=D(min_edge),
    )
    base.update(kw)
    return Opportunity(**base)


LADDER = [("0.50", "10"), ("0.52", "10"), ("0.60", "100")]
BIG = SizingBudget(max_loss=D("1000"), equity=D("100000"), kelly_fraction=D("1"))


# ------------------------------------------------------------ evaluator
def test_evaluator_sizes_at_actual_fill_and_fees() -> None:
    ev = ExecutionEvaluator(RATE)
    opp = ev.evaluate(directional(), {"M-A": book("M-A", LADDER)}, BIG)
    # Hand calculation: all-in average must stay <= q - min_edge = 0.57.
    # n=38 (10@.50, 10@.52, 18@.60): debit 21.66 = 0.57*38 exactly; n=39 -> 22.27 > 22.23.
    assert opp.status == DecisionStatus.CANDIDATE
    assert opp.contracts == D("38")
    expected_debit = ref_debit([("0.50", 10), ("0.52", 10), ("0.60", 18)])
    assert opp.cost == expected_debit == D("21.66")
    assert ref_debit([("0.50", 10), ("0.52", 10), ("0.60", 19)]) > D("0.57") * 39
    assert opp.fees == D("0.66") and opp.slippage == D("2.00")  # 21.00 notional vs 19.00
    assert opp.expected_profit_low == pytest.approx(0.60 * 38 - 21.66)
    assert opp.expected_profit == pytest.approx(0.62 * 38 - 21.66)
    assert opp.max_loss == opp.cost
    assert opp.primary.avg_price == D("21.00") / 38


def test_edge_that_vanishes_after_fees_is_rejected() -> None:
    # 0.53 vs a 0.50 ask looks like a 3c edge, but the 1-contract all-in cost is 0.52
    # (fee rounds up to the cent), and 10 contracts cost 5.18: edge <= 1.8c < 3c.
    ev = ExecutionEvaluator(RATE)
    opp = ev.evaluate(directional(p=0.55, q=0.53, limit="0.50"), {"M-A": book("M-A", LADDER)}, BIG)
    assert opp.status == DecisionStatus.REJECTED
    assert "net edge vanishes after fill+fees" in opp.reasons[0]


@pytest.mark.parametrize(
    "budget,expected",
    [
        (SizingBudget(D("5"), D("100000"), D("1")), 9),  # 9 -> 4.66, 10 -> 5.18
        (SizingBudget(D("10"), D("100000"), D("1"), size_multiplier=D("0.5")), 9),
        # Kelly: q=.60, cost ~.52 -> f*=.167; 10% of $100 x .167 = $1.67 -> 3 contracts
        (SizingBudget(D("1000"), D("100"), D("0.1")), 3),
    ],
)
def test_evaluator_respects_risk_budget_multiplier_and_kelly(
    budget: SizingBudget, expected: int
) -> None:
    ev = ExecutionEvaluator(RATE)
    opp = ev.evaluate(directional(limit="0.50"), {"M-A": book("M-A", LADDER)}, budget)
    assert opp.contracts == D(expected), opp.reasons
    assert opp.cost <= budget.max_loss * budget.size_multiplier


def test_liquidity_fraction_limits_size() -> None:
    ev = ExecutionEvaluator(RATE, liquidity_fraction=D("0.5"))
    opp = ev.evaluate(directional(limit="0.52"), {"M-A": book("M-A", LADDER)}, BIG)
    assert opp.contracts == D("10")  # half of 10 + half of 10


def test_baseline_without_model_is_not_edge_gated() -> None:
    ev = ExecutionEvaluator(RATE)
    opp = directional(fair=None, edge_gated=False, market_prob=0.49, limit="0.50")
    opp = ev.evaluate(opp, {"M-A": book("M-A", LADDER)}, SizingBudget(D("5"), D("1e6"), D("1")))
    assert opp.status == DecisionStatus.CANDIDATE and opp.contracts == D("9")
    assert opp.expected_profit == pytest.approx(0.49 * 9 - 4.66)  # negative: pays spread+fee


def basket(min_edge: str = "0.01") -> Opportunity:
    legs = [
        Leg("T-A", "EV-T", "yes", limit_price=D("0.99")),
        Leg("T-B", "EV-T", "yes", limit_price=D("0.99")),
    ]
    return Opportunity(
        strategy="cross_market",
        strategy_version="1+t",
        kind="basket",
        category="tennis",
        event_ticker="EV-T",
        legs=legs,
        observed_ts=1000.0,
        guaranteed_payout=D("1"),
        min_edge=D(min_edge),
    )


PAIR = {"T-A": book("T-A", [("0.45", "50")]), "T-B": book("T-B", [("0.50", "50")])}


def test_basket_guaranteed_profit_and_legging_risk() -> None:
    ev = ExecutionEvaluator(RATE)
    opp = ev.evaluate(basket(), PAIR, SizingBudget(D("10"), D("1e6"), D("1")))
    # Legging risk = cost of the dearer leg (B): n=19 -> 9.84 <= $10, n=20 -> 10.35.
    assert opp.contracts == D("19")
    a, b = ref_debit([("0.45", 19)]), ref_debit([("0.50", 19)])
    # B: 9.50 + fee 0.3325 = 9.8325, rounded UP to the cent.
    assert (a, b) == (D("8.88"), D("9.84")) and opp.cost == a + b
    assert opp.expected_profit_low == pytest.approx(19 - float(a + b))
    assert opp.max_loss == b  # never treated as risk-free


def test_basket_rejected_when_fees_eat_the_edge() -> None:
    ev = ExecutionEvaluator(RATE)
    pair = {"T-A": book("T-A", [("0.48", "50")]), "T-B": book("T-B", [("0.50", "50")])}
    opp = ev.evaluate(basket(), pair, BIG)  # 0.98 before fees, 1.00 after
    assert opp.status == DecisionStatus.REJECTED and "basket edge" in opp.reasons[0]


# ------------------------------------------------------------ allocator
def test_rank_by_conservative_return_on_capital_at_risk() -> None:
    a = directional(ticker="A")
    a.expected_profit_low, a.max_loss = 2.0, D("20")  # 10%
    b = directional(ticker="B")
    b.expected_profit_low, b.max_loss = 1.5, D("5")  # 30%
    base = directional(ticker="C", edge_gated=False)
    base.expected_profit_low, base.max_loss = 5.0, D("5")  # baseline: always after gated
    assert [o.market_ticker for o in rank([a, base, b])] == ["B", "A", "C"]
    hedged = directional(ticker="H")
    hedged.expected_profit_low, hedged.max_loss = 0.1, D("0")
    assert rank_score(hedged) == float("inf")


# ------------------------------------------------------------- pipeline
class Stub(Strategy):
    requires_score = False

    def __init__(self, name: str, mode: str, factory) -> None:
        super().__init__(StrategySettingsBase(mode=mode), Settings(_env_file=None).fees)
        self.name = name
        self.factory = factory

    def scan(self, ctx: ScanContext) -> list[Opportunity]:
        return self.factory(ctx)


def market(ticker: str, event: str) -> TrackedMarket:
    return TrackedMarket(
        ticker=ticker,
        event_ticker=event,
        series_ticker="S",
        player_name=ticker,
        competitor_id=None,
        status="active",
    )


def ctx_for(books: dict[str, OrderBook], events: dict[str, str], now: float = 1001.0):
    return ScanContext(
        now=now,
        markets={t: market(t, e) for t, e in events.items()},
        books=books,
    )


def stub_opp(name: str, ticker: str, event: str, q: float, limit: str = "0.60") -> Opportunity:
    o = directional(p=q + 0.02, q=q, ticker=ticker, event_ticker=event, limit=limit)
    o.strategy = name
    o.primary.event_ticker = event
    o.correlation_group = f"grp:{event}"
    return o


@pytest.fixture
def core_settings(settings: Settings) -> Settings:
    settings.paper.displayed_liquidity_fraction = D("1")
    return settings


def make_core(settings: Settings, strategies, db: Database | None = None) -> TradingCore:
    db = db or Database(":memory:")
    return TradingCore(settings, db, Repository(db), strategies, clock=lambda: 1001.0)


def test_correlated_opportunities_on_one_event_get_one_allocation(core_settings) -> None:
    books = {"M-A": book("M-A", LADDER), "M-B": book("M-B", LADDER)}
    s1 = Stub("s1", "paper", lambda c: [stub_opp("s1", "M-A", "EV-1", 0.60)])
    s2 = Stub("s2", "paper", lambda c: [stub_opp("s2", "M-B", "EV-1", 0.70)])
    core = make_core(core_settings, [s1, s2])
    opps = core.run_cycle(ctx_for(books, {"M-A": "EV-1", "M-B": "EV-1"}))
    by = {o.strategy: o for o in opps}
    # s2 has the higher conservative return on capital, so it is allocated first...
    assert by["s2"].status == DecisionStatus.FILLED
    # ...and s1, on the same underlying event, is refused rather than stacked on top.
    assert by["s1"].status == DecisionStatus.REJECTED
    assert any("already exposed" in r or "max trades" in r for r in by["s1"].reasons)
    trades = core.broker.open_trades()
    assert len(trades) == 1 and trades[0].strategy == "s2"
    # Every main-account position respects the per-trade cap (25% unvalidated x $20).
    assert trades[0].entry_cost + trades[0].entry_fee <= D("5")
    # Each strategy's shadow ledger traded its own opportunity independently.
    assert len(core.shadows["s1"].open_trades()) == 1
    assert len(core.shadows["s2"].open_trades()) == 1


def test_max_open_positions_goes_to_best_ranked(core_settings) -> None:
    core_settings.risk.max_open_positions = 1
    books = {"M-A": book("M-A", LADDER), "M-B": book("M-B", LADDER)}
    s1 = Stub(
        "s1",
        "paper",
        lambda c: [
            stub_opp("s1", "M-A", "EV-1", 0.60),
            stub_opp("s1", "M-B", "EV-2", 0.75),
        ],
    )
    core = make_core(core_settings, [s1])
    opps = core.run_cycle(ctx_for(books, {"M-A": "EV-1", "M-B": "EV-2"}))
    by = {o.market_ticker: o for o in opps}
    assert by["M-B"].status == DecisionStatus.FILLED
    assert by["M-A"].status == DecisionStatus.REJECTED
    assert any("max open positions" in r for r in by["M-A"].reasons)


def test_observe_mode_never_touches_main_account(core_settings) -> None:
    books = {"M-A": book("M-A", LADDER)}
    s = Stub("obs", "observe", lambda c: [stub_opp("obs", "M-A", "EV-1", 0.60)])
    core = make_core(core_settings, [s])
    (opp,) = core.run_cycle(ctx_for(books, {"M-A": "EV-1"}))
    assert opp.status == DecisionStatus.OBSERVED and opp.contracts > 0
    assert core.broker.open_trades() == []
    assert core.broker.account().cash == D("1000")
    assert len(core.shadows["obs"].open_trades()) == 1


def test_restart_does_not_duplicate_positions(core_settings, tmp_path) -> None:
    path = tmp_path / "restart.db"
    books = {"M-A": book("M-A", LADDER)}

    def strategy():
        return Stub("s", "paper", lambda c: [stub_opp("s", "M-A", "EV-1", 0.60)])

    first = make_core(core_settings, [strategy()], Database(path))
    (o1,) = first.run_cycle(ctx_for(books, {"M-A": "EV-1"}))
    assert o1.status == DecisionStatus.FILLED
    cash = first.broker.account().cash
    second = make_core(core_settings, [strategy()], Database(path))  # simulated restart
    (o2,) = second.run_cycle(ctx_for(books, {"M-A": "EV-1"}, now=2000.0))
    assert o2.status == DecisionStatus.REJECTED
    assert len(second.broker.open_trades()) == 1 and second.broker.account().cash == cash


def test_stale_book_is_rejected_before_execution(core_settings) -> None:
    books = {"M-A": book("M-A", LADDER)}  # received at t=1000
    s = Stub("s", "paper", lambda c: [stub_opp("s", "M-A", "EV-1", 0.60)])
    core = make_core(core_settings, [s])
    (opp,) = core.run_cycle(ctx_for(books, {"M-A": "EV-1"}, now=1100.0))
    assert opp.status == DecisionStatus.REJECTED
    assert any("order book stale" in r for r in opp.reasons)
    assert core.broker.open_trades() == []


def run_basket(core: TradingCore, books: dict[str, OrderBook]) -> Opportunity:
    opp = core.evaluator.evaluate(basket(), books, BIG)
    core._execute_basket(opp, core.broker, ctx_for(books, {"T-A": "EV-T", "T-B": "EV-T"}))
    return opp


def test_basket_fully_hedged_when_later_leg_is_deep(core_settings) -> None:
    core = make_core(core_settings, [Stub("s", "paper", lambda c: [])])
    books = {"T-A": book("T-A", [("0.45", "10")]), "T-B": book("T-B", [("0.50", "50")])}
    opp = run_basket(core, books)
    assert opp.contracts == D("10") and opp.status == DecisionStatus.FILLED
    trades = core.broker.open_trades()
    assert {t.entry_contracts for t in trades} == {D("10")}
    assert len({t.group_id for t in trades}) == 1  # one basket position
    assert core.broker.trade_counts("T-A", "EV-T") == (1, 1, True)
    # Hedged legs carry the basket's (small) max loss, not their full cost.
    assert sum(t.max_loss for t in trades) <= opp.max_loss + D("0.01")


def test_multi_leg_failure_unwinds_excess_into_bids(core_settings) -> None:
    """A (least liquid) fills 10; B may only take 50% of its 15 displayed -> 7. The 3
    unhedged A contracts are sold back into A's bid immediately."""
    core = make_core(core_settings, [Stub("s", "paper", lambda c: [])])
    books = {
        "T-A": book("T-A", [("0.45", "10")], yes_bids=[("0.43", "100")]),
        "T-B": book("T-B", [("0.50", "15")]),
    }
    opp = run_basket(core, books)
    trades = {t.market_ticker: t for t in core.broker.all_trades()}
    assert trades["T-B"].entry_contracts == D("7")
    assert trades["T-A"].entry_contracts == D("10") and trades["T-A"].contracts == D("7")
    assert trades["T-A"].exit_contracts == D("3")
    assert opp.status == DecisionStatus.PARTIAL
    assert any("hedged 7/10" in r for r in opp.reasons)
    assert any("unwound 3/3" in r for r in opp.reasons)


def test_failed_unwind_leaves_full_max_loss(core_settings) -> None:
    core = make_core(core_settings, [Stub("s", "paper", lambda c: [])])
    books = {
        "T-A": book("T-A", [("0.45", "10")], yes_bids=()),
        "T-B": book("T-B", [("0.50", "15")]),
    }
    opp = run_basket(core, books)  # A has no bids: the excess cannot be sold
    (a,) = [t for t in core.broker.open_trades() if t.market_ticker == "T-A"]
    assert any("unwound 0/3" in r for r in opp.reasons)
    assert a.contracts == D("10") and a.max_loss == a.entry_cost + a.entry_fee


def test_settlement_pays_main_and_shadow_ledgers(core_settings) -> None:
    books = {"M-A": book("M-A", LADDER)}
    s = Stub("s", "paper", lambda c: [stub_opp("s", "M-A", "EV-1", 0.60)])
    core = make_core(core_settings, [s])
    core.run_cycle(ctx_for(books, {"M-A": "EV-1"}))
    settled = core.settle("M-A", "yes", D("1"), 1500.0)
    assert len(settled) == 2  # main + shadow
    assert core.open_market_tickers() == set()


# --------------------------------------------------------------- registry
def outcome(pnl: float, event: str, price: float = 0.5) -> TradeOutcome:
    return TradeOutcome(0.0, 1.0, "s", price, 10, 10 * price, pnl, 0.1, event=event)


def test_registry_lifecycle_and_rules_change(memory_db: Database, settings: Settings) -> None:
    reg = StrategyRegistry(Repository(memory_db), settings.evidence)
    reg.sync([("s", "1", "aaaa", "observe")])
    assert reg.state("s").status == "experimental"
    assert reg.allows_main("s", "observe") == (False, "observe mode (shadow ledger only)")
    reg.sync([("s", "1", "aaaa", "paper")])
    assert reg.state("s").status == "paper_trading" and reg.allows_main("s", "paper")[0]
    # Significantly negative forward evidence rejects the strategy for the main account.
    losers = [outcome(-5.0, f"E{i}") for i in range(60)]
    result = reg.update_evidence("s", "paper", losers, 1000.0)
    assert result["negative"] and reg.state("s").status == "rejected"
    assert not reg.allows_main("s", "paper")[0]
    # Changing the rules resets the evidence: old results no longer apply.
    reg.sync([("s", "2", "bbbb", "paper")])
    assert reg.state("s").status == "paper_trading" and reg.state("s").evidence_json is None
    reg.set_paused("s", True, mode="paper")
    assert reg.allows_main("s", "paper") == (False, "strategy paused by operator")
    reg.set_paused("s", False, mode="paper")
    assert reg.state("s").status == "paper_trading"


def test_evidence_requires_enough_independent_events(settings: Settings) -> None:
    few = [outcome(4.0, "E1") for _ in range(80)]  # many trades, one event
    result = evaluate_evidence(few, settings.evidence, 1000.0)
    assert not result["validated"]
    assert any("independent events" in f for f in result["failures"])


def test_unvalidated_size_multiplier(memory_db: Database, settings: Settings) -> None:
    reg = StrategyRegistry(Repository(memory_db), settings.evidence)
    reg.sync([("s", "1", "h", "paper")])
    assert reg.size_multiplier("s", D("0.25")) == D("0.25")


def test_reason_keys_group_equivalent_causes() -> None:
    assert reason_key("order book stale (12.3s)") == reason_key("order book stale (4.0s)")
    assert reason_key("YES KXATPMATCH-26OCT30ZVEFER-ZVE: spread 0.09 > 0.04") == (
        "YES <market>: spread # > #"
    )
