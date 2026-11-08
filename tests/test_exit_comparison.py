"""SYNTHETIC accounting checks for selling early vs holding the same purchases."""

from decimal import Decimal as D

import pytest
from fastapi.testclient import TestClient

from kalshi_trader.config import Settings
from kalshi_trader.dashboard.app import create_app
from kalshi_trader.database.db import Database
from kalshi_trader.database.tables import MarketRow
from kalshi_trader.execution.paper import OrderIntent, PaperBroker
from kalshi_trader.market_data.orderbook import OrderBook
from kalshi_trader.research.exits import exit_comparison_report
from kalshi_trader.strategies import build_strategies

VERSION = "1+test"


def buy(broker: PaperBroker, ticker="M", side="yes", version=VERSION) -> int:
    book = OrderBook(ticker)
    bids = [(D("0.38"), D("1000"))]
    offers = [(D("0.60"), D("1000"))]
    book.apply_snapshot(
        bids if side == "yes" else offers,
        offers if side == "yes" else bids,
        source="ws",
    )
    report = broker.execute_buy(
        OrderIntent(
            f"buy:{broker.account_id}:{ticker}",
            "tennis_model_edge_exit",
            ticker,
            f"event:{ticker}",
            side,
            D("0.40"),
            D("10"),
            D("100"),
            strategy_version=version,
        ),
        book,
    )
    assert report.status == "filled" and report.trade_id is not None
    return report.trade_id


def sell(broker: PaperBroker, trade_id: int, side="yes", contracts=None) -> None:
    book = OrderBook("M")
    high, low = [(D("0.60"), D("1000"))], [(D("0.38"), D("1000"))]
    book.apply_snapshot(high if side == "yes" else low, low if side == "yes" else high, source="ws")
    assert (
        broker.execute_sell(
            trade_id, book, D("0.60"), f"sell:{trade_id}", contracts=contracts
        ).status
        == "filled"
    )


@pytest.mark.parametrize(
    ("side", "result", "held", "difference"),
    [("yes", "yes", 5.83, -4.17), ("yes", "no", -4.17, 5.83), ("no", "yes", -4.17, 5.83)],
)
def test_closed_trade_compares_exact_purchases_after_both_fees(
    settings: Settings, side: str, result: str, held: float, difference: float
) -> None:
    db = Database(":memory:")
    broker = PaperBroker(db, settings.paper, settings.fees)
    sell(broker, buy(broker, side=side), side=side)
    with db.session() as s:
        # Early-closed trades have no trade-level settlement, so use the final market.
        s.add(
            MarketRow(
                ticker="M",
                event_ticker="E",
                series_ticker="S",
                status="finalized",
                result=result,
                first_seen_ts=1,
                updated_ts=2,
            )
        )
    report = exit_comparison_report(db, broker.account_id, VERSION)
    assert report["compared_trades"] == report["compared_events"] == 1
    assert report["early_exit_trades"] == 1
    assert report["actual_pnl"] == 1.66  # $6 sale - $4 purchase - 17c fee on each side
    assert report["exit_fees"] == 0.17
    assert report["hold_pnl"] == held and report["difference"] == difference
    assert report["mean_difference_ci95"] is None  # one match cannot support a CI


def test_partial_no_exit_and_fractional_settlement(settings: Settings) -> None:
    db = Database(":memory:")
    broker = PaperBroker(db, settings.paper, settings.fees)
    sell(broker, buy(broker, side="no"), side="no", contracts=D("3"))
    broker.settle_market("M", "value", D("0.25"))
    report = exit_comparison_report(db, broker.account_id, VERSION)
    assert report["actual_pnl"] == 2.82  # 3 sold for $1.74 net + 7 * 75c - $4.17
    assert report["hold_pnl"] == 3.33  # same 10 NO contracts held for 75c each
    assert report["difference"] == -0.51
    assert report["trades"][0]["exit_contracts"] == 3


def test_no_sales_have_zero_difference(settings: Settings) -> None:
    db = Database(":memory:")
    broker = PaperBroker(db, settings.paper, settings.fees)
    for ticker, result in [("M", "yes"), ("N", "no")]:
        buy(broker, ticker)
        broker.settle_market(ticker, result, None)
    report = exit_comparison_report(db, broker.account_id, VERSION)
    assert report["compared_trades"] == 2
    assert report["actual_pnl"] == report["hold_pnl"] == 1.66
    assert report["early_exit_trades"] == report["difference"] == 0
    assert report["mean_difference_ci95"] == (0, 0)


def test_pending_positions_and_old_versions_never_inflate_comparison(settings: Settings) -> None:
    db = Database(":memory:")
    broker = PaperBroker(db, settings.paper, settings.fees)
    sell(broker, buy(broker))
    buy(broker, "OPEN")
    buy(broker, "OLD", version="old")
    broker.settle_market("OLD", "yes", None)
    other = PaperBroker(db, settings.paper, settings.fees, account_name="shadow:other")
    buy(other, "OTHER")
    other.settle_market("OTHER", "yes", None)
    with db.session() as s:
        # A provisional outcome must not count an early-realised win yet.
        s.add(
            MarketRow(
                ticker="M",
                event_ticker="E",
                series_ticker="S",
                status="determined",
                result="yes",
                first_seen_ts=1,
                updated_ts=2,
            )
        )
    report = exit_comparison_report(db, broker.account_id, VERSION)
    assert report["pending_trades"] == 2 and report["open_trades"] == 1
    assert report["compared_trades"] == 0
    assert report["actual_pnl"] == report["hold_pnl"] == report["difference"] == 0
    assert exit_comparison_report(db, None, VERSION)["pending_trades"] == 0


def test_dashboard_reports_current_observe_variant_and_preserves_main(settings: Settings) -> None:
    settings.strategies.tennis_model_variants = {
        "edge_exit": {"exit_edge": "0.03", "mode": "paper"}
    }
    strategies = {s.name: s for s in build_strategies(settings)}
    variant = strategies["tennis_model_edge_exit"]
    assert variant.mode == "observe"
    db = Database(settings.db_path)
    broker = PaperBroker(
        db, settings.paper, settings.fees, account_name="shadow:tennis_model_edge_exit"
    )
    buy(broker, version=variant.version_tag)
    broker.settle_market("M", "yes", None)
    client = TestClient(create_app(settings, db))
    response = client.get("/api/research/early-exits")
    assert response.status_code == 200
    (report,) = response.json()["strategies"]
    assert report["strategy"] == "tennis_model_edge_exit" and report["mode"] == "observe"
    assert report["compared_trades"] == 1 and report["difference"] == 0
    assert client.get("/api/trades").json()["trades"] == []
