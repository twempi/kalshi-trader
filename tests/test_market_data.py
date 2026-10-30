"""Order book maths, hub ownership rules and WebSocket resilience (local test server)."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal as D

import pytest
import websockets
from cryptography.hazmat.primitives.asymmetric import ed25519

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.api.schemas import OrderbookFp
from kalshi_trader.api.ws import KalshiWebSocketClient
from kalshi_trader.market_data.feeds import MarketDataHub
from kalshi_trader.market_data.orderbook import OrderBook


def make_book() -> OrderBook:
    book = OrderBook("T")
    book.apply_snapshot(
        [(D("0.40"), D("10")), (D("0.42"), D("5"))],
        [(D("0.55"), D("7")), (D("0.50"), D("20"))],
        source="test",
        received_at=100.0,
    )
    return book


def test_asks_derive_from_opposite_bids() -> None:
    book = make_book()
    assert book.best_bid("yes") == (D("0.42"), D("5"))
    assert book.best_ask("yes") == (D("0.45"), D("7"))  # 1 - best NO bid 0.55
    assert book.asks("yes") == [(D("0.45"), D("7")), (D("0.50"), D("20"))]
    assert book.best_ask("no") == (D("0.58"), D("5"))  # 1 - best YES bid 0.42
    assert book.spread("yes") == D("0.03")
    assert book.mid("yes") == D("0.435")
    assert book.ask_depth("yes", D("0.50")) == D("27")
    assert book.bid_depth("yes", D("0.41")) == D("5")


def test_deltas_update_and_remove_levels() -> None:
    book = make_book()
    book.apply_delta("no", D("0.55"), D("-7"), received_at=101.0)
    assert book.best_ask("yes") == (D("0.50"), D("20"))
    book.apply_delta("yes", D("0.43"), D("3"), received_at=102.0)
    assert book.best_bid("yes") == (D("0.43"), D("3"))
    assert book.age_s(now=105.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        book.apply_delta("yes", D("1.20"), D("1"))


def test_delta_below_zero_means_out_of_sync() -> None:
    book = make_book()
    with pytest.raises(ValueError, match="out of sync"):
        book.apply_delta("no", D("0.55"), D("-8"))  # only 7 resting
    assert not book.valid


def test_hub_queues_resync_when_ws_book_goes_out_of_sync() -> None:
    hub = MarketDataHub()
    hub.handle_ws_message(
        {
            "type": "orderbook_snapshot",
            "sid": 1,
            "seq": 1,
            "msg": {"market_ticker": "MKT-C", "yes_dollars_fp": [["0.30", "10"]]},
        }
    )
    hub.handle_ws_message(
        {
            "type": "orderbook_delta",
            "sid": 1,
            "seq": 2,
            "msg": {
                "market_ticker": "MKT-C",
                "price_dollars": "0.30",
                "delta_fp": "-11",
                "side": "yes",
            },
        }
    )
    assert hub.invalid_messages == 1 and not hub.ws_owns("MKT-C")
    assert hub.resync_requests == {"MKT-C"}
    rest = OrderbookFp.model_validate({"yes_dollars": [["0.31", "2"]], "no_dollars": []})
    assert hub.apply_rest_snapshot("MKT-C", rest) is True  # REST covers it meanwhile


def test_signature_changes_only_with_book() -> None:
    a, b = make_book(), make_book()
    assert a.signature(5) == b.signature(5)
    b.apply_delta("yes", D("0.40"), D("1"))
    assert a.signature(5) != b.signature(5)


def test_hub_rest_snapshot_not_mixed_into_ws_book() -> None:
    hub = MarketDataHub()
    seen: list[str] = []
    hub.add_book_listener(lambda t, b: seen.append(t))
    hub.handle_ws_message(
        {
            "type": "orderbook_snapshot",
            "sid": 1,
            "seq": 1,
            "msg": {
                "market_ticker": "MKT-A",
                "yes_dollars_fp": [["0.30", "10"]],
                "no_dollars_fp": [["0.60", "4"]],
            },
        }
    )
    assert hub.ws_owns("MKT-A")
    rest = OrderbookFp.model_validate({"yes_dollars": [["0.10", "1"]], "no_dollars": []})
    assert hub.apply_rest_snapshot("MKT-A", rest) is False
    assert hub.book("MKT-A").best_bid("yes") == (D("0.30"), D("10"))
    hub.handle_ws_message(
        {
            "type": "orderbook_delta",
            "sid": 1,
            "seq": 2,
            "msg": {
                "market_ticker": "MKT-A",
                "price_dollars": "0.30",
                "delta_fp": "-10.00",
                "side": "yes",
            },
        }
    )
    assert hub.book("MKT-A").best_bid("yes") is None
    hub.on_ws_gap("orderbook_delta", ["MKT-A"])  # gap: book invalid, REST may take over
    assert not hub.ws_owns("MKT-A")
    assert hub.apply_rest_snapshot("MKT-A", rest) is True
    assert seen == ["MKT-A", "MKT-A", "MKT-A"]


def test_hub_ignores_deltas_before_snapshot_and_rejects_garbage() -> None:
    hub = MarketDataHub()
    hub.handle_ws_message(
        {
            "type": "orderbook_delta",
            "msg": {
                "market_ticker": "MKT-B",
                "price_dollars": "0.30",
                "delta_fp": "5",
                "side": "yes",
            },
        }
    )
    assert "MKT-B" not in hub.books or not hub.books["MKT-B"].valid
    hub.handle_ws_message({"type": "orderbook_snapshot", "msg": {"market_ticker": "../x"}})
    hub.handle_ws_message(
        {
            "type": "orderbook_snapshot",
            "msg": {"market_ticker": "MKT-C", "yes_dollars_fp": [["7", "1"]], "no_dollars_fp": []},
        }
    )
    assert hub.invalid_messages == 2


class FakeKalshiServer:
    """Minimal stand-in for the Kalshi WS endpoint, scripted per connection."""

    def __init__(self) -> None:
        self.connections = 0
        self.subscribes: list[dict] = []
        self.headers: list[dict] = []

    async def handler(self, ws) -> None:
        self.connections += 1
        self.headers.append(dict(ws.request.headers))
        conn = self.connections
        raw = await ws.recv()
        cmd = json.loads(raw)
        self.subscribes.append(cmd)
        await ws.send(
            json.dumps(
                {
                    "id": cmd["id"],
                    "type": "subscribed",
                    "msg": {"channel": "orderbook_delta", "sid": 7},
                }
            )
        )
        snap = {
            "type": "orderbook_snapshot",
            "sid": 7,
            "seq": 1,
            "msg": {
                "market_ticker": "MKT-A",
                "yes_dollars_fp": [["0.40", "10"]],
                "no_dollars_fp": [["0.50", "10"]],
            },
        }
        await ws.send(json.dumps(snap))
        delta = {
            "type": "orderbook_delta",
            "sid": 7,
            "seq": 2,
            "msg": {
                "market_ticker": "MKT-A",
                "price_dollars": "0.41",
                "delta_fp": "3",
                "side": "yes",
            },
        }
        await ws.send(json.dumps(delta))
        await ws.send(json.dumps(delta))  # duplicate seq 2
        if conn == 1:
            gap = {**delta, "seq": 5}  # seq 3-4 missing
            await ws.send(json.dumps(gap))
            await ws.close()  # simulate a dropped connection
            return
        await asyncio.sleep(0.2)
        # then go silent to trigger the stale-connection guard
        await asyncio.sleep(5)


async def test_ws_reconnects_resubscribes_and_tracks_sequence() -> None:
    server = FakeKalshiServer()
    async with websockets.serve(server.handler, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        hub = MarketDataHub()
        gaps: list[tuple[str, list[str]]] = []

        def on_gap(channel: str, tickers: list[str]) -> None:
            gaps.append((channel, tickers))
            hub.on_ws_gap(channel, tickers)

        signer = KalshiSigner("kid", ed25519.Ed25519PrivateKey.generate())
        client = KalshiWebSocketClient(
            f"ws://127.0.0.1:{port}/trade-api/ws/v2",
            signer,
            hub.handle_ws_message,
            on_gap,
            backoff_initial_s=0.05,
            backoff_max_s=0.1,
            stale_after_s=1.0,
            ping_interval_s=5,
        )
        await client.set_markets("orderbook_delta", ["MKT-A"])
        stop = asyncio.Event()
        task = asyncio.create_task(client.run(stop))
        for _ in range(100):
            if server.connections >= 2 and client.health.messages >= 8:
                break
            await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    assert server.connections >= 2  # reconnected after the drop
    assert all(s["params"]["market_tickers"] == ["MKT-A"] for s in server.subscribes)
    assert all("kalshi-access-signature" in {k.lower() for k in h} for h in server.headers)
    assert client.health.duplicates_dropped >= 2
    assert client.health.gaps_detected == 1 and gaps == [("orderbook_delta", ["MKT-A"])]
    book = hub.book("MKT-A")
    assert book.valid and book.best_bid("yes") == (D("0.41"), D("3"))  # duplicate not re-applied
