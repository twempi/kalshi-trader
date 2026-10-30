"""REST client: read-only guard, retries, rate limiting, pagination, network failures."""

from __future__ import annotations

import json

import httpx
import pytest

from kalshi_trader.api.ratelimit import AsyncTokenBucket
from kalshi_trader.api.rest import KalshiAPIError, KalshiRestClient, ReadOnlyViolation
from kalshi_trader.config import KalshiSettings
from tests.conftest import load_fixture


def client_with(handler, **kwargs) -> KalshiRestClient:
    settings = KalshiSettings(max_requests_per_second=1000, burst=1000, max_retries=3)
    return KalshiRestClient(
        settings,
        transport=httpx.MockTransport(handler),
        backoff_base_s=0.001,
        backoff_max_s=0.01,
        **kwargs,
    )


async def test_refuses_non_get_requests() -> None:
    client = client_with(lambda r: httpx.Response(200, json={}))
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        with pytest.raises(ReadOnlyViolation):
            await client._request(method, "/portfolio/orders")
    await client.close()


def test_client_exposes_no_order_methods() -> None:
    import re

    names = [n.lower() for n in dir(KalshiRestClient)]
    forbidden = re.compile(r"create|cancel|amend|submit|place|decrease|transfer|order(?!book)")
    assert not [n for n in names if forbidden.search(n)]


async def test_retries_server_errors_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=load_fixture("exchange_status.json"))

    client = client_with(handler)
    status = await client.get_exchange_status()
    assert status.exchange_active in (True, False)
    assert calls["n"] == 3 and client.health.retries == 2
    assert client.health.consecutive_failures == 0
    await client.close()


async def test_recovers_from_transport_errors_and_429() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("network down", request=request)
        if calls["n"] == 2:
            return httpx.Response(429, json={"error": "too many requests"})
        return httpx.Response(200, json=load_fixture("exchange_status.json"))

    client = client_with(handler)
    await client.get_exchange_status()
    assert client.health.rate_limited == 1 and calls["n"] == 3
    assert client._bucket.rate == pytest.approx(1000 / 2 + 1000 * 0.02)  # halved, recovering
    await client.close()


async def test_token_bucket_backs_off_on_429_and_recovers() -> None:
    bucket = AsyncTokenBucket(8, 10, min_rate_per_s=1)
    for _ in range(10):
        bucket.throttle()
    assert bucket.rate == 1  # never below the floor
    await bucket.acquire(0.001)  # bursting is disabled right after a 429, but still works
    for _ in range(100):
        bucket.recover()
    assert bucket.rate == 8  # never above the configured rate


async def test_gives_up_after_max_retries() -> None:
    client = client_with(lambda r: httpx.Response(502))
    with pytest.raises(KalshiAPIError) as exc:
        await client.get_exchange_status()
    assert exc.value.status == 502
    assert client.health.failures == 4  # 1 try + 3 retries
    await client.close()


async def test_does_not_retry_client_errors() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"error": "not found"})

    client = client_with(handler)
    with pytest.raises(KalshiAPIError):
        await client.get_market("KXATPMATCH-NOPE")
    assert calls["n"] == 1
    await client.close()


async def test_pagination_follows_cursor_and_skips_invalid_records() -> None:
    page1 = load_fixture("markets_atp.json")["markets"][:2]
    page2 = [*load_fixture("markets_atp.json")["markets"][2:], {"ticker": "../bad"}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") == "abc":
            return httpx.Response(200, json={"markets": page2, "cursor": ""})
        return httpx.Response(200, json={"markets": page1, "cursor": "abc"})

    client = client_with(handler)
    markets = await client.get_markets(series_ticker="KXATPMATCH", status="open")
    assert [m.ticker for m in markets] == [m["ticker"] for m in page1 + page2[:2]]
    await client.close()


async def test_batch_orderbooks_chunk_and_ignore_unexpected_tickers() -> None:
    requested: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        tickers = request.url.params.get_list("tickers")
        requested.append(tickers)
        books = [
            {
                "ticker": t,
                "orderbook_fp": {"yes_dollars": [["0.40", "5"]], "no_dollars": [["0.55", "7"]]},
            }
            for t in tickers
        ]
        books.append({"ticker": "INJECTED-MKT", "orderbook_fp": {}})
        return httpx.Response(200, json={"orderbooks": books})

    client = client_with(handler)
    tickers = [f"KXTEST-{i}" for i in range(150)]
    books = await client.get_orderbooks(tickers)
    assert [len(r) for r in requested] == [100, 50]
    assert set(books) == set(tickers)
    await client.close()


async def test_signs_requests_when_credentials_present() -> None:
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from kalshi_trader.api.auth import KalshiSigner

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"balance": 0})

    client = client_with(handler, signer=KalshiSigner("kid", ed25519.Ed25519PrivateKey.generate()))
    await client.get_balance()
    assert seen["kalshi-access-key"] == "kid" and "kalshi-access-signature" in seen
    await client.close()


async def test_auth_required_endpoint_without_credentials_fails_cleanly() -> None:
    client = client_with(lambda r: httpx.Response(200, json={}))
    with pytest.raises(KalshiAPIError):
        await client.get_balance()
    await client.close()


async def test_invalid_json_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, content=b"<html>oops")
        return httpx.Response(200, content=json.dumps(load_fixture("exchange_status.json")))

    client = client_with(handler)
    await client.get_exchange_status()
    assert calls["n"] == 2
    await client.close()
