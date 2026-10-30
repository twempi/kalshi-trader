"""Resilient client for the Kalshi WebSocket API (https://docs.kalshi.com/websockets).

* Authenticated handshake (required by Kalshi even for public channels).
* Automatic reconnection with exponential backoff + jitter; a fresh signature per attempt.
* Desired subscriptions are remembered and replayed after every reconnect.
* Per-subscription ``seq`` tracking: duplicates/out-of-order messages are dropped, gaps are
  reported via ``on_gap`` so the order book can be resynchronised.
* A silent connection (no frames for ``stale_after_s``) is treated as dead and recycled.

The client only sends subscription-management commands; it has no order functionality.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

from kalshi_trader.api.auth import KalshiSigner
from kalshi_trader.logging_setup import get_logger

log = get_logger("kalshi.ws")

MessageHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
GapHandler = Callable[[str, list[str]], Awaitable[None] | None]
MARKET_SCOPED_CHANNELS = frozenset({"orderbook_delta", "ticker", "trade"})
TERMINAL_ERROR_CODES = frozenset({10, 25})


@dataclass
class WsHealth:
    connected: bool = False
    connects: int = 0
    disconnects: int = 0
    messages: int = 0
    duplicates_dropped: int = 0
    gaps_detected: int = 0
    errors: int = 0
    last_message_at: float | None = None
    last_connected_at: float | None = None
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class _Subscription:
    channel: str
    tickers: set[str] = field(default_factory=set)  # empty set => all markets
    sid: int | None = None


class KalshiWebSocketClient:
    def __init__(
        self,
        url: str,
        signer: KalshiSigner,
        on_message: MessageHandler,
        on_gap: GapHandler | None = None,
        *,
        connect: Callable[..., Any] = websockets.connect,
        backoff_initial_s: float = 1.0,
        backoff_max_s: float = 60.0,
        stale_after_s: float = 45.0,
        ping_interval_s: float = 10.0,
    ) -> None:
        self._url = url
        self._path = urlsplit(url).path or "/trade-api/ws/v2"
        self._signer = signer
        self._on_message = on_message
        self._on_gap = on_gap
        self._connect = connect
        self._backoff_initial = backoff_initial_s
        self._backoff_max = backoff_max_s
        self._stale_after = stale_after_s
        self._ping_interval = ping_interval_s
        self._subs: dict[str, _Subscription] = {}
        self._seq: dict[int, int] = {}
        self._pending: dict[int, str] = {}  # command id -> channel
        self._next_id = 1
        self._ws: Any = None
        self._send_lock = asyncio.Lock()
        self.health = WsHealth()

    # ---------------------------------------------------------------- public
    async def set_markets(self, channel: str, tickers: Iterable[str] | None) -> None:
        """Declare the markets wanted on `channel` (None/empty => all, where allowed)."""
        wanted = set(tickers or ())
        sub = self._subs.get(channel)
        if sub is None:
            self._subs[channel] = _Subscription(channel, wanted)
            if self.health.connected:
                await self._subscribe(self._subs[channel])
            return
        added, removed = wanted - sub.tickers, sub.tickers - wanted
        sub.tickers = wanted
        if not self.health.connected or sub.sid is None:
            return
        if added:
            await self._update(sub, sorted(added), "add_markets")
        if removed:
            await self._update(sub, sorted(removed), "delete_markets")

    async def request_snapshot(self, channel: str, tickers: list[str]) -> None:
        sub = self._subs.get(channel)
        if sub is not None and sub.sid is not None and self.health.connected:
            await self._update(sub, tickers, "get_snapshot")

    async def run(self, stop: asyncio.Event) -> None:
        delay = self._backoff_initial
        while not stop.is_set():
            started = time.monotonic()
            try:
                await self._session(stop)
            except (
                ConnectionClosed,
                InvalidHandshake,
                InvalidStatus,
                OSError,
                TimeoutError,
            ) as exc:
                self.health.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.warning("ws_disconnected", error=self.health.last_error)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the feed alive on unexpected errors
                self.health.errors += 1
                self.health.last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log.error("ws_unexpected_error", error=self.health.last_error)
            finally:
                if self.health.connected:
                    self.health.disconnects += 1
                self.health.connected = False
                self._ws = None
            if stop.is_set():
                break
            if time.monotonic() - started > 60:
                delay = self._backoff_initial  # the connection was healthy for a while
            sleep_for = min(self._backoff_max, delay) * random.uniform(0.5, 1.5)
            log.info("ws_reconnecting", delay_s=round(sleep_for, 1))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=sleep_for)
            delay = min(self._backoff_max, delay * 2)

    # -------------------------------------------------------------- internal
    async def _session(self, stop: asyncio.Event) -> None:
        headers = self._signer.headers("GET", self._path)
        async with self._connect(
            self._url,
            additional_headers=headers,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_interval * 2,
            max_size=8 * 1024 * 1024,
            open_timeout=15,
        ) as ws:
            self._ws = ws
            self._seq.clear()
            self._pending.clear()
            for sub in self._subs.values():
                sub.sid = None
            self.health.connected = True
            self.health.connects += 1
            self.health.last_connected_at = time.time()
            log.info("ws_connected", url=self._url)
            for sub in list(self._subs.values()):
                await self._subscribe(sub)
            while not stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._stale_after)
                except TimeoutError:
                    log.warning("ws_stale_connection", silent_s=self._stale_after)
                    return
                await self._handle_raw(raw)

    async def _send(self, payload: dict[str, Any]) -> int:
        async with self._send_lock:
            cmd_id = self._next_id
            self._next_id += 1
            payload = {"id": cmd_id, **payload}
            if self._ws is None:
                raise ConnectionError("websocket not connected")
            await self._ws.send(json.dumps(payload))
            return cmd_id

    async def _subscribe(self, sub: _Subscription) -> None:
        params: dict[str, Any] = {"channels": [sub.channel]}
        if sub.tickers:
            params["market_tickers"] = sorted(sub.tickers)
        elif sub.channel == "orderbook_delta":
            return  # orderbook requires explicit markets; wait until some are set
        cmd_id = await self._send({"cmd": "subscribe", "params": params})
        self._pending[cmd_id] = sub.channel

    async def _update(self, sub: _Subscription, tickers: list[str], action: str) -> None:
        if not tickers:
            return
        await self._send(
            {
                "cmd": "update_subscription",
                "params": {"sids": [sub.sid], "market_tickers": tickers, "action": action},
            }
        )

    async def _handle_raw(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            self.health.errors += 1
            log.warning("ws_invalid_json", sample=str(raw)[:200])
            return
        if not isinstance(message, dict):
            return
        self.health.messages += 1
        self.health.last_message_at = time.time()
        mtype = message.get("type")

        if mtype == "subscribed":
            msg = message.get("msg") or {}
            channel = msg.get("channel") or self._pending.pop(message.get("id"), None)
            sid = msg.get("sid")
            if channel in self._subs and isinstance(sid, int):
                self._subs[channel].sid = sid
                self._seq.pop(sid, None)
            return
        if mtype == "error":
            self.health.errors += 1
            msg = message.get("msg") or {}
            code = msg.get("code")
            log.warning("ws_error_message", code=code, error=str(msg.get("msg"))[:200])
            if code in TERMINAL_ERROR_CODES:
                await self._resubscribe_all()
            return

        sid, seq = message.get("sid"), message.get("seq")
        if isinstance(sid, int) and isinstance(seq, int):
            last = self._seq.get(sid)
            if last is not None and seq <= last:
                self.health.duplicates_dropped += 1
                return
            self._seq[sid] = seq
            if last is not None and seq > last + 1:
                self.health.gaps_detected += 1
                channel = next((c for c, s in self._subs.items() if s.sid == sid), None)
                log.warning("ws_sequence_gap", sid=sid, expected=last + 1, got=seq, channel=channel)
                if channel and self._on_gap is not None:
                    tickers = sorted(self._subs[channel].tickers)
                    result = self._on_gap(channel, tickers)
                    if asyncio.iscoroutine(result):
                        await result
        if mtype in ("ok", "unsubscribed", "list_subscriptions"):
            return
        result = self._on_message(message)
        if asyncio.iscoroutine(result):
            await result

    async def _resubscribe_all(self) -> None:
        for sub in self._subs.values():
            if sub.sid is not None:
                await self._send({"cmd": "unsubscribe", "params": {"sids": [sub.sid]}})
                self._seq.pop(sub.sid, None)
                sub.sid = None
            await self._subscribe(sub)
