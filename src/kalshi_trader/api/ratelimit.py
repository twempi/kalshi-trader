"""Client-side token bucket so we stay well inside Kalshi's server-side budgets."""

from __future__ import annotations

import asyncio
import time


class AsyncTokenBucket:
    """Token bucket with additive-increase / multiplicative-decrease pacing.

    Kalshi's 429s carry no ``Retry-After`` header and some endpoints cost more tokens than
    others, so a fixed client rate can still be too fast. A 429 halves the effective rate
    (never below ``min_rate_per_s``) and drops any banked burst; each success restores
    2% of the configured rate.
    """

    def __init__(self, rate_per_s: float, capacity: float, min_rate_per_s: float = 0.5) -> None:
        if rate_per_s <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.max_rate = rate_per_s
        self.rate = rate_per_s
        self.min_rate = min(min_rate_per_s, rate_per_s)
        self.capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self, tokens: float = 1.0) -> None:
        if tokens > self.capacity:
            raise ValueError("requested more tokens than bucket capacity")
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)

    def throttle(self) -> None:
        """The server rate-limited us: slow down and stop bursting."""
        self._refill()
        self.rate = max(self.min_rate, self.rate / 2)
        self._tokens = min(self._tokens, 0.0)

    def recover(self) -> None:
        self.rate = min(self.max_rate, self.rate + self.max_rate * 0.02)
