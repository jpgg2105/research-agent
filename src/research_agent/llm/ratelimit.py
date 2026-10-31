from __future__ import annotations

import asyncio
import time

_LIMITERS: dict[str, "RateLimiter"] = {}


class RateLimiter:
    """Spaces calls to at most ``rpm`` per minute. Shared by every client that
    uses the same endpoint and model, because provider quotas are per project
    and model, not per client object."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


def limiter_for(key: str, rpm: float | None) -> RateLimiter | None:
    if not rpm or rpm <= 0:
        return None
    lim = _LIMITERS.get(key)
    if lim is None or lim.interval < 60.0 / rpm:   # keep the strictest rate asked for
        lim = _LIMITERS[key] = RateLimiter(rpm)
    return lim
