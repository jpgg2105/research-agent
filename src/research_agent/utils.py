from __future__ import annotations

from .models import Source


def dedupe_sources(sources: list[Source]) -> list[Source]:
    """Stable de-duplication of sources by (title, locator, snippet-prefix)."""
    seen: set[tuple] = set()
    out: list[Source] = []
    for s in sources:
        key = (s.title, s.locator, (s.snippet or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
