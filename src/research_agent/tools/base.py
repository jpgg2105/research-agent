from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Source


@runtime_checkable
class RetrievalTool(Protocol):
    """A source of evidence. Implementations return a list of Source objects."""

    name: str

    async def search(self, query: str, *, top_k: int = 5) -> list[Source]: ...
