from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    """Running token count for one client instance: calls, input and output tokens.

    Every client records what the provider reports for each call, so a caller can
    read the cost of one question by building a client per question (the
    benchmark does this) or by diffing ``snapshot()`` before and after.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def snapshot(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
