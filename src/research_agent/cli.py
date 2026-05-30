from __future__ import annotations

import asyncio
import sys

from .config import get_settings
from .factory import build_orchestrator


async def _run(question: str) -> None:
    orchestrator = build_orchestrator(get_settings())
    state = await orchestrator.run(question)

    print("\n=== EXECUTION TRACE ===")
    for t in state.trace:
        print(f"[{t.agent:<11}] {t.action}: {t.detail}")

    print("\n=== REPORT ===")
    if state.report:
        print(f"\n{state.report.summary}\n")
        print(state.report.body, "\n")
        if state.report.citations:
            print("Sources:")
            for i, s in enumerate(state.report.citations, start=1):
                loc = f" ({s.locator})" if s.locator else ""
                print(f"  [{i}] {s.title or 'source'}{loc}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m research_agent.cli "your research question"')
        raise SystemExit(2)
    asyncio.run(_run(" ".join(sys.argv[1:])))


if __name__ == "__main__":
    main()
