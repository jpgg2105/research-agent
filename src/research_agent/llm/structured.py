from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import LLMClient

T = TypeVar("T", bound=BaseModel)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences a model may wrap its JSON in."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


async def structured_complete(
    client: LLMClient,
    *,
    system: str,
    user: str,
    response_model: Type[T],
    max_retries: int = 2,
    temperature: float = 0.2,
) -> T:
    """Get a validated Pydantic object out of any provider.

    We instruct the model with the target JSON Schema and validate the result.
    On a parse/validation failure we feed the error back and retry, up to
    ``max_retries`` times. This keeps a single code path across providers; in
    production you might prefer each provider's native structured-output mode.
    """
    schema = json.dumps(response_model.model_json_schema(), indent=2)
    instruction = (
        "Respond with ONLY a valid JSON object that conforms to this JSON Schema. "
        "Do not include markdown fences, comments, or any prose outside the JSON.\n\n"
        f"JSON Schema:\n{schema}"
    )
    full_system = f"{system}\n\n{instruction}"

    attempt_user = user
    last_error: str | None = None
    for _ in range(max_retries + 1):
        raw = await client.complete(system=full_system, user=attempt_user, temperature=temperature)
        candidate = _strip_fences(raw)
        try:
            return response_model.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            attempt_user = (
                f"{user}\n\nYour previous response could not be parsed into the required "
                f"schema. Error:\n{last_error}\n\nReturn corrected JSON only."
            )
    raise ValueError(
        f"Failed to obtain valid structured output after {max_retries + 1} attempts: {last_error}"
    )
