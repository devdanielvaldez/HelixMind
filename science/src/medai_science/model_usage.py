"""Validate provider-reported model token usage and convert it to session credits."""
from __future__ import annotations

import json
import os
from typing import Any


class ModelUsageError(ValueError):
    """A model response did not provide trustworthy token counts for budget metering."""


class ModelUsageBudgetError(RuntimeError):
    """Model usage could not be charged to the durable session budget."""


def model_tokens_per_credit(tokens_per_credit: int | None = None) -> int:
    if tokens_per_credit is None:
        raw_rate = os.getenv("MEDAI_MODEL_TOKENS_PER_CREDIT", "1000")
        try:
            tokens_per_credit = int(raw_rate)
        except ValueError as error:
            raise ModelUsageError("MEDAI_MODEL_TOKENS_PER_CREDIT must be an integer") from error
    if not 1 <= tokens_per_credit <= 1_000_000_000:
        raise ModelUsageError("tokens_per_credit must be 1..1000000000")
    return tokens_per_credit


def estimate_model_reservation(system: str, prompt: str, schema: Any, max_output_tokens: int) -> dict[str, int]:
    """Conservatively reserve one token per UTF-8 request byte, output ceiling, and protocol overhead."""
    if not isinstance(system, str) or not isinstance(prompt, str):
        raise ModelUsageError("model system prompt and prompt must be strings")
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens < 1:
        raise ModelUsageError("model output-token ceiling must be positive")
    try:
        schema_bytes = len(json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise ModelUsageError("model schema cannot be serialized for budget reservation") from error
    request_bytes = len(system.encode("utf-8")) + len(prompt.encode("utf-8")) + schema_bytes + 512
    token_ceiling = request_bytes + max_output_tokens
    rate = model_tokens_per_credit()
    return {"token_ceiling": token_ceiling, "tokens_per_credit": rate, "credits": (token_ceiling + rate - 1) // rate}


def meter_model_usage(usage: Any, tokens_per_credit: int | None = None) -> dict[str, int | None]:
    """Return validated token counts and rounded-up credits; 0 tokens cost 0 credits."""
    if not isinstance(usage, dict):
        raise ModelUsageError("provider response omitted token usage; model result cannot be retained")

    def read_count(key: str) -> int | None:
        value = usage.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ModelUsageError(f"provider reported invalid {key}")
        return value

    prompt_tokens = read_count("prompt_tokens")
    completion_tokens = read_count("completion_tokens")
    total_tokens = read_count("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        raise ModelUsageError("provider response omitted token usage; model result cannot be retained")
    if (prompt_tokens is None) != (completion_tokens is None):
        raise ModelUsageError("provider reported incomplete prompt/completion token usage")
    if prompt_tokens is not None and completion_tokens is not None:
        component_total = prompt_tokens + completion_tokens
        if total_tokens is not None and total_tokens != component_total:
            raise ModelUsageError("provider token usage total does not match prompt plus completion")
        total_tokens = component_total
    if total_tokens is None:
        raise ModelUsageError("provider response did not report a total token count")
    tokens_per_credit = model_tokens_per_credit(tokens_per_credit)
    credits = (total_tokens + tokens_per_credit - 1) // tokens_per_credit
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_credit": tokens_per_credit,
        "credits": credits,
    }
