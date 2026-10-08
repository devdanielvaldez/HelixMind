"""Session-budget enforcement around provider-neutral structured model calls."""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any

import httpx

from medai_science.llm import JSONModelProvider
from medai_science.mcp_auth import API, api_headers
from medai_science.model_usage import ModelUsageBudgetError, estimate_model_reservation, meter_model_usage


class SessionBudgetedJSONProvider:
    """Reserve a bounded worst-case token allowance before each provider request."""

    def __init__(self, provider: JSONModelProvider, session_id: str, task_id: str, transport: httpx.AsyncBaseTransport | None = None):
        self._provider = provider
        self._session_id = session_id
        self._task_id = task_id
        self._transport = transport

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def _settle(self, reservation_id: str, actual_credits: int | None) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=API, headers=api_headers(), timeout=15, transport=self._transport) as client:
            response = await client.post(
                f"/v1/research/{self._session_id}/budget/reservations/{reservation_id}/settle",
                json={"actual_credits": actual_credits},
            )
            response.raise_for_status()
            return response.json()

    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict[str, Any]) -> Any:
        max_output = getattr(self._provider, "max_output_tokens", 4096)
        estimate = estimate_model_reservation(system, prompt, schema, max_output)
        schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        request_sha256 = hashlib.sha256(system.encode("utf-8") + b"\0" + prompt.encode("utf-8") + b"\0" + schema_bytes).hexdigest()
        reservation_id = uuid.uuid4().hex
        provider_name = str(getattr(self._provider, "provider_name", "model"))[:24]
        model_name = str(getattr(self._provider, "model", "unknown"))[:32]
        async with httpx.AsyncClient(base_url=API, headers=api_headers(), timeout=15, transport=self._transport) as client:
            response = await client.post(
                f"/v1/research/{self._session_id}/budget/reservations",
                json={
                    "task_id": self._task_id,
                    "reservation_id": reservation_id,
                    "credits": estimate["credits"],
                    "reason": f"pre-call ceiling={estimate['token_ceiling']} rate={estimate['tokens_per_credit']}/credit {provider_name}/{model_name} request_sha256={request_sha256}",
                },
            )
            if response.status_code == 402:
                raise ModelUsageBudgetError("insufficient session or specialist credits for the model reservation")
            response.raise_for_status()
        try:
            completion = await self._provider.complete_json(system=system, prompt=prompt, schema_name=schema_name, schema=schema)
        except BaseException:
            # A disconnected provider can have accepted the request; retain the full reservation when cost is unknown.
            try:
                await asyncio.shield(self._settle(reservation_id, None))
            except Exception as settle_error:
                raise ModelUsageBudgetError("model call failed and its conservative budget reservation could not be settled") from settle_error
            raise
        try:
            metering = meter_model_usage(completion.usage)
            actual = metering["credits"]
            if not isinstance(actual, int) or actual > estimate["credits"]:
                raise ModelUsageBudgetError("provider-reported model usage exceeds the pre-call reservation")
            await self._settle(reservation_id, actual)
        except Exception as error:
            try:
                await asyncio.shield(self._settle(reservation_id, None))
            except Exception:
                pass
            if isinstance(error, ModelUsageBudgetError):
                raise
            raise ModelUsageBudgetError(f"model usage could not settle its pre-call reservation: {type(error).__name__}") from error
        return completion
