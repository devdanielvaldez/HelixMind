import json
import os
import unittest
from unittest.mock import patch

import httpx

from medai_science.llm import ModelCompletion
from medai_science.budgeted_model import SessionBudgetedJSONProvider


class FakeProvider:
    provider_name = "test"
    model = "model-1"
    max_output_tokens = 128
    base_url = "http://127.0.0.1:11434/v1"

    def __init__(self, order, *, fail=False):
        self.order = order
        self.fail = fail

    async def complete_json(self, *, system, prompt, schema_name, schema):
        self.order.append("provider")
        if self.fail:
            raise RuntimeError("provider disconnected")
        return ModelCompletion({"ok": True}, "test", "model-1", "a" * 64, "b" * 64,
                               {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}, 128)


class SessionBudgetedProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_reserves_before_call_and_settles_to_provider_usage(self):
        order = []
        settlements = []

        async def api(request):
            if request.url.path.endswith("/budget/reservations"):
                order.append("reserve")
                return httpx.Response(200, json={})
            order.append("settle")
            settlements.append(json.loads(request.content)["actual_credits"])
            return httpx.Response(200, json={"state": "settled"})

        transport = httpx.MockTransport(api)
        provider = SessionBudgetedJSONProvider(FakeProvider(order), "RES-1", "objective.normalize", transport)
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "1000"}), patch("medai_science.budgeted_model.api_headers", return_value={}):
            completion = await provider.complete_json(system="system", prompt="prompt", schema_name="test", schema={"type": "object"})
        self.assertEqual(completion.value, {"ok": True})
        self.assertEqual(order, ["reserve", "provider", "settle"])
        self.assertEqual(settlements, [1])

    async def test_failed_call_retains_full_reservation_as_uncertain(self):
        order = []
        settlements = []

        async def api(request):
            if request.url.path.endswith("/budget/reservations"):
                order.append("reserve")
                return httpx.Response(200, json={})
            order.append("settle")
            settlements.append(json.loads(request.content)["actual_credits"])
            return httpx.Response(200, json={"state": "uncertain"})

        provider = SessionBudgetedJSONProvider(FakeProvider(order, fail=True), "RES-1", "objective.normalize", httpx.MockTransport(api))
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "1000"}), patch("medai_science.budgeted_model.api_headers", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "provider disconnected"):
                await provider.complete_json(system="system", prompt="prompt", schema_name="test", schema={"type": "object"})
        self.assertEqual(order, ["reserve", "provider", "settle"])
        self.assertEqual(settlements, [None])

    async def test_budget_rejection_prevents_provider_call(self):
        order = []

        async def api(request):
            return httpx.Response(402, json={"error": "budget exhausted"})

        provider = SessionBudgetedJSONProvider(FakeProvider(order), "RES-1", "objective.normalize", httpx.MockTransport(api))
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "1000"}), patch("medai_science.budgeted_model.api_headers", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "insufficient session"):
                await provider.complete_json(system="system", prompt="prompt", schema_name="test", schema={"type": "object"})
        self.assertEqual(order, [])


if __name__ == "__main__":
    unittest.main()
