from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from medai_science.llm import AnthropicMessagesJSONProvider, GeminiGenerateContentJSONProvider, ModelProviderError, OpenAICompatibleJSONProvider, _validate_schema, configured_model_provider


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["query"], "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 40}}}


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks
        self.consumed = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self):
        return None


class ModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_selects_an_independent_registered_agent_model_policy_and_secret_reference(self):
        policies = {
            "scientific_critic": {
                "base_url": "http://127.0.0.1:11435/v1",
                "model": "critic-local",
                "provider_name": "local-critic",
                "api_key_env": "MEDAI_TEST_CRITIC_KEY",
                "timeout_seconds": 24,
                "max_output_tokens": 1536,
            }
        }
        env = {
            "MEDAI_AGENT_MODEL_POLICIES": json.dumps(policies),
            "MEDAI_MODEL_BASE_URL": "http://127.0.0.1:11434/v1",
            "MEDAI_MODEL_NAME": "shared-model",
            "MEDAI_TEST_CRITIC_KEY": "critic-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            provider = configured_model_provider("scientific_critic")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.base_url, "http://127.0.0.1:11435/v1")
            self.assertEqual(provider.model, "critic-local")
            self.assertEqual(provider.provider_name, "local-critic")
            self.assertEqual(provider.api_key, "critic-secret")
            self.assertEqual(provider.timeout, 24)
            self.assertEqual(provider.max_output_tokens, 1536)
            fallback = configured_model_provider("literature_librarian")
            self.assertEqual(fallback.model, "shared-model")

    async def test_shared_output_token_ceiling_is_validated_and_configured(self):
        common = {"MEDAI_MODEL_BASE_URL": "http://127.0.0.1:11434/v1", "MEDAI_MODEL_NAME": "shared"}
        with patch.dict(os.environ, {**common, "MEDAI_MODEL_MAX_OUTPUT_TOKENS": "2048"}, clear=True):
            self.assertEqual(configured_model_provider().max_output_tokens, 2048)
        for invalid in ("0", "32769", "not-a-number"):
            with patch.dict(os.environ, {**common, "MEDAI_MODEL_MAX_OUTPUT_TOKENS": invalid}, clear=True):
                with self.subTest(limit=invalid), self.assertRaises(ValueError):
                    configured_model_provider()

    async def test_rejects_invalid_agent_model_policies_and_inline_secrets(self):
        invalid = [
            "[]",
            json.dumps({"scientific_critic": {"base_url": "http://127.0.0.1/v1", "model": "m", "api_key": "inline-secret"}}),
            json.dumps({"bad role": {"base_url": "http://127.0.0.1/v1", "model": "m"}}),
        ]
        for raw in invalid:
            with patch.dict(os.environ, {"MEDAI_AGENT_MODEL_POLICIES": raw}, clear=True):
                with self.assertRaises(ValueError):
                    configured_model_provider("scientific_critic")

    async def test_selects_native_anthropic_adapter_for_an_agent_policy(self):
        policy = {"scientific_critic": {"provider_type": "anthropic", "base_url": "https://api.anthropic.com/v1", "model": "claude-test", "api_key_env": "MEDAI_TEST_ANTHROPIC_KEY"}}
        with patch.dict(os.environ, {"MEDAI_AGENT_MODEL_POLICIES": json.dumps(policy), "MEDAI_TEST_ANTHROPIC_KEY": "secret"}, clear=True):
            provider = configured_model_provider("scientific_critic")
        self.assertIsInstance(provider, AnthropicMessagesJSONProvider)
        self.assertEqual(provider.model, "claude-test")
        self.assertEqual(provider.provider_name, "anthropic")

    async def test_native_anthropic_structured_tool_call_is_validated_and_metered(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
            self.assertEqual(request.headers["x-api-key"], "anthropic-secret")
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            body = json.loads(request.content)
            self.assertEqual(body["max_tokens"], 4096)
            self.assertEqual(body["tool_choice"], {"type": "tool", "name": "query_plan"})
            self.assertEqual(body["tools"][0]["input_schema"], SCHEMA)
            return httpx.Response(200, json={"content": [{"type": "tool_use", "name": "query_plan", "input": {"query": "Alzheimer disease"}}], "usage": {"input_tokens": 19, "output_tokens": 7}})

        provider = AnthropicMessagesJSONProvider("https://api.anthropic.com/v1", "claude-test", "anthropic-secret", transport=httpx.MockTransport(handler))
        completion = await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        self.assertEqual(completion.value, {"query": "Alzheimer disease"})
        self.assertEqual(completion.usage, {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26})
        self.assertEqual(completion.max_output_tokens, 4096)
        self.assertEqual(len(completion.response_sha256), 64)

    async def test_native_anthropic_rejects_missing_tool_result_and_missing_key(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}], "usage": {"input_tokens": 1, "output_tokens": 1}}))
        provider = AnthropicMessagesJSONProvider("https://api.anthropic.com/v1", "claude-test", "key", transport=transport)
        with self.assertRaises(ModelProviderError):
            await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        with patch.dict(os.environ, {"MEDAI_AGENT_MODEL_POLICIES": json.dumps({"scientific_critic": {"provider_type": "anthropic", "base_url": "https://api.anthropic.com/v1", "model": "claude-test"}})}, clear=True):
            with self.assertRaisesRegex(ValueError, "require api_key_env"):
                configured_model_provider("scientific_critic")

    async def test_selects_gemini_and_uses_native_json_schema_endpoint_with_usage(self):
        policy = {"scientific_critic": {"provider_type": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta", "model": "gemini-test", "api_key_env": "MEDAI_TEST_GEMINI_KEY"}}
        with patch.dict(os.environ, {"MEDAI_AGENT_MODEL_POLICIES": json.dumps(policy), "MEDAI_TEST_GEMINI_KEY": "gemini-secret"}, clear=True):
            provider = configured_model_provider("scientific_critic")
        self.assertIsInstance(provider, GeminiGenerateContentJSONProvider)

        def handler(request):
            self.assertEqual(str(request.url), "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent")
            self.assertEqual(request.headers["x-goog-api-key"], "gemini-secret")
            body = json.loads(request.content)
            self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")
            self.assertEqual(body["generationConfig"]["maxOutputTokens"], 4096)
            self.assertEqual(body["generationConfig"]["responseSchema"]["required"], ["query"])
            self.assertNotIn("minLength", body["generationConfig"]["responseSchema"]["properties"]["query"])
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"query":"Alzheimer disease"}'}]}}], "usageMetadata": {"promptTokenCount": 19, "candidatesTokenCount": 7, "thoughtsTokenCount": 5, "totalTokenCount": 31}})

        provider.transport = httpx.MockTransport(handler)
        completion = await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        self.assertEqual(completion.value, {"query": "Alzheimer disease"})
        self.assertEqual(completion.usage, {"prompt_tokens": 19, "completion_tokens": 12, "total_tokens": 31})
        self.assertEqual(completion.max_output_tokens, 4096)

    async def test_gemini_missing_usage_cannot_be_used_for_billed_model_work(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"query":"x"}'}]}}]}))
        provider = GeminiGenerateContentJSONProvider("https://generativelanguage.googleapis.com/v1beta", "gemini-test", "key", transport=transport)
        result = await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        self.assertEqual(result.usage, {})

    async def test_all_providers_reject_reported_output_usage_above_requested_ceiling(self):
        ceiling = 4
        cases = [
            (OpenAICompatibleJSONProvider("http://127.0.0.1:11434/v1", "test", max_output_tokens=ceiling, transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"message": {"content": '{\"query\":\"x\"}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6}}))), "openai"),
            (AnthropicMessagesJSONProvider("https://api.anthropic.com/v1", "test", "key", max_output_tokens=ceiling, transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"content": [{"type": "tool_use", "name": "query_plan", "input": {"query": "x"}}], "usage": {"input_tokens": 1, "output_tokens": 5}}))), "anthropic"),
            (GeminiGenerateContentJSONProvider("https://generativelanguage.googleapis.com/v1beta", "test", "key", max_output_tokens=ceiling, transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{\"query\":\"x\"}'}]}}], "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 5, "totalTokenCount": 6}}))), "gemini"),
        ]
        for provider, name in cases:
            with self.subTest(provider=name), self.assertRaisesRegex(ModelProviderError, "above the configured output-token ceiling"):
                await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)

    async def test_all_provider_adapters_reject_nonstandard_nonfinite_json_numbers(self):
        numeric_schema = {"type": "object", "required": ["value"], "properties": {"value": {"type": "number"}}}
        with self.assertRaisesRegex(ValueError, "must be finite"):
            _validate_schema({"value": float("nan")}, numeric_schema)
        providers = [
            OpenAICompatibleJSONProvider(
                "https://model.example/v1", "test", transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json={"choices": [{"message": {"content": '{"value":NaN}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
                ),
            ),
            AnthropicMessagesJSONProvider(
                "https://api.anthropic.com/v1", "test", "key", transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, content=b'{"content":[{"type":"tool_use","name":"numbers","input":{"value":Infinity}}],"usage":{"input_tokens":1,"output_tokens":1}}')
                ),
            ),
            GeminiGenerateContentJSONProvider(
                "https://generativelanguage.googleapis.com/v1beta", "test", "key", transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"value":-Infinity}'}]}}], "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 2}})
                ),
            ),
        ]
        names = ["openai", "anthropic", "gemini"]
        for name, provider in zip(names, providers):
            with self.subTest(provider=name), self.assertRaises(ModelProviderError):
                await provider.complete_json(system="system", prompt="prompt", schema_name="numbers", schema=numeric_schema)

    async def test_sends_json_schema_and_returns_auditable_completion(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), "http://127.0.0.1:11434/v1/chat/completions")
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(body["max_tokens"], 4096)
            self.assertIn('"required":["query"]', body["messages"][0]["content"])
            self.assertEqual(request.headers["authorization"], "Bearer local-secret")
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"query":"Alzheimer disease"}'}}], "usage": {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}})

        provider = OpenAICompatibleJSONProvider("http://127.0.0.1:11434/v1", "local-model", "local-secret", transport=httpx.MockTransport(handler))
        result = await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        self.assertEqual(result.value, {"query": "Alzheimer disease"})
        self.assertEqual(result.usage["total_tokens"], 18)
        self.assertEqual(result.max_output_tokens, 4096)
        self.assertEqual(len(result.prompt_sha256), 64)
        self.assertEqual(len(result.response_sha256), 64)

    async def test_rejects_schema_violation(self):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [{"message": {"content": '{"wrong":"x"}'}}]}))
        provider = OpenAICompatibleJSONProvider("http://localhost:11434/v1", "model", transport=transport)
        with self.assertRaises(ModelProviderError):
            await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)

    async def test_rejects_remote_http_and_oversized_response(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleJSONProvider("http://example.com", "model")
        for limit in (0, -1, 32769, True):
            with self.subTest(max_output_tokens=limit), self.assertRaises(ValueError):
                OpenAICompatibleJSONProvider("http://localhost/v1", "model", max_output_tokens=limit)
        transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b" " * 1100))
        provider = OpenAICompatibleJSONProvider("http://localhost/v1", "model", max_response_bytes=1024, transport=transport)
        with self.assertRaises(ModelProviderError):
            await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)

    async def test_model_response_limit_stops_before_buffering_or_reading_later_chunks(self):
        stream = ChunkStream(b" " * 1024, b"x" * 4096, b"must not be read")
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
        provider = OpenAICompatibleJSONProvider("http://localhost/v1", "model", max_response_bytes=1024, transport=transport)
        with self.assertRaisesRegex(ModelProviderError, "configured byte limit"):
            await provider.complete_json(system="plan", prompt="objective", schema_name="query_plan", schema=SCHEMA)
        self.assertEqual(stream.consumed, 2)


if __name__ == "__main__":
    unittest.main()
