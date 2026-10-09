"""Provider-neutral structured JSON model calls with vendor-specific HTTP adapters."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
import math
import os
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
from medai_science.scientific_agents import AGENTS
from medai_science.model_usage import meter_model_usage

MAX_PROMPT_BYTES = 32_768
MAX_SCHEMA_BYTES = 65_536
MAX_RESPONSE_BYTES = 1_048_576
MAX_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32768


class ModelProviderError(RuntimeError):
    """The configured model did not return a valid bounded structured response."""


def _strict_json_loads(payload: str | bytes) -> Any:
    """Parse interoperable JSON and reject Python's permissive NaN/Infinity tokens."""
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def _validate_output_token_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"model max_output_tokens must be an integer from 1 to {MAX_OUTPUT_TOKENS}")
    return value


def _validate_completion_token_usage(usage: dict[str, int], max_output_tokens: int) -> dict[str, int]:
    """Reject a provider receipt that exceeds the output ceiling sent in the request."""
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is not None and completion_tokens > max_output_tokens:
        raise ModelProviderError("model provider reported completion usage above the configured output-token ceiling")
    return usage


async def _read_bounded_model_response(response: httpx.Response, max_bytes: int) -> bytes:
    """Bound decoded provider output before retaining each streamed chunk."""
    size_header = response.headers.get("content-length")
    if size_header and size_header.isdecimal() and int(size_header) > max_bytes:
        raise ModelProviderError("model response exceeds configured byte limit")
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        if len(chunk) > max_bytes - size:
            raise ModelProviderError("model response exceeds configured byte limit")
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class ModelCompletion:
    value: dict[str, Any]
    provider: str
    model: str
    prompt_sha256: str
    response_sha256: str
    usage: dict[str, int]
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


class JSONModelProvider(Protocol):
    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict[str, Any]) -> ModelCompletion:
        """Return a schema-validated JSON object without persisting prompt contents."""


def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ValueError("JSON schema nodes must be objects")
    kind = schema.get("type")
    valid = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
    }.get(kind)
    if valid is None:
        raise ValueError(f"unsupported JSON schema type at {path}: {kind}")
    if not valid(value):
        raise ValueError(f"model JSON does not match {kind} at {path}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"model JSON value is outside the allowed enum at {path}")
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"model JSON is missing required field(s) at {path}: {', '.join(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"model JSON has unexpected field(s) at {path}: {', '.join(sorted(extra))}")
        for key, item in value.items():
            if key in properties:
                _validate_schema(item, properties[key], f"{path}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", MAX_RESPONSE_BYTES):
            raise ValueError(f"model JSON array length is outside bounds at {path}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", MAX_RESPONSE_BYTES):
            raise ValueError(f"model JSON string length is outside bounds at {path}")
        pattern = schema.get("pattern")
        if pattern and not re.fullmatch(pattern, value):
            raise ValueError(f"model JSON string does not match its pattern at {path}")
    elif kind in {"number", "integer"}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"model JSON number must be finite at {path}")
        if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError(f"model JSON number is outside bounds at {path}")


async def complete_structured_json(
    provider: JSONModelProvider,
    *,
    system: str,
    prompt: str,
    schema_name: str,
    schema: dict[str, Any],
) -> ModelCompletion:
    """Run a provider-neutral task and enforce one common response contract.

    Provider adapters enforce these bounds before transport; this boundary also
    protects scientific operations that receive an injected/local provider.
    """
    if not isinstance(system, str) or not isinstance(prompt, str):
        raise ValueError("structured model system and prompt must be strings")
    if len(system.encode("utf-8")) + len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise ValueError(f"model prompt exceeds {MAX_PROMPT_BYTES} bytes")
    if not isinstance(schema_name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", schema_name):
        raise ValueError("schema_name must be a short identifier")
    if not isinstance(schema, dict):
        raise ValueError("model JSON schema must be an object")
    try:
        schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("model JSON schema must contain finite JSON values") from error
    if len(schema_bytes) > MAX_SCHEMA_BYTES:
        raise ValueError(f"model JSON schema exceeds {MAX_SCHEMA_BYTES} bytes")

    completion = await provider.complete_json(system=system, prompt=prompt, schema_name=schema_name, schema=schema)
    if not isinstance(completion, ModelCompletion) or not isinstance(completion.value, dict):
        raise ModelProviderError("model provider did not return a structured completion object")
    try:
        output_bytes = json.dumps(completion.value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ModelProviderError("model provider returned a non-JSON or non-finite structured value") from error
    if len(output_bytes) > MAX_RESPONSE_BYTES:
        raise ModelProviderError(f"model structured output exceeds {MAX_RESPONSE_BYTES} bytes")
    try:
        _validate_schema(completion.value, schema)
        meter_model_usage(completion.usage)
    except (TypeError, ValueError) as error:
        raise ModelProviderError(f"model completion failed the common schema or usage contract: {error}") from error
    for label, digest in (("prompt", completion.prompt_sha256), ("response", completion.response_sha256)):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ModelProviderError(f"model completion has an invalid {label} digest")
    if not isinstance(completion.provider, str) or not completion.provider.strip() or len(completion.provider) > 128:
        raise ModelProviderError("model completion has an invalid provider identity")
    if not isinstance(completion.model, str) or not completion.model.strip() or len(completion.model) > 256:
        raise ModelProviderError("model completion has an invalid model identity")
    try:
        _validate_output_token_limit(completion.max_output_tokens)
        _validate_completion_token_usage(completion.usage, completion.max_output_tokens)
    except ValueError as error:
        raise ModelProviderError(str(error)) from error
    return completion


def _validate_endpoint(base_url: str) -> str:
    if not isinstance(base_url, str) or len(base_url) > 2048:
        raise ValueError("model base URL must contain 1..2048 characters")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("model base URL must be an http(s) URL without credentials, query, or fragment")
    if parsed.scheme == "http":
        host = parsed.hostname.casefold()
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host in {"localhost", "host.docker.internal"}
        if not loopback:
            raise ValueError("unencrypted model HTTP is allowed only for loopback hosts")
    return base_url.rstrip("/")


class OpenAICompatibleJSONProvider:
    """A minimal HTTP adapter usable with local Ollama and compatible model gateways."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = 60,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        transport: httpx.AsyncBaseTransport | None = None,
        provider_name: str = "openai-compatible",
    ):
        self.base_url = _validate_endpoint(base_url)
        if not model.strip() or len(model) > 256:
            raise ValueError("model name must contain 1..256 characters")
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"model timeout must be 1..{MAX_TIMEOUT_SECONDS} seconds")
        if not 1024 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(f"model response limit must be 1024..{MAX_RESPONSE_BYTES} bytes")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_output_tokens = _validate_output_token_limit(max_output_tokens)
        self.transport = transport
        self.provider_name = provider_name[:128]

    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict[str, Any]) -> ModelCompletion:
        system_bytes, prompt_bytes = system.encode("utf-8"), prompt.encode("utf-8")
        schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(system_bytes) + len(prompt_bytes) > MAX_PROMPT_BYTES:
            raise ValueError(f"model prompt exceeds {MAX_PROMPT_BYTES} bytes")
        if len(schema_bytes) > MAX_SCHEMA_BYTES:
            raise ValueError(f"model JSON schema exceeds {MAX_SCHEMA_BYTES} bytes")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", schema_name):
            raise ValueError("schema_name must be a short identifier")
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        full_system = system + "\n\nReturn exactly one JSON object conforming to this schema (no markdown):\n" + schema_bytes.decode("utf-8")
        request_body = {
            "model": self.model,
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "stream": False,
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
                async with client.stream("POST", endpoint, json=request_body, headers=headers) as response:
                    if response.status_code >= 400:
                        raise ModelProviderError(f"model provider returned HTTP {response.status_code}")
                    raw = await _read_bounded_model_response(response, self.max_response_bytes)
        except ModelProviderError:
            raise
        except (httpx.HTTPError, OSError) as error:
            raise ModelProviderError(f"model request failed: {type(error).__name__}") from error
        try:
            envelope = _strict_json_loads(raw)
            content = envelope["choices"][0]["message"]["content"]
            value = _strict_json_loads(content) if isinstance(content, str) else content
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelProviderError("model response did not contain one JSON chat-completion object") from error
        if not isinstance(value, dict):
            raise ModelProviderError("model structured response must be a JSON object")
        try:
            _validate_schema(value, schema)
        except ValueError as error:
            raise ModelProviderError(str(error)) from error
        usage_data = envelope.get("usage") or {}
        usage = {key: usage_data[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if type(usage_data.get(key)) is int and usage_data[key] >= 0}
        _validate_completion_token_usage(usage, self.max_output_tokens)
        prompt_digest = hashlib.sha256(full_system.encode("utf-8") + b"\0" + prompt_bytes).hexdigest()
        return ModelCompletion(value, self.provider_name, self.model, prompt_digest, hashlib.sha256(raw).hexdigest(), usage, self.max_output_tokens)


class AnthropicMessagesJSONProvider:
    """Native Anthropic Messages adapter using a required structured tool call."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 60,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        transport: httpx.AsyncBaseTransport | None = None,
        provider_name: str = "anthropic",
    ):
        self.base_url = _validate_endpoint(base_url)
        if not model.strip() or len(model) > 256:
            raise ValueError("model name must contain 1..256 characters")
        if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096:
            raise ValueError("Anthropic API key must contain 1..4096 characters")
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"model timeout must be 1..{MAX_TIMEOUT_SECONDS} seconds")
        if not 1024 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(f"model response limit must be 1024..{MAX_RESPONSE_BYTES} bytes")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_output_tokens = _validate_output_token_limit(max_output_tokens)
        self.transport = transport
        self.provider_name = provider_name[:128]

    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict[str, Any]) -> ModelCompletion:
        system_bytes, prompt_bytes = system.encode("utf-8"), prompt.encode("utf-8")
        schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(system_bytes) + len(prompt_bytes) > MAX_PROMPT_BYTES:
            raise ValueError(f"model prompt exceeds {MAX_PROMPT_BYTES} bytes")
        if len(schema_bytes) > MAX_SCHEMA_BYTES:
            raise ValueError(f"model JSON schema exceeds {MAX_SCHEMA_BYTES} bytes")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", schema_name):
            raise ValueError("schema_name must be a short identifier")
        endpoint = self.base_url if self.base_url.endswith("/messages") else self.base_url + "/messages"
        full_system = system + "\n\nReturn a tool input conforming exactly to this JSON schema:\n" + schema_bytes.decode("utf-8")
        request_body = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": full_system,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"name": schema_name, "description": "Return the required structured JSON result.", "input_schema": schema}],
            "tool_choice": {"type": "tool", "name": schema_name},
            "temperature": 0,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
                async with client.stream("POST", endpoint, json=request_body, headers=headers) as response:
                    if response.status_code >= 400:
                        raise ModelProviderError(f"model provider returned HTTP {response.status_code}")
                    raw = await _read_bounded_model_response(response, self.max_response_bytes)
        except ModelProviderError:
            raise
        except (httpx.HTTPError, OSError) as error:
            raise ModelProviderError(f"model request failed: {type(error).__name__}") from error
        try:
            envelope = _strict_json_loads(raw)
            blocks = envelope["content"]
            tool_results = [block["input"] for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == schema_name]
            if len(tool_results) != 1:
                raise ValueError("expected one matching structured tool result")
            value = tool_results[0]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ModelProviderError("model response did not contain one matching Anthropic tool result") from error
        if not isinstance(value, dict):
            raise ModelProviderError("model structured response must be a JSON object")
        try:
            _validate_schema(value, schema)
        except ValueError as error:
            raise ModelProviderError(str(error)) from error
        usage_data = envelope.get("usage") or {}
        usage = {"prompt_tokens": usage_data["input_tokens"], "completion_tokens": usage_data["output_tokens"], "total_tokens": usage_data["input_tokens"] + usage_data["output_tokens"]} if all(type(usage_data.get(key)) is int and usage_data[key] >= 0 for key in ("input_tokens", "output_tokens")) else {}
        _validate_completion_token_usage(usage, self.max_output_tokens)
        prompt_digest = hashlib.sha256(full_system.encode("utf-8") + b"\0" + prompt_bytes).hexdigest()
        return ModelCompletion(value, self.provider_name, self.model, prompt_digest, hashlib.sha256(raw).hexdigest(), usage, self.max_output_tokens)


class GeminiGenerateContentJSONProvider:
    """Native Google Gemini GenerateContent adapter with local full-schema validation."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        timeout: float = 60,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        transport: httpx.AsyncBaseTransport | None = None,
        provider_name: str = "google-gemini",
    ):
        self.base_url = _validate_endpoint(base_url)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", model.strip()):
            raise ValueError("Gemini model must be a bounded model identifier")
        if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096:
            raise ValueError("Gemini API key must contain 1..4096 characters")
        if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"model timeout must be 1..{MAX_TIMEOUT_SECONDS} seconds")
        if not 1024 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(f"model response limit must be 1024..{MAX_RESPONSE_BYTES} bytes")
        self.model = model.strip()
        self.api_key = api_key
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_output_tokens = _validate_output_token_limit(max_output_tokens)
        self.transport = transport
        self.provider_name = provider_name[:128]

    @staticmethod
    def _provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Send only Gemini's documented schema subset; local validation keeps stricter bounds."""
        supported = {"type", "properties", "required", "items", "enum", "minimum", "maximum", "minItems", "maxItems", "additionalProperties", "format", "description"}
        def clean(node: Any) -> Any:
            if isinstance(node, dict):
                return {key: ({name: clean(value) for name, value in node[key].items()} if key == "properties" and isinstance(node[key], dict) else clean(node[key])) for key in node if key in supported}
            if isinstance(node, list):
                return [clean(value) for value in node]
            return node
        return clean(schema)

    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict[str, Any]) -> ModelCompletion:
        system_bytes, prompt_bytes = system.encode("utf-8"), prompt.encode("utf-8")
        if len(system_bytes) + len(prompt_bytes) > MAX_PROMPT_BYTES:
            raise ValueError(f"model prompt exceeds {MAX_PROMPT_BYTES} bytes")
        schema_bytes = json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(schema_bytes) > MAX_SCHEMA_BYTES:
            raise ValueError(f"model JSON schema exceeds {MAX_SCHEMA_BYTES} bytes")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", schema_name):
            raise ValueError("schema_name must be a short identifier")
        endpoint = self.base_url if re.search(r":generateContent$", self.base_url) else f"{self.base_url}/models/{self.model}:generateContent"
        provider_schema = self._provider_schema(schema)
        full_system = system + "\n\nReturn exactly one JSON object named " + schema_name + " conforming to this schema:\n" + schema_bytes.decode("utf-8")
        body = {
            "systemInstruction": {"parts": [{"text": full_system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": provider_schema, "temperature": 0, "candidateCount": 1, "maxOutputTokens": self.max_output_tokens},
        }
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
                async with client.stream("POST", endpoint, json=body, headers={"Accept": "application/json", "Content-Type": "application/json", "x-goog-api-key": self.api_key}) as response:
                    if response.status_code >= 400:
                        raise ModelProviderError(f"Gemini provider returned HTTP {response.status_code}")
                    raw = await _read_bounded_model_response(response, self.max_response_bytes)
        except ModelProviderError:
            raise
        except (httpx.HTTPError, OSError) as error:
            raise ModelProviderError(f"model request failed: {type(error).__name__}") from error
        try:
            envelope = _strict_json_loads(raw)
            parts = envelope["candidates"][0]["content"]["parts"]
            text = "".join(part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str))
            value = _strict_json_loads(text)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ModelProviderError("Gemini response did not contain one JSON candidate") from error
        if not isinstance(value, dict):
            raise ModelProviderError("model structured response must be a JSON object")
        try:
            _validate_schema(value, schema)
        except ValueError as error:
            raise ModelProviderError(str(error)) from error
        usage_data = envelope.get("usageMetadata") or {}
        if not isinstance(usage_data, dict):
            usage_data = {}
        prompt_tokens, total_tokens = usage_data.get("promptTokenCount"), usage_data.get("totalTokenCount")
        # Gemini may bill thought tokens in totalTokenCount without including them in candidatesTokenCount.
        # Derive the billed generation remainder so MedAI's invariant total=prompt+completion remains true.
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": total_tokens - prompt_tokens, "total_tokens": total_tokens} if all(type(value) is int and value >= 0 for value in (prompt_tokens, total_tokens)) and total_tokens >= prompt_tokens else {}
        _validate_completion_token_usage(usage, self.max_output_tokens)
        prompt_digest = hashlib.sha256(full_system.encode("utf-8") + b"\0" + prompt_bytes).hexdigest()
        return ModelCompletion(value, self.provider_name, self.model, prompt_digest, hashlib.sha256(raw).hexdigest(), usage, self.max_output_tokens)


def configured_model_provider(agent_id: str | None = None) -> OpenAICompatibleJSONProvider | AnthropicMessagesJSONProvider | GeminiGenerateContentJSONProvider | None:
    """Load a model policy for one scientific role, with a shared legacy fallback.

    MEDAI_AGENT_MODEL_POLICIES is a bounded JSON object keyed by registered
    agent ID. Secrets are referenced by environment-variable name and never
    embedded in the policy document.
    """
    policy_text = os.getenv("MEDAI_AGENT_MODEL_POLICIES", "").strip()
    selected: dict[str, Any] | None = None
    if policy_text:
        if len(policy_text.encode("utf-8")) > 65_536:
            raise ValueError("MEDAI_AGENT_MODEL_POLICIES exceeds 64 KiB")
        try:
            policies = json.loads(policy_text)
        except json.JSONDecodeError as error:
            raise ValueError("MEDAI_AGENT_MODEL_POLICIES must be a JSON object") from error
        if not isinstance(policies, dict) or len(policies) > 64:
            raise ValueError("MEDAI_AGENT_MODEL_POLICIES must contain at most 64 role profiles")
        if any(not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", key) for key in policies):
            raise ValueError("model policy keys must be safe scientific agent IDs")
        if set(policies) - set(AGENTS):
            raise ValueError("model policies may only name registered scientific agents")
        if agent_id is not None and agent_id in policies:
            candidate = policies[agent_id]
            if not isinstance(candidate, dict):
                raise ValueError(f"model policy for {agent_id} must be an object")
            allowed = {"provider_type", "base_url", "model", "provider_name", "api_key_env", "timeout_seconds", "max_output_tokens"}
            if set(candidate) - allowed or not {"base_url", "model"}.issubset(candidate):
                raise ValueError(f"model policy for {agent_id} has missing or unsupported fields")
            selected = candidate

    if selected is not None:
        base_url = selected["base_url"].strip() if isinstance(selected["base_url"], str) else ""
        model = selected["model"].strip() if isinstance(selected["model"], str) else ""
        provider_type = selected.get("provider_type", "openai-compatible")
        provider_name = selected.get("provider_name", provider_type)
        key_name = selected.get("api_key_env")
        if key_name is not None and (not isinstance(key_name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,127}", key_name)):
            raise ValueError(f"model policy for {agent_id} has an invalid api_key_env name")
        api_key = os.getenv(key_name) if key_name else None
        timeout = selected.get("timeout_seconds", 60)
        max_output_tokens = selected.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    else:
        base_url = os.getenv("MEDAI_MODEL_BASE_URL", "").strip()
        model = os.getenv("MEDAI_MODEL_NAME", "").strip()
        provider_type = os.getenv("MEDAI_MODEL_PROVIDER_TYPE", "openai-compatible")
        provider_name = os.getenv("MEDAI_MODEL_PROVIDER_NAME", provider_type)
        api_key = os.getenv("MEDAI_MODEL_API_KEY") or None
        timeout = float(os.getenv("MEDAI_MODEL_TIMEOUT_SECONDS", "60"))
        try:
            max_output_tokens = int(os.getenv("MEDAI_MODEL_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)))
        except ValueError as error:
            raise ValueError("MEDAI_MODEL_MAX_OUTPUT_TOKENS must be an integer") from error
    if not base_url and not model:
        return None
    if not base_url or not model:
        raise ValueError("model policy must configure both base_url and model")
    if not isinstance(provider_name, str) or not provider_name.strip() or len(provider_name) > 128:
        raise ValueError("model provider name must contain 1..128 characters")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("model policy timeout_seconds must be numeric")
    max_output_tokens = _validate_output_token_limit(max_output_tokens)
    if not isinstance(provider_type, str) or provider_type not in {"openai-compatible", "anthropic", "gemini"}:
        raise ValueError("model provider type must be openai-compatible, anthropic, or gemini")
    if provider_type == "anthropic":
        if not api_key:
            raise ValueError("Anthropic model policies require api_key_env or MEDAI_MODEL_API_KEY")
        return AnthropicMessagesJSONProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=float(timeout),
            max_output_tokens=max_output_tokens,
            provider_name=provider_name,
        )
    if provider_type == "gemini":
        if not api_key:
            raise ValueError("Gemini model policies require api_key_env or MEDAI_MODEL_API_KEY")
        return GeminiGenerateContentJSONProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=float(timeout),
            max_output_tokens=max_output_tokens,
            provider_name=provider_name,
        )
    return OpenAICompatibleJSONProvider(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=float(timeout),
        max_output_tokens=max_output_tokens,
        provider_name=provider_name,
    )
