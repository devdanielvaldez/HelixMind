"""Fail-closed execution for proposed Python capabilities using a pinned Docker image."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import selectors
import subprocess
import tempfile
import threading
import time
from typing import Any

from medai_science.contract_validation import capability_io_schemas, validate_value


MAX_SOURCE_BYTES = 256 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_TEST_CASES = 100
SANDBOX_TIMEOUT_SECONDS = 60
_IMAGE_REFERENCE = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}|sha256:[0-9a-f]{64})$"
)
_ALLOWED_IMPORTS = {"__future__", "math", "statistics", "re", "json", "collections", "itertools", "functools", "decimal", "fractions", "numpy", "scipy", "Bio", "rdkit"}
_DENIED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "breakpoint",
    "getattr", "setattr", "delattr", "hasattr", "vars", "dir", "globals", "locals",
}
_RESULT_PREFIX = b"MEDAI_SANDBOX_RESULT:"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("sandbox receipt contains a duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"sandbox receipt contains non-standard JSON number {value}")


def strict_json_loads(payload: bytes | str) -> Any:
    """Decode JSON while rejecting ambiguous keys and non-standard numbers."""
    return json.loads(payload, object_pairs_hook=_strict_json_object, parse_constant=_reject_json_constant)

_RUNNER_SOURCE = r'''import json, runpy, sys, time
code_path, request_path, nonce = sys.argv[1:4]
request = json.load(open(request_path, "r", encoding="utf-8"))
request.setdefault("input_schema", {"type": "object"})
request.setdefault("output_schema", {"type": "object"})
cpu_started = time.process_time()
actual = None
checks = []
try:
    namespace = runpy.run_path(code_path, run_name="medai_generated_capability")
    entry = namespace.get("run")
    if not callable(entry):
        raise TypeError("capability must define run(input: dict) -> dict")
    def invoke(value):
        validate(value, request["input_schema"])
        result = entry(value)
        if not isinstance(result, dict):
            raise TypeError("capability run() must return a JSON object")
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        result = json.loads(encoded)
        validate(result, request["output_schema"])
        return result
    def validate(value, schema):
        kind = schema["type"]
        matches = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool), "number": isinstance(value, (int, float)) and not isinstance(value, bool) and (not isinstance(value, float) or __import__("math").isfinite(value)), "boolean": isinstance(value, bool), "null": value is None}[kind]
        if not matches: raise TypeError("value violates capability contract type " + kind)
        if "const" in schema and value != schema["const"]: raise TypeError("value violates capability contract const")
        if "enum" in schema and value not in schema["enum"]: raise TypeError("value violates capability contract enum")
        if kind == "object":
            props = schema.get("properties", {})
            if set(schema.get("required", [])) - value.keys(): raise TypeError("value omits required capability contract fields")
            if schema.get("additionalProperties") is False and value.keys() - props.keys(): raise TypeError("value has undeclared capability contract fields")
            for key, item in value.items():
                if key in props: validate(item, props[key])
        elif kind == "array":
            if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")): raise TypeError("array violates capability contract bounds")
            if "items" in schema:
                for item in value: validate(item, schema["items"])
        elif kind == "string":
            if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")): raise TypeError("string violates capability contract bounds")
        elif kind in ("integer", "number"):
            if value < schema.get("minimum", float("-inf")) or value > schema.get("maximum", float("inf")): raise TypeError("number violates capability contract bounds")
    actual = invoke(request["input"])
    failed = False
    for index, case in enumerate(request["test_cases"]):
        case_started = time.perf_counter_ns()
        observed = invoke(case["input"])
        matches = observed == case["expected"]
        elapsed = int((time.perf_counter_ns() - case_started) / 1000)
        checks.append({"index": index, "passed": matches, "wall_time_microseconds": elapsed})
        failed = failed or not matches
    validation = {"performed": bool(checks), "passed": sum(row["passed"] for row in checks), "failed": sum(not row["passed"] for row in checks), "cases": checks}
    print("MEDAI_SANDBOX_RESULT:" + nonce + ":" + json.dumps({"ok": not failed, "result": actual, "cpu_time_microseconds": int((time.process_time() - cpu_started) * 1_000_000), "validation": validation}, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if failed:
        raise SystemExit(3)
except SystemExit:
    raise
except BaseException as error:
    print("MEDAI_SANDBOX_RESULT:" + nonce + ":" + json.dumps({"ok": False, "result": None, "execution_error_type": type(error).__name__, "cpu_time_microseconds": int((time.process_time() - cpu_started) * 1_000_000), "validation": {"passed": 0, "failed": 1, "cases": []}}, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    raise SystemExit(4)
'''


class SandboxError(RuntimeError):
    """Capability rejected, runner unavailable, or sandbox execution failed."""


class SandboxCancelledError(SandboxError):
    """The caller requested cancellation and the active container was stopped."""


def validate_python_source(source: str) -> dict[str, Any]:
    """Apply bounded syntax/import checks before isolated execution."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Python capability source cannot be empty")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError(f"Python capability source exceeds {MAX_SOURCE_BYTES} bytes")
    try:
        tree = ast.parse(source, filename="capability.py", mode="exec")
    except SyntaxError as error:
        raise ValueError(f"Python capability has invalid syntax: {error.msg}") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > 100_000:
        raise ValueError("Python capability AST exceeds 100000 nodes")
    imports: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("relative imports are not allowed in generated capabilities")
            if node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Name) and (node.id in _DENIED_NAMES or node.id.startswith("__") and node.id.endswith("__")):
            raise ValueError(f"generated capability references denied Python builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            raise ValueError(f"generated capability uses denied Python dunder attribute: {node.attr}")
        elif isinstance(node, ast.Attribute) and node.attr in {"format", "format_map"}:
            # str.format supports attribute and item traversal encoded inside a
            # string (for example "{value.__class__}"), which bypasses the AST
            # checks above because those names are not represented as AST
            # attributes. Scientific capabilities can use f-strings instead.
            raise ValueError("generated capability uses denied reflective string formatting: " + node.attr)
    denied = sorted(imports - _ALLOWED_IMPORTS)
    if denied:
        raise ValueError("generated capability imports modules outside the allowlist: " + ", ".join(denied))
    if any(isinstance(node, ast.AsyncFunctionDef) and node.name == "run" for node in tree.body):
        raise ValueError("Python capability run() must be synchronous")
    run_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if not run_functions:
        raise ValueError("Python capability must define a top-level run(input: dict) -> dict function")
    if len(run_functions) != 1:
        raise ValueError("Python capability must define exactly one top-level run(input) function")
    run = run_functions[0]
    positional = [*run.args.posonlyargs, *run.args.args]
    if (
        len(positional) != 1
        or positional[0].arg != "input"
        or run.args.defaults
        or run.args.vararg is not None
        or run.args.kwarg is not None
        or run.args.kwonlyargs
    ):
        raise ValueError("Python capability entry point must have exactly the signature run(input)")
    return {"source_sha256": hashlib.sha256(encoded).hexdigest(), "source_bytes": len(encoded), "ast_nodes": len(nodes), "imports": sorted(imports), "static_policy": "imports allowlisted; reflective builtins, dunder names, and dunder attributes denied; filesystem restricted to read-only bundle and bounded temporary storage; network disabled by container controls"}


def validate_capability_run_request(
    input_data: Any,
    test_cases: Any,
    contract: dict[str, Any] | None = None,
    *,
    require_test_cases: bool = True,
) -> dict[str, Any]:
    """Validate the bounded JSON request before charging and sandbox startup."""
    if not isinstance(require_test_cases, bool):
        raise ValueError("require_test_cases must be an explicit boolean")
    if not isinstance(input_data, dict):
        raise ValueError("capability input must be a JSON object")
    minimum_cases = 1 if require_test_cases else 0
    if not isinstance(test_cases, list) or not minimum_cases <= len(test_cases) <= MAX_TEST_CASES:
        if require_test_cases:
            raise ValueError(f"provide 1..{MAX_TEST_CASES} scientific validation cases")
        raise ValueError(f"provide 0..{MAX_TEST_CASES} execution cases")
    for index, case in enumerate(test_cases):
        if not isinstance(case, dict) or not isinstance(case.get("input"), dict) or "expected" not in case:
            raise ValueError(f"test case {index} must contain object input and expected result")
    input_schema: dict[str, Any] = {"type": "object"}
    output_schema: dict[str, Any] = {"type": "object"}
    if contract is not None:
        input_schema, output_schema = capability_io_schemas(contract)
        validate_value(input_data, input_schema)
        for index, case in enumerate(test_cases):
            validate_value(case["input"], input_schema, f"$.test_cases[{index}].input")
            validate_value(case["expected"], output_schema, f"$.test_cases[{index}].expected")
    request = {"input": input_data, "test_cases": test_cases, "input_schema": input_schema, "output_schema": output_schema}
    try:
        encoded = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("capability input and test cases must be finite JSON values") from error
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError(f"capability request exceeds {MAX_REQUEST_BYTES} bytes")
    return request


def capability_validation_request_hash(request: dict[str, Any]) -> str:
    """Hash canonical transient inputs and expected cases without retaining them."""
    if not isinstance(request, dict):
        raise ValueError("capability validation request has an invalid shape")
    if set(request) == {"input", "test_cases"}:
        validated = validate_capability_run_request(request["input"], request["test_cases"])
    elif set(request) == {"input", "test_cases", "input_schema", "output_schema"}:
        validated = validate_capability_run_request(request["input"], request["test_cases"], {"input": request["input_schema"], "output": request["output_schema"]})
    else:
        raise ValueError("capability validation request has an invalid shape")
    canonical = json.dumps(validated, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_limited(argv: list[str], *, timeout: int, output_limit: int = MAX_OUTPUT_BYTES, cancel_event: threading.Event | None = None, input_bytes: bytes | None = None) -> tuple[int, bytes, bytes]:
    process = subprocess.Popen(argv, stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    stdin_remaining = memoryview(input_bytes) if input_bytes is not None else None
    if process.stdin is not None:
        os.set_blocking(process.stdin.fileno(), False)
        if stdin_remaining:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map() or process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                raise SandboxCancelledError("sandbox execution cancelled by the research session")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("sandbox execution exceeded its wall-clock limit")
            for key, _ in selector.select(min(remaining, 0.2)):
                if key.data == "stdin":
                    try:
                        written = os.write(key.fileobj.fileno(), stdin_remaining[:16_384])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        stdin_remaining = memoryview(b"")
                        continue
                    stdin_remaining = stdin_remaining[written:]
                    if not stdin_remaining:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                try:
                    chunk = os.read(key.fileobj.fileno(), 16_384)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                target = stdout if key.data == "stdout" else stderr
                target.extend(chunk)
                if len(target) > output_limit:
                    raise SandboxError(f"sandbox {key.data} exceeded {output_limit} bytes")
        return process.wait(timeout=1), bytes(stdout), bytes(stderr)
    except (TimeoutError, SandboxError):
        try:
            os.killpg(process.pid, 9)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()


class DockerPythonSandbox:
    """Run deterministic Python capabilities in a no-network, resource-bounded container."""

    def __init__(self, image: str, docker_binary: str = "docker", timeout: int = SANDBOX_TIMEOUT_SECONDS):
        if not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("sandbox image must use an immutable sha256 repository digest or local image ID")
        if not 1 <= timeout <= SANDBOX_TIMEOUT_SECONDS:
            raise ValueError(f"sandbox timeout must be 1..{SANDBOX_TIMEOUT_SECONDS} seconds")
        self.image = image
        self.docker_binary = docker_binary
        self.timeout = timeout

    def execute(
        self,
        source: str,
        input_data: dict[str, Any],
        test_cases: list[dict[str, Any]],
        *,
        contract: dict[str, Any] | None = None,
        require_test_cases: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise SandboxCancelledError("sandbox execution cancelled before startup")
        source_receipt = validate_python_source(source)
        request = validate_capability_run_request(input_data, test_cases, contract, require_test_cases=require_test_cases)
        request_bytes = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="medai-capability-") as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir(mode=0o755)
            source_file = bundle / "capability.py"
            request_file = bundle / "request.json"
            runner_file = bundle / "runner.py"
            source_file.write_text(source, encoding="utf-8")
            request_file.write_bytes(request_bytes)
            runner_file.write_text(_RUNNER_SOURCE, encoding="utf-8")
            for file in (source_file, request_file, runner_file):
                file.chmod(0o444)
            bundle.chmod(0o755)
            cid_file = root / "container.cid"
            try:
                image_check = subprocess.run([self.docker_binary, "image", "inspect", "--format={{.Id}}", self.image], capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError) as error:
                raise SandboxError(f"could not inspect the pinned local sandbox image: {type(error).__name__}") from error
            if image_check.returncode != 0:
                detail = image_check.stderr.decode("utf-8", errors="replace")[-1000:]
                raise SandboxError("pinned sandbox image is unavailable or the Docker daemon cannot be reached; image pulls are disabled: " + detail)
            if cancel_event is not None and cancel_event.is_set():
                raise SandboxCancelledError("sandbox execution cancelled before container startup")
            command = [
                self.docker_binary, "run", "--pull=never", "--rm", f"--cidfile={cid_file}", "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32", "--memory=512m",
                "--memory-swap=512m", "--cpus=1.0", "--ulimit=cpu=30:30", "--ulimit=fsize=1048576:1048576", "--ulimit=nofile=64:64", "--ulimit=core=0:0",
                "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
                "--entrypoint=python", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONHASHSEED=0",
                # The interpreter reads these files by path; no executable or
                # device semantics are needed from the host-provided bundle.
                "--volume", f"{bundle}:/capability:ro,noexec,nosuid,nodev", self.image,
                "/capability/runner.py", "/capability/capability.py", "/capability/request.json", secrets.token_hex(24),
            ]
            nonce = command[-1]
            try:
                return_code, stdout, stderr = _run_limited(command, timeout=self.timeout, cancel_event=cancel_event)
            except TimeoutError as error:
                self._kill_container(cid_file)
                raise SandboxError(str(error)) from error
            except SandboxError:
                self._kill_container(cid_file)
                raise
            except (OSError, subprocess.SubprocessError) as error:
                self._kill_container(cid_file)
                raise SandboxError(f"could not start isolated Docker runner: {type(error).__name__}") from error
            receipt: dict[str, Any] | None = None
            result_prefix = _RESULT_PREFIX + nonce.encode("ascii") + b":"
            for line in stdout.splitlines():
                if line.startswith(result_prefix):
                    try:
                        receipt = strict_json_loads(line[len(result_prefix):])
                    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                        receipt = None
            if receipt is None:
                detail = stderr.decode("utf-8", errors="replace")[-4000:]
                raise SandboxError(f"sandbox produced no valid result receipt (exit {return_code}): {detail}")
            if not isinstance(receipt, dict):
                raise SandboxError("sandbox result receipt must be a JSON object")
            if not isinstance(receipt.get("ok"), bool):
                raise SandboxError("sandbox result receipt has an invalid execution status")
            if receipt["ok"] and not isinstance(receipt.get("result"), dict):
                raise SandboxError("successful sandbox result must be a JSON object")
            if receipt["ok"]:
                try:
                    validate_value(receipt["result"], request["output_schema"], "$.result")
                except ValueError as error:
                    raise SandboxError(f"sandbox result violates the host-validated output contract: {error}") from error
            cpu_time_microseconds = receipt.get("cpu_time_microseconds")
            if isinstance(cpu_time_microseconds, bool) or not isinstance(cpu_time_microseconds, int) or cpu_time_microseconds < 0:
                raise SandboxError("sandbox produced an invalid measured CPU usage receipt")
            validation = receipt.get("validation")
            if not isinstance(validation, dict):
                raise SandboxError("sandbox produced a malformed validation receipt")
            cases = validation.get("cases")
            passed = validation.get("passed")
            failed = validation.get("failed")
            if (not isinstance(cases, list) or len(cases) > MAX_TEST_CASES
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (passed, failed))):
                raise SandboxError("sandbox produced a malformed validation receipt")
            for index, case in enumerate(cases):
                if (not isinstance(case, dict) or case.get("index") != index or not isinstance(case.get("passed"), bool)
                        or isinstance(case.get("wall_time_microseconds"), bool)
                        or not isinstance(case.get("wall_time_microseconds"), int)
                        or case["wall_time_microseconds"] < 0):
                    raise SandboxError("sandbox produced a malformed per-case validation receipt")
            if receipt["ok"] and (len(cases) != len(test_cases)
                                   or passed != sum(case["passed"] for case in cases)
                                   or failed != sum(not case["passed"] for case in cases)
                                   or passed + failed != len(test_cases)):
                raise SandboxError("successful sandbox validation receipt does not match supplied test cases")
            return {
                **source_receipt,
                "runner": "docker-no-network-python",
                "image": self.image,
                "limits": {"wall_seconds": self.timeout, "cpu_seconds": 30, "cpus": 1, "memory_bytes": 512 * 1024 * 1024, "pids": 32, "output_bytes": MAX_OUTPUT_BYTES},
                "result": receipt.get("result"),
                "execution_error_type": receipt.get("execution_error_type"),
                "cpu_time_microseconds": cpu_time_microseconds,
                "cpu_measurement": "Python process CPU time for capability loading, requested invocation, and supplied test cases; excludes request parsing, Docker daemon, and host scheduling overhead",
                "validation": validation,
                "validation_performed": bool(test_cases),
                "case_tests_passed": bool(return_code == 0 and receipt.get("ok") and validation.get("passed") == len(test_cases) and validation.get("failed") == 0) if test_cases else None,
                "lifecycle_status": "experimental; no automatic registration or promotion",
                "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
                "limitations": ["Container isolation depends on the host Docker daemon and kernel; use a dedicated runner or stronger isolation such as gVisor/Kata for multi-tenant hostile workloads.", "Passing supplied cases validates those cases only; scientific correctness requires independently curated benchmarks."],
            }

    def _kill_container(self, cid_file: Path) -> None:
        try:
            container_id = cid_file.read_text(encoding="ascii").strip()
        except OSError:
            return
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            return
        try:
            subprocess.run([self.docker_binary, "kill", container_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return
