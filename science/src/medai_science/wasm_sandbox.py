"""Experimental WASI JSON capability runner inside a pinned no-network container."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Any

from medai_science.contract_validation import validate_value
from medai_science.sandbox import (
    MAX_OUTPUT_BYTES,
    MAX_REQUEST_BYTES,
    SANDBOX_TIMEOUT_SECONDS,
    SandboxCancelledError,
    SandboxError,
    _IMAGE_REFERENCE,
    _run_limited,
    strict_json_loads,
    validate_capability_run_request,
)


MAX_WASM_BYTES = 20 * 1024 * 1024
MAX_WASM_FUEL = 100_000_000
MAX_WASM_MEMORY_BYTES = 256 * 1024 * 1024
_WASM_V1_HEADER = b"\x00asm\x01\x00\x00\x00"


class DockerWasmSandbox:
    """Run one WASI Preview 1 JSON-in/JSON-out module per isolated invocation.

    Modules receive only one JSON object on stdin and must write one JSON object
    to stdout. No filesystem preopens, environment variables, host paths, or
    network interfaces are granted to the guest.
    """

    language = "wasm"

    def __init__(
        self,
        image: str,
        docker_binary: str = "docker",
        timeout: int = SANDBOX_TIMEOUT_SECONDS,
        *,
        fuel: int = MAX_WASM_FUEL,
        max_memory_bytes: int = MAX_WASM_MEMORY_BYTES,
    ):
        if not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("WASM sandbox image must use an immutable sha256 repository digest or local image ID")
        if not 1 <= timeout <= SANDBOX_TIMEOUT_SECONDS:
            raise ValueError(f"sandbox timeout must be 1..{SANDBOX_TIMEOUT_SECONDS} seconds")
        if isinstance(fuel, bool) or not isinstance(fuel, int) or not 1 <= fuel <= MAX_WASM_FUEL:
            raise ValueError(f"WASM fuel must be an integer from 1 to {MAX_WASM_FUEL}")
        if (isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int)
                or not 1_048_576 <= max_memory_bytes <= MAX_WASM_MEMORY_BYTES):
            raise ValueError(f"WASM memory must be 1 MiB..{MAX_WASM_MEMORY_BYTES} bytes")
        self.image = image
        self.docker_binary = docker_binary
        self.timeout = timeout
        self.fuel = fuel
        self.max_memory_bytes = max_memory_bytes

    def execute(
        self,
        module: bytes,
        input_data: dict[str, Any],
        test_cases: list[dict[str, Any]],
        *,
        contract: dict[str, Any] | None = None,
        require_test_cases: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if cancel_event is not None and cancel_event.is_set():
            raise SandboxCancelledError("WASM sandbox execution cancelled before startup")
        if not isinstance(module, bytes) or not _WASM_V1_HEADER == module[:8] or len(module) > MAX_WASM_BYTES:
            raise ValueError(f"WASM module must be a version 1 binary of 8..{MAX_WASM_BYTES} bytes")
        request = validate_capability_run_request(input_data, test_cases, contract, require_test_cases=require_test_cases)
        input_schema = request["input_schema"]
        output_schema = request["output_schema"]
        cases = request["test_cases"]
        module_hash = hashlib.sha256(module).hexdigest()
        started = time.monotonic()
        deadline = started + self.timeout
        checks = []
        result: dict[str, Any] | None = None
        with tempfile.TemporaryDirectory(prefix="medai-wasm-capability-") as temporary:
            root = Path(temporary)
            module_root = root / "module"
            module_root.mkdir(mode=0o755)
            module_file = module_root / "capability.wasm"
            module_file.write_bytes(module)
            module_file.chmod(0o444)
            # Keep Docker's host-side container ID file outside the only host
            # directory made visible to the WASI guest.
            cid_file = root / "container.cid"
            try:
                image_check = subprocess.run(
                    [self.docker_binary, "image", "inspect", "--format={{.Id}}", self.image],
                    capture_output=True, timeout=10, check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise SandboxError(f"could not inspect the pinned local WASM sandbox image: {type(error).__name__}") from error
            if image_check.returncode != 0:
                detail = image_check.stderr.decode("utf-8", errors="replace")[-1000:]
                raise SandboxError("pinned WASM sandbox image is unavailable or Docker daemon cannot be reached; image pulls are disabled: " + detail)

            def invoke(value: dict[str, Any]) -> tuple[dict[str, Any], int]:
                encoded_input = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
                if len(encoded_input) > MAX_REQUEST_BYTES:
                    raise ValueError("WASM invocation input exceeds the 64 KiB request limit")
                # Docker refuses to start when --cidfile already exists. Each
                # validation case is a new container, so remove the previous
                # invocation's identifier before asking Docker to create one.
                cid_file.unlink(missing_ok=True)
                command = [
                    self.docker_binary, "run", "--pull=never", "--rm", f"--cidfile={cid_file}", "-i",
                    "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                    "--pids-limit=16", "--memory=384m", "--memory-swap=384m", "--cpus=1.0",
                    "--ulimit=cpu=30:30", "--ulimit=fsize=1048576:1048576", "--ulimit=nofile=64:64", "--ulimit=core=0:0",
                    "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=32m",
                    "--volume", f"{module_root}:/capability:ro,noexec,nosuid,nodev", "--entrypoint=wasmtime", self.image,
                    "run", "-W", f"fuel={self.fuel}", "-W", f"max-memory-size={self.max_memory_bytes}",
                    "/capability/capability.wasm",
                ]
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise SandboxError("WASM sandbox exceeded its total wall-clock limit")
                try:
                    return_code, stdout, stderr = _run_limited(
                        command, timeout=remaining_seconds, cancel_event=cancel_event,
                        output_limit=MAX_OUTPUT_BYTES, input_bytes=encoded_input,
                    )
                except (TimeoutError, SandboxError) as error:
                    self._kill_container(cid_file)
                    if isinstance(error, TimeoutError):
                        raise SandboxError(str(error)) from error
                    raise
                except (OSError, subprocess.SubprocessError) as error:
                    self._kill_container(cid_file)
                    raise SandboxError(f"could not start isolated WASM runner: {type(error).__name__}") from error
                if return_code != 0:
                    detail = stderr.decode("utf-8", errors="replace")[-2000:]
                    raise SandboxError(f"WASM guest failed (exit {return_code}): {detail}")
                try:
                    output = strict_json_loads(stdout)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
                    raise SandboxError("WASM guest stdout must contain exactly one JSON result object") from error
                if not isinstance(output, dict):
                    raise SandboxError("WASM guest result must be a JSON object")
                validate_value(output, output_schema, "$.result")
                return output, len(encoded_input)

            validate_value(request["input"], input_schema, "$.input")
            result, _ = invoke(request["input"])
            for index, case in enumerate(cases):
                validate_value(case["input"], input_schema, f"$.test_cases[{index}].input")
                validate_value(case["expected"], output_schema, f"$.test_cases[{index}].expected")
                case_started = time.monotonic_ns()
                observed, _ = invoke(case["input"])
                checks.append({
                    "index": index,
                    "passed": observed == case["expected"],
                    "wall_time_microseconds": max(0, (time.monotonic_ns() - case_started) // 1000),
                })

        failed = sum(not row["passed"] for row in checks)
        return {
            "module_sha256": module_hash,
            "module_bytes": len(module),
            "runner": "docker-no-network-wasmtime-wasi-preview1",
            "image": self.image,
            "limits": {
                "wall_seconds_total": self.timeout,
                "docker_cpu_seconds": 30,
                "docker_cpus": 1,
                "docker_memory_bytes": 384 * 1024 * 1024,
                "wasm_memory_bytes": self.max_memory_bytes,
                "fuel_per_invocation": self.fuel,
                "pids": 16,
                "output_bytes": MAX_OUTPUT_BYTES,
                "request_bytes": MAX_REQUEST_BYTES,
            },
            "result": result,
            "validation": {"performed": bool(cases), "passed": len(checks) - failed, "failed": failed, "cases": checks},
            "validation_performed": bool(cases),
            "case_tests_passed": all(row["passed"] for row in checks) if cases else None,
            "wall_time_microseconds": max(0, int((time.monotonic() - started) * 1_000_000)),
            "fuel_metering": "Wasmtime enforces a per-invocation fuel ceiling; exact fuel consumed is not exposed by this CLI runner and is not reported as CPU time.",
            "lifecycle_status": "experimental; no automatic registration or promotion",
            "limitations": ["The pinned image must contain a Wasmtime CLI supporting -W fuel and -W max-memory-size.", "Only WASI imports provided by the Wasmtime CLI are available; no filesystem preopens or WASI networking are granted.", "A passing caller-supplied test set does not establish scientific correctness."],
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
