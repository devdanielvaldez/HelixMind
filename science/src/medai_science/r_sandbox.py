"""Fail-closed isolated execution for experimental generated R capabilities."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import secrets
import subprocess
import tempfile
import threading
from typing import Any

from medai_science.sandbox import (
    MAX_OUTPUT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_SOURCE_BYTES,
    SANDBOX_TIMEOUT_SECONDS,
    DockerPythonSandbox,
    SandboxCancelledError,
    SandboxError,
    _IMAGE_REFERENCE,
    _RESULT_PREFIX,
    _run_limited,
    validate_capability_run_request,
)
from medai_science.contract_validation import validate_value


_RUN_ENTRYPOINT = re.compile(r"(?m)^\s*run\s*<-\s*function\s*\(\s*input\s*\)")
_DENIED_CALL = re.compile(r"(?<![A-Za-z0-9._])(?:system2?|shell|download\.file|socketConnection|url|source|sys\.source|eval|parse|unlink|file\.remove|dir\.create|setwd|sink)\s*\(", re.IGNORECASE)

_RUNNER_SOURCE = r'''args <- commandArgs(trailingOnly=TRUE)
code_path <- args[[1]]
request_path <- args[[2]]
nonce <- args[[3]]
cpu_started <- proc.time()[["user.self"]] + proc.time()[["sys.self"]]
emit <- function(value) {
  cat("MEDAI_SANDBOX_RESULT:", nonce, ":", jsonlite::toJSON(value, auto_unbox=TRUE, null="null", na="null", digits=NA, dataframe="rows", force=TRUE), "\n", sep="")
}
tryCatch({
  request <- jsonlite::fromJSON(request_path, simplifyVector=FALSE)
  capability <- new.env(parent=baseenv())
  source(code_path, local=capability, chdir=FALSE, encoding="UTF-8")
  entry <- capability$run
  if (!is.function(entry) || !identical(names(formals(entry)), "input")) stop("capability must define run <- function(input)")
  invoke <- function(value) {
    output <- entry(value)
    if (!is.list(output) || is.null(names(output)) || any(!nzchar(names(output)))) stop("capability run(input) must return a named list representing a JSON object")
    output
  }
  actual <- invoke(request$input)
  observed <- lapply(request$test_cases, function(item) {
    started <- proc.time()[["elapsed"]]
    value <- invoke(item$input)
    elapsed <- proc.time()[["elapsed"]] - started
    list(result=value, wall_time_microseconds=as.integer(max(0, elapsed * 1000000)))
  })
  cpu_now <- proc.time()
  emit(list(ok=TRUE, result=actual, test_results=observed, cpu_time_microseconds=as.integer(max(0, (cpu_now[["user.self"]] + cpu_now[["sys.self"]] - cpu_started) * 1000000))))
}, error=function(error) {
  cpu_now <- proc.time()
  emit(list(ok=FALSE, execution_error_type=class(error)[[1]], cpu_time_microseconds=as.integer(max(0, (cpu_now[["user.self"]] + cpu_now[["sys.self"]] - cpu_started) * 1000000))))
  quit(status=4, save="no")
})
'''


def validate_r_source(source: str) -> dict[str, Any]:
    """Apply a bounded entry-point and obvious-dangerous-call policy before sandboxing."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("R capability source cannot be empty")
    encoded = source.encode("utf-8")
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValueError(f"R capability source exceeds {MAX_SOURCE_BYTES} bytes")
    if len(_RUN_ENTRYPOINT.findall(source)) != 1:
        raise ValueError("R capability must define exactly one run <- function(input) entry point")
    denied = _DENIED_CALL.search(source)
    if denied:
        raise ValueError(f"R capability references denied function: {denied.group(0).split('(')[0].strip()}")
    return {
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_bytes": len(encoded),
        "static_policy": "bounded source; one run(input) entry point; obvious process, network, dynamic evaluation, and filesystem calls denied; execution remains container-isolated",
    }


class DockerRSandbox:
    """Run R capabilities inside a locally cached digest-pinned, no-network container."""

    language = "r"

    def __init__(self, image: str, docker_binary: str = "docker", timeout: int = SANDBOX_TIMEOUT_SECONDS):
        if not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("sandbox image must be pinned by immutable sha256 digest")
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
        source_receipt = validate_r_source(source)
        request = validate_capability_run_request(input_data, test_cases, contract, require_test_cases=require_test_cases)
        request_bytes = json.dumps(request, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="medai-r-capability-") as temporary:
            root = Path(temporary)
            source_file, request_file, runner_file = root / "capability.R", root / "request.json", root / "runner.R"
            source_file.write_text(source, encoding="utf-8")
            request_file.write_bytes(request_bytes)
            runner_file.write_text(_RUNNER_SOURCE, encoding="utf-8")
            for file in (source_file, request_file, runner_file):
                file.chmod(0o444)
            root.chmod(0o755)
            cid_file = root / "container.cid"
            try:
                image_check = subprocess.run([self.docker_binary, "image", "inspect", "--format={{.Id}}", self.image], capture_output=True, timeout=10, check=False)
            except (OSError, subprocess.SubprocessError) as error:
                raise SandboxError(f"could not inspect the pinned local R sandbox image: {type(error).__name__}") from error
            if image_check.returncode != 0:
                detail = image_check.stderr.decode("utf-8", errors="replace")[-1000:]
                raise SandboxError("pinned R sandbox image is unavailable or Docker daemon cannot be reached; image pulls are disabled: " + detail)
            if cancel_event is not None and cancel_event.is_set():
                raise SandboxCancelledError("sandbox execution cancelled before container startup")
            command = [
                self.docker_binary, "run", "--pull=never", "--rm", f"--cidfile={cid_file}", "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32", "--memory=512m",
                "--memory-swap=512m", "--cpus=1.0", "--ulimit=cpu=30:30", "--ulimit=fsize=1048576:1048576",
                "--ulimit=nofile=64:64", "--ulimit=core=0:0", "--user=65534:65534",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--entrypoint=Rscript",
                "--volume", f"{root}:/capability:ro,noexec,nosuid,nodev", self.image,
                "--vanilla", "/capability/runner.R", "/capability/capability.R", "/capability/request.json", secrets.token_hex(24),
            ]
            nonce = command[-1]
            try:
                return_code, stdout, stderr = _run_limited(command, timeout=self.timeout, cancel_event=cancel_event)
            except TimeoutError as error:
                DockerPythonSandbox(self.image, self.docker_binary, self.timeout)._kill_container(cid_file)
                raise SandboxError(str(error)) from error
            except SandboxError:
                DockerPythonSandbox(self.image, self.docker_binary, self.timeout)._kill_container(cid_file)
                raise
            except (OSError, subprocess.SubprocessError) as error:
                DockerPythonSandbox(self.image, self.docker_binary, self.timeout)._kill_container(cid_file)
                raise SandboxError(f"could not start isolated R runner: {type(error).__name__}") from error
            receipt = None
            result_prefix = _RESULT_PREFIX + nonce.encode("ascii") + b":"
            for line in stdout.splitlines():
                if line.startswith(result_prefix):
                    try:
                        receipt = json.loads(line[len(result_prefix):])
                    except json.JSONDecodeError:
                        receipt = None
            if not isinstance(receipt, dict):
                detail = stderr.decode("utf-8", errors="replace")[-4000:]
                raise SandboxError(f"R sandbox produced no valid result receipt (exit {return_code}): {detail}")
            cpu_time = receipt.get("cpu_time_microseconds")
            if isinstance(cpu_time, bool) or not isinstance(cpu_time, int) or cpu_time < 0:
                raise SandboxError("R sandbox produced an invalid measured CPU usage receipt")
            result = receipt.get("result")
            output_schema = request["output_schema"]
            if result is not None:
                validate_value(result, output_schema)
            observations = receipt.get("test_results", [])
            if not isinstance(observations, list) or len(observations) != len(test_cases):
                if test_cases and receipt.get("ok") is True:
                    raise SandboxError("R sandbox test-result count does not match supplied cases")
                observations = []
            checks = []
            for index, (observation, case) in enumerate(zip(observations, test_cases)):
                if not isinstance(observation, dict) or set(observation) != {"result", "wall_time_microseconds"}:
                    raise SandboxError("R sandbox test receipt is malformed")
                observed = observation["result"]
                latency = observation["wall_time_microseconds"]
                if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
                    raise SandboxError("R sandbox returned invalid per-case wall time")
                validate_value(observed, output_schema, f"$.test_results[{index}]")
                checks.append({"index": index, "passed": observed == case["expected"], "wall_time_microseconds": latency})
            failed = sum(not row["passed"] for row in checks)
            execution_failed = receipt.get("ok") is not True or return_code != 0
            validation = {"performed": bool(test_cases), "passed": len(checks) - failed if not execution_failed else 0, "failed": failed if not execution_failed else max(1, failed), "cases": checks}
            ok = not execution_failed and failed == 0
            return {
                **source_receipt,
                "runner": "docker-no-network-r",
                "image": self.image,
                "limits": {"wall_seconds": self.timeout, "cpu_seconds": 30, "cpus": 1, "memory_bytes": 512 * 1024 * 1024, "pids": 32, "output_bytes": MAX_OUTPUT_BYTES, "request_bytes": MAX_REQUEST_BYTES},
                "result": result,
                "execution_error_type": receipt.get("execution_error_type"),
                "cpu_time_microseconds": cpu_time,
                "cpu_measurement": "R process user+system CPU time for capability loading, requested invocation, and supplied cases; excludes JSON parsing, Docker daemon, host scheduling, and child process usage",
                "validation": validation,
                "validation_performed": bool(test_cases),
                "case_tests_passed": ok if test_cases else None,
                "lifecycle_status": "experimental; no automatic registration or promotion",
                "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
                "limitations": ["The digest-pinned R image must include Rscript and jsonlite.", "Container isolation depends on the host Docker daemon and kernel; use a dedicated hardened runner for multi-tenant hostile workloads.", "Passing supplied cases validates those cases only; independent curation is required for scientific correctness."],
            }
