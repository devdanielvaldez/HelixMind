"""Strict loading and session submission for adjudicated benchmark inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable
from urllib.parse import quote

from medai_science.evaluation import evaluate_adjudicated_cases, evaluate_biomedical_domain_benchmark


MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES = 4 * 1024 * 1024
DOMAIN_BENCHMARK_ARTIFACT_KIND = "biomedical_domain_benchmark_input"
DOMAIN_BENCHMARK_ARTIFACT_SOURCE = "researcher-supplied adjudicated biomedical benchmark"
EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND = "evidence_quality_benchmark_input"
EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE = "researcher-supplied double-reviewed evidence-quality benchmark"


def load_domain_benchmark_artifact(path: str | Path, *, bins: int = 10) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load strict bounded JSON, validate the full benchmark, and hash original bytes."""
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("benchmark input must be a regular non-symlink file")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("benchmark input must be a regular non-symlink file")
        if metadata.st_size > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
            raise ValueError("benchmark input exceeds the 4 MiB retained-artifact limit")
        raw = handle.read(MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES + 1)
    if not raw or len(raw) > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
        raise ValueError("benchmark input must contain 1..4 MiB")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("benchmark input contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_nonfinite(value):
        raise ValueError(f"benchmark input contains non-standard JSON number {value}")

    try:
        document = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark input must be strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("benchmark input must be a JSON object")
    report = evaluate_biomedical_domain_benchmark(document, bins=bins)
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
        raise ValueError("serialized benchmark payload exceeds the 4 MiB retained-artifact limit")
    return document, report, hashlib.sha256(raw).hexdigest()


def submit_domain_benchmark_artifact(
    document: dict[str, Any],
    report: dict[str, Any],
    session_id: str,
    request: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Upload a previously validated corpus as one provenance-labeled session artifact."""
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", session_id):
        raise ValueError("session ID must be a stable 1..128 character identifier")
    return request(
        f"/v1/research/{quote(session_id, safe='')}/artifacts",
        {
            "task_id": "benchmark.evaluate",
            "kind": DOMAIN_BENCHMARK_ARTIFACT_KIND,
            "source": DOMAIN_BENCHMARK_ARTIFACT_SOURCE,
            "source_id": report["benchmark_id"],
            "payload": document,
        },
    )


def load_evidence_quality_benchmark_artifact(
    path: str | Path, *, bins: int = 10
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and evaluate a bounded double-reviewed ordinal quality benchmark."""
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("benchmark input must be a regular non-symlink file")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("benchmark input must be a regular non-symlink file")
        if metadata.st_size > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
            raise ValueError("benchmark input exceeds the 4 MiB retained-artifact limit")
        raw = handle.read(MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES + 1)
    return evaluate_evidence_quality_benchmark_bytes(raw, bins=bins)


def evaluate_evidence_quality_benchmark_bytes(
    raw: bytes, *, bins: int = 10
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate strict retained bytes and compute source-bound quality metrics."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
        raise ValueError("benchmark input must contain 1..4 MiB")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("benchmark input contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_nonfinite(value):
        raise ValueError(f"benchmark input contains non-standard JSON number {value}")

    try:
        document = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark input must be strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError("benchmark input must be a JSON object")
    if document.get("schema_version") != "3":
        raise ValueError("session evidence-quality benchmarks require ordinal schema version 3")
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES:
        raise ValueError("serialized benchmark payload exceeds the 4 MiB retained-artifact limit")
    report = evaluate_adjudicated_cases(document, bins=bins)
    return document, report, hashlib.sha256(raw).hexdigest()


def submit_evidence_quality_benchmark_artifact(
    document: dict[str, Any],
    report: dict[str, Any],
    session_id: str,
    request: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Attach one validated, double-reviewed quality benchmark to a session."""
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", session_id):
        raise ValueError("session ID must be a stable 1..128 character identifier")
    return request(
        f"/v1/research/{quote(session_id, safe='')}/artifacts",
        {
            "task_id": "benchmark.evaluate",
            "kind": EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND,
            "source": EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE,
            "source_id": report["benchmark_id"],
            "payload": document,
        },
    )
