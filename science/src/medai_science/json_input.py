"""Bounded strict JSON loading for researcher-supplied scientific inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any


def strict_json_loads(payload: bytes | str) -> Any:
    """Decode UTF-8 JSON while rejecting duplicate object keys and NaN/Infinity."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number is not allowed: {value}")

    try:
        text = payload.decode("utf-8", "strict") if isinstance(payload, bytes) else payload
        if not isinstance(text, str):
            raise ValueError("JSON input must be UTF-8 bytes or text")
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("JSON input is malformed or is not strict UTF-8") from error


def read_strict_json(path: str | Path, max_bytes: int, label: str) -> Any:
    """Read one bounded regular file and decode it with strict JSON rules."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("JSON input byte limit must be a positive integer")
    if not isinstance(label, str) or not label.strip() or len(label) > 120:
        raise ValueError("JSON input label must contain 1..120 characters")
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(Path(path).expanduser(), flags)
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError(f"{label} must be a regular file")
            payload = source.read(max_bytes + 1)
    except IsADirectoryError as error:
        raise ValueError(f"{label} must be a regular file") from error
    except OSError as error:
        raise ValueError(f"could not read {label}: {type(error).__name__}") from error
    if len(payload) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit")
    return strict_json_loads(payload)
