"""Authenticated, tenant-scoped research session status helpers for the CLI."""
from __future__ import annotations

import time
from typing import Any, Callable


ApiRequest = Callable[..., dict[str, Any]]


def validate_session_id(session_id: str) -> str:
    if not session_id or len(session_id) > 128 or not all(
        char.isascii() and (char.isalnum() or char in "-_.") for char in session_id
    ):
        raise ValueError("research session ID contains unsupported characters")
    return session_id


def list_sessions(request: ApiRequest) -> dict[str, Any]:
    result = request("/v1/research", method="GET")
    if not isinstance(result.get("items"), list):
        raise ValueError("MedAI API returned a malformed research session list")
    return result


def get_session_status(request: ApiRequest, session_id: str) -> dict[str, Any]:
    identifier = validate_session_id(session_id)
    result = request(f"/v1/research/{identifier}/status", method="GET")
    if not isinstance(result.get("tasks"), list) or not isinstance(result.get("task_counts"), dict):
        raise ValueError("MedAI API returned a malformed research status")
    return result


def cancel_session(request: ApiRequest, session_id: str) -> dict[str, Any]:
    identifier = validate_session_id(session_id)
    result = request(f"/v1/research/{identifier}/control", method="POST")
    if not isinstance(result, dict):
        raise ValueError("MedAI API returned a malformed research cancellation")
    return result


def watch_session(
    request: ApiRequest,
    session_id: str,
    poll_seconds: int,
    *,
    emit: Callable[[dict[str, Any]], None],
    sleep: Callable[[int], None] = time.sleep,
) -> None:
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, int) or not 5 <= poll_seconds <= 3600:
        raise ValueError("poll interval must be 5..3600 seconds")
    while True:
        status = get_session_status(request, session_id)
        emit(status)
        if status.get("terminal") is True:
            return
        sleep(poll_seconds)
