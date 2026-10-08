"""Read HTTP response bodies with an enforced decoded-byte ceiling."""
from __future__ import annotations

import httpx


async def read_bounded_response(response: httpx.Response, max_bytes: int, source: str) -> bytes:
    """Validate status and Content-Length, then cap bytes actually yielded by the transport."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("response byte limit must be a positive integer")
    if not isinstance(source, str) or not source.strip() or len(source) > 120:
        raise ValueError("response source label must contain 1..120 characters")
    response.raise_for_status()
    declared_length = response.headers.get("content-length")
    if declared_length and declared_length.isdecimal() and int(declared_length) > max_bytes:
        raise ValueError(f"{source} response exceeds the {max_bytes}-byte limit")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(chunk) > max_bytes - len(body):
            raise ValueError(f"{source} response exceeds the {max_bytes}-byte limit")
        body.extend(chunk)
    return bytes(body)
