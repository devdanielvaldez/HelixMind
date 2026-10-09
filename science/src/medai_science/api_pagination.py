"""Bounded keyset readers for complete, session-scoped MedAI API snapshots."""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


async def read_evidence_pages(
    client: httpx.AsyncClient,
    session_id: str,
    max_records: int = 50_000,
) -> list[dict[str, Any]]:
    """Read a bounded evidence snapshot using stable ID and rowid cursors."""
    if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1:
        raise ValueError("evidence snapshot limit must be a positive integer")
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    snapshot_rowid: int | None = None
    while True:
        params: dict[str, Any] = {"limit": 500}
        if cursor is not None:
            params["after_id"] = cursor
            params["snapshot_rowid"] = snapshot_rowid
        response = await client.get(
            f"/v1/research/{quote(session_id, safe='')}/evidence", params=params
        )
        response.raise_for_status()
        page = response.json()
        if (
            not isinstance(page, dict)
            or page.get("session_id") != session_id
            or not isinstance(page.get("items"), list)
        ):
            raise ValueError("evidence page is malformed or belongs to another research session")
        rows = page["items"]
        page_snapshot = page.get("snapshot_rowid")
        if (
            not isinstance(page_snapshot, int)
            or isinstance(page_snapshot, bool)
            or page_snapshot < 0
            or (snapshot_rowid is not None and page_snapshot != snapshot_rowid)
        ):
            raise ValueError("evidence page snapshot watermark is invalid or changed")
        if snapshot_rowid is None:
            snapshot_rowid = page_snapshot
        if len(rows) > 500 or any(
            not isinstance(row, dict) or row.get("session_id") != session_id for row in rows
        ):
            raise ValueError("evidence page contains malformed or cross-session records")
        previous_id = cursor
        for row in rows:
            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id or (
                previous_id is not None and row_id <= previous_id
            ):
                raise ValueError("evidence rows are not in strictly increasing ID order")
            previous_id = row_id
        items.extend(rows)
        if len(items) > max_records:
            raise ValueError(f"evidence exceeds the bounded snapshot limit of {max_records} records")
        has_more = page.get("has_more")
        next_cursor = page.get("next_cursor")
        if not isinstance(has_more, bool) or (
            has_more
            and (
                not rows
                or not isinstance(next_cursor, str)
                or not next_cursor
                or (cursor is not None and next_cursor <= cursor)
            )
        ):
            raise ValueError("evidence page cursor or completion flag is invalid")
        if not has_more:
            if next_cursor is not None:
                raise ValueError("exhausted evidence page unexpectedly returned a cursor")
            return items
        if rows[-1].get("id") != next_cursor:
            raise ValueError("evidence page cursor does not match its last record")
        cursor = next_cursor
