"""Short-lived, query-private cache for PubMed search identifiers and coverage only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any


_MAX_TTL_SECONDS = 3600
_MAX_CACHE_ENTRIES = 1000
_PMID = re.compile(r"^[0-9]{1,20}$")


def search_key(query: str, limit: int) -> str:
    """Hash exact normalized query inputs so the cache never stores search text."""
    if not isinstance(query, str) or not query.strip() or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("PubMed cache key requires a nonempty query and limit 1..200")
    encoded = json.dumps(["pubmed-esearch-v1", query, limit], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PubMedSearchCache:
    """Persist bounded PMID pages without queries, abstracts, or fetched records."""

    def __init__(self, path: str | Path, *, ttl_seconds: int = 300):
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= _MAX_TTL_SECONDS:
            raise ValueError(f"PubMed search cache TTL must be 1..{_MAX_TTL_SECONDS} seconds")
        self.path = Path(path).expanduser()
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("PubMed cache path cannot be a symlink")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink():
            raise ValueError("PubMed cache path cannot be a symlink")
        connection = sqlite3.connect(self.path, timeout=5)
        os.chmod(self.path, 0o600)
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS pubmed_search_cache (cache_key TEXT PRIMARY KEY, pmids_json TEXT NOT NULL, total_hits INTEGER NOT NULL, requested_limit INTEGER NOT NULL, stored_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)")

    def get(self, query: str, limit: int, *, now: int | None = None) -> tuple[list[str], dict[str, Any]] | None:
        key = search_key(query, limit)
        instant = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("DELETE FROM pubmed_search_cache WHERE expires_at<=?", (instant,))
            row = connection.execute("SELECT pmids_json,total_hits,requested_limit,expires_at FROM pubmed_search_cache WHERE cache_key=?", (key,)).fetchone()
            if row is None:
                return None
            pmids_raw, total_hits, stored_limit, expires_at = row
        try:
            pmids = json.loads(pmids_raw)
        except (TypeError, json.JSONDecodeError):
            self.delete(query, limit)
            return None
        if (stored_limit != limit or not isinstance(total_hits, int) or total_hits < 0
                or not isinstance(pmids, list) or len(pmids) > limit
                or any(not isinstance(pmid, str) or not _PMID.fullmatch(pmid) for pmid in pmids)
                or len(pmids) != len(set(pmids)) or len(pmids) > total_hits):
            self.delete(query, limit)
            return None
        return pmids, {
            "total_hits": total_hits,
            "requested_limit": limit,
            "returned_pmids": len(pmids),
            "search_truncated": total_hits > len(pmids),
            "coverage_semantics": "PubMed-reported total count versus the bounded ESearch PMID page; count does not establish relevance or completeness of the search strategy.",
            "cache": "hit",
        }

    def put(self, query: str, limit: int, pmids: list[str], coverage: dict[str, Any], *, now: int | None = None) -> None:
        key = search_key(query, limit)
        total_hits = coverage.get("total_hits") if isinstance(coverage, dict) else None
        if (not isinstance(pmids, list) or len(pmids) > limit
                or any(not isinstance(pmid, str) or not _PMID.fullmatch(pmid) for pmid in pmids)
                or len(pmids) != len(set(pmids)) or isinstance(total_hits, bool)
                or not isinstance(total_hits, int) or total_hits < len(pmids)):
            raise ValueError("PubMed search cache accepts only a validated bounded PMID page and total count")
        instant = int(time.time()) if now is None else now
        with self._connect() as connection:
            connection.execute("DELETE FROM pubmed_search_cache WHERE expires_at<=?", (instant,))
            connection.execute(
                "INSERT INTO pubmed_search_cache(cache_key,pmids_json,total_hits,requested_limit,stored_at,expires_at) VALUES(?,?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET pmids_json=excluded.pmids_json,total_hits=excluded.total_hits,requested_limit=excluded.requested_limit,stored_at=excluded.stored_at,expires_at=excluded.expires_at",
                (key, json.dumps(pmids, separators=(",", ":")), total_hits, limit, instant, instant + self.ttl_seconds),
            )
            connection.execute(
                "DELETE FROM pubmed_search_cache WHERE cache_key NOT IN (SELECT cache_key FROM pubmed_search_cache ORDER BY stored_at DESC,cache_key ASC LIMIT ?)",
                (_MAX_CACHE_ENTRIES,),
            )

    def delete(self, query: str, limit: int) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM pubmed_search_cache WHERE cache_key=?", (search_key(query, limit),))
