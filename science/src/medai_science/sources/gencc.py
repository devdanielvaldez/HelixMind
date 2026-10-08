"""Rate-aware cache and exact-identifier queries over the official GenCC new-format TSV."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import httpx


_REQUIRED_COLUMNS = {
    "sgc_id", "version_number", "gene_curie", "gene_symbol", "disease_curie", "disease_title",
    "classification_curie", "classification_title", "moi_curie", "moi_title", "submitter_curie",
    "submitter_title", "sub_date", "sub_public_report_url", "sub_assertion_criteria_url", "sub_pmids",
}
_HGNC = re.compile(r"^HGNC:[1-9][0-9]{0,7}$")
_MONDO = re.compile(r"^MONDO:[0-9]{7}$")
_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_MAX_EXPORT_BYTES = 64 * 1024 * 1024
_MAX_ROWS = 100_000
_MAX_MATCHES = 100
_REFRESH_INTERVAL_SECONDS = 24 * 60 * 60


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("GenCC cache timestamp is invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("GenCC cache timestamp is invalid") from error
    if result.tzinfo is None:
        raise ValueError("GenCC cache timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _validate_export(payload: bytes) -> tuple[list[str], int]:
    if not payload or len(payload) > _MAX_EXPORT_BYTES:
        raise ValueError("GenCC export is empty or exceeds the 64 MB safety limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("GenCC export is not UTF-8 TSV") from error
    stream = io.StringIO(text, newline="")
    reader = csv.DictReader(stream, delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError("GenCC export is missing its TSV header")
    headers = [header.strip() for header in reader.fieldnames]
    if len(headers) != len(set(headers)) or not _REQUIRED_COLUMNS.issubset(headers):
        raise ValueError("GenCC export does not match the documented new-format submission columns")
    reader.fieldnames = headers
    csv.field_size_limit(1_000_000)
    row_count = 0
    for row in reader:
        if row is None or None in row or any(not isinstance(row.get(column), str) for column in _REQUIRED_COLUMNS):
            raise ValueError("GenCC export contains a malformed TSV row")
        if not row["sgc_id"].startswith("SGC-") or not row.get("version_number", "").isdigit():
            raise ValueError("GenCC export contains a row without its new-format stable submission ID and version")
        row_count += 1
        if row_count > _MAX_ROWS:
            raise ValueError("GenCC export exceeds 100000 submission rows")
    if row_count == 0:
        raise ValueError("GenCC export contains no submission records")
    return headers, row_count


def _normalize_row(row: dict[str, str], raw_hash: str) -> dict[str, Any]:
    report_url = row.get("sub_public_report_url", "").strip()
    criteria_url = row.get("sub_assertion_criteria_url", "").strip()
    for value in (report_url, criteria_url):
        if value and not value.startswith("https://"):
            raise ValueError("GenCC submission contains a non-HTTPS evidence URL")
    pmids = []
    for token in row.get("sub_pmids", "").split(",")[:50]:
        candidate = token.strip()
        if candidate.isdigit() and 1 <= len(candidate) <= 9:
            pmids.append(candidate)
    return {
        "sgc_id": row["sgc_id"],
        "version_number": row["version_number"],
        "gene_curie": row.get("gene_curie", "")[:80],
        "gene_symbol": row.get("gene_symbol", "")[:100],
        "disease_curie": row.get("disease_curie", "")[:100],
        "disease_title": row.get("disease_title", "")[:500],
        "disease_original_curie": row.get("disease_original_curie", "")[:100],
        "disease_original_title": row.get("disease_original_title", "")[:500],
        "classification_curie": row.get("classification_curie", "")[:100],
        "classification": row.get("classification_title", "")[:100],
        "mode_of_inheritance_curie": row.get("moi_curie", "")[:100],
        "mode_of_inheritance": row.get("moi_title", "")[:200],
        "submitter_curie": row.get("submitter_curie", "")[:100],
        "submitter": row.get("submitter_title", "")[:300],
        "submission_date": row.get("sub_date", "")[:80],
        "public_report_url": report_url[:1000] or None,
        "assertion_criteria_url": criteria_url[:1000] or None,
        "pubmed_ids": pmids,
        "source_submission_sha256": raw_hash,
    }


class GenCCClient:
    """Fetch weekly GenCC snapshots conditionally and query exact gene/disease IDs."""

    endpoint = "https://thegencc.org/download/action/submissions-export-tsv?format=new"

    def __init__(self, cache_dir: str | Path | None = None, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 90.0):
        configured = cache_dir or os.getenv("MEDAI_GENCC_CACHE_DIR", ".medai/gencc")
        self.cache_dir = Path(configured).expanduser().resolve()
        self.transport = transport
        self.timeout = timeout

    @property
    def export_path(self) -> Path:
        return self.cache_dir / "submissions.tsv"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / "refresh.lock"

    @property
    def refresh_state_path(self) -> Path:
        return self.cache_dir / "refresh-state.json"

    def _refresh_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self.refresh_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _read_cached(self) -> tuple[bytes, dict[str, Any]] | None:
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            payload = self.export_path.read_bytes()
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or metadata.get("schema_version") != "1":
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("sha256", ""))) or metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
            return None
        try:
            _parse_datetime(metadata.get("downloaded_at"))
            _parse_datetime(metadata.get("last_checked_at"))
            _headers, row_count = _validate_export(payload)
        except (ValueError, TypeError):
            return None
        if metadata.get("row_count") != row_count:
            return None
        return payload, metadata

    def refresh_due(self, now: datetime | None = None) -> bool:
        cached = self._read_cached()
        state = self._refresh_state()
        timestamps = []
        if cached is not None:
            timestamps.append(_parse_datetime(cached[1].get("last_checked_at")))
        if state.get("last_attempt_at"):
            try:
                timestamps.append(_parse_datetime(state["last_attempt_at"]))
            except ValueError:
                pass
        if not timestamps:
            return True
        current = now or _utc_now()
        return (current - max(timestamps)).total_seconds() >= _REFRESH_INTERVAL_SECONDS

    def _write_refresh_state(self, state: dict[str, Any]) -> None:
        self._atomic_write(self.refresh_state_path, (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))

    async def refresh_if_due(self) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_dir.chmod(0o700)
        with self.lock_path.open("a+b") as lock:
            self.lock_path.chmod(0o600)
            await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_EX)
            cached = self._read_cached()
            now = _utc_now()
            state = self._refresh_state()
            event_times = []
            if cached is not None:
                event_times.append(_parse_datetime(cached[1]["last_checked_at"]))
            if state.get("last_attempt_at"):
                try:
                    event_times.append(_parse_datetime(state["last_attempt_at"]))
                except ValueError:
                    pass
            if event_times and (now - max(event_times)).total_seconds() < _REFRESH_INTERVAL_SECONDS:
                if cached is None:
                    raise ValueError("GenCC refresh is rate-limited for 24 hours and no valid cached snapshot is available")
                if state.get("error"):
                    return {**cached[1], "cache_status": "stale_on_refresh_error", "stale_seconds": max(0, int((now - _parse_datetime(cached[1]["last_checked_at"])).total_seconds())), "network_requests": 0}
                return {**cached[1], "cache_status": "fresh", "network_requests": 0}
            attempt_state = {"last_attempt_at": _iso(now), "error": None}
            self._write_refresh_state(attempt_state)
            headers = {"Accept": "text/tab-separated-values", "User-Agent": "MedAI/biomedical-research"}
            if cached is not None and cached[1].get("etag"):
                headers["If-None-Match"] = cached[1]["etag"]
            if cached is not None and cached[1].get("last_modified"):
                headers["If-Modified-Since"] = cached[1]["last_modified"]
            try:
                async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
                    async with client.stream("GET", self.endpoint, headers=headers) as response:
                        if response.status_code == 304:
                            if cached is None:
                                raise ValueError("GenCC returned 304 without a verified local snapshot")
                            metadata = {**cached[1], "last_checked_at": _iso(now), "cache_status": "not_modified", "network_requests": 1}
                            self._write_metadata(metadata)
                            self._write_refresh_state(attempt_state)
                            return metadata
                        response.raise_for_status()
                        payload = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(chunk) > _MAX_EXPORT_BYTES - len(payload):
                                raise ValueError("GenCC export exceeds the 64 MB safety limit")
                            payload.extend(chunk)
                        response_headers = response.headers
                content = bytes(payload)
                _headers, row_count = _validate_export(content)
                digest = hashlib.sha256(content).hexdigest()
                metadata = {
                    "schema_version": "1",
                    "source": "GenCC new-format submissions export",
                    "source_url": self.endpoint,
                    "data_license": "CC0-1.0",
                    "attribution": "GenCC and contributing submitters",
                    "etag": response_headers.get("etag"),
                    "last_modified": response_headers.get("last-modified"),
                    "sha256": digest,
                    "row_count": row_count,
                    "downloaded_at": _iso(now),
                    "last_checked_at": _iso(now),
                    "cache_status": "downloaded",
                    "network_requests": 1,
                }
                self._atomic_write(self.export_path, content)
                self._write_metadata(metadata)
                self._write_refresh_state(attempt_state)
                return metadata
            except Exception:
                attempt_state["error"] = "refresh_failed"
                self._write_refresh_state(attempt_state)
                if cached is not None:
                    stale_seconds = max(0, int((now - _parse_datetime(cached[1]["last_checked_at"])).total_seconds()))
                    return {**cached[1], "cache_status": "stale_on_refresh_error", "stale_seconds": stale_seconds, "network_requests": 1}
                raise
            finally:
                await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_UN)

    async def search_gene(self, gene: str, limit: int = 50) -> dict[str, Any]:
        gene = gene.strip()
        normalized_gene = gene.upper()
        if not 1 <= limit <= _MAX_MATCHES:
            raise ValueError("GenCC result limit must be 1..100")
        if _HGNC.fullmatch(normalized_gene):
            key, normalized_query = "gene_curie", normalized_gene
        elif _SYMBOL.fullmatch(normalized_gene):
            key, normalized_query = "gene_symbol", normalized_gene
        else:
            raise ValueError("gene must be an HGNC CURIE or one exact gene symbol")
        return await self._search(key, normalized_query, limit)

    async def search_disease(self, disease_curie: str, limit: int = 50) -> dict[str, Any]:
        if not isinstance(disease_curie, str) or not _MONDO.fullmatch(disease_curie):
            raise ValueError("GenCC disease lookup requires an exact MONDO CURIE")
        if not 1 <= limit <= _MAX_MATCHES:
            raise ValueError("GenCC result limit must be 1..100")
        return await self._search("disease_curie", disease_curie, limit)

    async def _search(self, field: str, query: str, limit: int) -> dict[str, Any]:
        meta = await self.refresh_if_due()
        cached = self._read_cached()
        if cached is None:
            raise ValueError("GenCC local snapshot failed integrity verification")
        payload, metadata = cached
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
        matches: list[dict[str, Any]] = []
        total = 0
        for row in reader:
            value = row.get(field, "").strip()
            if (value.upper() if field == "gene_symbol" else value) != query:
                continue
            source_record = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            normalized = _normalize_row(row, hashlib.sha256(source_record).hexdigest())
            total += 1
            if len(matches) < limit:
                matches.append(normalized)
        return {
            "source": "Gene Curation Coalition (GenCC)",
            "source_url": metadata["source_url"],
            "data_license": "CC0-1.0",
            "attribution": "GenCC and contributing submitters",
            "dataset_sha256": metadata["sha256"],
            "dataset_last_modified": metadata.get("last_modified"),
            "dataset_downloaded_at": metadata["downloaded_at"],
            "dataset_cache_status": meta["cache_status"],
            "query": {field: query},
            "total_matching_assertions": total,
            "returned_assertions": len(matches),
            "truncated": total > len(matches),
            "assertions": matches,
            "interpretation_limit": "GenCC submission-level classifications and submitter assertions are preserved as published. This response does not aggregate submitters, resolve conflicts, establish pathogenicity, or provide clinical interpretation.",
        }

    def _write_metadata(self, metadata: dict[str, Any]) -> None:
        content = (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        self._atomic_write(self.metadata_path, content)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".gencc-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            path.chmod(0o600)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
