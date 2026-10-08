"""Exact HGNC/MONDO lookup over ClinGen's published gene-disease validity CSV."""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import fcntl
import httpx
from medai_science.sources.bounded_http import read_bounded_response


SOURCE_URL = "https://search.clinicalgenome.org/kb/gene-validity/download"
_REQUIRED = (
    "GENE SYMBOL", "GENE ID (HGNC)", "DISEASE LABEL", "DISEASE ID (MONDO)", "MOI",
    "SOP", "CLASSIFICATION", "ONLINE REPORT", "CLASSIFICATION DATE", "GCEP",
)
_HGNC = re.compile(r"^HGNC:[0-9]+$", re.IGNORECASE)
_MONDO = re.compile(r"^MONDO:[0-9]{7}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,39}$")
_LIMIT = 100
_MAX_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 20_000
_INTERVAL = timedelta(hours=24)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("ClinGen cache timestamp is invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("ClinGen cache timestamp is invalid") from error
    if result.tzinfo is None:
        raise ValueError("ClinGen cache timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _is_separator(row: list[str]) -> bool:
    return bool(row) and all(value and set(value) == {"+"} for value in row)


def _parse_export(payload: bytes) -> tuple[list[dict[str, str]], str | None]:
    if not payload or len(payload) > _MAX_BYTES:
        raise ValueError("ClinGen export is empty or exceeds the 16 MB safety limit")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("ClinGen export is not UTF-8 CSV") from error
    reader = csv.reader(io.StringIO(text, newline=""))
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    created_at = None
    for row in reader:
        if len(row) > len(_REQUIRED) + 4:
            raise ValueError("ClinGen export contains an unexpected number of columns")
        if row:
            date_match = re.fullmatch(r"FILE CREATED:\s*(\d{4}-\d{2}-\d{2})", row[0].strip())
            if date_match:
                created_at = date_match.group(1)
            elif len(row) > 1 and row[0].strip() == "FILE CREATED:" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[1].strip()):
                created_at = row[1].strip()
        normalized = tuple(value.strip() for value in row)
        if normalized == _REQUIRED:
            header = list(_REQUIRED)
            continue
        if header is None or _is_separator(row) or not row:
            continue
        if len(row) != len(_REQUIRED):
            raise ValueError("ClinGen export contains a malformed gene-disease row")
        record = {key: value.strip() for key, value in zip(header, row)}
        if not _HGNC.fullmatch(record["GENE ID (HGNC)"]) or not _MONDO.fullmatch(record["DISEASE ID (MONDO)"]):
            raise ValueError("ClinGen record is missing the expected stable HGNC or MONDO identifier")
        if not record["GENE SYMBOL"] or not record["DISEASE LABEL"] or not record["CLASSIFICATION"]:
            raise ValueError("ClinGen record is missing a required gene, disease, or classification field")
        if not record["ONLINE REPORT"].startswith("https://search.clinicalgenome.org/kb/gene-validity/"):
            raise ValueError("ClinGen report URL is not on the official HTTPS gene-validity domain")
        if record["CLASSIFICATION DATE"]:
            try:
                datetime.fromisoformat(record["CLASSIFICATION DATE"].replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("ClinGen classification date is malformed") from error
        rows.append(record)
        if len(rows) > _MAX_ROWS:
            raise ValueError("ClinGen export exceeds 20000 curation rows")
    if header is None or not rows:
        raise ValueError("ClinGen export does not match the published gene-disease validity format")
    return rows, created_at


class ClinGenGeneValidityClient:
    """Cache the official summary export and preserve each expert-panel curation."""

    def __init__(self, cache_dir: str | Path | None = None, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 60):
        configured = cache_dir or os.getenv("MEDAI_CLINGEN_CACHE_DIR", ".medai/clingen")
        self.cache_dir = Path(configured).expanduser().resolve()
        self.transport = transport
        self.timeout = timeout

    @property
    def data_path(self) -> Path:
        return self.cache_dir / "gene-validity.csv"

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "metadata.json"

    @property
    def lock_path(self) -> Path:
        return self.cache_dir / "refresh.lock"

    @property
    def attempt_path(self) -> Path:
        return self.cache_dir / "refresh-state.json"

    def _attempt_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.attempt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _cached(self) -> tuple[bytes, dict[str, Any]] | None:
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            payload = self.data_path.read_bytes()
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or metadata.get("schema_version") != "1":
            return None
        if metadata.get("sha256") != hashlib.sha256(payload).hexdigest():
            return None
        try:
            rows, created_at = _parse_export(payload)
            _timestamp(metadata.get("checked_at"))
        except (ValueError, TypeError):
            return None
        if metadata.get("row_count") != len(rows) or metadata.get("source_file_date") != created_at:
            return None
        return payload, metadata

    def refresh_due(self, now: datetime | None = None) -> bool:
        cached = self._cached()
        moments = []
        if cached is not None:
            moments.append(_timestamp(cached[1]["checked_at"]))
        attempt = self._attempt_state()
        if attempt.get("last_attempt_at"):
            try:
                moments.append(_timestamp(attempt["last_attempt_at"]))
            except ValueError:
                pass
        return not moments or (now or _now()) - max(moments) >= _INTERVAL

    async def refresh_if_due(self) -> dict[str, Any]:
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_dir.chmod(0o700)
        with self.lock_path.open("a+b") as lock:
            self.lock_path.chmod(0o600)
            await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_EX)
            cached = self._cached()
            now = _now()
            state = self._attempt_state()
            try:
                moments = []
                if cached:
                    moments.append(_timestamp(cached[1]["checked_at"]))
                if state.get("last_attempt_at"):
                    try:
                        moments.append(_timestamp(state["last_attempt_at"]))
                    except ValueError:
                        pass
                if moments and now - max(moments) < _INTERVAL:
                    if cached is None:
                        raise ValueError("ClinGen refresh is rate-limited and no verified local export is available")
                    status = "stale_on_refresh_error" if state.get("error") else "fresh"
                    return {**cached[1], "cache_status": status, "network_requests": 0}
                state = {"last_attempt_at": _iso(now), "error": None}
                self._write(self.attempt_path, _json_bytes(state))
                headers = {"Accept": "text/csv", "User-Agent": "MedAI/biomedical-research"}
                if cached and cached[1].get("etag"):
                    headers["If-None-Match"] = cached[1]["etag"]
                if cached and cached[1].get("last_modified"):
                    headers["If-Modified-Since"] = cached[1]["last_modified"]
                async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
                    async with client.stream("GET", SOURCE_URL, headers=headers) as response:
                        if response.status_code == 304:
                            if cached is None:
                                raise ValueError("ClinGen returned 304 without a verified local export")
                            metadata = {**cached[1], "checked_at": _iso(now), "cache_status": "not_modified", "network_requests": 1}
                            self._write(self.metadata_path, _json_bytes(metadata))
                            self._write(self.attempt_path, _json_bytes(state))
                            return metadata
                        body = await read_bounded_response(response, _MAX_BYTES, "ClinGen export")
                        response_headers = response.headers
                payload = bytes(body)
                rows, file_date = _parse_export(payload)
                metadata = {
                    "schema_version": "1", "source": "ClinGen Gene-Disease Validity Curations",
                    "source_url": SOURCE_URL, "attribution": "Clinical Genome Resource (ClinGen)",
                    "sha256": hashlib.sha256(payload).hexdigest(), "row_count": len(rows),
                    "source_file_date": file_date, "checked_at": _iso(now),
                    "etag": response_headers.get("etag"), "last_modified": response_headers.get("last-modified"),
                }
                self._write(self.data_path, payload)
                self._write(self.metadata_path, _json_bytes(metadata))
                self._write(self.attempt_path, _json_bytes(state))
                return {**metadata, "cache_status": "downloaded", "network_requests": 1}
            except Exception:
                state["error"] = "refresh_failed"
                self._write(self.attempt_path, _json_bytes(state))
                if cached is not None:
                    return {**cached[1], "cache_status": "stale_on_refresh_error", "network_requests": 1}
                raise
            finally:
                await asyncio.to_thread(fcntl.flock, lock.fileno(), fcntl.LOCK_UN)

    async def search_gene(self, gene: str, limit: int = 50) -> dict[str, Any]:
        normalized = gene.strip().upper()
        if _HGNC.fullmatch(normalized):
            field, query = "GENE ID (HGNC)", normalized
        elif _SYMBOL.fullmatch(normalized):
            field, query = "GENE SYMBOL", normalized
        else:
            raise ValueError("gene must be an HGNC CURIE or exact gene symbol")
        return await self._search(field, query, limit)

    async def search_disease(self, mondo_id: str, limit: int = 50) -> dict[str, Any]:
        if not isinstance(mondo_id, str) or not _MONDO.fullmatch(mondo_id):
            raise ValueError("ClinGen disease lookup requires an exact MONDO CURIE")
        return await self._search("DISEASE ID (MONDO)", mondo_id, limit)

    async def _search(self, field: str, query: str, limit: int) -> dict[str, Any]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _LIMIT:
            raise ValueError("ClinGen result limit must be 1..100")
        meta = await self.refresh_if_due()
        cached = self._cached()
        if cached is None:
            raise ValueError("ClinGen local export failed integrity verification")
        rows, _ = _parse_export(cached[0])
        matches = []
        for row in rows:
            value = row[field].upper() if field == "GENE SYMBOL" else row[field]
            if value != query:
                continue
            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            matches.append({
                "gene_symbol": row["GENE SYMBOL"], "gene_hgnc_id": row["GENE ID (HGNC)"],
                "disease_label": row["DISEASE LABEL"], "disease_mondo_id": row["DISEASE ID (MONDO)"],
                "mode_of_inheritance": row["MOI"], "sop": row["SOP"],
                "classification": row["CLASSIFICATION"], "report_url": row["ONLINE REPORT"],
                "classification_date": row["CLASSIFICATION DATE"], "expert_panel": row["GCEP"],
                "record_sha256": hashlib.sha256(canonical).hexdigest(),
            })
        return {
            "source": "ClinGen Gene-Disease Validity", "source_url": SOURCE_URL,
            "attribution": "Clinical Genome Resource (ClinGen)", "dataset_sha256": meta["sha256"],
            "source_file_date": meta.get("source_file_date"), "dataset_cache_status": meta["cache_status"],
            "query": {"field": field, "value": query}, "total_matching_curations": len(matches),
            "curations": matches[:limit], "truncated": len(matches) > limit,
            "interpretation_limit": "ClinGen expert-panel gene-disease validity classification summarizes a curated, time-specific assessment; it is not a systematic review of all literature, variant classification, diagnostic result, or clinical recommendation. Do not merge its classification with GenCC, PanelApp, or association scores.",
        }

    @staticmethod
    def _write(path: Path, payload: bytes) -> None:
        descriptor, name = tempfile.mkstemp(prefix=".clingen-", dir=path.parent)
        temporary = Path(name)
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


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
