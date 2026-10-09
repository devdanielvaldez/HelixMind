"""Crossref DOI and bibliography resolution; citations remain metadata links, not evidence polarity."""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import event_loop_lock, read_bounded_response
from medai_science.json_input import strict_json_loads

_DOI = re.compile(r"^10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_MAX_BYTES = 2 * 1024 * 1024
_MAX_REFERENCES = 200
_MAX_AUTHORS = 50
_CACHE_MAX_TTL_SECONDS = 86_400
_CACHE_MAX_ENTRIES = 500
_CACHE_MAX_ENTRY_BYTES = 256 * 1024
_CACHE_WORK_FIELDS = {
    "doi", "title", "author", "container_title", "publisher", "type", "published_date",
    "volume", "issue", "page", "article_number", "issn", "url", "license_metadata",
    "references", "references_total_in_source_record", "references_truncated", "source",
    "source_work_url", "retrieved_at", "metadata_sha256", "reference_semantics",
}
_CACHE_REFERENCE_FIELDS = {
    "key", "DOI", "article-title", "author", "year", "journal-title", "volume", "issue",
    "first-page", "doi_validated_by_pattern", "reference_index",
}


def _cache_record_is_metadata_only(record: object) -> bool:
    """Accept only normalized bibliography fields; never persist arbitrary source prose."""
    if not isinstance(record, dict) or set(record) != _CACHE_WORK_FIELDS:
        return False
    if record.get("source") != "Crossref REST API" or not isinstance(record.get("doi"), str) or not _DOI.fullmatch(record["doi"]):
        return False
    references = record.get("references")
    if not isinstance(references, list) or len(references) > _MAX_REFERENCES:
        return False
    for reference in references:
        if not isinstance(reference, dict) or not set(reference) <= _CACHE_REFERENCE_FIELDS:
            return False
        index = reference.get("reference_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            return False
    licenses = record.get("license_metadata")
    if not isinstance(licenses, list) or len(licenses) > 20:
        return False
    for license_record in licenses:
        if not isinstance(license_record, dict) or not set(license_record) <= {"URL", "start", "delay-in-days"}:
            return False
    return True


def _cache_instant(now: int | None) -> int:
    if now is None:
        return int(time.time())
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise ValueError("cache timestamp must be a non-negative integer Unix time")
    return now


def _prune_invalid_timestamps(connection: sqlite3.Connection, instant: int, ttl_seconds: int) -> None:
    connection.execute(
        "DELETE FROM crossref_metadata_cache WHERE expires_at<=? OR typeof(stored_at)!='integer' OR stored_at<0 OR stored_at>? OR typeof(expires_at)!='integer' OR expires_at<=stored_at OR expires_at-stored_at>?",
        (instant, instant, ttl_seconds),
    )


def _date_part(record: dict) -> str | None:
    for field in ("published-print", "published-online", "published", "issued", "created"):
        value = record.get(field)
        parts = value.get("date-parts") if isinstance(value, dict) else None
        first = parts[0] if isinstance(parts, list) and parts else None
        if not isinstance(first, list) or not 1 <= len(first) <= 3:
            continue
        if any(isinstance(part, bool) or not isinstance(part, int) for part in first):
            continue
        year = first[0]
        if not 1 <= year <= 9999:
            continue
        if len(first) >= 2 and not 1 <= first[1] <= 12:
            continue
        if len(first) == 3:
            try:
                date(year, first[1], first[2])
            except ValueError:
                continue
        return "-".join(f"{part:02d}" if index > 0 else str(part) for index, part in enumerate(first))
    return None


def _text_list(value: object, limit: int = 1000) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:limit] for item in value if isinstance(item, str) and item.strip()][:10]


def _bounded_text(value: object, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def build_unresolved_reference_query(reference: object) -> dict[str, str]:
    """Build a bounded Crossref query from retained citation fields only."""
    if not isinstance(reference, dict):
        raise ValueError("citation reference must be an object")
    title = reference.get("article-title")
    if not isinstance(title, str) or not 12 <= len(title.strip()) <= 500:
        raise ValueError("unresolved citation requires an article title of 12..500 characters")
    fields = [("title", title.strip())]
    for key, label, limit in (("author", "author", 128), ("year", "year", 4), ("journal-title", "journal", 200)):
        value = reference.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"citation {label} metadata is malformed")
        normalized = str(value).strip()
        if not normalized or len(normalized) > limit or (label == "year" and not re.fullmatch(r"[0-9]{4}", normalized)):
            raise ValueError(f"citation {label} metadata is malformed")
        fields.append((label, normalized))
    query = " ".join(value for _, value in fields)
    if len(query.encode("utf-8")) > 500:
        raise ValueError("assembled bibliographic citation query exceeds 500 UTF-8 bytes")
    return {"query": query, "title": title.strip()}


def select_retained_reference(
    artifact: object,
    *,
    session_id: str,
    artifact_id: str,
    citing_doi: str,
    reference_index: int,
) -> dict:
    """Select one DOI-less reference from an exact hash-verified session artifact."""
    if not isinstance(artifact, dict) or artifact.get("session_id") != session_id or artifact.get("id") != artifact_id:
        raise ValueError("source artifact identity does not match the requested research session")
    if artifact.get("kind") != "crossref_citation_metadata" or artifact.get("source") != "Crossref REST API":
        raise ValueError("source artifact is not retained Crossref citation metadata")
    payload_text, payload_digest = artifact.get("payload"), artifact.get("sha256")
    if (not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > 250_000
            or not isinstance(payload_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload_digest)
            or hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != payload_digest):
        raise ValueError("source artifact payload is missing, oversized, or has a mismatched digest")
    payload = strict_json_loads(payload_text)
    if not isinstance(payload, dict) or payload.get("source") != "Crossref REST API":
        raise ValueError("source artifact payload is not a Crossref metadata object")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) > 5:
        raise ValueError("source artifact has a missing or oversized Crossref record list")
    canonical_records = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    records_digest = hashlib.sha256(canonical_records.encode("utf-8")).hexdigest()
    if records_digest != payload.get("retained_records_sha256"):
        raise ValueError("retained Crossref records do not match their digest")
    if not isinstance(citing_doi, str) or not _DOI.fullmatch(citing_doi):
        raise ValueError("citing DOI is invalid")
    if isinstance(reference_index, bool) or not isinstance(reference_index, int) or not 0 <= reference_index < 200:
        raise ValueError("reference_index must be an integer in 0..199")
    citing_work = next((
        row for row in records
        if isinstance(row, dict) and isinstance(row.get("doi"), str)
        and row["doi"].casefold() == citing_doi.casefold()
    ), None)
    if citing_work is None:
        raise ValueError("citing DOI is not present in the selected Crossref artifact")
    references = citing_work.get("references")
    if not isinstance(references, list):
        raise ValueError("citing work has no retained references array")
    reference = next((row for row in references if isinstance(row, dict) and row.get("reference_index") == reference_index), None)
    if reference is None:
        raise ValueError("reference index is not present in the selected citing work")
    source_cited_doi = reference.get("DOI") if isinstance(reference.get("DOI"), str) and reference.get("doi_validated_by_pattern") is True else None
    if source_cited_doi is None and not isinstance(reference.get("article-title"), str):
        raise ValueError("selected DOI-less citation has no bibliographic title")
    reference_digest = hashlib.sha256(
        json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "reference": reference,
        "reference_sha256": reference_digest,
        "source_artifact_sha256": payload_digest,
        "retained_records_sha256": records_digest,
        "source_cited_doi": source_cited_doi,
    }


def _strict_json_loads(payload: bytes) -> object:
    """Reject ambiguous object keys and non-standard numeric constants from the provider."""
    def object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Crossref JSON contains a duplicate object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"Crossref JSON contains a non-standard numeric constant: {value}")

    return json.loads(payload, object_pairs_hook=object_without_duplicate_keys, parse_constant=reject_constant)


def normalize_work(record: dict, retrieved_at: str | None = None) -> dict:
    doi = record.get("DOI")
    if not isinstance(doi, str) or len(doi) > 255 or not _DOI.fullmatch(doi):
        raise ValueError("Crossref work is missing a valid DOI")
    authors = []
    for author in record.get("author", [])[:_MAX_AUTHORS] if isinstance(record.get("author"), list) else []:
        if not isinstance(author, dict):
            continue
        authors.append({key: author[key][:256] for key in ("given", "family", "suffix", "ORCID") if isinstance(author.get(key), str)})
    references = []
    for index, ref in enumerate(record.get("reference", [])[:_MAX_REFERENCES] if isinstance(record.get("reference"), list) else []):
        if not isinstance(ref, dict):
            continue
        # Unstructured reference strings may contain copied article text. They
        # are unnecessary for DOI resolution and are not returned or cached.
        limits = {"key": 128, "DOI": 255, "article-title": 2000, "author": 256, "year": 16, "journal-title": 500, "volume": 64, "issue": 64, "first-page": 64}
        row = {key: ref[key][:limit] if isinstance(ref.get(key), str) else ref[key] for key, limit in limits.items() if isinstance(ref.get(key), (str, int))}
        if isinstance(row.get("DOI"), str):
            row["doi_validated_by_pattern"] = bool(_DOI.fullmatch(row["DOI"]))
        row["reference_index"] = index
        references.append(row)
    licenses = []
    for license_entry in record.get("license", [])[:20] if isinstance(record.get("license"), list) else []:
        if isinstance(license_entry, dict):
            license_row = {}
            if isinstance(license_entry.get("URL"), str):
                license_row["URL"] = license_entry["URL"][:2000]
            for key in ("start", "delay-in-days"):
                if isinstance(license_entry.get(key), int):
                    license_row[key] = license_entry[key]
            licenses.append(license_row)
    metadata = {
        "doi": doi.lower(),
        "title": _text_list(record.get("title"), 2000),
        "author": authors,
        "container_title": _text_list(record.get("container-title"), 500),
        "publisher": _bounded_text(record.get("publisher"), 500),
        "type": _bounded_text(record.get("type"), 100),
        "published_date": _date_part(record),
        "volume": _bounded_text(record.get("volume"), 64),
        "issue": _bounded_text(record.get("issue"), 64),
        "page": _bounded_text(record.get("page"), 128),
        "article_number": _bounded_text(record.get("article-number"), 128),
        "issn": _text_list(record.get("ISSN"), 32),
        "url": _bounded_text(record.get("URL"), 2048) or f"https://doi.org/{doi}",
        "license_metadata": licenses,
        "references": references,
        "references_total_in_source_record": record.get("references-count") if isinstance(record.get("references-count"), int) else None,
        "references_truncated": isinstance(record.get("reference"), list) and len(record["reference"]) > _MAX_REFERENCES,
    }
    canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    retrieved = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        **metadata,
        "source": "Crossref REST API",
        "source_work_url": f"https://api.crossref.org/works/{quote(doi, safe='/')}",
        "retrieved_at": retrieved,
        "metadata_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "reference_semantics": "References are publisher/member-deposited bibliographic links. A citation does not establish support, contradiction, quality, or causal evidence.",
    }


def build_retention_payload(result: list[dict], max_bytes: int = 240_000) -> dict:
    """Create a compact, bounded artifact receipt without abstracts or free-text references."""
    if not 1 <= max_bytes <= 250_000:
        raise ValueError("retained artifact size limit must be 1..250000 bytes")
    retained = []
    for row in result:
        item = {key: row[key] for key in ("doi", "title", "author", "container_title", "publisher", "type", "published_date", "url", "license_metadata", "metadata_sha256", "references_total_in_source_record", "references_truncated") if key in row}
        references = row.get("references", [])
        compact = []
        source_reference_count = len(references) if isinstance(references, list) else 0
        for reference in references[:150] if isinstance(references, list) else []:
            if not isinstance(reference, dict):
                continue
            citation = {key: str(reference[key])[:limit] if isinstance(reference[key], str) else reference[key] for key, limit in (("DOI", 256), ("key", 128), ("year", 16), ("volume", 64), ("issue", 64), ("first-page", 64), ("article-title", 500), ("author", 200), ("journal-title", 300)) if key in reference}
            citation["reference_index"] = reference.get("reference_index")
            compact.append(citation)
        item["references"] = compact
        item["references_retained_count"] = len(compact)
        item["references_retention_truncated"] = len(compact) < source_reference_count
        item["unstructured_reference_text_retained"] = False
        retained.append(item)
    payload = {"source": "Crossref REST API", "records": retained, "complete_result_sha256": hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "retention": "Publisher-deposited bibliographic references retained without abstracts or unstructured citation text; full tool result is returned to the caller."}
    while len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) > max_bytes and any(record.get("references") for record in payload["records"]):
        for record in payload["records"]:
            refs = record.get("references", [])
            record["references"] = refs[:max(0, len(refs) - 25)]
            record["references_retained_count"] = len(record["references"])
            record["references_retention_truncated"] = True
    if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) > max_bytes:
        raise ValueError("Crossref metadata summary cannot fit within artifact retention limit")
    payload["retained_records_sha256"] = hashlib.sha256(
        json.dumps(payload["records"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


class CrossrefMetadataCache:
    """Short-lived cache of normalized bibliography; never stores query text or full text."""

    def __init__(self, path: str | Path, *, ttl_seconds: int = 300):
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or not 1 <= ttl_seconds <= _CACHE_MAX_TTL_SECONDS:
            raise ValueError(f"Crossref metadata cache TTL must be 1..{_CACHE_MAX_TTL_SECONDS} seconds")
        self.path = Path(path).expanduser()
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("Crossref cache path cannot be a symlink")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink():
            raise ValueError("Crossref cache path cannot be a symlink")
        connection = sqlite3.connect(self.path, timeout=5)
        os.chmod(self.path, 0o600)
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS crossref_metadata_cache (cache_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, stored_at INTEGER NOT NULL, expires_at INTEGER NOT NULL)"
            )

    @staticmethod
    def key(operation: str, arguments: dict) -> str:
        if operation not in {"doi", "bibliographic_search"} or not isinstance(arguments, dict):
            raise ValueError("Crossref cache key requires a supported operation and object arguments")
        encoded = json.dumps(["crossref-normalized-metadata-v1", operation, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, cache_key: str, *, now: int | None = None) -> dict | list[dict] | None:
        if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise ValueError("Crossref cache key must be a lowercase SHA-256 digest")
        instant = _cache_instant(now)
        with self._connect() as connection:
            _prune_invalid_timestamps(connection, instant, self.ttl_seconds)
            row = connection.execute(
                "SELECT payload_json,payload_sha256,stored_at,expires_at FROM crossref_metadata_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        payload_json, expected_digest, stored_at, expires_at = row
        if (
            isinstance(stored_at, bool) or not isinstance(stored_at, int)
            or isinstance(expires_at, bool) or not isinstance(expires_at, int)
            or stored_at < 0 or stored_at > instant
            or expires_at <= instant or expires_at <= stored_at
            or expires_at - stored_at > self.ttl_seconds
        ):
            self.delete(cache_key)
            return None
        try:
            payload_bytes = payload_json.encode("utf-8")
            payload = _strict_json_loads(payload_bytes)
            if hashlib.sha256(payload_bytes).hexdigest() != expected_digest or len(payload_bytes) > _CACHE_MAX_ENTRY_BYTES:
                raise ValueError("Crossref cache payload digest or size is invalid")
            if isinstance(payload, dict):
                if not _cache_record_is_metadata_only(payload):
                    raise ValueError("Crossref cache payload has an invalid metadata-only record shape")
                return payload
            if isinstance(payload, list) and all(
                _cache_record_is_metadata_only(item)
                for item in payload
            ):
                return payload
            raise ValueError("Crossref cache payload has an invalid metadata-only record shape")
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            with self._connect() as connection:
                connection.execute("DELETE FROM crossref_metadata_cache WHERE cache_key=?", (cache_key,))
            return None

    def put(self, cache_key: str, payload: dict | list[dict], *, now: int | None = None) -> bool:
        if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
            raise ValueError("Crossref cache key must be a lowercase SHA-256 digest")
        payload_bytes = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(payload_bytes) > _CACHE_MAX_ENTRY_BYTES:
            return False
        records = payload if isinstance(payload, list) else [payload]
        if not all(_cache_record_is_metadata_only(record) for record in records):
            raise ValueError("Crossref cache accepts only normalized metadata-only records")
        instant = _cache_instant(now)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        with self._connect() as connection:
            _prune_invalid_timestamps(connection, instant, self.ttl_seconds)
            connection.execute(
                "INSERT INTO crossref_metadata_cache(cache_key,payload_json,payload_sha256,stored_at,expires_at) VALUES(?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET payload_json=excluded.payload_json,payload_sha256=excluded.payload_sha256,stored_at=excluded.stored_at,expires_at=excluded.expires_at",
                (cache_key, payload_bytes.decode("utf-8"), digest, instant, instant + self.ttl_seconds),
            )
            connection.execute(
                "DELETE FROM crossref_metadata_cache WHERE cache_key NOT IN (SELECT cache_key FROM crossref_metadata_cache ORDER BY stored_at DESC,cache_key ASC LIMIT ?)",
                (_CACHE_MAX_ENTRIES,),
            )
        return True

    def delete(self, cache_key: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM crossref_metadata_cache WHERE cache_key=?", (cache_key,))


class CrossrefClient:
    endpoint = "https://api.crossref.org/works"
    _last_request = 0.0

    def __init__(self, mailto: str | None = None, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0, cache_path: str | None = None):
        if mailto is not None and ("@" not in mailto or len(mailto) > 254):
            raise ValueError("mailto must be a valid contact email address")
        self.mailto = mailto
        self.transport = transport
        self.timeout = timeout
        configured_cache = cache_path or os.environ.get("MEDAI_CROSSREF_METADATA_CACHE")
        self.metadata_cache = CrossrefMetadataCache(configured_cache) if configured_cache else None

    @staticmethod
    def validate_doi(doi: str) -> str:
        value = doi.strip()
        value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
        if value.casefold().startswith("doi:"):
            value = value[4:].strip()
        if not _DOI.fullmatch(value):
            raise ValueError("provide a valid DOI beginning with 10.")
        return value

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
        async with event_loop_lock("crossref"):
            delay = 0.21 - (time.monotonic() - type(self)._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            type(self)._last_request = time.monotonic()
            async with client.stream("GET", url, params=params, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research"}) as response:
                if response.status_code == 404:
                    raise ValueError("Crossref did not find this DOI")
                body = await read_bounded_response(response, _MAX_BYTES, "Crossref")
        try:
            payload = _strict_json_loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError("Crossref returned invalid JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise ValueError("Crossref returned an unexpected record shape")
        return payload["message"]

    def _params(self, params: dict) -> dict:
        return {**params, **({"mailto": self.mailto} if self.mailto else {})}

    async def get_work(self, doi: str) -> dict:
        doi = self.validate_doi(doi)
        cache_key = self.metadata_cache.key("doi", {"doi": doi.casefold()}) if self.metadata_cache else None
        if self.metadata_cache and cache_key:
            cached = self.metadata_cache.get(cache_key)
            if isinstance(cached, dict) and cached.get("doi", "").casefold() == doi.casefold():
                return cached
            if cached is not None:
                self.metadata_cache.delete(cache_key)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            record = await self._get(client, f"{self.endpoint}/{quote(doi, safe='/')}", self._params({}))
        if str(record.get("DOI", "")).casefold() != doi.casefold():
            raise ValueError("Crossref response DOI does not match the requested DOI")
        result = normalize_work(record)
        if self.metadata_cache and cache_key:
            self.metadata_cache.put(cache_key, result)
        return result

    async def search_bibliographic(self, query: str, limit: int = 10) -> list[dict]:
        query = ensure_public_research_query(query)
        if len(query) > 500:
            raise ValueError("bibliographic query cannot exceed 500 characters")
        if not 1 <= limit <= 20:
            raise ValueError("Crossref search limit must be 1..20")
        cache_key = self.metadata_cache.key("bibliographic_search", {"query": query, "limit": limit}) if self.metadata_cache else None
        if self.metadata_cache and cache_key:
            cached = self.metadata_cache.get(cache_key)
            if isinstance(cached, list) and len(cached) <= limit:
                return cached
            if cached is not None:
                self.metadata_cache.delete(cache_key)
        fields = "DOI,title,author,container-title,publisher,type,published-print,published-online,published,issued,volume,issue,page,article-number,ISSN,URL,license,reference,references-count"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            message = await self._get(client, self.endpoint, self._params({"query.bibliographic": query, "rows": limit, "select": fields}))
        items = message.get("items")
        if not isinstance(items, list):
            raise ValueError("Crossref search response has no item list")
        result = []
        for record in items[:limit]:
            if isinstance(record, dict):
                try:
                    result.append(normalize_work(record))
                except ValueError:
                    continue
        if self.metadata_cache and cache_key:
            self.metadata_cache.put(cache_key, result)
        return result
