"""Crossref DOI and bibliography resolution; citations remain metadata links, not evidence polarity."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from urllib.parse import quote

import httpx

from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import read_bounded_response

_DOI = re.compile(r"^10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_MAX_BYTES = 2 * 1024 * 1024
_MAX_REFERENCES = 200
_MAX_AUTHORS = 50


def _date_part(record: dict) -> str | None:
    for field in ("published-print", "published-online", "published", "issued", "created"):
        value = record.get(field)
        parts = value.get("date-parts") if isinstance(value, dict) else None
        first = parts[0] if isinstance(parts, list) and parts else None
        if isinstance(first, list) and first and all(isinstance(x, int) for x in first[:3]):
            return "-".join(f"{x:02d}" if i > 0 else str(x) for i, x in enumerate(first[:3]))
    return None


def _text_list(value: object, limit: int = 1000) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:limit] for item in value if isinstance(item, str) and item.strip()][:10]


def _bounded_text(value: object, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) else None


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
        limits = {"key": 128, "DOI": 255, "article-title": 2000, "author": 256, "year": 16, "journal-title": 500, "volume": 64, "issue": 64, "first-page": 64, "unstructured": 2000}
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
            record["retained_reference_count"] = len(record["references"])
            record["references_retention_truncated"] = True
    if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()) > max_bytes:
        raise ValueError("Crossref metadata summary cannot fit within artifact retention limit")
    return payload


class CrossrefClient:
    endpoint = "https://api.crossref.org/works"
    _request_lock = asyncio.Lock()
    _last_request = 0.0

    def __init__(self, mailto: str | None = None, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        if mailto is not None and ("@" not in mailto or len(mailto) > 254):
            raise ValueError("mailto must be a valid contact email address")
        self.mailto = mailto
        self.transport = transport
        self.timeout = timeout

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
        async with type(self)._request_lock:
            delay = 0.21 - (time.monotonic() - type(self)._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            type(self)._last_request = time.monotonic()
            async with client.stream("GET", url, params=params, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research"}) as response:
                if response.status_code == 404:
                    raise ValueError("Crossref did not find this DOI")
                body = await read_bounded_response(response, _MAX_BYTES, "Crossref")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Crossref returned invalid JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise ValueError("Crossref returned an unexpected record shape")
        return payload["message"]

    def _params(self, params: dict) -> dict:
        return {**params, **({"mailto": self.mailto} if self.mailto else {})}

    async def get_work(self, doi: str) -> dict:
        doi = self.validate_doi(doi)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            record = await self._get(client, f"{self.endpoint}/{quote(doi, safe='/')}", self._params({}))
        if str(record.get("DOI", "")).casefold() != doi.casefold():
            raise ValueError("Crossref response DOI does not match the requested DOI")
        return normalize_work(record)

    async def search_bibliographic(self, query: str, limit: int = 10) -> list[dict]:
        query = ensure_public_research_query(query)
        if len(query) > 500:
            raise ValueError("bibliographic query cannot exceed 500 characters")
        if not 1 <= limit <= 20:
            raise ValueError("Crossref search limit must be 1..20")
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
        return result
