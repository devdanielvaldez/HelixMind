"""Metadata lookup for bioRxiv and medRxiv preprints by exact DOI."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import re
from urllib.parse import quote

import httpx
from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query
from medai_science.json_input import strict_json_loads


_DOI = re.compile(r"^10\.1101/[A-Za-z0-9._;()/:-]{1,200}$", re.IGNORECASE)
_MAX_RESPONSE_BYTES = 2_000_000


class PreprintClient:
    endpoints = {"biorxiv": "https://api.biorxiv.org", "medrxiv": "https://api.medrxiv.org"}

    def __init__(self, client_factory=None):
        self.client_factory = client_factory or httpx.AsyncClient

    @staticmethod
    def _retain_record(record: dict, server: str, retrieved_at: str) -> dict:
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        doi = record.get("doi", "")
        return {
            "doi": doi[:256] if isinstance(doi, str) else "",
            "server": server,
            "title": record.get("title", "")[:2000] if isinstance(record.get("title"), str) else None,
            "authors": record.get("authors", "")[:4000] if isinstance(record.get("authors"), str) else None,
            "date": record.get("date") if isinstance(record.get("date"), str) else None,
            "version": record.get("version") if isinstance(record.get("version"), (str, int)) else None,
            "type": record.get("type") if isinstance(record.get("type"), str) else None,
            "license": record.get("license") if isinstance(record.get("license"), str) else None,
            "category": record.get("category") if isinstance(record.get("category"), str) else None,
            "published": record.get("published") if isinstance(record.get("published"), (str, int, bool)) else None,
            "source_jats_path": record.get("jatsxml") if isinstance(record.get("jatsxml"), str) else None,
            "source": "bioRxiv/medRxiv API",
            "source_record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "retrieved_at": retrieved_at,
            "abstract_retained": False,
            "peer_review_status": "preprint_server_record; publication metadata is provider supplied and peer review is not established by this lookup",
        }

    @staticmethod
    def validate(server: str, doi: str) -> tuple[str, str]:
        if server not in {"biorxiv", "medrxiv"}:
            raise ValueError("preprint server must be biorxiv or medrxiv")
        normalized = doi.strip()
        if not _DOI.fullmatch(normalized):
            raise ValueError("preprint DOI must be an exact 10.1101 DOI")
        return server, normalized

    async def get_by_doi(self, server: str, doi: str) -> dict:
        server, doi = self.validate(server, doi)
        path_doi = quote(doi, safe="/.")
        async with self.client_factory(timeout=20, follow_redirects=False) as client:
            async with client.stream("GET", f"{self.endpoints[server]}/details/{server}/{path_doi}/na/json") as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "preprint API")
        try:
            payload = strict_json_loads(raw)
        except ValueError as error:
            raise ValueError(f"preprint API returned invalid strict JSON: {error}") from error
        collection = payload.get("collection") if isinstance(payload, dict) else None
        if not isinstance(collection, list) or len(collection) > 10:
            raise ValueError("preprint API returned an invalid collection")
        if any(not isinstance(record, dict) or not isinstance(record.get("doi"), str) for record in collection):
            raise ValueError("preprint API returned a malformed collection record")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        matches = []
        for record in collection:
            if record["doi"].casefold() != doi.casefold():
                continue
            matches.append(self._retain_record(record, server, retrieved_at))
        return {
            "query_doi": doi,
            "server": server,
            "records": matches,
            "returned_count": len(matches),
            "retrieved_at": retrieved_at,
            "interpretation_limit": "Preprint metadata and any published-version link are provider supplied. This lookup does not establish peer review, scientific validity, or publication equivalence.",
        }

    async def search(
        self,
        server: str,
        query: str,
        start_date: str,
        end_date: str,
        limit: int = 20,
        category: str | None = None,
        max_pages: int = 2,
    ) -> dict:
        """Lexically screen a bounded date/category slice; API abstracts are transient only."""
        if server not in self.endpoints:
            raise ValueError("preprint server must be biorxiv or medrxiv")
        query = ensure_public_research_query(query.strip())
        if not query or len(query) > 256:
            raise ValueError("preprint query must contain 1..256 characters")
        terms = sorted(set(term for term in re.findall(r"[a-z0-9]+", query.casefold()) if len(term) > 1))
        if not terms:
            raise ValueError("preprint query must contain at least one searchable term")
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except (TypeError, ValueError):
            raise ValueError("preprint date bounds must use YYYY-MM-DD") from None
        if end < start or end - start > timedelta(days=92):
            raise ValueError("preprint search interval must be ordered and at most 93 days")
        if not 1 <= limit <= 50 or not 1 <= max_pages <= 3:
            raise ValueError("preprint limit must be 1..50 and max_pages must be 1..3")
        if category is not None and (not category.strip() or len(category) > 100):
            raise ValueError("preprint category must contain 1..100 characters")
        page_size = 30 if server == "biorxiv" else 100
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        scanned: list[dict] = []
        pages = 0
        exhausted = False
        async with self.client_factory(timeout=20, follow_redirects=False) as client:
            for page in range(max_pages):
                cursor = page * page_size
                path = f"/details/{server}/{start.isoformat()}/{end.isoformat()}/{cursor}/json"
                async with client.stream("GET", f"{self.endpoints[server]}{path}", params={"category": category} if category else None) as response:
                    raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "preprint API")
                try:
                    payload = strict_json_loads(raw)
                except ValueError as error:
                    raise ValueError(f"preprint API returned invalid strict JSON: {error}") from error
                collection = payload.get("collection") if isinstance(payload, dict) else None
                if not isinstance(collection, list) or len(collection) > page_size:
                    raise ValueError("preprint API returned an invalid collection")
                if any(not isinstance(item, dict) or not isinstance(item.get("doi"), str) for item in collection):
                    raise ValueError("preprint API returned a malformed collection record")
                rows = collection
                scanned.extend(rows)
                pages += 1
                if len(rows) < page_size:
                    exhausted = True
                    break
        matches = []
        for record in scanned:
            title = record.get("title") if isinstance(record.get("title"), str) else ""
            abstract = record.get("abstract") if isinstance(record.get("abstract"), str) else ""
            words = set(re.findall(r"[a-z0-9]+", f"{title} {abstract}".casefold()))
            found = [term for term in terms if term in words]
            if len(found) != len(terms) or not isinstance(record.get("doi"), str):
                continue
            retained = self._retain_record(record, server, retrieved_at)
            retained["lexical_screen"] = {
                "query_terms": terms,
                "matched_terms": found,
                "matched_fields": [field for field, value in (("title", title), ("abstract", abstract)) if any(term in set(re.findall(r"[a-z0-9]+", value.casefold())) for term in found)],
                "method": "all query tokens matched title or transient abstract text; not semantic relevance ranking",
            }
            matches.append(retained)
            if len(matches) >= limit:
                break
        return {
            "server": server,
            "query": query,
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "category": category,
            "records": matches,
            "returned_count": len(matches),
            "scanned_count": len(scanned),
            "pages_requested": pages,
            "scan_exhausted": exhausted,
            "coverage_limit": None if exhausted else f"The date/category slice may contain more records than the {pages} page scan limit.",
            "abstracts_retained": False,
            "interpretation_limit": "This is a bounded lexical screen of provider-returned titles and transient abstracts, not a comprehensive or semantic search. Preprint metadata does not establish peer review, scientific validity, or publication equivalence.",
        }
