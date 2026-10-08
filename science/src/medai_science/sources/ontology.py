"""Bounded biomedical ontology term search through EMBL-EBI OLS4."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

import httpx

from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import read_bounded_response


_MAX_BYTES = 2_000_000
_MAX_ONTOLOGY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _bounded_strings(value: object, count: int = 20, length: int = 1000) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    return [item.strip()[:length] for item in values[:count] if isinstance(item, str) and item.strip()]


def _normalize_document(document: dict, retrieved_at: str) -> dict:
    strings = {key: document.get(key)[:2000] if isinstance(document.get(key), str) else None for key in (
        "iri", "short_form", "obo_id", "label", "ontology_name", "ontology_prefix", "ontology_iri", "ontology_link",
    )}
    description = _bounded_strings(document.get("description"), count=5)
    synonyms = _bounded_strings(document.get("synonym"), count=20, length=500)
    if not synonyms:
        synonyms = _bounded_strings(document.get("synonyms"), count=20, length=500)
    record = {
        **strings,
        "description": description,
        "synonyms": synonyms,
        "is_obsolete": document.get("is_obsolete") if isinstance(document.get("is_obsolete"), bool) else None,
        "is_defining_ontology": document.get("is_defining_ontology") if isinstance(document.get("is_defining_ontology"), bool) else None,
        "source": "EMBL-EBI Ontology Lookup Service (OLS4)",
        "retrieved_at": retrieved_at,
    }
    canonical_source = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record["source_record_sha256"] = hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()
    return record


class OntologyClient:
    endpoint = "https://www.ebi.ac.uk/ols4/api/search"

    def __init__(self, client_factory=httpx.AsyncClient):
        self._client_factory = client_factory

    @staticmethod
    def validate(query: str, ontology: str | None, limit: int) -> tuple[str, str | None, int]:
        query = ensure_public_research_query(query.strip())
        if not query or len(query) > 256:
            raise ValueError("ontology query must contain 1..256 characters")
        if ontology is not None:
            ontology = ontology.strip().casefold()
            if not _MAX_ONTOLOGY.fullmatch(ontology):
                raise ValueError("ontology must be a valid OLS ontology identifier")
        if isinstance(limit, bool) or not 1 <= limit <= 50:
            raise ValueError("ontology result limit must be 1..50")
        return query, ontology, limit

    async def search(self, query: str, ontology: str | None = None, limit: int = 20) -> dict:
        query, ontology, limit = self.validate(query, ontology, limit)
        params = {"q": query, "rows": limit, "start": 0, "groupField": "iri"}
        if ontology:
            params["ontology"] = ontology
        async with self._client_factory(timeout=25, follow_redirects=False) as client:
            async with client.stream("GET", self.endpoint, params=params, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research"}) as response:
                body = await read_bounded_response(response, _MAX_BYTES, "OLS4")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("OLS4 returned invalid JSON") from error
        response = payload.get("response") if isinstance(payload, dict) else None
        docs = response.get("docs") if isinstance(response, dict) else None
        if not isinstance(docs, list):
            raise ValueError("OLS4 returned an unsupported search response shape")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        records = [_normalize_document(row, retrieved_at) for row in docs[:limit] if isinstance(row, dict)]
        return {
            "query": query,
            "ontology_filter": ontology,
            "records": records,
            "returned_count": len(records),
            "provider_reported_count": response.get("numFound") if isinstance(response.get("numFound"), int) else None,
            "source": "EMBL-EBI Ontology Lookup Service (OLS4)",
            "source_url": self.endpoint,
            "retrieved_at": retrieved_at,
            "search_semantics": "OLS4 provider-ranked lexical/full-text search; labels, synonyms, descriptions, and imported terms may contribute. Result order is not a confidence score.",
            "interpretation_limit": "Ontology terms and cross-ontology mappings require scientific review. Search rank is not evidence strength, identity equivalence, or a disease association.",
        }

    async def resolve_identifier(self, identifier: str, ontology: str | None = None) -> dict:
        """Search and retain only records whose identifier field exactly matches."""
        result = await self.search(identifier, ontology, 50)
        needle = identifier.strip().casefold()
        matches = []
        for record in result["records"]:
            fields = (record.get("obo_id"), record.get("short_form"), record.get("iri"))
            if any(isinstance(value, str) and value.casefold() == needle for value in fields):
                matches.append(record)
        return {**result, "identifier_query": identifier, "records": matches, "returned_count": len(matches), "resolution_policy": "exact case-insensitive match against OLS obo_id, short_form, or IRI only; search rank is not used to infer identity"}
