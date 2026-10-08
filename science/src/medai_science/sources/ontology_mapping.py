"""Source-linked candidate disease ontology mappings through EMBL-EBI OxO."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

import httpx

from medai_science.sources.bounded_http import read_bounded_response


_MAX_BYTES = 1_000_000
_DISEASE_CURIE = re.compile(r"^(?:EFO_[0-9]{7,}|MONDO:[0-9]{7})$")
_PREFIX = re.compile(r"^[A-Z][A-Z0-9_-]{0,31}$")


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON numeric constant {value}")


class OntologyMappingClient:
    """Retrieve bounded OxO mappings as review candidates, never equivalences."""

    endpoint = "https://www.ebi.ac.uk/spot/oxo/api/search"

    def __init__(self, client_factory=httpx.AsyncClient):
        self._client_factory = client_factory

    @staticmethod
    def validate(identifier: str, target_prefixes: list[str] | None, distance: int, limit: int) -> tuple[str, list[str], int, int]:
        if not isinstance(identifier, str) or not _DISEASE_CURIE.fullmatch(identifier.strip()):
            raise ValueError("source identifier must be an exact EFO or MONDO disease CURIE")
        if isinstance(distance, bool) or distance != 1:
            raise ValueError("disease cross-mapping review permits direct OxO mappings only (distance=1)")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("ontology mapping result limit must be 1..50")
        prefixes = target_prefixes or (["MONDO"] if identifier.startswith("EFO_") else ["EFO"])
        if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 10:
            raise ValueError("target_prefixes must contain 1..10 ontology prefixes")
        normalized = []
        for prefix in prefixes:
            if not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix):
                raise ValueError("target ontology prefixes must be uppercase stable prefixes")
            canonical = prefix.upper()
            if canonical not in normalized:
                normalized.append(canonical)
        return identifier.strip(), normalized, distance, limit

    async def cross_map(self, identifier: str, target_prefixes: list[str] | None = None, distance: int = 1, limit: int = 20) -> dict:
        identifier, prefixes, distance, limit = self.validate(identifier, target_prefixes, distance, limit)
        params = {"ids": identifier, "distance": distance, "size": limit}
        async with self._client_factory(timeout=25, follow_redirects=False) as client:
            async with client.stream("GET", self.endpoint, params=params, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research"}) as response:
                body = await read_bounded_response(response, _MAX_BYTES, "EBI OxO")
        try:
            payload = json.loads(body, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("EBI OxO returned invalid JSON") from error
        except ValueError as error:
            raise ValueError(f"EBI OxO returned invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("EBI OxO returned an unsupported response")
        embedded = payload.get("_embedded")
        results = embedded.get("searchResults") if isinstance(embedded, dict) else None
        if not isinstance(results, list):
            raise ValueError("EBI OxO returned an unsupported mapping response shape")

        mappings_by_key = {}
        query_label = None
        truncated = False
        duplicate_mapping_rows = 0
        if len(results) > 1:
            raise ValueError("EBI OxO returned more than one result for the single requested identifier")
        for result in results:
            if not isinstance(result, dict) or result.get("curie") != identifier:
                raise ValueError("EBI OxO result identity does not match the requested source identifier")
            query_source = result.get("querySource")
            if isinstance(query_source, dict) and isinstance(query_source.get("label"), str):
                query_label = query_source["label"][:256]
            rows = result.get("mappingResponseList", [])
            if not isinstance(rows, list):
                raise ValueError("EBI OxO mapping response list is malformed")
            truncated = len(rows) > limit
            for row in rows[:limit]:
                if not isinstance(row, dict):
                    raise ValueError("EBI OxO mapping response contains a malformed record")
                curie = row.get("curie")
                prefix = row.get("targetPrefix")
                row_distance = row.get("distance")
                if (not isinstance(curie, str) or len(curie) > 128
                        or not isinstance(prefix, str) or not _PREFIX.fullmatch(prefix)
                        or isinstance(row_distance, bool) or not isinstance(row_distance, int)):
                    raise ValueError("EBI OxO mapping record has malformed identifier, prefix, or distance")
                curie_prefix = re.match(r"^([A-Z][A-Z0-9_-]*)(?:_|:)", curie)
                if curie_prefix is None or curie_prefix.group(1) != prefix:
                    raise ValueError("EBI OxO mapping CURIE does not agree with its declared target prefix")
                if prefix not in prefixes or row_distance != distance:
                    continue
                source_type = row.get("sourceType")
                label = row.get("label")
                if label is not None and not isinstance(label, str):
                    raise ValueError("EBI OxO mapping label is malformed")
                if source_type is not None and not isinstance(source_type, str):
                    raise ValueError("EBI OxO mapping source type is malformed")
                mapping = {
                    "target_curie": curie,
                    "target_prefix": prefix,
                    "target_label": label[:500] if isinstance(label, str) else None,
                    "distance": row_distance,
                    "source_type": source_type[:200] if isinstance(source_type, str) else None,
                }
                mapping_key = (prefix, curie, row_distance, mapping["source_type"])
                if mapping_key in mappings_by_key:
                    duplicate_mapping_rows += 1
                mappings_by_key[mapping_key] = mapping
        mappings = sorted(mappings_by_key.values(), key=lambda item: (item["target_prefix"], item["target_curie"], item["distance"], item["source_type"] or ""))
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return {
            "source": "EMBL-EBI Ontology Xref Service (OxO)",
            "source_url": self.endpoint,
            "source_response_sha256": hashlib.sha256(body).hexdigest(),
            "retrieved_at": retrieved_at,
            "source_identifier": identifier,
            "source_label": query_label,
            "target_prefixes": prefixes,
            "maximum_distance": distance,
            "mappings": mappings,
            "returned_count": len(mappings),
            "provider_mapping_count": len(rows) if results else 0,
            "truncated": truncated,
            "duplicate_mapping_rows": duplicate_mapping_rows,
            "mapping_interpretation": "OxO cross-references are source-provided mapping candidates. OxO does not validate mapping evidence; this result does not establish identity, disease equivalence, or gene-disease association and requires human review.",
        }
