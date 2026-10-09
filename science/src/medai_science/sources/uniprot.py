"""UniProtKB REST adapter with accession, sequence, and retrieval provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.parse import quote
import httpx
from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query
from medai_science.json_input import strict_json_loads

_MAX_ENTRY_BYTES = 8 * 1024 * 1024
_ACCESSION = re.compile(r"^[A-Z0-9]{6,10}(?:-[1-9][0-9]*)?$", re.IGNORECASE)


def _entry_features(record: dict) -> tuple[list[dict], list[dict]]:
    """Deterministically extract annotated regions and PDB cross references."""
    features = record.get("features", [])
    domains = []
    for feature in features if isinstance(features, list) else []:
        if not isinstance(feature, dict) or feature.get("type", "").casefold() not in {"domain", "repeat", "region", "motif"}:
            continue
        location = feature.get("location", {})
        domains.append({
            "feature_id": feature.get("featureId"),
            "type": feature.get("type"),
            "description": feature.get("description"),
            "start": (location.get("start") or {}).get("value") if isinstance(location, dict) else None,
            "end": (location.get("end") or {}).get("value") if isinstance(location, dict) else None,
            "sequence": feature.get("sequence"),
        })
    structures = []
    for reference in record.get("uniProtKBCrossReferences", []):
        if not isinstance(reference, dict) or reference.get("database") != "PDB":
            continue
        properties = {item.get("key"): item.get("value") for item in reference.get("properties", []) if isinstance(item, dict)}
        structures.append({"pdb_id": reference.get("id"), "method": properties.get("Method"), "resolution": properties.get("Resolution"), "chains": properties.get("Chains")})
    return domains, structures


class UniProtClient:
    endpoint = "https://rest.uniprot.org/uniprotkb/search"
    fields = "accession,id,protein_name,gene_names,organism_name,length,sequence,cc_function,go_id"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate_accession(accession: str) -> str:
        if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
            raise ValueError("provide a UniProt accession (optionally followed by an isoform suffix)")
        return accession.upper()

    async def get_entry(self, accession: str) -> dict:
        """Fetch a complete UniProtKB JSON record and expose sourced features/cross references."""
        accession = self.validate_accession(accession)
        endpoint = f"https://rest.uniprot.org/uniprotkb/{quote(accession, safe='-')}.json"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", endpoint, headers={"Accept": "application/json"}) as response:
                if response.status_code == 404:
                    raise ValueError(f"UniProtKB accession not found: {accession}")
                body = await read_bounded_response(response, _MAX_ENTRY_BYTES, "UniProtKB")
        try:
            record = strict_json_loads(body)
        except ValueError as error:
            raise ValueError(f"UniProtKB returned invalid strict JSON: {error}") from error
        if not isinstance(record, dict) or record.get("primaryAccession", "").upper() != accession.split("-")[0]:
            raise ValueError("UniProtKB response accession does not match the request")

        domains, structures = _entry_features(record)
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        digest = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "source": "UniProtKB", "accession": accession, "entry_url": f"https://www.uniprot.org/uniprotkb/{accession}",
            "retrieved_at": retrieved_at, "license": "CC BY 4.0", "record_sha256": digest,
            "domains_and_regions": domains, "experimental_structure_cross_references": structures,
            "structure_cross_reference_limit": "Provider cross-references are not independently validated here; structure interpretation requires reviewing the linked PDB entry and mapping.",
            "record": record,
        }

    async def search(self, query: str, limit: int = 10, organism_id: int = 9606) -> list[dict]:
        query = ensure_public_research_query(query)
        if len(query) > 256:
            raise ValueError("protein query cannot exceed 256 characters")
        if not 1 <= limit <= 20:
            raise ValueError("protein result limit must be 1..20")
        if not 1 <= organism_id <= 9_999_999:
            raise ValueError("organism_id must be a positive NCBI taxonomy identifier")

        # Keep the query as a value passed through HTTPX params; encoding is not
        # performed manually so the API receives the original query expression.
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream(
                "GET", self.endpoint,
                params={"query": f"({query}) AND (organism_id:{organism_id})", "format": "json", "size": limit, "fields": self.fields},
                headers={"Accept": "application/json"},
            ) as response:
                raw = await read_bounded_response(response, _MAX_ENTRY_BYTES, "UniProtKB search")
        try:
            payload = strict_json_loads(raw)
        except ValueError as error:
            raise ValueError(f"UniProtKB returned invalid strict search JSON: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or len(payload["results"]) > limit:
            raise ValueError("UniProtKB returned a malformed or over-limit search result collection")
        records = payload["results"]
        if any(not isinstance(record, dict) or not isinstance(record.get("primaryAccession"), str)
               or not _ACCESSION.fullmatch(record["primaryAccession"]) for record in records):
            raise ValueError("UniProtKB returned a malformed search record or missing accession")

        retrieved_at = datetime.now(timezone.utc).isoformat()
        return [
            {
                "source": "UniProtKB",
                "source_id": str(record["primaryAccession"]),
                "query": query,
                "organism_id": organism_id,
                "retrieved_at": retrieved_at,
                "license": "CC BY 4.0",
                "record": record,
            }
            for record in records
        ]
