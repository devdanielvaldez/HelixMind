"""Bounded Orphadata rare-disease gene associations by exact ORPHAcode."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import httpx

from medai_science.sources.bounded_http import read_bounded_response


_ORPHACODE = re.compile(r"^[0-9]{1,6}$")
_MAX_BYTES = 4_000_000
_MAX_ASSOCIATIONS = 500


class OrphadataClient:
    """Retrieve Orphanet's typed gene-disease assertions without score aggregation."""

    endpoint = "https://api.orphadata.com/rd-associated-genes/orphacodes"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate_orphacode(orphacode: str | int) -> str:
        if isinstance(orphacode, bool) or not _ORPHACODE.fullmatch(str(orphacode)) or int(orphacode) < 1:
            raise ValueError("Orphadata lookup requires an exact positive numeric ORPHAcode")
        return str(int(orphacode))

    async def disease_genes(self, orphacode: str | int) -> dict[str, Any]:
        code = self.validate_orphacode(orphacode)
        endpoint = f"{self.endpoint}/{code}"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", endpoint, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research"}) as response:
                body = await read_bounded_response(response, _MAX_BYTES, "Orphadata")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Orphadata returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Orphadata returned an unsupported response")
        data = payload.get("data")
        results = data.get("results") if isinstance(data, dict) else None
        rows = results if isinstance(results, list) else [results] if isinstance(results, dict) else []
        matching = [row for row in rows if isinstance(row, dict) and str(row.get("ORPHAcode", "")) == code]
        if len(matching) > 1:
            raise ValueError("Orphadata returned duplicate records for the exact ORPHAcode")
        if not matching:
            return self._receipt(code, body, None, [])

        disease = matching[0]
        raw_associations = disease.get("DisorderGeneAssociation", [])
        if isinstance(raw_associations, dict):
            raw_associations = [raw_associations]
        if not isinstance(raw_associations, list) or len(raw_associations) > _MAX_ASSOCIATIONS:
            raise ValueError("Orphadata gene association list is malformed or exceeds the retention limit")
        records = []
        for association in raw_associations:
            if not isinstance(association, dict):
                continue
            gene = association.get("Gene")
            if not isinstance(gene, dict):
                continue
            references = gene.get("ExternalReference", [])
            if isinstance(references, dict):
                references = [references]
            if not isinstance(references, list):
                references = []
            normalized_references = [
                {"source": ref.get("Source"), "reference": ref.get("Reference")}
                for ref in references[:100]
                if isinstance(ref, dict)
                and isinstance(ref.get("Source"), str)
                and isinstance(ref.get("Reference"), str)
            ]
            record = {
                "gene_symbol": gene.get("Symbol") if isinstance(gene.get("Symbol"), str) else None,
                "gene_name": gene.get("name") if isinstance(gene.get("name"), str) else None,
                "gene_type": gene.get("GeneType") if isinstance(gene.get("GeneType"), str) else None,
                "gene_synonyms": [value[:200] for value in gene.get("Synonym", [])[:50] if isinstance(value, str)] if isinstance(gene.get("Synonym"), list) else [],
                "gene_loci": [item.get("GeneLocus") for item in gene.get("Locus", [])[:20] if isinstance(item, dict) and isinstance(item.get("GeneLocus"), str)] if isinstance(gene.get("Locus"), list) else [],
                "gene_external_references": normalized_references,
                "association_type": association.get("DisorderGeneAssociationType") if isinstance(association.get("DisorderGeneAssociationType"), str) else None,
                "association_status": association.get("DisorderGeneAssociationStatus") if isinstance(association.get("DisorderGeneAssociationStatus"), str) else None,
                "source_of_validation": association.get("SourceOfValidation") if isinstance(association.get("SourceOfValidation"), str) else None,
                "record_sha256": hashlib.sha256(json.dumps(association, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            }
            records.append(record)
        return self._receipt(code, body, disease, records)

    @staticmethod
    def _receipt(code: str, body: bytes, disease: dict[str, Any] | None, records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "source": "Orphadata / Orphanet",
            "source_url": f"https://api.orphadata.com/rd-associated-genes/orphacodes/{code}",
            "source_response_sha256": hashlib.sha256(body).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "attribution": "Orphadata, CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "dataset_category": "Rare diseases and associated genes",
            "dataset_language": "en",
            "dataset_release_date": disease.get("Date") if isinstance(disease, dict) and isinstance(disease.get("Date"), str) else None,
            "disease": {"orphacode": code, "preferred_term": disease.get("Preferred term"), "typology": disease.get("Typology"), "group": disease.get("DisorderGroup"), "url": disease.get("OrphanetURL")} if disease else None,
            "records": records,
            "record_count": len(records),
            "found": disease is not None,
            "interpretation_limit": "Orphadata reports gene-disease relationship categories and its own association status. These source assertions are not aggregated into probabilities, are not equivalent to another provider's classifications, and do not constitute clinical interpretation.",
        }
