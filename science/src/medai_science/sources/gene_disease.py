"""Source-linked gene–disease evidence retrieval and conservative reconciliation."""
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query

MAX_MONARCH_RESPONSE_BYTES = 4 * 1024 * 1024
_MONARCH_GENE_DISEASE_CATEGORIES = {
    "biolink:CausalGeneToDiseaseAssociation",
    "biolink:CorrelatedGeneToDiseaseAssociation",
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"gene-disease source returned duplicate JSON key {key!r}")
        value[key] = item
    return value


def _load_json_response(raw: bytes, source: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"{source} returned non-standard JSON numeric constant {value}")

    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{source} returned invalid JSON") from error


class GeneDiseaseClient:
    """Adapters for Open Targets, GWAS Catalog, and Monarch Initiative."""

    opentargets_endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
    gwas_endpoint = "https://www.ebi.ac.uk/gwas/rest/api/v2/associations"
    monarch_endpoint = "https://api.monarchinitiative.org/v3/api/association"
    _disease_id = re.compile(r"^(?:EFO|MONDO|Orphanet|DOID|HP|OTAR)(?:_|:)?[A-Za-z0-9_.-]{2,40}$")

    def __init__(self, client_factory=httpx.AsyncClient):
        self._client_factory = client_factory

    @classmethod
    def validate_disease_id(cls, disease_id: str) -> str:
        value = ensure_public_research_query(disease_id.strip())
        if not cls._disease_id.fullmatch(value):
            raise ValueError("disease_id must be a stable ontology identifier (for example EFO_0000249 or MONDO_0004979)")
        return value

    @staticmethod
    def validate_limit(limit: int) -> None:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")

    async def opentargets(self, disease_id: str, limit: int = 50) -> dict:
        disease_id = self.validate_disease_id(disease_id)
        self.validate_limit(limit)
        query = """query DiseaseTargets($id: String!, $size: Int!) {
          disease(efoId: $id) {
            id name
            associatedTargets(page: {index: 0, size: $size}) {
              count rows { score target { id approvedSymbol } }
            }
          }
        }"""
        async with self._client_factory(timeout=30, follow_redirects=False) as client:
            async with client.stream("POST", self.opentargets_endpoint, json={"query": query, "variables": {"id": disease_id, "size": limit}}) as response:
                raw = await read_bounded_response(response, MAX_MONARCH_RESPONSE_BYTES, "Open Targets")
        payload = _load_json_response(raw, "Open Targets")
        if not isinstance(payload, dict):
            raise ValueError("Open Targets returned an unsupported response shape")
        if payload.get("errors"):
            raise ValueError("Open Targets returned GraphQL errors")
        disease = (payload.get("data") or {}).get("disease")
        if not isinstance(disease, dict) or disease.get("id") != disease_id:
            raise ValueError("Open Targets response disease identity does not exactly match the requested identifier")
        associated = disease.get("associatedTargets")
        if not isinstance(associated, dict):
            raise ValueError("Open Targets returned an unsupported associated-target response shape")
        rows = associated.get("rows")
        reported_count = associated.get("count")
        if (not isinstance(rows, list) or len(rows) > limit
                or isinstance(reported_count, bool) or not isinstance(reported_count, int) or reported_count < len(rows)):
            raise ValueError("Open Targets associated-target count or bounded rows are malformed")
        records = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("target"), dict):
                raise ValueError("Open Targets returned a malformed target association row")
            target = row["target"]
            gene_id = target.get("id")
            if not isinstance(gene_id, str) or not re.fullmatch(r"ENSG[0-9]{11}", gene_id):
                raise ValueError("Open Targets target row is missing an exact Ensembl gene identifier")
            score = row.get("score")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not 0 <= score <= 1 or isinstance(score, float) and not math.isfinite(score)):
                raise ValueError("Open Targets association score must be finite and within 0..1")
            symbol = target.get("approvedSymbol")
            if symbol is not None and (not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 128):
                raise ValueError("Open Targets approved gene symbol is malformed")
            records.append({"gene_id": gene_id, "symbol": symbol, "score": score, "evidence_type": "integrated_target_disease_association"})
        return {
            "source": "Open Targets Platform", "source_url": self.opentargets_endpoint,
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "disease_id": disease_id,
            "disease_name": disease.get("name") if isinstance(disease.get("name"), str) else None,
            "reported_count": reported_count, "returned_count": len(records),
            "truncated": reported_count > len(records), "records": records,
            "response_sha256": hashlib.sha256(raw).hexdigest(), "raw_response": payload,
            "interpretation_limit": "Open Targets association scores are integrated prioritisation scores, not probabilities or causal conclusions; this bounded result may be truncated and preserves provider identity and native scores.",
        }

    async def gwas_catalog(self, disease_id: str, limit: int = 50) -> dict:
        """Retrieve directly mapped GWAS Catalog traits for an EFO identifier."""
        disease_id = self.validate_disease_id(disease_id)
        if not disease_id.startswith("EFO_"):
            raise ValueError("GWAS Catalog trait lookup requires an EFO_ identifier")
        self.validate_limit(limit)
        async with self._client_factory(timeout=30, follow_redirects=False) as client:
            async with client.stream("GET", self.gwas_endpoint, params={"efo_trait": disease_id, "show_child_traits": "false", "page": 0, "size": limit}) as response:
                raw = await read_bounded_response(response, MAX_MONARCH_RESPONSE_BYTES, "GWAS Catalog")
        payload = _load_json_response(raw, "GWAS Catalog")
        if not isinstance(payload, dict):
            raise ValueError("GWAS Catalog returned an unsupported association response shape")
        embedded = payload.get("_embedded")
        if embedded is None:
            if "associations" not in payload:
                raise ValueError("GWAS Catalog response is missing its associations collection")
            entries = payload["associations"]
        elif isinstance(embedded, dict) and "associations" in embedded:
            entries = embedded["associations"]
        else:
            raise ValueError("GWAS Catalog response has a malformed _embedded.associations collection")
        if not isinstance(entries, list):
            raise ValueError("GWAS Catalog response has an unsupported association shape")
        records = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("GWAS Catalog returned a malformed association record")
            genes = entry.get("mappedGenes", entry.get("mapped_genes", []))
            if genes is None:
                genes = []
            if isinstance(genes, str):
                genes = [item.strip() for item in genes.split(",") if item.strip()]
            if isinstance(genes, dict):
                genes = [genes]
            if not isinstance(genes, list):
                raise ValueError("GWAS Catalog mapped-gene field has an unsupported shape")
            for gene in genes:
                if isinstance(gene, str):
                    gene_value = gene.strip()
                    gene_id = gene_value if re.fullmatch(r"ENSG[0-9]{11}", gene_value) else None
                    symbol = None if gene_id else gene_value
                elif isinstance(gene, dict):
                    gene_id = gene.get("ensemblId") or gene.get("ensembl_id")
                    symbol = gene.get("geneName") or gene.get("gene_name") or gene.get("symbol")
                else:
                    raise ValueError("GWAS Catalog returned a malformed mapped-gene record")
                # GWAS mapped names can be nearest-gene annotations. Keep them
                # available for later resolution, but never treat them as IDs.
                stable_gene_id = str(gene_id).strip() if isinstance(gene_id, str) and re.fullmatch(r"ENSG[0-9]{11}", gene_id.strip()) else None
                mapping_candidate = str(symbol or gene_id or "").strip() or None
                if stable_gene_id or mapping_candidate:
                    records.append({"gene_id": stable_gene_id, "gene_mapping_candidate": mapping_candidate if not stable_gene_id else None, "symbol": symbol, "association_id": entry.get("associationId") or entry.get("association_id"), "study_id": entry.get("studyId") or entry.get("study_id"), "pubmed_id": entry.get("pubmedId") or entry.get("pubmed_id"), "reported_trait": entry.get("reportedTrait") or entry.get("reported_trait"), "evidence_type": "curated_gwas_variant_mapped_gene"})
        return {"source": "NHGRI-EBI GWAS Catalog", "source_url": self.gwas_endpoint, "retrieved_at": datetime.now(timezone.utc).isoformat(), "disease_id": disease_id, "trait_scope": "direct EFO trait only; child traits excluded", "mapped_gene_policy": "GWAS Catalog default mapped genes; nearest genes may be included by source annotation", "records": records, "raw_response": payload}

    async def monarch_gene_disease(self, disease_id: str, limit: int = 50) -> dict:
        """Retrieve direct human gene–disease assertions with Monarch's native provenance."""
        disease_id = self.validate_disease_id(disease_id)
        self.validate_limit(limit)
        monarch_disease_id = disease_id.replace("_", ":", 1) if "_" in disease_id else disease_id
        params = [
            ("category", category) for category in sorted(_MONARCH_GENE_DISEASE_CATEGORIES)
        ]
        params.extend([
            ("subject_category", "biolink:Gene"),
            ("subject_taxon", "NCBITaxon:9606"),
            ("object", monarch_disease_id),
            ("object_category", "biolink:Disease"),
            ("direct", "true"),
            ("compact", "false"),
            ("format", "json"),
            ("limit", str(limit)),
            ("offset", "0"),
        ])
        async with self._client_factory(timeout=30, follow_redirects=False) as client:
            async with client.stream("GET", self.monarch_endpoint, params=params) as response:
                raw = await read_bounded_response(response, MAX_MONARCH_RESPONSE_BYTES, "Monarch association")
        payload = _load_json_response(raw, "Monarch association")
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("Monarch returned an unsupported association response shape")
        total = payload.get("total")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise ValueError("Monarch association response has an invalid total")
        records = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                continue
            if (
                item.get("object") != monarch_disease_id
                or item.get("subject_category") != "biolink:Gene"
                or item.get("object_category") != "biolink:Disease"
                or item.get("subject_taxon") != "NCBITaxon:9606"
                or item.get("category") not in _MONARCH_GENE_DISEASE_CATEGORIES
            ):
                continue
            gene_id = item.get("subject")
            if not isinstance(gene_id, str) or not re.fullmatch(r"(?:HGNC:[0-9]+|NCBIGene:[0-9]+|ENSG[0-9]{11})", gene_id):
                continue
            records.append({
                "gene_id": gene_id,
                "symbol": item.get("subject_label"),
                "association_id": item.get("id"),
                "association_category": item.get("category"),
                "predicate": item.get("predicate"),
                "knowledge_level": item.get("knowledge_level"),
                "agent_type": item.get("agent_type"),
                "primary_knowledge_source": item.get("primary_knowledge_source"),
                "aggregator_knowledge_source": item.get("aggregator_knowledge_source"),
                "publications": item.get("publications"),
                "has_evidence": item.get("has_evidence"),
                "evidence_count": item.get("evidence_count"),
                "negated": item.get("negated"),
                "evidence_type": "monarch_knowledge_graph_gene_disease_assertion",
            })
        return {
            "source": "Monarch Initiative Knowledge Graph",
            "source_url": self.monarch_endpoint,
            "api_version": "v3",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "disease_id": disease_id,
            "monarch_disease_id": monarch_disease_id,
            "taxon": "NCBITaxon:9606",
            "association_categories": sorted(_MONARCH_GENE_DISEASE_CATEGORIES),
            "direct_only": True,
            "reported_count": total,
            "truncated": total > len(payload["items"]),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "records": records,
            "raw_response": payload,
            "interpretation_limit": "Monarch is an aggregated knowledge graph; retain primary and aggregator sources per assertion. Retrieved association records are not independent-study counts, pathogenicity classifications, or clinical recommendations.",
        }


def reconcile_gene_disease_sources(disease_id: str, sources: list[dict]) -> dict:
    """Join exact stable gene IDs while preserving each source score and meaning."""
    groups: dict[str, dict] = {}
    for source_result in sources:
        source = str(source_result.get("source", "unknown"))
        for record in source_result.get("records", []):
            gene_id = str(record.get("gene_id", "")).strip()
            if not re.fullmatch(r"(?:ENSG[0-9]{11}|(?:HGNC|NCBIGene):[0-9]+)", gene_id):
                continue
            item = groups.setdefault(gene_id, {"gene_id": gene_id, "symbols": set(), "source_evidence": []})
            if record.get("symbol"):
                item["symbols"].add(str(record["symbol"]))
            item["source_evidence"].append({"source": source, **{key: value for key, value in record.items() if key not in {"gene_id", "symbol"}}})
    genes = []
    for gene_id, item in sorted(groups.items()):
        evidence = sorted(item["source_evidence"], key=lambda row: (row["source"], str(row.get("association_id", "")), str(row.get("study_id", ""))))
        genes.append({"gene_id": gene_id, "symbols": sorted(item["symbols"]), "distinct_source_count": len({row["source"] for row in evidence}), "source_evidence": evidence})
    return {"disease_id": disease_id, "genes": genes, "source_count": len({result.get("source") for result in sources}), "reconciliation_policy": "exact stable identifier join only; source scores and assertion types are incomparable and are never summed or presented as calibrated confidence", "interpretation_limit": "GWAS mapped genes can be positional. Open Targets scores are integrated prioritisation scores, not probabilities. Monarch is an aggregator and its records must be interpreted with their primary and aggregator source fields; distinct provider/source counts do not measure independent studies or establish causality."}
