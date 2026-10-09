"""Leased research worker that routes operations through an authenticated MCP endpoint."""
from __future__ import annotations

import hashlib
import json
import os
import asyncio
import math
import re
import unicodedata
import time
from urllib.parse import urlsplit
from typing import Any
from uuid import uuid4
from medai_science.sources.safety import ensure_public_research_query
from medai_science.literature import exact_mygene_identifier_matches, exact_uniprot_gene_matches, omit_pubmed_abstract_text, pubchem_cid_candidates_from_extraction_receipts, resolved_gene_symbols, source_entity_node, structure_pmc_receipt, structure_pubmed_abstract
from medai_science.research_planning import objective_plan_from_artifacts, objective_plan_record_from_artifacts, paper_limit_for_depth, RESEARCH_TASK_IDS
from medai_science.quality import is_valid_quality_annotation, summarize_pubmed_design_metadata
from medai_science.reproducibility import build_reproducibility_bundle
from medai_science.research_workflows import build_research_workflow_proposal
from medai_science.claim_alignment import build_claim_alignment_candidates, semantic_claim_pair_marker
from medai_science.contradiction_search import build_contradiction_query, select_contradiction_claims
from medai_science.cancellation import ResearchSessionCancelled
from medai_science.scientific_agents import AGENT_BY_TASK, AGENTS, ResearchDirector, agent_may_call
from medai_science.sources.gene_disease import GeneDiseaseClient
from medai_science.sources.pubmed_search import validate_pmid_snapshot
from medai_science.hypothesis_literature import build_hypothesis_literature_query, build_hypothesis_literature_receipt, bounded_paper_record
from medai_science.genomics_research import validate_local_vcf_summary
from medai_science.benchmark_artifacts import DOMAIN_BENCHMARK_ARTIFACT_KIND, DOMAIN_BENCHMARK_ARTIFACT_SOURCE, EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND, EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE

MAX_SEMANTIC_CLAIM_PAIRS_PER_TASK = 10
MCP_TRANSPORT_HEADER = "X-MedAI-MCP-Token"


def _worker_mcp_connection() -> tuple[str, dict[str, str]]:
    """Resolve the optional direct MedAI MCP route or the legacy AgentMesh route."""
    direct_url = os.getenv("MEDAI_DIRECT_MCP_URL", "").strip()
    if not direct_url:
        return os.getenv("MEDAI_AGENTMESH_URL", "http://127.0.0.1:8080/mcp"), {}

    parsed = urlsplit(direct_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("MEDAI_DIRECT_MCP_URL must be an http(s) endpoint without credentials, query, or fragment")
    token = os.getenv("MEDAI_MCP_BEARER_TOKEN", "")
    api_tokens = {
        value for value in (
            os.getenv("MEDAI_API_BEARER_TOKEN", ""),
            os.getenv("MEDAI_SPECIALIST_API_TOKEN", ""),
        ) if value
    }
    if len(token) < 32 or len(token) > 4096 or any(char.isspace() for char in token):
        raise ValueError("direct MCP transport requires MEDAI_MCP_BEARER_TOKEN with 32..4096 non-whitespace characters")
    if token in api_tokens:
        raise ValueError("MEDAI_MCP_BEARER_TOKEN must be distinct from API bearer credentials")
    return direct_url, {MCP_TRANSPORT_HEADER: token}


def _semantic_claim_screening_batch(
    links: list[dict[str, Any]], completed_source_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return the next bounded uncompleted batch and explicit remaining count."""
    if not isinstance(links, list):
        raise ValueError("semantic claim candidates must be a list")
    completed = completed_source_ids or set()
    pending = [
        link for link in links
        if semantic_claim_pair_marker(*link["sources"]) not in completed
    ]
    selected = pending[:MAX_SEMANTIC_CLAIM_PAIRS_PER_TASK]
    return selected, len(pending) - len(selected)


def _hypothesis_proposal_needs_review(proposal: dict[str, Any], literature: dict[str, Any]) -> bool:
    """Treat every omitted or unreported proposal/search bound as incomplete."""
    if not isinstance(proposal, dict) or not isinstance(literature, dict):
        return True
    omitted = literature.get("omitted_candidate_count")
    warnings = literature.get("warnings")
    return (
        not isinstance(proposal.get("artifact_id"), str)
        or not proposal["artifact_id"]
        or proposal.get("truncated") is not False
        or not isinstance(literature.get("artifact_id"), str)
        or not literature["artifact_id"]
        or isinstance(omitted, bool)
        or not isinstance(omitted, int)
        or omitted != 0
        or not isinstance(warnings, list)
        or bool(warnings)
    )


def _hypothesis_criticism_needs_review(critique: dict[str, Any], unlinked_candidate_count: int) -> bool:
    """Surface incomplete graph-route or retrieved-candidate criticism."""
    reviews = critique.get("reviews") if isinstance(critique, dict) else None
    return (
        not isinstance(critique, dict)
        or not isinstance(critique.get("artifact_id"), str)
        or not critique["artifact_id"]
        or unlinked_candidate_count > 0
        or not isinstance(reviews, list)
        or any(not isinstance(row, dict) or row.get("alternative_graph_routes_truncated") is not False for row in reviews)
    )


def _tool_value(result: Any) -> Any:
    if getattr(result, "isError", False):
        detail = "; ".join(getattr(item, "text", "") for item in getattr(result, "content", []))
        if "research session cancellation requested" in detail.lower():
            raise ResearchSessionCancelled(detail or "research session cancellation requested")
        raise RuntimeError(detail or "AgentMesh MCP tool returned an error")
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    blocks = [getattr(item, "text", None) for item in getattr(result, "content", [])]
    blocks = [item for item in blocks if item]
    if len(blocks) == 1:
        try:
            return json.loads(blocks[0])
        except json.JSONDecodeError:
            return blocks[0]
    if blocks:
        return [json.loads(item) for item in blocks]
    return None


def _is_budget_exhaustion_error(error: BaseException | str) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "specialist credit budget exhausted",
            "research session execution budget exhausted",
            "402 payment required",
        )
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()


def _literature_watch_pmids(value: Any) -> list[str]:
    """Normalize the API's JSON-encoded PMID snapshot before calling the MCP tool."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("stored literature watch PMID snapshot is invalid JSON") from error
    return validate_pmid_snapshot(value)


def _specialist_meta(agent_id: str | None, token: str | None) -> dict[str, str] | None:
    """Carry the specialist API credential as opaque MCP metadata, never as tool input."""
    if agent_id is None:
        return None
    if not isinstance(token, str) or len(token) < 32 or len(token) > 4096 or any(char.isspace() for char in token):
        raise ValueError("specialist workers require MEDAI_SPECIALIST_API_TOKEN with at least 32 non-whitespace bytes")
    return {
        "io.medai/specialist-api-bearer": token,
        "io.medai/scientific-agent": agent_id,
    }


class ArtifactCollectionLimitError(RuntimeError):
    """Raised when bounded worker artifact retrieval cannot return a complete set."""


def _record_control_plane_update_failure(
    summary: dict[str, Any], task_id: str, task_error: BaseException, update_error: BaseException,
) -> None:
    """Report uncertain durable task state without copying untrusted error text."""
    summary.setdefault("control_plane_update_failures", []).append({
        "task_id": task_id,
        "original_error_type": type(task_error).__name__,
        "update_error_type": type(update_error).__name__,
        "state": "unknown",
    })


def _source_identified_compound_artifacts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept compound identity receipts only from their registered chemistry stage."""
    return [
        item for item in items
        if item.get("kind") == "compound_chemical_identity"
        and item.get("task_id") == "compounds.analyze"
        and item.get("source") == "PubChem PUG REST"
    ]


def _biomedical_domain_benchmark_inputs(items: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """Select only researcher-designated domain benchmark artifacts from this session."""
    return [
        row for row in items
        if isinstance(row, dict)
        and row.get("session_id") == session_id
        and row.get("task_id") == "benchmark.evaluate"
        and row.get("kind") == DOMAIN_BENCHMARK_ARTIFACT_KIND
        and row.get("source") == DOMAIN_BENCHMARK_ARTIFACT_SOURCE
    ]


def _evidence_quality_benchmark_inputs(items: list[dict[str, Any]], session_id: str) -> list[dict[str, Any]]:
    """Select only researcher-designated double-reviewed ordinal quality benchmarks."""
    return [
        row for row in items
        if isinstance(row, dict)
        and row.get("session_id") == session_id
        and row.get("task_id") == "benchmark.evaluate"
        and row.get("kind") == EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND
        and row.get("source") == EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE
    ]


def _exact_pubchem_identity(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Validate the exact requested CID and a single standard chemical identity match."""
    payload = item.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(payload, dict) or payload.get("source") != "PubChem PUG REST":
        return None
    source_id = item.get("source_id")
    if not isinstance(source_id, str) or not source_id.isdecimal() or payload.get("query") != source_id:
        return None
    matches = payload.get("matches")
    if (not isinstance(matches, list) or len(matches) != 1
            or payload.get("ambiguous") is True or payload.get("too_many_matches") is True):
        return None
    match = matches[0]
    if not isinstance(match, dict) or isinstance(match.get("cid"), bool) or match.get("cid") != int(source_id):
        return None
    inchikey = match.get("inchikey")
    if not isinstance(inchikey, str) or not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", inchikey):
        return None
    return payload, match


def _pmc_extraction_receipts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept paper extraction inputs only from the PMC BioC extraction stage."""
    return [
        item for item in items
        if item.get("kind") == "paper_extraction_receipt"
        and item.get("task_id") == "literature.extract"
        and item.get("source") == "PMC BioC"
    ]


def _session_research_budget(session: Any) -> int:
    """Read a server-provided session budget without coercing malformed values."""
    budget = session.get("budget") if isinstance(session, dict) else None
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 100_000:
        raise ValueError("research session returned an invalid integer credit budget")
    return budget


async def _retrieve_source_identified_compounds(
    call: Any,
    session_id: str,
    task_id: str,
    worker_id: str,
    extraction_receipts: list[dict[str, Any]],
    existing_artifacts: list[dict[str, Any]],
    limit: int,
    cancellation: asyncio.Event,
    completed_source_ids: set[str],
    artifact_ids: list[str],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Resolve bounded BioC-provided PubChem CIDs and checkpoint retained identity artifacts."""
    extraction_receipts = _pmc_extraction_receipts(extraction_receipts)
    existing_artifacts = _source_identified_compound_artifacts(existing_artifacts)
    known_cids = {
        str(item.get("source_id"))
        for item in existing_artifacts
        if item.get("source") == "PubChem PUG REST"
    }
    chemical_artifacts = list(existing_artifacts[:limit])
    candidates, candidate_limit_reached = pubchem_cid_candidates_from_extraction_receipts(
        extraction_receipts, limit=limit, exclude=known_cids
    )
    warnings = []
    discovered = []
    for candidate in candidates:
        if len(chemical_artifacts) >= limit:
            break
        if cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during compound resolution")
        cid = candidate["cid"]
        try:
            result = await call(
                "medai_compounds_lookup_pubchem",
                {"session_id": session_id, "task_id": task_id, "identifier": cid},
            )
            artifact_id = result.get("artifact_id") if isinstance(result, dict) else None
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("PubChem identity artifact was not retained")
            artifact_ids.append(artifact_id)
            chemical_artifacts.append({
                "id": artifact_id,
                "kind": "compound_chemical_identity",
                "source": "PubChem PUG REST",
                "source_id": cid,
                "payload": result,
            })
            discovered.append({
                "cid": cid,
                "extraction_receipt_id": candidate["source_artifact_id"],
                "chemical_identity_artifact_id": artifact_id,
            })
            known_cids.add(cid)
            completed_source_ids.add(f"pubchem-cid:{cid}")
            checkpoint.update({
                "stage": "source_identified_compounds_resolved",
                "completed_source_ids": sorted(completed_source_ids),
                "artifact_ids": artifact_ids,
            })
            await call("medai_research_save_checkpoint", {
                "session_id": session_id,
                "task_id": task_id,
                "worker_id": worker_id,
                "checkpoint": checkpoint,
            })
        except ResearchSessionCancelled:
            raise
        except Exception as error:
            warnings.append({"cid": cid, "error": type(error).__name__})
    return {
        "chemical_artifacts": chemical_artifacts,
        "discovered": discovered,
        "warnings": warnings,
        "candidate_limit_reached": candidate_limit_reached,
    }


async def _retrieve_gene_disease_curations(
    call: Any,
    session_id: str,
    task_id: str,
    worker_id: str,
    genes: list[dict[str, str]],
    cancellation: asyncio.Event,
    completed_source_ids: set[str],
    artifact_ids: list[str],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Resume exact-symbol ClinGen and GenCC lookups without combining source scales."""
    source_counts = {"ClinGen": 0, "GenCC": 0}
    curation_counts = {"ClinGen": 0, "GenCC": 0}
    graph_edges_added = 0
    graph_warnings: list[dict[str, str]] = []
    warnings = []
    for gene in genes:
        if cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during gene-disease curation")
        symbol = gene["symbol"]
        for source, tool in (("ClinGen", "medai_gene_disease_clingen_gene"), ("GenCC", "medai_gene_disease_gencc_gene")):
            marker = f"{source.lower()}-curation:{gene['entrezgene']}"
            if marker in completed_source_ids:
                continue
            try:
                result = await call(tool, {"session_id": session_id, "task_id": task_id, "gene": symbol, "limit": 50})
                source_counts[source] += 1
                count_key = "total_matching_curations" if source == "ClinGen" else "total_matching_assertions"
                curation_counts[source] += int(result.get(count_key, 0)) if isinstance(result, dict) else 0
                artifact_id = result.get("artifact_id") if isinstance(result, dict) else None
                if isinstance(artifact_id, str) and artifact_id:
                    artifact_ids.append(artifact_id)
                    try:
                        graph_result = await _register_gene_disease_curation_graph(call, session_id, task_id, source, result, artifact_id)
                        graph_edges_added += graph_result["edges_added"]
                        graph_warnings.extend(graph_result["warnings"])
                    except ResearchSessionCancelled:
                        raise
                    except Exception as error:
                        graph_warnings.append({"source": source, "gene_symbol": symbol, "error": type(error).__name__})
                completed_source_ids.add(marker)
                checkpoint.update({"stage": "curated_gene_disease_sources_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
            except ResearchSessionCancelled:
                raise
            except Exception as error:
                warnings.append({"gene_symbol": symbol, "source": source, "error": type(error).__name__})
    return {"source_queries": source_counts, "source_assertions": curation_counts, "artifact_ids": artifact_ids, "graph_edges_added": graph_edges_added, "graph_warnings": graph_warnings, "warnings": warnings}


async def _register_gene_disease_curation_graph(call: Any, session_id: str, task_id: str, source: str, result: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Represent provider rows as distinct provenance-bearing assertion nodes."""
    source_prefix = {"ClinGen": "ClinGenCuration", "GenCC": "GenCCAssertion"}.get(source)
    records_key = {"ClinGen": "curations", "GenCC": "assertions"}.get(source)
    if source_prefix is None or records_key is None:
        raise ValueError("unsupported gene-disease curation source")
    rows = result.get(records_key)
    if not isinstance(rows, list):
        raise ValueError("gene-disease source record collection is malformed")
    edges_added = 0
    warnings: list[dict[str, str]] = []
    for row in rows[:50]:
        if not isinstance(row, dict):
            warnings.append({"source": source, "error": "malformed source row"})
            continue
        gene_id = row.get("gene_hgnc_id") if source == "ClinGen" else row.get("gene_curie")
        disease_id = row.get("disease_mondo_id") if source == "ClinGen" else row.get("disease_curie")
        record_hash = row.get("record_sha256")
        if not isinstance(gene_id, str) or not re.fullmatch(r"HGNC:[0-9]+", gene_id) or not isinstance(disease_id, str) or not re.fullmatch(r"MONDO:[0-9]{7}", disease_id) or not isinstance(record_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", record_hash):
            warnings.append({"source": source, "error": "source row lacks exact HGNC, MONDO, or record hash"})
            continue
        gene_node = f"HGNC:{gene_id.split(':', 1)[1]}"
        disease_node = f"MONDO:{disease_id.split(':', 1)[1]}"
        assertion_node = f"{source_prefix}:{record_hash}"
        classification = row.get("classification") or row.get("classification_title") or "Source assertion"
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": gene_node, "entity_type": "Gene", "label": str(row.get("gene_symbol") or gene_id)[:512], "metadata": {"identifier_namespace": "HGNC", "identifier": gene_id}})
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": disease_node, "entity_type": "Disease", "label": disease_id, "metadata": {"identifier_namespace": "MONDO", "identifier": disease_id}})
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": assertion_node, "entity_type": "GeneDiseaseAssertion", "label": f"{source} · {str(classification)[:400]}", "metadata": {"source": source, "source_record_sha256": record_hash, "source_artifact_id": artifact_id, "classification": str(classification)[:200], "mode_of_inheritance": str(row.get("mode_of_inheritance") or row.get("moi_title") or "")[:200]}})
        for endpoint, relation in ((gene_node, "SOURCE_ASSERTION_NAMES_GENE"), (disease_node, "SOURCE_ASSERTION_NAMES_DISEASE")):
            await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": assertion_node, "relation": relation, "target": endpoint, "evidence_ids": [], "artifact_ids": [artifact_id], "confidence": 1.0})
            edges_added += 1
    return {"edges_added": edges_added, "warnings": warnings}


async def _register_orphadata_gene_graph(call: Any, session_id: str, task_id: str, result: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Register only exact HGNC references from Orphadata as distinct assertions."""
    disease = result.get("disease")
    if not isinstance(disease, dict) or not isinstance(disease.get("orphacode"), str) or not re.fullmatch(r"[1-9][0-9]{0,5}", disease["orphacode"]):
        raise ValueError("Orphadata result lacks an exact ORPHAcode")
    disease_id = f"ORPHA:{disease['orphacode']}"
    rows = result.get("records")
    if not isinstance(rows, list):
        raise ValueError("Orphadata gene association records are malformed")
    disease_node = f"Disease:{disease_id}"
    await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": disease_node, "entity_type": "Disease", "label": str(disease.get("preferred_term") or disease_id)[:512], "metadata": {"identifier_namespace": "ORPHA", "identifier": disease_id, "source_artifact_id": artifact_id}})
    edges_added = 0
    warnings: list[dict[str, str]] = []
    for row in rows[:500]:
        if not isinstance(row, dict):
            warnings.append({"source": "Orphadata / Orphanet", "error": "malformed source row"})
            continue
        references = row.get("gene_external_references")
        hgnc_ids = sorted({
            reference.get("reference")
            for reference in references
            if isinstance(reference, dict)
            and isinstance(reference.get("source"), str)
            and reference["source"].casefold() == "hgnc"
            and isinstance(reference.get("reference"), str)
            and re.fullmatch(r"[0-9]+", reference["reference"])
        }) if isinstance(references, list) else []
        record_hash = row.get("record_sha256")
        if len(hgnc_ids) != 1 or not isinstance(record_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", record_hash):
            warnings.append({"source": "Orphadata / Orphanet", "error": "source row lacks one exact HGNC reference or a valid record digest"})
            continue
        gene_id = f"HGNC:{hgnc_ids[0]}"
        gene_node = gene_id
        assertion_node = f"OrphadataGeneDiseaseAssertion:{record_hash}"
        association_type = row.get("association_type") if isinstance(row.get("association_type"), str) else "Unspecified source association"
        association_status = row.get("association_status") if isinstance(row.get("association_status"), str) else "Unspecified source status"
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": gene_node, "entity_type": "Gene", "label": str(row.get("gene_symbol") or gene_id)[:512], "metadata": {"identifier_namespace": "HGNC", "identifier": gene_id}})
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": assertion_node, "entity_type": "GeneDiseaseAssertion", "label": f"Orphadata · {association_type[:300]}", "metadata": {"source": "Orphadata / Orphanet", "source_record_sha256": record_hash, "source_artifact_id": artifact_id, "association_type": association_type[:300], "association_status": association_status[:100], "source_of_validation": str(row.get("source_of_validation") or "")[:500], "source_dataset_date": str(result.get("dataset_release_date") or "")[:100]}})
        for endpoint, relation in ((gene_node, "SOURCE_ASSERTION_NAMES_GENE"), (disease_node, "SOURCE_ASSERTION_NAMES_DISEASE")):
            await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": assertion_node, "relation": relation, "target": endpoint, "evidence_ids": [], "artifact_ids": [artifact_id], "confidence": 1.0})
            edges_added += 1
    return {"edges_added": edges_added, "warnings": warnings}


async def _register_association_source_graph(call: Any, session_id: str, task_id: str, source_result: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Preserve each Open Targets/GWAS/Monarch source row as an artifact-linked assertion."""
    source = source_result.get("source")
    if source not in {"Open Targets Platform", "NHGRI-EBI GWAS Catalog", "Monarch Initiative Knowledge Graph"}:
        raise ValueError("unsupported association source")
    disease_id = source_result.get("disease_id")
    if not isinstance(disease_id, str) or not re.fullmatch(r"(?:EFO_[0-9]{7,}|MONDO:[0-9]{7}|MONDO_[0-9]{7})", disease_id):
        raise ValueError("association source did not return one exact EFO or MONDO identifier")
    records = source_result.get("records")
    if not isinstance(records, list):
        raise ValueError("association source records are malformed")
    disease_node = f"Disease:{disease_id}"
    disease_namespace = "EFO" if disease_id.startswith("EFO_") else "MONDO"
    await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": disease_node, "entity_type": "Disease", "label": disease_id, "metadata": {"identifier_namespace": disease_namespace, "identifier": disease_id}})
    edges_added = 0
    warnings: list[dict[str, str]] = []
    for record in records[:100]:
        if not isinstance(record, dict):
            warnings.append({"source": source, "error": "malformed association row"})
            continue
        gene_id = record.get("gene_id")
        mapping_candidate = record.get("gene_mapping_candidate")
        if gene_id is None and isinstance(mapping_candidate, str):
            gene_id = mapping_candidate.strip()
        if not isinstance(gene_id, str) or not gene_id.strip() or len(gene_id) > 128:
            warnings.append({"source": source, "error": "association row lacks a bounded gene identifier or mapping candidate"})
            continue
        record_hash = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assertion_prefix = {"Open Targets Platform": "OpenTargetsAssociation", "NHGRI-EBI GWAS Catalog": "GWASAssociation", "Monarch Initiative Knowledge Graph": "MonarchAssociation"}[source]
        assertion_node = f"{assertion_prefix}:{record_hash}"
        stable_ensembl = re.fullmatch(r"ENSG[0-9]{11}", gene_id)
        stable_curie = re.fullmatch(r"(?:HGNC|NCBIGene):[0-9]+", gene_id)
        stable_gene = bool(stable_ensembl or stable_curie)
        gene_node = gene_id if stable_curie else f"EnsemblGene:{gene_id}" if stable_ensembl else f"GeneMappingCandidate:{record_hash}"
        gene_kind = "Gene" if stable_gene else "GeneMappingCandidate"
        gene_label = str(record.get("symbol") or gene_id)[:512]
        gene_namespace = gene_id.split(":", 1)[0] if stable_curie else "ENSEMBL" if stable_ensembl else None
        gene_metadata = {"source": source, "source_gene_identifier": gene_id, "stable_gene_identifier": stable_gene, **({"mapping_candidate": gene_id} if not stable_gene else {}), **({"identifier_namespace": gene_namespace, "identifier": gene_id} if stable_gene else {})}
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": gene_node, "entity_type": gene_kind, "label": gene_label, "metadata": gene_metadata})
        score = record.get("score")
        assertion_metadata: dict[str, Any] = {"source": source, "source_record_sha256": record_hash, "source_artifact_id": artifact_id, "evidence_type": record.get("evidence_type")}
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            assertion_metadata["source_score"] = score
        for key in ("association_id", "study_id", "pubmed_id", "reported_trait", "association_category", "predicate", "knowledge_level", "agent_type", "primary_knowledge_source", "aggregator_knowledge_source", "publications", "has_evidence", "evidence_count", "negated"):
            value = record.get(key)
            if isinstance(value, str):
                assertion_metadata[key] = value[:256]
            elif isinstance(value, (int, bool)):
                assertion_metadata[key] = value
            elif isinstance(value, list):
                assertion_metadata[key] = [str(item)[:256] for item in value[:20] if isinstance(item, (str, int))]
        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": assertion_node, "entity_type": "GeneDiseaseAssociationRecord", "label": f"{source} · {str(record.get('evidence_type') or 'source association')[:400]}", "metadata": assertion_metadata})
        gene_relation = "SOURCE_ASSOCIATION_NAMES_GENE" if stable_gene else "SOURCE_ASSOCIATION_MAPS_GENE_CANDIDATE"
        for endpoint, relation in ((gene_node, gene_relation), (disease_node, "SOURCE_ASSOCIATION_NAMES_DISEASE")):
            await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": assertion_node, "relation": relation, "target": endpoint, "evidence_ids": [], "artifact_ids": [artifact_id], "confidence": 1.0})
            edges_added += 1
    return {"edges_added": edges_added, "warnings": warnings}


def _ontology_text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


async def _resolve_objective_disease_identifiers(call: Any, session_id: str, task_id: str, objective: str, concepts: list[str], cancellation: asyncio.Event, artifact_ids: list[str]) -> dict[str, Any]:
    """Resolve literal disease labels through OLS without synonym or cross-ontology inference."""
    objective_key = _ontology_text_key(objective)
    queries: list[str] = []
    for concept in concepts[:12]:
        if isinstance(concept, str) and concept.strip() and _ontology_text_key(concept) and f" {_ontology_text_key(concept)} " in f" {objective_key} ":
            queries.append(concept.strip())
    if not queries and objective.strip() and len(objective.strip()) <= 256:
        queries.append(objective.strip())
    unique_queries = list(dict.fromkeys(queries))[:3]
    if not unique_queries:
        return {"status": "unresolved", "identifiers": [], "candidates": [], "artifact_ids": [], "reason": "No bounded literal ontology search phrase occurs in the objective."}
    found: dict[str, dict[str, dict[str, str]]] = {"efo": {}, "mondo": {}}
    warnings: list[dict[str, str]] = []
    search_artifacts: list[str] = []
    for query in unique_queries:
        if cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during ontology resolution")
        for ontology in ("efo", "mondo"):
            try:
                result = await call("medai_ontology_search", {"session_id": session_id, "task_id": task_id, "query": query, "ontology": ontology, "limit": 50})
                artifact_id = result.get("artifact_id") if isinstance(result, dict) else None
                if isinstance(artifact_id, str) and artifact_id:
                    search_artifacts.append(artifact_id)
                    artifact_ids.append(artifact_id)
                records = result.get("records", []) if isinstance(result, dict) else []
                if not isinstance(records, list):
                    raise ValueError("ontology result records were malformed")
                for record in records:
                    if not isinstance(record, dict) or not isinstance(record.get("label"), str):
                        continue
                    label_key = _ontology_text_key(record["label"])
                    if not label_key or f" {label_key} " not in f" {objective_key} ":
                        continue
                    identifier = record.get("obo_id")
                    if ontology == "efo":
                        match = re.fullmatch(r"EFO[:_]([0-9]{7,})", str(identifier or ""), re.IGNORECASE)
                        canonical = f"EFO_{match.group(1)}" if match else None
                    else:
                        match = re.fullmatch(r"MONDO[:_]([0-9]{7})", str(identifier or ""), re.IGNORECASE)
                        canonical = f"MONDO:{match.group(1)}" if match else None
                    if canonical:
                        found[ontology][canonical] = {"identifier": canonical, "label": record["label"][:256], "ontology": ontology, "artifact_id": artifact_id if isinstance(artifact_id, str) else ""}
            except ResearchSessionCancelled:
                raise
            except Exception as error:
                warnings.append({"ontology": ontology, "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "error": type(error).__name__})
    identifiers = []
    candidates = []
    ambiguous_ontologies = []
    for ontology in ("efo", "mondo"):
        rows = list(found[ontology].values())
        if len(rows) == 1:
            identifiers.append(rows[0]["identifier"])
        elif len(rows) > 1:
            ambiguous_ontologies.append(ontology)
            candidates.extend(rows)
    status = "resolved" if identifiers and not ambiguous_ontologies and not warnings else "needs_review" if candidates or warnings else "unresolved"
    return {"status": status, "identifiers": identifiers, "candidates": candidates, "ambiguous_ontologies": ambiguous_ontologies, "warnings": warnings, "search_artifact_ids": search_artifacts, "method": "OLS stable identifier accepted only when its preferred label occurs literally in the user objective; synonyms, provider rank, and cross-ontology equivalence are not used"}


async def _retrieve_objective_disease_evidence(
    call: Any,
    session_id: str,
    task_id: str,
    worker_id: str,
    objective: str,
    cancellation: asyncio.Event,
    completed_source_ids: set[str],
    artifact_ids: list[str],
    checkpoint: dict[str, Any],
    ontology_terms: list[str] | None = None,
) -> dict[str, Any]:
    """Use explicit disease CURIEs or a literal preferred-label match from OLS."""
    identifiers = re.findall(r"(?<![A-Za-z0-9])(?:EFO_[0-9]{7,}|MONDO[:_][0-9]{7}|ORPHA[:_][0-9]{1,6})(?![A-Za-z0-9])", objective, re.IGNORECASE)
    if not identifiers:
        resolution = checkpoint.get("objective_disease_resolution")
        objective_sha256 = hashlib.sha256(objective.encode()).hexdigest()
        concepts_sha256 = hashlib.sha256(json.dumps(ontology_terms or [], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if not isinstance(resolution, dict) or resolution.get("objective_sha256") != objective_sha256 or resolution.get("concepts_sha256") != concepts_sha256:
            resolved = await _resolve_objective_disease_identifiers(call, session_id, task_id, objective, ontology_terms or [], cancellation, artifact_ids)
            resolution = {**resolved, "objective_sha256": objective_sha256, "concepts_sha256": concepts_sha256}
            checkpoint.update({"stage": "objective_disease_identifier_resolution", "artifact_ids": artifact_ids, "objective_disease_resolution": resolution})
            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
        resolved_ids = resolution.get("identifiers", [])
        if not isinstance(resolved_ids, list) or not resolved_ids:
            return {"status": "needs_review" if resolution.get("status") == "needs_review" else "skipped", "disease_id": None, "disease_identifier_resolution": resolution, "reason": "No unique stable disease identifier could be resolved from a preferred ontology label that appears literally in the objective."}
        results = []
        for identifier in resolved_ids:
            if not isinstance(identifier, str):
                continue
            results.append(await _retrieve_objective_disease_evidence(call, session_id, task_id, worker_id, f"{objective} {identifier}", cancellation, completed_source_ids, artifact_ids, checkpoint, ontology_terms))
        return {"status": "needs_review" if resolution.get("status") != "resolved" or any(row.get("status") != "succeeded" for row in results) else "succeeded", "disease_id": None, "resolved_disease_ids": resolved_ids, "disease_identifier_resolution": resolution, "source_results": results, "artifact_ids": sorted(set(artifact_ids)), "graph_edges_added": sum(int(row.get("graph_edges_added", 0)) for row in results), "graph_warnings": [warning for row in results for warning in row.get("graph_warnings", [])]}
    if len(identifiers) == 1 and identifiers[0].upper().startswith("ORPHA"):
        match = re.fullmatch(r"ORPHA[:_]([0-9]{1,6})", identifiers[0], re.IGNORECASE)
        if not match or int(match.group(1)) < 1:
            return {"status": "needs_review", "disease_id": identifiers[0], "reason": "The explicit ORPHAcode is malformed."}
        disease_id = f"ORPHA:{int(match.group(1))}"
        marker = f"disease-association:{disease_id}"
        if marker in completed_source_ids:
            previous_results = checkpoint.get("disease_association_results", {})
            previous = previous_results.get(disease_id) if isinstance(previous_results, dict) else None
            return previous if isinstance(previous, dict) else {"status": "needs_review", "disease_id": disease_id, "reason": "Checkpoint marks this query complete but lacks its result receipt."}
        if cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during Orphadata retrieval")
        try:
            result = await call("medai_gene_disease_orphadata", {"session_id": session_id, "task_id": task_id, "orphacode": str(int(match.group(1)))})
            if not isinstance(result, dict) or result.get("found") is not True or not isinstance(result.get("artifact_id"), str):
                raise ValueError("Orphadata returned a malformed or unretained exact-ORPHAcode result")
            artifact_id = result["artifact_id"]
            artifact_ids.append(artifact_id)
            graph_warnings = []
            graph_edges_added = 0
            try:
                graph_result = await _register_orphadata_gene_graph(call, session_id, task_id, result, artifact_id)
                graph_warnings.extend(graph_result["warnings"])
                graph_edges_added = graph_result["edges_added"]
            except ResearchSessionCancelled:
                raise
            except Exception as error:
                graph_warnings.append({"source": "Orphadata / Orphanet", "error": type(error).__name__})
            review = {"status": "needs_review" if graph_warnings else "succeeded", "disease_id": disease_id, "source_results": [{"source": result.get("source"), "disease": result.get("disease"), "record_count": result.get("record_count"), "artifact_id": artifact_id}], "artifact_ids": [artifact_id], "interpretation_limit": result.get("interpretation_limit"), "graph_edges_added": graph_edges_added, "graph_warnings": graph_warnings}
            completed_source_ids.add(marker)
            previous_results = checkpoint.get("disease_association_results", {})
            previous_results = dict(previous_results) if isinstance(previous_results, dict) else {}
            previous_results[disease_id] = review
            checkpoint.update({"stage": "objective_disease_associations_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids, "disease_association_results": previous_results, "disease_association_result": review})
            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
            return review
        except ResearchSessionCancelled:
            raise
        except Exception as error:
            return {"status": "needs_review", "disease_id": disease_id, "error": type(error).__name__, "reason": "Orphadata did not return a retained source receipt for the explicit ORPHAcode."}
    if len(identifiers) != 1:
        return {"status": "skipped", "disease_id": None, "reason": "Provide exactly one explicit EFO, MONDO, or ORPHA identifier, or use a disease label that resolves uniquely through OLS; MedAI does not infer ontology identity from synonyms or map between ontologies."}
    disease_id = identifiers[0].upper().replace("MONDO:", "MONDO_")
    disease_id = GeneDiseaseClient.validate_disease_id(disease_id)
    if disease_id.startswith("MONDO_"):
        disease_id = disease_id.replace("MONDO_", "MONDO:")
    marker = f"disease-association:{disease_id}"
    if marker in completed_source_ids:
        previous_results = checkpoint.get("disease_association_results", {})
        previous = previous_results.get(disease_id) if isinstance(previous_results, dict) else None
        if not isinstance(previous, dict):
            previous = checkpoint.get("disease_association_result")
        return previous if isinstance(previous, dict) else {"status": "needs_review", "disease_id": disease_id, "reason": "Checkpoint marks this query complete but lacks its result receipt."}
    if cancellation.is_set():
        raise ResearchSessionCancelled("research session cancellation requested during disease association retrieval")
    try:
        cross_mapping_cache = checkpoint.get("ontology_cross_mapping_reviews", {})
        cross_mapping_cache = dict(cross_mapping_cache) if isinstance(cross_mapping_cache, dict) else {}
        cross_mapping_review = cross_mapping_cache.get(disease_id)
        if not isinstance(cross_mapping_review, dict):
            target_prefix = "MONDO" if disease_id.startswith("EFO_") else "EFO"
            try:
                mapping_result = await call("medai_ontology_cross_mapping_review", {
                    "session_id": session_id,
                    "task_id": task_id,
                    "source_identifier": disease_id,
                    "target_prefixes": [target_prefix],
                    "distance": 1,
                    "limit": 50,
                })
                if not isinstance(mapping_result, dict) or not isinstance(mapping_result.get("artifact_id"), str) or not isinstance(mapping_result.get("mappings"), list):
                    raise ValueError("OxO mapping tool returned a malformed or unretained result")
                truncated = mapping_result.get("truncated", False)
                provider_mapping_count = mapping_result.get("provider_mapping_count", len(mapping_result["mappings"]))
                duplicate_mapping_rows = mapping_result.get("duplicate_mapping_rows", 0)
                if (type(truncated) is not bool or isinstance(provider_mapping_count, bool)
                        or not isinstance(provider_mapping_count, int) or provider_mapping_count < len(mapping_result["mappings"])
                        or isinstance(duplicate_mapping_rows, bool) or not isinstance(duplicate_mapping_rows, int)
                        or duplicate_mapping_rows < 0):
                    raise ValueError("OxO mapping tool returned malformed completeness metadata")
                cross_mapping_review = {
                    "status": "pending_researcher_review" if mapping_result["mappings"] or truncated else "no_direct_candidate_found",
                    "source_identifier": disease_id,
                    "target_prefix": target_prefix,
                    "mappings": mapping_result["mappings"],
                    "provider_mapping_count": provider_mapping_count,
                    "truncated": truncated,
                    "duplicate_mapping_rows": duplicate_mapping_rows,
                    "artifact_id": mapping_result["artifact_id"],
                    "interpretation": mapping_result.get("mapping_interpretation"),
                }
                artifact_ids.append(mapping_result["artifact_id"])
            except ResearchSessionCancelled:
                raise
            except Exception as error:
                cross_mapping_review = {
                    "status": "mapping_unavailable",
                    "source_identifier": disease_id,
                    "target_prefix": target_prefix,
                    "error": type(error).__name__,
                    "interpretation": "No ontology identity equivalence was inferred; mapping retrieval requires review.",
                }
            cross_mapping_cache[disease_id] = cross_mapping_review
            checkpoint.update({"stage": "ontology_cross_mapping_review", "artifact_ids": artifact_ids, "ontology_cross_mapping_reviews": cross_mapping_cache})
            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
        if disease_id.startswith("EFO_"):
            result = await call("medai_gene_disease_evidence", {"session_id": session_id, "task_id": task_id, "disease_id": disease_id, "limit": 50})
            if not isinstance(result, dict) or not isinstance(result.get("reconciliation"), dict):
                raise ValueError("gene-disease evidence tool returned a malformed result")
            disease_graph_edges = 0
            disease_graph_warnings = []
            sources = result.get("sources", [])
            source_artifact_ids = result.get("artifact_ids", [])
            if not isinstance(sources, list) or not isinstance(source_artifact_ids, list) or len(source_artifact_ids) < len(sources):
                disease_graph_warnings.append({"source": "EFO association retrieval", "error": "source artifact receipts do not align with provider records"})
            else:
                for source_result, source_artifact_id in zip(sources[:20], source_artifact_ids[:20]):
                    if not isinstance(source_result, dict) or not isinstance(source_artifact_id, str):
                        disease_graph_warnings.append({"source": "EFO association retrieval", "error": "malformed provider source receipt"})
                        continue
                    try:
                        graph_result = await _register_association_source_graph(call, session_id, task_id, source_result, source_artifact_id)
                        disease_graph_edges += graph_result["edges_added"]
                        disease_graph_warnings.extend(graph_result["warnings"])
                    except ResearchSessionCancelled:
                        raise
                    except Exception as error:
                        disease_graph_warnings.append({"source": str(source_result.get("source")), "error": type(error).__name__})
        else:
            # MONDO is queried only against sources that natively curate MONDO IDs.
            # Keep the source receipts independent; never synthesize a crosswalk.
            tool_requests = [
                ("ClinGen", "medai_gene_disease_clingen_disease", {"session_id": session_id, "task_id": task_id, "disease_curie": disease_id, "limit": 50}),
                ("GenCC", "medai_gene_disease_gencc_disease", {"session_id": session_id, "task_id": task_id, "disease_curie": disease_id, "limit": 50}),
                ("Monarch Initiative Knowledge Graph", "medai_gene_disease_evidence", {"session_id": session_id, "task_id": task_id, "disease_id": disease_id, "limit": 50}),
            ]
            outcomes = await asyncio.gather(*(call(tool, arguments) for _, tool, arguments in tool_requests), return_exceptions=True)
            source_errors = [{"source": label, "error": type(value).__name__} for (label, _, _), value in zip(tool_requests, outcomes) if isinstance(value, Exception)]
            source_results = [value for value in outcomes[:2] if isinstance(value, dict) and isinstance(value.get("artifact_id"), str)]
            monarch_result = outcomes[2]
            if isinstance(monarch_result, dict):
                monarch_sources = monarch_result.get("sources", [])
                monarch_artifacts = monarch_result.get("artifact_ids", [])
                if isinstance(monarch_sources, list) and isinstance(monarch_artifacts, list):
                    source_results.extend({**source, "artifact_id": artifact_id} for source, artifact_id in zip(monarch_sources, monarch_artifacts) if isinstance(source, dict) and isinstance(artifact_id, str))
                nested_errors = monarch_result.get("source_errors", [])
                if isinstance(nested_errors, list):
                    source_errors.extend(error for error in nested_errors if isinstance(error, dict))
            elif not isinstance(monarch_result, Exception):
                source_errors.append({"source": "Monarch Initiative Knowledge Graph", "error": "malformed source result"})
            if not source_results:
                raise ValueError("no native MONDO gene-disease source returned a receipt")
            disease_graph_edges = 0
            disease_graph_warnings = []
            for source_result in source_results:
                source_name = source_result.get("source")
                if source_name == "Monarch Initiative Knowledge Graph":
                    try:
                        graph_result = await _register_association_source_graph(call, session_id, task_id, source_result, source_result["artifact_id"])
                        disease_graph_edges += graph_result["edges_added"]
                        disease_graph_warnings.extend(graph_result["warnings"])
                    except ResearchSessionCancelled:
                        raise
                    except Exception as error:
                        disease_graph_warnings.append({"source": str(source_name), "error": type(error).__name__})
                    continue
                source_key = "GenCC" if source_name == "Gene Curation Coalition (GenCC)" else "ClinGen" if source_name == "ClinGen Gene-Disease Validity" else None
                if source_key is None:
                    disease_graph_warnings.append({"source": str(source_name), "error": "unsupported native disease-curation source"})
                    continue
                try:
                    graph_result = await _register_gene_disease_curation_graph(call, session_id, task_id, source_key, source_result, source_result["artifact_id"])
                    disease_graph_edges += graph_result["edges_added"]
                    disease_graph_warnings.extend(graph_result["warnings"])
                except ResearchSessionCancelled:
                    raise
                except Exception as error:
                    disease_graph_warnings.append({"source": source_key, "error": type(error).__name__})
            result = {"source_errors": source_errors, "artifact_ids": [value["artifact_id"] for value in source_results], "source_results": [{"source": value.get("source"), "query": value.get("query"), "record_count": value.get("total_matching_assertions", value.get("total_matching_curations")), "artifact_id": value["artifact_id"]} for value in source_results]}
        receipts = result.get("artifact_ids", [])
        if isinstance(receipts, list):
            artifact_ids.extend(value for value in receipts if isinstance(value, str) and value)
        source_errors = result.get("source_errors", [])
        mapping_requires_review = cross_mapping_review.get("status") in {"pending_researcher_review", "mapping_unavailable"}
        retained_receipts = [value for value in receipts if isinstance(value, str)] if isinstance(receipts, list) else []
        mapping_artifact_id = cross_mapping_review.get("artifact_id")
        if isinstance(mapping_artifact_id, str) and mapping_artifact_id:
            retained_receipts.append(mapping_artifact_id)
        review = {"status": "needs_review" if source_errors or mapping_requires_review or (disease_id.startswith(("MONDO:", "EFO_")) and disease_graph_warnings) else "succeeded", "disease_id": disease_id, "ontology_cross_mapping_review": cross_mapping_review, "reconciliation": result.get("reconciliation"), "source_results": result.get("source_results", []), "source_errors": source_errors if isinstance(source_errors, list) else [], "artifact_ids": sorted(set(retained_receipts)), **({"graph_edges_added": disease_graph_edges, "graph_warnings": disease_graph_warnings} if disease_id.startswith(("MONDO:", "EFO_")) else {})}
        completed_source_ids.add(marker)
        previous_results = checkpoint.get("disease_association_results", {})
        previous_results = dict(previous_results) if isinstance(previous_results, dict) else {}
        previous_results[disease_id] = review
        checkpoint.update({"stage": "objective_disease_associations_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids, "disease_association_results": previous_results, "disease_association_result": review})
        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
        return review
    except ResearchSessionCancelled:
        raise
    except Exception as error:
        return {"status": "needs_review", "disease_id": disease_id, "error": type(error).__name__, "reason": "Native disease evidence sources did not produce the expected receipts."}


async def _search_hypothesis_literature(
    call: Any,
    session_id: str,
    task_id: str,
    worker_id: str,
    proposal: dict[str, Any],
    paper_limit: int,
    cancellation: asyncio.Event,
    completed_source_ids: set[str],
    artifact_ids: list[str],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Run bounded exact-entity searches for candidates; never assign novelty."""
    candidates = proposal.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("hypothesis proposal candidates are malformed")
    valid = [row for row in candidates if isinstance(row, dict) and isinstance(row.get("id"), str)][:10]
    budget_state = await call("medai_research_get_budget", {"session_id": session_id})
    remaining = budget_state.get("remaining", 0) if isinstance(budget_state, dict) else 0
    # Reserve three credits for later criticism/report stages and allow five per
    # search for E-utilities plus bounded title ranking/model usage.
    query_limit = min(3, paper_limit, max(0, (int(remaining) - 3) // 5)) if isinstance(remaining, int) and not isinstance(remaining, bool) else 0
    prior_results = checkpoint.get("hypothesis_literature_results", {})
    if not isinstance(prior_results, dict):
        prior_results = {}
    reviews = []
    warnings = []
    searched = 0
    for candidate in valid[:query_limit]:
        if cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during hypothesis literature search")
        candidate_id = candidate["id"]
        try:
            query = build_hypothesis_literature_query(candidate)
            query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
            marker = f"hypothesis-literature:{candidate_id}:{query_sha256}"
            cached = prior_results.get(candidate_id)
            if marker in completed_source_ids:
                cached_searches = cached.get("candidate_searches") if isinstance(cached, dict) else None
                if cached.get("query_sha256") == query_sha256 and isinstance(cached_searches, list) and cached_searches and isinstance(cached_searches[0], dict):
                    reviews.append(cached_searches[0])
                else:
                    warnings.append({"hypothesis_id": candidate_id, "reason": "completed search checkpoint lacks a matching query receipt"})
                continue
            search = await call("medai_papers_search_pubmed", {"session_id": session_id, "task_id": task_id, "query": query, "limit": min(5, paper_limit)})
            if not isinstance(search, dict):
                raise ValueError("PubMed search response is malformed")
            papers = search.get("results", [])
            if not isinstance(papers, list):
                raise ValueError("PubMed result list is malformed")
            stored_papers = []
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                metadata = bounded_paper_record(paper)
                artifact = await call("medai_research_store_artifact", {
                    "session_id": session_id, "task_id": task_id, "kind": "hypothesis_literature_candidate",
                    "source": "PubMed exact-entity hypothesis discovery", "source_id": metadata["pmid"],
                    "payload": {"hypothesis_id": candidate_id, "query_sha256": query_sha256, **metadata},
                })
                if not isinstance(artifact, dict) or not isinstance(artifact.get("id"), str):
                    raise ValueError("PubMed candidate artifact was not persisted")
                artifact_receipt = {"id": artifact["id"], "record_sha256": metadata["record_sha256"]}
                stored_papers.append(artifact_receipt)
                artifact_ids.append(artifact["id"])
            review = build_hypothesis_literature_receipt(candidate, query, search, stored_papers)
            review["status"] = "searched"
            review["candidate_papers"] = len(stored_papers)
            review["hypothesis_id"] = candidate_id
            reviews.append(review)
            prior_results[candidate_id] = {"query_sha256": query_sha256, "candidate_searches": [review]}
            completed_source_ids.add(marker)
            searched += 1
            ranking_id = search.get("ranking_artifact_id")
            if isinstance(ranking_id, str) and ranking_id:
                artifact_ids.append(ranking_id)
            coverage_id = search.get("coverage_artifact_id")
            if isinstance(coverage_id, str) and coverage_id:
                artifact_ids.append(coverage_id)
            checkpoint.update({"stage": "hypothesis_literature_leads_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids, "hypothesis_literature_results": prior_results})
            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": worker_id, "checkpoint": checkpoint})
        except ResearchSessionCancelled:
            raise
        except Exception as error:
            warnings.append({"hypothesis_id": candidate_id, "error": type(error).__name__})
    omitted = max(0, len(valid) - min(len(valid), query_limit))
    receipt = {
        "schema_version": 1,
        "proposal_artifact_id": proposal.get("artifact_id"),
        "searched_candidate_count": sum(1 for row in reviews if row.get("status") == "searched"),
        "query_attempts_this_run": searched,
        "total_candidate_count": len(candidates),
        "omitted_candidate_count": max(0, len(candidates) - len(valid)) + omitted,
        "search_budget_remaining_at_start": remaining if isinstance(remaining, int) else None,
        "estimated_credits_per_search": 5,
        "reserved_downstream_credits": 3,
        "candidate_searches": reviews,
        "warnings": warnings,
        "novelty_assessment": None,
        "review_status": "requires_human_relevance_review",
        "interpretation_limit": "Exact-entity PubMed retrieval provides bounded literature leads only; it is not source-complete and does not establish semantic relevance, absence of prior work, or novelty. Novelty remains unscored.",
    }
    stored = await call("medai_research_store_artifact", {"session_id": session_id, "task_id": task_id, "kind": "hypothesis_literature_comparison", "source": "MedAI bounded PubMed candidate search", "source_id": str(proposal.get("artifact_id") or session_id), "payload": receipt})
    if isinstance(stored, dict) and isinstance(stored.get("id"), str):
        artifact_ids.append(stored["id"])
        receipt["artifact_id"] = stored["id"]
    receipt["artifact_ids"] = artifact_ids
    receipt["query_attempts"] = searched
    receipt["warnings"] = warnings
    return receipt


async def _collect_artifact_pages(
    call: Any,
    session_id: str,
    kind: str | None = None,
    cancellation: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Collect cursor pages with explicit page, item, and serialized-byte ceilings."""
    max_pages, max_items, max_bytes = 50, 10_000, 8 * 1024 * 1024
    items: list[dict[str, Any]] = []
    byte_count = 0
    cursor: dict[str, Any] | None = None
    previous_cursor: tuple[int, str] | None = None
    pages_read = 0
    for _ in range(max_pages):
        if cancellation is not None and cancellation.is_set():
            raise ResearchSessionCancelled("research session cancellation requested during artifact pagination")
        arguments: dict[str, Any] = {"session_id": session_id, "limit": 200}
        if kind is not None:
            arguments["kind"] = kind
        if cursor is not None:
            arguments.update({"after_recorded_at": cursor["recorded_at"], "after_id": cursor["id"]})
        page = await call("medai_research_list_artifacts", arguments)
        pages_read += 1
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise ValueError("AgentMesh artifact pagination returned a malformed page")
        if len(page["items"]) > 200:
            raise ValueError("AgentMesh artifact page exceeded its declared item limit")
        for item in page["items"]:
            if not isinstance(item, dict):
                raise ValueError("AgentMesh artifact page contains a malformed row")
            byte_count += len(_json_bytes(item))
            if len(items) >= max_items or byte_count > max_bytes:
                raise ArtifactCollectionLimitError(
                    f"complete artifact set exceeds worker bound ({max_items} items or {max_bytes} bytes)"
                )
            items.append(item)
        cursor = page.get("next_cursor")
        if cursor is None:
            return {"items": items, "truncated": False, "pages_read": pages_read, "serialized_bytes": byte_count}
        if not isinstance(cursor, dict) or not isinstance(cursor.get("recorded_at"), int) or not isinstance(cursor.get("id"), str) or not cursor["id"]:
            raise ValueError("AgentMesh artifact page returned an invalid continuation cursor")
        if not page["items"]:
            raise ValueError("AgentMesh artifact page returned a cursor without any rows")
        last = page["items"][-1]
        if not isinstance(last.get("recorded_at"), int) or not isinstance(last.get("id"), str):
            raise ValueError("AgentMesh artifact page row lacks stable cursor fields")
        cursor_key = (cursor["recorded_at"], cursor["id"])
        if cursor_key != (last["recorded_at"], last["id"]) or (previous_cursor is not None and cursor_key <= previous_cursor):
            raise ValueError("AgentMesh artifact continuation cursor did not advance from the final row")
        previous_cursor = cursor_key
    raise ArtifactCollectionLimitError(f"complete artifact set exceeds worker bound ({max_pages} pages)")


class ResearchWorker:
    def __init__(self, gateway_url: str, worker_id: str | None = None, lease_seconds: int = 900, agent_id: str | None = None, agent_api_token: str | None = None, transport_headers: dict[str, str] | None = None):
        if not gateway_url.startswith(("http://", "https://")):
            raise ValueError("AgentMesh gateway URL must use http:// or https://")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("lease duration must be 30..3600 seconds")
        if agent_id is not None and agent_id not in AGENTS:
            raise ValueError(f"unknown scientific agent ID: {agent_id}")
        specialist_meta = _specialist_meta(agent_id, agent_api_token)
        self.gateway_url = gateway_url
        self.transport_headers = dict(transport_headers or {})
        self.agent_id = agent_id
        self.agent_api_token = agent_api_token if specialist_meta else None
        if agent_id:
            self.worker_id = f"{worker_id or 'medai'}.{agent_id}.{uuid4().hex[:12]}"
        else:
            self.worker_id = worker_id or f"medai-{uuid4().hex}"
        if len(self.worker_id) > 128:
            raise ValueError("worker ID plus specialist ID exceeds the 128 character lease limit")
        self.lease_seconds = lease_seconds

    async def execute(self, session_id: str, max_tasks: int = 32, paper_limit: int | None = None) -> dict:
        if not session_id.strip():
            raise ValueError("session_id cannot be empty")
        if not 1 <= max_tasks <= 100:
            raise ValueError("max_tasks must be 1..100")
        if paper_limit is not None and (isinstance(paper_limit, bool) or not isinstance(paper_limit, int) or not 1 <= paper_limit <= 20):
            raise ValueError("paper_limit must be 1..20")

        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        summary = {"session_id": session_id, "worker_id": self.worker_id, "claimed": 0, "succeeded": [], "needs_review": [], "failed": [], "skipped": [], "cancelled": [], "control_plane_update_failures": [], "remaining_limit": max_tasks}
        async with streamable_http_client(self.gateway_url, headers=self.transport_headers or None) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                active_cancellation: asyncio.Event | None = None
                active_agent_id: str | None = None
                active_policy_violations: list[str] = []

                async def call(name: str, arguments: dict | None = None) -> Any:
                    if active_agent_id is not None and not agent_may_call(active_agent_id, name):
                        if name not in active_policy_violations:
                            active_policy_violations.append(name)
                        raise PermissionError(f"specialist {active_agent_id} is not authorized to call MCP tool {name}")
                    try:
                        return _tool_value(await client.call_tool(name, arguments=arguments or {}, meta=_specialist_meta(self.agent_id, self.agent_api_token)))
                    except ResearchSessionCancelled:
                        if active_cancellation is not None:
                            active_cancellation.set()
                        raise

                if self.agent_id is None:
                    await call("medai_research_recover_expired_tasks", {"session_id": session_id, "limit": 100})
                session_rows = await call("medai_research_list_sessions")
                sessions = session_rows.get("items", []) if isinstance(session_rows, dict) else []
                session = next((row for row in sessions if row.get("id") == session_id), None)
                if session is None:
                    raise ValueError(f"research session {session_id} was not found through the configured MCP endpoint")
                requested_paper_limit = paper_limit
                research_depth = session.get("depth", "standard")
                paper_limit = paper_limit_for_depth(research_depth, requested_paper_limit)
                summary["paper_limit"] = paper_limit
                summary["paper_limit_source"] = "operator_override" if requested_paper_limit is not None else "session_depth_default"
                summary["research_depth"] = research_depth
                objective = session["objective"].strip()
                persisted_plan = await call("medai_research_get_plan", {"session_id": session_id})
                coordination = ResearchDirector().coordinate(
                    session_id=session_id,
                    objective=objective,
                    depth=session.get("depth", "standard"),
                    budget=_session_research_budget(session),
                    tasks=persisted_plan.get("tasks", []) if isinstance(persisted_plan, dict) else [],
                )
                assignments = {item["task_id"]: item for item in coordination["task_assignments"]}
                summary["research_plan_sha256"] = coordination["plan_sha256"]
                if self.agent_id is not None:
                    summary["specialist_agent_id"] = self.agent_id
                    actor_task_ids = [task_id for task_id, specialist in AGENT_BY_TASK.items() if specialist == self.agent_id]
                    if not actor_task_ids:
                        raise ValueError(f"scientific agent {self.agent_id} has no assigned research tasks")
                else:
                    actor_task_ids = None

                for _ in range(max_tasks):
                    control = await call("medai_research_get_control", {"session_id": session_id})
                    if control.get("cancellation_requested"):
                        summary["cancelled"].append("session")
                        break
                    claimed = await call(
                        "medai_research_claim_tasks",
                        {"session_id": session_id, "worker_id": self.worker_id, "limit": 1, "lease_seconds": self.lease_seconds, **({"task_ids": actor_task_ids} if actor_task_ids is not None else {})},
                    )
                    tasks = claimed.get("tasks", []) if isinstance(claimed, dict) else []
                    if not tasks:
                        break
                    task = tasks[0]
                    task_id = task["id"]
                    summary["claimed"] += 1
                    summary["remaining_limit"] -= 1
                    try:
                        checkpoint = json.loads(task.get("checkpoint") or "{}")
                    except (TypeError, json.JSONDecodeError):
                        checkpoint = {}
                    if not isinstance(checkpoint, dict):
                        checkpoint = {}
                    completed_source_ids = set(checkpoint.get("completed_source_ids", []))
                    artifact_ids = list(checkpoint.get("artifact_ids", []))
                    assignment = assignments[task_id]
                    safe_objective = objective
                    if task_id in {"literature.discover", "trials.discover"}:
                        try:
                            safe_objective = ensure_public_research_query(objective)
                        except ValueError as error:
                            message = str(error)
                            await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "needs_review", "output": json.dumps({"reason": message, "specialist_agent": assignment, "research_plan_sha256": coordination["plan_sha256"]}, ensure_ascii=False), "worker_id": self.worker_id})
                            summary["needs_review"].append(task_id)
                            continue
                    active_agent_id = assignment["agent_id"]
                    active_policy_violations = []
                    cancellation = asyncio.Event()
                    active_cancellation = cancellation
                    lease_lost: list[str] = []
                    heartbeat = asyncio.create_task(self._maintain_lease(call, session_id, task_id, cancellation, lease_lost))
                    try:
                        output: dict[str, Any]
                        if task_id == "objective.normalize":
                            coordination_artifact = await call(
                                "medai_research_store_artifact",
                                {
                                    "session_id": session_id,
                                    "task_id": task_id,
                                    "kind": "research_agent_coordination_plan",
                                    "source": "MedAI Research Director",
                                    "source_id": coordination["plan_sha256"],
                                    "payload": coordination,
                                },
                            )
                            depth = session.get("depth", "standard")
                            existing_plan_data = await _collect_artifact_pages(call, session_id, "research_objective_plan", cancellation)
                            existing_plan_rows = existing_plan_data.get("items", []) if isinstance(existing_plan_data, dict) else []
                            existing_plan_record = objective_plan_record_from_artifacts(existing_plan_rows, objective=objective, depth=depth, session_id=session_id)
                            if existing_plan_record:
                                existing_planning, existing_artifact = existing_plan_record
                                existing_envelope = json.loads(existing_artifact["payload"])
                                model_plan = {
                                    "available": True,
                                    "planning": existing_planning,
                                    "artifact_id": existing_artifact["id"],
                                    "model": existing_envelope.get("model"),
                                    "provider": existing_envelope.get("provider"),
                                    "response_sha256": existing_envelope.get("response_sha256"),
                                    "usage": existing_envelope.get("usage", {}),
                                    "reused_artifact": True,
                                }
                            else:
                                try:
                                    model_plan = await call("medai_research_plan_objective", {"session_id": session_id, "task_id": task_id, "objective": objective, "depth": depth})
                                except Exception as error:
                                    model_plan = {"available": False, "reason": type(error).__name__}
                            if isinstance(model_plan, dict) and model_plan.get("available"):
                                scope_application = await call("medai_research_apply_task_scope", {"session_id": session_id, "worker_id": self.worker_id, "plan_artifact_id": model_plan["artifact_id"]})
                                scope_skipped = scope_application.get("skipped_task_ids", []) if isinstance(scope_application, dict) else []
                                summary["skipped"].extend(task for task in scope_skipped if task not in summary["skipped"])
                                output = {"normalized_objective": model_plan["planning"]["normalized_question"], "depth": depth, "method": "configured model planned search queries and selected a validated AgentMesh task scope; no evidence inferred", "pubmed_queries": model_plan["planning"]["pubmed_queries"], "clinical_trials_query": model_plan["planning"]["clinical_trials_query"], "concepts": model_plan["planning"]["concepts"], "scope_notes": model_plan["planning"]["scope_notes"], "selected_task_ids": model_plan["planning"]["selected_task_ids"], "task_scope_notes": model_plan["planning"]["task_scope_notes"], "scope_skipped_task_ids": scope_skipped, "plan_reused_after_retry": model_plan.get("reused_artifact", False), "model": model_plan.get("model"), "provider": model_plan.get("provider"), "plan_artifact_id": model_plan.get("artifact_id"), "response_sha256": model_plan.get("response_sha256"), "usage": model_plan.get("usage", {}), "coordination_artifact_id": coordination_artifact.get("id")}
                            else:
                                output = {"normalized_objective": objective, "depth": depth, "method": "deterministic whitespace normalization; all registered research tasks remain enabled", "model_planning": "unavailable; model provider was not configured or returned an error", "model_planning_error": model_plan.get("reason") if isinstance(model_plan, dict) else "invalid provider response", "selected_task_ids": list(RESEARCH_TASK_IDS), "coordination_artifact_id": coordination_artifact.get("id")}
                            status = "succeeded"
                        elif task_id == "literature.discover":
                            plan_data = await _collect_artifact_pages(call, session_id, "research_objective_plan", cancellation)
                            plan_artifacts = plan_data.get("items", []) if isinstance(plan_data, dict) else []
                            objective_plan = objective_plan_from_artifacts(plan_artifacts, objective=objective, depth=session.get("depth", "standard"), session_id=session_id)
                            pubmed_queries = objective_plan.get("pubmed_queries", []) if objective_plan else [safe_objective]
                            pubmed_queries = list(dict.fromkeys(query.strip() for query in pubmed_queries if isinstance(query, str) and query.strip()))[:3] or [safe_objective]
                            # Give every planned query at least one PMID slot; a lower run budget
                            # must not produce invalid zero-result PubMed requests.
                            pubmed_queries = pubmed_queries[:paper_limit]
                            crossref_query = objective_plan.get("normalized_question", safe_objective) if objective_plan else safe_objective
                            query_hash = hashlib.sha256(crossref_query.encode()).hexdigest()
                            source_errors: list[dict[str, str]] = []
                            source_successes: list[str] = []
                            records_found: dict[str, int] = {}
                            ranking_runs: list[dict[str, Any]] = []
                            search_coverage: list[dict[str, Any]] = []
                            coverage_needs_review = False
                            saved_coverage = checkpoint.get("pubmed_search_coverage", {})
                            if not isinstance(saved_coverage, dict):
                                saved_coverage = {}
                            completed_pubmed_queries = [query for query in pubmed_queries if f"pubmed-query:{hashlib.sha256(query.encode()).hexdigest()}" in completed_source_ids]
                            pending_pubmed_queries = [query for query in pubmed_queries if query not in completed_pubmed_queries]
                            records_found["PubMed"] = 0
                            if completed_pubmed_queries:
                                source_successes.append("PubMed")
                                for completed_query in completed_pubmed_queries:
                                    query_digest = hashlib.sha256(completed_query.encode()).hexdigest()
                                    prior_coverage = saved_coverage.get(query_digest)
                                    if isinstance(prior_coverage, dict):
                                        search_coverage.append({"query_sha256": query_digest, **prior_coverage, "reused_checkpoint": True})
                                        coverage_needs_review = coverage_needs_review or prior_coverage.get("fetch_incomplete") is True
                                        count = prior_coverage.get("fetched_records")
                                        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                                            records_found["PubMed"] += count
                                    else:
                                        search_coverage.append({"query_sha256": query_digest, "coverage_status": "unavailable_for_completed_query", "reused_checkpoint": True})
                                        coverage_needs_review = True
                            query_count = len(pubmed_queries)
                            query_limits = [paper_limit // query_count + (index < paper_limit % query_count) for index in range(query_count)]
                            for query in pending_pubmed_queries:
                                if cancellation.is_set():
                                    break
                                query_hash_marker = f"pubmed-query:{hashlib.sha256(query.encode()).hexdigest()}"
                                query_index = pubmed_queries.index(query)
                                try:
                                    papers = await call("medai_papers_search_pubmed", {"session_id": session_id, "task_id": task_id, "query": query, "limit": query_limits[query_index]})
                                    records = papers if isinstance(papers, list) else papers.get("results", [])
                                    if isinstance(papers, dict):
                                        query_digest = hashlib.sha256(query.encode()).hexdigest()
                                        coverage = papers.get("coverage")
                                        if papers.get("coverage_artifact_id"):
                                            artifact_ids.append(papers["coverage_artifact_id"])
                                        if isinstance(coverage, dict):
                                            saved_coverage[query_digest] = coverage
                                            search_coverage.append({"query_sha256": query_digest, **coverage, "reused_checkpoint": False})
                                            coverage_needs_review = coverage_needs_review or coverage.get("fetch_incomplete") is True
                                            checkpoint.update({"stage": "pubmed_search_coverage_retained", "pubmed_search_coverage": saved_coverage, "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                        else:
                                            source_errors.append({"source": "PubMed", "query_sha256": query_digest, "error": "search_coverage_missing"})
                                            coverage_needs_review = True
                                        ranking = papers.get("ranking", {})
                                        ranking_runs.append({"query_sha256": hashlib.sha256(query.encode()).hexdigest(), "available": bool(ranking.get("available")), "provider": ranking.get("provider"), "model": ranking.get("model"), "ranked_pmids": ranking.get("ranked_pmids", []), "reason": ranking.get("reason"), "error": ranking.get("error")})
                                        if papers.get("ranking_artifact_id"):
                                            artifact_ids.append(papers["ranking_artifact_id"])
                                    else:
                                        query_digest = hashlib.sha256(query.encode()).hexdigest()
                                        source_errors.append({"source": "PubMed", "query_sha256": query_digest, "error": "search_coverage_missing"})
                                        coverage_needs_review = True
                                    records_found["PubMed"] += len(records)
                                    for paper in records:
                                        if cancellation.is_set():
                                            break
                                        if not isinstance(paper, dict):
                                            continue
                                        provenance = paper.get("provenance", {})
                                        source_id = str(provenance.get("source_id") or provenance.get("sourceId") or "")
                                        if not source_id or source_id in completed_source_ids:
                                            continue
                                        retained_paper = omit_pubmed_abstract_text(paper)
                                        artifact = await self._store_record(call, session_id, task_id, "paper", "PubMed", source_id, retained_paper)
                                        completed_source_ids.add(source_id)
                                        if artifact.get("id"):
                                            artifact_ids.append(artifact["id"])
                                        checkpoint.update({"stage": "literature_records_persisted", "last_query": query, "pubmed_search_coverage": saved_coverage, "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                    if not cancellation.is_set():
                                        completed_source_ids.add(query_hash_marker)
                                        if "PubMed" not in source_successes:
                                            source_successes.append("PubMed")
                                        checkpoint.update({"stage": "pubmed_query_complete", "pubmed_search_coverage": saved_coverage, "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except Exception as error:
                                    source_errors.append({"source": "PubMed", "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "error": type(error).__name__})

                            crossref_marker = f"crossref-query:{query_hash}"
                            citation_links_indexed = 0
                            if crossref_marker in completed_source_ids:
                                source_successes.append("Crossref")
                            elif not cancellation.is_set():
                                try:
                                    crossref = await call("medai_papers_resolve_crossref", {"session_id": session_id, "task_id": task_id, "query": crossref_query, "limit": min(paper_limit, 5)})
                                    records = crossref.get("records", []) if isinstance(crossref, dict) else []
                                    records_found["Crossref"] = len(records)
                                    citation_links_indexed = int(crossref.get("citation_links_indexed", 0))
                                    if crossref.get("artifact_id"):
                                        artifact_ids.append(crossref["artifact_id"])
                                    completed_source_ids.add(crossref_marker)
                                    source_successes.append("Crossref")
                                    checkpoint.update({"stage": "crossref_citations_indexed", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                    await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except Exception as error:
                                    source_errors.append({"source": "Crossref", "error": type(error).__name__})
                            if not source_successes and not cancellation.is_set():
                                raise RuntimeError("all literature discovery sources failed")
                            output = {"queries": pubmed_queries, "crossref_query": crossref_query, "model_planned": objective_plan is not None, "title_ranking_runs": ranking_runs, "search_coverage": search_coverage, "sources": source_successes, "source_errors": source_errors, "records_found": records_found, "citation_links_indexed": citation_links_indexed, "artifact_ids": artifact_ids}
                            status = "needs_review" if coverage_needs_review else "succeeded"
                        elif task_id == "literature.extract":
                            source_data = await _collect_artifact_pages(call, session_id, "paper", cancellation=cancellation)
                            paper_artifacts = source_data.get("items", []) if isinstance(source_data, dict) else []
                            pubmed_artifacts = [item for item in paper_artifacts if item.get("source") == "PubMed"]
                            processed = 0
                            pmc_processed = 0
                            abstract_processed = 0
                            model_annotations = 0
                            model_annotation_artifact_ids: list[str] = []
                            warnings: list[dict[str, str]] = []
                            for source_artifact in pubmed_artifacts[:paper_limit]:
                                if cancellation.is_set():
                                    break
                                raw_payload = source_artifact.get("payload")
                                try:
                                    paper = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
                                    source_id = str(paper.get("provenance", {}).get("source_id", ""))
                                    if not source_id or source_id in completed_source_ids:
                                        continue
                                    identifiers = paper.get("identifiers", {})
                                    pmc_id = identifiers.get("pmc") if isinstance(identifiers, dict) else None
                                    if isinstance(pmc_id, str) and pmc_id.upper().startswith("PMC"):
                                        try:
                                            fulltext = await call("medai_papers_understand_pmc", {"session_id": session_id, "task_id": task_id, "identifier": pmc_id})
                                            parsed = fulltext.get("paper", {}) if isinstance(fulltext, dict) else {}
                                            retained = structure_pmc_receipt(parsed, pmid=source_id, pmc_id=pmc_id, source_artifact_id=source_artifact["id"], source_artifact_sha256=source_artifact["sha256"], extraction_artifact_id=fulltext.get("artifact_id"))
                                            artifact = await self._store_record(call, session_id, task_id, "paper_extraction_receipt", "PMC BioC", pmc_id, retained)
                                            annotation = fulltext.get("model_extraction", {}) if isinstance(fulltext, dict) else {}
                                            if annotation.get("available"):
                                                model_annotations += len(annotation.get("annotations", []))
                                                if annotation.get("artifact_id"):
                                                    model_annotation_artifact_ids.append(annotation["artifact_id"])
                                                    artifact_ids.append(annotation["artifact_id"])
                                            pmc_processed += 1
                                        except Exception as error:
                                            warnings.append({"source_artifact_id": str(source_artifact.get("id", "")), "stage": "PMC full text; abstract fallback used", "error": type(error).__name__})
                                            structured = structure_pubmed_abstract(paper, source_artifact["id"], source_artifact["sha256"])
                                            artifact = await self._store_record(call, session_id, task_id, "paper_abstract_structure", "MedAI PubMed abstract section parser", source_id, structured)
                                            abstract_processed += 1
                                    else:
                                        structured = structure_pubmed_abstract(paper, source_artifact["id"], source_artifact["sha256"])
                                        artifact = await self._store_record(call, session_id, task_id, "paper_abstract_structure", "MedAI PubMed abstract section parser", source_id, structured)
                                        abstract_processed += 1
                                    completed_source_ids.add(source_id)
                                    if artifact.get("id"):
                                        artifact_ids.append(artifact["id"])
                                    processed += 1
                                    checkpoint.update({"stage": "literature_records_extracted", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                    await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                                    warnings.append({"source_artifact_id": str(source_artifact.get("id", "")), "error": type(error).__name__})
                            output = {"source": "PubMed", "records_structured": processed, "pmc_full_text_records": pmc_processed, "abstract_only_records": abstract_processed, "model_result_annotations": model_annotations, "model_annotation_artifact_ids": model_annotation_artifact_ids, "source_records_available": len(pubmed_artifacts), "warnings": warnings, "artifact_ids": artifact_ids, "interpretation": "Provider-labeled abstract sections and PMC annotations are retained with provenance. Optional model labels are categorical screening annotations only; no claim polarity or evidence strength is inferred."}
                            status = "needs_review" if warnings and processed == 0 else "succeeded"
                        elif task_id == "trials.discover":
                            plan_data = await _collect_artifact_pages(call, session_id, "research_objective_plan", cancellation)
                            plan_artifacts = plan_data.get("items", []) if isinstance(plan_data, dict) else []
                            objective_plan = objective_plan_from_artifacts(plan_artifacts, objective=objective, depth=session.get("depth", "standard"), session_id=session_id)
                            trials_query = objective_plan.get("clinical_trials_query", safe_objective) if objective_plan else safe_objective
                            trials = await call(
                                "medai_trials_search_clinicaltrials_gov",
                                {"session_id": session_id, "task_id": task_id, "query": trials_query, "limit": min(paper_limit, 20)},
                            )
                            records = trials if isinstance(trials, list) else trials.get("results", [])
                            for trial in records:
                                source_id = str(trial.get("nct_id") or "")
                                if not source_id or source_id in completed_source_ids:
                                    continue
                                artifact = await self._store_record(call, session_id, task_id, "clinical_trial", "ClinicalTrials.gov", source_id, trial)
                                completed_source_ids.add(source_id)
                                if artifact.get("id"):
                                    artifact_ids.append(artifact["id"])
                                checkpoint.update({"stage": "records_persisted", "query": trials_query, "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                            output = {"query": trials_query, "model_planned": objective_plan is not None, "source": "ClinicalTrials.gov", "records_found": len(records), "artifact_ids": artifact_ids, "research_only": True}
                            status = "succeeded"
                        elif task_id == "entities.resolve":
                            source_data = await _collect_artifact_pages(call, session_id, "paper_extraction_receipt", cancellation=cancellation)
                            receipts = source_data.get("items", []) if isinstance(source_data, dict) else []
                            nodes_added = 0
                            gene_lookups = 0
                            gene_exact_matches = 0
                            warnings = []
                            discovered_gene_ids: set[str] = set()
                            looked_up_gene_ids = {item.removeprefix("gene-lookup:") for item in completed_source_ids if item.startswith("gene-lookup:")}
                            for receipt_artifact in receipts:
                                if cancellation.is_set():
                                    break
                                receipt_id = str(receipt_artifact.get("id", ""))
                                if not receipt_id:
                                    continue
                                try:
                                    record = receipt_artifact.get("payload")
                                    record = json.loads(record) if isinstance(record, str) else record
                                    entities = record.get("normalized_entities", []) if isinstance(record, dict) else []
                                    ncbi_gene_ids: set[str] = set()
                                    for entity in entities:
                                        node = source_entity_node(entity, receipt_id=receipt_id, receipt_sha256=str(receipt_artifact.get("sha256", "")), pmc_id=str(record.get("pmc_id", "")))
                                        if node is None:
                                            continue
                                        if node["id"].startswith("NCBIGene:"):
                                            ncbi_gene_ids.add(node["id"].split(":", 1)[1])
                                        if receipt_id not in completed_source_ids:
                                            await call("medai_graph_add_entity", {
                                                "session_id": session_id,
                                                "task_id": task_id,
                                                "entity_id": node["id"],
                                                "entity_type": node["kind"],
                                                "label": node["label"],
                                                "metadata": node["metadata"],
                                            })
                                            nodes_added += 1
                                    discovered_gene_ids.update(ncbi_gene_ids)
                                    if receipt_id not in completed_source_ids:
                                        completed_source_ids.add(receipt_id)
                                        checkpoint.update({"stage": "source_identifiers_resolved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except Exception as error:
                                    warnings.append({"source_artifact_id": receipt_id, "error": type(error).__name__})
                            for gene_id in sorted(discovered_gene_ids - looked_up_gene_ids)[:paper_limit]:
                                if cancellation.is_set():
                                    break
                                try:
                                    lookup = await call("medai_genes_search_mygene", {"session_id": session_id, "task_id": task_id, "query": f"entrezgene:{gene_id}", "limit": 5, "species": "human"})
                                    exact = exact_mygene_identifier_matches(lookup.get("results", []) if isinstance(lookup, dict) else [], gene_id)
                                    resolution = {"source": "MyGene.info exact NCBI Gene identifier resolution", "query": f"entrezgene:{gene_id}", "ncbi_gene_id": gene_id, "exact_match_count": len(exact), "matches": exact, "provider_artifact_id": lookup.get("artifact_id") if isinstance(lookup, dict) else None, "policy": "Search results are retained by MyGene.info; only exact entrezgene identifier matches are included here."}
                                    stored = await self._store_record(call, session_id, task_id, "gene_identifier_resolution", "MyGene.info", gene_id, resolution)
                                    if stored.get("id"):
                                        artifact_ids.append(stored["id"])
                                    gene_lookups += 1
                                    gene_exact_matches += len(exact)
                                    looked_up_gene_ids.add(gene_id)
                                    completed_source_ids.add(f"gene-lookup:{gene_id}")
                                    checkpoint.update({"stage": "source_identifiers_resolved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                    await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except Exception as error:
                                    warnings.append({"gene_id": gene_id, "stage": "MyGene exact identifier lookup", "error": type(error).__name__})
                            output = {"source_receipts": len(receipts), "source_identifiers_registered": nodes_added, "exact_gene_id_lookups": gene_lookups, "exact_gene_id_matches": gene_exact_matches, "warnings": warnings, "relationships_created": 0, "interpretation": "Only BioC-provided stable identifiers are registered; MyGene records are retained only after exact NCBI Gene ID matching. No biological relationships are inferred."}
                            status = "needs_review" if warnings else "skipped" if not receipts else "succeeded"
                            if not receipts:
                                output["reason"] = "No PMC BioC extraction receipts with source-annotated entities were available."
                        elif task_id == "genes.curate":
                            source_data = await _collect_artifact_pages(call, session_id, "gene_identifier_resolution", cancellation)
                            artifacts = source_data.get("items", []) if isinstance(source_data, dict) else []
                            genes = resolved_gene_symbols(artifacts, paper_limit)
                            source_counts = {"ClinGen": 0, "GenCC": 0}
                            curation_counts = {"ClinGen": 0, "GenCC": 0}
                            graph_edges_added = 0
                            graph_warnings = []
                            warnings = []
                            if genes:
                                retrieved = await _retrieve_gene_disease_curations(call, session_id, task_id, self.worker_id, genes, cancellation, completed_source_ids, artifact_ids, checkpoint)
                                source_counts, curation_counts = retrieved["source_queries"], retrieved["source_assertions"]
                                warnings = retrieved["warnings"]
                                graph_edges_added = retrieved["graph_edges_added"]
                                graph_warnings = retrieved["graph_warnings"]
                            ontology_terms = []
                            try:
                                plan_data = await _collect_artifact_pages(call, session_id, "research_objective_plan", cancellation=cancellation)
                                plan_artifacts = plan_data.get("items", []) if isinstance(plan_data, dict) else []
                                objective_plan = objective_plan_from_artifacts(plan_artifacts, objective=objective, depth=session.get("depth", "standard"), session_id=session_id)
                                if objective_plan and isinstance(objective_plan.get("concepts"), list):
                                    ontology_terms = [term for term in objective_plan["concepts"] if isinstance(term, str)]
                            except ResearchSessionCancelled:
                                raise
                            except Exception as error:
                                warnings.append({"stage": "objective disease label planning", "error": type(error).__name__})
                            objective_disease = await _retrieve_objective_disease_evidence(call, session_id, task_id, self.worker_id, objective, cancellation, completed_source_ids, artifact_ids, checkpoint, ontology_terms)
                            if objective_disease["status"] == "needs_review":
                                warnings.append({"stage": "objective disease association", "reason": objective_disease.get("reason"), "error": objective_disease.get("error")})
                            output = {"resolved_human_genes": genes, "genes_considered": len(genes), "source_queries": source_counts, "source_assertions": curation_counts, "curation_warnings": warnings, "graph_edges_added": graph_edges_added, "graph_warnings": graph_warnings, "objective_disease_associations": objective_disease, "artifact_ids": artifact_ids, "interpretation": "ClinGen expert-panel classifications and GenCC submitter assertions remain separate, provenance-linked assertion nodes. Graph links record the identifiers and source artifact for each exact provider row; they do not combine source classifications, prove causality, or create a synthetic gene-disease consensus. Disease association retrieval accepts explicit stable identifiers or a unique OLS preferred label that occurs literally in the objective; it never uses synonyms, provider rank, or cross-ontology mapping. MedAI does not combine incomparable source records or scores."}
                            if not genes and objective_disease["status"] == "skipped":
                                output["reason"] = "No exact human gene mappings or uniquely resolved stable EFO/MONDO disease identifier were available."
                                status = "skipped"
                            else:
                                status = "needs_review" if warnings or graph_warnings or objective_disease["status"] == "needs_review" else "succeeded"
                        elif task_id == "pathways.map":
                            source_data = await _collect_artifact_pages(call, session_id, "gene_identifier_resolution", cancellation=cancellation)
                            artifacts = source_data.get("items", []) if isinstance(source_data, dict) else []
                            genes = resolved_gene_symbols(artifacts, paper_limit)
                            if not genes:
                                status = "skipped"
                                output = {"reason": "No exact MyGene.info Entrez ID to gene symbol mappings are available for pathway analysis.", "pathway_records": 0}
                            else:
                                analysis = await call("medai_pathways_analyze_reactome", {"session_id": session_id, "task_id": task_id, "genes": [item["symbol"] for item in genes], "species": "Homo sapiens", "page_size": 20})
                                artifact_id = analysis.get("artifact_id") if isinstance(analysis, dict) else None
                                if artifact_id:
                                    artifact_ids.append(artifact_id)
                                pathway_result = analysis.get("analysis", {}) if isinstance(analysis, dict) else {}
                                source_result = pathway_result.get("result", {}) if isinstance(pathway_result, dict) else {}
                                pathway_rows = source_result.get("pathways", []) if isinstance(source_result, dict) else []
                                graph_edges_added = 0
                                graph_warnings = []
                                analysis_hash = pathway_result.get("input_sha256") if isinstance(pathway_result, dict) else None
                                if artifact_id and isinstance(analysis_hash, str) and re.fullmatch(r"[0-9a-f]{64}", analysis_hash):
                                    analysis_node = f"ReactomeAnalysis:{analysis_hash}"
                                    await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": analysis_node, "entity_type": "Analysis", "label": "Reactome gene-set over-representation analysis", "metadata": {"source": "Reactome Analysis Service", "source_artifact_id": artifact_id, "input_sha256": analysis_hash, "interpretation": "computational analysis record; not a biological entity"}})
                                    for gene in genes:
                                        gene_node = f"NCBIGene:{gene['entrezgene']}"
                                        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": gene_node, "entity_type": "Gene", "label": gene["symbol"], "metadata": {"source": "MyGene.info exact Entrez identifier mapping", "source_artifact_id": gene.get("source_artifact_id"), "entrez_gene_id": gene["entrezgene"]}})
                                        await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": gene_node, "relation": "SUBMITTED_TO_REACTOME_ANALYSIS", "target": analysis_node, "evidence_ids": [], "artifact_ids": [artifact_id], "confidence": 1.0})
                                        graph_edges_added += 1
                                    for pathway in pathway_rows[:20] if isinstance(pathway_rows, list) else []:
                                        if not isinstance(pathway, dict):
                                            graph_warnings.append("Reactome pathway row was malformed")
                                            continue
                                        stable_id = pathway.get("stId")
                                        name = pathway.get("name")
                                        if not isinstance(stable_id, str) or not re.fullmatch(r"R-[A-Z]{3}-[0-9]+", stable_id) or not isinstance(name, str) or not name.strip():
                                            graph_warnings.append("Reactome pathway was missing a valid stable ID or name")
                                            continue
                                        pathway_node = f"ReactomePathway:{stable_id}"
                                        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": pathway_node, "entity_type": "Pathway", "label": name[:512], "metadata": {"source": "Reactome Analysis Service", "source_artifact_id": artifact_id, "reactome_stable_id": stable_id}})
                                        await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": analysis_node, "relation": "RETURNED_PATHWAY_CANDIDATE", "target": pathway_node, "evidence_ids": [], "artifact_ids": [artifact_id], "confidence": 1.0})
                                        graph_edges_added += 1
                                else:
                                    graph_warnings.append("Reactome analysis artifact or input digest was unavailable; no graph links were created")
                                output = {"source": "Reactome Analysis Service", "gene_inputs": genes, "input_sha256": analysis_hash, "pathway_count": len(pathway_rows), "pathways": pathway_rows, "artifact_id": artifact_id, "graph_edges_added": graph_edges_added, "graph_warnings": graph_warnings, "interpretation_limit": "Graph links record the exact gene inputs and pathways returned by Reactome for one over-representation analysis; they do not mean every submitted gene participates in every listed pathway or establish causality."}
                                status = "needs_review" if graph_warnings else "succeeded"
                                checkpoint.update({"stage": "pathway_analysis_complete", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                        elif task_id == "proteins.analyze":
                            source_data = await _collect_artifact_pages(call, session_id, "gene_identifier_resolution", cancellation=cancellation)
                            artifacts = source_data.get("items", []) if isinstance(source_data, dict) else []
                            budget_state = await call("medai_research_get_budget", {"session_id": session_id})
                            remaining_credits = int(budget_state.get("remaining", 0)) if isinstance(budget_state, dict) else 0
                            # Reserve credits for the rest of the DAG; account for the highest-cost optional SIFTS call per protein.
                            protein_limit = min(paper_limit, max(0, (remaining_credits - 10) // 9))
                            all_genes = resolved_gene_symbols(artifacts, max(1, min(paper_limit, 20)))
                            genes = [gene for gene in all_genes if f"protein-gene:{gene['entrezgene']}" not in completed_source_ids][:protein_limit]
                            if not all_genes:
                                status = "skipped"
                                output = {"reason": "No exact MyGene.info Entrez ID to gene symbol mappings are available for protein analysis.", "proteins_analyzed": 0}
                            elif protein_limit == 0 and genes == [] and any(f"protein-gene:{gene['entrezgene']}" not in completed_source_ids for gene in all_genes):
                                status = "skipped"
                                output = {"reason": "The remaining execution-credit budget is reserved for downstream research tasks.", "proteins_analyzed": 0, "remaining_credits": remaining_credits, "estimated_worst_case_credits_per_protein": 9, "reserved_downstream_credits": 10}
                            else:
                                exact_accessions: list[str] = []
                                protein_records = []
                                warnings = []
                                for gene in genes:
                                    marker = f"protein-gene:{gene['entrezgene']}"
                                    if marker in completed_source_ids or cancellation.is_set():
                                        continue
                                    try:
                                        search = await call("medai_proteins_search_uniprot", {"session_id": session_id, "task_id": task_id, "query": f"gene:{gene['symbol']}", "limit": 5, "organism_id": 9606})
                                        artifact_ids.extend(search.get("artifact_ids", []))
                                        candidates = exact_uniprot_gene_matches(search.get("results", []), gene["symbol"])
                                        if candidates:
                                            selected = candidates[0]
                                            accession = selected["accession"]
                                            exact_accessions.append(accession)
                                            record = await call("medai_proteins_get_uniprot_record", {"session_id": session_id, "task_id": task_id, "uniprot_accession": accession})
                                            if record.get("artifact_id"):
                                                artifact_ids.append(record["artifact_id"])
                                            await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": accession, "entity_type": "Protein", "label": accession, "metadata": {"source": "UniProtKB exact gene-symbol search", "gene_symbol": gene["symbol"], "gene_resolution_artifact_id": gene["source_artifact_id"], "protein_record_artifact_id": record.get("artifact_id"), "record_sha256": record.get("record_sha256")}})
                                            structure = await call("medai_proteins_get_alphafold_structure", {"session_id": session_id, "task_id": task_id, "uniprot_accession": accession})
                                            if structure.get("artifact_id"):
                                                artifact_ids.append(structure["artifact_id"])
                                            interactions = await call("medai_proteins_search_intact_interactions", {"session_id": session_id, "task_id": task_id, "uniprot_accession": accession, "limit": 25, "organism_id": 9606})
                                            if interactions.get("artifact_id"):
                                                artifact_ids.append(interactions["artifact_id"])
                                            mapping = None
                                            if record.get("experimental_structure_cross_references"):
                                                mapping = await call("medai_proteins_map_experimental_residues", {"session_id": session_id, "task_id": task_id, "uniprot_accession": accession, "max_structures": 2})
                                                if mapping.get("artifact_id"):
                                                    artifact_ids.append(mapping["artifact_id"])
                                            protein_records.append({"gene_symbol": gene["symbol"], "accession": accession, "uniprot_record_artifact_id": record.get("artifact_id"), "record_sha256": record.get("record_sha256"), "domain_region_count": len(record.get("domains_and_regions", [])), "experimental_structure_count": len(record.get("experimental_structure_cross_references", [])), "alphafold_artifact_id": structure.get("artifact_id"), "intact_artifact_id": interactions.get("artifact_id"), "intact_record_count": len(interactions.get("records", [])), "residue_mapping_artifact_id": mapping.get("artifact_id") if isinstance(mapping, dict) else None})
                                        completed_source_ids.add(marker)
                                        checkpoint.update({"stage": "protein_source_records_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                    except Exception as error:
                                        warnings.append({"gene_id": gene["entrezgene"], "symbol": gene["symbol"], "error": type(error).__name__})
                                output = {"source_genes": genes, "exact_uniprot_accessions": exact_accessions, "proteins_analyzed": len(protein_records), "records": protein_records, "warnings": warnings, "artifact_ids": artifact_ids, "remaining_credits_at_start": remaining_credits, "estimated_worst_case_credits_per_protein": 9, "reserved_downstream_credits": 10, "interpretation": "Stable protein records, predicted structures, and interaction observations are source artifacts; UniProt search does not establish a disease mechanism, and AlphaFold predictions are not experimental structures."}
                                status = "needs_review" if warnings else "succeeded"
                        elif task_id == "evidence.grade":
                            evidence_data = await call("medai_evidence_list", {"session_id": session_id})
                            evidence = evidence_data.get("items", []) if isinstance(evidence_data, dict) else []
                            papers_data = await _collect_artifact_pages(call, session_id, "paper", cancellation=cancellation)
                            paper_artifacts = papers_data.get("items", []) if isinstance(papers_data, dict) else []
                            counts = {"supports": 0, "contradicts": 0, "context": 0}
                            quality_by_label = {key: [] for key in counts}
                            sources: dict[str, int] = {}
                            publication_type_counts: dict[str, int] = {}
                            mesh_descriptor_counts: dict[str, int] = {}
                            pubmed_records = []
                            malformed = 0
                            for item in evidence:
                                polarity = item.get("polarity") if isinstance(item, dict) else None
                                quality = item.get("quality") if isinstance(item, dict) else None
                                if polarity not in counts or not is_valid_quality_annotation(quality) or not item.get("source"):
                                    malformed += 1
                                    continue
                                counts[polarity] += 1
                                quality_by_label[polarity].append(float(quality))
                                sources[item["source"]] = sources.get(item["source"], 0) + 1
                            for paper_artifact in paper_artifacts:
                                payload = paper_artifact.get("payload")
                                try:
                                    paper = json.loads(payload) if isinstance(payload, str) else payload
                                except json.JSONDecodeError:
                                    continue
                                provenance = paper.get("provenance") if isinstance(paper, dict) else None
                                if not isinstance(paper, dict) or not isinstance(provenance, dict) or provenance.get("source") != "PubMed":
                                    continue
                                pubmed_records.append(paper)
                                source_publication_types = paper.get("publication_types", [])
                                if not isinstance(source_publication_types, list):
                                    source_publication_types = []
                                source_mesh_headings = paper.get("mesh_headings", [])
                                if not isinstance(source_mesh_headings, list):
                                    source_mesh_headings = []
                                for kind in source_publication_types:
                                    if isinstance(kind, str) and kind:
                                        publication_type_counts[kind] = publication_type_counts.get(kind, 0) + 1
                                for heading in source_mesh_headings:
                                    descriptor_id = heading.get("descriptor_ui") if isinstance(heading, dict) else None
                                    if isinstance(descriptor_id, str) and descriptor_id:
                                        mesh_descriptor_counts[descriptor_id] = mesh_descriptor_counts.get(descriptor_id, 0) + 1
                            design_inventory = summarize_pubmed_design_metadata(pubmed_records)
                            pubmed_records_with_metadata = design_inventory["pubmed_records"]
                            malformed_design_metadata = design_inventory["records_with_malformed_design_metadata"]
                            malformed_related_publication_metadata = design_inventory["records_with_malformed_related_publication_metadata"]
                            if malformed:
                                status = "needs_review"
                                output = {"reason": "Evidence records with invalid or missing source annotations were not summarized.", "malformed_records": malformed, "valid_records": sum(counts.values())}
                            elif not evidence and not pubmed_records_with_metadata:
                                status = "skipped"
                                output = {"reason": "No source-linked evidence annotations or PubMed records are available for metadata inventory.", "evidence_count": 0, "pubmed_records": 0}
                            else:
                                inventory = {"counts_by_existing_polarity_annotation": counts, "mean_existing_quality_annotation_by_polarity": {key: round(sum(values) / len(values), 4) if values else None for key, values in quality_by_label.items()}, "source_counts": sources, "evidence_ids": [item["id"] for item in evidence], "pubmed_records_with_metadata": pubmed_records_with_metadata, "provider_publication_type_counts": publication_type_counts, "provider_mesh_descriptor_id_counts": mesh_descriptor_counts, "study_design_inventory": design_inventory, "interpretation": "Descriptive summary of pre-existing user/source annotations and PubMed-supplied publication metadata. It does not assign evidence polarity or independently grade study quality."}
                                stored = await call("medai_research_store_artifact", {"session_id": session_id, "task_id": task_id, "kind": "evidence_inventory", "source": "MedAI descriptive evidence inventory", "source_id": session_id, "payload": inventory})
                                output = {**inventory, "artifact_id": stored.get("id")}
                                status = "needs_review" if malformed_design_metadata or malformed_related_publication_metadata else "succeeded"
                                if malformed_design_metadata or malformed_related_publication_metadata:
                                    output["review_reason"] = "Some PubMed records contained malformed publication-type, MeSH, or related-publication metadata; see the separate inventory counters."
                        elif task_id == "contradictions.search":
                            annotation_data = await _collect_artifact_pages(call, session_id, "paper_result_annotations", cancellation=cancellation)
                            annotation_artifacts = annotation_data.get("items", []) if isinstance(annotation_data, dict) else []
                            alignment = build_claim_alignment_candidates(annotation_artifacts, session_id=session_id)
                            alignment_artifact_id = None
                            semantic_alignment_artifact_ids: list[str] = []
                            semantic_alignment_warnings: list[dict[str, str]] = []
                            semantic_links, semantic_alignment_omitted_count = _semantic_claim_screening_batch(
                                alignment["links"], completed_source_ids,
                            )
                            if alignment["links"]:
                                stored_alignment = await call("medai_research_store_artifact", {
                                    "session_id": session_id,
                                    "task_id": task_id,
                                    "kind": "paper_claim_alignment_candidates",
                                    "source": "MedAI exact-entity directional review leads",
                                    "source_id": session_id,
                                    "payload": alignment,
                                })
                                alignment_artifact_id = stored_alignment.get("id")
                                if alignment_artifact_id:
                                    artifact_ids.append(alignment_artifact_id)
                                for link in semantic_links:
                                    if cancellation.is_set():
                                        break
                                    left, right = link["sources"]
                                    pair_marker = semantic_claim_pair_marker(left, right)
                                    if pair_marker in completed_source_ids:
                                        continue
                                    try:
                                        comparison = await call("medai_papers_align_pmc_claims", {
                                            "session_id": session_id,
                                            "task_id": task_id,
                                            "annotation_artifact_a": left["artifact_id"],
                                            "candidate_id_a": left["candidate_id"],
                                            "annotation_artifact_b": right["artifact_id"],
                                            "candidate_id_b": right["candidate_id"],
                                        })
                                        if comparison.get("available") and comparison.get("artifact_id"):
                                            semantic_alignment_artifact_ids.append(comparison["artifact_id"])
                                            artifact_ids.append(comparison["artifact_id"])
                                            completed_source_ids.add(pair_marker)
                                            checkpoint.update({"stage": "semantic_claim_pairs_screened", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                            await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                        else:
                                            semantic_alignment_warnings.append({"pair_sha256": pair_marker.split(":", 1)[1], "reason": str(comparison.get("reason", "local model screening unavailable"))[:200]})
                                    except ResearchSessionCancelled:
                                        raise
                                    except Exception as error:
                                        semantic_alignment_warnings.append({"pair_sha256": pair_marker.split(":", 1)[1], "error": type(error).__name__})
                            evidence_data = await call("medai_evidence_list", {"session_id": session_id})
                            evidence = evidence_data.get("items", []) if isinstance(evidence_data, dict) else []
                            claims = select_contradiction_claims(evidence, limit=min(10, paper_limit))
                            if not claims:
                                status = "needs_review" if alignment_artifact_id or alignment["needs_review"] or semantic_alignment_omitted_count else "skipped"
                                output = {"reason": "No source-linked supporting claim annotations are available for targeted PubMed contradiction searching.", "alignment_artifact_id": alignment_artifact_id, "alignment_candidate_count": len(alignment["links"]), "alignment_invalid_annotation_artifact_count": alignment["invalid_annotation_artifact_count"], "alignment_truncated": alignment["truncated"], "semantic_alignment_omitted_count": semantic_alignment_omitted_count, "semantic_alignment_artifact_ids": semantic_alignment_artifact_ids, "semantic_alignment_warnings": semantic_alignment_warnings, "interpretation": "Directional article-pair leads are unreviewed and semantic model screenings require human review. Invalid or truncated source annotations require review. No claim-level PubMed search was possible; this is not evidence that contradictions are absent. At most ten semantic pairs are screened per task; omitted pairs remain pending review."}
                            else:
                                searched = []
                                warnings = []
                                for evidence_item in claims:
                                    if cancellation.is_set():
                                        break
                                    evidence_id = str(evidence_item.get("id", ""))
                                    marker = "contradiction-claim:" + evidence_id
                                    if marker in completed_source_ids:
                                        continue
                                    try:
                                        safe_query = build_contradiction_query(evidence_item["claim"], max_bytes=256)
                                        search = await call("medai_papers_search_pubmed", {"session_id": session_id, "task_id": task_id, "query": safe_query, "limit": min(paper_limit, 5)})
                                        records = search if isinstance(search, list) else search.get("results", [])
                                        ranking_artifact_id = search.get("ranking_artifact_id") if isinstance(search, dict) else None
                                        if ranking_artifact_id:
                                            artifact_ids.append(ranking_artifact_id)
                                        candidate_ids = []
                                        for paper in records:
                                            provenance = paper.get("provenance", {}) if isinstance(paper, dict) else {}
                                            source_id = str(provenance.get("source_id") or "")
                                            if not source_id:
                                                continue
                                            retained = {"target_evidence_id": evidence_id, "paper": paper, "interpretation": "Retrieved as a contradiction-search candidate only; no contradictory relation or evidence polarity was inferred."}
                                            artifact = await self._store_record(call, session_id, task_id, "contradiction_search_candidate", "PubMed", source_id, retained)
                                            if artifact.get("id"):
                                                artifact_ids.append(artifact["id"])
                                                candidate_ids.append(artifact["id"])
                                        completed_source_ids.add(marker)
                                        searched.append({"target_evidence_id": evidence_id, "query_sha256": hashlib.sha256(safe_query.encode()).hexdigest(), "candidate_count": len(candidate_ids), "candidate_artifact_ids": candidate_ids})
                                        checkpoint.update({"stage": "contradiction_candidates_retrieved", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": artifact_ids})
                                        await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                    except Exception as error:
                                        warnings.append({"target_evidence_id": evidence_id, "error": type(error).__name__})
                                status = "needs_review" if alignment["needs_review"] or semantic_alignment_omitted_count or (warnings and not searched) or semantic_alignment_warnings else "succeeded"
                                output = {"searched_claims": searched, "warnings": warnings, "alignment_artifact_id": alignment_artifact_id, "alignment_candidate_count": len(alignment["links"]), "alignment_invalid_annotation_artifact_count": alignment["invalid_annotation_artifact_count"], "alignment_truncated": alignment["truncated"], "semantic_alignment_omitted_count": semantic_alignment_omitted_count, "semantic_alignment_artifact_ids": semantic_alignment_artifact_ids, "semantic_alignment_warnings": semantic_alignment_warnings, "artifact_ids": artifact_ids, "interpretation": "Retrieved papers are unassessed candidates for human contradiction review. At most ten semantic pairs are screened per task; omitted pairs remain pending review. Semantic pair outputs are local-model screening labels; neither retrieval nor model labeling confirms contradiction. Invalid or truncated source annotations require review."}
                        elif task_id == "graph.build":
                            graph = await call("medai_graph_snapshot", {"session_id": session_id})
                            if not graph.get("nodes") and not graph.get("edges"):
                                status = "skipped"
                                output = {"reason": "No stable-identifier graph entities or evidence-linked relationships are available."}
                            else:
                                snapshot = {"nodes": [{key: row.get(key) for key in ("id", "kind", "label")} for row in graph.get("nodes", [])], "edges": [{key: row.get(key) for key in ("source", "relation", "target", "evidence_ids", "artifact_ids")} for row in graph.get("edges", [])], "interpretation": "Session graph snapshot contains explicitly registered stable entities and edges linked to same-session evidence annotations and/or source artifacts; no relationships were inferred."}
                                stored = await call("medai_research_store_artifact", {"session_id": session_id, "task_id": task_id, "kind": "biomedical_graph_snapshot", "source": "MedAI session graph", "source_id": session_id, "payload": snapshot})
                                output = {"node_count": len(snapshot["nodes"]), "edge_count": len(snapshot["edges"]), "artifact_id": stored.get("id"), "interpretation": snapshot["interpretation"]}
                                status = "succeeded"
                        elif task_id == "hypotheses.propose":
                            proposal = await call("medai_hypotheses_propose", {"session_id": session_id, "task_id": task_id, "limit": min(50, max(1, paper_limit * 5))})
                            literature = await _search_hypothesis_literature(call, session_id, task_id, self.worker_id, proposal, paper_limit, cancellation, completed_source_ids, artifact_ids, checkpoint)
                            output = {"candidate_count": proposal.get("candidate_count", 0), "returned_count": proposal.get("returned_count", 0), "candidate_limit_omitted_count": proposal.get("candidate_limit_omitted_count", 0), "candidate_byte_omitted_count": proposal.get("candidate_byte_omitted_count", 0), "candidate_payload_bytes": proposal.get("candidate_payload_bytes", 0), "proposal_truncated": proposal.get("truncated") is not False, "traversal_truncated": proposal.get("traversal_truncated") is True, "traversed_path_count": proposal.get("traversed_path_count", 0), "artifact_id": proposal.get("artifact_id"), "literature_comparison_artifact_id": literature.get("artifact_id"), "literature_candidates_searched": literature.get("searched_candidate_count", 0), "literature_candidates_omitted": literature.get("omitted_candidate_count", 0), "literature_search_warnings": literature.get("warnings", []), "novelty_assessment": None, "interpretation": proposal.get("interpretation", "Candidate hypotheses require scientific review."), "literature_interpretation_limit": literature.get("interpretation_limit")}
                            status = "needs_review" if _hypothesis_proposal_needs_review(proposal, literature) else "succeeded"
                        elif task_id == "hypotheses.criticize":
                            artifact_data = await _collect_artifact_pages(call, session_id, "hypothesis_candidates", cancellation=cancellation)
                            proposals = artifact_data.get("items", []) if isinstance(artifact_data, dict) else []
                            if not proposals:
                                status = "skipped"
                                output = {"reason": "No candidate hypothesis artifact is available for criticism."}
                            else:
                                proposal = proposals[-1]
                                contradiction_data = await _collect_artifact_pages(call, session_id, "contradiction_search_candidate", cancellation=cancellation)
                                contradiction_candidates = contradiction_data.get("items", []) if isinstance(contradiction_data, dict) else []
                                contradiction_artifact_ids = [row["id"] for row in contradiction_candidates if isinstance(row, dict) and isinstance(row.get("id"), str)][:50]
                                critique = await call("medai_hypotheses_criticize", {"session_id": session_id, "task_id": task_id, "proposal_artifact_id": proposal["id"], "contradiction_artifact_ids": contradiction_artifact_ids})
                                alternative_route_truncated_count = sum(
                                    review.get("alternative_graph_routes_truncated") is True
                                    for review in critique.get("reviews", []) if isinstance(review, dict)
                                )
                                output = {"proposal_artifact_id": proposal["id"], "review_count": len(critique.get("reviews", [])), "artifact_id": critique.get("artifact_id"), "contradiction_candidate_count": len(contradiction_candidates), "contradiction_candidates_supplied": len(contradiction_artifact_ids), "contradiction_candidates_truncated": len(contradiction_candidates) > len(contradiction_artifact_ids), "alternative_graph_routes_truncated_count": alternative_route_truncated_count, "interpretation": critique.get("interpretation")}
                                status = "needs_review" if _hypothesis_criticism_needs_review(critique, max(0, len(contradiction_candidates) - len(contradiction_artifact_ids))) or alternative_route_truncated_count > 0 else "succeeded"
                        elif task_id == "compounds.analyze":
                            source_data = await _collect_artifact_pages(call, session_id, cancellation=cancellation)
                            available = source_data.get("items", []) if isinstance(source_data, dict) else []
                            all_chemical_artifacts = _source_identified_compound_artifacts(available)
                            compound_limit = min(20, paper_limit)
                            extraction_receipts = _pmc_extraction_receipts(available)
                            resolution = await _retrieve_source_identified_compounds(call, session_id, task_id, self.worker_id, extraction_receipts, all_chemical_artifacts[:compound_limit], compound_limit, cancellation, completed_source_ids, artifact_ids, checkpoint)
                            unresolved = [
                                item.get("id") for item in resolution["chemical_artifacts"]
                                if _exact_pubchem_identity(item) is None
                            ]
                            status = "needs_review" if resolution["warnings"] or len(all_chemical_artifacts) > compound_limit or resolution["candidate_limit_reached"] or unresolved else "succeeded"
                            if not extraction_receipts and not all_chemical_artifacts:
                                status = "skipped"
                            output = {"chemical_identity_artifact_ids": [item.get("id") for item in resolution["chemical_artifacts"]], "source_identified_compounds": resolution["discovered"], "compound_lookup_warnings": resolution["warnings"], "existing_identity_count": len(all_chemical_artifacts), "chemical_artifacts_processed": len(resolution["chemical_artifacts"]), "compound_limit": compound_limit, "candidate_limit_reached": resolution["candidate_limit_reached"], "unresolved_identity_artifact_ids": unresolved, "interpretation": "Only stable PubChem CIDs explicitly identified in paper extraction are resolved; chemical identities are not inferred from free text."}
                        elif task_id == "drugs.investigate":
                            source_data = await _collect_artifact_pages(call, session_id, cancellation=cancellation)
                            available = source_data.get("items", []) if isinstance(source_data, dict) else []
                            all_chemical_artifacts = _source_identified_compound_artifacts(available)
                            chemical_artifact_count = len(all_chemical_artifacts)
                            compound_limit = min(20, paper_limit)
                            chemical_artifacts = all_chemical_artifacts[:compound_limit]
                            compound_lookup_warnings = []
                            candidate_limit_reached = len(all_chemical_artifacts) > compound_limit
                            discovered_compounds = []
                            resolved = []
                            rejected = []
                            pharmacology_graph_edges = 0
                            pharmacology_graph_warnings = []
                            for item in chemical_artifacts:
                                exact_identity = _exact_pubchem_identity(item)
                                if exact_identity is None:
                                    rejected.append(item.get("id"))
                                    continue
                                _payload, match = exact_identity
                                inchikey = match["inchikey"]
                                result = await call("medai_drug_profile_chembl", {"session_id": session_id, "task_id": task_id, "inchikey": inchikey})
                                profile_artifact_id = result.get("artifact_id")
                                graph_target_count = 0
                                if isinstance(profile_artifact_id, str) and profile_artifact_id:
                                    try:
                                        compound_node = f"InChIKey:{inchikey}"
                                        await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": compound_node, "entity_type": "Compound", "label": str(match.get("title") or inchikey)[:512], "metadata": {"source": item.get("source"), "chemical_identity_artifact_id": item.get("id"), "inchikey": inchikey}})
                                        for mechanism in result.get("mechanisms", []) if isinstance(result.get("mechanisms"), list) else []:
                                            record = mechanism.get("record") if isinstance(mechanism, dict) else None
                                            target_id = record.get("target_chembl_id") if isinstance(record, dict) else None
                                            if not isinstance(target_id, str) or not re.fullmatch(r"CHEMBL[0-9]+", target_id):
                                                continue
                                            target_node = f"ChEMBLTarget:{target_id}"
                                            await call("medai_graph_add_entity", {"session_id": session_id, "task_id": task_id, "entity_id": target_node, "entity_type": "Target", "label": target_id, "metadata": {"source": "ChEMBL web services", "pharmacology_artifact_id": profile_artifact_id, "target_chembl_id": target_id}})
                                            await call("medai_graph_link_evidence", {"session_id": session_id, "task_id": task_id, "source": compound_node, "relation": "CHEMBL_PROVIDER_REPORTS_MECHANISM_TARGET", "target": target_node, "evidence_ids": [], "artifact_ids": [profile_artifact_id], "confidence": 1.0})
                                            graph_target_count += 1
                                            pharmacology_graph_edges += 1
                                    except ResearchSessionCancelled:
                                        raise
                                    except Exception as error:
                                        pharmacology_graph_warnings.append({"chemical_artifact_id": item.get("id"), "error": type(error).__name__})
                                else:
                                    pharmacology_graph_warnings.append({"chemical_artifact_id": item.get("id"), "error": "pharmacology profile artifact was not retained"})
                                resolved.append({"chemical_artifact_id": item.get("id"), "inchikey": inchikey, "pharmacology_artifact_id": profile_artifact_id, "parent_chembl_ids": result.get("parent_chembl_ids", []), "mechanism_count": len(result.get("mechanisms", [])), "indication_count": len(result.get("indications", [])), "warning_count": len(result.get("warnings", [])), "graph_target_links": graph_target_count, "interpretation": result.get("interpretation")})
                            if resolved:
                                status = "needs_review" if pharmacology_graph_warnings or chemical_artifact_count > compound_limit or candidate_limit_reached or rejected else "succeeded"
                                output = {"profiles": resolved, "pharmacology_graph_edges": pharmacology_graph_edges, "pharmacology_graph_warnings": pharmacology_graph_warnings, "chemical_artifact_count": chemical_artifact_count, "chemical_artifacts_processed": len(chemical_artifacts), "compound_limit": compound_limit, "candidate_limit_reached": candidate_limit_reached, "rejected_ambiguous_or_unresolved_artifact_ids": rejected, "research_only": True, "interpretation": "ChEMBL target links report source-recorded mechanism annotations; they do not establish therapeutic effect, clinical efficacy, or drug-drug interaction."}
                            else:
                                status = "needs_review" if chemical_artifacts else "skipped"
                                output = {"reason": "No unambiguous PubChem identity with an InChIKey was available." if chemical_artifacts else "No source-identified chemical identity artifact is available.", "chemical_artifact_ids": [item.get("id") for item in chemical_artifacts], "interpretation": "MedAI does not infer chemical identities from free-text mentions or disease names."}
                        elif task_id == "genomics.analyze":
                            source_data = await _collect_artifact_pages(call, session_id, cancellation=cancellation)
                            available = source_data.get("items", []) if isinstance(source_data, dict) else []
                            inputs = [item for item in available if item.get("kind") in {"genomic_dataset", "vcf_dataset"}]
                            processed_inputs = []
                            rejected_inputs = []
                            receipt_artifact_ids = []
                            for item in inputs:
                                if cancellation.is_set():
                                    break
                                artifact_id = item.get("id")
                                source_id = item.get("source_id")
                                marker = f"genomic-dataset:{artifact_id}"
                                if not isinstance(artifact_id, str) or not isinstance(source_id, str):
                                    continue
                                try:
                                    payload = item.get("payload")
                                    if isinstance(payload, str):
                                        payload = json.loads(payload)
                                    if item.get("source") != "MedAI local VCF aggregate receipt; variant and sample data remain local":
                                        raise ValueError("unsupported genomic dataset source; submit a privacy-minimized local VCF receipt through the authenticated endpoint")
                                    summary = validate_local_vcf_summary(payload, expected_source_id=source_id)
                                    retained = await self._store_record(call, session_id, task_id, "genomics_analysis_receipt", "MedAI VCF analysis summary", source_id, {**summary, "source_artifact_id": artifact_id, "source_artifact_sha256": item.get("sha256")})
                                    if retained.get("id"):
                                        receipt_artifact_ids.append(retained["id"])
                                    processed_inputs.append({"source_artifact_id": artifact_id, "source_vcf_sha256": source_id, "genome_build": summary["genome_build"], "alternate_alleles_emitted": summary["alternate_alleles_emitted"], "variant_class_counts": summary["variant_class_counts"], "reference_validated": summary["reference_validated"], "analysis_receipt_id": retained.get("id")})
                                    completed_source_ids.add(marker)
                                    checkpoint.update({"stage": "local_genomic_reports_summarized", "completed_source_ids": sorted(completed_source_ids), "artifact_ids": [*artifact_ids, *receipt_artifact_ids]})
                                    await call("medai_research_save_checkpoint", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "checkpoint": checkpoint})
                                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                                    rejected_inputs.append({"source_artifact_id": artifact_id, "error": type(error).__name__})
                            if processed_inputs:
                                status = "needs_review" if rejected_inputs else "succeeded"
                                output = {"source": "authenticated researcher-submitted local VCF report", "datasets_analyzed": len(processed_inputs), "datasets": processed_inputs, "rejected_inputs": rejected_inputs, "analysis_receipt_artifact_ids": receipt_artifact_ids, "remote_variant_sources_queried": False, "clinical_interpretation": False, "interpretation_limit": "The worker summarizes a submitted local report, not the underlying VCF. It does not repeat the local parser, verify file-to-report correspondence, query remote variant services, or assess pathogenicity."}
                            else:
                                status = "needs_review" if rejected_inputs else "skipped"
                                output = {"reason": "No valid authenticated local VCF analysis report was available." if inputs else "No compatible user-supplied dataset is present in this research session.", "compatible_input_artifact_ids": [item.get("id") for item in inputs], "rejected_inputs": rejected_inputs, "remote_variant_sources_queried": False}
                        elif task_id == "skills.learn":
                            output = await call("medai_skills_learn_research_sequence", {"session_id": session_id, "task_id": task_id})
                            status = "succeeded"
                        elif task_id == "benchmark.evaluate":
                            artifact_data = await _collect_artifact_pages(call, session_id, cancellation=cancellation)
                            artifact_rows = artifact_data.get("items", []) if isinstance(artifact_data, dict) else []
                            domain_inputs = _biomedical_domain_benchmark_inputs(artifact_rows, session_id)
                            quality_inputs = _evidence_quality_benchmark_inputs(artifact_rows, session_id)
                            benchmark_inputs = domain_inputs + quality_inputs
                            if not benchmark_inputs:
                                status = "needs_review"
                                output = {"reason": "Attach exactly one source-digest-bound, researcher-adjudicated biomedical domain or double-reviewed ordinal evidence-quality benchmark input before evaluation.", "input_artifacts_found": 0}
                            elif len(benchmark_inputs) != 1:
                                status = "needs_review"
                                output = {"reason": "Multiple biomedical benchmark inputs are present; select one explicitly before evaluation.", "input_artifact_ids": [item.get("id") for item in benchmark_inputs]}
                            else:
                                benchmark_input = benchmark_inputs[0]
                                artifact_id = benchmark_input.get("id")
                                if not isinstance(artifact_id, str) or not artifact_id:
                                    raise ValueError("biomedical benchmark input has no stable artifact ID")
                                tool_name = (
                                    "medai_biomedical_domain_benchmark_evaluate"
                                    if benchmark_input.get("kind") == DOMAIN_BENCHMARK_ARTIFACT_KIND
                                    else "medai_evidence_quality_benchmark_evaluate"
                                )
                                result = await call(tool_name, {"session_id": session_id, "task_id": task_id, "input_artifact_id": artifact_id})
                                output = {
                                    "input_artifact_id": artifact_id,
                                    "input_artifact_sha256": benchmark_input.get("sha256"),
                                    **result,
                                }
                                status = "succeeded" if result.get("artifact_id") and result.get("benchmark_sha256") else "needs_review"
                        elif task_id == "report.reproduce":
                            plan = await call("medai_research_get_plan", {"session_id": session_id})
                            artifacts_data = await _collect_artifact_pages(call, session_id, cancellation=cancellation)
                            evidence_data = await call("medai_evidence_list", {"session_id": session_id})
                            graph = await call("medai_graph_snapshot", {"session_id": session_id})
                            citation_network = await call("medai_papers_citation_network", {"session_id": session_id})
                            artifact_rows = artifacts_data.get("items", []) if isinstance(artifacts_data, dict) else []
                            evidence_rows = evidence_data.get("items", []) if isinstance(evidence_data, dict) else []
                            workflow = build_research_workflow_proposal(session, plan)
                            workflow_artifact = await call("medai_research_store_artifact", {"session_id": session_id, "task_id": task_id, "kind": "research_workflow_proposal", "source": "MedAI versioned research workflow generator", "source_id": workflow["workflow_id"] + "@" + workflow["version"], "payload": workflow})
                            if not isinstance(workflow_artifact, dict) or not isinstance(workflow_artifact.get("id"), str) or not isinstance(workflow_artifact.get("sha256"), str):
                                raise ValueError("research workflow proposal was not retained with a content hash")
                            artifact_rows.append(workflow_artifact)
                            citation_links = citation_network.get("items", []) if isinstance(citation_network, dict) else []
                            bundle = build_reproducibility_bundle(session, plan, artifact_rows, graph, evidence_rows, citation_links)
                            stored = await call("medai_research_store_artifact", {"session_id": session_id, "task_id": task_id, "kind": "research_reproducibility_bundle", "source": "MedAI reproducibility engine", "source_id": session_id, "payload": bundle})
                            output = {"artifact_id": stored.get("id"), "workflow_artifact_id": workflow_artifact["id"], "workflow_id": workflow["workflow_id"], "workflow_version": workflow["version"], "workflow_topology_sha256": workflow["topology_sha256"], "bundle_sha256": bundle["bundle_sha256"], "report_markdown": bundle["report_markdown"], "source_artifact_count": len(bundle["artifacts"]), "evidence_count": len(bundle["evidence_annotations"]), "graph_node_count": len(bundle["graph"]["nodes"]), "graph_edge_count": len(bundle["graph"]["edges"]), "citation_link_count": len(bundle["citation_graph"]["edges"]), "interpretation": "Versioned workflow topology proposal, reproducibility bundle, and report; the workflow remains disabled pending review and provides no clinical advice or independent scientific validation."}
                            status = "succeeded" if stored.get("id") else "needs_review"
                        else:
                            status = "needs_review"
                            output = {"reason": "No validated scientific executor is registered for this task.", "task_id": task_id}
                        if lease_lost:
                            raise RuntimeError(lease_lost[0])
                        if active_policy_violations:
                            raise PermissionError(
                                "specialist MCP policy violation(s): "
                                + ", ".join(active_policy_violations)
                            )
                        if cancellation.is_set():
                            status = "skipped"
                            output = {"reason": "Session cancellation was requested; the active bounded operation has completed.", "checkpoint_stage": checkpoint.get("stage")}
                        output = {**output, "specialist_agent": assignment, "research_plan_sha256": coordination["plan_sha256"]}
                        await call(
                            "medai_research_update_task",
                            {"session_id": session_id, "task_id": task_id, "status": status, "output": json.dumps(output, ensure_ascii=False), "worker_id": self.worker_id},
                        )
                        summary[{"succeeded": "succeeded", "needs_review": "needs_review", "skipped": "cancelled"}[status]].append(task_id)
                    except Exception as error:
                        message = f"{type(error).__name__}: {error}"[:4000]
                        if not cancellation.is_set():
                            try:
                                control = await call("medai_research_get_control", {"session_id": session_id})
                                if control.get("cancellation_requested"):
                                    cancellation.set()
                            except Exception:
                                pass
                        try:
                            if cancellation.is_set():
                                await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "skipped", "output": json.dumps({"reason": "cancelled", "checkpoint_stage": checkpoint.get("stage")}), "worker_id": self.worker_id})
                                summary["cancelled"].append(task_id)
                            elif _is_budget_exhaustion_error(error) and not lease_lost:
                                reason = "The specialist or research session reached its configured execution credit limit."
                                await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "needs_review", "output": json.dumps({"reason": reason, "specialist_agent": assignment, "research_plan_sha256": coordination["plan_sha256"]}, ensure_ascii=False), "worker_id": self.worker_id})
                                summary["needs_review"].append(task_id)
                            elif isinstance(error, ArtifactCollectionLimitError) and not lease_lost:
                                await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "needs_review", "output": json.dumps({"reason": "The full session artifact set exceeds bounded worker retrieval limits; no incomplete artifact subset was analyzed.", "detail": message}, ensure_ascii=False), "worker_id": self.worker_id})
                                summary["needs_review"].append(task_id)
                            elif not lease_lost:
                                await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "failed", "error": message, "worker_id": self.worker_id})
                                if int(task.get("attempts", 1)) < 5:
                                    await call("medai_research_update_task", {"session_id": session_id, "task_id": task_id, "status": "ready"})
                        except Exception as update_error:
                            # A failed task transition can leave a lease active or
                            # its retry state uncertain. Surface that separately
                            # without echoing provider/API error bodies into logs.
                            _record_control_plane_update_failure(summary, task_id, error, update_error)
                        if not cancellation.is_set() and not isinstance(error, ArtifactCollectionLimitError):
                            summary["failed"].append({"task_id": task_id, "error": message})
                    finally:
                        heartbeat.cancel()
                        await asyncio.gather(heartbeat, return_exceptions=True)
                        active_cancellation = None
                        active_agent_id = None
                return summary

    async def _maintain_lease(self, call: Any, session_id: str, task_id: str, cancellation: asyncio.Event, lease_lost: list[str]) -> None:
        """Renew long tasks and poll cancellation while a handler is awaiting I/O."""
        interval = max(1, min(30, self.lease_seconds / 3))
        try:
            while True:
                control = await call("medai_research_get_control", {"session_id": session_id})
                if control.get("cancellation_requested"):
                    cancellation.set()
                    return
                await call("medai_research_renew_task_lease", {"session_id": session_id, "task_id": task_id, "worker_id": self.worker_id, "lease_seconds": self.lease_seconds})
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            lease_lost.append(f"task lease could not be renewed: {type(error).__name__}: {error}")

    @staticmethod
    async def _store_record(call: Any, session_id: str, task_id: str, kind: str, source: str, source_id: str, record: dict) -> dict:
        raw = _json_bytes(record)
        retained: dict[str, Any] = record
        if len(raw) > 240_000:
            retained = {
                "source_record_omitted_for_size": True,
                "source_record_sha256": hashlib.sha256(raw).hexdigest(),
                "source_record_bytes": len(raw),
                "source_id": source_id,
            }
        return await call(
            "medai_research_store_artifact",
            {"session_id": session_id, "task_id": task_id, "kind": kind, "source": source, "source_id": source_id, "payload": retained},
        )


async def execute_session(session_id: str, max_tasks: int = 32, paper_limit: int | None = None, agent_id: str | None = None) -> dict:
    url, transport_headers = _worker_mcp_connection()
    worker = ResearchWorker(
        url,
        worker_id=os.getenv("MEDAI_WORKER_ID"),
        lease_seconds=int(os.getenv("MEDAI_WORKER_LEASE_SECONDS", "900")),
        agent_id=agent_id,
        agent_api_token=os.getenv("MEDAI_SPECIALIST_API_TOKEN"),
        transport_headers=transport_headers,
    )
    return await worker.execute(session_id, max_tasks=max_tasks, paper_limit=paper_limit)


async def run_worker(poll_seconds: int = 30, max_tasks: int = 32, paper_limit: int | None = None, watch: bool = True, agent_id: str | None = None) -> None:
    if not 1 <= poll_seconds <= 3600:
        raise ValueError("poll_seconds must be 1..3600")
    if not 1 <= max_tasks <= 100 or (paper_limit is not None and (isinstance(paper_limit, bool) or not isinstance(paper_limit, int) or not 1 <= paper_limit <= 20)):
        raise ValueError("max_tasks must be 1..100 and paper_limit, when specified, must be 1..20")
    if agent_id is not None and agent_id not in AGENTS:
        raise ValueError(f"unknown scientific agent ID: {agent_id}")
    specialist_meta = _specialist_meta(agent_id, os.getenv("MEDAI_SPECIALIST_API_TOKEN"))
    url, transport_headers = _worker_mcp_connection()
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    trial_watch_worker_id = os.getenv("MEDAI_TRIAL_WATCH_WORKER_ID") or f"medai-watch-{uuid4().hex}"
    literature_watch_worker_id = os.getenv("MEDAI_LITERATURE_WATCH_WORKER_ID") or f"medai-literature-watch-{uuid4().hex}"

    while True:
        available: list[str] = []
        async with streamable_http_client(url, headers=transport_headers or None) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                await client.initialize()
                sessions = _tool_value(await client.call_tool("medai_research_list_sessions", arguments={}, meta=specialist_meta))
                for row in sessions.get("items", []) if isinstance(sessions, dict) else []:
                    if row.get("status") not in {"planned", "running", "needs_review", "cancelling"}:
                        continue
                    plan = _tool_value(
                        await client.call_tool("medai_research_get_plan", arguments={"session_id": row["id"]}, meta=specialist_meta)
                    )
                    now = int(time.time())
                    allowed_task_ids = {task_id for task_id, specialist in AGENT_BY_TASK.items() if agent_id is None or specialist == agent_id}
                    if any(
                        task.get("status") == "ready"
                        or (
                            task.get("status") == "running"
                            and task.get("lease_until") is not None
                            and task["lease_until"] <= now
                        )
                        for task in plan.get("tasks", []) if task.get("id") in allowed_task_ids
                    ):
                        available.append(row["id"])
                try:
                    if agent_id is not None:
                        raise LookupError("scheduled source watches are owned by the general worker")
                    claimed_watches = _tool_value(await client.call_tool(
                        "medai_trials_watch_claim_due",
                        arguments={"worker_id": trial_watch_worker_id, "limit": 5, "lease_seconds": 900},
                    ))
                    for trial_watch in claimed_watches.get("items", []) if isinstance(claimed_watches, dict) else []:
                        try:
                            outcome = _tool_value(await client.call_tool(
                                "medai_trials_watch_refresh",
                                arguments={"session_id": trial_watch["session_id"], "nct_id": trial_watch["nct_id"], "worker_id": trial_watch_worker_id, "last_snapshot": trial_watch.get("last_snapshot")},
                            ))
                            print(json.dumps({"trial_watch": trial_watch["nct_id"], "session_id": trial_watch["session_id"], "changed": (outcome.get("comparison") or {}).get("changed"), "event_id": (outcome.get("event") or {}).get("id")}, ensure_ascii=False), flush=True)
                        except Exception as error:
                            print(json.dumps({"trial_watch": trial_watch.get("nct_id"), "session_id": trial_watch.get("session_id"), "error": f"{type(error).__name__}: {error}"}), flush=True)
                except LookupError:
                    pass
                except Exception as error:
                    print(json.dumps({"trial_watch_scheduler_error": f"{type(error).__name__}: {error}"}), flush=True)
                try:
                    if agent_id is not None:
                        raise LookupError("scheduled source watches are owned by the general worker")
                    due_literature = _tool_value(await client.call_tool(
                        "medai_literature_watch_claim_due",
                        arguments={"worker_id": literature_watch_worker_id, "limit": 5, "lease_seconds": 900},
                    ))
                    for literature_watch in due_literature.get("items", []) if isinstance(due_literature, dict) else []:
                        try:
                            last_pmids = _literature_watch_pmids(literature_watch.get("last_pmids"))
                            pending_metadata_pmids = _literature_watch_pmids(
                                literature_watch.get("pending_metadata_pmids")
                            )
                            outcome = _tool_value(await client.call_tool(
                                "medai_literature_watch_refresh",
                                arguments={
                                    "session_id": literature_watch["session_id"],
                                    "watch_id": literature_watch["watch_id"],
                                    "query": literature_watch["query"],
                                    "interval_seconds": literature_watch["interval_seconds"],
                                    "worker_id": literature_watch_worker_id,
                                    "last_pmids": last_pmids,
                                    "pending_metadata_pmids": pending_metadata_pmids,
                                },
                            ))
                            print(json.dumps({"literature_watch": literature_watch["watch_id"], "session_id": literature_watch["session_id"], "new_pmids": outcome.get("new_pmids"), "retained_record_count": outcome.get("retained_record_count"), "metadata_pending_count": outcome.get("metadata_pending_count"), "metadata_overflow_count": outcome.get("metadata_overflow_count"), "incomplete": outcome.get("incomplete")}, ensure_ascii=False), flush=True)
                        except Exception as error:
                            print(json.dumps({"literature_watch": literature_watch.get("watch_id"), "session_id": literature_watch.get("session_id"), "error": f"{type(error).__name__}: {error}"}), flush=True)
                except LookupError:
                    pass
                except Exception as error:
                    print(json.dumps({"literature_watch_scheduler_error": f"{type(error).__name__}: {error}"}), flush=True)
        for session_id in available:
            try:
                result = await execute_session(session_id, max_tasks=max_tasks, paper_limit=paper_limit, agent_id=agent_id)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as error:
                print(json.dumps({"session_id": session_id, "error": f"{type(error).__name__}: {error}"}), flush=True)
        if not watch:
            return
        await asyncio.sleep(poll_seconds)
