"""MedAI MCP tools. Scientific writes remain explicit and source linked."""
import os
import asyncio
import json
import hashlib
import re
from pathlib import Path
import threading
from typing import Any
from urllib.parse import quote
import httpx
from medai_science.api_pagination import read_evidence_pages as _read_evidence_pages
from mcp.server.fastmcp import FastMCP
from medai_science.sources.pubmed import PubMedClient
from medai_science.sources.pubmed_search import validate_pmid_snapshot
from medai_science.sources.clinical_trials import ClinicalTrialsClient, compare_study_snapshots, latest_retained_study_snapshot
from medai_science.sources.alphafold import AlphaFoldClient
from medai_science.structure_analysis import summarize_pdb
from medai_science.sources.mygene import MyGeneClient
from medai_science.sources.clinvar import ClinVarClient
from medai_science.sources.dbsnp import DbSNPClient
from medai_science.sources.gnomad import GnomADClient
from medai_science.sources.ensembl_vep import EnsemblVEPClient
from medai_science.sources.panelapp import PanelAppClient
from medai_science.sources.gencc import GenCCClient
from medai_science.sources.clingen_gene_validity import ClinGenGeneValidityClient
from medai_science.sources.orphadata import OrphadataClient
from medai_science.sources.uniprot import UniProtClient
from medai_science.sources.intact import IntActClient, build_retention_payload as build_intact_retention_payload
from medai_science.sources.sifts import SIFTSClient
from medai_science.sources.reactome import ReactomeClient
from medai_science.sources.gene_disease import GeneDiseaseClient, reconcile_gene_disease_sources
from medai_science.sources.pmc_bioc import PMCBioCClient
from medai_science.paper_understanding import build_extraction_retention_receipt
from medai_science.sources.pubchem import PubChemClient
from medai_science.sources.chembl import ChEMBLClient
from medai_science.sources.openfda_labels import OpenFDALabelClient
from medai_science.sources.dailymed_labels import DailyMedLabelClient, compare_label_distributions
from medai_science.sources.drug_interaction_literature import build_interaction_evidence_query, candidate_retention_receipt, characterize_interaction_candidate, retain_candidate_paper
from medai_science.sources.crossref import CrossrefClient, build_retention_payload, build_unresolved_reference_query, select_retained_reference
from medai_science.sources.preprint import PreprintClient
from medai_science.sources.ontology import OntologyClient
from medai_science.sources.ontology_mapping import OntologyMappingClient
from medai_science.cancellation import ResearchSessionCancelled, run_cancel_aware, run_cancel_aware_thread
from medai_science.sources.safety import ensure_public_research_query
from medai_science.hypotheses import (
    propose_candidates,
    critique_candidate,
    link_contradiction_search_candidates,
    validate_hypothesis_proposal_artifact,
)
from medai_science.capabilities import LocalSentenceTransformer, SemanticCapabilityIndex, apply_authoritative_lifecycle, load_amcp_catalog
from medai_science.capabilities import runtime_capability_statuses
from medai_science.capability_lifecycle import fetch_approved_lifecycle, fetch_capability_registry
from medai_science.generated_execution import resolve_production_generated_package, require_exact_active_generated_route
from medai_science.evaluation import evaluate_biomedical_domain_benchmark, load_pinned_capability_calibration
from medai_science.benchmark_artifacts import DOMAIN_BENCHMARK_ARTIFACT_KIND, DOMAIN_BENCHMARK_ARTIFACT_SOURCE, EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND, EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE, evaluate_evidence_quality_benchmark_bytes
from medai_science.agentmesh_control import AgentMeshControlPlaneClient
from medai_science.amcp_generation import build_generated_capability_amcp_package
from medai_science.code_generation import generate_python_capability, generate_r_capability, validate_python_capability_request
from medai_science.capability_evolution import run_paired_capability_benchmark
from medai_science.provenance_signing import sign_capability_proposal, verify_capability_proposal_signature
from medai_science.skill_learning import learn_research_skill_batch, persist_skill_candidate, read_skill_index, tenant_skill_registry_dir, write_skill_index
from medai_science.skill_export import read_verified_skill, render_agentmesh_skill_review_package
from medai_science.llm import JSONModelProvider, configured_model_provider
from medai_science.budgeted_model import SessionBudgetedJSONProvider
from medai_science.scientific_agents import AGENT_BY_TASK
from medai_science.research_planning import plan_research_objective as _plan_research_objective
from medai_science.research_criticism import generate_critic_leads
from medai_science.structured_extraction import extract_local_result_annotations, extraction_batch_count, license_policy_allows_model_processing, provider_is_local
from medai_science.claim_semantics import claim_context_receipt, compare_claim_pair
from medai_science.claim_alignment import decode_verified_annotation_artifact, validate_semantic_claim_pair
from medai_science.drug_interaction_review import screen_sentence_candidates, screen_drug_label_candidates, rank_candidate_pairs
from medai_science.drug_interaction_fulltext import build_interaction_fulltext_screen, rehydrate_interaction_fulltext_candidates, validate_interaction_fulltext_binding
from medai_science.literature_reranking import rerank_pubmed_titles
from medai_science.literature import omit_pubmed_abstract_text
from medai_science.sandbox import DockerPythonSandbox, SandboxError, capability_validation_request_hash, validate_capability_run_request, validate_python_source
from medai_science.r_sandbox import DockerRSandbox, validate_r_source
from medai_science.model_usage import ModelUsageBudgetError, ModelUsageError, meter_model_usage
from medai_science.compute_usage import DEFAULT_CPU_SECONDS_PER_CREDIT, meter_cpu_time
from medai_science.mcp_auth import API, AuthenticatedMCPTransportASGIMiddleware, SpecialistCredentialASGIMiddleware, api_headers as _api_headers

mcp = FastMCP("MedAI Biomedical Research")
_semantic_index = None
_semantic_index_config = None
_semantic_lock = threading.RLock()

async def _tenant_skills_dir() -> Path:
    """Resolve the Skill store from the authenticated API principal, never a tool argument."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=10) as client:
        response = await client.get("/v1/auth/whoami")
        response.raise_for_status()
    identity = response.json()
    tenant_id = identity.get("tenant_id") if isinstance(identity, dict) else None
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("authenticated API identity did not include a tenant")
    return tenant_skill_registry_dir(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills"), tenant_id)

async def _run_cancel_aware(session_id: str, operation: Any, *, poll_seconds: float = 1.0) -> Any:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=3) as control_client:
        async def cancellation_check(active_session: str) -> bool:
            response = await control_client.get(f"/v1/research/{active_session}/control")
            response.raise_for_status()
            return bool(response.json().get("cancellation_requested"))

        return await run_cancel_aware(
            session_id,
            operation,
            cancellation_check=cancellation_check,
            poll_seconds=poll_seconds,
        )

async def _run_cancel_aware_thread(session_id: str, function: Any, *args: Any, poll_seconds: float = 1.0, **kwargs: Any) -> Any:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=3) as control_client:
        async def cancellation_check(active_session: str) -> bool:
            response = await control_client.get(f"/v1/research/{active_session}/control")
            response.raise_for_status()
            return bool(response.json().get("cancellation_requested"))

        return await run_cancel_aware_thread(
            session_id,
            function,
            *args,
            cancellation_check=cancellation_check,
            poll_seconds=poll_seconds,
            **kwargs,
        )

async def _charge(session_id: str, task_id: str, credits: int, reason: str) -> None:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/budget/consume",
            json={"task_id": task_id, "credits": credits, "reason": reason},
        )
        if response.status_code == 429 and "specialist credit budget exhausted" in response.text.lower():
            raise RuntimeError("specialist credit budget exhausted; research task requires review")
        if response.status_code == 402 and "research session execution budget exhausted" in response.text.lower():
            raise RuntimeError("research session execution budget exhausted; research task requires review")
        response.raise_for_status()

def _model_provider_for_task(task_id: str, session_id: str) -> JSONModelProvider | None:
    """Select model policy from the registered specialist owning this DAG step."""
    provider = configured_model_provider(AGENT_BY_TASK.get(task_id))
    return SessionBudgetedJSONProvider(provider, session_id, task_id) if provider is not None else None


async def _charge_model_usage(session_id: str, task_id: str, provider: str, model: str, usage: dict[str, int]) -> dict[str, int | None]:
    # SessionBudgetedJSONProvider reserved before the request and settled this receipt.
    return meter_model_usage(usage)

@mcp.tool(name="medai_research_create_session")
async def create_research_session(objective: str, depth: str = "standard", budget: int = 100) -> dict:
    """Create a durable computational research session. Does not claim to run the investigation."""
    if not objective.strip() or len(objective) > 2000:
        raise ValueError("objective must contain 1..2000 characters")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research", json={"objective": objective, "depth": depth, "budget": budget})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_plan_objective")
async def plan_objective_with_model(session_id: str, task_id: str, objective: str, depth: str = "standard") -> dict:
    """Plan source queries and a validated AgentMesh task scope; make no scientific findings."""
    provider = _model_provider_for_task(task_id, session_id)
    if provider is None:
        return {"available": False, "reason": "No MEDAI_MODEL_BASE_URL and MEDAI_MODEL_NAME are configured; deterministic planning remains available."}
    await _charge(session_id, task_id, 1, "configured model-assisted objective query planning")
    completion = await _run_cancel_aware(session_id, _plan_research_objective(provider, objective, depth))
    usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
    payload = {
        "schema_version": "2",
        "objective_sha256": hashlib.sha256(objective.strip().encode()).hexdigest(),
        "depth": depth,
        "provider": completion.provider,
        "model": completion.model,
        "max_output_tokens": completion.max_output_tokens,
        "prompt_sha256": completion.prompt_sha256,
        "response_sha256": completion.response_sha256,
        "usage": completion.usage,
        "usage_metering": usage_metering,
        "planning": completion.value,
        "interpretation_limit": "Generated query plan only; not a statement of scientific evidence or results.",
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "research_objective_plan", "source": "MedAI configured model planner", "source_id": payload["objective_sha256"], "payload": payload})
        response.raise_for_status()
        artifact = response.json()
    return {"available": True, "planning": completion.value, "model": completion.model, "provider": completion.provider, "max_output_tokens": completion.max_output_tokens, "prompt_sha256": completion.prompt_sha256, "response_sha256": completion.response_sha256, "usage": completion.usage, "usage_metering": usage_metering, "artifact_id": artifact["id"], "charged_credits": 1 + int(usage_metering["credits"] or 0)}

@mcp.tool(name="medai_research_list_sessions")
async def list_research_sessions() -> dict:
    """List persistent research session metadata."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_get_plan")
async def get_research_plan(session_id: str) -> dict:
    """Read task dependencies, current states, and the immutable transition history."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/plan")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_get_control")
async def get_research_control(session_id: str) -> dict:
    """Check durable cooperative cancellation state for a research session."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/control")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_cancel_session")
async def cancel_research_session(session_id: str) -> dict:
    """Request audited cooperative cancellation and skip all queued DAG work."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(f"/v1/research/{session_id}/control")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_claim_tasks")
async def claim_research_tasks(session_id: str, worker_id: str, limit: int = 1, lease_seconds: int = 300, task_ids: list[str] | None = None) -> dict:
    """Atomically lease runnable tasks, optionally restricting a separately scheduled specialist actor."""
    if not worker_id.strip() or not 1 <= limit <= 16 or not 5 <= lease_seconds <= 3600:
        raise ValueError("worker_id is required, limit must be 1..16, and lease_seconds must be 5..3600")
    if task_ids is not None and (not 1 <= len(task_ids) <= 100 or len(set(task_ids)) != len(task_ids)):
        raise ValueError("task_ids filter must contain 1..100 unique task IDs")
    payload: dict[str, Any] = {"worker_id": worker_id, "limit": limit, "lease_seconds": lease_seconds}
    if task_ids is not None:
        payload["task_ids"] = task_ids
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/tasks/claim",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_apply_task_scope")
async def apply_research_task_scope(session_id: str, worker_id: str, plan_artifact_id: str) -> dict:
    """Atomically skip unselected queued DAG tasks using the verified model plan, before they consume task credits."""
    if not worker_id.strip() or not plan_artifact_id.strip():
        raise ValueError("worker_id and plan_artifact_id are required")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            f"/v1/research/{session_id}/tasks/scope",
            json={"worker_id": worker_id, "plan_artifact_id": plan_artifact_id},
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_renew_task_lease")
async def renew_research_task_lease(session_id: str, task_id: str, worker_id: str, lease_seconds: int = 300) -> dict:
    """Extend a still-live task lease owned by this worker."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/tasks/" + task_id + "/lease",
            json={"worker_id": worker_id, "lease_seconds": lease_seconds},
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_save_checkpoint")
async def save_research_task_checkpoint(session_id: str, task_id: str, worker_id: str, checkpoint: dict) -> dict:
    """Persist bounded task progress while the worker owns an unexpired lease."""
    if len(json.dumps(checkpoint, ensure_ascii=False).encode()) > 262_144:
        raise ValueError("checkpoint exceeds 256 KiB")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(f"/v1/research/{session_id}/tasks/{task_id}/checkpoint", json={"worker_id": worker_id, "checkpoint": checkpoint})
        response.raise_for_status()
        return {"session_id": session_id, "task_id": task_id, "saved": True}

def _semantic_capability_search(need: str, requirement: dict, limit: int) -> dict:
    global _semantic_index, _semantic_index_config
    model_path = os.getenv("MEDAI_EMBEDDING_MODEL")
    if not model_path:
        raise ValueError("MEDAI_EMBEDDING_MODEL must point to a pre-downloaded local embedding model")
    catalog_dir = Path(os.getenv("MEDAI_AGENTMESH_CAPABILITY_DIR", "integrations/agentmesh/capabilities"))
    index_path = Path(os.getenv("MEDAI_CAPABILITY_INDEX", ".medai/capability-vectors.sqlite"))
    if os.getenv("MEDAI_CAPABILITY_STATUS_OVERRIDES"):
        raise ValueError("semantic reuse decisions use the authenticated MedAI lifecycle registry; local lifecycle overrides are not permitted")
    api_authorization = _api_headers().get("Authorization", "")
    if not api_authorization.startswith("Bearer "):
        raise ValueError("semantic reuse decisions require an authenticated MedAI API bearer")
    registry_statuses, registry_sha256 = fetch_approved_lifecycle(
        API, api_authorization[7:], allow_insecure_hosts={"api"}
    )
    calibration_path = os.getenv("MEDAI_CAPABILITY_CALIBRATION_FILE")
    calibration_sha256 = os.getenv("MEDAI_CAPABILITY_CALIBRATION_SHA256", "")
    if bool(calibration_path) != bool(calibration_sha256):
        raise ValueError("calibrated semantic reuse requires both MEDAI_CAPABILITY_CALIBRATION_FILE and MEDAI_CAPABILITY_CALIBRATION_SHA256")
    service = os.getenv("MEDAI_AGENTMESH_SERVICE", "")
    control_url = os.getenv("AGENTMESH_CONTROL_URL", "")
    admin_token = os.getenv("AGENTMESH_ADMIN_TOKEN", "")
    if not service or not control_url or not admin_token:
        raise ValueError("live runtime route checks require MEDAI_AGENTMESH_SERVICE, AGENTMESH_CONTROL_URL, and AGENTMESH_ADMIN_TOKEN")
    control = AgentMeshControlPlaneClient(control_url, admin_token, os.getenv("AGENTMESH_TENANT", "medai"), os.getenv("AGENTMESH_NAMESPACE", "biomedical"))
    runtime_snapshot = asyncio.run(control.authenticated_runtime_snapshot())
    config = (str(Path(model_path).resolve()), str(catalog_dir.resolve()), str(index_path.resolve()))
    with _semantic_lock:
        if _semantic_index is None or _semantic_index_config != config:
            if _semantic_index is not None:
                _semantic_index.close()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            _semantic_index = SemanticCapabilityIndex(index_path, LocalSentenceTransformer(model_path))
            _semantic_index_config = config
        # Model weights and vector identity stay fixed for this process lifetime.
        # Restart the MCP service after replacing the local model.
        embedder = _semantic_index.embedder
        calibration = load_pinned_capability_calibration(
            calibration_path, calibration_sha256, expected_embedding_model_id=embedder.model_id
        ) if calibration_path else None
        packages = apply_authoritative_lifecycle(load_amcp_catalog(catalog_dir), registry_statuses)
        runtime_statuses = runtime_capability_statuses(packages, runtime_snapshot, service)
        indexed = _semantic_index.index(packages)
        matches = _semantic_index.search(
            need,
            requirement,
            limit=limit,
            reuse_similarity_threshold=calibration["threshold"] if calibration else None,
            permitted_effects=set(requirement.get("permitted_effects", [])),
            permitted_permissions=set(requirement.get("permitted_permissions", [])),
            runtime_statuses=runtime_statuses,
        )
    return {"indexed_or_updated": indexed, "matches": matches, "recommendation": matches[0]["decision"] if matches else "build", "lifecycle_registry_sha256": registry_sha256, "lifecycle_source": "authenticated MedAI capability registry", "reuse_calibration": calibration or {"status": "not_configured", "reuse_enabled": False}, "runtime_snapshot": {key: runtime_snapshot.get(key) for key in ("control_plane_status", "tenant", "namespace", "revision", "digest", "empty")}, "decision_policy": "reuse requires the exact operator-pinned holdout-passing threshold for the active embedding model, verified/production lifecycle from the authenticated MedAI registry, an equivalent permitted contract, and an exact active route in the authenticated AgentMesh runtime snapshot; no capability is executed; without calibration, reuse is disabled"}

@mcp.tool(name="medai_capabilities_search_semantic")
async def search_capabilities_semantic(session_id: str, task_id: str, need: str, requirement: dict, limit: int = 10) -> dict:
    """Rank contracts with schema checks and authenticated active-route confirmation."""
    need = ensure_public_research_query(need)
    if not isinstance(requirement, dict) or not 1 <= limit <= 100:
        raise ValueError("requirement must be an object and limit must be 1..100")
    if len(json.dumps(requirement, ensure_ascii=False).encode()) > 65_536:
        raise ValueError("capability requirement exceeds 64 KiB")
    await _charge(session_id, task_id, 1, "semantic capability discovery")
    result = await asyncio.to_thread(_semantic_capability_search, need, requirement, limit)
    query_id = hashlib.sha256((need + json.dumps(requirement, sort_keys=True)).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "capability_discovery", "source": "MedAI local semantic index and authenticated AgentMesh runtime", "source_id": query_id, "payload": {"need_sha256": hashlib.sha256(need.encode()).hexdigest(), "result": result}})
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "executed": False}

@mcp.tool(name="medai_capabilities_generate_python_proposal")
async def generate_python_capability_proposal(session_id: str, task_id: str, capability_id: str, description: str, contract: dict, implementation_requirements: list[str]) -> dict:
    """Draft source code for a disabled capability; does not run, register, or promote it."""
    provider = _model_provider_for_task(task_id, session_id)
    if provider is None:
        return {"available": False, "reason": "Configure MEDAI_MODEL_BASE_URL and MEDAI_MODEL_NAME to draft code."}
    validate_python_capability_request(
        capability_id=capability_id,
        description=description,
        source_session=session_id,
        contract=contract,
        implementation_requirements=implementation_requirements,
    )
    await _charge(session_id, task_id, 5, "provider-generated Python capability proposal; sandbox and independent scientific validation remain separate")
    proposal, completion = await _run_cancel_aware(
        session_id,
        generate_python_capability(
            provider,
            capability_id=capability_id,
            description=description,
            source_session=session_id,
            contract=contract,
            implementation_requirements=implementation_requirements,
        ),
    )
    usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
    proposal["generation"]["usage_metering"] = usage_metering
    proposal = sign_capability_proposal(proposal)
    source_hash = proposal["static_review"]["source_sha256"]
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "generated_capability_proposal", "source": f"{completion.provider}/{completion.model}", "source_id": source_hash, "payload": proposal},
        )
        response.raise_for_status()
        artifact = response.json()
        identity_response = await client.get("/v1/auth/whoami")
        identity_response.raise_for_status()
        identity = identity_response.json()
        source_tenant_id = identity.get("tenant_id") if isinstance(identity, dict) else None
        if not isinstance(source_tenant_id, str) or not source_tenant_id:
            raise ValueError("generated capability AMCP packaging requires an authenticated tenant identity")
        amcp_package = build_generated_capability_amcp_package(
            proposal,
            source_tenant_id=source_tenant_id,
            source_session_id=session_id,
            proposal_artifact_id=artifact["id"],
            proposal_artifact_sha256=artifact["sha256"],
        )
        amcp_proposal = {
            "schema_version": "1",
            "language": "python",
            "package": amcp_package,
            "source_tenant_id": source_tenant_id,
            "source_session_id": session_id,
            "source_proposal_artifact_id": artifact["id"],
            "source_proposal_artifact_sha256": artifact["sha256"],
            "source_sha256": source_hash,
            "execution": {"enabled": False, "lifecycle": "pending_review"},
            "registered": False,
            "promoted": False,
        }
        amcp_response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "generated_capability_amcp_proposal", "source": "MedAI deterministic AMCP proposal generator", "source_id": f"{amcp_package['capability']['id']}@{amcp_package['capability']['version']}", "payload": amcp_proposal},
        )
        amcp_response.raise_for_status()
        amcp_artifact = amcp_response.json()
    return {"available": True, "proposal": proposal, "artifact_id": artifact["id"], "amcp_package": amcp_package, "amcp_artifact_id": amcp_artifact["id"], "charged_credits": 5 + int(usage_metering["credits"] or 0), "executed": False, "registered": False, "promoted": False}


@mcp.tool(name="medai_capabilities_generate_r_proposal")
async def generate_r_capability_proposal(session_id: str, task_id: str, capability_id: str, description: str, contract: dict, implementation_requirements: list[str]) -> dict:
    """Draft a signed disabled R capability proposal; does not execute or register it."""
    provider = _model_provider_for_task(task_id, session_id)
    if provider is None:
        return {"available": False, "reason": "Configure a model policy for the capability_learner specialist to draft R code."}
    validate_python_capability_request(capability_id=capability_id, description=description, source_session=session_id, contract=contract, implementation_requirements=implementation_requirements)
    await _charge(session_id, task_id, 5, "provider-generated R capability proposal; sandbox and independent scientific validation remain separate")
    proposal, completion = await _run_cancel_aware(
        session_id,
        generate_r_capability(provider, capability_id=capability_id, description=description, source_session=session_id, contract=contract, implementation_requirements=implementation_requirements),
    )
    usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
    proposal["generation"]["usage_metering"] = usage_metering
    proposal = sign_capability_proposal(proposal)
    source_hash = proposal["static_review"]["source_sha256"]
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "generated_r_capability_proposal", "source": f"{completion.provider}/{completion.model}", "source_id": source_hash, "payload": proposal},
        )
        response.raise_for_status()
        artifact = response.json()
        identity_response = await client.get("/v1/auth/whoami")
        identity_response.raise_for_status()
        identity = identity_response.json()
        source_tenant_id = identity.get("tenant_id") if isinstance(identity, dict) else None
        if not isinstance(source_tenant_id, str) or not source_tenant_id:
            raise ValueError("generated R AMCP packaging requires an authenticated tenant identity")
        amcp_package = build_generated_capability_amcp_package(proposal, source_tenant_id=source_tenant_id, source_session_id=session_id, proposal_artifact_id=artifact["id"], proposal_artifact_sha256=artifact["sha256"])
        amcp_proposal = {"schema_version": "1", "language": "r", "package": amcp_package, "source_tenant_id": source_tenant_id, "source_session_id": session_id, "source_proposal_artifact_id": artifact["id"], "source_proposal_artifact_sha256": artifact["sha256"], "source_sha256": source_hash, "execution": {"enabled": False, "lifecycle": "pending_review"}, "registered": False, "promoted": False}
        amcp_response = await client.post(f"/v1/research/{session_id}/artifacts", json={"task_id": task_id, "kind": "generated_capability_amcp_proposal", "source": "MedAI deterministic R AMCP proposal generator", "source_id": f"{amcp_package['capability']['id']}@{amcp_package['capability']['version']}", "payload": amcp_proposal})
        amcp_response.raise_for_status()
        amcp_artifact = amcp_response.json()
    return {"available": True, "proposal": proposal, "artifact_id": artifact["id"], "source_sha256": source_hash, "amcp_package": amcp_package, "amcp_artifact_id": amcp_artifact["id"], "charged_credits": 5 + int(usage_metering["credits"] or 0), "executed": False, "registered": False, "promoted": False}


@mcp.tool(name="medai_capabilities_validate_python_proposal")
async def validate_python_capability_proposal(session_id: str, task_id: str, proposal_artifact_id: str, input_data: dict, test_cases: list[dict]) -> dict:
    """Run an independently tested proposal in the pinned no-network sandbox; never promote it."""
    image = os.getenv("MEDAI_PYTHON_SANDBOX_IMAGE", "").strip()
    if not image:
        return {"available": False, "registered": False, "promoted": False, "reason": "Set MEDAI_PYTHON_SANDBOX_IMAGE to an already cached immutable digest-pinned image."}
    request = validate_capability_run_request(input_data, test_cases)
    timeout = int(os.getenv("MEDAI_PYTHON_SANDBOX_TIMEOUT_SECONDS", "60"))
    sandbox = DockerPythonSandbox(
        image=image,
        docker_binary=os.getenv("MEDAI_DOCKER_BINARY", "docker"),
        timeout=timeout,
    )
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.get(
            f"/v1/research/{session_id}/artifacts",
            params={"kind": "generated_capability_proposal"},
        )
        response.raise_for_status()
        items = response.json().get("items", [])
    proposal_artifact = next((item for item in items if item.get("id") == proposal_artifact_id), None)
    if proposal_artifact is None:
        raise ValueError("generated capability proposal was not found in this research session")
    if proposal_artifact.get("session_id") != session_id or hashlib.sha256(str(proposal_artifact.get("payload", "")).encode()).hexdigest() != proposal_artifact.get("sha256"):
        raise ValueError("generated capability proposal failed its session or content-hash check")
    proposal = proposal_artifact.get("payload")
    if isinstance(proposal, str):
        proposal = json.loads(proposal)
    if not isinstance(proposal, dict) or proposal.get("execution", {}).get("enabled") is not False:
        raise ValueError("artifact is not a disabled generated capability proposal")
    contract = proposal.get("contract")
    request = validate_capability_run_request(input_data, test_cases, contract)
    signature_receipt = verify_capability_proposal_signature(proposal)
    source = proposal.get("source")
    if not isinstance(source, str):
        raise ValueError("generated capability proposal does not contain source code")
    source_receipt = validate_python_source(source)
    expected_source_hash = proposal.get("static_review", {}).get("source_sha256")
    if source_receipt["source_sha256"] != expected_source_hash:
        raise ValueError("generated proposal source hash does not match its retained static review")
    try:
        cpu_seconds_per_credit = float(os.getenv("MEDAI_CPU_SECONDS_PER_CREDIT", str(DEFAULT_CPU_SECONDS_PER_CREDIT)))
        meter_cpu_time(0, cpu_seconds_per_credit)
    except ValueError as error:
        raise ValueError("MEDAI_CPU_SECONDS_PER_CREDIT must be a positive finite number") from error
    await _charge(session_id, task_id, 2, "bounded isolated execution and independent expected-output validation for a disabled capability proposal")
    try:
        execution = await _run_cancel_aware_thread(
            session_id,
            sandbox.execute,
            source,
            request["input"],
            request["test_cases"],
            contract=contract,
            poll_seconds=0.5,
        )
        cpu_metering = meter_cpu_time(execution["cpu_time_microseconds"], cpu_seconds_per_credit)
        if cpu_metering["credits"]:
            cpu_reason = f"measured isolated Python process CPU time={cpu_metering['cpu_time_microseconds']}us rate={cpu_metering['seconds_per_credit']}s/credit"
            await _charge(session_id, task_id, int(cpu_metering["credits"]), cpu_reason)
        outcome = {
            "runner": execution["runner"],
            "image": execution["image"],
            "limits": execution["limits"],
            "cpu_usage": cpu_metering,
            "cpu_measurement": execution["cpu_measurement"],
            "validation": execution["validation"],
            "case_tests_passed": execution["case_tests_passed"],
            "execution_error_type": execution.get("execution_error_type"),
        }
    except SandboxError as error:
        outcome = {
            "runner": "docker-no-network-python",
            "image": sandbox.image,
            "validation": {"passed": 0, "failed": len(request["test_cases"]), "cases": []},
            "case_tests_passed": False,
            "execution_error_type": type(error).__name__,
        }
    receipt = {
        "schema_version": "1",
        "proposal_artifact_id": proposal_artifact_id,
        "proposal_artifact_sha256": proposal_artifact["sha256"],
        "provenance_signature": signature_receipt,
        "source_sha256": source_receipt["source_sha256"],
        "test_case_count": len(request["test_cases"]),
        "validation_request_sha256": capability_validation_request_hash(request),
        **outcome,
        "lifecycle_status": "experimental; validation receipt does not verify, register, enable, or promote the capability",
        "limitations": [
            "Passing caller-supplied cases validates only those cases; scientific correctness needs independently curated benchmark coverage.",
            "Raw inputs, expected outputs, generated results, and sandbox stderr are omitted from the persisted receipt.",
            "Docker daemon and host kernel remain part of the isolation trust boundary.",
        ],
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "capability_sandbox_validation", "source": "MedAI pinned Docker Python sandbox", "source_id": proposal_artifact_id, "payload": receipt},
        )
        response.raise_for_status()
        artifact = response.json()
    cpu_credits = int(outcome.get("cpu_usage", {}).get("credits", 0))
    return {"available": True, **receipt, "artifact_id": artifact["id"], "charged_credits": 2 + cpu_credits, "registered": False, "promoted": False}


@mcp.tool(name="medai_capabilities_validate_r_proposal")
async def validate_r_capability_proposal(session_id: str, task_id: str, proposal_artifact_id: str, input_data: dict, test_cases: list[dict]) -> dict:
    """Verify and execute one signed R proposal in the pinned sandbox; never promote it."""
    image = os.getenv("MEDAI_R_SANDBOX_IMAGE", "").strip()
    if not image:
        return {"available": False, "registered": False, "promoted": False, "reason": "Set MEDAI_R_SANDBOX_IMAGE to a locally cached immutable digest-pinned R image containing jsonlite."}
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.get(f"/v1/research/{session_id}/artifacts", params={"kind": "generated_r_capability_proposal"})
        response.raise_for_status()
        items = response.json().get("items", [])
    proposal_artifact = next((item for item in items if isinstance(item, dict) and item.get("id") == proposal_artifact_id), None)
    if proposal_artifact is None:
        raise ValueError("generated R capability proposal was not found in this research session")
    if proposal_artifact.get("session_id") != session_id or hashlib.sha256(str(proposal_artifact.get("payload", "")).encode()).hexdigest() != proposal_artifact.get("sha256"):
        raise ValueError("generated R capability proposal failed its session or content-hash check")
    proposal = proposal_artifact.get("payload")
    if isinstance(proposal, str):
        proposal = json.loads(proposal)
    if not isinstance(proposal, dict) or proposal.get("language") != "r" or proposal.get("execution", {}).get("enabled") is not False:
        raise ValueError("artifact is not a disabled generated R capability proposal")
    request = validate_capability_run_request(input_data, test_cases, proposal.get("contract"))
    signature_receipt = verify_capability_proposal_signature(proposal)
    if signature_receipt.get("verified") is not True:
        raise ValueError("generated R proposal installation signature did not verify")
    source = proposal.get("source")
    if not isinstance(source, str):
        raise ValueError("generated R capability proposal does not contain source code")
    source_receipt = validate_r_source(source)
    if source_receipt["source_sha256"] != proposal.get("static_review", {}).get("source_sha256"):
        raise ValueError("generated R source hash does not match its retained static review")
    try:
        cpu_seconds_per_credit = float(os.getenv("MEDAI_CPU_SECONDS_PER_CREDIT", str(DEFAULT_CPU_SECONDS_PER_CREDIT)))
        meter_cpu_time(0, cpu_seconds_per_credit)
    except ValueError as error:
        raise ValueError("MEDAI_CPU_SECONDS_PER_CREDIT must be a positive finite number") from error
    sandbox = DockerRSandbox(image=image, docker_binary=os.getenv("MEDAI_DOCKER_BINARY", "docker"), timeout=int(os.getenv("MEDAI_R_SANDBOX_TIMEOUT_SECONDS", "60")))
    await _charge(session_id, task_id, 2, "bounded isolated R execution and independent expected-output validation for a disabled proposal")
    try:
        execution = await _run_cancel_aware_thread(session_id, sandbox.execute, source, request["input"], request["test_cases"], contract=proposal["contract"], poll_seconds=0.5)
        cpu_metering = meter_cpu_time(execution["cpu_time_microseconds"], cpu_seconds_per_credit)
        if cpu_metering["credits"]:
            await _charge(session_id, task_id, int(cpu_metering["credits"]), f"measured isolated R process CPU time={execution['cpu_time_microseconds']}us")
        outcome = {
            "runner": execution["runner"], "image": execution["image"], "limits": execution["limits"],
            "cpu_usage": cpu_metering, "cpu_measurement": execution["cpu_measurement"],
            "validation": execution["validation"], "case_tests_passed": execution["case_tests_passed"],
            "execution_error_type": execution.get("execution_error_type"),
        }
    except (SandboxError, ValueError) as error:
        outcome = {
            "runner": "docker-no-network-r", "image": sandbox.image,
            "validation": {"performed": True, "passed": 0, "failed": max(1, len(request["test_cases"])), "cases": []},
            "case_tests_passed": False, "execution_error_type": type(error).__name__,
        }
    receipt = {
        "schema_version": "1", "proposal_artifact_id": proposal_artifact_id,
        "proposal_artifact_sha256": proposal_artifact["sha256"], "provenance_signature": signature_receipt,
        "source_sha256": source_receipt["source_sha256"], "test_case_count": len(request["test_cases"]),
        "validation_request_sha256": capability_validation_request_hash(request), **outcome,
        "lifecycle_status": "experimental; validation receipt does not verify, register, enable, or promote the capability",
        "limitations": [
            "Caller-supplied cases validate only those cases; scientific correctness requires independently curated benchmarks.",
            "Raw inputs, expected values, results, and sandbox stderr are omitted from the persisted receipt.",
            "The R image must contain Rscript and jsonlite; Docker daemon and host kernel remain part of the isolation boundary.",
        ],
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(f"/v1/research/{session_id}/artifacts", json={"task_id": task_id, "kind": "r_capability_sandbox_validation", "source": "MedAI pinned Docker R sandbox", "source_id": proposal_artifact_id, "payload": receipt})
        response.raise_for_status()
        artifact = response.json()
    cpu_credits = int(outcome.get("cpu_usage", {}).get("credits", 0))
    return {"available": True, **receipt, "artifact_id": artifact["id"], "charged_credits": 2 + cpu_credits, "registered": False, "promoted": False}


@mcp.tool(name="medai_capabilities_execute_generated")
async def execute_generated_python_capability(
    session_id: str,
    task_id: str,
    source_tenant_id: str,
    source_session_id: str,
    proposal_artifact_id: str,
    proposal_artifact_sha256: str,
    proposal_capability_id: str,
    capability_id: str,
    version: str,
    input_data: dict[str, Any],
) -> dict:
    """Invoke one exact production-approved generated proposal in its isolated language sandbox."""
    identifier = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
    for label, value in (("session", session_id), ("task", task_id), ("source tenant", source_tenant_id), ("source session", source_session_id), ("proposal artifact", proposal_artifact_id), ("proposal capability", proposal_capability_id), ("capability", capability_id), ("version", version)):
        if not isinstance(value, str) or not identifier.fullmatch(value):
            raise ValueError(f"generated execution {label} identity is malformed")
    if not isinstance(proposal_artifact_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", proposal_artifact_sha256):
        raise ValueError("proposal artifact digest must be a lowercase SHA-256")
    image = ""
    service = os.getenv("MEDAI_AGENTMESH_SERVICE", "")
    control_url = os.getenv("AGENTMESH_CONTROL_URL", "")
    admin_token = os.getenv("AGENTMESH_ADMIN_TOKEN", "")
    if not service or not control_url or not admin_token:
        raise ValueError("generated execution requires authenticated AgentMesh runtime configuration")

    # The authenticated service identity scopes the registry and source/current sessions to one tenant.
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        identity_response = await client.get("/v1/auth/whoami")
        identity_response.raise_for_status()
        identity = identity_response.json()
        if not isinstance(identity, dict) or identity.get("tenant_id") != source_tenant_id:
            raise ValueError("generated proposal belongs to a different authenticated tenant")
        current_session_response = await client.get(f"/v1/research/{session_id}/plan")
        current_session_response.raise_for_status()
        source_response = await client.get(f"/v1/research/{source_session_id}/artifacts/{proposal_artifact_id}")
        source_response.raise_for_status()
        source_artifact = source_response.json()
    if not isinstance(source_artifact, dict) or source_artifact.get("session_id") != source_session_id or source_artifact.get("kind") not in {"generated_capability_proposal", "generated_r_capability_proposal"} or source_artifact.get("sha256") != proposal_artifact_sha256:
        raise ValueError("source proposal artifact identity, session, kind, or digest does not match the AMCP binding")
    payload = source_artifact.get("payload")
    if not isinstance(payload, str) or hashlib.sha256(payload.encode("utf-8")).hexdigest() != proposal_artifact_sha256:
        raise ValueError("source proposal artifact payload digest verification failed")
    proposal = json.loads(payload)
    if not isinstance(proposal, dict) or proposal.get("execution", {}).get("enabled") is not False:
        raise ValueError("source proposal is malformed or was not retained as execution-disabled")
    manifest = proposal.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("id") != proposal_capability_id or manifest.get("version") != version:
        raise ValueError("signed proposal identity does not match the requested AMCP capability")
    signature = verify_capability_proposal_signature(proposal)
    if not signature.get("verified"):
        raise ValueError("generated proposal installation signature did not verify")
    language = proposal.get("language", "python")
    expected_kind = "generated_r_capability_proposal" if language == "r" else "generated_capability_proposal" if language == "python" else None
    if expected_kind is None or source_artifact.get("kind") != expected_kind:
        raise ValueError("generated proposal language does not match its retained artifact kind")
    source = proposal.get("source")
    if not isinstance(source, str):
        raise ValueError(f"signed proposal has no {language.upper()} source")
    source_receipt = validate_r_source(source) if language == "r" else validate_python_source(source)
    if source_receipt["source_sha256"] != proposal.get("static_review", {}).get("source_sha256"):
        raise ValueError("generated source differs from its signed static review hash")

    auth = _api_headers().get("Authorization", "")
    if not isinstance(auth, str) or not auth.startswith("Bearer "):
        raise ValueError("authenticated API service credential is required for lifecycle checks")
    _, registry_digest, registry_rows = await asyncio.to_thread(
        lambda: fetch_capability_registry(API, auth[7:], allow_insecure_hosts={"api"})
    )
    package, package_receipt = resolve_production_generated_package(
        registry_rows,
        capability_id=capability_id,
        version=version,
        source_tenant_id=source_tenant_id,
        source_session_id=source_session_id,
        proposal_artifact_id=proposal_artifact_id,
        proposal_artifact_sha256=proposal_artifact_sha256,
        proposal_capability_id=proposal_capability_id,
    )
    expected_package = build_generated_capability_amcp_package(
        proposal, source_tenant_id=source_tenant_id, source_session_id=source_session_id,
        proposal_artifact_id=proposal_artifact_id, proposal_artifact_sha256=proposal_artifact_sha256,
    )
    if package != expected_package or package["capability"]["id"] != capability_id:
        raise ValueError("approved registry AMCP package does not match this exact signed source proposal")
    image_env = "MEDAI_R_SANDBOX_IMAGE" if language == "r" else "MEDAI_PYTHON_SANDBOX_IMAGE"
    image = os.getenv(image_env, "").strip()
    if not image:
        raise ValueError(f"generated execution requires {image_env} pinned to an immutable local image digest")
    control = AgentMeshControlPlaneClient(
        control_url, admin_token, os.getenv("AGENTMESH_TENANT", "medai"),
        os.getenv("AGENTMESH_NAMESPACE", "biomedical"),
    )
    snapshot = await control.authenticated_runtime_snapshot()
    route = require_exact_active_generated_route(package, snapshot, service)
    if snapshot.get("tenant") != os.getenv("AGENTMESH_TENANT", "medai") or snapshot.get("namespace") != os.getenv("AGENTMESH_NAMESPACE", "biomedical"):
        raise ValueError("AgentMesh runtime snapshot scope does not match the configured tenant and namespace")

    contract = {"input": package["contract"]["inputs"]["properties"]["input_data"], "output": package["contract"]["outputs"]["properties"]["result"]}
    request = validate_capability_run_request(input_data, [], contract, require_test_cases=False)
    try:
        cpu_seconds_per_credit = float(os.getenv("MEDAI_CPU_SECONDS_PER_CREDIT", str(DEFAULT_CPU_SECONDS_PER_CREDIT)))
        meter_cpu_time(0, cpu_seconds_per_credit)
    except ValueError as error:
        raise ValueError("MEDAI_CPU_SECONDS_PER_CREDIT must be a positive finite number") from error
    sandbox_type = DockerRSandbox if language == "r" else DockerPythonSandbox
    timeout_env = "MEDAI_R_SANDBOX_TIMEOUT_SECONDS" if language == "r" else "MEDAI_PYTHON_SANDBOX_TIMEOUT_SECONDS"
    sandbox = sandbox_type(
        image=image, docker_binary=os.getenv("MEDAI_DOCKER_BINARY", "docker"),
        timeout=int(os.getenv(timeout_env, "60")),
    )
    await _charge(session_id, task_id, 2, f"production-approved generated {language.upper()} capability invocation in isolated no-network sandbox")
    execution: dict[str, Any] = {}
    cpu_metering: dict[str, Any] = meter_cpu_time(0, cpu_seconds_per_credit)
    try:
        execution = await _run_cancel_aware_thread(
            session_id, sandbox.execute, source, request["input"], [], contract=contract,
            require_test_cases=False, poll_seconds=0.5,
        )
        cpu_metering = meter_cpu_time(execution["cpu_time_microseconds"], cpu_seconds_per_credit)
        if cpu_metering["credits"]:
            await _charge(session_id, task_id, int(cpu_metering["credits"]), f"measured isolated generated {language.upper()} CPU time={cpu_metering['cpu_time_microseconds']}us")
        result_value = execution.get("result")
        if execution.get("execution_error_type"):
            raise SandboxError("generated capability raised an exception")
        from medai_science.contract_validation import validate_value
        try:
            validate_value(result_value, contract["output"], "$.result")
        except ValueError as error:
            raise SandboxError("generated capability output violated its retained contract") from error
        status, error_type = "completed", None
    except SandboxError as error:
        if not execution:
            execution = {"runner": f"docker-no-network-{language}", "image": image, "limits": {"wall_seconds": sandbox.timeout}, "cpu_measurement": "unavailable after runner failure"}
        result_value = None
        status, error_type = "failed", type(error).__name__

    invocation_digest = capability_validation_request_hash(request)
    result_digest = hashlib.sha256(json.dumps(result_value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest() if status == "completed" else None
    receipt = {
        "schema_version": "1", "session_id": session_id, "task_id": task_id,
        "capability_id": capability_id, "version": version, "language": language,
        "source_tenant_id": source_tenant_id, "source_session_id": source_session_id,
        "proposal_artifact_id": proposal_artifact_id, "proposal_artifact_sha256": proposal_artifact_sha256,
        "proposal_signature_key_id": signature.get("key_id"), "source_sha256": source_receipt["source_sha256"],
        "registry_package_sha256": package_receipt["package_sha256"], "registry_snapshot_sha256": registry_digest,
        "agentmesh_route": route, "input_sha256": invocation_digest, "result_sha256": result_digest,
        "status": status, "error_type": error_type, "runner": execution.get("runner"),
        "image": execution.get("image"), "limits": execution.get("limits"),
        "cpu_usage": cpu_metering, "cpu_measurement": execution.get("cpu_measurement"),
        "retention": "input and result were not persisted; result is returned transiently",
    }
    receipt["execution_receipt_sha256"] = hashlib.sha256(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "generated_capability_execution", "source": f"MedAI production-gated {language.upper()} sandbox", "source_id": proposal_artifact_id, "payload": receipt},
        )
        response.raise_for_status()
        artifact = response.json()
    response = {
        "available": status == "completed", "capability_id": capability_id, "version": version,
        "status": status, "execution_receipt_sha256": receipt["execution_receipt_sha256"],
        "validation_performed": False, "registered": False, "promoted": False,
        "charged_credits": 2 + int(cpu_metering["credits"]), "artifact_id": artifact["id"],
    }
    if status == "completed":
        response["result"] = result_value
    else:
        response["error_type"] = error_type
    return response


@mcp.tool(name="medai_capabilities_benchmark_proposal_versions")
async def benchmark_generated_capability_versions(
    session_id: str,
    task_id: str,
    baseline_proposal_artifact_id: str,
    candidate_proposal_artifact_id: str,
    benchmark_id: str,
    cases: list[dict[str, Any]],
    repeats: int = 5,
) -> dict:
    """Compare same-capability signed proposals on a paired benchmark without promotion."""
    if baseline_proposal_artifact_id == candidate_proposal_artifact_id:
        raise ValueError("baseline and candidate proposal artifacts must differ")
    if not isinstance(benchmark_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", benchmark_id):
        raise ValueError("benchmark_id must be a bounded stable identifier")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 15:
        raise ValueError("repeats must be 3..15")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        async def fetch_proposal(artifact_id: str) -> dict:
            response = await client.get(f"/v1/research/{session_id}/artifacts/{artifact_id}")
            response.raise_for_status()
            artifact = response.json()
            payload_text = artifact.get("payload")
            if not isinstance(payload_text, str) or hashlib.sha256(payload_text.encode()).hexdigest() != artifact.get("sha256"):
                raise ValueError("proposal artifact payload or digest is invalid")
            if artifact.get("session_id") != session_id or artifact.get("kind") not in {"generated_capability_proposal", "generated_r_capability_proposal"}:
                raise ValueError("proposal artifact must be a generated proposal from this session")
            proposal = json.loads(payload_text)
            if not isinstance(proposal, dict) or proposal.get("execution", {}).get("enabled") is not False:
                raise ValueError("proposal must remain execution-disabled")
            language = proposal.get("language", "python")
            expected_kind = "generated_r_capability_proposal" if language == "r" else "generated_capability_proposal" if language == "python" else None
            if expected_kind is None or artifact.get("kind") != expected_kind:
                raise ValueError("proposal language does not match its artifact kind")
            signature = verify_capability_proposal_signature(proposal)
            source = proposal.get("source")
            if not isinstance(source, str):
                raise ValueError("proposal is missing source code")
            source_receipt = validate_r_source(source) if language == "r" else validate_python_source(source)
            if source_receipt["source_sha256"] != proposal.get("static_review", {}).get("source_sha256"):
                raise ValueError("proposal source hash does not match its static review")
            manifest = proposal.get("manifest")
            if not isinstance(manifest, dict) or not isinstance(manifest.get("id"), str):
                raise ValueError("proposal is missing its capability identity")
            contract = proposal.get("contract")
            if not isinstance(contract, dict):
                raise ValueError("proposal is missing its retained input/output contract")
            return {"artifact": artifact, "proposal": proposal, "signature": signature, "source": source, "manifest": manifest, "language": language, "contract": contract}

        baseline, candidate = await asyncio.gather(
            fetch_proposal(baseline_proposal_artifact_id),
            fetch_proposal(candidate_proposal_artifact_id),
        )
    if baseline["manifest"]["id"] != candidate["manifest"]["id"]:
        raise ValueError("capability proposals must share the exact capability ID")
    if baseline["language"] != candidate["language"]:
        raise ValueError("capability proposals must use the same sandbox language")
    if baseline["contract"] != candidate["contract"]:
        raise ValueError("capability proposals must retain the exact same input/output contract")
    language = baseline["language"]
    image_env = "MEDAI_R_SANDBOX_IMAGE" if language == "r" else "MEDAI_PYTHON_SANDBOX_IMAGE"
    image = os.getenv(image_env, "").strip()
    if not image:
        return {"available": False, "reason": f"Set {image_env} to an already cached immutable digest-pinned image.", "promoted": False}
    sandbox_type = DockerRSandbox if language == "r" else DockerPythonSandbox
    timeout_env = "MEDAI_R_SANDBOX_TIMEOUT_SECONDS" if language == "r" else "MEDAI_PYTHON_SANDBOX_TIMEOUT_SECONDS"
    sandbox = sandbox_type(image=image, docker_binary=os.getenv("MEDAI_DOCKER_BINARY", "docker"), timeout=int(os.getenv(timeout_env, "60")))
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
        raise ValueError("benchmark cases must be a non-empty list of structured cases")
    if not 10 <= len(cases) <= 100:
        raise ValueError("benchmark requires 10..100 paired cases")
    case_ids: set[str] = set()
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != {"case_id", "input", "expected"}:
            raise ValueError(f"benchmark case {index} requires only case_id, input, and expected")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", case_id) or case_id in case_ids:
            raise ValueError(f"benchmark case {index} has an invalid or duplicate case_id")
        if not isinstance(row["input"], dict) or not isinstance(row["expected"], dict):
            raise ValueError(f"benchmark case {index} input and expected must be objects")
        case_ids.add(case_id)
    benchmark_contract = {"input": baseline["contract"].get("input"), "output": baseline["contract"].get("output")}
    if not isinstance(benchmark_contract["input"], dict) or not isinstance(benchmark_contract["output"], dict):
        raise ValueError("proposal benchmark requires retained input_schema and output_schema")
    validate_capability_run_request(
        cases[0].get("input"),
        [{"input": row.get("input"), "expected": row.get("expected")} for row in cases if isinstance(row, dict)],
        benchmark_contract,
    )
    cpu_seconds_per_credit = float(os.getenv("MEDAI_CPU_SECONDS_PER_CREDIT", str(DEFAULT_CPU_SECONDS_PER_CREDIT)))
    meter_cpu_time(0, cpu_seconds_per_credit)
    benchmark = {
        "schema_version": "1",
        "capability_id": baseline["manifest"]["id"],
        "baseline_version": baseline_proposal_artifact_id,
        "candidate_version": candidate_proposal_artifact_id,
        "benchmark_id": benchmark_id,
        "cases": cases,
    }
    # Reserve the bounded container-start/execution charge before either proposal runs.
    await _charge(session_id, task_id, 4 * repeats, f"paired generated-capability benchmark: {repeats} runs per version in pinned sandbox")
    try:
        report = await _run_cancel_aware_thread(
            session_id,
            run_paired_capability_benchmark,
            sandbox,
            baseline["source"],
            candidate["source"],
            benchmark,
            repeats=repeats,
            language=language,
            contract=benchmark_contract,
            poll_seconds=0.5,
        )
        total_cpu = sum(report["execution"]["baseline_cpu_microseconds"] + report["execution"]["candidate_cpu_microseconds"])
        cpu_metering = meter_cpu_time(total_cpu, cpu_seconds_per_credit)
        if cpu_metering["credits"]:
            await _charge(session_id, task_id, int(cpu_metering["credits"]), f"paired generated-capability benchmark measured sandbox CPU={total_cpu}us")
        report.update({"status": "completed", "cpu_usage": cpu_metering})
    except (SandboxError, ValueError) as error:
        benchmark_digest = hashlib.sha256(json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        report = {
            "schema_version": "1",
            "status": "execution_failed_or_incomplete",
            "failure_type": type(error).__name__,
            "capability_id": baseline["manifest"]["id"],
            "language": language,
            "benchmark_id": benchmark_id,
            "benchmark_sha256": benchmark_digest,
            "baseline_proposal_artifact_id": baseline_proposal_artifact_id,
            "baseline_proposal_artifact_sha256": baseline["artifact"]["sha256"],
            "baseline_source_sha256": baseline["proposal"]["static_review"]["source_sha256"],
            "baseline_proposal_signature": baseline["signature"],
            "candidate_proposal_artifact_id": candidate_proposal_artifact_id,
            "candidate_proposal_artifact_sha256": candidate["artifact"]["sha256"],
            "candidate_source_sha256": candidate["proposal"]["static_review"]["source_sha256"],
            "candidate_proposal_signature": candidate["signature"],
            "sandbox_image": sandbox.image,
            "repeats": repeats,
            "base_credits_charged": 4 * repeats,
            "recommendation": "no_comparison_available",
            "lifecycle_mutation": False,
            "registered": False,
            "promoted": False,
            "interpretation_limit": "A failed or incomplete sandbox run has no comparative performance result and cannot support lifecycle decisions.",
        }
    report.setdefault("baseline_proposal_artifact_id", baseline_proposal_artifact_id)
    report.setdefault("baseline_proposal_artifact_sha256", baseline["artifact"]["sha256"])
    report.setdefault("baseline_proposal_signature", baseline["signature"])
    report.setdefault("candidate_proposal_artifact_id", candidate_proposal_artifact_id)
    report.setdefault("candidate_proposal_artifact_sha256", candidate["artifact"]["sha256"])
    report.setdefault("candidate_proposal_signature", candidate["signature"])
    report.setdefault("registered", False)
    report.setdefault("promoted", False)
    if "comparison_sha256" not in report:
        report["comparison_sha256"] = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if "execution_receipt_sha256" not in report:
        report["execution_receipt_sha256"] = hashlib.sha256(json.dumps({key: report[key] for key in ("status", "benchmark_sha256", "baseline_proposal_artifact_sha256", "candidate_proposal_artifact_sha256", "sandbox_image", "repeats", "failure_type") if key in report}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "capability_version_benchmark" if report.get("status") == "completed" else "capability_version_benchmark_failure", "source": "MedAI paired pinned Docker capability benchmark", "source_id": report.get("execution_receipt_sha256", report["benchmark_sha256"]), "payload": report},
        )
        response.raise_for_status()
        report["artifact_id"] = response.json()["id"]
    return report


async def _load_completed_skill_histories(registry_index: dict[str, Any]) -> list[dict[str, Any]]:
    consumed = set(registry_index.get("consumed_session_ids", []))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        response = await client.get("/v1/research")
        response.raise_for_status()
        sessions = response.json().get("items", [])
        if not isinstance(sessions, list) or len(sessions) > 10_000:
            raise ValueError("research session history exceeds the Skill learning bound")
        candidates = [item for item in sessions if isinstance(item, dict) and item.get("status") == "completed" and item.get("id") not in consumed]
        candidates.sort(key=lambda item: (int(item.get("created_at", 0) or 0), str(item.get("id", ""))))
        histories = []
        for session in candidates[:100]:
            session_id = session.get("id")
            if not isinstance(session_id, str):
                continue
            plan_response = await client.get(f"/v1/research/{session_id}/plan")
            plan_response.raise_for_status()
            plan = plan_response.json()
            live_tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
            if not isinstance(live_tasks, list) or not any(isinstance(task, dict) and task.get("id") == "report.reproduce" and task.get("status") == "succeeded" for task in live_tasks):
                continue
            artifact_response = await client.get(f"/v1/research/{session_id}/artifacts", params={"kind": "research_reproducibility_bundle", "limit": 20})
            artifact_response.raise_for_status()
            artifacts = artifact_response.json().get("items", [])
            matches = [item for item in artifacts if isinstance(item, dict) and item.get("source") == "MedAI reproducibility engine" and item.get("session_id") == session_id]
            matches.sort(key=lambda item: (int(item.get("recorded_at", 0) or 0), str(item.get("id", ""))))
            if not matches:
                continue
            artifact = matches[-1]
            raw_payload = artifact.get("payload")
            if not isinstance(raw_payload, str) or hashlib.sha256(raw_payload.encode("utf-8")).hexdigest() != artifact.get("sha256"):
                raise ValueError("completed research bundle artifact failed its stored payload hash")
            try:
                bundle = json.loads(raw_payload)
            except json.JSONDecodeError as error:
                raise ValueError("completed research bundle artifact is malformed JSON") from error
            histories.append({"session_id": session_id, "session_status": "completed", "created_at": session.get("created_at", 0), "plan": plan, "bundle": bundle})
            if len(histories) >= 20:
                break
        return histories


@mcp.tool(name="medai_skills_learn_research_sequence")
async def skills_learn_research_sequence(session_id: str, task_id: str) -> dict[str, Any]:
    """Learn one disabled, signed research Skill from a fresh batch of completed reproducible sessions."""
    await _charge(session_id, task_id, 1, "cross-session research Skill pattern evaluation")
    registry_dir = await _tenant_skills_dir()
    histories: list[dict[str, Any]] = []
    integrity_review_required = False
    try:
        prior_index = await asyncio.to_thread(read_skill_index, registry_dir)
        histories = await _run_cancel_aware(session_id, _load_completed_skill_histories(prior_index))
        candidate, next_index, outcome = await asyncio.to_thread(learn_research_skill_batch, histories, prior_index)
    except ValueError:
        # Bad historical/index data blocks self-learning, but must not block the report.
        prior_index = {}
        next_index = {}
        candidate = None
        integrity_review_required = True
        outcome = {"status": "needs_review", "reason_code": "skill_history_or_registry_integrity_failed", "review_required": True, "execution_enabled": False}
    persisted = None
    if candidate is not None:
        persisted = await asyncio.to_thread(persist_skill_candidate, candidate, registry_dir)
        outcome = {**outcome, "registry_path": persisted["path"], "skill_sha256": persisted["skill_sha256"], "created": persisted["created"], "review_required": True, "execution_enabled": False}
    index_changed = json.dumps(prior_index, sort_keys=True) != json.dumps(next_index, sort_keys=True)
    if index_changed and not integrity_review_required:
        await asyncio.to_thread(write_skill_index, registry_dir, next_index)
    artifact_id = None
    if candidate is not None or index_changed or integrity_review_required:
        source_id = candidate.get("id", "skill-learning-batch") + "@" + str(candidate.get("version", outcome.get("status"))) if candidate else hashlib.sha256(json.dumps(outcome, sort_keys=True).encode()).hexdigest()
        payload = {"outcome": outcome, "candidate": candidate, "index_sha256": hashlib.sha256(json.dumps(next_index, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
            artifact_kind = "skill_learning_integrity_review" if integrity_review_required else "learned_biomedical_skill"
            response = await client.post(f"/v1/research/{session_id}/artifacts", json={"task_id": task_id, "kind": artifact_kind, "source": "MedAI research Skill learner", "source_id": source_id, "payload": payload})
            response.raise_for_status()
            artifact_id = response.json()["id"]
    return {**outcome, "artifact_id": artifact_id, "evaluated_sessions": len(histories), "charged_credits": 1, "review_required": candidate is not None or integrity_review_required, "execution_enabled": False}

@mcp.tool(name="medai_skills_get_candidate")
async def skills_get_candidate(skill_id: str, version: str) -> dict[str, Any]:
    """Retrieve one locally signed, immutable learned Skill for human review; never runs it."""
    registry_dir = await _tenant_skills_dir()
    skill = await asyncio.to_thread(read_verified_skill, registry_dir, skill_id, version)
    canonical = json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "skill": skill,
        "skill_markdown": skill["skill_markdown"],
        "skill_sha256": hashlib.sha256(canonical).hexdigest(),
        "signature_verified": True,
        "execution_enabled": False,
        "review_required": True,
    }

@mcp.tool(name="medai_skills_export_agentmesh")
async def skills_export_agentmesh(skill_id: str, version: str) -> dict[str, Any]:
    """Export a verified learned Skill as an installable read-only AMCP review package."""
    registry_dir = await _tenant_skills_dir()
    skill = await asyncio.to_thread(read_verified_skill, registry_dir, skill_id, version)
    package_yaml, package_sha256 = render_agentmesh_skill_review_package(skill)
    return {
        "skill_id": skill_id,
        "version": version,
        "package_yaml": package_yaml,
        "package_sha256": package_sha256,
        "signature_verified": True,
        "execution_enabled": False,
        "review_required": True,
        "installation": "Save the YAML and run agentmesh capabilities validate, then install it into the reviewed local capability catalog.",
    }

@mcp.tool(name="medai_research_recover_expired_tasks")
async def recover_expired_research_tasks(session_id: str, limit: int = 25) -> dict:
    """Requeue expired leases or fail tasks at the bounded retry ceiling."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/tasks/recover",
            json={"limit": limit},
        )
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_get_budget")
async def get_research_budget(session_id: str) -> dict:
    """Inspect remaining execution credits and the durable charge history."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research/" + session_id + "/budget")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_consume_budget")
async def consume_research_budget(session_id: str, task_id: str, credits: int, reason: str) -> dict:
    """Atomically charge a bounded research task before using metered resources."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/budget/consume", json={"task_id":task_id,"credits":credits,"reason":reason})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_update_task")
async def update_research_task(session_id: str, task_id: str, status: str, output: str | None = None, error: str | None = None, worker_id: str | None = None) -> dict:
    """Record a task transition; dependency order and allowed states are enforced by MedAI."""
    payload = {"status":status,"output":output,"error":error}
    if worker_id is not None:
        payload["worker_id"] = worker_id
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.patch(f"/v1/research/{session_id}/tasks/{task_id}", json=payload)
        response.raise_for_status()
        return {"session_id": session_id, "task_id": task_id, "status": status, "recorded": True}

@mcp.tool(name="medai_evidence_record")
async def record_evidence(session_id: str, claim: str, source: str, polarity: str, quality: float, source_id: str | None = None, provenance: dict | None = None, task_id: str | None = None) -> dict:
    """Record a sourced claim; quality is an explicit annotation, not calibrated confidence."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/evidence", json={
            "claim": claim,
            "task_id": task_id,
            "source": source,
            "source_id": source_id,
            "polarity": polarity,
            "quality": quality,
            "provenance": json.dumps(provenance or {"recorded_by":"mcp"}),
        })
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_evidence_list")
async def list_evidence(session_id: str) -> dict:
    """List retained claims and their source provenance for one research session."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        items = await _read_evidence_pages(client, session_id)
        return {"session_id": session_id, "items": items, "complete": True}

@mcp.tool(name="medai_research_store_artifact")
async def store_research_artifact(session_id: str, task_id: str, kind: str, source: str, source_id: str, payload: dict | list) -> dict:
    """Persist a source artifact with content hash and research-session provenance."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/artifacts", json={"task_id":task_id,"kind":kind,"source":source,"source_id":source_id,"payload":payload})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_research_list_artifacts")
async def list_research_artifacts(
    session_id: str,
    kind: str | None = None,
    limit: int = 200,
    after_recorded_at: int | None = None,
    after_id: str | None = None,
) -> dict:
    """Read one bounded cursor page of immutable, content-hashed session artifacts."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
        raise ValueError("artifact page limit must be 1..200")
    if (after_recorded_at is None) != (after_id is None):
        raise ValueError("artifact cursor requires both recorded_at and id")
    if after_id is not None and (not isinstance(after_id, str) or not after_id):
        raise ValueError("artifact cursor id must be a non-empty string")
    params: dict[str, Any] = {"limit": limit}
    if kind is not None:
        params["kind"] = kind
    if after_recorded_at is not None:
        params["after_recorded_at"] = after_recorded_at
        params["after_id"] = after_id
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research/" + session_id + "/artifacts", params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError("artifact page response is malformed")
    return data

@mcp.tool(name="medai_graph_add_entity")
async def add_graph_entity(session_id: str, entity_id: str, entity_type: str, label: str, metadata: dict | None = None, task_id: str | None = None) -> dict:
    """Add or update a biomedical graph entity; relationships require evidence records."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/graph", json={"session_id":session_id,"id":entity_id,"kind":entity_type,"label":label,"metadata":json.dumps(metadata or {}),"task_id":task_id})
        response.raise_for_status()
        return {"id":entity_id,"saved":True}

@mcp.tool(name="medai_graph_link_evidence")
async def link_graph_entities(session_id: str, source: str, relation: str, target: str, evidence_ids: list[str], confidence: float, artifact_ids: list[str] | None = None, task_id: str | None = None) -> dict:
    """Create a relationship only when every linked evidence or source artifact exists in-session."""
    if not evidence_ids and not artifact_ids:
        raise ValueError("at least one evidence id or source artifact id is required")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/graph/edges", json={"session_id":session_id,"source":source,"relation":relation,"target":target,"evidence_ids":json.dumps(evidence_ids or []),"artifact_ids":json.dumps(artifact_ids or []),"confidence":confidence,"task_id":task_id})
        response.raise_for_status()
        return {"source":source,"relation":relation,"target":target,"evidence_ids":evidence_ids,"artifact_ids":artifact_ids or [],"saved":True}

@mcp.tool(name="medai_papers_search_pubmed")
async def search_pubmed(session_id: str, task_id: str, query: str, limit: int = 10) -> dict:
    """Search a bounded PubMed page and report provider total and fetch coverage."""
    query = ensure_public_research_query(query)
    if not 1 <= limit <= 20:
        raise ValueError("limit must be 1..20")
    email = os.getenv("NCBI_EMAIL")
    if not email:
        raise ValueError("NCBI_EMAIL must be configured for PubMed access")
    await _charge(session_id, task_id, 2, "PubMed ESearch and EFetch batch")
    paper_records, coverage = await _run_cancel_aware(
        session_id,
        PubMedClient(email, os.getenv("NCBI_API_KEY")).search_and_fetch_with_coverage(query, min(limit, 20)),
    )
    papers = [paper.model_dump(mode="json") for paper in paper_records]
    query_sha256 = hashlib.sha256(query.encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        coverage_response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={
                "task_id": task_id,
                "kind": "literature_search_coverage",
                "source": "PubMed ESearch/EFetch coverage",
                "source_id": query_sha256,
                "payload": {
                    "schema_version": "1",
                    "query_sha256": query_sha256,
                    "coverage": coverage,
                    "interpretation_limit": "PubMed-reported hit counts and EFetch omissions describe this bounded retrieval only; they do not establish relevance or completeness of the query strategy.",
                },
            },
        )
        coverage_response.raise_for_status()
        coverage_artifact_id = coverage_response.json()["id"]
    ranking = {"available": False, "reason": "no configured model provider"}
    artifact_id = None
    try:
        provider = _model_provider_for_task(task_id, session_id)
        if provider is not None and papers:
            await _charge(session_id, task_id, 1, "model-assisted PubMed title relevance ranking")
            papers, completion = await _run_cancel_aware(session_id, rerank_pubmed_titles(provider, query, papers))
            usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
            ranking = {
                "available": True,
                "ranked_pmids": completion.value["ranked_pmids"],
                "relevance": completion.value["relevance"],
                "interpretation_limit": completion.value["interpretation_limit"],
                "provider": completion.provider,
                "model": completion.model,
                "max_output_tokens": completion.max_output_tokens,
                "prompt_sha256": completion.prompt_sha256,
                "response_sha256": completion.response_sha256,
                "usage": completion.usage,
                "usage_metering": usage_metering,
            }
            async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
                response = await client.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "literature_title_ranking", "source": "MedAI title-only model reranker", "source_id": query_sha256, "payload": ranking})
                response.raise_for_status()
                artifact_id = response.json()["id"]
    except ResearchSessionCancelled:
        raise
    except (ModelUsageBudgetError, ModelUsageError):
        raise
    except Exception as error:
        ranking = {"available": False, "reason": "model reranking unavailable", "error": type(error).__name__}
    return {"results": [omit_pubmed_abstract_text(paper) for paper in papers], "coverage": coverage, "coverage_artifact_id": coverage_artifact_id, "ranking": ranking, "ranking_artifact_id": artifact_id}

@mcp.tool(name="medai_papers_resolve_crossref")
async def resolve_crossref_paper(session_id: str, task_id: str, query: str, limit: int = 5) -> dict:
    """Resolve one DOI or search bibliography in Crossref; citation links are not evidence polarity."""
    if not 1 <= limit <= 5:
        raise ValueError("Crossref search limit must be 1..5")
    client = CrossrefClient(mailto=os.getenv("CROSSREF_MAILTO"))
    try:
        doi = client.validate_doi(query)
    except ValueError:
        doi = None
    if doi is None:
        query = ensure_public_research_query(query)
        if len(query) > 500:
            raise ValueError("bibliographic query cannot exceed 500 characters")
    await _charge(session_id, task_id, 1, "Crossref DOI or bibliographic metadata lookup")
    if doi:
        result = [await _run_cancel_aware(session_id, client.get_work(doi))]
        source_id = doi.lower()
    else:
        result = await _run_cancel_aware(session_id, client.search_bibliographic(query, limit))
        source_id = "query-" + hashlib.sha256(query.encode()).hexdigest()
    retained = build_retention_payload(result)
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "crossref_citation_metadata", "source": "Crossref REST API", "source_id": source_id, "payload": retained},
        )
        response.raise_for_status()
        artifact = response.json()
        index_response = await client_http.post("/v1/research/" + session_id + "/citation-links", json={"artifact_id": artifact["id"]})
        index_response.raise_for_status()
        indexed = index_response.json()
    return {"records": result, "artifact_id": artifact["id"], "citation_links_indexed": indexed["inserted"], "retention": retained["retention"]}


@mcp.tool(name="medai_papers_resolve_crossref_reference")
async def resolve_crossref_reference(session_id: str, task_id: str, source_artifact_id: str, citing_doi: str, reference_index: int, limit: int = 5) -> dict:
    """Search Crossref for a DOI-less retained reference and keep results as unverified candidates."""
    if task_id != "literature.discover":
        raise ValueError("citation reference resolution is restricted to literature.discover")
    if not 1 <= limit <= 5:
        raise ValueError("Crossref candidate limit must be 1..5")
    if isinstance(reference_index, bool) or not isinstance(reference_index, int) or not 0 <= reference_index < 200:
        raise ValueError("reference_index must be an integer in 0..199")
    if not isinstance(source_artifact_id, str) or not source_artifact_id or len(source_artifact_id) > 128:
        raise ValueError("source_artifact_id must be a bounded artifact identifier")
    client = CrossrefClient(mailto=os.getenv("CROSSREF_MAILTO"))
    citing_doi = client.validate_doi(citing_doi)
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client_http:
        response = await client_http.get(
            f"/v1/research/{quote(session_id, safe='')}/artifacts/{quote(source_artifact_id, safe='')}"
        )
        response.raise_for_status()
        artifact = response.json()
    selected = select_retained_reference(artifact, session_id=session_id, artifact_id=source_artifact_id, citing_doi=citing_doi, reference_index=reference_index)
    reference = selected["reference"]
    payload_digest = selected["source_artifact_sha256"]
    reference_sha256 = selected["reference_sha256"]
    existing_doi = selected["source_cited_doi"]
    if existing_doi is not None:
        return {"status": "source_doi_present", "citing_doi": citing_doi, "reference_index": reference_index, "cited_doi": client.validate_doi(existing_doi), "candidate_artifact_id": None, "interpretation": "Crossref supplied this DOI; MedAI did not independently validate the citation relationship."}
    query_record = build_unresolved_reference_query(reference)
    query = ensure_public_research_query(query_record["query"])
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
    await _charge(session_id, task_id, 1, "Crossref lookup for retained DOI-less citation candidate")
    result = await _run_cancel_aware(session_id, client.search_bibliographic(query, limit))
    retained = build_retention_payload(result, max_bytes=200_000)
    candidate_payload = {
        "schema_version": "1",
        "source": "Crossref REST API",
        "status": "unverified_candidates",
        "citing_doi": citing_doi,
        "reference_index": reference_index,
        "source_artifact_id": source_artifact_id,
        "source_artifact_sha256": payload_digest,
        "reference_sha256": reference_sha256,
        "query_sha256": query_sha256,
        "candidate_records": retained["records"],
        "candidate_records_sha256": retained["retained_records_sha256"],
        "complete_result_sha256": retained["complete_result_sha256"],
        "candidate_count": len(retained["records"]),
        "interpretation": "Bibliographic search results are candidates only. No DOI was assigned to the cited reference and no citation edge was created; researcher adjudication is required.",
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        save = await client_http.post(
            f"/v1/research/{quote(session_id, safe='')}/artifacts",
            json={"task_id": task_id, "kind": "crossref_reference_resolution_candidates", "source": "Crossref REST API", "source_id": "reference-" + reference_sha256, "payload": candidate_payload},
        )
        save.raise_for_status()
        saved_artifact = save.json()
    return {"status": "unverified_candidates", "candidate_artifact_id": saved_artifact["id"], "candidate_count": len(retained["records"]), "candidate_records": retained["records"], "citing_doi": citing_doi, "reference_index": reference_index, "source_artifact_id": source_artifact_id, "reference_sha256": reference_sha256, "query_sha256": query_sha256, "interpretation": candidate_payload["interpretation"]}


@mcp.tool(name="medai_papers_citation_network")
async def get_citation_network(session_id: str) -> dict[str, Any]:
    """Read a session's retained Crossref citation graph and descriptive connectivity metrics."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/citation-links")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_papers_lookup_preprint")
async def lookup_preprint(session_id: str, task_id: str, server: str, doi: str) -> dict:
    """Retrieve one exact bioRxiv/medRxiv DOI record and retain metadata with its license and provenance."""
    server, doi = PreprintClient.validate(server, doi)
    await _charge(session_id, task_id, 1, "bioRxiv/medRxiv DOI metadata lookup")
    result = await _run_cancel_aware(session_id, PreprintClient().get_by_doi(server, doi))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "preprint_metadata", "source": "bioRxiv/medRxiv API", "source_id": doi.lower(), "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_papers_search_preprints")
async def search_preprints(session_id: str, task_id: str, server: str, query: str, start_date: str, end_date: str, limit: int = 20, category: str | None = None, max_pages: int = 2) -> dict:
    """Search a bounded preprint date/category slice by lexical title/transient-abstract screening."""
    if server not in PreprintClient.endpoints:
        raise ValueError("preprint server must be biorxiv or medrxiv")
    await _charge(session_id, task_id, 1, "bounded bioRxiv/medRxiv date/category search")
    result = await _run_cancel_aware(
        session_id,
        PreprintClient().search(server, query, start_date, end_date, limit, category, max_pages),
    )
    source_id = "query-" + hashlib.sha256(json.dumps({"server": server, "query": query, "start": start_date, "end": end_date, "category": category}, sort_keys=True).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "preprint_search_results", "source": "bioRxiv/medRxiv API bounded lexical screen", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_ontology_search")
async def ontology_search(session_id: str, task_id: str, query: str, ontology: str | None = None, limit: int = 20) -> dict:
    """Search OLS4 for provider-ranked ontology terms; rankings are not evidence or identity matches."""
    OntologyClient.validate(query, ontology, limit)
    await _charge(session_id, task_id, 1, "EMBL-EBI OLS4 biomedical ontology search")
    result = await _run_cancel_aware(session_id, OntologyClient().search(query, ontology, limit))
    source_id = "ols4-search-" + hashlib.sha256(json.dumps({"query": query, "ontology": ontology, "limit": limit}, sort_keys=True).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "ontology_search", "source": "EMBL-EBI OLS4", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_ontology_resolve_identifier")
async def ontology_resolve_identifier(session_id: str, task_id: str, identifier: str, ontology: str | None = None) -> dict:
    """Resolve a term only on exact returned identifier equality; do not infer equivalence."""
    OntologyClient.validate(identifier, ontology, 50)
    await _charge(session_id, task_id, 1, "EMBL-EBI OLS4 exact ontology identifier resolution")
    result = await _run_cancel_aware(session_id, OntologyClient().resolve_identifier(identifier, ontology))
    source_id = "ols4-id-" + hashlib.sha256(json.dumps({"identifier": identifier, "ontology": ontology}, sort_keys=True).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "ontology_identifier_resolution", "source": "EMBL-EBI OLS4", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_ontology_cross_mapping_review")
async def ontology_cross_mapping_review(
    session_id: str,
    task_id: str,
    source_identifier: str,
    target_prefixes: list[str] | None = None,
    distance: int = 1,
    limit: int = 20,
) -> dict:
    """Retrieve direct OxO ontology cross-references for human review; never asserts equivalence."""
    OntologyMappingClient.validate(source_identifier, target_prefixes, distance, limit)
    await _charge(session_id, task_id, 1, "EMBL-EBI OxO direct disease ontology mapping review")
    result = await _run_cancel_aware(
        session_id,
        OntologyMappingClient().cross_map(source_identifier, target_prefixes, distance, limit),
    )
    source_id = "oxo-crossmap-" + hashlib.sha256(json.dumps({
        "source_identifier": source_identifier,
        "target_prefixes": target_prefixes,
        "distance": distance,
        "limit": limit,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "ontology_cross_mapping_review", "source": result["source"], "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True, "human_review_required": True}

@mcp.tool(name="medai_trials_search_clinicaltrials_gov")
async def search_clinical_trials(session_id: str, task_id: str, query: str, limit: int = 10) -> list[dict]:
    """Search public ClinicalTrials.gov records. This is not treatment advice."""
    query = ensure_public_research_query(query)
    if not 1 <= limit <= 20:
        raise ValueError("limit must be 1..20")
    await _charge(session_id, task_id, 1, "ClinicalTrials.gov study search")
    return await _run_cancel_aware(session_id, ClinicalTrialsClient().search(query, min(limit, 20)))

@mcp.tool(name="medai_trials_get_clinicaltrials_gov")
async def get_clinical_trial(session_id: str, task_id: str, nct_id: str) -> dict:
    """Retrieve and retain one ClinicalTrials.gov record snapshot by NCT ID; not treatment advice."""
    if not nct_id.startswith("NCT") or len(nct_id) != 11 or not nct_id[3:].isdigit():
        raise ValueError("nct_id must match NCT followed by eight digits")
    await _charge(session_id, task_id, 1, "ClinicalTrials.gov study record lookup")
    study = await _run_cancel_aware(session_id, ClinicalTrialsClient().get_study(nct_id))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "clinical_trial_snapshot", "source": "ClinicalTrials.gov", "source_id": nct_id, "payload": study},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"study": study, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_trials_refresh_and_compare")
async def refresh_and_compare_clinical_trial(session_id: str, task_id: str, nct_id: str) -> dict:
    """Retain the current public registry snapshot and compare it with this session's latest prior capture."""
    if not isinstance(nct_id, str) or len(nct_id) != 11 or not nct_id.startswith("NCT") or not nct_id[3:].isdigit():
        raise ValueError("nct_id must match NCT followed by eight digits")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        history_response = await client.get(
            f"/v1/research/{session_id}/artifacts",
            params={"kind": "clinical_trial_snapshot", "source_id": nct_id, "limit": 200},
        )
        history_response.raise_for_status()
        history_data = history_response.json()
        if history_data.get("next_cursor") is not None:
            raise ValueError("retained ClinicalTrials.gov snapshot history exceeds the bounded comparison page")
        previous = latest_retained_study_snapshot(history_data.get("items", []), nct_id)
    await _charge(session_id, task_id, 1, "ClinicalTrials.gov current record refresh and same-session snapshot comparison")
    current = await _run_cancel_aware(session_id, ClinicalTrialsClient().get_study(nct_id))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "clinical_trial_snapshot", "source": "ClinicalTrials.gov", "source_id": nct_id, "payload": current},
        )
        response.raise_for_status()
        snapshot_artifact = response.json()
        comparison = compare_study_snapshots(previous, current) if previous else None
        comparison_artifact = None
        if comparison is not None:
            comparison_payload = {**comparison, "history_scope": "MedAI snapshots captured in this research session; no pre-capture registry history was retrieved."}
            comparison_response = await client.post(
                "/v1/research/" + session_id + "/artifacts",
                json={"task_id": task_id, "kind": "clinical_trial_snapshot_comparison", "source": "MedAI ClinicalTrials.gov snapshot comparison", "source_id": nct_id, "payload": comparison_payload},
            )
            comparison_response.raise_for_status()
            comparison_artifact = comparison_response.json()
    return {
        "study": current,
        "snapshot_artifact_id": snapshot_artifact["id"],
        "comparison": comparison,
        "comparison_artifact_id": comparison_artifact["id"] if comparison_artifact else None,
        "history_scope": "session-observed snapshots only; the ClinicalTrials.gov v2 API returns the current record and no historical version was backfilled",
        "research_only": True,
    }

@mcp.tool(name="medai_trials_watch_register")
async def register_clinical_trial_watch(session_id: str, nct_id: str, interval_seconds: int = 86400) -> dict:
    """Persist a scheduled registry snapshot watch for one NCT record in a research session."""
    if not isinstance(nct_id, str) or len(nct_id) != 11 or not nct_id.startswith("NCT") or not nct_id[3:].isdigit():
        raise ValueError("nct_id must match NCT followed by eight digits")
    if not 3600 <= interval_seconds <= 31_536_000:
        raise ValueError("interval_seconds must be 3600..31536000")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/trial-watches", json={"nct_id": nct_id, "interval_seconds": interval_seconds})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_register")
async def register_literature_watch(session_id: str, query: str, interval_seconds: int = 86400) -> dict:
    """Schedule recurring PubMed discovery; discoveries are retained as papers, not validated claims."""
    query = ensure_public_research_query(query)
    if not 86400 <= interval_seconds <= 31_536_000:
        raise ValueError("interval_seconds must be 86400..31536000")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(f"/v1/research/{session_id}/literature-watches", json={"query": query, "interval_seconds": interval_seconds})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_list")
async def list_literature_watches(session_id: str) -> dict:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/literature-watches")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_disable")
async def disable_literature_watch(session_id: str, watch_id: str) -> dict:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(f"/v1/research/{session_id}/literature-watches/{watch_id}/disable")
        response.raise_for_status()
        return {"session_id": session_id, "watch_id": watch_id, "disabled": True}

@mcp.tool(name="medai_literature_watch_events")
async def literature_watch_events(session_id: str, watch_id: str) -> dict:
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/literature-watches/{watch_id}/events")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_papers")
async def literature_watch_papers(session_id: str, watch_id: str | None = None, limit: int = 100) -> dict:
    """List bounded PubMed bibliographic records discovered by scheduled literature watches."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be 1..500")
    params = {"limit": limit}
    if watch_id:
        params["watch_id"] = watch_id
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(f"/v1/research/{session_id}/literature-watch-papers", params=params)
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_claim_due")
async def claim_due_literature_watches(worker_id: str, limit: int = 10, lease_seconds: int = 900) -> dict:
    if not 1 <= limit <= 100 or not 30 <= lease_seconds <= 3600:
        raise ValueError("limit must be 1..100 and lease_seconds must be 30..3600")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/literature-watches/due/claim", json={"worker_id": worker_id, "limit": limit, "lease_seconds": lease_seconds})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_literature_watch_refresh")
async def refresh_literature_watch(session_id: str, watch_id: str, query: str, interval_seconds: int, worker_id: str, last_pmids: list[str] | None = None, pending_metadata_pmids: list[str] | None = None) -> dict:
    """Search a bounded overlapping PubMed window, retain new papers, and record discovery without asserting evidence."""
    query = ensure_public_research_query(query)
    if not 86400 <= interval_seconds <= 31_536_000:
        raise ValueError("interval_seconds must be 86400..31536000")
    endpoint = f"/v1/research/{session_id}/literature-watches/{watch_id}"
    try:
        last_pmids = validate_pmid_snapshot(last_pmids)
        pending_metadata_pmids = validate_pmid_snapshot(pending_metadata_pmids, max_items=20)
        async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
            charged = await client.post(endpoint + "/budget/consume", json={"worker_id": worker_id, "credits": 1, "reason": "scheduled PubMed literature discovery"})
            charged.raise_for_status()
        window_days = min(365, max(2, 2 * ((interval_seconds + 86399) // 86400)))
        email = os.getenv("NCBI_EMAIL", "")
        api_key = os.getenv("NCBI_API_KEY")
        count, pmids = await _run_cancel_aware(session_id, PubMedClient(email, api_key).search_recent_ids(query, window_days, 200))
        seen = set(last_pmids)
        new_pmids = [pmid for pmid in pmids if pmid not in seen]
        metadata_candidates = list(dict.fromkeys(pending_metadata_pmids + new_pmids))[:20]
        papers = []
        if metadata_candidates:
            async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
                charged = await client.post(endpoint + "/budget/consume", json={"worker_id": worker_id, "credits": 1, "reason": "bounded PubMed bibliographic metadata retrieval"})
                charged.raise_for_status()
            papers = await _run_cancel_aware(session_id, PubMedClient(email, api_key).fetch_bibliographic(metadata_candidates))
        async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
            response = await client.post(endpoint + "/complete", json={"worker_id": worker_id, "pmids": pmids, "result_count": count, "papers": papers})
            response.raise_for_status()
        event = response.json()
        return {"event": event, "window_days": window_days, "returned_pmids": len(pmids), "new_pmids": len(new_pmids), "new_pmid_identifiers": new_pmids, "metadata_candidates": len(metadata_candidates), "retained_record_count": len(papers), "metadata_pending_count": event.get("metadata_pending_count"), "metadata_overflow_count": event.get("metadata_overflow_count"), "incomplete": event.get("incomplete"), "research_only": True, "evidence_validated": False}
    except Exception as error:
        try:
            async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
                await client.post(endpoint + "/fail", json={"worker_id": worker_id, "error": f"{type(error).__name__}: {error}"[:4000]})
        except Exception:
            pass
        raise

@mcp.tool(name="medai_trials_watch_list")
async def list_clinical_trial_watches(session_id: str) -> dict:
    """List durable ClinicalTrials.gov watch schedules and latest outcomes for one session."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research/" + session_id + "/trial-watches")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_trials_watch_disable")
async def disable_clinical_trial_watch(session_id: str, nct_id: str) -> dict:
    """Disable a recurring ClinicalTrials.gov snapshot watch without deleting its audit history."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/disable")
        response.raise_for_status()
        return {"session_id": session_id, "nct_id": nct_id, "disabled": True}

@mcp.tool(name="medai_trials_watch_events")
async def clinical_trial_watch_events(session_id: str, nct_id: str) -> dict:
    """Read retained snapshot and field-diff history for one watched trial."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/events")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_trials_watch_latest")
async def latest_clinical_trial_watch_snapshot(session_id: str, nct_id: str) -> dict:
    """Read the latest complete registry snapshot retained by one session-specific watch."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/latest")
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_trials_watch_claim_due")
async def claim_due_clinical_trial_watches(worker_id: str, limit: int = 10, lease_seconds: int = 300) -> dict:
    """Claim due durable trial watches for the AgentMesh research worker."""
    if not 1 <= limit <= 100 or not 30 <= lease_seconds <= 3600:
        raise ValueError("limit must be 1..100 and lease_seconds must be 30..3600")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post("/v1/trial-watches/due/claim", json={"worker_id": worker_id, "limit": limit, "lease_seconds": lease_seconds})
        response.raise_for_status()
        return response.json()

@mcp.tool(name="medai_trials_watch_refresh")
async def refresh_clinical_trial_watch(session_id: str, nct_id: str, worker_id: str, last_snapshot: str | None = None) -> dict:
    """Refresh a claimed NCT watch, charge its research budget, and retain a versioned snapshot/diff event."""
    if not isinstance(nct_id, str) or len(nct_id) != 11 or not nct_id.startswith("NCT") or not nct_id[3:].isdigit():
        raise ValueError("nct_id must match NCT followed by eight digits")
    try:
        async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
            charge = await client.post("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/budget/consume", json={"worker_id": worker_id, "credits": 1, "reason": "scheduled ClinicalTrials.gov snapshot refresh"})
            charge.raise_for_status()
        current = await _run_cancel_aware(session_id, ClinicalTrialsClient().get_study(nct_id))
        previous = json.loads(last_snapshot) if isinstance(last_snapshot, str) else last_snapshot
        comparison = compare_study_snapshots(previous, current) if isinstance(previous, dict) else None
        async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
            response = await client.post("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/complete", json={"worker_id": worker_id, "snapshot": current, "comparison": comparison})
            response.raise_for_status()
        return {"event": response.json(), "comparison": comparison, "research_only": True}
    except Exception as error:
        try:
            async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
                await client.post("/v1/research/" + session_id + "/trial-watches/" + nct_id + "/fail", json={"worker_id": worker_id, "error": f"{type(error).__name__}: {error}"[:4000]})
        except Exception:
            pass
        raise

@mcp.tool(name="medai_proteins_get_alphafold_structure")
async def get_alphafold_structure(session_id: str, task_id: str, uniprot_accession: str) -> dict:
    """Retrieve and retain AlphaFold DB model metadata; predicted structures are not experimental evidence."""
    uniprot_accession = AlphaFoldClient.validate_accession(uniprot_accession)
    await _charge(session_id, task_id, 1, "AlphaFold DB protein structure lookup")
    predictions = await _run_cancel_aware(session_id, AlphaFoldClient().predictions(uniprot_accession))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "predicted_protein_structures", "source": "AlphaFold Protein Structure Database", "source_id": uniprot_accession, "payload": {"accession": uniprot_accession, "predictions": predictions}},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"accession": uniprot_accession, "predictions": predictions, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_proteins_analyze_alphafold_coordinates")
async def analyze_alphafold_coordinates(session_id: str, task_id: str, uniprot_accession: str, model_id: str) -> dict:
    """Download and summarize one bounded AlphaFold PDB prediction; retain only a hash-linked summary."""
    accession = AlphaFoldClient.validate_accession(uniprot_accession)
    if not isinstance(model_id, str) or not model_id.startswith("AF-") or model_id.split("-")[1] != accession.split("-")[0]:
        raise ValueError("model_id must be a prediction belonging to the requested UniProt accession")
    await _charge(session_id, task_id, 4, "AlphaFold model lookup, PDB coordinate download, and bounded structural analysis")
    predictions = await _run_cancel_aware(session_id, AlphaFoldClient().predictions(accession))
    if not any(prediction["model_id"] == model_id for prediction in predictions):
        raise ValueError("model_id is not present in the AlphaFold response for this accession")
    downloaded = await _run_cancel_aware(session_id, AlphaFoldClient().coordinate_file(model_id, "pdb"))
    summary = summarize_pdb(downloaded["content"], label=model_id)
    receipt = {
        "accession": accession,
        "model_id": model_id,
        "source": "AlphaFold Protein Structure Database",
        "source_url": downloaded["url"],
        "license": downloaded["license"],
        "attribution": downloaded["attribution"],
        "coordinate_sha256": downloaded["sha256"],
        "coordinate_bytes": downloaded["byte_length"],
        "retrieved_at": downloaded["retrieved_at"],
        "summary": summary,
        "experimental_structure": False,
        "research_only": True,
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "alphafold_coordinate_analysis", "source": receipt["source"], "source_id": model_id, "payload": receipt},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**receipt, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_proteins_get_uniprot_record")
async def get_uniprot_protein_record(session_id: str, task_id: str, uniprot_accession: str) -> dict:
    """Retrieve UniProt feature annotations and experimental-structure cross-reference metadata."""
    uniprot_accession = UniProtClient.validate_accession(uniprot_accession)
    await _charge(session_id, task_id, 1, "UniProtKB protein feature and structure cross-reference lookup")
    result = await _run_cancel_aware(session_id, UniProtClient().get_entry(uniprot_accession))
    record = result["record"]
    sequence = (record.get("sequence") or {}).get("value", "")
    artifact_payload = result
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) > 240_000:
        artifact_payload = {key: value for key, value in result.items() if key != "record"}
        artifact_payload["record_summary"] = {
            "primaryAccession": record.get("primaryAccession"),
            "sequence_length": (record.get("sequence") or {}).get("length"),
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest() if sequence else None,
            "source_record_omitted_for_size": True,
        }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "protein_feature_and_pdb_cross_references", "source": "UniProtKB", "source_id": uniprot_accession, "payload": artifact_payload},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"accession": uniprot_accession, "domains_and_regions": result["domains_and_regions"], "experimental_structure_cross_references": result["experimental_structure_cross_references"], "record_sha256": result["record_sha256"], "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_proteins_map_experimental_residues")
async def map_experimental_protein_residues(session_id: str, task_id: str, uniprot_accession: str, max_structures: int = 10) -> dict:
    """Map experimental PDB residues to UniProt positions via retained SIFTS evidence."""
    accession = UniProtClient.validate_accession(uniprot_accession)
    if not 1 <= max_structures <= 20:
        raise ValueError("max_structures must be 1..20")
    await _charge(session_id, task_id, 1 + 2 * max_structures, "UniProtKB, bounded SIFTS residue mappings, and updated mmCIF coordinate-observation metadata")
    entry = await _run_cancel_aware(session_id, UniProtClient().get_entry(accession))
    pdb_ids = [item["pdb_id"] for item in entry["experimental_structure_cross_references"] if item.get("pdb_id")]
    result = await _run_cancel_aware(session_id, SIFTSClient().map_residues(accession, pdb_ids, max_structures))
    result["uniprot_record_sha256"] = entry["record_sha256"]
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "experimental_residue_mappings", "source": "PDBe SIFTS", "source_id": accession, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"accession": accession, "coverage": result["coverage"], "entries": result["entries"], "residue_mappings": result["residue_mappings"], "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_proteins_search_intact_interactions")
async def search_intact_protein_interactions(session_id: str, task_id: str, uniprot_accession: str, limit: int = 100, organism_id: int | None = 9606) -> dict:
    """Retrieve source-linked IntAct interaction records; source scores are not probabilities."""
    # Validate before charging the research session.
    uniprot_accession = IntActClient.validate_accession(uniprot_accession)
    if not 1 <= limit <= 500:
        raise ValueError("interaction limit must be 1..500")
    if organism_id is not None and not 1 <= organism_id <= 9_999_999:
        raise ValueError("organism_id must be a positive NCBI taxonomy identifier or None")
    await _charge(session_id, task_id, 1, "IntAct protein interaction evidence search")
    result = await _run_cancel_aware(session_id, IntActClient().interactions(uniprot_accession, limit, organism_id))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "protein_interaction_evidence", "source": "IntAct", "source_id": result["query_accession"], "payload": build_intact_retention_payload(result)},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"accession": result["query_accession"], "records": result["records"], "reconciliation": result["reconciliation"], "response_sha256": result["response_sha256"], "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_genes_search_mygene")
async def search_genes_mygene(session_id: str, task_id: str, query: str, limit: int = 10, species: str = "human") -> dict:
    """Search MyGene.info annotations and retain the provider response as a task artifact."""
    query = ensure_public_research_query(query)
    if len(query) > 256 or not 1 <= limit <= 20:
        raise ValueError("query cannot exceed 256 characters and limit must be 1..20")
    if species not in {"human", "mouse", "rat", "zebrafish", "pig", "fruitfly", "nematode", "thale-cress", "frog"}:
        raise ValueError("unsupported MyGene.info common species filter")
    await _charge(session_id, task_id, 1, "MyGene.info annotation search")
    results = await _run_cancel_aware(session_id, MyGeneClient().search(query, limit, species))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "gene_annotations", "source": "MyGene.info", "source_id": query, "payload": {"query": query, "species": species, "results": results}},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"query": query, "species": species, "results": results, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_variants_search_clinvar")
async def search_variants_clinvar(session_id: str, task_id: str, query: str, limit: int = 10) -> dict:
    """Search ClinVar by rsID, ClinVar accession, gene, or variant term; archived submissions are not clinical advice."""
    query = ensure_public_research_query(query)
    if len(query) > 256 or not 1 <= limit <= 20:
        raise ValueError("query cannot exceed 256 characters and limit must be 1..20")
    email = os.getenv("NCBI_EMAIL")
    if not email:
        raise ValueError("NCBI_EMAIL must be configured for ClinVar E-utilities access")
    await _charge(session_id, task_id, 1, "NCBI ClinVar ESearch and ESummary")
    result = await _run_cancel_aware(
        session_id,
        ClinVarClient(api_key=os.getenv("NCBI_API_KEY"), email=email).search(query, limit),
    )
    source_id = "query-" + hashlib.sha256(query.encode("utf-8")).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "clinvar_variant_records", "source": "NCBI ClinVar E-utilities", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


@mcp.tool(name="medai_variants_lookup_dbsnp")
async def lookup_variant_dbsnp(session_id: str, task_id: str, rsid: str) -> dict:
    """Retrieve one source-bound dbSNP RefSNP record; assembly reconciliation remains local and explicit."""
    canonical_rsid = DbSNPClient.normalize_rsid(rsid)
    await _charge(session_id, task_id, 1, "NCBI dbSNP RefSNP lookup")
    result = await _run_cancel_aware(session_id, DbSNPClient().lookup(canonical_rsid))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "dbsnp_refsnp_record", "source": result["source"], "source_id": canonical_rsid, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


@mcp.tool(name="medai_variants_lookup_gnomad_frequency")
async def lookup_variants_gnomad_frequency(session_id: str, task_id: str, variant_id: str, assembly: str = "GRCh38") -> dict:
    """Retrieve one GRCh38 variant-ID frequency; callers must supply reference-normalized IDs for indels."""
    if assembly != "GRCh38":
        raise ValueError("gnomAD v4 lookup requires GRCh38 coordinates")
    normalized_id = GnomADClient.parse_variant_id(variant_id)
    await _charge(session_id, task_id, 1, "gnomAD v4 exact population frequency lookup")
    result = await _run_cancel_aware(session_id, GnomADClient().lookup(normalized_id, assembly))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "gnomad_variant_population_frequency", "source": "gnomAD Browser GraphQL API", "source_id": normalized_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


@mcp.tool(name="medai_variants_annotate_ensembl_vep")
async def annotate_variants_ensembl_vep(session_id: str, task_id: str, genomic_report: dict[str, Any], genome_build: str | None = None) -> dict:
    """Cross-check reference-validated local variants with Ensembl VEP and retain source hashes."""
    encoded = json.dumps(genomic_report, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > 2_000_000:
        raise ValueError("local genomic report exceeds the 2 MB MCP input limit")
    count = EnsemblVEPClient.queryable_count(genomic_report)
    cost = (count + EnsemblVEPClient.max_variants - 1) // EnsemblVEPClient.max_variants
    if cost:
        await _charge(session_id, task_id, cost, f"Ensembl VEP annotation for {count} reference-validated variant alleles")
    result = await _run_cancel_aware(session_id, EnsemblVEPClient().annotate_report(genomic_report, genome_build))
    if cost == 0:
        return {**result, "artifact_id": None, "research_only": True}
    source_id = "report-" + result["input_report_sha256"]
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "ensembl_vep_consequences", "source": "Ensembl VEP REST API", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_proteins_search_uniprot")
async def search_proteins_uniprot(session_id: str, task_id: str, query: str, limit: int = 10, organism_id: int = 9606) -> dict:
    """Search UniProtKB with stable accessions; persist every bounded source record."""
    query = ensure_public_research_query(query)
    if len(query) > 256 or not 1 <= limit <= 20:
        raise ValueError("query cannot exceed 256 characters and limit must be 1..20")
    if not 1 <= organism_id <= 9_999_999:
        raise ValueError("organism_id must be a positive NCBI taxonomy identifier")
    await _charge(session_id, task_id, 1, "UniProtKB protein annotation search")
    results = await _run_cancel_aware(session_id, UniProtClient().search(query, limit, organism_id))
    artifacts = []
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        for result in results:
            record_payload = result
            encoded_size = len(json.dumps(record_payload, ensure_ascii=False, separators=(",", ":")).encode())
            if encoded_size > 240_000:
                record = result["record"]
                sequence = (record.get("sequence") or {}).get("value", "")
                record_payload = {
                    **{key: value for key, value in result.items() if key != "record"},
                    "record_summary": {
                        "primaryAccession": result["source_id"],
                        "entryType": record.get("entryType"),
                        "proteinDescription": record.get("proteinDescription"),
                        "genes": record.get("genes"),
                        "organism": record.get("organism"),
                        "sequence_length": (record.get("sequence") or {}).get("length"),
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest() if sequence else None,
                        "source_record_omitted_for_size": True,
                    },
                }
            response = await client.post(
                "/v1/research/" + session_id + "/artifacts",
                json={"task_id": task_id, "kind": "protein_annotation", "source": "UniProtKB", "source_id": result["source_id"], "payload": record_payload},
            )
            response.raise_for_status()
            artifacts.append(response.json()["id"])
    return {"query": query, "organism_id": organism_id, "results": results, "artifact_ids": artifacts}

@mcp.tool(name="medai_pathways_analyze_reactome")
async def analyze_reactome_pathways(session_id: str, task_id: str, genes: list[str], species: str = "Homo sapiens", page_size: int = 20) -> dict:
    """Run Reactome over-representation analysis and retain complete provenance."""
    ReactomeClient.validate(genes, species, page_size)
    await _charge(session_id, task_id, 2, "Reactome gene-list pathway analysis")
    result = await _run_cancel_aware(session_id, ReactomeClient().analyze_genes(genes, species, page_size))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "pathway_analysis", "source": "Reactome Analysis Service", "source_id": result["input_sha256"], "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {"analysis": result, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_compounds_lookup_pubchem")
async def lookup_compound_pubchem(session_id: str, task_id: str, identifier: str) -> dict:
    """Retrieve chemical identity from PubChem; optional RDKit descriptors are not activity predictions."""
    identifier = PubChemClient.validate_identifier(identifier)
    await _charge(session_id, task_id, 2 if not identifier.isdecimal() else 1, "PubChem compound identity and structure lookup")
    result = await _run_cancel_aware(session_id, PubChemClient().lookup(identifier))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "compound_chemical_identity", "source": "PubChem PUG REST", "source_id": identifier, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_drug_profile_chembl")
async def drug_profile_chembl(session_id: str, task_id: str, inchikey: str) -> dict:
    """Retrieve ChEMBL drug mechanisms, indications and warnings by exact standard InChIKey."""
    await _charge(session_id, task_id, 4, "exact-identity ChEMBL drug mechanism, indication, and warning lookup")
    result = await _run_cancel_aware(session_id, ChEMBLClient().investigate(inchikey))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "drug_pharmacology_profile", "source": "ChEMBL web services", "source_id": inchikey, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_drug_shared_targets_chembl")
async def drug_shared_targets_chembl(session_id: str, task_id: str, inchikey_a: str, inchikey_b: str) -> dict:
    """Screen exact-identity ChEMBL mechanism records for shared target IDs; not a DDI assessment."""
    await _charge(session_id, task_id, 8, "exact-identity ChEMBL mechanism lookup for two compounds and descriptive shared-target screening")
    result = await _run_cancel_aware(session_id, ChEMBLClient().compare_shared_targets(inchikey_a, inchikey_b))
    source_id = hashlib.sha256((result["compound_a"]["inchikey"] + "\0" + result["compound_b"]["inchikey"]).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "shared_pharmacology_target_screen", "source": "ChEMBL web services", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_drug_interaction_labels_openfda")
async def drug_interaction_labels_openfda(session_id: str, task_id: str, drug_a: str, drug_b: str) -> dict:
    """Compare exact-name mentions in FDA drug-interaction label sections; this is not a DDI verdict."""
    drug_a = OpenFDALabelClient.validate_name(drug_a)
    drug_b = OpenFDALabelClient.validate_name(drug_b)
    await _charge(session_id, task_id, 8, "RxNorm ingredient resolution and bounded openFDA interaction-section retrieval for two public generic drug names")
    result = await _run_cancel_aware(session_id, OpenFDALabelClient().investigate_pair(drug_a, drug_b))
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "drug_interaction_label_screen", "source": "openFDA drug labeling", "source_id": hashlib.sha256((drug_a + "\0" + drug_b).encode()).hexdigest(), "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"]}

@mcp.tool(name="medai_drug_interaction_cross_source_review")
async def drug_interaction_cross_source_review(session_id: str, task_id: str, drug_a: str, drug_b: str) -> dict:
    """Compare public drug labels and retrieve PubMed co-mention candidates; neither yields a DDI verdict."""
    drug_a = OpenFDALabelClient.validate_name(drug_a)
    drug_b = OpenFDALabelClient.validate_name(drug_b)
    await _charge(session_id, task_id, 40, "RxNorm identity, bounded openFDA labels, up to six current and twelve historical DailyMed SPL records, and bounded PubMed discovery")
    openfda = await _run_cancel_aware(session_id, OpenFDALabelClient().investigate_pair(drug_a, drug_b))
    try:
        dailymed = await _run_cancel_aware(
            session_id,
            DailyMedLabelClient().investigate_pair(drug_a, drug_b, openfda.get("ingredient_resolutions", [])),
        )
        dailymed_error = None
    except (httpx.HTTPError, ValueError) as error:
        # Keep a usable, source-attributed openFDA result when the second public
        # label distribution is unavailable; never classify this as no evidence.
        dailymed = {"drug_queries": [drug_a, drug_b], "source": "DailyMed REST API v2", "labels": {}, "research_only": True, "clinical_advice": False}
        dailymed_error = type(error).__name__
    comparison = compare_label_distributions(openfda, dailymed)
    comparison["dailymed_retrieval_status"] = "available" if dailymed_error is None else "unavailable_or_invalid_response"
    if dailymed_error:
        comparison["dailymed_error_type"] = dailymed_error
    digest = hashlib.sha256((drug_a + "\0" + drug_b).encode()).hexdigest()
    ingredient_resolutions = openfda.get("ingredient_resolutions", [])
    literature = {"status": "not_configured", "source": "PubMed", "query": None, "candidate_count": 0, "candidates": [], "coverage": None, "resolved_ingredient_names": [], "independent_evidence_established": False, "interpretation_limit": "PubMed is an independent retrieval source from label distribution, but retrieved articles are only candidate records; independent underlying study evidence and interaction claims require full-text review."}
    literature_candidates: list[dict] = []
    if len(ingredient_resolutions) == 2 and all(row.get("resolution_status") == "resolved_unique_ingredient" and row.get("ingredients") for row in ingredient_resolutions):
        email = os.getenv("NCBI_EMAIL")
        if email:
            ingredient_name_groups = []
            for index, row in enumerate(ingredient_resolutions):
                ingredient = row["ingredients"][0]
                names = ingredient.get("names", [])
                fallback = drug_a if index == 0 else drug_b
                ingredient_name_groups.append([str(name) for name in names[:6] if isinstance(name, str)] or [fallback])
            literature["resolved_ingredient_names"] = ingredient_name_groups
            if ingredient_resolutions[0]["ingredients"][0].get("rxcui") == ingredient_resolutions[1]["ingredients"][0].get("rxcui"):
                literature["status"] = "same_resolved_ingredient"
                query = None
            else:
                query = build_interaction_evidence_query(*ingredient_name_groups)
            if query is not None:
                literature["query"] = query
                literature["query_sha256"] = hashlib.sha256(query.encode()).hexdigest()
                try:
                    await _charge(session_id, task_id, 2, "bounded PubMed search for RxNorm-resolved drug pair and PK/PD or clinical outcome signals")
                    papers, coverage = await _run_cancel_aware(session_id, PubMedClient(email, os.getenv("NCBI_API_KEY")).search_and_fetch_with_coverage(query, 5))
                    literature_candidates = [retain_candidate_paper(paper) for paper in papers[:5]]
                    for candidate in literature_candidates:
                        candidate["evidence_screen"] = characterize_interaction_candidate(candidate)
                    literature.update({"status": "retrieved", "candidate_count": len(literature_candidates), "pmids": [paper["pmid"] for paper in literature_candidates], "retrieval_limit": 5, "coverage": coverage, "search_scope": "ingredient name/synonym pair plus interaction, pharmacokinetic, pharmacodynamic, or selected exposure/outcome terms"})
                except ResearchSessionCancelled:
                    raise
                except (httpx.HTTPError, ValueError, RuntimeError) as error:
                    literature.update({"status": "unavailable_or_invalid_response", "error_type": type(error).__name__, "retrieval_limit": 5})
        else:
            literature["status"] = "not_configured_ncbi_email_required"
    else:
        literature["status"] = "skipped_unresolved_or_ambiguous_ingredient_identity"
    literature["independent_evidence_established"] = False
    literature["interpretation_limit"] = "PubMed article retrieval is separate from regulatory label distribution, but may include evidence already incorporated into labels. Publication type, MeSH, and PK/PD/outcome terms only help prioritize candidates. Independent study design, actual pair exposure, comparator, dose, route, direction, magnitude, conflicts, and clinical relevance require article-level review; retrieval does not establish an interaction or its absence."
    literature_coverage = {
        "source": "PubMed ESearch/EFetch",
        "query_sha256": literature.get("query_sha256"),
        "status": literature["status"],
        "provider_coverage": literature["coverage"] if isinstance(literature.get("coverage"), dict) else {},
        "interpretation_limit": "Provider hit count and EFetch omissions describe this bounded candidate retrieval only; they do not establish search-strategy completeness, clinical relevance, or an interaction verdict.",
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        receipts = []
        for kind, source, source_id, payload in (
            ("drug_interaction_label_screen", "openFDA drug labeling", digest, openfda),
            ("dailymed_interaction_label_screen", "DailyMed REST API v2", digest, dailymed),
            ("drug_interaction_distribution_comparison", "MedAI openFDA-DailyMed label comparator", digest, comparison),
        ):
            response = await client.post(
                f"/v1/research/{session_id}/artifacts",
                json={"task_id": task_id, "kind": kind, "source": source, "source_id": source_id, "payload": payload},
            )
            response.raise_for_status()
            receipts.append(response.json()["id"])
        for paper in literature_candidates:
            response = await client.post(
                f"/v1/research/{session_id}/artifacts",
                json={"task_id": task_id, "kind": "drug_interaction_literature_candidate", "source": "PubMed", "source_id": paper["pmid"], "payload": candidate_retention_receipt(paper)},
            )
            response.raise_for_status()
            receipts.append(response.json()["id"])
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "drug_interaction_literature_screen", "source": "MedAI PubMed title-abstract co-mention search", "source_id": digest, "payload": {key: value for key, value in literature.items() if key != "candidates"}},
        )
        response.raise_for_status()
        literature["artifact_id"] = response.json()["id"]
        receipts.append(literature["artifact_id"])
        response = await client.post(
            f"/v1/research/{session_id}/artifacts",
            json={"task_id": task_id, "kind": "drug_interaction_literature_coverage", "source": "PubMed ESearch/EFetch bounded coverage", "source_id": digest, "payload": literature_coverage},
        )
        response.raise_for_status()
        literature_coverage_artifact_id = response.json()["id"]
        receipts.append(literature_coverage_artifact_id)
    literature["candidates"] = [candidate_retention_receipt(paper) for paper in literature_candidates]
    literature["artifact_ids"] = [value for value in receipts[3:]]
    return {"drug_queries": [drug_a, drug_b], "openfda": openfda, "dailymed": dailymed, "distribution_comparison": comparison, "literature_screen": literature, "literature_coverage": literature_coverage, "literature_coverage_artifact_id": literature_coverage_artifact_id, "artifact_ids": receipts, "research_only": True, "clinical_advice": False}


@mcp.tool(name="medai_drug_interaction_pmc_fulltext_screen")
async def screen_drug_interaction_pmc_fulltext(
    session_id: str,
    task_id: str,
    literature_artifact_id: str,
    candidate_artifact_id: str,
    limit: int = 20,
) -> dict:
    """Fetch one linked PMC candidate and retain only lexical review leads, hashes, and offsets."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be 1..20")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        async def fetch_artifact(artifact_id: str) -> dict:
            response = await client.get(f"/v1/research/{quote(session_id, safe='')}/artifacts/{quote(artifact_id, safe='')}")
            response.raise_for_status()
            artifact = response.json()
            payload_text = artifact.get("payload")
            if not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > 262_144:
                raise ValueError("selected source artifact is malformed or exceeds 256 KiB")
            if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != artifact.get("sha256"):
                raise ValueError("selected source artifact hash does not match its retained payload")
            payload = json.loads(payload_text)
            if not isinstance(payload, dict):
                raise ValueError("selected source artifact payload must be a JSON object")
            artifact["decoded_payload"] = payload
            return artifact

        literature_artifact, candidate_artifact = await _run_cancel_aware(
            session_id, asyncio.gather(fetch_artifact(literature_artifact_id), fetch_artifact(candidate_artifact_id)),
        )

    if literature_artifact.get("kind") != "drug_interaction_literature_screen":
        raise ValueError("literature_artifact_id must identify a drug-interaction PubMed screen")
    if candidate_artifact.get("kind") != "drug_interaction_literature_candidate" or candidate_artifact.get("source") != "PubMed":
        raise ValueError("candidate_artifact_id must identify a retained PubMed drug-interaction candidate")
    if literature_artifact.get("task_id") != task_id or candidate_artifact.get("task_id") != task_id:
        raise ValueError("selected literature artifacts must belong to the current research task")
    literature = literature_artifact["decoded_payload"]
    candidate = candidate_artifact["decoded_payload"]
    binding = validate_interaction_fulltext_binding(literature, candidate, str(candidate_artifact.get("source_id", "")))
    pmid, pmcid = binding["pmid"], binding["pmcid"]

    await _charge(session_id, task_id, 2, "bounded PMC full-text retrieval and source-anchored interaction review leads")
    article = await _run_cancel_aware(session_id, PMCBioCClient().fetch(pmcid))
    result = build_interaction_fulltext_screen(
        article,
        pmcid=pmcid,
        pmid=pmid,
        candidate_artifact_id=candidate_artifact["id"],
        candidate_artifact_sha256=candidate_artifact["sha256"],
        ingredient_names_a=binding["ingredient_names_a"],
        ingredient_names_b=binding["ingredient_names_b"],
        limit=limit,
    )
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            f"/v1/research/{quote(session_id, safe='')}/artifacts",
            json={"task_id": task_id, "kind": "drug_interaction_pmc_fulltext_screen", "source": "PMC BioC hash-and-offset lexical review leads", "source_id": pmcid, "payload": result},
        )
        response.raise_for_status()
        result["artifact_id"] = response.json()["id"]
    return result

@mcp.tool(name="medai_drug_interaction_review_sources")
async def drug_interaction_review_sources(session_id: str, task_id: str, label_artifact_id: str, literature_artifact_id: str, max_pairs: int = 10) -> dict:
    """Screen selected versioned label and PubMed or licensed PMC sentences for human review."""
    if not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or not 1 <= max_pairs <= 10:
        raise ValueError("max_pairs must be 1..10")
    provider = _model_provider_for_task(task_id, session_id)
    if provider is None:
        raise RuntimeError("source-anchored sentence screening requires a configured model provider")

    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        async def fetch_artifact(artifact_id: str) -> dict:
            response = await client.get(f"/v1/research/{session_id}/artifacts/{artifact_id}")
            response.raise_for_status()
            artifact = response.json()
            payload_text = artifact.get("payload")
            if not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > 262_144:
                raise ValueError("selected source artifact is malformed or exceeds 256 KiB")
            if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != artifact.get("sha256"):
                raise ValueError("selected source artifact hash does not match its retained payload")
            payload = json.loads(payload_text)
            if not isinstance(payload, dict):
                raise ValueError("selected source artifact payload must be a JSON object")
            artifact["decoded_payload"] = payload
            return artifact
        label_artifact, paper_artifact = await _run_cancel_aware(
            session_id, asyncio.gather(fetch_artifact(label_artifact_id), fetch_artifact(literature_artifact_id)),
        )

    if label_artifact.get("kind") not in {"drug_interaction_label_screen", "dailymed_interaction_label_screen"}:
        raise ValueError("label_artifact_id must identify an openFDA or DailyMed interaction label artifact")
    is_pmc = paper_artifact.get("kind") == "drug_interaction_pmc_fulltext_screen"
    if not is_pmc and (paper_artifact.get("kind") != "drug_interaction_literature_candidate" or paper_artifact.get("source") != "PubMed"):
        raise ValueError("literature_artifact_id must identify a retained PubMed candidate or PMC full-text screen")
    label_payload, paper_payload = label_artifact["decoded_payload"], paper_artifact["decoded_payload"]
    paper_source_name = "PubMed"
    paper_source_id = str(paper_artifact.get("source_id", ""))
    paper_field = "abstract"
    paper_source_sha256 = None
    paper_context = {}
    if is_pmc:
        if paper_payload.get("source") != "PMC BioC" or paper_payload.get("status") != "review_leads_available":
            raise ValueError("selected PMC artifact has no full-text review leads")
        if paper_payload.get("pmcid") != paper_source_id or not isinstance(paper_payload.get("source_sha256"), str):
            raise ValueError("PMC source identity or BioC digest does not match its artifact envelope")
        if not provider_is_local(provider):
            raise ValueError("PMC full-text model screening is restricted to a local inference provider")
        if not license_policy_allows_model_processing(paper_payload.get("source_license"), os.getenv("MEDAI_PMC_ALLOWED_MODEL_LICENSES", "")):
            raise ValueError("PMC full-text model processing requires its exact article license in MEDAI_PMC_ALLOWED_MODEL_LICENSES")
        candidate_artifact_id = paper_payload.get("candidate_artifact_id")
        candidate_artifact = await _run_cancel_aware(session_id, fetch_artifact(str(candidate_artifact_id or "")))
        if candidate_artifact.get("kind") != "drug_interaction_literature_candidate" or candidate_artifact.get("source") != "PubMed" or candidate_artifact.get("task_id") != task_id:
            raise ValueError("PMC screen must link to a same-task PubMed candidate")
        if candidate_artifact.get("sha256") != paper_payload.get("candidate_artifact_sha256"):
            raise ValueError("linked PubMed candidate digest differs from the retained PMC screen")
        candidate_payload = candidate_artifact["decoded_payload"]
        pmc_ids = candidate_payload.get("identifiers", {})
        if candidate_payload.get("pmid") != paper_payload.get("pmid") or not isinstance(pmc_ids, dict) or pmc_ids.get("pmc") != paper_source_id:
            raise ValueError("PMC full-text screen does not match its linked PubMed record")
        await _charge(session_id, task_id, 2, "PMC BioC source revalidation for selected drug-interaction claim screening")
        article = await _run_cancel_aware(session_id, PMCBioCClient().fetch(paper_source_id))
        paper_sentences = rehydrate_interaction_fulltext_candidates(paper_payload, article)
        paper_source_name, paper_field = "PMC BioC", "full_text"
        paper_source_sha256 = paper_payload["source_sha256"]
        paper_context = {"pmid": paper_payload.get("pmid"), "pmcid": paper_source_id, "section": "selected PMC full-text sentence; source section is retained in the full-text screen", "article_title": str(paper_payload.get("article_title", ""))[:500]}
    else:
        if paper_payload.get("source") != "PubMed" or str(paper_payload.get("pmid", "")) != paper_source_id:
            raise ValueError("PubMed candidate source identity does not match its artifact envelope")
        abstract = paper_payload.get("abstract")
        if not isinstance(abstract, str) or not abstract:
            raise ValueError("PubMed abstract text was not retained because reuse rights were not established; select an exact-license PMC full-text screen instead")
        paper_candidate_coverage = screen_sentence_candidates(abstract, "PMID")
        paper_sentences = paper_candidate_coverage["candidates"]
        paper_context = {"title": str(paper_payload.get("title", ""))[:500], "journal": paper_payload.get("journal"), "publication_date": paper_payload.get("publication_date"), "publication_types": paper_payload.get("publication_types", [])[:10]}

    labels = label_payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("selected label artifact has no label records")
    label_screen = screen_drug_label_candidates(labels)
    label_sentences = label_screen["candidates"]
    if not paper_sentences or not label_sentences:
        raise ValueError("selected artifacts do not contain bounded label and source sentences to compare")
    pairs = rank_candidate_pairs(label_sentences, paper_sentences, max_pairs)
    await _charge(session_id, task_id, len(pairs), "source-anchored model screening of selected paper and versioned-label sentence pairs")
    reviews = []
    for pair in pairs:
        label, paper = pair["label"], pair["paper"]
        completion = await _run_cancel_aware(
            session_id,
            compare_claim_pair(
                provider,
                candidate_a=label["candidate_id"],
                text_a=label["text"],
                candidate_b=paper["candidate_id"],
                text_b=paper["text"],
                metadata_a={"source": label_artifact.get("source"), "drug_name": label.get("drug_name"), "set_id": label.get("set_id"), "version": label.get("version"), "effective_time": label.get("effective_time"), "product_name": label.get("product_name")},
                metadata_b={"source": paper_source_name, "source_id": paper_source_id, **paper_context, **({"section": paper.get("section"), "source_section": paper.get("source_section"), "matched_ingredient_names_a": paper.get("matched_ingredient_names_a"), "matched_ingredient_names_b": paper.get("matched_ingredient_names_b")} if is_pmc else {})},
            ),
        )
        usage = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
        reviews.append({
            **completion.value,
            "lexical_screen_rank": pair["screen_rank"],
            "shared_lexical_terms": pair["shared_terms"],
            "lexical_similarity": pair["lexical_overlap"],
            "sources": [
                {"artifact_id": label_artifact["id"], "artifact_sha256": label_artifact["sha256"], "source": label_artifact["source"], "source_id": label_artifact["source_id"], "candidate_id": label["candidate_id"], "field": "labels.drug_interactions", "set_id": label.get("set_id"), "version": label.get("version"), "offset": label["offset"], "length": label["length"], "text_sha256": label["text_sha256"]},
                {"artifact_id": paper_artifact["id"], "artifact_sha256": paper_artifact["sha256"], "source": paper_source_name, "source_id": paper_source_id, "candidate_id": paper["candidate_id"], "field": paper_field, "offset": paper["offset"], "length": paper["length"], "text_sha256": paper["text_sha256"], **({"source_sha256": paper_source_sha256, "section": paper.get("section"), "source_section": paper.get("source_section"), "passage_index": paper.get("passage_index")} if paper_source_sha256 else {})},
            ],
            "model": {"provider": completion.provider, "name": completion.model, "max_output_tokens": completion.max_output_tokens, "prompt_sha256": completion.prompt_sha256, "response_sha256": completion.response_sha256, "usage": completion.usage, "usage_metering": usage},
        })
    result = {
        "label_artifact_id": label_artifact_id,
        "literature_artifact_id": literature_artifact_id,
        "reviews": reviews,
        "candidate_pair_count": len(reviews),
        "candidate_coverage": {
            "pubmed_or_pmc": paper_candidate_coverage if not is_pmc else {
                "source_sentence_count": paper_payload.get("anchored_source_sentence_count", len(paper_sentences)),
                "retained_candidate_count": len(paper_sentences),
                "omitted_overlong_sentence_count": paper_payload.get("omitted_overlong_sentence_count", 0),
                "reported_omitted_candidate_count": paper_payload.get("omitted_candidate_count", 0),
                "source_scan_truncated": bool(paper_payload.get("candidate_scan_truncated")),
                "truncated": bool(paper_payload.get("candidate_scan_truncated")) or paper_payload.get("omitted_candidate_count", 0) > 0 or paper_payload.get("omitted_overlong_sentence_count", 0) > 0,
                "coverage_source": "bounded PMC BioC full-text screen receipt",
            },
            "label": label_screen["coverage"],
        },
        "review_required": True,
        "interaction_verdict": None,
        "interpretation_limit": "Model screening compares source-anchored sentences and returns categorical review leads only. It does not establish contradiction, interaction, evidence quality, causality, pharmacokinetic/pharmacodynamic direction, or clinical relevance.",
    }
    source_id = hashlib.sha256((label_artifact["id"] + "\0" + paper_artifact["id"]).encode()).hexdigest()
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(f"/v1/research/{session_id}/artifacts", json={"task_id": task_id, "kind": "drug_interaction_semantic_review", "source": f"{reviews[0]['model']['provider']}/{reviews[0]['model']['name']} categorical screening" if reviews else "MedAI interaction review", "source_id": source_id, "payload": result})
        response.raise_for_status()
        result["artifact_id"] = response.json()["id"]
    return result

@mcp.tool(name="medai_gene_disease_evidence")
async def gene_disease_evidence(session_id: str, task_id: str, disease_id: str, limit: int = 50) -> dict:
    """Retrieve source-linked gene–disease evidence, including Monarch's direct human assertions."""
    disease_id = GeneDiseaseClient.validate_disease_id(disease_id)
    GeneDiseaseClient.validate_limit(limit)
    client = GeneDiseaseClient()
    source_names = []
    requested = []
    if disease_id.startswith("EFO_"):
        source_names.append("Open Targets Platform")
        requested.append(client.opentargets(disease_id, limit))
        source_names.append("NHGRI-EBI GWAS Catalog")
        requested.append(client.gwas_catalog(disease_id, limit))
    source_names.append("Monarch Initiative Knowledge Graph")
    requested.append(client.monarch_gene_disease(disease_id, limit))
    await _charge(session_id, task_id, len(requested), "source-specific gene-disease evidence retrieval")
    outcomes = await _run_cancel_aware(session_id, asyncio.gather(*requested, return_exceptions=True))
    sources = [value for value in outcomes if isinstance(value, dict)]
    errors = [{"source": source_names[index], "error": type(value).__name__} for index, value in enumerate(outcomes) if isinstance(value, Exception)]
    if not sources:
        raise RuntimeError("all gene-disease sources failed")
    reconciliation = reconcile_gene_disease_sources(disease_id, sources)
    payload = {"disease_id": disease_id, "sources": sources, "source_errors": errors, "reconciliation": reconciliation}
    artifacts = []
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        for source in sources:
            response = await client_http.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "gene_disease_evidence", "source": source["source"], "source_id": disease_id, "payload": source})
            response.raise_for_status()
            artifacts.append(response.json()["id"])
        response = await client_http.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "gene_disease_reconciliation", "source": "MedAI exact-ID reconciliation", "source_id": disease_id, "payload": reconciliation})
        response.raise_for_status()
        artifacts.append(response.json()["id"])
    return {**payload, "artifact_ids": artifacts}


@mcp.tool(name="medai_gene_disease_panelapp")
async def gene_disease_panelapp(session_id: str, task_id: str, panel_id: int, version: str | None = None) -> dict:
    """Retrieve one exact version of a curated PanelApp gene panel without score aggregation."""
    PanelAppClient._validate(panel_id, version)
    await _charge(session_id, task_id, 1, "Genomics England PanelApp versioned gene panel lookup")
    result = await _run_cancel_aware(session_id, PanelAppClient().get_panel(panel_id, version))
    if not result.get("found", True):
        raise ValueError("PanelApp panel/version was not found")
    source_id = f"{result['panel_id']}@{result['version']}"
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "panelapp_curated_gene_panel", "source": "Genomics England PanelApp", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


async def _gencc_query(session_id: str, task_id: str, query: str, by_disease: bool, limit: int) -> dict[str, Any]:
    client = GenCCClient()
    refresh_due = await asyncio.to_thread(client.refresh_due)
    if refresh_due:
        await _charge(session_id, task_id, 1, "conditional GenCC weekly bulk snapshot refresh")
    result = await _run_cancel_aware(
        session_id,
        client.search_disease(query, limit) if by_disease else client.search_gene(query, limit),
    )
    source_id = f"{result['query'].get('disease_curie') or result['query'].get('gene_curie') or result['query'].get('gene_symbol')}@{result['dataset_sha256']}"
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "gencc_gene_disease_assertions", "source": "Gene Curation Coalition (GenCC)", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


@mcp.tool(name="medai_gene_disease_gencc_gene")
async def gene_disease_gencc_gene(session_id: str, task_id: str, gene: str, limit: int = 50) -> dict[str, Any]:
    """Search cached GenCC submission-level assertions by exact HGNC ID or gene symbol."""
    return await _gencc_query(session_id, task_id, gene, False, limit)


@mcp.tool(name="medai_gene_disease_gencc_disease")
async def gene_disease_gencc_disease(session_id: str, task_id: str, disease_curie: str, limit: int = 50) -> dict[str, Any]:
    """Search cached GenCC submissions by exact MONDO CURIE without consensus aggregation."""
    return await _gencc_query(session_id, task_id, disease_curie, True, limit)


async def _clingen_gene_validity_query(session_id: str, task_id: str, query: str, by_disease: bool, limit: int) -> dict[str, Any]:
    client = ClinGenGeneValidityClient()
    refresh_due = await asyncio.to_thread(client.refresh_due)
    if refresh_due:
        await _charge(session_id, task_id, 1, "conditional ClinGen gene-disease validity snapshot refresh")
    result = await _run_cancel_aware(
        session_id,
        client.search_disease(query, limit) if by_disease else client.search_gene(query, limit),
    )
    source_id = f"{result['query']['value']}@{result['dataset_sha256']}"
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "clingen_gene_disease_validity", "source": "ClinGen Gene-Disease Validity", "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}


@mcp.tool(name="medai_gene_disease_clingen_gene")
async def gene_disease_clingen_gene(session_id: str, task_id: str, gene: str, limit: int = 50) -> dict[str, Any]:
    """Search ClinGen expert-panel validity curations by exact HGNC ID or symbol."""
    return await _clingen_gene_validity_query(session_id, task_id, gene, False, limit)


@mcp.tool(name="medai_gene_disease_clingen_disease")
async def gene_disease_clingen_disease(session_id: str, task_id: str, disease_curie: str, limit: int = 50) -> dict[str, Any]:
    """Search ClinGen expert-panel validity curations by exact MONDO ID."""
    return await _clingen_gene_validity_query(session_id, task_id, disease_curie, True, limit)


@mcp.tool(name="medai_gene_disease_orphadata")
async def gene_disease_orphadata(session_id: str, task_id: str, orphacode: str) -> dict[str, Any]:
    """Retrieve Orphadata's typed gene-disease assertions for one exact ORPHAcode."""
    client = OrphadataClient()
    code = client.validate_orphacode(orphacode)
    await _charge(session_id, task_id, 1, "Orphadata exact ORPHAcode gene-disease retrieval")
    result = await _run_cancel_aware(session_id, client.disease_genes(code))
    if not result["found"]:
        raise ValueError("Orphadata did not return the exact requested ORPHAcode")
    source_id = f"ORPHA:{code}@{result['source_response_sha256']}"
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client_http:
        response = await client_http.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "orphadata_gene_disease_assertions", "source": result["source"], "source_id": source_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"], "research_only": True}

@mcp.tool(name="medai_papers_understand_pmc")
async def understand_pmc_paper(session_id: str, task_id: str, identifier: str) -> dict:
    """Parse PMC BioC transiently and return only hash-linked extraction metadata."""
    identifier = PMCBioCClient.validate_identifier(identifier)
    await _charge(session_id, task_id, 2, "PMC BioC full-text retrieval and structural parsing")
    result = await _run_cancel_aware(session_id, PMCBioCClient().fetch(identifier))
    retained = build_extraction_retention_receipt(result)
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post("/v1/research/" + session_id + "/artifacts", json={"task_id": task_id, "kind": "paper_extraction_metadata", "source": "PMC BioC", "source_id": identifier, "payload": retained})
        response.raise_for_status()
        artifact = response.json()
    model_extraction = {"available": False, "reason": "No loopback structured model provider is configured."}
    try:
        provider = _model_provider_for_task(task_id, session_id)
        if provider is not None:
            if not provider_is_local(provider):
                model_extraction = {"available": False, "reason": "PMC full-text extraction sends article text only to a loopback model endpoint."}
            elif not license_policy_allows_model_processing(result.get("source_license"), os.getenv("MEDAI_PMC_ALLOWED_MODEL_LICENSES", "")):
                model_extraction = {"available": False, "reason": "PMC model processing is disabled unless the operator allowlists the exact source license after review."}
            else:
                batches = extraction_batch_count(result)
                await _charge(session_id, task_id, batches, "local-model structured PMC result annotation batches")
                completion = await _run_cancel_aware(session_id, extract_local_result_annotations(provider, result))
                usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
                if completion.value.get("document_id") != result.get("document_id"):
                    raise ValueError("model extraction returned a different PMC document identifier")
                async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
                    response = await client.post(
                        "/v1/research/" + session_id + "/artifacts",
                        json={"task_id": task_id, "kind": "paper_result_annotations", "source": "PMC BioC local-model result annotation", "source_id": identifier, "payload": {**completion.value, "provider": completion.provider, "model": completion.model, "max_output_tokens": completion.max_output_tokens, "prompt_sha256": completion.prompt_sha256, "response_sha256": completion.response_sha256, "usage": completion.usage, "usage_metering": usage_metering}},
                    )
                    response.raise_for_status()
                    annotation_artifact = response.json()
                model_extraction = {"available": True, "artifact_id": annotation_artifact["id"], "annotations": completion.value["annotations"], "batches": batches, "provider": completion.provider, "model": completion.model, "max_output_tokens": completion.max_output_tokens, "prompt_sha256": completion.prompt_sha256, "response_sha256": completion.response_sha256, "usage": completion.usage, "usage_metering": usage_metering, "retention": "No full-text sentence text is persisted in this annotation artifact."}
    except ResearchSessionCancelled:
        raise
    except (ModelUsageBudgetError, ModelUsageError):
        raise
    except Exception as error:
        model_extraction = {"available": False, "reason": "local model annotation failed", "error": type(error).__name__}
    return {"paper": retained, "artifact_id": artifact["id"], "model_extraction": model_extraction, "retention": "Full-text content is neither returned through MCP nor persisted by this operation; any licensed text is used transiently only, and model processing requires an exact operator-reviewed license allowlist and a loopback endpoint."}

@mcp.tool(name="medai_papers_align_pmc_claims")
async def align_pmc_result_claims(
    session_id: str,
    task_id: str,
    annotation_artifact_a: str,
    candidate_id_a: str,
    annotation_artifact_b: str,
    candidate_id_b: str,
) -> dict:
    """Screen two retained PMC result candidates semantically with a local model; persist labels and provenance only."""
    provider = _model_provider_for_task(task_id, session_id)
    if provider is None:
        return {"available": False, "review_required": False, "reason": "No configured local model provider is available."}
    if not provider_is_local(provider):
        return {"available": False, "review_required": False, "reason": "PMC full-text claim comparison is restricted to a loopback model endpoint."}
    if not annotation_artifact_a or not annotation_artifact_b or annotation_artifact_a == annotation_artifact_b:
        raise ValueError("claim candidates must come from two distinct retained same-session PMC annotation artifacts")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        async def fetch_annotation(artifact_id: str) -> dict[str, Any]:
            response = await client.get(f"/v1/research/{session_id}/artifacts/{artifact_id}")
            response.raise_for_status()
            return decode_verified_annotation_artifact(response.json(), session_id)

        artifact_a, artifact_b = await _run_cancel_aware(
            session_id,
            asyncio.gather(fetch_annotation(annotation_artifact_a), fetch_annotation(annotation_artifact_b)),
        )
    payload_a, payload_b = artifact_a["decoded_payload"], artifact_b["decoded_payload"]
    if payload_a.get("document_id") == payload_b.get("document_id"):
        raise ValueError("semantic claim comparison requires two different PMC documents")
    candidates: list[dict[str, Any]] = []
    for artifact, payload, candidate_id in (
        (artifact_a, payload_a, candidate_id_a), (artifact_b, payload_b, candidate_id_b),
    ):
        annotations = payload.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError("PMC annotation artifact has no annotation array")
        annotation = next((row for row in annotations if isinstance(row, dict) and row.get("candidate_id") == candidate_id), None)
        if annotation is None:
            raise ValueError("candidate identifier is not present in its retained annotation artifact")
        pmc_id = PMCBioCClient.validate_identifier(str(artifact.get("source_id", "")))
        candidates.append({"artifact": artifact, "payload": payload, "annotation": annotation, "candidate_id": candidate_id, "pmc_id": pmc_id})

    shared_entity_ids = validate_semantic_claim_pair(candidates[0]["annotation"], candidates[1]["annotation"])
    await _charge(session_id, task_id, 5, "two PMC BioC refreshes and one local-model semantic claim-pair screening call")
    parsed_records = [await _run_cancel_aware(session_id, PMCBioCClient().fetch(item["pmc_id"])) for item in candidates]
    for item, parsed in zip(candidates, parsed_records):
        if parsed.get("source_sha256") != item["payload"].get("source_sha256"):
            raise ValueError("refetched PMC source differs from the retained annotation source hash")
        if not license_policy_allows_model_processing(parsed.get("source_license"), os.getenv("MEDAI_PMC_ALLOWED_MODEL_LICENSES", "")):
            raise ValueError("both PMC article licenses must be explicitly allowlisted for local model processing")

    sentence_rows = []
    for item, parsed in zip(candidates, parsed_records):
        candidate = next((row for row in parsed.get("result_sentence_candidates", []) if row.get("candidate_id") == item["candidate_id"]), None)
        expected_hash = item["payload"].get("candidate_text_hashes", {}).get(item["candidate_id"])
        if candidate is None or not isinstance(expected_hash, str) or hashlib.sha256(candidate["text"].encode("utf-8")).hexdigest() != expected_hash:
            raise ValueError("PMC result sentence candidate differs from its retained source anchor")
        sentence_rows.append(candidate)

    pair_contexts = [
        {
            **{key: item["annotation"].get(key) for key in ("result_kind", "direction", "study_system", "entity_ids")},
            "shared_exact_source_entity_ids": shared_entity_ids,
            "title": str(parsed.get("title", ""))[:500],
            "abstract_context": str(parsed.get("abstract", ""))[:1000],
            "methods_context": str(parsed.get("methods", ""))[:1000],
        }
        for item, parsed in zip(candidates, parsed_records)
    ]
    completion = await _run_cancel_aware(
        session_id,
        compare_claim_pair(
            provider,
            candidate_a=candidate_id_a,
            text_a=sentence_rows[0]["text"],
            candidate_b=candidate_id_b,
            text_b=sentence_rows[1]["text"],
            metadata_a=pair_contexts[0],
            metadata_b=pair_contexts[1],
        ),
    )
    usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
    source_receipts = []
    for item, parsed, sentence, context in zip(candidates, parsed_records, sentence_rows, pair_contexts):
        source_receipts.append({
            "annotation_artifact_id": item["artifact"].get("id"),
            "pmc_id": item["pmc_id"],
            "document_id": parsed["document_id"],
            "source_sha256": parsed["source_sha256"],
            "candidate_id": item["candidate_id"],
            "candidate_text_sha256": hashlib.sha256(sentence["text"].encode("utf-8")).hexdigest(),
            "passage_index": sentence["passage_index"],
            "sentence_index": sentence["sentence_index"],
            "source_offset": sentence["source_offset"],
            "source_local_offset": sentence["source_local_offset"],
            "source_length": sentence["source_length"],
            "license": parsed.get("source_license"),
            "claim_context_receipt": claim_context_receipt(context),
        })
    result = {
        "schema_version": "1",
        "shared_exact_source_entity_ids": shared_entity_ids,
        "sources": source_receipts,
        "assessment": completion.value,
        "provider": completion.provider,
        "model": completion.model,
        "max_output_tokens": completion.max_output_tokens,
        "prompt_sha256": completion.prompt_sha256,
        "response_sha256": completion.response_sha256,
        "usage": completion.usage,
        "usage_metering": usage_metering,
        "retention": "Full-text sentence content is transient and is not included in this artifact.",
    }
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=20) as client:
        response = await client.post(f"/v1/research/{session_id}/artifacts", json={"task_id": task_id, "kind": "paper_claim_semantic_alignment", "source": f"{completion.provider}/{completion.model} local screening", "source_id": hashlib.sha256((candidate_id_a + "\0" + candidate_id_b).encode()).hexdigest(), "payload": result})
        response.raise_for_status()
        artifact = response.json()
    return {"available": True, "artifact_id": artifact["id"], "assessment": completion.value, "usage": completion.usage, "usage_metering": usage_metering, "review_required": True}

async def _read_graph_pages(client: httpx.AsyncClient, session_id: str, max_records: int = 50000) -> dict:
    """Read a complete, bounded graph snapshot through the API keyset cursors."""
    nodes: list[dict] = []
    edges: list[dict] = []
    node_cursor: dict | None = None
    edge_cursor: dict | None = None
    nodes_done = False
    edges_done = False
    while True:
        params: dict[str, Any] = {"limit": 500, "include_nodes": not nodes_done, "include_edges": not edges_done}
        if node_cursor:
            params["after_node_label"] = node_cursor["label"]
            params["after_node_id"] = node_cursor["id"]
        if edge_cursor:
            params["after_edge_source"] = edge_cursor["source"]
            params["after_edge_relation"] = edge_cursor["relation"]
            params["after_edge_target"] = edge_cursor["target"]
        response = await client.get(f"/v1/research/{quote(session_id, safe='')}/graph", params=params)
        response.raise_for_status()
        page = response.json()
        if page.get("session_id") != session_id:
            raise ValueError("graph page belongs to a different research session")
        nodes.extend(page.get("nodes", []))
        edges.extend(page.get("edges", []))
        if len(nodes) > max_records or len(edges) > max_records:
            raise ValueError(f"graph exceeds the bounded snapshot limit of {max_records} nodes or edges")
        node_cursor = page.get("next_node_cursor")
        edge_cursor = page.get("next_edge_cursor")
        nodes_done = nodes_done or node_cursor is None
        edges_done = edges_done or edge_cursor is None
        if nodes_done and edges_done:
            return {"session_id": session_id, "nodes": nodes, "edges": edges}


@mcp.tool(name="medai_graph_search")
async def search_knowledge_graph(session_id: str, query: str) -> dict:
    """Find existing graph nodes by case-insensitive ID, label, or entity type."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=15) as client:
        response = await client.get(
            f"/v1/research/{quote(session_id, safe='')}/graph/search",
            params={"q": query.strip(), "limit": 100},
        )
        response.raise_for_status()
        result = response.json()
    if result.get("session_id") != session_id:
        raise ValueError("graph search returned a different research session")
    return result

@mcp.tool(name="medai_graph_snapshot")
async def get_knowledge_graph_snapshot(session_id: str) -> dict:
    """Read the complete session-scoped biomedical graph for auditing and reproduction."""
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        result = await _read_graph_pages(client, session_id)
    if any(node.get("session_id") != session_id for node in result.get("nodes", [])) or any(edge.get("session_id") != session_id for edge in result.get("edges", [])):
        raise ValueError("graph snapshot returned cross-session records")
    return result

@mcp.tool(name="medai_hypotheses_propose")
async def propose_research_hypotheses(session_id: str, task_id: str, limit: int = 50) -> dict:
    """Enumerate bounded evidence-backed simple paths of two to four edges."""
    if not 1 <= limit <= 200:
        raise ValueError("limit must be 1..200")
    await _charge(session_id, task_id, 1, "Evidence-linked hypothesis candidate generation")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        graph_result, evidence = await asyncio.gather(
            _read_graph_pages(client, session_id),
            _read_evidence_pages(client, session_id),
        )
        graph = graph_result
        candidates = propose_candidates(graph["nodes"], graph["edges"], evidence, limit)
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "hypothesis_candidates", "source": "MedAI graph-path engine", "source_id": session_id, "payload": candidates},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**candidates, "artifact_id": artifact["id"]}


@mcp.tool(name="medai_biomedical_domain_benchmark_evaluate")
async def evaluate_biomedical_domain_benchmark_in_session(
    session_id: str,
    task_id: str,
    input_artifact_id: str,
    bins: int = 10,
) -> dict[str, Any]:
    """Evaluate one exact retained benchmark input and keep only hash-bound metrics."""
    if task_id != "benchmark.evaluate":
        raise ValueError("biomedical benchmark evaluation requires the benchmark.evaluate task")
    if not isinstance(input_artifact_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", input_artifact_id):
        raise ValueError("input_artifact_id must be a bounded artifact identifier")
    await _charge(session_id, task_id, 1, "four-domain biomedical benchmark evaluation")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        source_response = await client.get(
            "/v1/research/" + quote(session_id, safe="") + "/artifacts/" + quote(input_artifact_id, safe="")
        )
        source_response.raise_for_status()
        source_artifact = source_response.json()
        if source_artifact.get("session_id") != session_id or source_artifact.get("id") != input_artifact_id:
            raise ValueError("benchmark input artifact is outside the requested research session")
        if (
            source_artifact.get("kind") != DOMAIN_BENCHMARK_ARTIFACT_KIND
            or source_artifact.get("task_id") != task_id
            or source_artifact.get("source") != DOMAIN_BENCHMARK_ARTIFACT_SOURCE
        ):
            raise ValueError("artifact is not a biomedical domain benchmark input for this task")
        raw_payload = source_artifact.get("payload")
        if not isinstance(raw_payload, str) or len(raw_payload.encode("utf-8")) > 4 * 1024 * 1024:
            raise ValueError("benchmark input payload is missing or exceeds 4 MiB")
        source_digest = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
        if source_digest != source_artifact.get("sha256"):
            raise ValueError("benchmark input artifact failed its retained payload hash check")
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("benchmark input contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_nonfinite(value):
        raise ValueError(f"benchmark input contains non-standard JSON number {value}")

    try:
        benchmark = json.loads(raw_payload, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("benchmark input artifact does not contain strict JSON") from error
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark input must be a JSON object")
    canonical_input = json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    benchmark_digest = hashlib.sha256(canonical_input).hexdigest()
    report = evaluate_biomedical_domain_benchmark(benchmark, bins=bins)
    if report["benchmark_sha256"] != benchmark_digest:
        raise ValueError("benchmark digest changed during evaluation")
    report["input_artifact_id"] = input_artifact_id
    report["input_artifact_sha256"] = source_digest
    report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(report_bytes) > 4 * 1024 * 1024:
        raise ValueError("biomedical benchmark evaluation report exceeds 4 MiB")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        response = await client.post(
            "/v1/research/" + quote(session_id, safe="") + "/artifacts",
            json={
                "task_id": task_id,
                "kind": "biomedical_domain_benchmark",
                "source": "MedAI source-bound biomedical benchmark evaluator",
                "source_id": input_artifact_id,
                "payload": report,
            },
        )
        response.raise_for_status()
        artifact = response.json()
    return {
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": benchmark_digest,
        "total_cases": report["total_cases"],
        "complete_domain_coverage": report["complete_domain_coverage"],
        "missing_domains": report["missing_domains"],
        "domains": {domain: {"status": result["status"], "cases": result.get("overall", {}).get("n", 0)} for domain, result in report["domains"].items()},
        "adjudication_integrity": report["adjudication_integrity"],
        "artifact_id": artifact["id"],
        "interpretation_limit": report["interpretation_limit"],
    }


@mcp.tool(name="medai_evidence_quality_benchmark_evaluate")
async def evaluate_evidence_quality_benchmark_in_session(
    session_id: str,
    task_id: str,
    input_artifact_id: str,
    bins: int = 10,
) -> dict[str, Any]:
    """Evaluate one retained, double-reviewed ordinal quality benchmark by domain."""
    if task_id != "benchmark.evaluate":
        raise ValueError("evidence-quality benchmark evaluation requires the benchmark.evaluate task")
    if not isinstance(input_artifact_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", input_artifact_id):
        raise ValueError("input_artifact_id must be a bounded artifact identifier")
    await _charge(session_id, task_id, 1, "Double-reviewed evidence-quality benchmark evaluation")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        response = await client.get(
            f"/v1/research/{quote(session_id, safe='')}/artifacts/{quote(input_artifact_id, safe='')}"
        )
        response.raise_for_status()
        source_artifact = response.json()
        if source_artifact.get("session_id") != session_id or source_artifact.get("id") != input_artifact_id:
            raise ValueError("benchmark input artifact is outside the requested research session")
        if (
            source_artifact.get("kind") != EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND
            or source_artifact.get("task_id") != task_id
            or source_artifact.get("source") != EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE
        ):
            raise ValueError("artifact is not a double-reviewed evidence-quality benchmark input for this task")
        raw_payload = source_artifact.get("payload")
        stored_digest = source_artifact.get("sha256")
        if not isinstance(raw_payload, str):
            raise ValueError("benchmark input payload is missing")
        raw = raw_payload.encode("utf-8")
        source_digest = hashlib.sha256(raw).hexdigest()
        if source_digest != stored_digest:
            raise ValueError("benchmark input artifact failed its retained payload hash check")
        _, report, benchmark_digest = evaluate_evidence_quality_benchmark_bytes(raw, bins=bins)
        if report.get("benchmark_sha256") != benchmark_digest:
            raise ValueError("benchmark digest changed during evaluation")
        report["input_artifact_id"] = input_artifact_id
        report["input_artifact_sha256"] = source_digest
        report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(report_bytes) > 4 * 1024 * 1024:
            raise ValueError("evidence-quality benchmark report exceeds 4 MiB")
        saved_response = await client.post(
            f"/v1/research/{quote(session_id, safe='')}/artifacts",
            json={
                "task_id": task_id,
                "kind": "evidence_quality_benchmark_report",
                "source": "MedAI double-reviewed evidence-quality benchmark evaluator",
                "source_id": input_artifact_id,
                "payload": report,
            },
        )
        saved_response.raise_for_status()
        saved_artifact = saved_response.json()
    return {
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": benchmark_digest,
        "input_artifact_id": input_artifact_id,
        "input_artifact_sha256": source_digest,
        "cases": report["cases"],
        "overall_by_domain": report["overall_by_domain"],
        "by_evidence_type": report["by_evidence_type"],
        "primary_review_agreement_by_domain": {
            domain: metrics["primary_reviewer_agreement"]
            for domain, metrics in report["overall_by_domain"].items()
        },
        "artifact_id": saved_artifact["id"],
        "interpretation_limit": report["interpretation_limit"],
    }

@mcp.tool(name="medai_hypotheses_criticize")
async def criticize_research_hypotheses(
    session_id: str,
    task_id: str,
    proposal_artifact_id: str,
    contradiction_artifact_ids: list[str] | None = None,
) -> dict:
    """Produce review leads and explicit weaknesses for retained candidate hypotheses."""
    contradiction_artifact_ids = [] if contradiction_artifact_ids is None else contradiction_artifact_ids
    if not isinstance(contradiction_artifact_ids, list) or len(contradiction_artifact_ids) > 50 or any(not isinstance(item, str) or not item or len(item) > 128 for item in contradiction_artifact_ids) or len(set(contradiction_artifact_ids)) != len(contradiction_artifact_ids):
        raise ValueError("contradiction_artifact_ids must contain at most 50 unique artifact IDs")
    await _charge(session_id, task_id, 1, "Hypothesis criticism and contradictory evidence review")
    async with httpx.AsyncClient(base_url=API, headers=_api_headers(), timeout=30) as client:
        artifacts_response, graph, evidence = await asyncio.gather(
            client.get(f"/v1/research/{session_id}/artifacts/{proposal_artifact_id}"),
            _read_graph_pages(client, session_id),
            _read_evidence_pages(client, session_id),
        )
        artifacts_response.raise_for_status()
        proposal_artifact = artifacts_response.json()
        proposal = validate_hypothesis_proposal_artifact(proposal_artifact, session_id)
        candidate_responses = await asyncio.gather(*(
            client.get(f"/v1/research/{quote(session_id, safe='')}/artifacts/{quote(artifact_id, safe='')}")
            for artifact_id in contradiction_artifact_ids
        ))
        contradiction_artifacts = []
        for candidate_response in candidate_responses:
            candidate_response.raise_for_status()
            candidate_artifact = candidate_response.json()
            if candidate_artifact.get("session_id") != session_id:
                raise ValueError("contradiction-search artifact is outside the requested research session")
            contradiction_artifacts.append(candidate_artifact)
        candidates = proposal.get("candidates", [])
        reviews = [critique_candidate(row, graph["nodes"], graph["edges"], evidence) for row in candidates]
        for candidate, review in zip(candidates, reviews):
            review["retrieved_contradiction_candidates"] = link_contradiction_search_candidates(candidate, contradiction_artifacts, evidence)
        supplied_candidate_ids = set(contradiction_artifact_ids)
        matched_candidate_ids = {
            item["candidate_artifact_id"]
            for review in reviews
            for item in review["retrieved_contradiction_candidates"]
        }
        try:
            provider = _model_provider_for_task(task_id, session_id)
        except (ValueError, TypeError):
            provider = None
        model_reviewed = 0
        model_review_errors = []
        if provider is not None:
            for candidate, review in zip(candidates[:10], reviews[:10]):
                try:
                    await _charge(session_id, task_id, 1, "model-assisted adversarial hypothesis review")
                    completion = await _run_cancel_aware(session_id, generate_critic_leads(provider, candidate, review, evidence))
                    usage_metering = await _charge_model_usage(session_id, task_id, completion.provider, completion.model, completion.usage)
                    review["model_generated_leads"] = completion.value
                    review["model_provenance"] = {"provider": completion.provider, "model": completion.model, "max_output_tokens": completion.max_output_tokens, "prompt_sha256": completion.prompt_sha256, "response_sha256": completion.response_sha256, "usage": completion.usage, "usage_metering": usage_metering}
                    model_reviewed += 1
                except ResearchSessionCancelled:
                    raise
                except (ModelUsageBudgetError, ModelUsageError):
                    raise
                except Exception as error:
                    review["model_criticism_status"] = "unavailable"
                    model_review_errors.append({"hypothesis_id": review.get("hypothesis_id"), "error": type(error).__name__})
        result = {
            "proposal_artifact_id": proposal_artifact_id,
            "proposal_artifact_sha256": proposal_artifact.get("sha256"),
            "reviews": reviews,
            "contradiction_search_artifacts_supplied": len(contradiction_artifacts),
            "contradiction_search_artifacts_linked": len(matched_candidate_ids),
            "contradiction_search_artifact_ids_unlinked": sorted(supplied_candidate_ids - matched_candidate_ids),
            "model_reviewed_count": model_reviewed,
            "model_review_errors": model_review_errors,
            "model_review_limit": 10,
            "interpretation": "Deterministic evidence leads are combined with optional model-generated questions; both require human scientific review and neither is a scientific verdict.",
        }
        response = await client.post(
            "/v1/research/" + session_id + "/artifacts",
            json={"task_id": task_id, "kind": "hypothesis_criticism", "source": "MedAI scientific critic", "source_id": proposal_artifact_id, "payload": result},
        )
        response.raise_for_status()
        artifact = response.json()
    return {**result, "artifact_id": artifact["id"]}

def main() -> None:
    transport = os.getenv("MEDAI_MCP_TRANSPORT", "streamable-http")
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("MEDAI_MCP_TRANSPORT must be stdio or streamable-http")
    if transport == "stdio":
        mcp.run(transport=transport)
    else:
        import uvicorn

        app = AuthenticatedMCPTransportASGIMiddleware(
            SpecialistCredentialASGIMiddleware(mcp.streamable_http_app(stateless_http=True))
        )
        uvicorn.run(app, host=os.getenv("MEDAI_MCP_HOST", "127.0.0.1"), port=int(os.getenv("MEDAI_MCP_PORT", "8091")))

if __name__ == "__main__":
    main()
