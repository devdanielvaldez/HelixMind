from __future__ import annotations
import asyncio
from dataclasses import asdict
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote, urlsplit
import httpx
import typer
import yaml
from medai_science.sources.pubmed import PubMedClient
from medai_science.sources.clinical_trials import ClinicalTrialsClient
from medai_science.sources.alphafold import AlphaFoldClient
from medai_science.sources.mygene import MyGeneClient
from medai_science.sources.clinvar import ClinVarClient
from medai_science.sources.dbsnp import DbSNPClient
from medai_science.sources.gnomad import GnomADClient
from medai_science.sources.uniprot import UniProtClient
from medai_science.sources.intact import IntActClient
from medai_science.sources.sifts import SIFTSClient
from medai_science.sources.reactome import ReactomeClient
from medai_science.sources.gene_disease import GeneDiseaseClient, reconcile_gene_disease_sources
from medai_science.sources.pmc_bioc import PMCBioCClient
from medai_science.sources.pubchem import PubChemClient
from medai_science.sources.chembl import ChEMBLClient
from medai_science.sources.openfda_labels import OpenFDALabelClient
from medai_science.sources.crossref import CrossrefClient
from medai_science.sources.preprint import PreprintClient
from medai_science.sources.ontology import OntologyClient
from medai_science.sources.ensembl_vep import EnsemblVEPClient
from medai_science.sources.panelapp import PanelAppClient
from medai_science.sources.gencc import GenCCClient
from medai_science.sources.clingen_gene_validity import ClinGenGeneValidityClient
from medai_science.sources.orphadata import OrphadataClient
from medai_science.agentmesh_control import AgentMeshControlPlaneClient, build_capability_route_plan, load_gateway_upstream
from medai_science.amcp_export import export_generated_amcp_artifact
from medai_science.agentmesh_sync import AgentMeshSyncError, inspect_capability_sync, reconcile_runtime_lifecycle, seal_route_plan_for_review, sync_capability_routes
from medai_science.genomics import analyze_vcf, inspect_fasta, inspect_fastq
from medai_science.genomics_research import summarize_local_vcf_report
from medai_science.json_input import read_strict_json
from medai_science.structure_analysis import MAX_MMCIF_BYTES, MAX_PDB_BYTES, summarize_mmcif, summarize_pdb
from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
from medai_science.dbsnp_reconciliation import reconcile_dbsnp_placements
from medai_science.variant_source_reconciliation import reconcile_gnomad_frequency, reconcile_variant_sources
from medai_science.orchestrator import plan_research
from medai_science.capabilities import SemanticCapabilityIndex, LocalSentenceTransformer, apply_authoritative_lifecycle, canonical_amcp_registry_package, draft_manifest, load_amcp_catalog, runtime_capability_statuses
from medai_science.capability_lifecycle import fetch_approved_lifecycle, fetch_capability_registry
from medai_science.reproducibility import compare_reproducibility_bundles, create_chembl_bundle, create_clinical_trial_bundle, create_compound_bundle, create_genomics_bundle, create_intact_bundle, create_pathway_bundle, create_protein_bundle, create_sequence_bundle, create_structure_bundle, parse_research_bundle_artifact, verify_analysis_bundle
from medai_science.sandbox import DockerPythonSandbox
from medai_science.r_sandbox import DockerRSandbox
from medai_science.wasm_sandbox import DockerWasmSandbox
from medai_science.evaluation import calibrate_capability_reuse_threshold, evaluate_adjudicated_cases, evaluate_biomedical_domain_benchmark, evaluate_drug_pair_benchmark, load_pinned_capability_calibration, write_private_report_no_overwrite
from medai_science.benchmark_artifacts import load_domain_benchmark_artifact, load_evidence_quality_benchmark_artifact, submit_domain_benchmark_artifact, submit_evidence_quality_benchmark_artifact
from medai_science.capability_evolution import compare_capability_versions, run_paired_capability_benchmark
from medai_science.skill_learning import read_skill_index
from medai_science.provenance_signing import verify_capability_proposal_signature
from medai_science.skill_export import read_verified_skill, render_agentmesh_skill_review_package, register_agentmesh_skill_review_package
from medai_science.skill_sync import build_signed_skill_route_plan, reconcile_skill_agentmesh_status
from medai_science.research_cli import cancel_session, get_session_status, list_sessions, watch_session

app = typer.Typer(help="Biomedical computational research source adapters")
agentmesh_app = typer.Typer(help="Authenticated AgentMesh administrative control-plane operations")
app.add_typer(agentmesh_app, name="agentmesh")
skills_app = typer.Typer(help="Locally learned, review-gated biomedical Skills")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(registry_dir: Path = typer.Option(Path(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills")), exists=False, file_okay=False, dir_okay=True)):
    """List locally persisted Skill candidates and verify their installation signatures."""
    try:
        index = read_skill_index(registry_dir)
        packages = []
        for manifest_path in sorted(registry_dir.glob("*/*/skill.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification = verify_capability_proposal_signature(manifest)
            if not verification.get("verified"):
                raise ValueError(f"Skill signature verification failed: {manifest_path}")
            packages.append({"id": manifest.get("id"), "version": manifest.get("version"), "status": manifest.get("status"), "review_required": manifest.get("review_required"), "execution": manifest.get("execution"), "path": str(manifest_path.parent)})
        typer.echo(json.dumps({"skills": packages, "count": len(packages), "consumed_sessions": len(index.get("consumed_session_ids", [])), "execution": "disabled pending review"}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@skills_app.command("export-agentmesh")
def skills_export_agentmesh(
    skill_id: str,
    version: str,
    output: Path,
    registry_dir: Path = typer.Option(Path(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills")), exists=False, file_okay=False, dir_okay=True),
):
    """Export one verified disabled Skill as an AgentMesh review-only AMCP package."""
    try:
        skill = read_verified_skill(registry_dir, skill_id, version)
        package_yaml, package_sha256 = render_agentmesh_skill_review_package(skill)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(package_yaml)
        output.chmod(0o600)
    except FileExistsError as error:
        raise typer.BadParameter("refusing to overwrite an existing AgentMesh package") from error
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps({
        "skill_id": skill_id,
        "version": version,
        "package_path": str(output.resolve()),
        "package_sha256": package_sha256,
        "signature_verified": True,
        "review_required": True,
        "execution_enabled": False,
        "next_steps": [f"agentmesh capabilities validate {output}", f"agentmesh capabilities install {output} --catalog $MEDAI_AGENTMESH_CATALOG"],
    }, ensure_ascii=False, indent=2))


@skills_app.command("register-agentmesh")
def skills_register_agentmesh(
    skill_id: str,
    version: str,
    catalog: Path = typer.Option(..., "--catalog", exists=False, file_okay=False, dir_okay=True),
    registry_dir: Path = typer.Option(Path(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills")), exists=False, file_okay=False, dir_okay=True),
    agentmesh_cli: str | None = typer.Option(None, "--agentmesh-cli", envvar="AGENTMESH_CLI"),
):
    """Validate and install a signed Skill review package into a local AgentMesh catalog."""
    try:
        skill = read_verified_skill(registry_dir, skill_id, version)
        receipt = register_agentmesh_skill_review_package(skill, catalog, agentmesh_cli)
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))


@skills_app.command("agentmesh-status")
def skills_agentmesh_status(
    registry_dir: Path = typer.Option(Path(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills")), exists=True, file_okay=False, dir_okay=True),
    catalog_dir: Path = typer.Option(Path(os.getenv("MEDAI_AGENTMESH_CAPABILITY_DIR", "integrations/agentmesh/capabilities")), exists=True, file_okay=False, dir_okay=True),
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url"),
    service: str = typer.Option(os.getenv("MEDAI_AGENTMESH_SERVICE", ""), "--service"),
):
    """Reconcile signed local Skills against exact MedAI lifecycle and AgentMesh route state."""
    try:
        if not service:
            raise ValueError("Set MEDAI_AGENTMESH_SERVICE or pass --service")
        skills = []
        manifest_paths = sorted(registry_dir.glob("*/*/skill.json"))
        if len(manifest_paths) > 1000:
            raise ValueError("local Skill registry exceeds 1000 packages")
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("id"), str) or not isinstance(manifest.get("version"), str):
                raise ValueError(f"local Skill manifest is malformed: {manifest_path}")
            skills.append(read_verified_skill(registry_dir, manifest["id"], manifest["version"]))
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required to reconcile approved Skill lifecycle")
        _, lifecycle_digest, registry_rows = fetch_capability_registry(api_url, token)
        runtime = asyncio.run(_agentmesh_control_client().authenticated_runtime_snapshot())
        report = reconcile_skill_agentmesh_status(
            skills,
            load_amcp_catalog(catalog_dir),
            registry_rows,
            runtime,
            service,
        )
        report["lifecycle_registry_sha256"] = lifecycle_digest
        report["summary"] = {
            "approved_and_routed_review_only": sum(row["reconciliation"] == "approved_and_routed_review_only" for row in report["skills"]),
            "awaiting_administrative_approval": sum(row["reconciliation"] == "awaiting_administrative_approval" for row in report["skills"]),
            "attention_required": sum(row["reconciliation"] not in {"approved_and_routed_review_only", "awaiting_administrative_approval"} for row in report["skills"]),
        }
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@skills_app.command("plan-agentmesh")
def skills_plan_agentmesh(
    service: str = typer.Option(os.getenv("MEDAI_AGENTMESH_SERVICE", ""), "--service"),
    registry_dir: Path = typer.Option(Path(os.getenv("MEDAI_SKILLS_DIR", ".medai/skills")), exists=True, file_okay=False, dir_okay=True),
    catalog_dir: Path = typer.Option(Path(os.getenv("MEDAI_AGENTMESH_CAPABILITY_DIR", "integrations/agentmesh/capabilities")), exists=True, file_okay=False, dir_okay=True),
    gateway_config: Path = typer.Option(Path("integrations/agentmesh/config/agentmesh-medai.yaml"), exists=True, dir_okay=False, readable=True),
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url"),
    output: Path | None = typer.Option(None, "--output", help="Write the signed-Skill route plan for explicit review"),
):
    """Plan routes from exact signed Skill packages and authenticated MedAI lifecycle only."""
    try:
        if not service:
            raise ValueError("Set MEDAI_AGENTMESH_SERVICE or pass --service")
        manifest_paths = sorted(registry_dir.glob("*/*/skill.json"))
        if len(manifest_paths) > 1000:
            raise ValueError("local Skill registry exceeds 1000 packages")
        skills = []
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("id"), str) or not isinstance(manifest.get("version"), str):
                raise ValueError(f"local Skill manifest is malformed: {manifest_path}")
            skills.append(read_verified_skill(registry_dir, manifest["id"], manifest["version"]))
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required to plan runtime routes")
        _, registry_sha256, registry_rows = fetch_capability_registry(api_url, token)
        plan = build_signed_skill_route_plan(
            skills,
            load_amcp_catalog(catalog_dir),
            registry_rows,
            registry_sha256,
            service,
            load_gateway_upstream(gateway_config),
            hashlib.sha256(gateway_config.read_bytes()).hexdigest(),
            os.getenv("AGENTMESH_TENANT", "medai"),
            os.getenv("AGENTMESH_NAMESPACE", "biomedical"),
        )
        rendered = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
            output.chmod(0o600)
        typer.echo(rendered)
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@skills_app.command("sync-agentmesh")
def skills_sync_agentmesh(
    plan_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    approved_plan_sha256: str = typer.Option(..., "--approve-plan-sha256"),
    approval_receipt_id: str = typer.Option(..., "--approval-receipt-id"),
    state_file: Path = typer.Option(Path(os.getenv("MEDAI_AGENTMESH_SKILLS_SYNC_STATE", ".medai/agentmesh/skills-sync-state.json")), "--state-file"),
):
    """Apply only the exact reviewed Skill route plan and verify the authenticated runtime."""
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or plan.get("lifecycle_source") != "authenticated-medai-api-exact-package-match":
            raise ValueError("Skill route sync requires a plan from skills plan-agentmesh")
        if not re.fullmatch(r"[a-f0-9]{64}", approved_plan_sha256) or plan.get("plan_review_sha256") != approved_plan_sha256:
            raise ValueError("approved plan digest must exactly match the reviewed Skill route plan")
        approval = _medai_admin_request(
            f"/v1/admin/agentmesh/route-approvals/{quote(approval_receipt_id, safe='')}/validate",
            {"plan": plan},
        )
        receipt = asyncio.run(sync_capability_routes(
            _agentmesh_control_client(), plan, state_file,
            approved_plan_sha256=approved_plan_sha256, approval_receipt=approval,
        ))
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError, AgentMeshSyncError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))


def _agentmesh_control_client() -> AgentMeshControlPlaneClient:
    base_url = os.getenv("AGENTMESH_CONTROL_URL", "")
    token = os.getenv("AGENTMESH_ADMIN_TOKEN", "")
    if not base_url or not token:
        raise typer.BadParameter("Set AGENTMESH_CONTROL_URL and AGENTMESH_ADMIN_TOKEN")
    try:
        return AgentMeshControlPlaneClient(
            base_url,
            token,
            os.getenv("AGENTMESH_TENANT", "medai"),
            os.getenv("AGENTMESH_NAMESPACE", "biomedical"),
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _medai_admin_request(path: str, body: dict | None = None, *, method: str = "POST") -> dict:
    token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
    if not token:
        raise ValueError("MEDAI_API_BEARER_TOKEN is required for authenticated MedAI API requests")
    base_url = os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("MedAI API URL must be absolute HTTP(S) without credentials, query, or fragment")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not loopback:
            raise ValueError("MedAI API bearer tokens require HTTPS except for loopback API servers")
    with httpx.Client(
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=20, follow_redirects=False, trust_env=False,
    ) as client:
        response = client.request(method, base_url + path, json=body)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise ValueError("MedAI API returned a malformed JSON object")
    return result


@app.command("research-create")
def research_create(
    objective: str,
    depth: str = typer.Option("standard", "--depth", case_sensitive=False),
    budget: int = typer.Option(100, "--budget", min=1),
):
    """Create a tenant-scoped research session through the authenticated MedAI API."""
    if depth not in {"standard", "deep"}:
        raise typer.BadParameter("depth must be standard or deep")
    if not objective.strip():
        raise typer.BadParameter("objective must not be empty")
    try:
        session = _medai_admin_request(
            "/v1/research", {"objective": objective, "depth": depth, "budget": budget}
        )
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"could not create research session: {error}") from error
    typer.echo(json.dumps(session, ensure_ascii=False, indent=2))


@app.command("research-list")
def research_list():
    """List research sessions visible to the authenticated API tenant."""
    try:
        result = list_sessions(_medai_admin_request)
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"could not list research sessions: {error}") from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("research-status")
def research_status(session_id: str):
    """Read one authenticated session's operational task and budget status."""
    try:
        result = get_session_status(_medai_admin_request, session_id)
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"could not read research status: {error}") from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("research-watch")
def research_watch(
    session_id: str,
    poll_seconds: int = typer.Option(30, "--poll-seconds", min=5, max=3600),
):
    """Poll authenticated research status until the DAG has no ready or running work."""
    try:
        watch_session(
            _medai_admin_request,
            session_id,
            poll_seconds,
            emit=lambda result: typer.echo(json.dumps(result, ensure_ascii=False), color=False),
        )
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"research watch stopped: {error}") from error


@app.command("research-cancel")
def research_cancel(session_id: str):
    """Request cooperative cancellation of an authenticated tenant research session."""
    try:
        result = cancel_session(_medai_admin_request, session_id)
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"could not cancel research session: {error}") from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@agentmesh_app.command("status")
def agentmesh_status():
    """Read authenticated control-plane health/status."""
    try:
        typer.echo(json.dumps(asyncio.run(_agentmesh_control_client().status()), ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("snapshot")
def agentmesh_snapshot():
    """Read the active AgentMesh runtime snapshot for the configured scope."""
    try:
        typer.echo(json.dumps(asyncio.run(_agentmesh_control_client().snapshot()), ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("apply")
def agentmesh_apply(resource_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True), expected_revision: int | None = typer.Option(None, "--expected-revision", min=0)):
    """Apply one explicit JSON desired-state resource using optimistic revision checks."""
    try:
        resource = json.loads(resource_file.read_text(encoding="utf-8"))
        result = asyncio.run(_agentmesh_control_client().apply(resource, expected_revision))
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("reconcile")
def agentmesh_reconcile():
    """Reconcile desired state into the active AgentMesh gateway snapshot."""
    try:
        typer.echo(json.dumps(asyncio.run(_agentmesh_control_client().reconcile()), ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("register-capabilities")
def agentmesh_register_capabilities(
    catalog_dir: Path = typer.Option(Path("integrations/agentmesh/capabilities"), exists=True, file_okay=False, readable=True),
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url"),
):
    """Import AMCP metadata into MedAI as disabled pending-review proposals; never promote or enable routes."""
    try:
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required; registration requires a platform_admin credential")
        # Validate the origin and credential policy before fetching the full metadata records.
        fetch_approved_lifecycle(api_url, token)
        with httpx.Client(base_url=api_url.rstrip("/"), headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=15, follow_redirects=False) as client:
            response = client.get("/v1/capabilities")
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ValueError("MedAI capability registry response exceeds 2 MiB")
            rows = response.json().get("items")
            if not isinstance(rows, list):
                raise ValueError("MedAI capability registry response is malformed")
            existing = {}
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("version"), str):
                    raise ValueError("MedAI capability registry contains a malformed identity")
                existing[f"{row['id']}@{row['version']}"] = row
            registered = []
            unchanged = []
            for package in load_amcp_catalog(catalog_dir):
                capability = package["capability"]
                capability_id, version = capability["id"], capability["version"]
                key = f"{capability_id}@{version}"
                # Lifecycle labels in the checked-in AMCP are not approval; the registry starts at pending_review.
                package, package_sha256 = canonical_amcp_registry_package(package)
                prior = existing.get(key)
                if prior is not None:
                    try:
                        prior_manifest = json.loads(prior.get("manifest", "{}"))
                    except (TypeError, json.JSONDecodeError) as error:
                        raise ValueError(f"existing capability {key} has an unreadable registry manifest") from error
                    if prior_manifest.get("source") != "agentmesh_amcp" or prior_manifest.get("package_sha256") != package_sha256:
                        raise ValueError(f"existing capability {key} differs from this AMCP package; refusing to overwrite")
                    unchanged.append(key)
                    continue
                result = client.post("/v1/capabilities", json={"id": capability_id, "version": version, "kind": "mcp", "description": capability.get("intent") or capability_id, "package": package})
                result.raise_for_status()
                registered.append(key)
        typer.echo(json.dumps({"registered_pending_review": registered, "already_registered_unchanged": unchanged, "automatic_promotion": False}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("export-generated-amcp")
def agentmesh_export_generated_amcp(
    session_id: str,
    artifact_id: str,
    output: Path,
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url"),
):
    """Verify and export a signed generated AMCP artifact for human review and AgentMesh validation."""
    try:
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required to retrieve the tenant-scoped proposal")
        result = export_generated_amcp_artifact(api_url, token, session_id, artifact_id, output)
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("register-generated-amcp")
def agentmesh_register_generated_amcp(
    session_id: str,
    artifact_id: str,
    catalog: Path = typer.Option(..., "--catalog", exists=False, file_okay=False, dir_okay=True),
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url"),
    agentmesh_cli: str | None = typer.Option(None, "--agentmesh-cli", envvar="AGENTMESH_CLI"),
):
    """Verify a generated package, validate/install it locally, and register it pending review."""
    try:
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required; registration requires platform_admin")
        executable = agentmesh_cli or shutil.which("agentmesh")
        if not executable:
            raise ValueError("AgentMesh CLI is required; set AGENTMESH_CLI or add agentmesh to PATH")
        # Verify both signed source artifacts before passing a package to either registry.
        with tempfile.TemporaryDirectory(prefix="medai-generated-amcp-") as temporary:
            package_path = Path(temporary) / "generated-review.yaml"
            export_generated_amcp_artifact(api_url, token, session_id, artifact_id, package_path)
            package = yaml.safe_load(package_path.read_text(encoding="utf-8"))
            capability = package["capability"]
            for command in (
                [executable, "capabilities", "validate", str(package_path)],
                [executable, "capabilities", "install", "--catalog", str(catalog.expanduser().resolve()), str(package_path)],
            ):
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30, shell=False)
                if result.returncode != 0:
                    detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()[:2000]
                    raise ValueError(f"AgentMesh package registration failed: {detail}")
            canonical, package_sha256 = canonical_amcp_registry_package(package)
            request_package = canonical
            with httpx.Client(
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                timeout=20, follow_redirects=False, trust_env=False,
            ) as client:
                response = client.post(api_url.rstrip("/") + "/v1/capabilities", json={
                    "id": capability["id"], "version": capability["version"], "kind": "mcp",
                    "description": capability["intent"], "package": request_package,
                })
                response.raise_for_status()
                registered = response.json()
        typer.echo(json.dumps({
            "capability_id": capability["id"], "version": capability["version"],
            "package_sha256": package_sha256, "catalog": str(catalog.expanduser().resolve()),
            "registry": registered, "registered_pending_review": True,
            "execution_enabled": False, "route_enabled": False,
            "agentmesh_validation": result.stdout.strip()[:1000],
        }, ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("plan-capabilities")
def agentmesh_plan_capabilities(
    service: str = typer.Option(..., "--service", help="AgentMesh registered service identifier for the MedAI MCP server"),
    catalog_dir: Path = typer.Option(Path("integrations/agentmesh/capabilities"), exists=True, file_okay=False, readable=True),
    gateway_config: Path = typer.Option(Path("integrations/agentmesh/config/agentmesh-medai.yaml"), exists=True, dir_okay=False, readable=True),
    status_overrides_json: Path | None = typer.Option(None, exists=True, dir_okay=False, readable=True, help="Reviewed JSON map from capability-id@version to lifecycle status"),
    use_approved_api_lifecycle: bool = typer.Option(False, "--use-approved-api-lifecycle", help="Fetch authenticated lifecycle decisions from the MedAI API registry"),
    output: Path | None = typer.Option(None, "--output", help="Write the reviewable plan JSON to a local file"),
):
    """Build an audited desired-route plan from AMCP lifecycle metadata; does not apply it."""
    try:
        if use_approved_api_lifecycle and status_overrides_json is not None:
            raise ValueError("choose either the authenticated API registry or a reviewed local override file")
        registry_sha256 = None
        lifecycle_source = "local-catalog-defaults"
        if use_approved_api_lifecycle:
            token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
            if not token:
                raise ValueError("MEDAI_API_BEARER_TOKEN is required to read approved lifecycle decisions")
            api_url = os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090")
            registry_statuses, registry_sha256 = fetch_approved_lifecycle(api_url, token)
            catalog_packages = load_amcp_catalog(catalog_dir)
            catalog_keys = {
                f"{package['capability']['id']}@{package['capability']['version']}"
                for package in catalog_packages
            }
            matched = catalog_keys.intersection(registry_statuses)
            if not matched:
                raise ValueError("authenticated MedAI registry has no capability versions in the selected AgentMesh catalog")
            # The API is authoritative: a catalog package absent from it remains experimental.
            overrides = {key: "experimental" for key in catalog_keys}
            overrides.update({key: registry_statuses[key] for key in matched})
            lifecycle_source = "authenticated-medai-api"
        else:
            overrides = json.loads(status_overrides_json.read_text(encoding="utf-8")) if status_overrides_json else None
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("capability lifecycle overrides must be a JSON object")
        packages = load_amcp_catalog(catalog_dir, status_overrides=overrides)
        plan = build_capability_route_plan(packages, service)
        upstream = load_gateway_upstream(gateway_config)
        plan["gateway_upstream"] = upstream
        plan["gateway_config_sha256"] = hashlib.sha256(gateway_config.read_bytes()).hexdigest()
        plan["lifecycle_source"] = lifecycle_source
        plan["lifecycle_registry_sha256"] = registry_sha256
        plan["lifecycle_overrides_sha256"] = (
            registry_sha256 if use_approved_api_lifecycle else
            hashlib.sha256(status_overrides_json.read_bytes()).hexdigest() if status_overrides_json else None
        )
        plan = seal_route_plan_for_review(
            plan,
            os.getenv("AGENTMESH_TENANT", "medai"),
            os.getenv("AGENTMESH_NAMESPACE", "biomedical"),
        )
        rendered = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(rendered)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@agentmesh_app.command("sync-capabilities")
def agentmesh_sync_capabilities(
    plan_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    approved_plan_sha256: str = typer.Option(..., "--approve-plan-sha256", help="Exact plan_review_sha256 inspected before rollout"),
    approval_receipt_id: str = typer.Option(..., "--approval-receipt-id", help="Unexpired receipt created by agentmesh approve-plan"),
    state_file: Path = typer.Option(Path(os.getenv("MEDAI_AGENTMESH_SYNC_STATE", ".medai/agentmesh/sync-state.json")), "--state-file"),
):
    """Apply an inspected route plan, verify the reconciled snapshot, and roll back failed updates."""
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("route plan must be a JSON object")
        if not re.fullmatch(r"[a-f0-9]{64}", approved_plan_sha256) or plan.get("plan_review_sha256") != approved_plan_sha256:
            raise ValueError("approved plan digest must exactly match the reviewed AgentMesh route plan")
        approval = _medai_admin_request(
            f"/v1/admin/agentmesh/route-approvals/{quote(approval_receipt_id, safe='')}/validate",
            {"plan": plan},
        )
        receipt = asyncio.run(sync_capability_routes(
            _agentmesh_control_client(), plan, state_file,
            approved_plan_sha256=approved_plan_sha256, approval_receipt=approval,
        ))
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError, AgentMeshSyncError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))


@agentmesh_app.command("approve-plan")
def agentmesh_approve_plan(
    plan_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
):
    """Record platform-admin approval for the exact reviewed AgentMesh route plan."""
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("route plan must be a JSON object")
        receipt = _medai_admin_request("/v1/admin/agentmesh/route-approvals", {"plan": plan})
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))


@agentmesh_app.command("sync-status")
def agentmesh_sync_status(
    state_file: Path = typer.Option(Path(os.getenv("MEDAI_AGENTMESH_SYNC_STATE", ".medai/agentmesh/sync-state.json")), "--state-file"),
):
    """Compare the signed local route journal with live AgentMesh state without changing either."""
    try:
        report = asyncio.run(inspect_capability_sync(_agentmesh_control_client(), state_file))
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required to reconcile live routes with MedAI lifecycle decisions")
        api_url = os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090")
        _, registry_sha256, registry_rows = fetch_capability_registry(api_url, token)
        lifecycle = reconcile_runtime_lifecycle(report, registry_rows, registry_sha256)
        report["lifecycle_reconciliation"] = lifecycle
        receipt_id = report.get("last_approval_receipt_id")
        if receipt_id:
            try:
                remote_receipt = _medai_admin_request(
                    f"/v1/admin/agentmesh/route-approvals/{quote(receipt_id, safe='')}",
                    method="GET",
                )
                receipt_status = "matched" if remote_receipt == report.get("approval_receipt") else "mismatch"
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    raise
                remote_receipt = None
                receipt_status = "missing"
        elif report.get("state_file_present"):
            remote_receipt = None
            receipt_status = "unverified_legacy_journal"
        else:
            remote_receipt = None
            receipt_status = "not_applicable"
        report["approval_receipt_reconciliation"] = {
            "status": receipt_status,
            "receipt_id": receipt_id,
            "remote_receipt_present": remote_receipt is not None,
            "mutation_performed": False,
        }
        report["cross_system_status"] = (
            "drift"
            if report["status"] in {"drift", "unjournaled_routes", "recovery_required"}
            or lifecycle["status"] == "drift"
            or receipt_status in {"missing", "mismatch"}
            else "unverified" if receipt_status == "unverified_legacy_journal"
            else "in_sync"
        )
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command("capability-run-python")
def capability_run_python(
    source_file: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="Proposed Python capability source; it must define run(input) -> dict"),
    input_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON object passed to run()"),
    test_cases_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON list of {input, expected} validation cases"),
    image: str = typer.Option(..., envvar="MEDAI_PYTHON_SANDBOX_IMAGE", help="Locally installed Docker image pinned by sha256 digest"),
    timeout: int = typer.Option(60, min=1, max=60),
):
    """Run an experimental Python capability in a no-network, resource-limited Docker sandbox."""
    try:
        code = source_file.read_text(encoding="utf-8")
        input_data = json.loads(input_json.read_text(encoding="utf-8"))
        test_cases = json.loads(test_cases_json.read_text(encoding="utf-8"))
        result = DockerPythonSandbox(image=image, timeout=timeout).execute(code, input_data, test_cases)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if not result.get("case_tests_passed"):
        raise typer.Exit(code=1)


@app.command("capability-run-r")
def capability_run_r(
    source_file: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="Proposed R capability source; it must define run <- function(input)"),
    input_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON object passed to run()"),
    test_cases_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON list of {input, expected} validation cases"),
    image: str = typer.Option(..., envvar="MEDAI_R_SANDBOX_IMAGE", help="Locally installed R image pinned by sha256 digest; it must include jsonlite"),
    timeout: int = typer.Option(60, min=1, max=60),
):
    """Run an experimental R capability in a no-network, resource-limited Docker sandbox."""
    try:
        code = source_file.read_text(encoding="utf-8")
        input_data = json.loads(input_json.read_text(encoding="utf-8"))
        test_cases = json.loads(test_cases_json.read_text(encoding="utf-8"))
        result = DockerRSandbox(image=image, timeout=timeout).execute(code, input_data, test_cases)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if not result.get("case_tests_passed"):
        raise typer.Exit(code=1)


@app.command("capability-run-wasm")
def capability_run_wasm(
    module_file: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="WASI Preview 1 version 1 module implementing JSON stdin/stdout"),
    input_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON object written to the guest stdin"),
    test_cases_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON list of {input, expected} validation cases"),
    image: str = typer.Option(..., envvar="MEDAI_WASM_SANDBOX_IMAGE", help="Locally cached Wasmtime image pinned by sha256 digest"),
    contract_json: Path | None = typer.Option(None, exists=True, dir_okay=False, readable=True, help="Optional JSON object with input and output schemas"),
    timeout: int = typer.Option(60, min=1, max=60),
    fuel: int = typer.Option(100_000_000, min=1, max=100_000_000),
):
    """Run an experimental WASI module in a no-network, fuel- and memory-limited sandbox."""
    try:
        input_data = json.loads(input_json.read_text(encoding="utf-8"))
        test_cases = json.loads(test_cases_json.read_text(encoding="utf-8"))
        contract = json.loads(contract_json.read_text(encoding="utf-8")) if contract_json is not None else None
        result = DockerWasmSandbox(image=image, timeout=timeout, fuel=fuel).execute(
            module_file.read_bytes(), input_data, test_cases, contract=contract,
        )
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if not result.get("case_tests_passed"):
        raise typer.Exit(code=1)

@app.command()
def plan(objective: str, depth: str = "standard", budget: int = 100):
    """Create a deterministic DAG plan; this describes work and does not execute it."""
    typer.echo(json.dumps(asdict(plan_research(objective, depth, budget)), indent=2, default=str))

@app.command("capability-draft")
def capability_draft(capability_id: str, description: str, session_id: str):
    """Emit a disabled capability proposal for review, never executable code."""
    typer.echo(json.dumps(draft_manifest(capability_id, description, session_id), indent=2))

@app.command("capabilities-search")
def capabilities_search(
    need: str,
    requirement_json: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSON object with inputs, outputs, permitted_effects, and permitted_permissions"),
    model_path: Path = typer.Option(..., exists=True, file_okay=False, readable=True, envvar="MEDAI_EMBEDDING_MODEL"),
    catalog_dir: Path = typer.Option(Path("integrations/agentmesh/capabilities"), exists=True, file_okay=False, readable=True),
    index_path: Path = typer.Option(Path(".medai/capability-vectors.sqlite")),
    api_url: str = typer.Option(os.getenv("MEDAI_API_URL", "http://127.0.0.1:8090"), "--api-url", help="MedAI API origin that owns approved capability lifecycle state"),
    calibration_file: Path | None = typer.Option(None, "--calibration-file", exists=True, dir_okay=False, readable=True, envvar="MEDAI_CAPABILITY_CALIBRATION_FILE"),
    calibration_sha256: str = typer.Option(os.getenv("MEDAI_CAPABILITY_CALIBRATION_SHA256", ""), "--calibration-sha256"),
    limit: int = 10,
):
    """Rank capabilities using contracts, lifecycle metadata, and live AgentMesh routes."""
    try:
        requirement = json.loads(requirement_json.read_text(encoding="utf-8"))
        if not isinstance(requirement, dict):
            raise ValueError("requirement JSON must contain an object")
        token = os.getenv("MEDAI_API_BEARER_TOKEN", "")
        if not token:
            raise ValueError("MEDAI_API_BEARER_TOKEN is required to read authoritative capability lifecycle state")
        registry_statuses, registry_sha256 = fetch_approved_lifecycle(api_url, token)
        packages = apply_authoritative_lifecycle(load_amcp_catalog(catalog_dir), registry_statuses)
        if bool(calibration_file) != bool(calibration_sha256):
            raise ValueError("calibrated reuse requires both --calibration-file and its operator-pinned --calibration-sha256")
        embedder = LocalSentenceTransformer(model_path)
        calibration = load_pinned_capability_calibration(calibration_file, calibration_sha256, expected_embedding_model_id=embedder.model_id) if calibration_file else None
        service = os.getenv("MEDAI_AGENTMESH_SERVICE", "")
        if not service:
            raise ValueError("Set MEDAI_AGENTMESH_SERVICE to the exact AgentMesh registered service ID")
        control = _agentmesh_control_client()
        runtime_snapshot = asyncio.run(control.authenticated_runtime_snapshot())
        runtime_statuses = runtime_capability_statuses(packages, runtime_snapshot, service)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index = SemanticCapabilityIndex(index_path, embedder)
        try:
            indexed = index.index(packages)
            results = index.search(
                need,
                requirement,
                limit=limit,
                reuse_similarity_threshold=calibration["threshold"] if calibration else None,
                permitted_effects=set(requirement.get("permitted_effects", [])),
                permitted_permissions=set(requirement.get("permitted_permissions", [])),
                runtime_statuses=runtime_statuses,
            )
        finally:
            index.close()
        typer.echo(json.dumps({"indexed_or_updated": indexed, "results": results, "recommendation": results[0]["decision"] if results else "build", "lifecycle_source": "authenticated MedAI capability registry", "lifecycle_registry_sha256": registry_sha256, "reuse_calibration": calibration or {"status": "not_configured", "reuse_enabled": False}, "runtime_snapshot": {key: runtime_snapshot.get(key) for key in ("control_plane_status", "tenant", "namespace", "revision", "digest", "empty")}, "execution": "search only; no capability was invoked"}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command()
def papers(query: str, limit: int = 20):
    """Search PubMed and return structured papers with source provenance."""
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise typer.BadParameter("Set NCBI_EMAIL before calling NCBI E-utilities")
    result = asyncio.run(PubMedClient(email, os.environ.get("NCBI_API_KEY")).search_and_fetch(query, limit))
    for paper in result:
        typer.echo(paper.model_dump_json())


@app.command("benchmark-evaluate")
def benchmark_evaluate(
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    bins: int = typer.Option(10, min=2, max=100, help="Equal-width calibration bins for prediction correctness"),
):
    """Evaluate predictions against a frozen independently adjudicated JSON benchmark."""
    try:
        if benchmark_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("benchmark file exceeds 20 MiB")
        document = json.loads(benchmark_file.read_text(encoding="utf-8"))
        typer.echo(json.dumps(evaluate_adjudicated_cases(document, bins=bins), ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("drug-pair-benchmark-evaluate")
def drug_pair_benchmark_evaluate(
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    bins: int = typer.Option(10, min=2, max=100, help="Equal-width calibration bins for prediction correctness"),
):
    """Evaluate source-bound drug-pair retrieval and conditional conflict screening."""
    try:
        if benchmark_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("drug-pair benchmark file exceeds 20 MiB")

        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_nonfinite(value):
            raise ValueError(f"non-finite JSON number is not allowed: {value}")

        document = json.loads(
            benchmark_file.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
        report = evaluate_drug_pair_benchmark(document, bins=bins)
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("biomedical-domain-benchmark-evaluate")
def biomedical_domain_benchmark_evaluate(
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    bins: int = typer.Option(10, min=2, max=100, help="Equal-width calibration bins for prediction correctness"),
):
    """Evaluate source-bound biomedical cases separately across four scientific domains."""
    try:
        if benchmark_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("biomedical domain benchmark file exceeds 20 MiB")

        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_nonfinite(value):
            raise ValueError(f"non-finite JSON number is not allowed: {value}")

        document = json.loads(
            benchmark_file.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
        report = evaluate_biomedical_domain_benchmark(document, bins=bins)
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("biomedical-domain-benchmark-submit")
def biomedical_domain_benchmark_submit(
    session_id: str = typer.Argument(..., help="Tenant-owned research session to receive the benchmark"),
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    bins: int = typer.Option(10, min=2, max=100, help="Calibration bins used to validate benchmark computability"),
):
    """Validate and attach one independently adjudicated four-domain benchmark to a research session."""
    try:
        document, report, input_file_sha256 = load_domain_benchmark_artifact(benchmark_file, bins=bins)
        artifact = submit_domain_benchmark_artifact(document, report, session_id, _medai_admin_request)
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps({
        "submitted": True,
        "session_id": session_id,
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": report["benchmark_sha256"],
        "input_file_sha256": input_file_sha256,
        "artifact_id": artifact.get("id"),
        "artifact_sha256": artifact.get("sha256"),
        "total_cases": report["total_cases"],
        "complete_domain_coverage": report["complete_domain_coverage"],
        "missing_domains": report["missing_domains"],
        "selected_task_required": "benchmark.evaluate must be selected in the Research Director plan",
        "interpretation_limit": report["interpretation_limit"],
    }, ensure_ascii=False, indent=2))


@app.command("evidence-quality-benchmark-submit")
def evidence_quality_benchmark_submit(
    session_id: str = typer.Argument(..., help="Tenant-owned research session to receive the benchmark"),
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    bins: int = typer.Option(10, min=2, max=100, help="Calibration bins used to validate benchmark computability"),
):
    """Validate and attach a double-reviewed ordinal evidence-quality benchmark."""
    try:
        document, report, input_file_sha256 = load_evidence_quality_benchmark_artifact(benchmark_file, bins=bins)
        artifact = submit_evidence_quality_benchmark_artifact(document, report, session_id, _medai_admin_request)
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps({
        "submitted": True,
        "session_id": session_id,
        "benchmark_id": report["benchmark_id"],
        "benchmark_sha256": report["benchmark_sha256"],
        "input_file_sha256": input_file_sha256,
        "artifact_id": artifact.get("id"),
        "artifact_sha256": artifact.get("sha256"),
        "cases": report["cases"],
        "domains": sorted(report["overall_by_domain"]),
        "selected_task_required": "benchmark.evaluate must be selected in the Research Director plan",
        "interpretation_limit": report["interpretation_limit"],
    }, ensure_ascii=False, indent=2))


@app.command("capability-calibrate")
def capability_calibrate(
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", help="Write the holdout-checked calibration policy JSON"),
    minimum_precision: float = typer.Option(0.95, "--minimum-precision", min=0.51, max=1.0),
):
    """Fit a semantic reuse threshold from an independently adjudicated pair benchmark."""
    try:
        if benchmark_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("capability benchmark file exceeds 20 MiB")
        document = json.loads(benchmark_file.read_text(encoding="utf-8"))
        report = calibrate_capability_reuse_threshold(document, minimum_precision=minimum_precision)
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        digest = write_private_report_no_overwrite(output, rendered)
        typer.echo(json.dumps({"report": report, "report_path": str(output), "report_sha256": digest, "ready_for_operator_review": report["reuse_enabled"]}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("capability-compare-versions")
def capability_compare_versions(
    comparison_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
):
    """Compare two capability versions on the same paired benchmark without changing lifecycle state."""
    try:
        if comparison_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("capability comparison file exceeds 20 MiB")
        document = json.loads(comparison_file.read_text(encoding="utf-8"))
        report = compare_capability_versions(document)
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("capability-benchmark-versions")
def capability_benchmark_versions(
    baseline_source: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    candidate_source: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    benchmark_file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    image: str = typer.Option(..., "--image", envvar="MEDAI_PYTHON_SANDBOX_IMAGE", help="Locally cached immutable digest-pinned sandbox image"),
    repeats: int = typer.Option(5, min=3, max=15),
    timeout: int = typer.Option(60, min=1, max=60),
):
    """Run paired capability versions on identical cases in the pinned no-network sandbox."""
    try:
        if benchmark_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("capability benchmark file exceeds 20 MiB")
        benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
        sandbox = DockerPythonSandbox(image=image, timeout=timeout)
        report = run_paired_capability_benchmark(
            sandbox,
            baseline_source.read_text(encoding="utf-8"),
            candidate_source.read_text(encoding="utf-8"),
            benchmark,
            repeats=repeats,
        )
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command("variant")
def variant_records(query: str, limit: int = 10):
    """Search ClinVar by variant, accession, or gene term; classifications are research records only."""
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise typer.BadParameter("Set NCBI_EMAIL before calling ClinVar E-utilities")
    try:
        result = asyncio.run(ClinVarClient(api_key=os.environ.get("NCBI_API_KEY"), email=email).search(query, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("dbsnp")
def dbsnp_record(rsid: str):
    """Retrieve one bounded NCBI dbSNP RefSNP record; placements need assembly validation."""
    try:
        result = asyncio.run(DbSNPClient().lookup(rsid))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

@app.command("variant-frequency")
def variant_frequency(variant_id: str, assembly: str = typer.Option("GRCh38", "--assembly")):
    """Look up one exact GRCh38 variant ID in gnomAD v4; reference normalization is not performed here."""
    try:
        result = asyncio.run(GnomADClient().lookup(variant_id, assembly))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

@app.command("variant-reconcile")
def variant_reconcile(
    genomic_report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    clinvar_results: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    reference_fasta: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    genome_build: str = typer.Option(..., "--genome-build", help="Reference assembly: GRCh37 or GRCh38"),
    output: Path | None = typer.Option(None, "--output", help="Optional output path; genomic identifiers remain local"),
):
    """Reconcile local reference-validated alleles with exact ClinVar assembly locations."""
    try:
        local = json.loads(genomic_report.read_text(encoding="utf-8"))
        clinvar = json.loads(clinvar_results.read_text(encoding="utf-8"))
        if isinstance(clinvar, dict):
            clinvar = clinvar.get("records")
        result = reconcile_clinvar_variants(local, clinvar, reference_fasta, genome_build)
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if output is not None:
            output.write_text(encoded + "\n", encoding="utf-8")
        else:
            typer.echo(encoded)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command("preprint")
def preprint(server: str, doi: str):
    """Look up one exact bioRxiv or medRxiv DOI without retaining its abstract."""
    try:
        result = asyncio.run(PreprintClient().get_by_doi(server, doi))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("variant-annotate-vep")
def variant_annotate_vep(
    genomic_report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    genome_build: str | None = typer.Option(None, "--genome-build", help="Must match the GRCh37/GRCh38 build in the local analysis report"),
    output: Path | None = typer.Option(None, "--output", help="Optional output file for the Ensembl transcript annotations"),
):
    """Cross-check reference-validated local variants with Ensembl VEP consequences."""
    try:
        if genomic_report.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("genomic analysis report exceeds 20 MiB")
        report = json.loads(genomic_report.read_text(encoding="utf-8"))
        result = asyncio.run(EnsemblVEPClient().annotate_report(report, genome_build))
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if output:
            output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(rendered)
    except (OSError, ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command("variant-reconcile-sources")
def variant_reconcile_sources(
    genomic_report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    clinvar_reconciliation: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    vep_annotation: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", help="Optional output path; identifiers remain local"),
):
    """Join ClinVar and VEP receipts to exact local reference-normalized alleles."""
    try:
        paths = (genomic_report, clinvar_reconciliation, vep_annotation)
        if any(path.stat().st_size > 20 * 1024 * 1024 for path in paths):
            raise ValueError("source reconciliation input exceeds 20 MiB")
        report, clinvar, vep = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        result = reconcile_variant_sources(report, clinvar, vep)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if output is not None:
            output.write_text(rendered + "\n", encoding="utf-8")
        else:
            typer.echo(rendered)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("variant-reconcile-gnomad")
def variant_reconcile_gnomad(
    genomic_report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    frequency_receipts: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(None, "--output", help="Optional output path for the local allele join"),
):
    """Join gnomAD population frequency receipts to exact local normalized GRCh38 alleles."""
    try:
        if genomic_report.stat().st_size > 20 * 1024 * 1024 or frequency_receipts.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("genomic report or gnomAD receipt input exceeds its size limit")
        report = json.loads(genomic_report.read_text(encoding="utf-8"))
        payload = json.loads(frequency_receipts.read_text(encoding="utf-8"))
        receipts = payload.get("receipts") if isinstance(payload, dict) and isinstance(payload.get("receipts"), list) else payload if isinstance(payload, list) else [payload]
        result = reconcile_gnomad_frequency(report, receipts)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if output is not None:
            output.write_text(rendered + "\n", encoding="utf-8")
        else:
            typer.echo(rendered)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command("gene-panel")
def gene_panel(panel_id: int, version: str | None = typer.Option(None, "--version", help="Exact numeric PanelApp version; omit for current")):
    """Retrieve one versioned Genomics England PanelApp curated gene panel."""
    try:
        result = asyncio.run(PanelAppClient().get_panel(panel_id, version))
    except (ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

@app.command("variant-reconcile-dbsnp")
def variant_reconcile_dbsnp(
    genomic_report: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    dbsnp_receipt: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    reference_fasta: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    sequence_map: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    genome_build: str = typer.Option(..., "--genome-build", help="Reference assembly: GRCh37 or GRCh38"),
):
    """Join dbSNP RefSNP placements to exact local normalized alleles using an explicit RefSeq map."""
    try:
        paths = (genomic_report, dbsnp_receipt, sequence_map)
        if any(path.stat().st_size > 8_000_000 for path in paths):
            raise ValueError("dbSNP reconciliation input exceeds the 8 MB limit")
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("dbSNP reconciliation input contains duplicate JSON keys")
                result[key] = value
            return result

        def reject_nonfinite(value):
            raise ValueError(f"dbSNP reconciliation input contains non-standard JSON number {value}")

        report, receipt, mapping = [
            json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
            for path in paths
        ]
        result = reconcile_dbsnp_placements(report, receipt, reference_fasta, genome_build, mapping)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


@app.command("gencc-gene")
def gencc_gene(gene: str, limit: int = typer.Option(50, min=1, max=100)):
    """Read exact gene assertions from the locally cached GenCC export."""
    try:
        result = asyncio.run(GenCCClient().search_gene(gene, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


@app.command("gencc-disease")
def gencc_disease(disease_curie: str, limit: int = typer.Option(50, min=1, max=100)):
    """Read exact MONDO disease assertions from the locally cached GenCC export."""
    try:
        result = asyncio.run(GenCCClient().search_disease(disease_curie, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


@app.command("clingen-gene")
def clingen_gene(gene: str, limit: int = typer.Option(50, min=1, max=100)):
    """Read ClinGen expert-panel validity curations by exact HGNC ID or symbol."""
    try:
        result = asyncio.run(ClinGenGeneValidityClient().search_gene(gene, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


@app.command("clingen-disease")
def clingen_disease(disease_curie: str, limit: int = typer.Option(50, min=1, max=100)):
    """Read ClinGen expert-panel validity curations by exact MONDO identifier."""
    try:
        result = asyncio.run(ClinGenGeneValidityClient().search_disease(disease_curie, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


@app.command("orphadata-disease")
def orphadata_disease(orphacode: str):
    """Retrieve Orphadata gene-disease assertions by exact numeric ORPHAcode."""
    try:
        result = asyncio.run(OrphadataClient().disease_genes(orphacode))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

@app.command("preprints-search")
def preprints_search(
    server: str,
    query: str,
    start_date: str = typer.Option(..., "--start-date", help="Inclusive YYYY-MM-DD"),
    end_date: str = typer.Option(..., "--end-date", help="Inclusive YYYY-MM-DD"),
    limit: int = typer.Option(20, min=1, max=50),
    category: str | None = typer.Option(None, help="Provider category filter"),
    max_pages: int = typer.Option(2, min=1, max=3),
):
    """Lexically screen a bounded bioRxiv/medRxiv date/category slice."""
    try:
        result = asyncio.run(PreprintClient().search(server, query, start_date, end_date, limit, category, max_pages))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("ontology-search")
def ontology_search(query: str, ontology: str | None = typer.Option(None, "--ontology"), limit: int = typer.Option(20, min=1, max=50)):
    """Search biomedical ontology labels and synonyms through OLS4."""
    try:
        result = asyncio.run(OntologyClient().search(query, ontology, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("ontology-term")
def ontology_term(identifier: str, ontology: str | None = typer.Option(None, "--ontology")):
    """Resolve an ontology identifier only when OLS4 returns an exact identifier match."""
    try:
        result = asyncio.run(OntologyClient().resolve_identifier(identifier, ontology))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command()
def trials(query: str, limit: int = 20):
    """Search public ClinicalTrials.gov registry records."""
    for row in asyncio.run(ClinicalTrialsClient().search(query, limit)):
        typer.echo(__import__("json").dumps(row))

@app.command("compound")
def compound(identifier: str, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private replayable bundle for one exact PubChem match with computed RDKit descriptors")):
    """Look up PubChem chemical identity and optional local RDKit descriptors."""
    try:
        result = asyncio.run(PubChemClient().lookup(identifier))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    if bundle_dir is not None:
        matches = result.get("matches", [])
        if result.get("ambiguous") or len(matches) != 1:
            raise typer.BadParameter("compound bundle requires exactly one unambiguous PubChem CID")
        try:
            result = {"chemical_identity": result, "reproducibility_bundle": create_compound_bundle(bundle_dir, matches[0])}
        except (OSError, ValueError) as error:
            raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("drug")
def drug(identifier: str, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Capture exact ChEMBL records in a private offline-replayable bundle")):
    """Resolve compound identity and retrieve exact-identity ChEMBL pharmacology records."""
    try:
        identity = asyncio.run(PubChemClient().lookup(identifier))
        matches = identity.get("matches", [])
        if identity.get("ambiguous") or len(matches) != 1:
            result = {"chemical_identity": identity, "pharmacology": None, "reason": "Choose a single exact compound identity before joining pharmacology records."}
        else:
            inchikey = matches[0].get("inchikey")
            pharmacology = asyncio.run(ChEMBLClient().investigate(inchikey)) if inchikey else None
            bundle = None
            if bundle_dir is not None:
                if not inchikey or not pharmacology or not pharmacology.get("molecule_matches"):
                    raise ValueError("drug bundle requires one exact PubChem and ChEMBL identity match")
                bundle = create_chembl_bundle(bundle_dir, pharmacology)
            result = {"chemical_identity": identity, "pharmacology": pharmacology, "reproducibility_bundle": bundle, "reason": None if inchikey else "PubChem returned no standard InChIKey; pharmacology was not joined."}
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("drug-interactions")
def drug_interactions(drug_a: str, drug_b: str):
    """Compare public FDA label interaction sections for two generic drug names; not clinical advice."""
    try:
        result = asyncio.run(OpenFDALabelClient().investigate_pair(drug_a, drug_b))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("crossref")
def crossref_lookup(query: str, limit: int = 5, mailto: str | None = typer.Option(None, envvar="CROSSREF_MAILTO")):
    """Resolve a DOI or search bibliographic metadata and publisher-deposited references."""
    client = CrossrefClient(mailto=mailto)
    try:
        try:
            result = asyncio.run(client.get_work(query))
        except ValueError as error:
            if "valid DOI" not in str(error):
                raise
            result = asyncio.run(client.search_bibliographic(query, limit))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("trial")
def trial(nct_id: str, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private offline-replayable ClinicalTrials.gov snapshot bundle")):
    """Fetch one versioned registry snapshot by NCT identifier; not treatment advice."""
    try:
        result = asyncio.run(ClinicalTrialsClient().get_study(nct_id))
        if bundle_dir is not None:
            result["reproducibility_bundle"] = create_clinical_trial_bundle(bundle_dir, result)
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command()
def gene(query: str, limit: int = 10, species: str = "human"):
    """Search MyGene.info annotations; records retain provider IDs and retrieval time."""
    for row in asyncio.run(MyGeneClient().search(query, limit, species)):
        typer.echo(json.dumps(row))

@app.command("gene-disease")
def gene_disease(disease_id: str, limit: int = 50):
    """Compare source-linked gene–disease records by exact gene ID across applicable sources."""
    async def run():
        client = GeneDiseaseClient()
        source_names = []
        calls = []
        if disease_id.startswith("EFO_"):
            source_names.append("Open Targets Platform")
            calls.append(client.opentargets(disease_id, limit))
            source_names.append("NHGRI-EBI GWAS Catalog")
            calls.append(client.gwas_catalog(disease_id, limit))
        source_names.append("Monarch Initiative Knowledge Graph")
        calls.append(client.monarch_gene_disease(disease_id, limit))
        outcomes = await asyncio.gather(*calls, return_exceptions=True)
        sources = [item for item in outcomes if isinstance(item, dict)]
        if not sources:
            raise RuntimeError("all gene-disease sources failed")
        return {
            "sources": sources,
            "source_errors": [{"source": source_names[i], "error": type(item).__name__} for i, item in enumerate(outcomes) if isinstance(item, Exception)],
            "reconciliation": reconcile_gene_disease_sources(disease_id, sources),
        }
    try:
        typer.echo(json.dumps(asyncio.run(run()), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command("paper-fulltext")
def paper_fulltext(identifier: str):
    """Parse one PMC Open Access BioC full-text document; check its article-specific license."""
    try:
        result = asyncio.run(PMCBioCClient().fetch(identifier))
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error

@app.command()
def protein(query: str, limit: int = 10, organism_id: int = 9606):
    """Search UniProtKB protein names, genes, sequences, and function annotations."""
    for row in asyncio.run(UniProtClient().search(query, limit, organism_id)):
        typer.echo(json.dumps(row, ensure_ascii=False))

@app.command("structure")
def protein_structure(uniprot_accession: str):
    """Retrieve predicted protein structure metadata from AlphaFold DB."""
    try:
        records = asyncio.run(AlphaFoldClient().predictions(uniprot_accession))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps({"accession": uniprot_accession, "predictions": records}, ensure_ascii=False, indent=2))

@app.command("protein-structure-analyze")
def protein_structure_analyze(
    input_file: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, readable=True),
    label: str | None = typer.Option(None, "--label", help="Optional stable structure accession/model label"),
    structure_format: str = typer.Option("auto", "--format", help="auto, pdb, or mmcif"),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Capture a private offline-replayable structure analysis bundle"),
):
    """Summarize a local PDB or mmCIF coordinate file without retaining coordinates."""
    try:
        if structure_format not in {"auto", "pdb", "mmcif"}:
            raise ValueError("format must be auto, pdb, or mmcif")
        selected_format = structure_format
        if selected_format == "auto":
            suffix = input_file.suffix.casefold()
            if suffix in {".cif", ".mmcif"}:
                selected_format = "mmcif"
            elif suffix in {".pdb", ".ent"}:
                selected_format = "pdb"
            else:
                raise ValueError("cannot infer structure format from the filename; pass --format pdb or --format mmcif")
        maximum = MAX_MMCIF_BYTES if selected_format == "mmcif" else MAX_PDB_BYTES
        with input_file.open("rb") as structure_stream:
            payload = structure_stream.read(maximum + 1)
        if not 1 <= len(payload) <= maximum:
            raise ValueError(f"{selected_format} input must contain 1..{maximum} bytes")
        result = summarize_mmcif(payload, label=label) if selected_format == "mmcif" else summarize_pdb(payload, label=label)
        if bundle_dir is not None:
            result["reproducibility_bundle"] = create_structure_bundle(bundle_dir, result, input_file, selected_format)
    except (OSError, ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("protein-record")
def protein_record(uniprot_accession: str, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private, offline-replayable UniProtKB feature bundle")):
    """Retrieve a UniProtKB entry and optionally capture its replayable feature analysis."""
    try:
        result = asyncio.run(UniProtClient().get_entry(uniprot_accession))
        if bundle_dir is not None:
            result["reproducibility_bundle"] = create_protein_bundle(bundle_dir, result)
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("protein-mapping")
def protein_residue_mapping(uniprot_accession: str, max_structures: int = 10):
    """Map PDB residues to UniProt positions through PDBe SIFTS XML."""
    try:
        accession = UniProtClient.validate_accession(uniprot_accession)
        entry = asyncio.run(UniProtClient().get_entry(accession))
        pdb_ids = [item["pdb_id"] for item in entry["experimental_structure_cross_references"] if item.get("pdb_id")]
        mapping = asyncio.run(SIFTSClient().map_residues(accession, pdb_ids, max_structures))
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(mapping, ensure_ascii=False, indent=2))

@app.command("interactions")
def protein_interactions(uniprot_accession: str, limit: int = 100, organism_id: int | None = 9606, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Capture minimized IntAct source receipts for offline reconciliation replay")):
    """Search IntAct interaction evidence and optionally capture an offline replay bundle."""
    try:
        result = asyncio.run(IntActClient().interactions(uniprot_accession, limit, organism_id))
        bundle = create_intact_bundle(bundle_dir, result) if bundle_dir is not None else None
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps({**result, **({"reproducibility_bundle": bundle} if bundle else {})}, ensure_ascii=False, indent=2))

@app.command()
def pathways(genes: str, species: str = "Homo sapiens", page_size: int = 20, bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private offline-verifiable Reactome response bundle")):
    """Analyze a comma-separated gene list against Reactome pathways."""
    identifiers = [item for item in genes.replace("\n", ",").split(",")]
    result = asyncio.run(ReactomeClient().analyze_genes(identifiers, species, page_size))
    if bundle_dir is not None:
        result["reproducibility_bundle"] = create_pathway_bundle(bundle_dir, result)
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("genome-vcf")
def genome_vcf(
    input_file: Path,
    genome_build: str = typer.Option(..., "--genome-build", help="Reference assembly: GRCh37 or GRCh38"),
    max_variants: int = typer.Option(2_000_000, min=1, max=2_000_000),
    reference_fasta: Path | None = typer.Option(None, "--reference-fasta", help="Local uncompressed FASTA reference"),
    transcript_gff3: Path | None = typer.Option(None, "--transcript-gff3", help="Local GFF3 transcript annotation; requires --reference-fasta"),
    annotate_vep: bool = typer.Option(False, "--annotate-vep", help="Cross-check reference-validated alleles with Ensembl VEP and include its receipt in the output/bundle"),
    clinvar_results: Path | None = typer.Option(None, "--clinvar-results", help="Local JSON file containing sourced ClinVar records; requires --reference-fasta"),
    gnomad_frequency_receipts: Path | None = typer.Option(None, "--gnomad-frequency-receipts", help="Local gnomAD frequency receipt JSON; requires --reference-fasta"),
    dbsnp_receipt_file: Path | None = typer.Option(None, "--dbsnp-receipt", help="Local bounded dbSNP RefSNP receipt; requires --reference-fasta and --dbsnp-sequence-map"),
    dbsnp_sequence_map_file: Path | None = typer.Option(None, "--dbsnp-sequence-map", help="JSON map from RefSeq accession versions to exact FASTA contig names"),
    output: Path | None = typer.Option(None, "--output", help="Write the local JSON report to this path"),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private replayable bundle; copies genomic inputs into this new directory"),
    session_id: str | None = typer.Option(None, "--submit-session", help="Upload only an aggregate receipt; the VCF and variant records stay local"),
):
    """Validate local VCF alleles and optionally map them to local GFF3 transcripts."""
    try:
        result = analyze_vcf(input_file, genome_build, max_variants, reference_fasta, transcript_gff3)
        vep_result = None
        if annotate_vep:
            if reference_fasta is None:
                raise ValueError("--annotate-vep requires --reference-fasta so submitted variants are locally reference-validated")
            if session_id:
                raise ValueError("--annotate-vep output is retained locally; do not combine it with --submit-session")
            vep_result = asyncio.run(EnsemblVEPClient().annotate_report(result, genome_build))
        clinvar_records = None
        clinvar_receipt = None
        source_reconciliation = None
        gnomad_receipts = None
        gnomad_reconciliation = None
        dbsnp_receipt = None
        dbsnp_sequence_map = None
        dbsnp_reconciliation = None
        if clinvar_results is not None:
            if reference_fasta is None:
                raise ValueError("--clinvar-results requires --reference-fasta for exact allele normalization")
            if session_id:
                raise ValueError("--clinvar-results data are retained locally; do not combine them with --submit-session")
            clinvar_payload = read_strict_json(clinvar_results, 8_000_000, "ClinVar results")
            clinvar_records = clinvar_payload.get("records") if isinstance(clinvar_payload, dict) else clinvar_payload
            clinvar_receipt = reconcile_clinvar_variants(result, clinvar_records, reference_fasta, genome_build)
            if vep_result is not None:
                source_reconciliation = reconcile_variant_sources(result, clinvar_receipt, vep_result)
        if gnomad_frequency_receipts is not None:
            if reference_fasta is None:
                raise ValueError("--gnomad-frequency-receipts requires --reference-fasta for exact allele identity")
            if session_id:
                raise ValueError("--gnomad-frequency-receipts data are retained locally; do not combine them with --submit-session")
            gnomad_payload = read_strict_json(gnomad_frequency_receipts, 8_000_000, "gnomAD frequency receipts")
            gnomad_receipts = gnomad_payload.get("receipts") if isinstance(gnomad_payload, dict) and isinstance(gnomad_payload.get("receipts"), list) else gnomad_payload if isinstance(gnomad_payload, list) else [gnomad_payload]
            gnomad_reconciliation = reconcile_gnomad_frequency(result, gnomad_receipts)
        if (dbsnp_receipt_file is None) != (dbsnp_sequence_map_file is None):
            raise ValueError("--dbsnp-receipt and --dbsnp-sequence-map must be provided together")
        if dbsnp_receipt_file is not None:
            if reference_fasta is None:
                raise ValueError("--dbsnp-receipt requires --reference-fasta for exact SPDI normalization")
            if session_id:
                raise ValueError("--dbsnp-receipt data are retained locally; do not combine them with --submit-session")
            dbsnp_receipt = read_strict_json(dbsnp_receipt_file, 8_000_000, "dbSNP receipt")
            dbsnp_sequence_map = read_strict_json(dbsnp_sequence_map_file, 128_000, "dbSNP sequence map")
            dbsnp_reconciliation = reconcile_dbsnp_placements(result, dbsnp_receipt, reference_fasta, genome_build, dbsnp_sequence_map)
    except (OSError, ValueError, httpx.HTTPError) as error:
        raise typer.BadParameter(str(error)) from error
    if output is not None and bundle_dir is not None:
        raise typer.BadParameter("choose either --output or --bundle-dir")
    if session_id and bundle_dir is not None:
        raise typer.BadParameter("submit a local report with --output or stdout; private replay bundles are not uploaded")
    if bundle_dir is not None:
        try:
            bundle = create_genomics_bundle(bundle_dir, result, input_file, genome_build, max_variants, reference_fasta, transcript_gff3, vep_result, clinvar_records=clinvar_records, gnomad_frequency_receipts=gnomad_receipts, dbsnp_receipt=dbsnp_receipt, dbsnp_sequence_accession_to_contig=dbsnp_sequence_map)
        except (OSError, ValueError, httpx.HTTPError) as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(json.dumps(bundle, ensure_ascii=False, indent=2))
        return
    if vep_result is not None or clinvar_receipt is not None or gnomad_reconciliation is not None or dbsnp_reconciliation is not None:
        rendered_payload = {"genomic_report": result}
        if vep_result is not None:
            rendered_payload["external_vep_validation"] = vep_result
        if clinvar_receipt is not None:
            rendered_payload["external_clinvar_reconciliation"] = clinvar_receipt
        if source_reconciliation is not None:
            rendered_payload["cross_source_reconciliation"] = source_reconciliation
        if gnomad_receipts is not None:
            rendered_payload["external_gnomad_frequency_receipts"] = gnomad_receipts
            rendered_payload["external_gnomad_frequency_reconciliation"] = gnomad_reconciliation
        if dbsnp_reconciliation is not None:
            rendered_payload["external_dbsnp_receipt"] = dbsnp_receipt
            rendered_payload["external_dbsnp_reconciliation"] = dbsnp_reconciliation
    else:
        rendered_payload = result
    rendered = json.dumps(rendered_payload, ensure_ascii=False, indent=2)
    if output is None and not session_id:
        typer.echo(rendered)
        return
    if output is not None:
        protected_inputs = [input_file, *([reference_fasta] if reference_fasta else []), *([transcript_gff3] if transcript_gff3 else [])]
        if any(output.expanduser().resolve() == source.expanduser().resolve() for source in protected_inputs):
            raise typer.BadParameter("output path must not overwrite a VCF, FASTA, or GFF3 input")
        output.expanduser().write_text(rendered + "\n", encoding="utf-8")
    if session_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", session_id):
            raise typer.BadParameter("session ID must be a stable 1..128 character identifier")
        try:
            summary = summarize_local_vcf_report(result, expected_source_id=result["input_sha256"])
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        if len(json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 250_000:
            raise typer.BadParameter("session upload aggregate exceeds the 250000-byte payload limit")
        try:
            receipt = _medai_admin_request(
                f"/v1/research/{quote(session_id, safe='')}/genomic-datasets",
                {"report": summary},
            )
        except (ValueError, httpx.HTTPError, json.JSONDecodeError) as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(json.dumps({"submitted": True, "session_id": session_id, "artifact_id": receipt.get("id"), "artifact_sha256": receipt.get("sha256"), "source_vcf_sha256": result["input_sha256"], "report_path": str(output.expanduser()) if output else None, "original_vcf_uploaded": False, "raw_variant_records_uploaded": False}, ensure_ascii=False, indent=2))
    elif output is not None:
        typer.echo(f"Local VCF report written to {output.expanduser()}")

@app.command("bundle-verify")
def bundle_verify(
    bundle_dir: Path,
    replay: bool = typer.Option(False, "--replay", help="Recompute the report using bundled inputs, matching code/runtime versions, and no network"),
):
    """Verify a local scientific analysis bundle and optionally replay its analysis offline."""
    try:
        result = verify_analysis_bundle(bundle_dir, replay)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("bundle-diff")
def bundle_diff(baseline: Path, current: Path):
    """Compare two hash-verified research-session bundle JSON files."""
    try:
        before = json.loads(baseline.expanduser().read_text(encoding="utf-8"))
        after = json.loads(current.expanduser().read_text(encoding="utf-8"))
        result = compare_reproducibility_bundles(before, after)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command("research-diff")
def research_diff(session_id: str, baseline_artifact_id: str, current_artifact_id: str):
    """Compare two stored research bundles through the authenticated tenant API."""
    try:
        session_path = quote(session_id, safe="")
        baseline_record = _medai_admin_request(
            f"/v1/research/{session_path}/artifacts/{quote(baseline_artifact_id, safe='')}",
            method="GET",
        )
        current_record = _medai_admin_request(
            f"/v1/research/{session_path}/artifacts/{quote(current_artifact_id, safe='')}",
            method="GET",
        )
        baseline = parse_research_bundle_artifact(baseline_record, session_id)
        current = parse_research_bundle_artifact(current_record, session_id)
        result = compare_reproducibility_bundles(baseline, current)
    except (httpx.HTTPError, ValueError) as error:
        raise typer.BadParameter(f"could not compare stored research bundles: {error}") from error
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command("sequence-fasta")
def sequence_fasta(
    input_file: Path,
    molecule_type: str = typer.Option("nucleotide", "--molecule-type", help="nucleotide or protein"),
    max_records: int = typer.Option(2_000_000, min=1, max=2_000_000),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private replayable bundle with an exact copy of the FASTA input"),
):
    """Validate a FASTA file locally and report aggregate sequence statistics."""
    try:
        report = inspect_fasta(input_file, molecule_type, max_records)
        if bundle_dir is not None:
            created = create_sequence_bundle(bundle_dir, report, input_file, "fasta", max_records, molecule_type)
            typer.echo(json.dumps(created, ensure_ascii=False, indent=2))
            return
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command("sequence-fastq")
def sequence_fastq(
    input_file: Path,
    max_records: int = typer.Option(2_000_000, min=1, max=2_000_000),
    bundle_dir: Path | None = typer.Option(None, "--bundle-dir", help="Create a private replayable bundle with an exact copy of the FASTQ input"),
):
    """Validate FASTQ sequence/quality lengths locally using Phred+33."""
    try:
        report = inspect_fastq(input_file, max_records)
        if bundle_dir is not None:
            created = create_sequence_bundle(bundle_dir, report, input_file, "fastq", max_records)
            typer.echo(json.dumps(created, ensure_ascii=False, indent=2))
            return
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))

@app.command()
def execute(session_id: str, max_tasks: int = 32, paper_limit: int | None = typer.Option(None, "--paper-limit", min=1, max=20, help="Override the depth-based source-record ceiling (standard 10, deep 20)"), agent_id: str | None = typer.Option(None, "--agent-id", help="Restrict leases to one registered scientific specialist")):
    """Run validated research handlers through the configured authenticated MCP endpoint."""
    from medai_science.worker import execute_session
    try:
        agent_id = agent_id or os.getenv("MEDAI_SCIENTIFIC_AGENT_ID") or None
        result = asyncio.run(execute_session(session_id, max_tasks=max_tasks, paper_limit=paper_limit, agent_id=agent_id))
    except ModuleNotFoundError as error:
        if error.name == "mcp" or (error.name and error.name.startswith("mcp.")):
            raise typer.BadParameter("Install the MCP extra with: pip install -e '.[mcp]'") from error
        raise
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

@app.command()
def work(poll_seconds: int = 30, max_tasks: int = 32, paper_limit: int | None = typer.Option(None, "--paper-limit", min=1, max=20, help="Override the depth-based source-record ceiling (standard 10, deep 20)"), watch: bool = True, agent_id: str | None = typer.Option(None, "--agent-id", help="Run as one registered specialist actor with a task allowlist")):
    """Continuously discover and execute runnable sessions through authenticated MCP."""
    from medai_science.worker import run_worker
    try:
        agent_id = agent_id or os.getenv("MEDAI_SCIENTIFIC_AGENT_ID") or None
        asyncio.run(run_worker(poll_seconds=poll_seconds, max_tasks=max_tasks, paper_limit=paper_limit, watch=watch, agent_id=agent_id))
    except ModuleNotFoundError as error:
        if error.name == "mcp" or (error.name and error.name.startswith("mcp.")):
            raise typer.BadParameter("Install the MCP extra with: pip install -e '.[mcp]'") from error
        raise
