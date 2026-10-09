"""Deterministic Research Director and least-authority specialist routing.

The specialists are policy-scoped roles inside the AgentMesh worker. They do
not claim to be independently running LLM agents; this registry gives each DAG
stage an auditable owner, purpose, and allowed MCP tool surface.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from medai_science.orchestrator import DEFAULT_PLAN


@dataclass(frozen=True)
class ScientificAgent:
    id: str
    name: str
    specialty: str
    mandate: str
    allowed_tool_prefixes: tuple[str, ...]


AGENTS: dict[str, ScientificAgent] = {
    "research_director": ScientificAgent("research_director", "Research Director", "planning", "Coordinate the persisted research DAG, select objective-relevant registered stages, and delegate execution to their declared specialists.", ("medai_research_plan_objective", "medai_research_apply_task_scope")),
    "literature_librarian": ScientificAgent("literature_librarian", "Literature Librarian", "literature", "Retrieve bounded scholarly records and citation metadata, and resolve DOI-less references to unverified candidates; do not assert scientific support or create citation edges from search matches.", ("medai_papers_search_pubmed", "medai_papers_resolve_crossref")),
    "paper_curator": ScientificAgent("paper_curator", "Paper Understanding Agent", "literature", "Extract source-labeled structure and source-annotated entities; do not infer claims from unclassified text.", ("medai_papers_understand_pmc",)),
    "entity_curator": ScientificAgent("entity_curator", "Entity Resolution Agent", "knowledge_graph", "Resolve only stable source identifiers and register provenance-bearing entities.", ("medai_graph_add_entity", "medai_genes_search_mygene")),
    "gene_disease_curator": ScientificAgent("gene_disease_curator", "Gene-Disease Evidence Agent", "gene_disease", "Resolve only literal preferred disease labels through OLS, retrieve independent gene-disease assertions without merging source scales, and create source-record nodes linked to retained artifacts. Retrieve direct OxO cross-reference candidates for human review without using them as identity equivalences.", ("medai_gene_disease_", "medai_ontology_search", "medai_ontology_cross_mapping_review", "medai_graph_add_entity", "medai_graph_link_evidence")),
    "evidence_synthesizer": ScientificAgent("evidence_synthesizer", "Evidence Agent", "evidence", "Summarize existing source annotations descriptively without regrading study quality.", ("medai_evidence_list",)),
    "genomics_analyst": ScientificAgent("genomics_analyst", "Genomics Agent", "genomics", "Assess whether authorized, validated genomic inputs can be analyzed; do not infer pathogenicity.", ()),
    "protein_analyst": ScientificAgent("protein_analyst", "Protein Agent", "proteomics", "Retrieve accession-bound protein records, structures, residue mappings, and observed interactions; register exact source-identified proteins in the session graph.", ("medai_proteins_", "medai_graph_add_entity")),
    "pathway_analyst": ScientificAgent("pathway_analyst", "Pathway Agent", "pathways", "Run bounded Reactome over-representation analysis and add result- and artifact-linked analysis nodes without inferring pathway causality.", ("medai_pathways_", "medai_graph_add_entity", "medai_graph_link_evidence")),
    "trial_librarian": ScientificAgent("trial_librarian", "Clinical Trials Agent", "clinical_trials", "Retrieve public registry records without assessing personal suitability or recommending treatment.", ("medai_trials_",)),
    "chemistry_analyst": ScientificAgent("chemistry_analyst", "Chemistry Agent", "chemistry", "Resolve only source-identified PubChem CIDs to unambiguous chemical identities; never infer identity from free-text mentions.", ("medai_compounds_lookup_pubchem",)),
    "pharmacology_analyst": ScientificAgent("pharmacology_analyst", "Pharmacology Agent", "pharmacology", "Retrieve source-linked pharmacology records for exact compound identities and record only artifact-backed provider target assertions.", ("medai_drug_", "medai_graph_add_entity", "medai_graph_link_evidence")),
    "scientific_critic": ScientificAgent("scientific_critic", "Scientific Critic", "scientific_criticism", "Find review leads and alternative explanations; retrieved or model-screened records are not refutations.", ("medai_papers_search_pubmed", "medai_papers_align_pmc_claims", "medai_evidence_list", "medai_hypotheses_criticize")),
    "knowledge_graph_curator": ScientificAgent("knowledge_graph_curator", "Knowledge Graph Agent", "knowledge_graph", "Snapshot only registered stable entities and evidence-linked relationships.", ("medai_graph_snapshot",)),
    "hypothesis_generator": ScientificAgent("hypothesis_generator", "Hypothesis Agent", "hypothesis", "Generate traceable candidate hypotheses, search bounded source literature, and preserve uncertainty; make no discovery claims.", ("medai_hypotheses_propose", "medai_papers_search_pubmed")),
    "capability_learner": ScientificAgent("capability_learner", "Capability Learning Agent", "self_extension", "Draft disabled Python/R capability proposals, validate signed source only in the bounded no-network sandbox, and compare paired sandbox benchmarks; never invoke production capabilities, activate a Skill, or promote a capability.", ("medai_skills_", "medai_capabilities_generate_python_proposal", "medai_capabilities_generate_r_proposal", "medai_capabilities_validate_python_proposal", "medai_capabilities_validate_r_proposal", "medai_capabilities_benchmark_")),
    "reproducibility_agent": ScientificAgent("reproducibility_agent", "Reproducibility Agent", "reproducibility", "Package source, task, graph, evidence, and citation receipts with interpretation limits.", ("medai_graph_snapshot", "medai_evidence_list", "medai_papers_citation_network")),
    "benchmark_analyst": ScientificAgent("benchmark_analyst", "Scientific Evaluation Agent", "scientific_evaluation", "Evaluate only source-digest-bound, independently adjudicated domain or double-reviewed evidence-quality benchmark inputs; report domains separately and never claim performance from synthetic fixtures.", ("medai_biomedical_domain_benchmark_evaluate", "medai_evidence_quality_benchmark_evaluate")),
}

# These tools form the worker's bounded execution substrate and are available
# to every specialist. Domain/source tools remain restricted per agent above.
COMMON_TOOLS = frozenset({
    "medai_research_get_control",
    "medai_research_renew_task_lease",
    "medai_research_save_checkpoint",
    "medai_research_update_task",
    "medai_research_store_artifact",
    "medai_research_list_artifacts",
    "medai_research_get_plan",
    "medai_research_get_budget",
})


AGENT_BY_TASK: dict[str, str] = {
    "objective.normalize": "research_director",
    "literature.discover": "literature_librarian",
    "literature.extract": "paper_curator",
    "entities.resolve": "entity_curator",
    "genes.curate": "gene_disease_curator",
    "evidence.grade": "evidence_synthesizer",
    "genomics.analyze": "genomics_analyst",
    "proteins.analyze": "protein_analyst",
    "pathways.map": "pathway_analyst",
    "trials.discover": "trial_librarian",
    "compounds.analyze": "chemistry_analyst",
    "drugs.investigate": "pharmacology_analyst",
    "contradictions.search": "scientific_critic",
    "graph.build": "knowledge_graph_curator",
    "hypotheses.propose": "hypothesis_generator",
    "hypotheses.criticize": "scientific_critic",
    "skills.learn": "capability_learner",
    "report.reproduce": "reproducibility_agent",
    "benchmark.evaluate": "benchmark_analyst",
}


def agent_may_call(agent_id: str, tool_name: str) -> bool:
    """Apply the specialist's least-authority MCP tool policy."""
    agent = AGENTS.get(agent_id)
    if agent is None:
        return False
    return tool_name in COMMON_TOOLS or any(tool_name.startswith(prefix) for prefix in agent.allowed_tool_prefixes)


class ResearchDirector:
    """Validate and delegate the persisted research DAG without inventing steps."""

    def coordinate(
        self,
        *,
        session_id: str,
        objective: str,
        depth: str,
        budget: int,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not session_id or not isinstance(objective, str) or not objective.strip():
            raise ValueError("research director requires a session and objective")
        if depth not in {"standard", "deep"} or isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 100_000:
            raise ValueError("research director received invalid depth or budget")
        expected = {step.id: step for step in DEFAULT_PLAN}
        if not isinstance(tasks, list) or len(tasks) != len(expected):
            raise ValueError("persisted research plan does not match the registered specialist roster")
        by_id: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_id = task.get("id") if isinstance(task, dict) else None
            if not isinstance(task_id, str) or task_id not in expected or task_id in by_id:
                raise ValueError("persisted research plan contains an unknown or duplicate task")
            step = expected[task_id]
            if task.get("domain") != step.domain:
                raise ValueError(f"persisted task domain differs from the research plan: {task_id}")
            dependencies = task.get("depends_on")
            if not isinstance(dependencies, list) or set(dependencies) != set(step.depends_on) or len(dependencies) != len(step.depends_on):
                raise ValueError(f"persisted task dependencies differ from the research plan: {task_id}")
            by_id[task_id] = task
        if set(by_id) != set(expected) or set(AGENT_BY_TASK) != set(expected):
            raise ValueError("research plan and specialist roster coverage differ")

        # A topological traversal rejects cycles even if a future persisted plan
        # changes dependency declarations and the comparison policy is extended.
        remaining = {task_id: set(step.depends_on) for task_id, step in expected.items()}
        ordered: list[str] = []
        while remaining:
            ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("research plan dependency graph contains a cycle")
            ordered.extend(ready)
            for task_id in ready:
                remaining.pop(task_id)
            for deps in remaining.values():
                deps.difference_update(ready)

        assignments = []
        for task_id in ordered:
            step = expected[task_id]
            agent = AGENTS[AGENT_BY_TASK[task_id]]
            assignments.append({
                "task_id": task_id,
                "domain": step.domain,
                "depends_on": list(step.depends_on),
                "agent_id": agent.id,
                "agent_name": agent.name,
                "specialty": agent.specialty,
                "mandate": agent.mandate,
                "allowed_mcp_tool_prefixes": list(agent.allowed_tool_prefixes),
            })
        payload = {
            "schema": "medai.scientific-agent-coordination.v1",
            "session_id": session_id,
            "objective_sha256": hashlib.sha256(objective.strip().encode("utf-8")).hexdigest(),
            "depth": depth,
            "budget": budget,
            "director": {"agent_id": AGENTS["research_director"].id, "policy": "validate persisted DAG; delegate only to registered specialists; no implicit task execution"},
            "execution_plane": "AgentMesh MCP gateway",
            "task_assignments": assignments,
            "interpretation_limit": "Assignments describe deterministic worker roles and tool authority; they do not claim independent agent reasoning or scientific validation.",
        }
        payload["plan_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return payload
