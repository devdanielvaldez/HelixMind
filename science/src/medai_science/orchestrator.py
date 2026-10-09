"""Deterministic research planning primitives. A plan is not a claim that work ran."""
from dataclasses import dataclass, replace
from enum import Enum
from uuid import uuid4

class StepState(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    SKIPPED = "skipped"

@dataclass(frozen=True)
class ResearchStep:
    id: str
    domain: str
    purpose: str
    depends_on: tuple[str, ...] = ()
    state: StepState = StepState.READY

DEFAULT_PLAN = (
    ResearchStep("objective.normalize", "planning", "Normalize scope, entity names, and requested depth"),
    ResearchStep("literature.discover", "literature", "Search approved scholarly sources", ("objective.normalize",)),
    ResearchStep("literature.extract", "literature", "Extract structured methods, results, limitations, and citations", ("literature.discover",)),
    ResearchStep("entities.resolve", "knowledge_graph", "Resolve disease, gene, variant, protein, pathway, drug, and trial entities", ("literature.extract",)),
    ResearchStep("genes.curate", "gene_disease", "Retrieve independent curated gene-disease validity assertions for exactly resolved human genes", ("entities.resolve",)),
    ResearchStep("evidence.grade", "evidence", "Summarize recorded evidence annotations and PubMed publication metadata without inferring evidence quality", ("literature.extract",)),
    ResearchStep("genomics.analyze", "genomics", "Analyze variants and gene associations when input data is authorized", ("entities.resolve",)),
    ResearchStep("proteins.analyze", "proteomics", "Map protein functions, structures, and interactions", ("entities.resolve",)),
    ResearchStep("pathways.map", "pathways", "Connect entities to pathway evidence", ("entities.resolve",)),
    ResearchStep("trials.discover", "clinical_trials", "Retrieve registry records without making treatment recommendations", ("objective.normalize",)),
    ResearchStep("compounds.analyze", "chemistry", "Resolve source-identified compounds to exact chemical identities", ("literature.extract",)),
    ResearchStep("drugs.investigate", "pharmacology", "Connect exact compound identities and targets to sourced pharmacology evidence", ("compounds.analyze", "entities.resolve", "trials.discover")),
    ResearchStep("contradictions.search", "evidence", "Search for contrary findings and methodological differences", ("evidence.grade", "entities.resolve")),
    ResearchStep("graph.build", "knowledge_graph", "Persist only source-linked nodes and edges", ("entities.resolve", "evidence.grade", "genomics.analyze", "proteins.analyze", "pathways.map", "genes.curate", "compounds.analyze", "drugs.investigate")),
    ResearchStep("hypotheses.propose", "hypothesis", "Propose candidate hypotheses from traceable graph paths", ("graph.build", "contradictions.search")),
    ResearchStep("hypotheses.criticize", "scientific_criticism", "Attempt to refute candidate hypotheses", ("hypotheses.propose",)),
    ResearchStep("skills.learn", "self_extension", "Evaluate completed research histories and draft a disabled reusable Skill when a common procedure is supported", ("hypotheses.criticize",)),
    ResearchStep("report.reproduce", "reproducibility", "Emit report with source, tool, version, and session provenance", ("hypotheses.criticize", "skills.learn", "trials.discover")),
    ResearchStep("benchmark.evaluate", "scientific_evaluation", "Evaluate independently adjudicated domain benchmark cases supplied to this research session", ("objective.normalize",)),
)

@dataclass
class ResearchPlan:
    id: str
    objective: str
    depth: str
    budget: int
    steps: tuple[ResearchStep, ...]

def plan_research(objective: str, depth: str = "standard", budget: int = 100) -> ResearchPlan:
    objective = objective.strip()
    if not objective:
        raise ValueError("research objective cannot be empty")
    if depth not in {"standard", "deep"}:
        raise ValueError("depth must be standard or deep")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 100_000:
        raise ValueError("budget must be in the range 1..100000")
    steps = tuple(replace(step, state=StepState.READY if not step.depends_on else StepState.BLOCKED) for step in DEFAULT_PLAN)
    return ResearchPlan(id=f"PLAN-{uuid4().hex[:12]}", objective=objective, depth=depth, budget=budget, steps=steps)
