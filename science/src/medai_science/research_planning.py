"""Model-assisted query and AgentMesh task-scope planning, without scientific claims."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from medai_science.llm import JSONModelProvider, ModelCompletion, _validate_schema
from medai_science.sources.safety import ensure_public_research_query
from medai_science.orchestrator import DEFAULT_PLAN


RESEARCH_TASK_IDS = tuple(step.id for step in DEFAULT_PLAN)
MANDATORY_RESEARCH_TASK_IDS = frozenset({
    "objective.normalize", "literature.discover", "literature.extract", "entities.resolve",
    "evidence.grade", "contradictions.search", "graph.build", "hypotheses.propose",
    "hypotheses.criticize", "report.reproduce",
})
OPTIONAL_DEPENDENCY_AGGREGATORS = frozenset({"graph.build", "report.reproduce"})


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("objective-plan artifact contains duplicate JSON keys")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"objective-plan artifact contains non-standard number {value}")


def paper_limit_for_depth(depth: str, requested_limit: int | None = None) -> int:
    """Choose a bounded source-record ceiling; an explicit operator ceiling always wins."""
    if depth not in {"standard", "deep"}:
        raise ValueError("depth must be standard or deep")
    if requested_limit is None:
        return 20 if depth == "deep" else 10
    if isinstance(requested_limit, bool) or not isinstance(requested_limit, int) or not 1 <= requested_limit <= 20:
        raise ValueError("paper limit must be 1..20")
    return requested_limit


OBJECTIVE_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["normalized_question", "pubmed_queries", "clinical_trials_query", "concepts", "scope_notes", "selected_task_ids", "task_scope_notes"],
    "properties": {
        "normalized_question": {"type": "string", "minLength": 1, "maxLength": 500},
        "pubmed_queries": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
        },
        "clinical_trials_query": {"type": "string", "minLength": 1, "maxLength": 256},
        "concepts": {
            "type": "array", "minItems": 1, "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        "scope_notes": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
        },
        "selected_task_ids": {
            "type": "array", "minItems": 1, "maxItems": len(RESEARCH_TASK_IDS),
            "items": {"type": "string", "enum": list(RESEARCH_TASK_IDS)},
        },
        "task_scope_notes": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 256},
        },
    },
}

_SYSTEM = (
    "You are a biomedical computational research planner. Produce bounded search and task-scope planning only. "
    "Do not answer the research question, state findings, infer causal relationships, invent stable identifiers, "
    "make clinical recommendations, or claim evidence. Use concepts present in or directly entailed by the objective. "
    "Return only the requested JSON object. The queries will be sent to public scholarly sources."
)


async def plan_research_objective(provider: JSONModelProvider, objective: str, depth: str = "standard") -> ModelCompletion:
    objective = ensure_public_research_query(objective.strip())
    if not objective or len(objective) > 2000:
        raise ValueError("research objective must contain 1..2000 characters")
    if depth not in {"standard", "deep"}:
        raise ValueError("depth must be standard or deep")
    prompt = (
        f"Research objective: {objective}\n"
        f"Requested depth: {depth}\n"
        "Registered task catalog (ID | domain | purpose): "
        + "; ".join(f"{step.id} | {step.domain} | {step.purpose}" for step in DEFAULT_PLAN)
        + "\n"
        "Select which registered scientific tasks are needed for this objective. Always include these mandatory tasks: "
        + ", ".join(sorted(MANDATORY_RESEARCH_TASK_IDS)) + ". Optional task IDs: "
        + ", ".join(task_id for task_id in RESEARCH_TASK_IDS if task_id not in MANDATORY_RESEARCH_TASK_IDS)
        + ". Include a task only when relevant; do not invent task IDs. graph.build and report.reproduce can consume partial upstream results.\n"
        "Prepare 1-3 distinct PubMed search queries, one ClinicalTrials.gov search query, "
        "a concise normalized question, grounded concepts, and scope notes. Query text must contain no patient identifiers."
    )
    completion = await provider.complete_json(
        system=_SYSTEM,
        prompt=prompt,
        schema_name="biomedical_research_objective_plan_v2",
        schema=OBJECTIVE_PLAN_SCHEMA,
    )
    plan = completion.value
    validate_task_scope(plan)
    for field in ("normalized_question", "clinical_trials_query"):
        plan[field] = ensure_public_research_query(plan[field].strip())
    for field in ("pubmed_queries", "concepts", "scope_notes", "task_scope_notes"):
        plan[field] = [ensure_public_research_query(value.strip()) for value in plan[field]]
    return completion


def validate_task_scope(plan: Any) -> list[str]:
    """Validate a model-selected task set against the registered AgentMesh DAG."""
    if not isinstance(plan, dict):
        raise ValueError("research task scope must be an object")
    selected = plan.get("selected_task_ids")
    if not isinstance(selected, list) or not selected or any(not isinstance(item, str) for item in selected):
        raise ValueError("research task scope must select registered task IDs")
    if len(selected) != len(set(selected)) or not set(selected) <= set(RESEARCH_TASK_IDS):
        raise ValueError("research task scope contains duplicate or unregistered task IDs")
    selected_set = set(selected)
    if not MANDATORY_RESEARCH_TASK_IDS <= selected_set:
        raise ValueError("research task scope omits mandatory synthesis or provenance tasks")
    steps = {step.id: step for step in DEFAULT_PLAN}
    for task_id in selected_set - OPTIONAL_DEPENDENCY_AGGREGATORS:
        if not set(steps[task_id].depends_on) <= selected_set:
            raise ValueError(f"research task scope omits a prerequisite for {task_id}")
    return [task_id for task_id in RESEARCH_TASK_IDS if task_id in selected_set]


def objective_plan_record_from_artifacts(artifacts: Any, *, objective: str, depth: str, session_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load a verified planning payload and its receipt for the exact research run."""
    if not isinstance(artifacts, list):
        return None
    if not isinstance(objective, str) or not objective.strip() or not isinstance(depth, str) or depth not in {"standard", "deep"}:
        raise ValueError("objective plan binding requires a research objective and supported depth")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("objective plan binding requires the current research session")
    expected_objective_sha256 = hashlib.sha256(objective.strip().encode()).hexdigest()
    for artifact in reversed(artifacts):
        if (
            not isinstance(artifact, dict)
            or artifact.get("kind") != "research_objective_plan"
            or artifact.get("source") != "MedAI configured model planner"
            or artifact.get("session_id") != session_id
            or artifact.get("task_id") != "objective.normalize"
        ):
            continue
        payload = artifact.get("payload")
        try:
            if not isinstance(payload, str) or len(payload.encode("utf-8")) > 32_768:
                continue
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or not hashlib.sha256(payload.encode("utf-8")).hexdigest() == digest:
                continue
            payload = json.loads(payload, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
            if not isinstance(payload, dict):
                continue
            if (
                payload.get("schema_version") != "2"
                or payload.get("objective_sha256") != expected_objective_sha256
                or payload.get("depth") != depth
                or artifact.get("source_id") != expected_objective_sha256
            ):
                continue
            plan = payload.get("planning")
            _validate_schema(plan, OBJECTIVE_PLAN_SCHEMA)
            plan["selected_task_ids"] = validate_task_scope(plan)
            for field in ("normalized_question", "clinical_trials_query"):
                ensure_public_research_query(plan[field])
            for field in ("pubmed_queries", "concepts", "scope_notes", "task_scope_notes"):
                for value in plan[field]:
                    ensure_public_research_query(value)
            return plan, artifact
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return None


def objective_plan_from_artifacts(artifacts: Any, *, objective: str, depth: str, session_id: str) -> dict[str, Any] | None:
    """Load the safe planning data only when its objective and depth match this run."""
    record = objective_plan_record_from_artifacts(artifacts, objective=objective, depth=depth, session_id=session_id)
    return record[0] if record else None
