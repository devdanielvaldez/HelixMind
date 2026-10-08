"""Versioned, review-gated templates derived from persisted MedAI research DAGs."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


MAX_WORKFLOW_TASKS = 256
_TASK_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_plan(tasks: Any) -> list[dict[str, Any]]:
    if not isinstance(tasks, list) or not tasks or len(tasks) > MAX_WORKFLOW_TASKS:
        raise ValueError(f"research workflow requires 1..{MAX_WORKFLOW_TASKS} tasks")
    rows: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"workflow task {index} must be an object")
        task_id = task.get("id")
        domain = task.get("domain")
        description = task.get("description")
        dependencies = task.get("depends_on")
        if not isinstance(task_id, str) or not _TASK_ID.fullmatch(task_id) or task_id in rows:
            raise ValueError(f"workflow task {index} has an invalid or duplicate ID")
        if not isinstance(domain, str) or not _TASK_ID.fullmatch(domain):
            raise ValueError(f"workflow task {task_id} has an invalid domain")
        if not isinstance(description, str) or not description.strip() or len(description) > 1000:
            raise ValueError(f"workflow task {task_id} requires a bounded description")
        if not isinstance(dependencies, list) or len(dependencies) > MAX_WORKFLOW_TASKS or any(not isinstance(value, str) or not _TASK_ID.fullmatch(value) for value in dependencies):
            raise ValueError(f"workflow task {task_id} has invalid dependencies")
        rows[task_id] = {"id": task_id, "domain": domain, "description": description.strip(), "depends_on": sorted(set(dependencies))}
    for task_id, task in rows.items():
        if task_id in task["depends_on"] or any(dependency not in rows for dependency in task["depends_on"]):
            raise ValueError(f"workflow task {task_id} has a self-reference or missing dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("research plan contains a dependency cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in rows[task_id]["depends_on"]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in sorted(rows):
        visit(task_id)
    # Emit a deterministic dependency-first sequence for readers and future runtimes.
    ordered: list[str] = []
    emitted: set[str] = set()

    def emit(task_id: str) -> None:
        if task_id in emitted:
            return
        for dependency in rows[task_id]["depends_on"]:
            emit(dependency)
        emitted.add(task_id)
        ordered.append(task_id)

    for task_id in sorted(rows):
        emit(task_id)
    return [rows[task_id] for task_id in ordered]


def build_research_workflow_proposal(session: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Derive an immutable topology version and hash-only run provenance; never execute it."""
    if not isinstance(session, dict) or not isinstance(plan, dict):
        raise ValueError("workflow source session and task plan must be objects")
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise ValueError("workflow source session requires a stable identifier")
    if plan.get("session_id") not in (None, session_id):
        raise ValueError("research task plan belongs to another session")
    steps = _validate_plan(plan.get("tasks"))
    by_id = {row["id"]: row for row in plan["tasks"]}
    topology = {"schema_version": "1", "workflow_id": "biomedical.research.plan", "steps": steps, "execution_target": "medai-research-dag-v1"}
    topology_sha256 = _canonical_digest(topology)
    observed_tasks = []
    for task_id in sorted(by_id):
        row = by_id[task_id]
        status = row.get("status")
        attempts = row.get("attempts")
        if not isinstance(status, str) or status not in {"blocked", "ready", "running", "succeeded", "failed", "skipped", "needs_review"}:
            raise ValueError(f"workflow source task {task_id} has an unknown lifecycle state")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError(f"workflow source task {task_id} has invalid attempt metadata")
        observed_tasks.append({
            "task_id": task_id,
            "status": status,
            "attempts": attempts,
            "output_sha256": _canonical_digest(row["output"]) if row.get("output") is not None else None,
            "error_sha256": _canonical_digest(row["error"]) if row.get("error") is not None else None,
        })
    executed = sum(row["status"] == "succeeded" for row in observed_tasks)
    return {
        **topology,
        "version": "1.0.0+" + topology_sha256[:16],
        "topology_sha256": topology_sha256,
        "source_run": {
            "session_id": session_id,
            "session_status": session.get("status"),
            "task_outcomes": observed_tasks,
            "completed_task_count": executed,
            "task_count": len(observed_tasks),
            "outcome_coverage": executed / len(observed_tasks),
        },
        "lifecycle_status": "experimental_pending_review",
        "execution": {"enabled": False, "requires_authenticated_approval": True},
        "interpretation_limit": "This is a versioned topology template for MedAI's existing research DAG, not an AgentMesh Teach Workflow IR package or a scientific validation. Task outcome hashes document one source run; they do not establish workflow effectiveness or permit execution.",
    }
