"""Deterministic, review-gated Skills learned from completed research histories."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from medai_science.orchestrator import DEFAULT_PLAN
from medai_science.provenance_signing import sign_capability_proposal, verify_capability_proposal_signature


_MAX_HISTORY = 10_000
_CORE_STEPS = (
    "objective.normalize", "literature.discover", "literature.extract", "entities.resolve",
    "evidence.grade", "hypotheses.propose", "hypotheses.criticize", "report.reproduce",
)
_TASK_STATES = {"blocked", "ready", "running", "succeeded", "failed", "needs_review", "skipped"}
_OPTIONAL_DEPENDENCY_AGGREGATORS = {"graph.build", "report.reproduce"}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def tenant_skill_registry_dir(registry_root: str | Path, tenant_id: str) -> Path:
    """Return a stable, non-identifying tenant partition for persisted learned Skills."""
    if not isinstance(tenant_id, str) or not tenant_id or len(tenant_id) > 128:
        raise ValueError("authenticated tenant identity is required for the Skill registry")
    root = Path(registry_root).expanduser().resolve()
    partition = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return root / "tenants" / partition


def verify_research_bundle(bundle: Any, session_id: str) -> dict[str, Any]:
    if not isinstance(bundle, dict) or not isinstance(bundle.get("session"), dict) or bundle["session"].get("id") != session_id:
        raise ValueError("research bundle does not belong to its declared session")
    expected = bundle.get("bundle_sha256")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or _canonical_hash(unsigned) != expected:
        raise ValueError("research bundle integrity verification failed")
    if not isinstance(bundle.get("tasks"), list) or not isinstance(bundle.get("artifacts"), list):
        raise ValueError("research bundle task and artifact receipts must be arrays")
    return bundle


def learn_research_skill_batch(
    histories: list[dict[str, Any]],
    index: dict[str, Any] | None = None,
    *,
    minimum_sessions: int = 3,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Select one unconsumed completed-session batch and derive its common task routine."""
    if not isinstance(histories, list) or len(histories) > _MAX_HISTORY:
        raise ValueError("research history must be a bounded list")
    if not isinstance(minimum_sessions, int) or isinstance(minimum_sessions, bool) or not 3 <= minimum_sessions <= 20:
        raise ValueError("skill learning requires a minimum of 3..20 sessions")
    registry = dict(index or {})
    consumed = set(registry.get("consumed_session_ids", []))
    valid = []
    for row in histories:
        if not isinstance(row, dict) or row.get("session_status") != "completed":
            continue
        session_id = row.get("session_id")
        if not isinstance(session_id, str) or not re.fullmatch(r"RES-[A-Za-z0-9-]{1,120}", session_id):
            continue
        bundle = verify_research_bundle(row.get("bundle"), session_id)
        plan = row.get("plan")
        tasks = plan.get("tasks") if isinstance(plan, dict) else None
        if not isinstance(tasks, list) or plan.get("session_id") not in (None, session_id):
            raise ValueError("completed session is missing its same-session live task plan")
        bundle_tasks = bundle["tasks"]
        live_by_id: dict[str, dict[str, Any]] = {}
        bundle_by_id: dict[str, dict[str, Any]] = {}
        for rows, destination, label in ((tasks, live_by_id, "live task plan"), (bundle_tasks, bundle_by_id, "bundle task receipts")):
            for task in rows:
                if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
                    raise ValueError(f"completed session has malformed {label}")
                task_id = task["id"]
                if task_id in destination:
                    raise ValueError(f"completed session has duplicate {label} identity")
                status = task.get("status")
                if not isinstance(status, str) or status not in _TASK_STATES:
                    raise ValueError(f"completed session has invalid {label} status")
                destination[task_id] = task
        for task_id, live_task in live_by_id.items():
            receipt = bundle_by_id.get(task_id)
            if receipt is None:
                continue  # Older bundles can predate a registered optional DAG stage.
            observed = receipt["status"]
            if task_id == "report.reproduce" and observed == "running" and live_task["status"] == "succeeded":
                continue  # The report snapshot is necessarily taken before this task is finalized.
            if observed != live_task["status"]:
                raise ValueError("completed session live task outcomes differ from the hash-verified bundle")
        for task_id in _CORE_STEPS:
            if task_id not in live_by_id or task_id not in bundle_by_id:
                raise ValueError("completed session is missing a core task outcome from its bundle")
        if live_by_id.get("report.reproduce", {}).get("status") != "succeeded":
            continue
        valid.append({"session_id": session_id, "bundle_sha256": bundle["bundle_sha256"], "tasks": live_by_id, "created_at": row.get("created_at", 0)})
    valid.sort(key=lambda item: (int(item["created_at"] or 0), item["session_id"]))
    candidates = [item for item in valid if item["session_id"] not in consumed]
    if len(candidates) < minimum_sessions:
        return None, registry, {"status": "insufficient_history", "eligible_sessions": len(candidates), "minimum_sessions": minimum_sessions}
    batch = candidates[:minimum_sessions]
    registry["schema_version"] = "1"
    registry["consumed_session_ids"] = sorted(consumed | {item["session_id"] for item in batch})
    core_missing = [step for step in _CORE_STEPS if any(item["tasks"].get(step, {}).get("status") != "succeeded" for item in batch)]
    if core_missing:
        registry.setdefault("evaluated_batches", []).append({"session_ids": [item["session_id"] for item in batch], "result": "no_common_core", "missing_steps": core_missing})
        return None, registry, {"status": "no_common_core_procedure", "eligible_sessions": len(batch), "missing_core_steps": core_missing}
    steps = []
    for step in DEFAULT_PLAN:
        if step.id == "skills.learn":
            continue
        outcomes = [item["tasks"].get(step.id) for item in batch]
        attempted = [task for task in outcomes if isinstance(task, dict) and task.get("status") not in {"blocked", "ready", "running", None}]
        passed = sum(task.get("status") == "succeeded" for task in attempted)
        if len(attempted) < minimum_sessions or passed < minimum_sessions:
            continue
        steps.append({
            "task_id": step.id,
            "domain": step.domain,
            "purpose": step.purpose,
            "depends_on": list(step.depends_on),
            "successful_sessions": passed,
            "attempted_sessions": len(attempted),
            "observed_success_fraction": passed / len(attempted),
        })
    # A sequence learned from older or partial sessions must still be a valid
    # executable subset of today's DAG. Aggregators may consume partial inputs;
    # every other stage requires its registered prerequisites in the procedure.
    step_by_id = {step["task_id"]: step for step in steps}
    changed = True
    while changed:
        changed = False
        for step in tuple(steps):
            task = next(registered for registered in DEFAULT_PLAN if registered.id == step["task_id"])
            if task.id in _OPTIONAL_DEPENDENCY_AGGREGATORS:
                continue
            if any(dependency not in step_by_id for dependency in task.depends_on):
                step_by_id.pop(task.id)
                steps.remove(step)
                changed = True
    step_ids = {step["task_id"] for step in steps}
    if not set(_CORE_STEPS).issubset(step_ids):
        registry.setdefault("evaluated_batches", []).append({"session_ids": [item["session_id"] for item in batch], "result": "no_common_core_procedure"})
        return None, registry, {"status": "no_common_core_procedure", "eligible_sessions": len(batch)}
    sources = [{"session_id": item["session_id"], "bundle_sha256": item["bundle_sha256"]} for item in batch]
    sequence_sha256 = _canonical_hash([step["task_id"] for step in steps])
    source_sha256 = _canonical_hash(sources)
    version_digest = _canonical_hash({"sequence": sequence_sha256, "sources": sources})[:16]
    version = f"0.1.0+{version_digest}"
    skill_markdown = _render_skill(steps, len(batch), sources, version)
    manifest = {
        "schema_version": "1",
        "id": "biomedical.research_sequence",
        "version": version,
        "kind": "skill",
        "description": "Reusable MedAI computational research procedure derived from repeated completed sessions.",
        "status": "pending_review",
        "source_sessions": sources,
        "source_sha256": source_sha256,
        "procedure_sha256": sequence_sha256,
        "procedure": steps,
        "evaluation": {
            "completed_sessions": len(batch),
            "required_minimum_sessions": minimum_sessions,
            "interpretation": "Observed task completion frequency in a small selected batch; not scientific accuracy, causal validity, or calibrated reliability.",
        },
        "execution": {"enabled": False, "network": False, "filesystem": "read_only"},
        "review_required": True,
        "skill_markdown": skill_markdown,
    }
    signed = sign_capability_proposal(manifest)
    registry.setdefault("evaluated_batches", []).append({"session_ids": [item["session_id"] for item in batch], "result": "candidate_created", "skill_id": signed["id"], "version": signed["version"], "procedure_sha256": sequence_sha256})
    return signed, registry, {"status": "candidate_created", "eligible_sessions": len(batch), "skill_id": signed["id"], "version": signed["version"], "procedure_sha256": sequence_sha256}


def _render_skill(steps: list[dict[str, Any]], sample_count: int, sources: list[dict[str, str]], version: str) -> str:
    lines = [
        "---",
        'name: "biomedical.research_sequence"',
        'description: "Review-gated computational research workflow derived from completed MedAI sessions."',
        f'version: "{version}"',
        'status: "pending_review"',
        "---",
        "",
        "# Biomedical Research Sequence",
        "",
        f"Candidate procedure derived from {sample_count} completed, hash-verified research sessions.",
        "",
        "## Procedure",
        "",
    ]
    lines.extend(f"{index}. **{step['task_id']}** — {step['purpose']} (observed {step['successful_sessions']}/{step['attempted_sessions']} sessions)" for index, step in enumerate(steps, 1))
    lines.extend([
        "",
        "## Operational constraints",
        "",
        "- Execution is disabled until independent review and registry approval.",
        "- Preserve source artifacts and provenance for each run.",
        "- Treat retrieved claims and hypotheses as research evidence requiring scientific review.",
        "- This procedure does not make clinical recommendations.",
        "- The observed task completion fraction is not a scientific accuracy or reliability estimate.",
        "",
        "## Source sessions",
        "",
    ])
    lines.extend(f"- `{row['session_id']}` — bundle SHA-256 `{row['bundle_sha256']}`" for row in sources)
    lines.append("")
    return "\n".join(lines)


def persist_skill_candidate(skill: dict[str, Any], registry_dir: str | Path, signing_key: bytes | None = None) -> dict[str, Any]:
    """Atomically write an immutable signed Skill package into the local private registry."""
    if not isinstance(skill, dict) or skill.get("status") != "pending_review" or skill.get("execution", {}).get("enabled") is not False:
        raise ValueError("only disabled pending-review Skill candidates can be persisted")
    verify_capability_proposal_signature(skill, signing_key)
    skill_id, version = skill.get("id"), skill.get("version")
    if not isinstance(skill_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", skill_id):
        raise ValueError("Skill id is not a valid namespaced identifier")
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\+[a-f0-9]{16}", version):
        raise ValueError("Skill version is not a deterministic content version")
    root = Path(registry_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    target = root / skill_id / version
    if target.exists():
        existing = json.loads((target / "skill.json").read_text(encoding="utf-8"))
        if _canonical_hash(existing) != _canonical_hash(skill):
            raise ValueError("immutable Skill version already exists with different content")
        return {"path": str(target), "created": False, "skill_sha256": _canonical_hash(skill)}
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = Path(tempfile.mkdtemp(prefix=".skill-", dir=target.parent))
    try:
        (temporary / "skill.json").write_text(json.dumps(skill, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "SKILL.md").write_text(skill["skill_markdown"], encoding="utf-8")
        (temporary / "skill.json").chmod(0o600)
        (temporary / "SKILL.md").chmod(0o600)
        temporary.chmod(0o700)
        target.parent.chmod(0o700)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"path": str(target), "created": True, "skill_sha256": _canonical_hash(skill)}


def read_skill_index(registry_dir: str | Path) -> dict[str, Any]:
    path = Path(registry_dir).expanduser().resolve() / "index.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": "1", "consumed_session_ids": [], "evaluated_batches": []}
    except json.JSONDecodeError as error:
        raise ValueError("Skill registry index is malformed") from error
    if not isinstance(value, dict) or value.get("schema_version") != "1" or not isinstance(value.get("consumed_session_ids"), list) or not isinstance(value.get("evaluated_batches"), list):
        raise ValueError("Skill registry index has an unsupported format")
    verify_capability_proposal_signature(value)
    return value


def write_skill_index(registry_dir: str | Path, index: dict[str, Any]) -> None:
    root = Path(registry_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    target = root / "index.json"
    signed_index = sign_capability_proposal(index)
    payload = (json.dumps(signed_index, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=".skill-index-", dir=root)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, target)
        target.chmod(0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(name).unlink(missing_ok=True)
        raise
