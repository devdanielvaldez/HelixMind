from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medai_science.orchestrator import DEFAULT_PLAN
from medai_science.provenance_signing import verify_capability_proposal_signature
from medai_science.skill_learning import learn_research_skill_batch, persist_skill_candidate, tenant_skill_registry_dir, verify_research_bundle
from medai_science.skill_export import build_agentmesh_skill_review_package, read_verified_skill, render_agentmesh_skill_review_package, register_agentmesh_skill_review_package


def make_history(index: int, status_override: dict[str, str] | None = None):
    session_id = f"RES-session-{index}"
    task_rows = []
    for task in DEFAULT_PLAN:
        status = (status_override or {}).get(task.id, "succeeded")
        task_rows.append({"id": task.id, "status": status, "attempts": 1, "output_sha256": hashlib.sha256(f"{session_id}:{task.id}".encode()).hexdigest()})
    bundle = {
        "schema_version": "1",
        "session": {"id": session_id, "objective": f"Research objective {index}", "status": "running"},
        "tasks": task_rows,
        "artifacts": [],
        "evidence_annotations": [],
        "graph": {"nodes": [], "edges": []},
        "limitations": ["Fixture only"],
    }
    bundle["bundle_sha256"] = hashlib.sha256(json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    live_tasks = [{"id": row["id"], "status": row["status"], "domain": task.domain, "depends_on": list(task.depends_on)} for row, task in zip(task_rows, DEFAULT_PLAN)]
    return {"session_id": session_id, "session_status": "completed", "created_at": index, "plan": {"session_id": session_id, "tasks": live_tasks}, "bundle": bundle}


class SkillLearningTests(unittest.TestCase):
    def test_tenant_skill_registries_are_stably_partitioned_without_plaintext_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            first = tenant_skill_registry_dir(folder, "research-team-a")
            second = tenant_skill_registry_dir(folder, "research-team-b")
            self.assertEqual(first, tenant_skill_registry_dir(folder, "research-team-a"))
            self.assertNotEqual(first, second)
            self.assertNotIn("research-team-a", str(first))

    def test_learns_only_shared_successful_task_sequence_and_signs_candidate(self):
        histories = [make_history(index) for index in range(1, 4)]
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": "test-only-signing-key-with-at-least-32-bytes"}, clear=True):
            candidate, index, result = learn_research_skill_batch(histories)
            self.assertEqual(result["status"], "candidate_created")
            self.assertEqual(candidate["status"], "pending_review")
            self.assertFalse(candidate["execution"]["enabled"])
            self.assertFalse(candidate["execution"]["network"])
            self.assertEqual(candidate["evaluation"]["completed_sessions"], 3)
            self.assertEqual(len(candidate["procedure"]), len(DEFAULT_PLAN) - 1)
            self.assertTrue(verify_capability_proposal_signature(candidate)["verified"])
            self.assertEqual(len(index["consumed_session_ids"]), 3)
            self.assertIn("not scientific accuracy", candidate["evaluation"]["interpretation"])

    def test_refuses_tampered_bundles_and_requires_common_core_procedure(self):
        histories = [make_history(index) for index in range(1, 4)]
        damaged = make_history(1)
        damaged["bundle"]["session"]["objective"] = "tampered"
        with self.assertRaisesRegex(ValueError, "integrity"):
            verify_research_bundle(damaged["bundle"], damaged["session_id"])
        failed = [make_history(index, {"literature.extract": "skipped"}) for index in range(4, 7)]
        candidate, index, result = learn_research_skill_batch(failed)
        self.assertIsNone(candidate)
        self.assertEqual(result["status"], "no_common_core_procedure")
        self.assertEqual(len(index["consumed_session_ids"]), 3)
        self.assertEqual(len(histories), 3)

    def test_rejects_duplicate_or_plan_bundle_outcome_drift(self):
        duplicate = make_history(1)
        duplicate["plan"]["tasks"].append(dict(duplicate["plan"]["tasks"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate live task plan"):
            learn_research_skill_batch([duplicate])

        drifted = make_history(2)
        drifted["bundle"]["tasks"][0]["status"] = "failed"
        unsigned = {key: value for key, value in drifted["bundle"].items() if key != "bundle_sha256"}
        drifted["bundle"]["bundle_sha256"] = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "outcomes differ"):
            learn_research_skill_batch([drifted])

    def test_learned_procedure_prunes_a_successful_stage_with_missing_prerequisite(self):
        histories = [make_history(index, {"compounds.analyze": "skipped", "drugs.investigate": "succeeded"}) for index in range(1, 4)]
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": "test-only-signing-key-with-at-least-32-bytes"}, clear=True):
            candidate, _, result = learn_research_skill_batch(histories)
        self.assertEqual(result["status"], "candidate_created")
        procedure_ids = {step["task_id"] for step in candidate["procedure"]}
        self.assertNotIn("drugs.investigate", procedure_ids)
        self.assertNotIn("compounds.analyze", procedure_ids)

    def test_enforces_three_complete_reproducible_sessions_and_batches_history(self):
        candidate, index, result = learn_research_skill_batch([make_history(1), make_history(2)])
        self.assertIsNone(candidate)
        self.assertEqual(index, {})
        self.assertEqual(result["status"], "insufficient_history")

    def test_persists_an_immutable_private_versioned_skill_package(self):
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": "test-only-signing-key-with-at-least-32-bytes"}, clear=True):
            candidate, _, _ = learn_research_skill_batch([make_history(index) for index in range(1, 4)])
        with tempfile.TemporaryDirectory() as temp:
            key = b"test-only-signing-key-with-at-least-32-bytes"
            first = persist_skill_candidate(candidate, temp, key)
            second = persist_skill_candidate(candidate, temp, key)
            self.assertTrue(first["created"])
            self.assertFalse(second["created"])
            directory = Path(first["path"])
            self.assertEqual((directory / "skill.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual((directory / "SKILL.md").stat().st_mode & 0o777, 0o600)
            loaded = json.loads((directory / "skill.json").read_text())
            self.assertTrue(verify_capability_proposal_signature(loaded, b"test-only-signing-key-with-at-least-32-bytes")["verified"])
            self.assertIn("pending_review", (directory / "SKILL.md").read_text())

    def test_exports_only_verified_disabled_skills_as_agentmesh_review_packages(self):
        key = b"test-only-signing-key-with-at-least-32-bytes"
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": key.decode()}, clear=True):
            candidate, _, _ = learn_research_skill_batch([make_history(index) for index in range(1, 4)])
        with tempfile.TemporaryDirectory() as temp:
            persisted = persist_skill_candidate(candidate, temp, key)
            loaded = read_verified_skill(temp, candidate["id"], candidate["version"], key)
            package = build_agentmesh_skill_review_package(loaded, key)
            self.assertEqual(package["schemaVersion"], "amcp/0.1")
            self.assertEqual(package["capability"]["id"], "biomedical.skill.biomedical.research-sequence")
            self.assertEqual(package["implementations"][0]["reference"], "mcp://medai/medai_skills_get_candidate")
            self.assertEqual(package["contract"]["effects"], ["state_read"])
            self.assertFalse(package["x-medai-learned-skill"]["executionEnabled"])
            rendered, digest = render_agentmesh_skill_review_package(loaded, key)
            self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), digest)
            self.assertIn("x-medai-learned-skill:", rendered)
            artifact = Path(persisted["path"]) / "skill.json"
            tampered = json.loads(artifact.read_text())
            tampered["skill_markdown"] += "\nUnreviewed change.\n"
            artifact.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "signature|digest|hash"):
                read_verified_skill(temp, candidate["id"], candidate["version"], key)

    def test_registration_validates_before_installing_and_never_enables_execution(self):
        import subprocess

        key = b"test-only-signing-key-with-at-least-32-bytes"
        with patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": key.decode()}, clear=True):
            candidate, _, _ = learn_research_skill_batch([make_history(index) for index in range(1, 4)])
        with tempfile.TemporaryDirectory() as catalog, patch("medai_science.skill_export.subprocess.run") as run:
            run.side_effect = [subprocess.CompletedProcess([], 0, "valid", ""), subprocess.CompletedProcess([], 0, "installed", "")]
            receipt = register_agentmesh_skill_review_package(candidate, catalog, "/usr/bin/agentmesh", key)
        self.assertTrue(receipt["validated"])
        self.assertTrue(receipt["installed"])
        self.assertTrue(receipt["review_required"])
        self.assertFalse(receipt["execution_enabled"])
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][1:3], ["capabilities", "validate"])
        self.assertEqual(run.call_args_list[1].args[0][1:3], ["capabilities", "install"])


if __name__ == "__main__":
    unittest.main()
