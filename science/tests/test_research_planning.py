from __future__ import annotations

import unittest
import hashlib
import json
import ast
from pathlib import Path

from medai_science.llm import ModelCompletion
from medai_science.research_planning import OBJECTIVE_PLAN_SCHEMA, MANDATORY_RESEARCH_TASK_IDS, RESEARCH_TASK_IDS, objective_plan_from_artifacts, paper_limit_for_depth, plan_research_objective, validate_task_scope
from medai_science.orchestrator import DEFAULT_PLAN
from medai_science.scientific_agents import AGENT_BY_TASK, ResearchDirector, agent_may_call


def valid_plan():
    return {"normalized_question": "How is APOE associated with Alzheimer disease?", "pubmed_queries": ["APOE Alzheimer disease", "APOE neuroinflammation"], "clinical_trials_query": "Alzheimer APOE", "concepts": ["APOE", "Alzheimer disease"], "scope_notes": ["Computational research only"], "selected_task_ids": list(RESEARCH_TASK_IDS), "task_scope_notes": ["Broad disease objective"]}


def planner_artifact(plan, objective="APOE and Alzheimer disease", depth="standard", source="MedAI configured model planner", session_id="RES-1", task_id="objective.normalize", kind="research_objective_plan"):
    digest = hashlib.sha256(objective.strip().encode()).hexdigest()
    payload = json.dumps({"schema_version": "2", "objective_sha256": digest, "depth": depth, "planning": plan}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"kind": kind, "source": source, "source_id": digest, "payload": payload, "sha256": hashlib.sha256(payload.encode()).hexdigest(), "session_id": session_id, "task_id": task_id}


class FakeProvider:
    async def complete_json(self, **kwargs):
        self.kwargs = kwargs
        return ModelCompletion(valid_plan(), "test", "fixture", "a" * 64, "b" * 64, {"total_tokens": 12})


class ResearchPlanningTests(unittest.IsolatedAsyncioTestCase):
    def test_execution_source_ceiling_scales_by_depth_and_preserves_explicit_override(self):
        self.assertEqual(paper_limit_for_depth("standard"), 10)
        self.assertEqual(paper_limit_for_depth("deep"), 20)
        self.assertEqual(paper_limit_for_depth("deep", 6), 6)
        for invalid in (0, 21, True, 2.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "1..20"):
                paper_limit_for_depth("deep", invalid)

    async def test_generates_bounded_query_plan_only(self):
        provider = FakeProvider()
        result = await plan_research_objective(provider, "  APOE and Alzheimer disease  ")
        self.assertEqual(result.value["pubmed_queries"], valid_plan()["pubmed_queries"])
        self.assertEqual(provider.kwargs["schema"], OBJECTIVE_PLAN_SCHEMA)
        self.assertIn("claim evidence", provider.kwargs["system"])

    async def test_rejects_patient_identifiers_and_unsafe_model_queries(self):
        provider = FakeProvider()
        with self.assertRaises(ValueError):
            await plan_research_objective(provider, "Patient john@example.com has Alzheimer")
        async def unsafe(**kwargs):
            plan = valid_plan()
            plan["pubmed_queries"][0] = "patient john@example.com"
            return ModelCompletion(plan, "test", "fixture", "a" * 64, "b" * 64, {})
        provider.complete_json = unsafe
        with self.assertRaises(ValueError):
            await plan_research_objective(provider, "Alzheimer disease")
        async def unsafe_scope_notes(**kwargs):
            plan = valid_plan()
            plan["task_scope_notes"] = ["Patient john@example.com records are out of scope"]
            return ModelCompletion(plan, "test", "fixture", "a" * 64, "b" * 64, {})
        provider.complete_json = unsafe_scope_notes
        with self.assertRaises(ValueError):
            await plan_research_objective(provider, "Alzheimer disease")

    def test_artifact_loader_accepts_only_valid_safe_plans(self):
        rows = [planner_artifact(valid_plan())]
        self.assertEqual(objective_plan_from_artifacts(rows, objective="APOE and Alzheimer disease", depth="standard", session_id="RES-1"), valid_plan())
        unsafe = valid_plan()
        unsafe["clinical_trials_query"] = "patient@example.com"
        self.assertIsNone(objective_plan_from_artifacts([planner_artifact(unsafe)], objective="APOE and Alzheimer disease", depth="standard", session_id="RES-1"))
        self.assertIsNone(objective_plan_from_artifacts([planner_artifact(valid_plan(), source="untrusted")], objective="APOE and Alzheimer disease", depth="standard", session_id="RES-1"))

    def test_task_scope_is_registered_mandatory_and_dependency_safe(self):
        plan = valid_plan()
        plan["selected_task_ids"] = ["objective.normalize", "literature.discover", "literature.extract", "entities.resolve", "evidence.grade", "contradictions.search", "graph.build", "hypotheses.propose", "hypotheses.criticize", "report.reproduce"]
        self.assertEqual(set(validate_task_scope(plan)), set(plan["selected_task_ids"]))
        plan["selected_task_ids"].remove("contradictions.search")
        with self.assertRaisesRegex(ValueError, "mandatory"):
            validate_task_scope(plan)
        plan["selected_task_ids"] = list(MANDATORY_RESEARCH_TASK_IDS) + ["drugs.investigate"]
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            validate_task_scope(plan)
        plan["selected_task_ids"] = list(RESEARCH_TASK_IDS) + ["invented.tool"]
        with self.assertRaisesRegex(ValueError, "duplicate or unregistered"):
            validate_task_scope(plan)

    def test_artifact_loader_rejects_objective_or_depth_drift_and_falls_back_to_matching_plan(self):
        objective = "APOE and Alzheimer disease"
        valid = planner_artifact(valid_plan(), objective=objective, depth="deep")
        stale_schema = planner_artifact(valid_plan(), objective=objective, depth="deep")
        stale_schema["payload"] = stale_schema["payload"].replace('"schema_version":"2"', '"schema_version":"1"')
        stale_schema["sha256"] = hashlib.sha256(stale_schema["payload"].encode()).hexdigest()
        stale_objective = planner_artifact(valid_plan(), objective="Parkinson disease", depth="deep")
        stale_depth = planner_artifact(valid_plan(), objective=objective, depth="standard")
        self.assertIsNone(objective_plan_from_artifacts([stale_objective], objective=objective, depth="deep", session_id="RES-1"))
        self.assertIsNone(objective_plan_from_artifacts([stale_depth], objective=objective, depth="deep", session_id="RES-1"))
        self.assertIsNone(objective_plan_from_artifacts([stale_schema], objective=objective, depth="deep", session_id="RES-1"))
        self.assertEqual(objective_plan_from_artifacts([valid, stale_objective], objective=objective, depth="deep", session_id="RES-1"), valid_plan())

    def test_artifact_loader_rechecks_payload_digest_and_rejects_ambiguous_json(self):
        objective = "APOE and Alzheimer disease"
        artifact = planner_artifact(valid_plan(), objective=objective)
        artifact["payload"] = artifact["payload"].replace('"APOE"', '"APOE","unexpected":"tampered"', 1)
        self.assertIsNone(objective_plan_from_artifacts([artifact], objective=objective, depth="standard", session_id="RES-1"))
        ambiguous = planner_artifact(valid_plan(), objective=objective)
        ambiguous["payload"] = ambiguous["payload"].replace('"depth":"standard"', '"depth":"deep","depth":"standard"')
        ambiguous["sha256"] = hashlib.sha256(ambiguous["payload"].encode()).hexdigest()
        self.assertIsNone(objective_plan_from_artifacts([ambiguous], objective=objective, depth="standard", session_id="RES-1"))
        non_finite = planner_artifact(valid_plan(), objective=objective)
        non_finite["payload"] = '{"schema_version":"2","objective_sha256":"' + hashlib.sha256(objective.encode()).hexdigest() + '","depth":"standard","planning":' + json.dumps(valid_plan()) + ',"unused":NaN}'
        non_finite["sha256"] = hashlib.sha256(non_finite["payload"].encode()).hexdigest()
        self.assertIsNone(objective_plan_from_artifacts([non_finite], objective=objective, depth="standard", session_id="RES-1"))

    def test_artifact_loader_rejects_plans_from_other_sessions_or_tasks(self):
        objective = "APOE and Alzheimer disease"
        valid = planner_artifact(valid_plan())
        wrong_session = planner_artifact(valid_plan(), session_id="RES-2")
        wrong_task = planner_artifact(valid_plan(), task_id="literature.discover")
        wrong_kind = planner_artifact(valid_plan(), kind="generic_note")
        for artifact in (wrong_session, wrong_task, wrong_kind):
            self.assertIsNone(objective_plan_from_artifacts([artifact], objective=objective, depth="standard", session_id="RES-1"))
        self.assertEqual(objective_plan_from_artifacts([valid], objective=objective, depth="standard", session_id="RES-1"), valid_plan())

    def test_research_director_assigns_every_persisted_step_deterministically(self):
        tasks = [
            {"id": step.id, "domain": step.domain, "depends_on": list(step.depends_on), "status": "ready"}
            for step in DEFAULT_PLAN
        ]
        director = ResearchDirector()
        first = director.coordinate(session_id="RES-1", objective="APOE and Alzheimer disease", depth="deep", budget=500, tasks=tasks)
        second = director.coordinate(session_id="RES-1", objective="APOE and Alzheimer disease", depth="deep", budget=500, tasks=tasks)
        self.assertEqual(first, second)
        self.assertEqual({row["task_id"] for row in first["task_assignments"]}, set(AGENT_BY_TASK))
        self.assertEqual(first["schema"], "medai.scientific-agent-coordination.v1")
        self.assertEqual(first["execution_plane"], "AgentMesh MCP gateway")
        self.assertIn("not claim independent agent reasoning", first["interpretation_limit"])

    def test_research_director_rejects_plan_drift_or_ambiguous_dependencies(self):
        tasks = [
            {"id": step.id, "domain": step.domain, "depends_on": list(step.depends_on)}
            for step in DEFAULT_PLAN
        ]
        tasks[1]["domain"] = "pharmacology"
        with self.assertRaisesRegex(ValueError, "domain differs"):
            ResearchDirector().coordinate(session_id="RES-1", objective="Alzheimer disease", depth="standard", budget=100, tasks=tasks)
        tasks[1]["domain"] = DEFAULT_PLAN[1].domain
        tasks[1]["depends_on"] = ["missing.task"]
        with self.assertRaisesRegex(ValueError, "dependencies differ"):
            ResearchDirector().coordinate(session_id="RES-1", objective="Alzheimer disease", depth="standard", budget=100, tasks=tasks)

    def test_specialist_mcp_authority_is_domain_scoped(self):
        self.assertTrue(agent_may_call("research_director", "medai_research_apply_task_scope"))
        self.assertFalse(agent_may_call("literature_librarian", "medai_research_apply_task_scope"))
        self.assertTrue(agent_may_call("gene_disease_curator", "medai_graph_add_entity"))
        self.assertTrue(agent_may_call("gene_disease_curator", "medai_graph_link_evidence"))
        self.assertTrue(agent_may_call("gene_disease_curator", "medai_ontology_search"))
        self.assertTrue(agent_may_call("gene_disease_curator", "medai_ontology_cross_mapping_review"))
        self.assertFalse(agent_may_call("gene_disease_curator", "medai_drug_profile_chembl"))
        self.assertTrue(agent_may_call("protein_analyst", "medai_proteins_get_uniprot_record"))
        self.assertTrue(agent_may_call("protein_analyst", "medai_research_save_checkpoint"))
        self.assertFalse(agent_may_call("protein_analyst", "medai_drug_profile_chembl"))
        self.assertFalse(agent_may_call("unknown_agent", "medai_research_store_artifact"))
        self.assertTrue(agent_may_call("hypothesis_generator", "medai_papers_search_pubmed"))
        self.assertFalse(agent_may_call("hypothesis_generator", "medai_drug_interaction_labels_openfda"))
        self.assertTrue(agent_may_call("capability_learner", "medai_capabilities_benchmark_proposal_versions"))
        self.assertTrue(agent_may_call("capability_learner", "medai_capabilities_generate_python_proposal"))
        self.assertTrue(agent_may_call("capability_learner", "medai_capabilities_generate_r_proposal"))
        self.assertTrue(agent_may_call("capability_learner", "medai_capabilities_validate_python_proposal"))
        self.assertTrue(agent_may_call("capability_learner", "medai_capabilities_validate_r_proposal"))
        self.assertFalse(agent_may_call("capability_learner", "medai_capabilities_execute_generated"))

    def test_worker_task_tool_calls_are_allowed_for_the_assigned_specialist(self):
        worker_path = Path(__file__).parents[1] / "src/medai_science/worker.py"
        tree = ast.parse(worker_path.read_text(encoding="utf-8"))
        task_branches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            if not isinstance(node.test.left, ast.Name) or node.test.left.id != "task_id" or node.col_offset < 20:
                continue
            task_ids = [item.value for item in node.test.comparators if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            if task_ids:
                task_branches.append((task_ids[0], node))
        self.assertEqual({task_id for task_id, _ in task_branches}, set(AGENT_BY_TASK))
        for task_id, branch in task_branches:
            agent_id = AGENT_BY_TASK[task_id]
            tool_names = {
                node.args[0].value
                for statement in branch.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "call"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            }
            unauthorized = sorted(name for name in tool_names if not agent_may_call(agent_id, name))
            self.assertEqual(unauthorized, [], f"{task_id} ({agent_id}) calls MCP tools outside its allowlist")


if __name__ == "__main__":
    unittest.main()
