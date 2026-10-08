from __future__ import annotations

import unittest

from medai_science.research_workflows import build_research_workflow_proposal


SESSION = {"id": "RES-1", "status": "completed"}
TASKS = [
    {"id": "b", "domain": "literature", "description": "Read papers", "depends_on": ["a"], "status": "succeeded", "attempts": 1, "output": {"papers": ["PMID:1"]}},
    {"id": "a", "domain": "planning", "description": "Plan research", "depends_on": [], "status": "succeeded", "attempts": 1, "output": {"private": "must not be copied"}},
]


class ResearchWorkflowProposalTests(unittest.TestCase):
    def test_produces_deterministic_review_gated_topology_and_hash_only_run_receipts(self):
        first = build_research_workflow_proposal(SESSION, {"session_id": "RES-1", "tasks": TASKS})
        second = build_research_workflow_proposal(SESSION, {"session_id": "RES-1", "tasks": list(reversed(TASKS))})
        self.assertEqual(first["version"], second["version"])
        self.assertEqual(first["topology_sha256"], second["topology_sha256"])
        self.assertEqual([step["id"] for step in first["steps"]], ["a", "b"])
        self.assertEqual(first["steps"][1]["depends_on"], ["a"])
        self.assertEqual(first["source_run"]["outcome_coverage"], 1)
        self.assertTrue(all("output" not in row and "error" not in row for row in first["source_run"]["task_outcomes"]))
        self.assertNotIn("must not be copied", str(first))
        self.assertFalse(first["execution"]["enabled"])
        self.assertEqual(first["lifecycle_status"], "experimental_pending_review")

    def test_rejects_cross_session_plan(self):
        with self.assertRaisesRegex(ValueError, "another session"):
            build_research_workflow_proposal(SESSION, {"session_id": "RES-2", "tasks": TASKS})

    def test_rejects_invalid_graphs(self):
        cases = [
            ([{"id": "a", "domain": "x", "description": "x", "depends_on": ["a"], "status": "ready", "attempts": 0}], "self-reference"),
            ([{"id": "a", "domain": "x", "description": "x", "depends_on": ["missing"], "status": "ready", "attempts": 0}], "missing dependency"),
            ([{"id": "a", "domain": "x", "description": "x", "depends_on": ["b"], "status": "ready", "attempts": 0}, {"id": "b", "domain": "x", "description": "x", "depends_on": ["a"], "status": "ready", "attempts": 0}], "cycle"),
        ]
        for tasks, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_research_workflow_proposal(SESSION, {"session_id": "RES-1", "tasks": tasks})


if __name__ == "__main__":
    unittest.main()
