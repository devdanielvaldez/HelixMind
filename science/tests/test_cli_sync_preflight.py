from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import typer

from medai_science.cli import agentmesh_sync_capabilities, skills_sync_agentmesh


class AgentMeshSyncPreflightTests(unittest.TestCase):
    def _write_plan(self, directory: str, plan: dict) -> Path:
        path = Path(directory) / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path

    def test_capability_sync_rejects_malformed_digest_before_admin_request(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._write_plan(directory, {"plan_review_sha256": "a" * 64})
            with patch("medai_science.cli._medai_admin_request") as request:
                with self.assertRaisesRegex(typer.BadParameter, "exactly match"):
                    agentmesh_sync_capabilities(plan, "A" * 64, "receipt", Path(directory) / "state.json")
                request.assert_not_called()

    def test_capability_sync_rejects_digest_for_another_plan_before_admin_request(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._write_plan(directory, {"plan_review_sha256": "a" * 64})
            with patch("medai_science.cli._medai_admin_request") as request:
                with self.assertRaisesRegex(typer.BadParameter, "exactly match"):
                    agentmesh_sync_capabilities(plan, "b" * 64, "receipt", Path(directory) / "state.json")
                request.assert_not_called()

    def test_skill_sync_rejects_wrong_plan_kind_before_admin_request(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self._write_plan(directory, {"plan_review_sha256": "a" * 64})
            with patch("medai_science.cli._medai_admin_request") as request:
                with self.assertRaisesRegex(typer.BadParameter, "requires a plan"):
                    skills_sync_agentmesh(plan, "a" * 64, "receipt", Path(directory) / "state.json")
                request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
