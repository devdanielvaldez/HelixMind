import unittest
from pathlib import Path

import yaml


class CapabilityBenchmarkContractTests(unittest.TestCase):
    def test_agentmesh_contract_matches_the_mcp_benchmark_policy(self):
        path = Path(__file__).resolve().parents[2] / "integrations/agentmesh/capabilities/biomedical.capability-benchmark-proposal-versions.yaml"
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(contract["schemaVersion"], "amcp/0.1")
        self.assertEqual(contract["capability"]["id"], "biomedical.capability-benchmark-proposal-versions")
        self.assertEqual(contract["contract"]["inputs"]["properties"]["cases"]["minItems"], 10)
        self.assertEqual(contract["contract"]["inputs"]["properties"]["repeats"]["maximum"], 15)
        outputs = contract["contract"]["outputs"]["properties"]
        self.assertIs(outputs["registered"]["const"], False)
        self.assertIs(outputs["promoted"]["const"], False)
        self.assertIn("benchmark_proposal_versions", contract["implementations"][0]["reference"])


if __name__ == "__main__":
    unittest.main()
