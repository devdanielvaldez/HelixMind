from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medai_science.reproducibility import _canonical_hash, create_pathway_bundle, verify_analysis_bundle, verify_pathway_bundle
from medai_science.sources.reactome import ReactomeClient


def pathway_result() -> dict:
    genes = ["APOE", "TREM2"]
    parameters = {"pageSize": 20, "sortBy": "ENTITIES_PVALUE", "order": "ASC", "species": "Homo sapiens"}
    return {
        "source": "Reactome Analysis Service",
        "source_url": ReactomeClient.endpoint,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "species": "Homo sapiens",
        "input_identifiers": genes,
        "input_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "analysis_parameters": parameters,
        "request_sha256": _canonical_hash({"endpoint": ReactomeClient.endpoint, "parameters": parameters, "identifiers": genes}),
        "result": {"pathways": [{"stId": "R-HSA-123", "name": "Example pathway"}]},
        "interpretation_limit": "Pathway over-representation is an association analysis, not causal evidence.",
    }


class PathwayBundleTests(unittest.TestCase):
    def test_bundle_verifies_and_replays_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "pathway-bundle"
            create_pathway_bundle(bundle, pathway_result())
            self.assertTrue(verify_pathway_bundle(bundle, replay=True)["network_accessed"] is False)
            self.assertTrue(verify_analysis_bundle(bundle, replay=True)["replayed"])

    def test_rejects_mismatched_input_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bad = pathway_result()
            bad["input_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "exact input"):
                create_pathway_bundle(Path(temp) / "bad", bad)

    def test_tampered_response_fails_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "pathway-bundle"
            create_pathway_bundle(bundle, pathway_result())
            report = bundle / "pathways.json"
            payload = json.loads(report.read_text())
            payload["result"]["pathways"] = []
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_pathway_bundle(bundle, replay=True)


if __name__ == "__main__":
    unittest.main()
