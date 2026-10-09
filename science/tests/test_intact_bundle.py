from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from medai_science.reproducibility import create_intact_bundle, verify_analysis_bundle
from medai_science.sources.intact import IntActClient


def _mitab() -> str:
    return "\t".join([
        "uniprotkb:P05067", "uniprotkb:Q9Y6K9", "-", "-", "gene name:APP", "gene name:ABC1",
        "psi-mi:MI:0018", "author:Example et al.", "pubmed:12345678",
        "taxid:9606(Homo sapiens)", "taxid:9606(Homo sapiens)", "psi-mi:MI:0915(physical association)",
        "psi-mi:MI:0469(intact)", "intact:EBI-123456", "intact-miscore:0.72",
    ])


class IntActBundleTests(unittest.TestCase):
    def _result(self) -> dict:
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=_mitab() + "\n")))
        return asyncio.run(client.interactions("P05067"))

    def test_creates_minimized_offline_replay_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-intact-bundle-") as temp:
            bundle_dir = Path(temp) / "bundle"
            create_intact_bundle(bundle_dir, self._result())
            report = json.loads((bundle_dir / "interactions.json").read_text(encoding="utf-8"))
            self.assertNotIn("raw_mitab", report["records"][0])
            self.assertNotIn("aliases_a", report["records"][0])
            self.assertEqual(verify_analysis_bundle(bundle_dir)["valid"], True)
            self.assertEqual(verify_analysis_bundle(bundle_dir, replay=True)["network_accessed"], False)

    def test_rejects_source_record_tampering_even_when_bundle_hashes_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-intact-bundle-tamper-") as temp:
            bundle_dir = Path(temp) / "bundle"
            create_intact_bundle(bundle_dir, self._result())
            report_path = bundle_dir / "interactions.json"
            manifest_path = bundle_dir / "manifest.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["records"][0]["interactor_b"] = "uniprotkb:Q8N158"
            report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            manifest["report_sha256"] = report_digest
            for artifact in manifest["artifacts"]:
                if artifact["path"] == "interactions.json":
                    artifact["size"] = report_path.stat().st_size
                    artifact["sha256"] = report_digest
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reconciliation validation"):
                verify_analysis_bundle(bundle_dir)


if __name__ == "__main__":
    unittest.main()
