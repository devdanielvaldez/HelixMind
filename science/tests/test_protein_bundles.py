from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medai_science.reproducibility import create_protein_bundle, verify_analysis_bundle, verify_protein_bundle
from medai_science.sources.uniprot import _entry_features


def protein_record() -> dict:
    record = {
        "primaryAccession": "P05067",
        "features": [
            {"type": "Domain", "featureId": "PRO_000000001", "description": "E1 domain", "location": {"start": {"value": 20}, "end": {"value": 120}}},
            {"type": "Signal", "description": "Signal peptide", "location": {"start": {"value": 1}, "end": {"value": 17}}},
        ],
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "1AAP", "properties": [{"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.40 A"}, {"key": "Chains", "value": "A=18-120"}]},
            {"database": "GO", "id": "GO:0000000", "properties": []},
        ],
        "sequence": {"length": 770, "value": "M" * 770},
    }
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    domains, structures = _entry_features(record)
    return {
        "source": "UniProtKB", "accession": "P05067", "entry_url": "https://www.uniprot.org/uniprotkb/P05067",
        "retrieved_at": "2026-10-08T00:00:00Z", "license": "CC BY 4.0",
        "record_sha256": hashlib.sha256(canonical).hexdigest(), "record": record,
        "domains_and_regions": domains, "experimental_structure_cross_references": structures,
        "structure_cross_reference_limit": "provider metadata only",
    }


class ProteinBundleTests(unittest.TestCase):
    def test_bundle_verifies_and_replays_exact_uniprot_feature_extraction(self):
        with tempfile.TemporaryDirectory(prefix="medai-protein-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            result = protein_record()
            created = create_protein_bundle(bundle, result)
            self.assertEqual(created["artifact_count"], 2)
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual((bundle / "protein.json").stat().st_mode & 0o777, 0o600)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_license"], "CC BY 4.0")
            self.assertFalse(manifest["network_accessed_during_replay"])
            self.assertTrue(verify_protein_bundle(bundle)["valid"])
            replay = verify_analysis_bundle(bundle, replay=True)
            self.assertTrue(replay["valid"])
            self.assertTrue(replay["replayed"])
            self.assertEqual(replay["report_sha256"], created["report_sha256"])

    def test_bundle_rejects_modified_features_and_wrong_source_digest(self):
        result = protein_record()
        with tempfile.TemporaryDirectory(prefix="medai-protein-bundle-") as temp:
            with self.assertRaisesRegex(ValueError, "do not match"):
                create_protein_bundle(Path(temp) / "bad-features", {**result, "domains_and_regions": []})
            with self.assertRaisesRegex(ValueError, "digest"):
                create_protein_bundle(Path(temp) / "bad-digest", {**result, "record_sha256": "0" * 64})

    def test_bundle_detects_tampering_before_replay(self):
        with tempfile.TemporaryDirectory(prefix="medai-protein-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            create_protein_bundle(bundle, protein_record())
            report = json.loads((bundle / "protein.json").read_text(encoding="utf-8"))
            report["record"]["features"][0]["description"] = "tampered"
            (bundle / "protein.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                verify_protein_bundle(bundle, replay=True)


if __name__ == "__main__":
    unittest.main()
