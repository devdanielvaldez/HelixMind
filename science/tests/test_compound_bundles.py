from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch

from medai_science.reproducibility import create_compound_bundle, verify_analysis_bundle, verify_compound_bundle


def compound_record() -> dict:
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    return {
        "cid": 2244,
        "title": "Aspirin",
        "molecular_formula": "C9H8O4",
        "molecular_weight_g_mol": 180.16,
        "exact_mass": 180.042,
        "xlogp": 1.2,
        "smiles": smiles,
        "inchi": "InChI=1S/example",
        "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
        "rdkit": {
            "status": "computed",
            "software": "RDKit",
            "version": "2025.03.1",
            "input_smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
            "descriptors": {
                "molecular_weight_g_mol": 180.159,
                "tpsa_angstrom2": 63.6,
                "h_bond_donors": 1,
                "h_bond_acceptors": 4,
                "rotatable_bonds": 3,
                "lipophilicity_logp": 1.3101,
            },
            "limitations": "Test fixture; descriptors are not a safety or efficacy prediction.",
        },
        "source": "PubChem PUG REST",
        "source_url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244",
        "record_sha256": "a" * 64,
        "retrieved_at": "2026-01-01T00:00:00Z",
    }


class CompoundBundleTests(unittest.TestCase):
    def _rdkit_modules(self):
        package = ModuleType("rdkit")
        base = ModuleType("rdkit.rdBase")
        base.rdkitVersion = "2025.03.1"
        package.rdBase = base
        return {"rdkit": package, "rdkit.rdBase": base}

    def test_exact_compound_bundle_verifies_and_replays_without_network(self):
        with tempfile.TemporaryDirectory(prefix="medai-compound-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            expected = compound_record()["rdkit"]
            with patch("medai_science.sources.pubchem.PubChemClient._rdkit_descriptors", return_value=expected):
                created = create_compound_bundle(bundle, compound_record())
                self.assertEqual(created["artifact_count"], 2)
                self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
                self.assertEqual((bundle / "compound.json").stat().st_mode & 0o777, 0o600)
                self.assertTrue(verify_compound_bundle(bundle)["valid"])
                with patch.dict(sys.modules, self._rdkit_modules()):
                    replay = verify_analysis_bundle(bundle, replay=True)
                self.assertTrue(replay["replayed"])
                self.assertEqual(replay["report_sha256"], created["report_sha256"])

    def test_bundle_rejects_ambiguous_identity_or_missing_rdkit_calculation(self):
        record = compound_record()
        record["rdkit"] = {"status": "not_installed"}
        with tempfile.TemporaryDirectory(prefix="medai-compound-bundle-") as temp:
            with self.assertRaisesRegex(ValueError, "computed RDKit"):
                create_compound_bundle(Path(temp) / "bundle", record)
            with self.assertRaisesRegex(ValueError, "standard InChIKey"):
                create_compound_bundle(Path(temp) / "other", {**compound_record(), "inchikey": "ambiguous"})

    def test_bundle_tampering_is_detected_before_replay(self):
        with tempfile.TemporaryDirectory(prefix="medai-compound-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            with patch("medai_science.sources.pubchem.PubChemClient._rdkit_descriptors", return_value=compound_record()["rdkit"]):
                create_compound_bundle(bundle, compound_record())
            report = json.loads((bundle / "compound.json").read_text(encoding="utf-8"))
            report["rdkit"]["descriptors"]["molecular_weight_g_mol"] = 1
            (bundle / "compound.json").write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                verify_compound_bundle(bundle)

    def test_manifest_cannot_rebind_report_to_another_pubchem_cid(self):
        with tempfile.TemporaryDirectory(prefix="medai-compound-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            with patch("medai_science.sources.pubchem.PubChemClient._rdkit_descriptors", return_value=compound_record()["rdkit"]):
                create_compound_bundle(bundle, compound_record())
            report_path = bundle / "compound.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["cid"] = 999
            report_path.write_text(json.dumps(report), encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
            manifest["report_sha256"] = report_hash
            manifest["artifacts"][0]["sha256"] = report_hash
            manifest["artifacts"][0]["size"] = report_path.stat().st_size
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CID does not match"):
                verify_compound_bundle(bundle)

    def test_replay_fails_closed_when_rdkit_is_not_installed(self):
        with tempfile.TemporaryDirectory(prefix="medai-compound-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            with patch("medai_science.sources.pubchem.PubChemClient._rdkit_descriptors", return_value=compound_record()["rdkit"]):
                create_compound_bundle(bundle, compound_record())
            with patch.dict(sys.modules, {"rdkit": None, "rdkit.rdBase": None}):
                with self.assertRaisesRegex(ValueError, "RDKit is required"):
                    verify_compound_bundle(bundle, replay=True)


if __name__ == "__main__":
    unittest.main()
