from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from medai_science.reproducibility import create_structure_bundle, verify_analysis_bundle, verify_structure_bundle
from medai_science.structure_analysis import summarize_pdb


def atom(serial: int, name: str, residue: str, number: int) -> str:
    return f"ATOM  {serial:5d} {name:<4} {residue:>3} A{number:4d}    {1.0:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{25.0:6.2f}          C \n"


class StructureBundleTests(unittest.TestCase):
    def test_pdb_bundle_replays_exact_parser_privately(self):
        with tempfile.TemporaryDirectory(prefix="medai-structure-bundle-") as temp:
            root = Path(temp)
            source = root / "local.pdb"
            source.write_text(atom(1, "CA", "ALA", 1) + "END\n", encoding="ascii")
            report = summarize_pdb(source.read_bytes(), label="fixture-1")
            bundle = root / "bundle"
            created = create_structure_bundle(bundle, report, source, "pdb")
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(created["artifact_count"], 3)
            self.assertEqual(manifest["analysis"]["workflow"], "protein.coordinate_summary")
            self.assertFalse(manifest["network_accessed"])
            self.assertEqual((bundle / "inputs/structure.pdb").read_bytes(), source.read_bytes())
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual((bundle / "inputs/structure.pdb").stat().st_mode & 0o777, 0o600)
            self.assertTrue(verify_analysis_bundle(bundle)["valid"])
            self.assertTrue(verify_structure_bundle(bundle, replay=True)["replayed"])

    def test_verifier_rejects_changed_coordinate_bytes(self):
        with tempfile.TemporaryDirectory(prefix="medai-structure-bundle-") as temp:
            root = Path(temp)
            source = root / "local.pdb"
            source.write_text(atom(1, "CA", "ALA", 1) + "END\n", encoding="ascii")
            bundle = root / "bundle"
            create_structure_bundle(bundle, summarize_pdb(source.read_bytes()), source, "pdb")
            (bundle / "inputs/structure.pdb").write_text("tampered\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "artifact hash or size mismatch"):
                verify_structure_bundle(bundle)

    def test_creation_requires_report_to_match_exact_input_and_format(self):
        with tempfile.TemporaryDirectory(prefix="medai-structure-bundle-") as temp:
            root = Path(temp)
            source = root / "local.pdb"
            source.write_text(atom(1, "CA", "ALA", 1) + "END\n", encoding="ascii")
            report = summarize_pdb(source.read_bytes())
            with self.assertRaisesRegex(ValueError, "does not match"):
                create_structure_bundle(root / "bundle", {**report, "atom_count": 9}, source, "pdb")
            with self.assertRaisesRegex(ValueError, "format must be"):
                create_structure_bundle(root / "invalid", report, source, "cif")


if __name__ == "__main__":
    unittest.main()
