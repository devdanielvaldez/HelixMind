from __future__ import annotations

import hashlib
import unittest

from medai_science.structure_analysis import MAX_PDB_BYTES, summarize_pdb


def atom(serial: int, name: str, residue: str, number: int, *, factor: float = 25.0, x: float = 1.0, altloc: str = " ") -> str:
    return (
        f"ATOM  {serial:5d} {name:<4}{altloc}{residue:>3} A{number:4d}    "
        f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}{1.0:6.2f}{factor:6.2f}          C \n"
    )


class ProteinStructureSummaryTests(unittest.TestCase):
    def test_reports_hash_linked_chain_residue_and_temperature_factor_aggregates(self):
        payload = (atom(1, "N", "ALA", 1, factor=10) + atom(2, "CA", "ALA", 1, factor=20)
                   + atom(3, "CA", "GLY", 2, factor=30) + "END\n").encode("ascii")
        result = summarize_pdb(payload, label="AF-P12345-F1")
        self.assertEqual(result["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual((result["atom_count"], result["residue_count"], result["alpha_carbon_count"]), (3, 2, 2))
        self.assertEqual(result["chains"][0]["sequence_sha256"], hashlib.sha256(b"AG").hexdigest())
        self.assertEqual(result["atom_temperature_factor"]["median"], 20)
        self.assertIn("not identified as pLDDT", result["atom_temperature_factor"]["interpretation"])
        self.assertFalse(result["coordinates_retained"])

    def test_counts_models_and_omits_nonprimary_alternate_locations(self):
        payload = ("MODEL        1\n" + atom(1, "CA", "ALA", 1) + atom(2, "CA", "ALA", 1, altloc="B")
                   + "ENDMDL\nMODEL        2\n" + atom(3, "CA", "GLY", 1) + "ENDMDL\nEND\n").encode("ascii")
        result = summarize_pdb(payload)
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(result["atom_count"], 2)
        self.assertEqual(result["chain_count"], 2)
        self.assertEqual(result["unique_chain_id_count"], 1)
        self.assertEqual(result["discarded_alternate_atom_count"], 1)
        self.assertEqual([row["model"] for row in result["chains"]], [1, 2])

    def test_rejects_ambiguous_or_corrupt_coordinate_records(self):
        good = atom(1, "CA", "ALA", 1)
        with self.assertRaisesRegex(ValueError, "duplicate accepted atom"):
            summarize_pdb((good + good).encode("ascii"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_pdb(atom(1, "CA", "ALA", 1, x=float("nan")).encode("ascii"))
        with self.assertRaisesRegex(ValueError, "truncated"):
            summarize_pdb(b"ATOM  00001 CA")
        malformed_alternate = atom(2, "CA", "ALA", 1, altloc="B").replace("   1.000", "     nan", 1)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_pdb(malformed_alternate.encode("ascii"))

    def test_enforces_payload_and_identifier_bounds(self):
        with self.assertRaisesRegex(ValueError, "PDB input must contain"):
            summarize_pdb(b"x" * (MAX_PDB_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "safe identifier"):
            summarize_pdb(atom(1, "CA", "ALA", 1).encode("ascii"), label="bad label")

    def test_rejects_non_line_ending_ascii_control_characters(self):
        valid = atom(1, "CA", "ALA", 1).encode("ascii")
        for control in (b"\x00", b"\t", b"\x0b", b"\x1b", b"\x7f"):
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "forbidden control"):
                summarize_pdb(valid.replace(b"ALA", b"A" + control + b"A"))


if __name__ == "__main__":
    unittest.main()
