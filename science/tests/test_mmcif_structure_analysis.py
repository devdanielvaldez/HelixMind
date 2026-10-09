from __future__ import annotations

import hashlib
from importlib.util import find_spec
import sys
import types
import unittest
from unittest.mock import patch

from medai_science.structure_analysis import summarize_mmcif


def mmcif_columns():
    return {
        "_atom_site.group_PDB": ["ATOM", "ATOM", "ATOM", "HETATM"],
        "_atom_site.label_atom_id": ["N", "CA", "CA", "C1"],
        "_atom_site.label_alt_id": [".", ".", "B", "."],
        "_atom_site.label_comp_id": ["ALA", "ALA", "ALA", "LIG"],
        "_atom_site.auth_asym_id": ["A", "A", "A", "A"],
        "_atom_site.auth_seq_id": ["1", "1", "1", "2"],
        "_atom_site.pdbx_PDB_ins_code": ["?", "?", "?", "?"],
        "_atom_site.Cartn_x": ["0", "1", "2", "3"],
        "_atom_site.Cartn_y": ["0", "1", "2", "3"],
        "_atom_site.Cartn_z": ["0", "1", "2", "3"],
        "_atom_site.occupancy": ["1", "1", "0.5", "1"],
        "_atom_site.B_iso_or_equiv": ["10", "20", "25", "30"],
        "_atom_site.pdbx_PDB_model_num": ["1", "1", "1", "1"],
    }


class MMCIFStructureAnalysisTests(unittest.TestCase):
    def _with_fake_parser(self, columns):
        bio = types.ModuleType("Bio")
        bio.__path__ = []
        pdb = types.ModuleType("Bio.PDB")
        pdb.__path__ = []
        parser = types.ModuleType("Bio.PDB.MMCIF2Dict")

        class FakeMMCIF2Dict(dict):
            def __init__(self, source):
                self.source_text = source.read()
                self.update(columns)

        parser.MMCIF2Dict = FakeMMCIF2Dict
        return patch.dict(sys.modules, {"Bio": bio, "Bio.PDB": pdb, "Bio.PDB.MMCIF2Dict": parser})

    def test_summarizes_only_primary_polymer_atom_rows_with_hash_linked_chains(self):
        payload = b"data_structure\nloop_\n_atom_site.group_PDB\nATOM\n"
        with self._with_fake_parser(mmcif_columns()):
            result = summarize_mmcif(payload, label="AF-P12345-F1")
        self.assertEqual(result["source"], "user-supplied mmCIF coordinate file")
        self.assertEqual(result["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["atom_count"], 2)
        self.assertEqual(result["discarded_alternate_atom_count"], 1)
        self.assertEqual(result["residue_count"], 1)
        self.assertEqual(result["alpha_carbon_count"], 1)
        self.assertEqual(result["atom_temperature_factor"]["median"], 15)
        self.assertFalse(result["coordinates_retained"])

    def test_rejects_misaligned_atom_columns_and_nonfinite_coordinates(self):
        columns = mmcif_columns()
        columns["_atom_site.Cartn_x"] = ["0"]
        with self._with_fake_parser(columns), self.assertRaisesRegex(ValueError, "does not match"):
            summarize_mmcif(b"data_structure")
        columns = mmcif_columns()
        columns["_atom_site.Cartn_x"][0] = "NaN"
        with self._with_fake_parser(columns), self.assertRaisesRegex(ValueError, "non-finite"):
            summarize_mmcif(b"data_structure")

    def test_rejects_forbidden_control_characters_before_invoking_cif_parser(self):
        payload = b"data_structure\n\x00\n"
        with self._with_fake_parser(mmcif_columns()), self.assertRaisesRegex(ValueError, "forbidden control"):
            summarize_mmcif(payload)

    def test_rejects_duplicate_accepted_atom_rows(self):
        columns = mmcif_columns()
        for key, values in columns.items():
            columns[key] = values[:2] + [values[1]]
        columns["_atom_site.label_atom_id"][2] = "CA"
        with self._with_fake_parser(columns), self.assertRaisesRegex(ValueError, "duplicate accepted atom"):
            summarize_mmcif(b"data_structure")

    def test_uses_label_chain_and_residue_ids_when_author_ids_are_missing_per_row(self):
        columns = mmcif_columns()
        columns["_atom_site.auth_asym_id"] = ["?"] * 4
        columns["_atom_site.auth_seq_id"] = ["."] * 4
        columns["_atom_site.label_asym_id"] = ["Z", "Z", "Z", "Z"]
        columns["_atom_site.label_seq_id"] = ["7", "7", "7", "8"]
        with self._with_fake_parser(columns):
            result = summarize_mmcif(b"data_structure")
        self.assertEqual(result["chains"][0]["chain"], "Z")
        self.assertEqual(result["chains"][0]["residue_count"], 1)

    def test_requires_biopython_bio_extra_when_parser_is_unavailable(self):
        with patch.dict(sys.modules, {"Bio": None, "Bio.PDB": None, "Bio.PDB.MMCIF2Dict": None}):
            with self.assertRaisesRegex(RuntimeError, r"medai-science\[bio\]"):
                summarize_mmcif(b"data_structure")

    @unittest.skipUnless(find_spec("Bio") is not None, "Biopython is provided by the optional bio extra")
    def test_parses_a_real_minimal_mmcif_atom_site_loop(self):
        payload = b"""data_medai
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 0.0 0.0 0.0 1.0 10.0 1 ALA A N 1
ATOM 2 C CA . ALA A 1 1 ? 1.0 1.0 1.0 1.0 20.0 1 ALA A CA 1
"""
        result = summarize_mmcif(payload)
        self.assertEqual((result["atom_count"], result["residue_count"], result["alpha_carbon_count"]), (2, 1, 1))
        self.assertEqual(result["atom_temperature_factor"]["median"], 15)


if __name__ == "__main__":
    unittest.main()
