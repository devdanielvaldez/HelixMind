from __future__ import annotations

import asyncio
import gzip
import unittest

import httpx

from medai_science.sources.sifts import SIFTSClient


def _sifts_xml() -> bytes:
    return b'''<?xml version="1.0"?>
<entry xmlns="http://www.ebi.ac.uk/pdbe/docs/sifts/eFamily.xsd" dbSource="PDBe">
  <entity type="protein" entityId="1">
    <segment segId="1" start="10" end="11">
      <listResidue>
        <residue dbResNum="10" dbResMon="GLY">
          <crossRefDb dbSource="PDB" dbAccessionId="1abc" dbResNum="10" dbResName="GLY" dbChainId="A" dbAuthorResNum="12" dbAuthorChainId="X" dbCoordSys="PDBresnum"/>
          <crossRefDb dbSource="UniProt" dbAccessionId="P05067" dbResNum="42" dbResName="GLY" dbCoordSys="UniProt"/>
          <residueDetail dbSource="PDBe" property="annotation">Observed</residueDetail>
        </residue>
        <residue dbResNum="11" dbResMon="ALA">
          <crossRefDb dbSource="PDB" dbAccessionId="1abc" dbResNum="11" dbResName="ALA" dbChainId="A" dbAuthorResNum="13A" dbAuthorChainId="X" dbCoordSys="PDBresnum"/>
          <crossRefDb dbSource="UniProt" dbAccessionId="P05067" dbResNum="43" dbResName="ALA" dbCoordSys="UniProt"/>
        </residue>
        <residue dbResNum="12" dbResMon="SER">
          <crossRefDb dbSource="PDB" dbAccessionId="1abc" dbResNum="12" dbResName="SER" dbChainId="A"/>
          <crossRefDb dbSource="UniProt" dbAccessionId="Q9Y6K9" dbResNum="20" dbResName="SER"/>
        </residue>
      </listResidue>
    </segment>
  </entity>
</entry>'''


def _observed_mmcif() -> bytes:
    return b'''data_1abc
loop_
_pdbx_sifts_xref_db.entity_id
_pdbx_sifts_xref_db.asym_id
_pdbx_sifts_xref_db.seq_id
_pdbx_sifts_xref_db.unp_acc
_pdbx_sifts_xref_db.unp_num
_pdbx_sifts_xref_db.observed
1 A 10 P05067 42 Y
1 A 11 P05067 43 N
1 A 12 Q9Y6K9 20 Y
#
'''


class SIFTSClientTests(unittest.TestCase):
    def test_maps_each_uniprot_residue_and_preserves_auth_numbering_and_details(self) -> None:
        packed = gzip.compress(_sifts_xml())

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "ftp.ebi.ac.uk":
                self.assertEqual(request.url.path, "/pub/databases/msd/sifts/xml/1abc.xml.gz")
                return httpx.Response(200, content=packed)
            self.assertEqual(request.url.host, "www.ebi.ac.uk")
            self.assertEqual(request.url.path, "/pdbe/static/entry/1abc_updated.cif")
            return httpx.Response(200, content=_observed_mmcif())

        result = asyncio.run(SIFTSClient(transport=httpx.MockTransport(handler)).map_residues("P05067", ["1ABC"]))
        self.assertEqual(result["coverage"]["mapped_uniprot_positions"], 2)
        self.assertEqual(len(result["residue_mappings"]), 2)
        first, second = result["residue_mappings"]
        self.assertEqual((first["uniprot_residue_number"], first["pdb_residue_number"]), (42, "10"))
        self.assertEqual((first["chain_id"], first["pdb_author_chain_id"], first["pdb_author_residue_number"]), ("A", "X", "12"))
        self.assertEqual(second["pdb_author_residue_number"], "13A")
        self.assertEqual(first["coordinate_observation"]["status"], "observed")
        self.assertTrue(first["coordinate_observation"]["has_atomic_coordinates"])
        self.assertEqual(second["coordinate_observation"]["status"], "not_observed")
        self.assertFalse(second["coordinate_observation"]["has_atomic_coordinates"])
        self.assertEqual(result["coverage"]["residues_with_coordinates"], 1)
        self.assertEqual(result["coverage"]["residues_without_coordinates"], 1)
        self.assertEqual(result["coverage"]["residues_with_unknown_coordinate_status"], 0)
        self.assertEqual(len(result["entries"][0]["xml_sha256"]), 64)
        self.assertEqual(len(result["entries"][0]["coordinate_metadata"]["mmcif_sha256"]), 64)

    def test_missing_or_conflicting_coordinate_observation_stays_unknown(self) -> None:
        packed = gzip.compress(_sifts_xml())

        def unavailable(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=packed) if request.url.path.endswith(".xml.gz") else httpx.Response(404)

        result = asyncio.run(SIFTSClient(transport=httpx.MockTransport(unavailable)).map_residues("P05067", ["1ABC"]))
        self.assertTrue(all(row["coordinate_observation"]["status"] == "unknown" for row in result["residue_mappings"]))
        self.assertEqual(result["coverage"]["residues_with_unknown_coordinate_status"], 2)

        ambiguous_data = _observed_mmcif().replace(b"1 A 10 P05067 42 Y", b"1 A 10 P05067 42 Y\n1 A 10 P05067 42 N")

        def conflicting(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=packed) if request.url.path.endswith(".xml.gz") else httpx.Response(200, content=ambiguous_data)

        result = asyncio.run(SIFTSClient(transport=httpx.MockTransport(conflicting)).map_residues("P05067", ["1ABC"]))
        self.assertEqual(result["residue_mappings"][0]["coordinate_observation"]["status"], "ambiguous")

    def test_missing_files_and_bounded_inputs(self) -> None:
        missing = SIFTSClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
        result = asyncio.run(missing.map_residues("P05067", ["1ABC"]))
        self.assertEqual(result["entries"], [{"pdb_id": "1abc", "status": "not_found"}])
        with self.assertRaisesRegex(ValueError, "PDB identifiers"):
            asyncio.run(missing.map_residues("P05067", ["../../evil"]))
        with self.assertRaisesRegex(ValueError, "max_structures"):
            asyncio.run(missing.map_residues("P05067", [], 21))

    def test_rejects_bad_gzip_and_malformed_xml(self) -> None:
        bad_gzip = SIFTSClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"not-gzip")))
        with self.assertRaisesRegex(ValueError, "invalid gzip"):
            asyncio.run(bad_gzip.map_residues("P05067", ["1ABC"]))
        malformed = SIFTSClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=gzip.compress(b"<entry>"))))
        with self.assertRaisesRegex(ValueError, "malformed XML"):
            asyncio.run(malformed.map_residues("P05067", ["1ABC"]))


if __name__ == "__main__":
    unittest.main()
