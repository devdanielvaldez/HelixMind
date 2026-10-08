from __future__ import annotations

import asyncio
import unittest

import httpx

from medai_science.sources.chembl import ChEMBLClient

KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


class ChEMBLTests(unittest.TestCase):
    def test_exact_structure_join_retains_mechanism_indication_and_warning_receipts(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path.endswith("/molecule.json"):
                self.assertEqual(request.url.params["molecule_structures__standard_inchi_key__exact"], KEY)
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL25"}, "pref_name": "ASPIRIN"}]})
            key = request.url.path.rsplit("/", 1)[-1]
            if key == "mechanism.json":
                return httpx.Response(200, json={"mechanisms": [{"molecule_chembl_id": "CHEMBL25", "mechanism_of_action": "example", "target_chembl_id": "CHEMBL210"}], "page_meta": {"total_count": 51}})
            if key == "drug_indication.json":
                return httpx.Response(200, json={"drug_indications": [{"molecule_chembl_id": "CHEMBL25", "mesh_heading": "example", "efo_id": "EFO_0000001"}]})
            return httpx.Response(200, json={"drug_warnings": [{"molecule_chembl_id": "CHEMBL25", "warning_type": "example"}]})

        result = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))
        self.assertEqual(result["parent_chembl_ids"], ["CHEMBL25"])
        self.assertEqual(len(result["mechanisms"]), 1)
        self.assertEqual(len(result["indications"]), 1)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(len(result["mechanisms"][0]["record_sha256"]), 64)
        self.assertTrue(result["record_limits"]["mechanisms"]["truncated"])
        self.assertEqual(len(calls), 4)
        self.assertTrue(result["research_only"])

    def test_rejects_malformed_key_and_provider_mismatch(self):
        with self.assertRaisesRegex(ValueError, "InChIKey"):
            asyncio.run(ChEMBLClient().investigate("not-a-key"))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": "AAAAAAAAAAAAAA-BBBBBBBBBB-C"}}]})
            return httpx.Response(200, json={})

        result = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))
        self.assertEqual(result["molecule_matches"], [])
        self.assertEqual(result["mechanisms"], [])

    def test_rejects_mechanism_rows_for_unmatched_parent(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL25"}}]})
            if request.url.path.endswith("/mechanism.json"):
                return httpx.Response(200, json={"mechanisms": [{"molecule_chembl_id": "CHEMBL999", "mechanism_of_action": "wrong identity"}]})
            return httpx.Response(200, json={})

        with self.assertRaisesRegex(ValueError, "did not match"):
            asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))

    def test_shared_target_screen_intersects_only_exact_identity_mechanism_records(self):
        key_b = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                key = request.url.params["molecule_structures__standard_inchi_key__exact"]
                molecule_id = "CHEMBL25" if key == KEY else "CHEMBL26"
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": molecule_id, "molecule_structures": {"standard_inchi_key": key}, "molecule_hierarchy": {"parent_chembl_id": molecule_id}}]})
            if request.url.path.endswith("/mechanism.json"):
                molecule_id = request.url.params["molecule_chembl_id__in"]
                target = "CHEMBL210" if molecule_id == "CHEMBL25" else "CHEMBL210"
                unique = "CHEMBL211" if molecule_id == "CHEMBL25" else "CHEMBL212"
                return httpx.Response(200, json={"mechanisms": [{"molecule_chembl_id": molecule_id, "target_chembl_id": target, "mechanism_of_action": "provider reported"}, {"molecule_chembl_id": molecule_id, "target_chembl_id": unique, "mechanism_of_action": "provider reported"}]})
            return httpx.Response(200, json={"drug_indications": [], "drug_warnings": []})

        result = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).compare_shared_targets(KEY, key_b))
        self.assertEqual([row["target_chembl_id"] for row in result["shared_targets"]], ["CHEMBL210"])
        self.assertEqual(len(result["shared_targets"][0]["compound_a_mechanism_records"][0]["record_sha256"]), 64)
        self.assertTrue(any("not evidence of a drug-drug interaction" in text for text in result["limitations"]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            asyncio.run(ChEMBLClient().compare_shared_targets(KEY, KEY))


if __name__ == "__main__":
    unittest.main()
