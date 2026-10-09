from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from medai_science.sources.chembl import ChEMBLClient
from medai_science.reproducibility import create_chembl_bundle, verify_analysis_bundle

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

    def test_profile_bundle_replays_exact_retained_records_without_network(self):
        def handler(request: httpx.Request) -> httpx.Response:
            resource = request.url.path.rsplit("/", 1)[-1]
            if resource == "molecule.json":
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL25"}}], "page_meta": {"total_count": 1}})
            if resource == "mechanism.json":
                return httpx.Response(200, json={"mechanisms": [{"molecule_chembl_id": "CHEMBL25", "target_chembl_id": "CHEMBL210", "mechanism_of_action": "source reported"}], "page_meta": {"total_count": 1}})
            if resource == "drug_indication.json":
                return httpx.Response(200, json={"drug_indications": [{"molecule_chembl_id": "CHEMBL25", "mesh_heading": "source reported"}], "page_meta": {"total_count": 1}})
            return httpx.Response(200, json={"drug_warnings": [], "page_meta": {"total_count": 0}})

        profile = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))
        with tempfile.TemporaryDirectory(prefix="medai-chembl-bundle-") as temp:
            bundle = Path(temp) / "bundle"
            created = create_chembl_bundle(bundle, profile)
            self.assertEqual(created["artifact_count"], 2)
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual((bundle / "drug.json").stat().st_mode & 0o777, 0o600)
            self.assertTrue(verify_analysis_bundle(bundle)["valid"])
            with patch.object(ChEMBLClient, "investigate", side_effect=AssertionError("replay attempted network access")):
                replayed = verify_analysis_bundle(bundle, replay=True)
            self.assertTrue(replayed["replayed"])
            self.assertEqual(replayed["report_sha256"], created["report_sha256"])

    def test_bundle_creation_rejects_changed_source_record_digest(self):
        def handler(request: httpx.Request) -> httpx.Response:
            resource = request.url.path.rsplit("/", 1)[-1]
            if resource == "molecule.json":
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL25"}}]})
            return httpx.Response(200, json={"mechanisms": [], "drug_indications": [], "drug_warnings": []})

        profile = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))
        profile["mechanisms"].append({"source": "ChEMBL mechanism", "record": {"molecule_chembl_id": "CHEMBL25"}, "record_sha256": "0" * 64, "retrieved_at": profile["molecule_matches"][0]["retrieved_at"]})
        profile["record_limits"]["mechanisms"].update({"returned": 1, "provider_total": 1})
        with tempfile.TemporaryDirectory(prefix="medai-chembl-bundle-") as temp:
            with self.assertRaisesRegex(ValueError, "source record digest"):
                create_chembl_bundle(Path(temp) / "bundle", profile)

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

    def test_rejects_ambiguous_duplicate_key_and_nonstandard_json_numbers(self):
        for body in (b'{"molecules":[],"molecules":[]}', b'{"molecules":[{"x":NaN}]}'):
            with self.subTest(body=body):
                async def handler(_: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, content=body)

                with self.assertRaisesRegex(ValueError, "invalid, ambiguous, or non-standard JSON"):
                    asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))

    def test_rejects_mechanism_rows_for_unmatched_parent(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": {"parent_chembl_id": "CHEMBL25"}}]})
            if request.url.path.endswith("/mechanism.json"):
                return httpx.Response(200, json={"mechanisms": [{"molecule_chembl_id": "CHEMBL999", "mechanism_of_action": "wrong identity"}]})
            return httpx.Response(200, json={})

        with self.assertRaisesRegex(ValueError, "did not match"):
            asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))

    def test_rejects_malformed_molecule_identity_metadata_without_attribute_errors(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": []}}]})
            return httpx.Response(200, json={})

        result = asyncio.run(ChEMBLClient(httpx.MockTransport(handler)).investigate(KEY))
        self.assertEqual(result["molecule_matches"], [])

        def malformed_hierarchy(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/molecule.json"):
                return httpx.Response(200, json={"molecules": [{"molecule_chembl_id": "CHEMBL25", "molecule_structures": {"standard_inchi_key": KEY}, "molecule_hierarchy": "invalid"}]})
            return httpx.Response(200, json={})

        with self.assertRaisesRegex(ValueError, "hierarchy metadata"):
            asyncio.run(ChEMBLClient(httpx.MockTransport(malformed_hierarchy)).investigate(KEY))

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
