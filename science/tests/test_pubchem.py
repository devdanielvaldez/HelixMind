from __future__ import annotations

import asyncio
import time
import unittest

import httpx

from medai_science.sources.pubchem import PubChemClient


def _properties(cid: int, title: str) -> dict:
    return {"CID": cid, "Title": title, "MolecularFormula": "C9H8O4", "MolecularWeight": 180.16, "SMILES": "CC(=O)Oc1ccccc1C(=O)O", "InChI": "InChI=1S/example", "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "XLogP": 1.2, "ExactMass": 180.042}


class PubChemTests(unittest.TestCase):
    def test_process_wide_pacing_serializes_concurrent_client_requests(self):
        request_times = []

        def handler(_: httpx.Request) -> httpx.Response:
            request_times.append(time.monotonic())
            return httpx.Response(200, json={"record": True})

        async def run_concurrently():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                await asyncio.gather(
                    PubChemClient()._json(client, "https://example.test/first"),
                    PubChemClient()._json(client, "https://example.test/second"),
                )

        with PubChemClient._request_schedule_lock:
            PubChemClient._next_request_at = 0.0
        asyncio.run(run_concurrently())
        self.assertEqual(len(request_times), 2)
        self.assertGreaterEqual(request_times[1] - request_times[0], 0.19)

    def test_resolves_name_and_returns_sourced_identity_without_claiming_activity(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/cids/JSON"):
                return httpx.Response(200, json={"IdentifierList": {"CID": [2244]}})
            self.assertIn("/compound/cid/2244/property/", request.url.path)
            return httpx.Response(200, json={"PropertyTable": {"Properties": [_properties(2244, "Aspirin")]}})

        result = asyncio.run(PubChemClient(httpx.MockTransport(handler)).lookup("aspirin"))
        self.assertFalse(result["ambiguous"])
        self.assertEqual(result["matches"][0]["cid"], 2244)
        self.assertEqual(len(result["matches"][0]["record_sha256"]), 64)
        self.assertIn("no target", result["interpretation"])
        self.assertIn("rdkit", result["matches"][0])

    def test_reports_ambiguous_names_without_silently_selecting_one(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/cids/JSON"):
                return httpx.Response(200, json={"IdentifierList": {"CID": [1, 2]}})
            return httpx.Response(200, json={"PropertyTable": {"Properties": [_properties(1, "Compound one"), _properties(2, "Compound two")]}})

        result = asyncio.run(PubChemClient(httpx.MockTransport(handler)).lookup("ambiguous name"))
        self.assertTrue(result["ambiguous"])
        self.assertEqual([row["cid"] for row in result["matches"]], [1, 2])

    def test_rejects_identifiers_that_look_like_patient_data_and_mismatched_provider_records(self):
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            PubChemClient.validate_identifier("patient id: ABC1234")

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"PropertyTable": {"Properties": [_properties(7, "Other")]}})

        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(PubChemClient(httpx.MockTransport(handler)).lookup("2244"))

    def test_bounds_match_count_and_response_body(self):
        def many_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"IdentifierList": {"CID": list(range(1, 12))}})

        result = asyncio.run(PubChemClient(httpx.MockTransport(many_handler)).lookup("many matches"))
        self.assertTrue(result["too_many_matches"])

        def large_handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"{" + b" " * (512 * 1024 + 1))

        with self.assertRaisesRegex(ValueError, "PubChem response exceeds the 524288-byte limit"):
            asyncio.run(PubChemClient(httpx.MockTransport(large_handler)).lookup("2244"))


if __name__ == "__main__":
    unittest.main()
