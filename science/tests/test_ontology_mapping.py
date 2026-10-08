import json
import unittest

import httpx

from medai_science.sources.ontology_mapping import OntologyMappingClient


class OntologyMappingClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_direct_oxo_cross_references_only_as_review_candidates(self):
        async def handler(request):
            self.assertEqual(request.url.host, "www.ebi.ac.uk")
            self.assertEqual(request.url.path, "/spot/oxo/api/search")
            self.assertEqual(request.url.params["ids"], "EFO_0000249")
            self.assertEqual(request.url.params["distance"], "1")
            payload = {
                "_embedded": {"searchResults": [{
                    "curie": "EFO_0000249",
                    "querySource": {"label": "Alzheimer disease"},
                    "mappingResponseList": [
                        {"curie": "MONDO:0004979", "label": "Alzheimer disease", "targetPrefix": "MONDO", "distance": 1, "sourceType": "OLS cross-reference"},
                        {"curie": "MONDO:0004980", "label": "Related disease", "targetPrefix": "MONDO", "distance": 2, "sourceType": "UMLS"},
                        {"curie": "DOID:10652", "label": "Alzheimer disease", "targetPrefix": "DOID", "distance": 1},
                    ],
                }]},
            }
            return httpx.Response(200, content=json.dumps(payload).encode())

        client = OntologyMappingClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.cross_map("EFO_0000249", ["MONDO"], distance=1, limit=10)

        self.assertEqual(result["source_identifier"], "EFO_0000249")
        self.assertEqual(result["source_label"], "Alzheimer disease")
        self.assertEqual(result["mappings"], [{
            "target_curie": "MONDO:0004979", "target_prefix": "MONDO", "target_label": "Alzheimer disease",
            "distance": 1, "source_type": "OLS cross-reference",
        }])
        self.assertEqual(len(result["source_response_sha256"]), 64)
        self.assertIn("requires human review", result["mapping_interpretation"])

    async def test_rejects_unsupported_ids_transitive_mappings_and_invalid_targets(self):
        with self.assertRaisesRegex(ValueError, "exact EFO or MONDO"):
            await OntologyMappingClient().cross_map("Alzheimer disease")
        with self.assertRaisesRegex(ValueError, "distance=1"):
            await OntologyMappingClient().cross_map("MONDO:0004979", distance=2)
        with self.assertRaisesRegex(ValueError, "prefixes"):
            await OntologyMappingClient().cross_map("MONDO:0004979", ["mondo"])

    async def test_rejects_unexpected_oxo_payload_shape(self):
        client = OntologyMappingClient(lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"unexpected": []})), **kwargs
        ))
        with self.assertRaisesRegex(ValueError, "response shape"):
            await client.cross_map("MONDO:0004979")

    async def test_rejects_wrong_source_identity_and_malformed_candidate_prefix(self):
        for result, message in [
            ({"curie": "EFO_0000555", "mappingResponseList": []}, "does not match"),
            ({"curie": "EFO_0000249", "mappingResponseList": [{"curie": "EFO_0001", "targetPrefix": "MONDO", "distance": 1}]}, "does not agree"),
        ]:
            client = OntologyMappingClient(lambda **kwargs: httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _, result=result: httpx.Response(200, json={"_embedded": {"searchResults": [result]}})), **kwargs
            ))
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                await client.cross_map("EFO_0000249", ["MONDO"])

    async def test_reports_truncated_provider_rows_and_deduplicates_exact_candidates(self):
        candidate = {"curie": "MONDO:0004979", "label": "Alzheimer disease", "targetPrefix": "MONDO", "distance": 1, "sourceType": "OLS cross-reference"}
        payload = {"_embedded": {"searchResults": [{"curie": "EFO_0000249", "mappingResponseList": [candidate, dict(candidate), {**candidate, "curie": "MONDO:0004980"}]}]}}
        client = OntologyMappingClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)), **kwargs))
        result = await client.cross_map("EFO_0000249", ["MONDO"], limit=2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["provider_mapping_count"], 3)
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["duplicate_mapping_rows"], 1)

    async def test_rejects_nonstandard_json_constants(self):
        client = OntologyMappingClient(lambda **kwargs: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b'{"_embedded":{"searchResults":NaN}}')), **kwargs
        ))
        with self.assertRaisesRegex(ValueError, "non-standard JSON"):
            await client.cross_map("EFO_0000249")


if __name__ == "__main__":
    unittest.main()
