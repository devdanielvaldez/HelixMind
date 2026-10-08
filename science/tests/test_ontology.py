import unittest

import httpx

from medai_science.sources.ontology import OntologyClient


class OntologyClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_retains_bounded_terms_and_provider_rank_is_not_confidence(self):
        async def handler(request):
            self.assertEqual(request.url.host, "www.ebi.ac.uk")
            self.assertEqual(request.url.path, "/ols4/api/search")
            self.assertEqual(request.url.params["ontology"], "mondo")
            self.assertEqual(request.url.params["rows"], "10")
            return httpx.Response(200, json={"response": {"numFound": 1, "docs": [{
                "iri": "http://purl.obolibrary.org/obo/MONDO_0004979",
                "short_form": "MONDO_0004979",
                "obo_id": "MONDO:0004979",
                "label": "Alzheimer disease",
                "synonym": ["Alzheimer's disease", "A" * 600],
                "description": ["A bounded description."],
                "ontology_name": "Mondo Disease Ontology",
                "ontology_prefix": "MONDO",
                "is_obsolete": False,
                "abstract": "must not be retained",
            }]}})

        client = OntologyClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.search("Alzheimer disease", "MONDO", 10)
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["records"][0]["label"], "Alzheimer disease")
        self.assertEqual(len(result["records"][0]["synonyms"][1]), 500)
        self.assertNotIn("abstract", result["records"][0])
        self.assertEqual(len(result["records"][0]["source_record_sha256"]), 64)
        self.assertIn("not a confidence score", result["search_semantics"])

    async def test_identifier_resolution_requires_exact_identifier_field(self):
        async def handler(request):
            return httpx.Response(200, json={"response": {"numFound": 2, "docs": [
                {"iri": "http://purl.obolibrary.org/obo/MONDO_0004979", "short_form": "MONDO_0004979", "label": "Alzheimer disease"},
                {"iri": "http://purl.obolibrary.org/obo/MONDO_0004980", "short_form": "MONDO_0004980", "label": "Related term"},
            ]}})

        client = OntologyClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.resolve_identifier("MONDO_0004979", "mondo")
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["records"][0]["short_form"], "MONDO_0004979")

    async def test_search_rejects_patient_identifier_and_unbounded_limit(self):
        client = OntologyClient()
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            await client.search("patient MRN 123456789")
        with self.assertRaisesRegex(ValueError, "1..50"):
            await client.search("Alzheimer", limit=51)


if __name__ == "__main__":
    unittest.main()
