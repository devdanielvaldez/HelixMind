import asyncio
import unittest

import httpx

from medai_science.sources.reactome import ReactomeClient


class ReactomeTests(unittest.TestCase):
    def test_posts_bounded_gene_list_and_preserves_analysis_provenance(self):
        payload = {"summary": {"token": "R-HSA-1"}, "pathways": [{"stId": "R-HSA-1", "name": "Example pathway"}]}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.params["species"], "Homo sapiens")
            self.assertEqual(request.content, b"APOE\nTREM2")
            self.assertEqual(request.headers["Content-Type"], "text/plain")
            return httpx.Response(200, json=payload)

        result = asyncio.run(ReactomeClient(httpx.MockTransport(handler)).analyze_genes(["APOE", "TREM2", "APOE"]))
        self.assertEqual(result["input_identifiers"], ["APOE", "TREM2"])
        self.assertEqual(len(result["input_sha256"]), 64)
        self.assertEqual(result["analysis_parameters"], {"pageSize": 20, "sortBy": "ENTITIES_PVALUE", "order": "ASC", "species": "Homo sapiens"})
        self.assertEqual(len(result["request_sha256"]), 64)
        self.assertEqual(result["result"], payload)
        self.assertIn("not causal evidence", result["interpretation_limit"])

    def test_rejects_empty_identifiers_unsupported_species_and_patient_identifiers(self):
        with self.assertRaisesRegex(ValueError, "1..5000"):
            ReactomeClient.validate([], "Homo sapiens", 20)
        with self.assertRaisesRegex(ValueError, "supported Reactome species"):
            ReactomeClient.validate(["APOE"], "unknown", 20)
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            ReactomeClient.validate(["patient id: ABC1234"], "Homo sapiens", 20)


if __name__ == "__main__":
    unittest.main()
