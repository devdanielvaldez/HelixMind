from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from medai_science.sources.crossref import CrossrefClient, build_retention_payload, normalize_work


def _work(doi: str = "10.1234/example.1") -> dict:
    return {
        "DOI": doi,
        "title": ["A biomedical research article"],
        "author": [{"given": "Ada", "family": "Researcher", "ORCID": "https://orcid.org/0000-0000-0000-0000"}],
        "container-title": ["Journal of Examples"],
        "publisher": "Example publisher",
        "type": "journal-article",
        "published-print": {"date-parts": [[2024, 3, 2]]},
        "reference": [{"key": "ref-1", "DOI": "10.9999/cited", "article-title": "Cited work", "year": "2020", "unstructured": "Citation metadata only"}],
        "references-count": 1,
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/", "start": {"date-parts": [[2024, 3, 2]]}}],
    }


class CrossrefTests(unittest.TestCase):
    def test_doi_lookup_normalizes_bibliography_and_citation_edges_without_evidence_polarity(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn("/works/10.1234/example.1", request.url.path)
            self.assertEqual(request.url.params.get("mailto"), "research@example.org")
            return httpx.Response(200, json={"status": "ok", "message": _work()})

        result = asyncio.run(CrossrefClient(mailto="research@example.org", transport=httpx.MockTransport(handler)).get_work("https://doi.org/10.1234/example.1"))
        self.assertEqual(result["doi"], "10.1234/example.1")
        self.assertEqual(result["published_date"], "2024-03-02")
        self.assertEqual(result["references"][0]["DOI"], "10.9999/cited")
        self.assertNotIn("polarity", result["references"][0])
        self.assertIn("does not establish support", result["reference_semantics"])
        self.assertEqual(len(result["metadata_sha256"]), 64)
        self.assertNotIn("abstract", result)

    def test_search_is_bounded_and_uses_polite_pool_contact(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.params.get("query.bibliographic"), "APOE microglia")
            self.assertEqual(request.url.params.get("rows"), "2")
            self.assertEqual(request.url.params.get("mailto"), "research@example.org")
            return httpx.Response(200, json={"message": {"items": [_work(), _work("10.1234/example.2"), _work("10.1234/example.3")]}})

        result = asyncio.run(CrossrefClient(mailto="research@example.org", transport=httpx.MockTransport(handler)).search_bibliographic("APOE microglia", limit=2))
        self.assertEqual(len(result), 2)

    def test_rejects_patient_identifying_search_and_doi_response_mismatch(self):
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            asyncio.run(CrossrefClient(transport=httpx.MockTransport(lambda _: httpx.Response(200))).search_bibliographic("patient id: ABC1234"))

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": _work("10.9999/wrong")})

        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(CrossrefClient(transport=httpx.MockTransport(handler)).get_work("10.1234/example.1"))

    def test_reference_count_is_capped_and_doi_validation_is_strict(self):
        work = _work()
        work["reference"] = [{"DOI": f"10.9999/ref{i}"} for i in range(250)]
        normalized = normalize_work(work)
        self.assertEqual(len(normalized["references"]), 200)
        self.assertTrue(normalized["references_truncated"])
        with self.assertRaises(ValueError):
            CrossrefClient.validate_doi("patient id: 123456")

    def test_artifact_retention_strips_free_text_and_enforces_medai_payload_limit(self):
        records = []
        for index in range(5):
            row = normalize_work(_work(f"10.1234/example.{index}"))
            row["references"] = [{"DOI": "10.9999/cited", "article-title": "title " * 80, "unstructured": "sensitive copyright text " * 100, "reference_index": j} for j in range(150)]
            records.append(row)
        payload = build_retention_payload(records)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertLessEqual(len(encoded), 240_000)
        self.assertNotIn("sensitive copyright text", encoded.decode())
        self.assertEqual(len(payload["complete_result_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
