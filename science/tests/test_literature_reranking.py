from __future__ import annotations

import unittest

from medai_science.llm import ModelCompletion
from medai_science.literature_reranking import RERANK_SCHEMA, rerank_pubmed_titles


PAPERS = [
    {"provenance": {"source": "PubMed", "source_id": "101", "title": "Microglial APOE signaling in Alzheimer disease"}, "abstract": "This abstract must never be sent to the reranker."},
    {"provenance": {"source": "PubMed", "source_id": "202", "title": "Unrelated cardiac imaging methods"}, "abstract": "This abstract also must never be sent."},
]


class FakeProvider:
    async def complete_json(self, **kwargs):
        self.kwargs = kwargs
        value = {"ranked_pmids": ["101", "202"], "relevance": [{"pmid": "101", "tier": "higher"}, {"pmid": "202", "tier": "lower"}]}
        return ModelCompletion(value, "test", "fixture", "a" * 64, "b" * 64, {"total_tokens": 6})


class LiteratureRerankingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reorders_source_records_and_sends_titles_only(self):
        provider = FakeProvider()
        papers, completion = await rerank_pubmed_titles(provider, "Alzheimer microglia", PAPERS)
        self.assertEqual([row["provenance"]["source_id"] for row in papers], ["101", "202"])
        self.assertIn("Microglial APOE", provider.kwargs["prompt"])
        self.assertNotIn("abstract", provider.kwargs["prompt"])
        self.assertNotIn("This abstract", provider.kwargs["prompt"])
        self.assertEqual(provider.kwargs["schema"], RERANK_SCHEMA)
        self.assertEqual(completion.value["relevance"][0]["tier"], "higher")

    async def test_rejects_missing_duplicate_or_invented_pmids(self):
        provider = FakeProvider()
        provider.complete_json = async_bad_ids
        with self.assertRaisesRegex(ValueError, "every supplied PMID"):
            await rerank_pubmed_titles(provider, "Alzheimer", PAPERS)

    async def test_rejects_patient_identifiers_and_duplicate_source_records(self):
        provider = FakeProvider()
        with self.assertRaises(ValueError):
            await rerank_pubmed_titles(provider, "patient john@example.com", PAPERS)
        duplicates = [PAPERS[0], PAPERS[0]]
        with self.assertRaisesRegex(ValueError, "unique PMIDs"):
            await rerank_pubmed_titles(provider, "APOE", duplicates)


async def async_bad_ids(**kwargs):
    return ModelCompletion({"ranked_pmids": ["101", "999"], "relevance": [{"pmid": "101", "tier": "higher"}, {"pmid": "999", "tier": "lower"}]}, "test", "fixture", "a" * 64, "b" * 64, {"total_tokens": 6})


if __name__ == "__main__":
    unittest.main()
