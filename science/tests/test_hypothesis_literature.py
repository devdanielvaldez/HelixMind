import unittest

from medai_science.hypothesis_literature import (
    bounded_paper_record,
    build_hypothesis_literature_query,
    build_hypothesis_literature_receipt,
)


class HypothesisLiteratureTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "id": "HYP-test",
            "statement": "APOE may be connected to Alzheimer disease through TREM2.",
        }
        self.paper = {
            "provenance": {"source": "PubMed", "source_id": "12345", "title": "Mechanism study"},
            "abstract": "This abstract must not be persisted in the candidate receipt.",
            "journal": "Journal",
            "publication_date": "2025",
            "publication_types": ["Journal Article"],
            "identifiers": {"doi": "10.1000/example", "pmc": "PMC123"},
        }

    def test_query_binds_all_three_path_entities(self):
        query = build_hypothesis_literature_query(self.candidate)
        self.assertIn('"APOE"[All Fields]', query)
        self.assertIn('"TREM2"[All Fields]', query)
        self.assertIn('"Alzheimer disease"[All Fields]', query)
        with self.assertRaisesRegex(ValueError, "three distinct"):
            build_hypothesis_literature_query({"id": "HYP-2", "statement": "APOE may be connected to APOE through TREM2."})

    def test_source_candidate_receipt_omits_abstract_and_binds_digest(self):
        record = bounded_paper_record(self.paper)
        self.assertFalse(record["abstract_retained"])
        self.assertNotIn("abstract", record)
        self.assertEqual(record["pmid"], "12345")
        search = {"results": [self.paper], "coverage": {"total_hits": 12, "search_truncated": False}, "coverage_artifact_id": "ART-COVERAGE"}
        receipt = build_hypothesis_literature_receipt(self.candidate, build_hypothesis_literature_query(self.candidate), search, [{"id": "ART-PAPER", "record_sha256": record["record_sha256"]}])
        self.assertEqual(receipt["candidate_leads"][0]["artifact_id"], "ART-PAPER")
        self.assertIsNone(receipt["novelty_assessment"])
        self.assertIn("does not establish novelty", receipt["interpretation_limit"])

    def test_rejects_unbound_pmid_and_misaligned_receipts(self):
        query = build_hypothesis_literature_query(self.candidate)
        bad = {**self.paper, "provenance": {"source": "Crossref", "source_id": "12345"}}
        with self.assertRaisesRegex(ValueError, "source provenance"):
            bounded_paper_record(bad)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            build_hypothesis_literature_receipt(self.candidate, query, {"results": [self.paper]}, [])


if __name__ == "__main__":
    unittest.main()
