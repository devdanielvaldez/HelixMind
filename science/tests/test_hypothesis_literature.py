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
        self.coverage = {
            "total_hits": 12,
            "requested_limit": 20,
            "returned_pmids": 1,
            "search_truncated": True,
            "fetched_records": 1,
            "missing_record_pmids": [],
            "fetch_incomplete": False,
            "retrieval_incomplete": True,
        }

    def test_query_binds_all_three_path_entities(self):
        query = build_hypothesis_literature_query(self.candidate)
        self.assertIn('"APOE"[All Fields]', query)
        self.assertIn('"TREM2"[All Fields]', query)
        self.assertIn('"Alzheimer disease"[All Fields]', query)
        with self.assertRaisesRegex(ValueError, "distinct path entity labels"):
            build_hypothesis_literature_query({"id": "HYP-2", "statement": "APOE may be connected to APOE through TREM2."})

    def test_query_binds_every_entity_in_a_multi_hop_path(self):
        candidate = {
            "id": "HYP-path-4",
            "statement": "APOE may be connected to Alzheimer disease through TREM2 → lipid metabolism.",
            "entity_labels": ["APOE", "TREM2", "lipid metabolism", "Alzheimer disease"],
        }
        query = build_hypothesis_literature_query(candidate)
        self.assertEqual(query.count("[All Fields]"), 4)
        self.assertIn('"lipid metabolism"[All Fields]', query)
        candidate["entity_labels"] = ["APOE", "APP", "lipid metabolism", "Alzheimer disease"]
        with self.assertRaisesRegex(ValueError, "does not match its ordered graph-path entities"):
            build_hypothesis_literature_query(candidate)

    def test_source_candidate_receipt_omits_abstract_and_binds_digest(self):
        record = bounded_paper_record(self.paper)
        self.assertFalse(record["abstract_retained"])
        self.assertNotIn("abstract", record)
        self.assertEqual(record["pmid"], "12345")
        search = {"results": [self.paper], "coverage": self.coverage, "coverage_artifact_id": "ART-COVERAGE"}
        receipt = build_hypothesis_literature_receipt(self.candidate, build_hypothesis_literature_query(self.candidate), search, [{"id": "ART-PAPER", "record_sha256": record["record_sha256"]}])
        self.assertEqual(receipt["candidate_leads"][0]["artifact_id"], "ART-PAPER")
        self.assertEqual(receipt["retrieval_coverage"]["status"], "retrieval_incomplete")
        self.assertFalse(receipt["retrieval_coverage"]["novelty_assessable"])
        self.assertIsNone(receipt["novelty_assessment"])
        self.assertIn("does not establish novelty", receipt["interpretation_limit"])

    def test_rejects_unbound_pmid_and_misaligned_receipts(self):
        query = build_hypothesis_literature_query(self.candidate)
        bad = {**self.paper, "provenance": {"source": "Crossref", "source_id": "12345"}}
        with self.assertRaisesRegex(ValueError, "source provenance"):
            bounded_paper_record(bad)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            build_hypothesis_literature_receipt(self.candidate, query, {"results": [self.paper]}, [])

    def test_receipt_rejects_query_for_another_path_and_paper_digest_mismatch(self):
        query = build_hypothesis_literature_query(self.candidate)
        search = {"results": [self.paper], "coverage": self.coverage, "coverage_artifact_id": "ART-COVERAGE"}
        artifact = {"id": "ART-PAPER", "record_sha256": bounded_paper_record(self.paper)["record_sha256"]}
        with self.assertRaisesRegex(ValueError, "does not match the exact hypothesis"):
            build_hypothesis_literature_receipt(self.candidate, query.replace('"APOE"', '"APP"'), search, [artifact])
        artifact["record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match its stored"):
            build_hypothesis_literature_receipt(self.candidate, query, search, [artifact])

    def test_receipt_rejects_artifact_without_stable_id(self):
        query = build_hypothesis_literature_query(self.candidate)
        with self.assertRaisesRegex(ValueError, "exact PMID or stored artifact"):
            build_hypothesis_literature_receipt(
                self.candidate,
                query,
                {"results": [self.paper], "coverage": self.coverage, "coverage_artifact_id": "ART-COVERAGE"},
                [{"id": "", "record_sha256": bounded_paper_record(self.paper)["record_sha256"]}],
            )

    def test_retrieval_coverage_rejects_contradictions_and_preserves_bounded_completeness(self):
        complete = {**self.coverage, "total_hits": 1, "search_truncated": False, "retrieval_incomplete": False}
        receipt = build_hypothesis_literature_receipt(
            self.candidate,
            build_hypothesis_literature_query(self.candidate),
            {"results": [self.paper], "coverage": complete, "coverage_artifact_id": "ART-COVERAGE"},
            [{"id": "ART-PAPER", "record_sha256": bounded_paper_record(self.paper)["record_sha256"]}],
        )
        self.assertEqual(receipt["retrieval_coverage"]["status"], "retrieved_page_complete")
        self.assertFalse(receipt["retrieval_coverage"]["source_complete"])
        self.assertFalse(receipt["retrieval_coverage"]["query_strategy_complete"])

        malformed = {**complete, "retrieval_incomplete": True}
        with self.assertRaisesRegex(ValueError, "flags conflict"):
            build_hypothesis_literature_receipt(
                self.candidate,
                build_hypothesis_literature_query(self.candidate),
                {"results": [self.paper], "coverage": malformed, "coverage_artifact_id": "ART-COVERAGE"},
                [{"id": "ART-PAPER", "record_sha256": bounded_paper_record(self.paper)["record_sha256"]}],
            )

        overlapping = {
            **complete,
            "total_hits": 2,
            "returned_pmids": 2,
            "fetched_records": 1,
            "missing_record_pmids": ["12345"],
            "fetch_incomplete": True,
            "retrieval_incomplete": True,
        }
        with self.assertRaisesRegex(ValueError, "missing-record identifiers"):
            build_hypothesis_literature_receipt(
                self.candidate,
                build_hypothesis_literature_query(self.candidate),
                {"results": [self.paper], "coverage": overlapping, "coverage_artifact_id": "ART-COVERAGE"},
                [{"id": "ART-PAPER", "record_sha256": bounded_paper_record(self.paper)["record_sha256"]}],
            )


if __name__ == "__main__":
    unittest.main()
