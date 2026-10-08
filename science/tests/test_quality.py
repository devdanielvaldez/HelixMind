from __future__ import annotations

import math
import unittest

from medai_science.quality import classify_pubmed_design_metadata, is_valid_quality_annotation, summarize_pubmed_design_metadata


class QualityAnnotationValidationTests(unittest.TestCase):
    def test_accepts_finite_numeric_annotations_in_range(self):
        for value in (0, 0.0, 0.42, 1, 1.0):
            with self.subTest(value=value):
                self.assertTrue(is_valid_quality_annotation(value))

    def test_rejects_boolean_non_finite_non_numeric_and_out_of_range_values(self):
        for value in (True, False, math.nan, math.inf, -math.inf, -0.01, 1.01, "0.5", None):
            with self.subTest(value=value):
                self.assertFalse(is_valid_quality_annotation(value))


class PubMedDesignMetadataTests(unittest.TestCase):
    def test_emits_only_exact_provider_publication_type_signals(self):
        result = classify_pubmed_design_metadata(
            ["Journal Article", "Randomized Controlled Trial", "Clinical Trial, Phase III"],
            [{"descriptor": "Cohort Studies", "descriptor_ui": "D000067"}],
        )
        self.assertEqual(result["methodology"], "pubmed-publication-type-exact-v2")
        self.assertEqual(result["design_categories"], ["phase_iii_clinical_trial", "randomized_controlled_trial"])
        self.assertEqual(result["mesh_descriptor_context"], ["Cohort Studies"])
        self.assertIsNone(result["evidence_grade"])
        self.assertTrue(all(value == "not_assessed" for value in result["quality_domains"].values()))

    def test_uses_exact_pubmed_publication_type_identifiers_separately_from_mesh_topic_terms(self):
        result = classify_pubmed_design_metadata(
            ["Clinical Trial, Phase III"], [], ["D017428", "D017326"]
        )
        self.assertEqual(result["design_categories"], ["phase_iii_clinical_trial"])
        id_signals = [item for item in result["publication_type_signals"] if item["source_field"] == "publication_type_ids"]
        self.assertEqual(len(id_signals), 1)
        self.assertEqual(id_signals[0]["source_term"], "D017428")
        self.assertEqual(id_signals[0]["source_url"], "https://meshb.nlm.nih.gov/record/ui?ui=D017428")
        self.assertEqual(len(result["publication_type_signals"]), 2)
        self.assertIsNone(result["evidence_grade"])

    def test_deduplicates_provider_signals_and_leaves_unsupported_types_unclassified(self):
        result = classify_pubmed_design_metadata(
            [" Randomized Controlled Trial ", "randomized controlled trial", "Journal Article"],
            [],
        )
        self.assertEqual(result["design_categories"], ["randomized_controlled_trial"])
        self.assertEqual(len(result["publication_type_signals"]), 1)

    def test_rejects_malformed_or_unbounded_provider_metadata(self):
        with self.assertRaisesRegex(ValueError, "publication types"):
            classify_pubmed_design_metadata("Randomized Controlled Trial", [])
        with self.assertRaisesRegex(ValueError, "MeSH heading records"):
            classify_pubmed_design_metadata([], ["not-an-object"])
        with self.assertRaisesRegex(ValueError, "valid MeSH UI"):
            classify_pubmed_design_metadata([], [], ["D017428/../../"])

    def test_summarizes_overlapping_categories_with_bounded_pmid_examples_and_gaps(self):
        records = [
            {"provenance": {"source": "PubMed", "source_id": str(index)}, "publication_types": ["Randomized Controlled Trial", "Clinical Trial, Phase III"], "publication_type_ids": ["D016449", "D017428"], "mesh_headings": []}
            for index in range(1, 103)
        ]
        records.extend([
            {"provenance": {"source": "PubMed", "source_id": "999"}, "publication_types": ["Journal Article"], "mesh_headings": []},
            {"provenance": {"source": "PubMed", "source_id": "1000"}, "publication_types": "malformed", "mesh_headings": []},
            {"provenance": {"source": "Crossref", "source_id": "doi:1"}, "publication_types": ["Randomized Controlled Trial"], "mesh_headings": []},
        ])
        inventory = summarize_pubmed_design_metadata(records)
        self.assertEqual(inventory["pubmed_records"], 104)
        self.assertEqual(inventory["counts_by_provider_publication_type_signal"]["randomized_controlled_trial"], 102)
        self.assertEqual(inventory["counts_by_provider_publication_type_signal"]["phase_iii_clinical_trial"], 102)
        self.assertEqual(len(inventory["example_pmids_by_signal"]["randomized_controlled_trial"]), 100)
        self.assertTrue(inventory["example_pmids_truncated_by_signal"]["randomized_controlled_trial"])
        self.assertEqual(inventory["records_without_recognized_design_signal"], 1)
        self.assertEqual(inventory["records_with_malformed_design_metadata"], 1)
        self.assertIsNone(inventory["evidence_grade"])

    def test_summarizes_source_linked_corrections_and_retractions_without_grading(self):
        records = [
            {"provenance": {"source": "PubMed", "source_id": "100"}, "publication_types": ["Journal Article"], "mesh_headings": [], "related_publications": [{"ref_type": "RetractionIn", "pmid": "200"}, {"ref_type": "ErratumIn", "pmid": "300"}]},
            {"provenance": {"source": "PubMed", "source_id": "200"}, "publication_types": ["Retraction Notice"], "mesh_headings": [], "related_publications": [{"ref_type": "RetractionOf", "pmid": "100"}]},
        ]
        inventory = summarize_pubmed_design_metadata(records)
        self.assertEqual(inventory["related_publication_links"], 3)
        self.assertEqual(inventory["counts_by_pubmed_related_publication_type"], {"RetractionIn": 1, "ErratumIn": 1, "RetractionOf": 1})
        self.assertEqual(inventory["example_pmids_by_related_publication_type"]["RetractionIn"], ["200"])
        self.assertIsNone(inventory["evidence_grade"])
        self.assertIn("do not independently establish article validity", inventory["interpretation"])

    def test_malformed_related_publication_metadata_routes_record_to_review(self):
        inventory = summarize_pubmed_design_metadata([{"provenance": {"source": "PubMed", "source_id": "100"}, "publication_types": [], "mesh_headings": [], "related_publications": [{"ref_type": "RetractionIn", "pmid": "bad"}]}])
        self.assertEqual(inventory["records_with_malformed_design_metadata"], 0)
        self.assertEqual(inventory["records_with_malformed_related_publication_metadata"], 1)
        self.assertEqual(inventory["related_publication_links"], 0)

    def test_malformed_related_link_does_not_hide_valid_design_metadata(self):
        inventory = summarize_pubmed_design_metadata([{"provenance": {"source": "PubMed", "source_id": "100"}, "publication_types": ["Randomized Controlled Trial"], "mesh_headings": [], "related_publications": [{"ref_type": "RetractionIn", "pmid": "bad"}]}])
        self.assertEqual(inventory["counts_by_provider_publication_type_signal"], {"randomized_controlled_trial": 1})
        self.assertEqual(inventory["records_with_malformed_design_metadata"], 0)
        self.assertEqual(inventory["records_with_malformed_related_publication_metadata"], 1)


if __name__ == "__main__":
    unittest.main()
