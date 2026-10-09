from __future__ import annotations

import json
import unittest

from medai_science.drug_interaction_fulltext import build_interaction_fulltext_screen, rehydrate_interaction_fulltext_candidates, validate_interaction_fulltext_binding


class DrugInteractionFullTextScreenTests(unittest.TestCase):
    def setUp(self):
        self.article = {
            "source": "PMC BioC",
            "document_id": "PMC12345",
            "title": "A pharmacokinetic study",
            "source_sha256": "a" * 64,
            "source_license": "CC BY 4.0",
            "parser_version": "0.2.0",
            "sections": [
                {"section": "methods", "source_section": "METHODS", "passage_index": 2, "sentence_index": 0, "offset": 100, "text": "Metformin and warfarin were coadministered in a randomized crossover study."},
                {"section": "results", "source_section": "RESULTS", "passage_index": 5, "sentence_index": 1, "offset": 240, "text": "The warfarin group had a measured AUC endpoint after treatment."},
                {"section": "introduction", "source_section": "INTRO", "passage_index": 1, "offset": 10, "text": "Metformin and warfarin have both been prescribed for decades."},
            ],
        }
        self.arguments = {
            "pmcid": "PMC12345",
            "pmid": "98765432",
            "candidate_artifact_id": "artifact-paper",
            "candidate_artifact_sha256": "b" * 64,
            "ingredient_names_a": ["metformin", "metformin hydrochloride"],
            "ingredient_names_b": ["warfarin"],
        }

    def test_returns_exact_name_fulltext_leads_with_offsets_and_hashes_only(self):
        result = build_interaction_fulltext_screen(self.article, **self.arguments)
        self.assertEqual(result["status"], "review_leads_available")
        self.assertEqual(result["candidate_count"], 2)
        self.assertTrue(result["candidates"][0]["both_ingredients_in_candidate"])
        self.assertEqual(result["candidates"][0]["source_offset"], 100)
        self.assertEqual(len(result["candidates"][0]["text_sha256"]), 64)
        self.assertNotIn("Metformin and warfarin were coadministered", json.dumps(result))
        self.assertIsNone(result["interaction_verdict"])
        self.assertTrue(result["review_required"])
        self.assertEqual(result["source_license"], "CC BY 4.0")

    def test_unanchored_matches_and_nonreview_sections_are_not_emitted_as_candidates(self):
        article = {**self.article, "sections": [
            {"section": "methods", "offset": None, "text": "Metformin and warfarin were dosed in a study."},
            {"section": "introduction", "offset": 10, "text": "Metformin and warfarin were dosed in a study."},
        ]}
        result = build_interaction_fulltext_screen(article, **self.arguments)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["unanchored_lexical_match_count"], 1)
        self.assertEqual(result["status"], "no_anchored_lexical_leads")

    def test_reports_anchored_source_sentence_and_overlong_omission_counts(self):
        long_sentence = "Metformin and warfarin were administered " + "x" * 1201 + "."
        text = long_sentence + " A second short sentence documents an assay."
        article = {**self.article, "sections": [{"section": "methods", "source_section": "METHODS", "passage_index": 4, "offset": 300, "text": text}]}
        result = build_interaction_fulltext_screen(article, **self.arguments)
        self.assertEqual(result["anchored_source_sentence_count"], 2)
        self.assertEqual(result["omitted_overlong_sentence_count"], 1)
        self.assertEqual(result["candidate_count"], 0)

    def test_emits_sentence_level_offsets_and_hashes_for_adjudication(self):
        text = "Metformin and warfarin were coadministered. A second unrelated sentence describes the assay."
        article = {**self.article, "sections": [{"section": "methods", "source_section": "METHODS", "passage_index": 4, "offset": 300, "text": text}]}
        result = build_interaction_fulltext_screen(article, **self.arguments)
        self.assertEqual(result["candidate_count"], 1)
        candidate = result["candidates"][0]
        expected = "Metformin and warfarin were coadministered."
        self.assertEqual(candidate["source_offset"], 300)
        self.assertEqual(candidate["source_length"], len(expected))
        self.assertEqual(candidate["text_sha256"], __import__("hashlib").sha256(expected.encode()).hexdigest())
        self.assertEqual(len(candidate["candidate_id"]), 64)
        self.assertNotIn(expected, json.dumps(result))

    def test_rehydrates_only_exact_candidates_from_matching_fresh_bioc(self):
        screen = build_interaction_fulltext_screen(self.article, **self.arguments)
        rows = rehydrate_interaction_fulltext_candidates(screen, self.article)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["text_sha256"] == screen["candidates"][index]["text_sha256"] for index, row in enumerate(rows)))
        with self.assertRaisesRegex(ValueError, "digest"):
            rehydrate_interaction_fulltext_candidates(screen, {**self.article, "source_sha256": "c" * 64})
        changed = {**screen, "candidates": [{**screen["candidates"][0], "source_offset": 101}]}
        self.assertEqual(rehydrate_interaction_fulltext_candidates(changed, self.article), [])

    def test_rejects_invalid_source_hash_and_ambiguous_candidate_limit(self):
        with self.assertRaisesRegex(ValueError, "source hash"):
            build_interaction_fulltext_screen({**self.article, "source_sha256": "bad"}, **self.arguments)
        with self.assertRaisesRegex(ValueError, "limit"):
            build_interaction_fulltext_screen(self.article, **self.arguments, limit=True)

    def test_candidate_must_be_a_pmc_linked_member_of_the_exact_pair_search(self):
        literature = {
            "status": "retrieved",
            "query_sha256": "c" * 64,
            "pmids": ["98765432"],
            "resolved_ingredient_names": [["metformin"], ["warfarin"]],
        }
        candidate = {"source": "PubMed", "pmid": "98765432", "identifiers": {"pmc": "PMC12345"}}
        binding = validate_interaction_fulltext_binding(literature, candidate, "98765432")
        self.assertEqual(binding["pmcid"], "PMC12345")
        with self.assertRaisesRegex(ValueError, "exact drug-pair search"):
            validate_interaction_fulltext_binding({**literature, "pmids": ["111"]}, candidate, "98765432")
        with self.assertRaisesRegex(ValueError, "valid linked PMC"):
            validate_interaction_fulltext_binding(literature, {**candidate, "identifiers": {"pmc": "https://evil.example"}}, "98765432")


if __name__ == "__main__":
    unittest.main()
