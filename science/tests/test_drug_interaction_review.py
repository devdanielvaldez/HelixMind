import unittest

from medai_science.drug_interaction_review import rank_candidate_pairs, screen_drug_label_candidates, screen_sentence_candidates, sentence_candidates


class DrugInteractionReviewTests(unittest.TestCase):
    def test_sentence_spans_are_exact_hash_linked_and_exclude_overlong_sentences(self):
        text = "  Warfarin exposure increased.\n" + ("x" * 1201) + ". Aspirin was also measured."
        rows = sentence_candidates(text, "PMID")
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(text[row["offset"]:row["offset"] + row["length"]], row["text"])
        self.assertEqual(rows[0]["text"], "Warfarin exposure increased.")
        self.assertEqual(len(rows[0]["text_sha256"]), 64)

    def test_pair_ranking_is_bounded_deterministic_and_does_not_claim_conflict(self):
        labels = sentence_candidates("Concomitant warfarin exposure increases bleeding risk.", "LABEL")
        papers = sentence_candidates("Warfarin exposure was associated with increased bleeding.", "PMID")
        pairs = rank_candidate_pairs(labels, papers, 1)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["shared_terms"], ["bleeding", "exposure", "warfarin"])
        with self.assertRaisesRegex(ValueError, "1..10"):
            rank_candidate_pairs(labels, papers, 11)

    def test_sentence_screen_reports_omitted_candidates_and_overlong_sentences(self):
        many = screen_sentence_candidates(" ".join(["Warfarin exposure." for _ in range(101)]), "PMID")
        self.assertEqual(many["source_sentence_count"], 101)
        self.assertEqual(many["retained_candidate_count"], 100)
        self.assertEqual(many["omitted_by_limit_count"], 1)
        self.assertTrue(many["truncated"])

        mixed = screen_sentence_candidates("Valid sentence. " + "x" * 1201 + ". Another valid sentence.", "LABEL")
        self.assertEqual(mixed["eligible_candidate_count"], 2)
        self.assertEqual(mixed["omitted_overlong_sentence_count"], 1)
        self.assertTrue(mixed["truncated"])

    def test_label_candidate_inventory_counts_global_cap_and_preserves_version_context(self):
        labels = {
            "warfarin": [
                {"drug_interactions": "Warfarin exposure is studied.", "set_id": "set-1", "version": "20250101"},
                {"drug_interactions": "Warfarin metabolism may change.", "set_id": "set-2", "version": "20260101"},
            ],
            "metformin": [{"drug_interactions": "Metformin has a separate interaction section."}],
        }
        result = screen_drug_label_candidates(labels, limit=1)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["version"], "20250101")
        self.assertEqual(result["coverage"]["source_sentence_count"], 3)
        self.assertEqual(result["coverage"]["omitted_by_limit_count"], 2)
        self.assertTrue(result["coverage"]["truncated"])


if __name__ == "__main__":
    unittest.main()
