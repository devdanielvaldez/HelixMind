import unittest

from medai_science.drug_interaction_review import rank_candidate_pairs, sentence_candidates


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


if __name__ == "__main__":
    unittest.main()
