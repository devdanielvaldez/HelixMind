from __future__ import annotations

import unittest

from medai_science.claim_semantics import ALIGNMENT_SCHEMA, claim_context_receipt, compare_claim_pair
from medai_science.llm import ModelCompletion, ModelProviderError


class FakeProvider:
    def __init__(self, value):
        self.value = value

    async def complete_json(self, **kwargs):
        self.kwargs = kwargs
        return ModelCompletion(self.value, "local-test", "fixture", "a" * 64, "b" * 64, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})


def response(candidate_a="candidate-a", candidate_b="candidate-b"):
    return {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "relation": "same_question_opposing",
        "population_comparability": "unclear",
        "method_comparability": "not_reported",
        "reason_codes": ["same_entity", "same_outcome", "opposite_direction", "insufficient_context"],
    }


METADATA_A = {"result_kind": "association", "direction": "increase", "entity_ids": ["NCBI:1"]}
METADATA_B = {"result_kind": "association", "direction": "decrease", "entity_ids": ["NCBI:1"]}


class ClaimSemanticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_receipt_binds_model_visible_methods_and_population_without_text(self):
        first = {
            "result_kind": "association",
            "direction": "increase",
            "study_system": "human cohort",
            "entity_ids": ["NCBI:1"],
            "title": "Study title",
            "abstract_context": "Cohort summary",
            "methods_context": "Prospective cohort with matched controls",
        }
        receipt = claim_context_receipt(first)
        self.assertEqual(len(receipt["metadata_sha256"]), 64)
        self.assertEqual(receipt["context_character_lengths"]["methods_context"], 40)
        self.assertNotIn("Prospective cohort", str(receipt))
        changed = claim_context_receipt({**first, "methods_context": "Retrospective case-control study"})
        self.assertNotEqual(receipt["metadata_sha256"], changed["metadata_sha256"])
        with self.assertRaisesRegex(ValueError, "exceeds 500"):
            claim_context_receipt({**first, "title": "x" * 501})

    async def test_returns_bounded_screening_label_and_keeps_full_sentences_out_of_result(self):
        provider = FakeProvider(response())
        result = await compare_claim_pair(
            provider,
            candidate_a="candidate-a",
            text_a="Gene X was associated with increased outcome Y in this cohort.",
            candidate_b="candidate-b",
            text_b="Gene X was associated with decreased outcome Y in an independent cohort.",
            metadata_a=METADATA_A,
            metadata_b=METADATA_B,
        )
        self.assertEqual(result.value["assessment_type"], "local_model_semantic_screening")
        self.assertEqual(result.value["review_status"], "requires_human_scientific_review")
        self.assertEqual(result.value["relation"], "same_question_opposing")
        self.assertEqual(len(result.value["source_context_receipts"][0]["metadata_sha256"]), 64)
        self.assertEqual(provider.kwargs["schema"], ALIGNMENT_SCHEMA)
        self.assertNotIn("Gene X was associated", str(result.value))

    async def test_allows_cross_source_review_without_fabricating_source_annotations(self):
        provider = FakeProvider(response("label-claim", "paper-claim"))
        result = await compare_claim_pair(
            provider, candidate_a="label-claim",
            text_a="The versioned label reports altered exposure with coadministration.",
            candidate_b="paper-claim",
            text_b="The study reports no detectable exposure change.",
            metadata_a={"source": "DailyMed", "version": "2025-01"},
            metadata_b={"source": "PubMed", "pmid": "123456"},
        )
        self.assertEqual(result.value["relation"], "same_question_opposing")
        self.assertEqual(result.value["review_status"], "requires_human_scientific_review")
        self.assertIn("different evidence distributions", provider.kwargs["system"])

    async def test_rejects_wrong_candidate_ids_and_out_of_schema_model_labels(self):
        with self.assertRaisesRegex(ValueError, "different source candidate"):
            await compare_claim_pair(
                FakeProvider(response("invented", "candidate-b")),
                candidate_a="candidate-a", text_a="A sufficiently long source result sentence.",
                candidate_b="candidate-b", text_b="Another sufficiently long source result sentence.",
                metadata_a=METADATA_A, metadata_b=METADATA_B,
            )
        invalid = response()
        invalid["relation"] = "confirmed_contradiction"
        with self.assertRaisesRegex(ModelProviderError, "outside the allowed enum"):
            await compare_claim_pair(
                FakeProvider(invalid), candidate_a="candidate-a", text_a="A sufficiently long source result sentence.",
                candidate_b="candidate-b", text_b="Another sufficiently long source result sentence.",
                metadata_a=METADATA_A, metadata_b=METADATA_B,
            )

    async def test_bounds_text_and_requires_distinct_source_candidates(self):
        provider = FakeProvider(response())
        with self.assertRaisesRegex(ValueError, "distinct"):
            await compare_claim_pair(provider, candidate_a="same", text_a="Sentence one long enough to be passed.", candidate_b="same", text_b="Sentence two long enough to be passed.", metadata_a=METADATA_A, metadata_b=METADATA_B)
        with self.assertRaisesRegex(ValueError, "characters"):
            await compare_claim_pair(provider, candidate_a="a", text_a="x" * 1201, candidate_b="b", text_b="Sentence two long enough to be passed.", metadata_a=METADATA_A, metadata_b=METADATA_B)
        for candidate in ("", " " * 3, "c" * 65):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(ValueError, "distinct source candidate"):
                await compare_claim_pair(provider, candidate_a=candidate, text_a="Sentence one long enough to be passed.", candidate_b="other", text_b="Sentence two long enough to be passed.", metadata_a=METADATA_A, metadata_b=METADATA_B)

    async def test_pair_comparison_applies_context_bounds_and_strict_json_validation(self):
        args = {
            "candidate_a": "candidate-a", "text_a": "A sufficiently long source result sentence.",
            "candidate_b": "candidate-b", "text_b": "Another sufficiently long source result sentence.",
            "metadata_a": {**METADATA_A, "title": "x" * 501}, "metadata_b": METADATA_B,
        }
        with self.assertRaisesRegex(ValueError, "exceeds 500"):
            await compare_claim_pair(FakeProvider(response()), **args)
        args["metadata_a"] = {**METADATA_A, "unused": float("nan")}
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            await compare_claim_pair(FakeProvider(response()), **args)

    async def test_rejects_model_relations_that_conflict_with_source_directions(self):
        conflicting = response()
        conflicting["relation"] = "same_question_consistent"
        conflicting["reason_codes"] = ["same_entity", "same_outcome"]
        with self.assertRaisesRegex(ValueError, "conflicts with source-annotated direction"):
            await compare_claim_pair(
                FakeProvider(conflicting), candidate_a="candidate-a",
                text_a="An increase was observed for the source outcome.",
                candidate_b="candidate-b", text_b="A decrease was observed for that outcome.",
                metadata_a=METADATA_A, metadata_b=METADATA_B,
            )

        opposing = response()
        opposing["reason_codes"] = ["same_entity", "same_outcome"]
        with self.assertRaisesRegex(ValueError, "conflicts with source-annotated direction"):
            await compare_claim_pair(
                FakeProvider(opposing), candidate_a="candidate-a",
                text_a="Gene X increases outcome Y.", text_b="Gene X increases outcome Y.",
                candidate_b="candidate-b", metadata_a=METADATA_A,
                metadata_b={**METADATA_A},
            )

    async def test_rejects_internally_inconsistent_relation_reason_codes(self):
        missing_outcome = response()
        missing_outcome["reason_codes"] = ["same_entity", "opposite_direction"]
        with self.assertRaisesRegex(ValueError, "same-outcome and opposite-direction"):
            await compare_claim_pair(
                FakeProvider(missing_outcome), candidate_a="candidate-a",
                text_a="Gene X was associated with increased outcome Y.",
                candidate_b="candidate-b", text_b="Gene X was associated with decreased outcome Y.",
                metadata_a={"source": "source-a"}, metadata_b={"source": "source-b"},
            )

        duplicate_reasons = response()
        duplicate_reasons["reason_codes"] = ["same_entity", "same_outcome", "same_outcome", "opposite_direction"]
        with self.assertRaisesRegex(ValueError, "duplicate reason codes"):
            await compare_claim_pair(
                FakeProvider(duplicate_reasons), candidate_a="candidate-a",
                text_a="Gene X was associated with increased outcome Y.",
                candidate_b="candidate-b", text_b="Gene X was associated with decreased outcome Y.",
                metadata_a={"source": "source-a"}, metadata_b={"source": "source-b"},
            )

        conflicting_reasons = response()
        conflicting_reasons["reason_codes"] = ["same_entity", "same_outcome", "different_outcome", "opposite_direction"]
        with self.assertRaisesRegex(ValueError, "conflicting outcome reason codes"):
            await compare_claim_pair(
                FakeProvider(conflicting_reasons), candidate_a="candidate-a",
                text_a="Gene X was associated with increased outcome Y.",
                candidate_b="candidate-b", text_b="Gene X was associated with decreased outcome Y.",
                metadata_a={"source": "source-a"}, metadata_b={"source": "source-b"},
            )

    async def test_opposing_screen_requires_explicit_opposite_source_directions(self):
        for direction_a, direction_b in (("mixed", "increase"), ("unclear", "decrease"), ("no_detected_effect", "increase")):
            with self.subTest(direction_a=direction_a, direction_b=direction_b):
                with self.assertRaisesRegex(ValueError, "conflicts with source-annotated direction"):
                    await compare_claim_pair(
                        FakeProvider(response()), candidate_a="candidate-a",
                        text_a="A sufficiently long source result sentence.",
                        candidate_b="candidate-b", text_b="Another sufficiently long source result sentence.",
                        metadata_a={**METADATA_A, "direction": direction_a},
                        metadata_b={**METADATA_B, "direction": direction_b},
                    )

    async def test_rejects_pairs_without_matching_source_entities_or_result_kind(self):
        with self.assertRaisesRegex(ValueError, "shared source entity"):
            await compare_claim_pair(
                FakeProvider(response()), candidate_a="candidate-a",
                text_a="A sufficiently long source result sentence.", candidate_b="candidate-b",
                text_b="Another sufficiently long source result sentence.",
                metadata_a=METADATA_A,
                metadata_b={**METADATA_B, "entity_ids": ["NCBI:2"]},
            )
        with self.assertRaisesRegex(ValueError, "result category"):
            await compare_claim_pair(
                FakeProvider(response()), candidate_a="candidate-a",
                text_a="A sufficiently long source result sentence.", candidate_b="candidate-b",
                text_b="Another sufficiently long source result sentence.",
                metadata_a=METADATA_A,
                metadata_b={**METADATA_B, "result_kind": "clinical"},
            )
        with self.assertRaisesRegex(ValueError, "source-annotated directions"):
            await compare_claim_pair(
                FakeProvider(response()), candidate_a="candidate-a",
                text_a="A sufficiently long source result sentence.", candidate_b="candidate-b",
                text_b="Another sufficiently long source result sentence.",
                metadata_a={**METADATA_A, "direction": {"unexpected": "object"}},
                metadata_b=METADATA_B,
            )


if __name__ == "__main__":
    unittest.main()
