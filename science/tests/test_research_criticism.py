from __future__ import annotations

import unittest

from medai_science.llm import ModelCompletion
from medai_science.research_criticism import CRITICISM_SCHEMA, generate_critic_leads


class FakeProvider:
    def __init__(self, result):
        self.result = result

    async def complete_json(self, **kwargs):
        self.kwargs = kwargs
        return ModelCompletion(self.result, "test", "fixture", "a" * 64, "b" * 64, {"total_tokens": 7})


def candidate():
    return {"id": "HYP-1", "statement": "Gene A may be connected to disease B", "path": [{"source": "g1", "relation": "ASSOCIATED", "target": "p1"}, {"source": "p1", "relation": "INVOLVED_IN", "target": "d1"}], "evidence_ids": ["E1"]}


def completion():
    return {"hypothesis_id": "HYP-1", "refutation_questions": ["Could this association vary by study population?"], "alternative_explanations": ["The observed relation may reflect confounding."], "methodological_gaps": ["Were the findings independently replicated?"], "evidence_refs": ["E1"]}


class ResearchCriticismTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_bounded_leads_and_marks_them_for_human_review(self):
        provider = FakeProvider(completion())
        records = [{"id": "E1", "claim": "Gene A associated with disease B", "polarity": "supports", "quality": 0.6}]
        result = await generate_critic_leads(provider, candidate(), {"major_weaknesses": []}, records)
        self.assertEqual(result.value["classification"], "MODEL-GENERATED REVIEW LEADS")
        self.assertEqual(result.value["review_status"], "requires_human_scientific_review")
        self.assertIn("Gene A associated", provider.kwargs["prompt"])
        self.assertEqual(provider.kwargs["schema"], CRITICISM_SCHEMA)

    async def test_rejects_unprovided_evidence_references_and_wrong_candidate(self):
        result = completion()
        result["evidence_refs"] = ["INVENTED"]
        with self.assertRaisesRegex(ValueError, "not supplied"):
            await generate_critic_leads(FakeProvider(result), candidate(), {}, [{"id": "E1", "claim": "A B", "polarity": "supports"}])
        result = completion()
        result["hypothesis_id"] = "HYP-other"
        with self.assertRaisesRegex(ValueError, "different hypothesis"):
            await generate_critic_leads(FakeProvider(result), candidate(), {}, [{"id": "E1", "claim": "A B", "polarity": "supports"}])

    async def test_rejects_patient_data_in_model_generated_search_leads(self):
        result = completion()
        result["refutation_questions"] = ["Review patient@example.com records"]
        with self.assertRaises(ValueError):
            await generate_critic_leads(FakeProvider(result), candidate(), {}, [{"id": "E1", "claim": "A B", "polarity": "supports"}])

    async def test_does_not_send_patient_identifiers_from_retained_evidence_to_provider(self):
        provider = FakeProvider(completion())
        records = [{"id": "E1", "claim": "Patient identifier: MRN 1234567 has disease", "polarity": "supports"}]
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            await generate_critic_leads(provider, candidate(), {}, records)
        self.assertFalse(hasattr(provider, "kwargs"))


if __name__ == "__main__":
    unittest.main()
