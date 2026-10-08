from __future__ import annotations

import asyncio
import hashlib
import unittest

from medai_science.code_generation import generate_python_capability, generate_r_capability
from medai_science.llm import ModelCompletion


class _Provider:
    def __init__(self, source: str):
        self.source = source
        self.schema_name = None

    async def complete_json(self, *, system: str, prompt: str, schema_name: str, schema: dict) -> ModelCompletion:
        self.schema_name = schema_name
        self.prompt = prompt
        return ModelCompletion(
            {"source": self.source, "implementation_summary": "Deterministic pure transformation."},
            "test-provider",
            "test-model",
            "a" * 64,
            "b" * 64,
            {"total_tokens": 42},
        )


class CapabilityCodeGenerationTests(unittest.TestCase):
    def _generate(self, provider: _Provider):
        return asyncio.run(generate_python_capability(
            provider,
            capability_id="genomics.variant_summary",
            description="Summarize validated variant records.",
            source_session="RES-2026-0001",
            contract={"input": {"type": "object"}, "output": {"type": "object"}},
            implementation_requirements=["Count variant records by chromosome."],
        ))

    def test_r_proposal_is_signed_lifecycle_ready_shape_but_disabled_pending_review(self):
        provider = _Provider("run <- function(input) list(count = length(input$variants))\n")
        proposal, completion = asyncio.run(generate_r_capability(
            provider,
            capability_id="genomics.variant_summary_r",
            description="Summarize validated variant records.",
            source_session="RES-2026-0001",
            contract={"input": {"type": "object"}, "output": {"type": "object"}},
            implementation_requirements=["Count variant records by chromosome."],
        ))
        self.assertEqual(provider.schema_name, "medai_r_capability_source")
        self.assertEqual(proposal["language"], "r")
        self.assertFalse(proposal["execution"]["enabled"])
        self.assertEqual(proposal["lifecycle_status"], "experimental_pending_review")
        self.assertEqual(proposal["generation"]["response_sha256"], completion.response_sha256)
        self.assertEqual(proposal["static_review"]["source_sha256"], hashlib.sha256(provider.source.encode()).hexdigest())
        self.assertIn("not registered", " ".join(proposal["limitations"]))

    def test_r_proposal_rejects_dangerous_source_before_artifact_persistence(self):
        provider = _Provider("run <- function(input) system('curl https://example.invalid')\n")
        with self.assertRaisesRegex(ValueError, "denied function"):
            asyncio.run(generate_r_capability(
                provider,
                capability_id="genomics.variant_summary_r",
                description="Summarize variants.",
                source_session="RES-1",
                contract={"input": {"type": "object"}, "output": {"type": "object"}},
                implementation_requirements=["Count variants."],
            ))

    def test_returns_disabled_source_artifact_with_model_and_static_provenance(self):
        provider = _Provider("def run(input):\n    return {'count': len(input['variants'])}\n")
        proposal, completion = self._generate(provider)
        self.assertEqual(provider.schema_name, "medai_python_capability_source")
        self.assertEqual(proposal["manifest"]["id"], "genomics.variant_summary")
        self.assertFalse(proposal["execution"]["enabled"])
        self.assertEqual(proposal["manifest"]["permissions"], [])
        self.assertEqual(proposal["lifecycle_status"], "experimental_pending_review")
        self.assertEqual(proposal["generation"]["response_sha256"], completion.response_sha256)
        self.assertEqual(proposal["static_review"]["source_sha256"], hashlib.sha256(provider.source.encode()).hexdigest())
        self.assertIn("independent expected-output cases pending", proposal["validation"]["scientific"])
        self.assertIn("not registered", " ".join(proposal["limitations"]))

    def test_generated_source_is_rejected_by_sandbox_static_policy(self):
        provider = _Provider("import os\ndef run(input):\n    return {'cwd': os.getcwd()}\n")
        with self.assertRaisesRegex(ValueError, "outside the allowlist"):
            self._generate(provider)

    def test_proposal_size_is_bounded_for_the_persistent_artifact_store(self):
        provider = _Provider("def run(input):\n    return {'value': '" + "x" * 260_000 + "'}\n")
        with self.assertRaisesRegex(ValueError, "artifact payload limit"):
            self._generate(provider)

    def test_rejects_patient_data_and_invalid_namespace_before_model_call(self):
        provider = _Provider("def run(input): return {}\n")
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            asyncio.run(generate_python_capability(
                provider,
                capability_id="genomics.variant_summary",
                description="Summarize patient id: MRN-928271 variant records.",
                source_session="RES-1",
                contract={"input": {"type": "object"}, "output": {"type": "object"}},
                implementation_requirements=["Count variants."],
            ))
        self.assertIsNone(provider.schema_name)
        with self.assertRaisesRegex(ValueError, "supported type"):
            asyncio.run(generate_python_capability(
                provider,
                capability_id="genomics.variant_summary",
                description="Summarize variants.",
                source_session="RES-1",
                contract={"input": {"type": "unvalidated"}, "output": {"type": "object"}},
                implementation_requirements=["Count variants."],
            ))
        self.assertIsNone(provider.schema_name)
        with self.assertRaisesRegex(ValueError, "namespaced"):
            asyncio.run(generate_python_capability(
                provider,
                capability_id="variant_summary",
                description="Summarize variants.",
                source_session="RES-1",
                contract={"input": {"type": "object"}, "output": {"type": "object"}},
                implementation_requirements=["Count variants."],
            ))


if __name__ == "__main__":
    unittest.main()
