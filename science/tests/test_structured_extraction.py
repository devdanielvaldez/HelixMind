from __future__ import annotations

import unittest

from medai_science.llm import ModelCompletion
from medai_science.paper_understanding import parse_bioc_xml
from medai_science.structured_extraction import EXTRACTION_SCHEMA, _candidate_records, extract_local_result_annotations, extraction_batch_count, license_policy_allows_model_processing, provider_is_local


BIOC = b'''<collection><document><id>PMC12345</id>
<passage><infon key="section_type">TITLE</infon><text>Case-control study of Protein X.</text></passage>
<passage><infon key="section_type">ABSTRACT</infon><text>We studied patients in a case-control cohort.</text></passage>
<passage><infon key="section_type">METHODS</infon><text>Case-control methods compared matched patients and controls.</text></passage>
<passage><infon key="section_type">RESULTS</infon><offset>0</offset><text>Protein X increased in the treated cell model. The sample size was small and should be interpreted carefully.</text>
<annotation id="1"><infon key="type">Protein</infon><infon key="identifier">P04156</infon><infon key="database">UniProt</infon><text>Protein X</text><location offset="0" length="9"/></annotation></passage>
</document></collection>'''


class FakeProvider:
    base_url = "http://127.0.0.1:11434/v1"

    def __init__(self, value):
        self.value = value

    async def complete_json(self, **kwargs):
        self.kwargs = kwargs
        return ModelCompletion(self.value, "local", "fixture", "a" * 64, "b" * 64, {"total_tokens": 8})


class BatchProvider:
    base_url = "http://127.0.0.1:11434/v1"

    async def complete_json(self, **kwargs):
        import json
        prompt_data = json.loads(kwargs["prompt"].split("\n", 1)[1])
        rows = prompt_data["result_candidates"]
        result = {"document_id": "PMC998877", "paper_context": {"study_design": "unclear", "population": "unclear", "method_family": "unclear"}, "annotations": [{"candidate_id": row["candidate_id"], "result_kind": "other", "direction": "unclear", "study_system": "unclear", "entity_ids": []} for row in rows]}
        return ModelCompletion(result, "local", "fixture", "c" * 64, "d" * 64, {"total_tokens": len(rows)})


def parsed():
    return parse_bioc_xml(BIOC)


def valid_output(record):
    return {
        "document_id": record["document_id"],
        "paper_context": {"study_design": "observational", "population": "patients", "method_family": "cohort_or_case_control"},
        "annotations": [{"candidate_id": row["candidate_id"], "result_kind": "cell_model", "direction": "increase", "study_system": "cell", "entity_ids": ["UniProtKB:P04156"]} for row in record["result_sentence_candidates"]],
    }


class StructuredExtractionTests(unittest.IsolatedAsyncioTestCase):
    def test_license_policy_requires_exact_operator_allowlist(self):
        self.assertFalse(license_policy_allows_model_processing("CC BY 4.0", ""))
        self.assertTrue(license_policy_allows_model_processing("CC BY 4.0", "cc by 4.0|CC0 1.0"))
        self.assertFalse(license_policy_allows_model_processing("CC BY-NC-ND 4.0", "CC BY 4.0"))
        self.assertFalse(license_policy_allows_model_processing(None, "CC BY 4.0"))

    async def test_classifies_every_result_sentence_and_retains_no_sentence_text(self):
        record = parsed()
        provider = FakeProvider(valid_output(record))
        result = await extract_local_result_annotations(provider, record)
        self.assertEqual(len(result.value["annotations"]), len(record["result_sentence_candidates"]))
        self.assertEqual(result.value["source_sha256"], record["source_sha256"])
        self.assertNotIn("Protein X increased", str(result.value))
        self.assertIn("Protein X increased", provider.kwargs["prompt"])
        self.assertEqual(provider.kwargs["schema"], EXTRACTION_SCHEMA)
        self.assertEqual(result.value["paper_context"]["population"], "patients")
        self.assertEqual(result.value["paper_context_provenance"]["included_sections"], ["title", "abstract", "methods"])
        self.assertNotIn("case-control methods", str(result.value))
        self.assertIn("evidence polarity", result.value["interpretation_limit"])
        first_candidate = record["result_sentence_candidates"][0]
        first_anchor = result.value["source_anchors"][first_candidate["candidate_id"]]
        self.assertEqual(first_anchor["source_offset"], first_candidate["source_offset"])
        self.assertEqual(first_anchor["source_length"], len(first_candidate["text"]))

    async def test_rejects_remote_provider_for_full_text_and_wrong_entity_or_candidate(self):
        record = parsed()
        provider = FakeProvider(valid_output(record))
        provider.base_url = "https://model.example/v1"
        self.assertFalse(provider_is_local(provider))
        with self.assertRaisesRegex(ValueError, "loopback"):
            await extract_local_result_annotations(provider, record)
        provider.base_url = "http://localhost:11434/v1"
        provider.value["annotations"][0]["entity_ids"] = ["invented-id"]
        with self.assertRaisesRegex(ValueError, "not annotated"):
            await extract_local_result_annotations(provider, record)

    async def test_rejects_incomplete_annotation_set(self):
        record = parsed()
        output = valid_output(record)
        output["annotations"].pop()
        with self.assertRaisesRegex(ValueError, "every supplied"):
            await extract_local_result_annotations(FakeProvider(output), record)

    async def test_processes_long_articles_in_complete_chargable_batches(self):
        record = {"source": "PMC BioC", "document_id": "PMC998877", "source_sha256": "f" * 64, "normalized_entities": [], "result_sentence_candidates": [{"candidate_id": "candidate-%03d" % index, "passage_index": index, "source_offset": index * 20, "text": "Result candidate sentence number %d is long enough for a source result." % index} for index in range(21)]}
        provider = BatchProvider()
        result = await extract_local_result_annotations(provider, record)
        self.assertEqual(extraction_batch_count(record), 2)
        self.assertEqual(len(result.value["annotations"]), 21)
        self.assertEqual(result.usage["total_tokens"], 21)
        self.assertEqual(result.max_output_tokens, 4096)
        self.assertEqual(len(result.value["source_anchors"]), 21)

    async def test_conflicting_paper_context_labels_across_batches_abstain(self):
        record = {"source": "PMC BioC", "document_id": "PMC998877", "source_sha256": "f" * 64, "normalized_entities": [], "methods": "A bounded methods description.", "result_sentence_candidates": [{"candidate_id": "candidate-%03d" % index, "passage_index": index, "source_offset": index * 20, "text": "Result candidate sentence number %d is long enough for a source result." % index} for index in range(21)]}

        class ConflictingBatchProvider(BatchProvider):
            def __init__(self):
                self.calls = 0

            async def complete_json(self, **kwargs):
                import json
                prompt_data = json.loads(kwargs["prompt"].split("\n", 1)[1])
                rows = prompt_data["result_candidates"]
                self.calls += 1
                context = {"study_design": "observational", "population": "patients", "method_family": "cohort_or_case_control"}
                if self.calls == 2:
                    context["population"] = "healthy_volunteers"
                result = {"document_id": "PMC998877", "paper_context": context, "annotations": [{"candidate_id": row["candidate_id"], "result_kind": "other", "direction": "unclear", "study_system": "unclear", "entity_ids": []} for row in rows]}
                return ModelCompletion(result, "local", "fixture", "c" * 64, "d" * 64, {"total_tokens": len(rows)})

        result = await extract_local_result_annotations(ConflictingBatchProvider(), record)
        self.assertTrue(result.value["paper_context_batch_conflict"])
        self.assertEqual(result.value["paper_context"]["population"], "unclear")

    def test_sentence_bioc_entity_context_does_not_leak_between_sibling_sentences(self):
        first = "Protein A changed under treatment and showed a robust response."
        second = "Protein B remained stable throughout all experiments and controls."
        second_offset = 100 + len(first) + 1
        payload = (
            '<collection><document><id>PMC998878</id><passage>'
            '<infon key="section_type">RESULTS</infon><offset>100</offset>'
            f'<sentence><offset>100</offset><text>{first}</text><annotation id="1">'
            '<infon key="type">Protein</infon><infon key="identifier">P04156</infon>'
            '<infon key="database">UniProt</infon><text>Protein A</text>'
            '<location offset="100" length="9"/></annotation></sentence>'
            f'<sentence><offset>{second_offset}</offset><text>{second}</text><annotation id="2">'
            '<infon key="type">Protein</infon><infon key="identifier">Q9Y6K9</infon>'
            '<infon key="database">UniProt</infon><text>Protein B</text>'
            f'<location offset="{second_offset}" length="9"/></annotation></sentence>'
            '</passage></document></collection>'
        ).encode()
        candidates, _ = _candidate_records(parse_bioc_xml(payload))
        self.assertEqual([row["entity_ids"] for row in candidates], [["UniProtKB:P04156"], ["UniProtKB:Q9Y6K9"]])

    def test_candidate_ids_are_source_namespace_qualified(self):
        payload = (
            b"<collection><document><id>PMC998879</id><passage>"
            b"<infon key=\"section_type\">RESULTS</infon><offset>0</offset>"
            b"<text>Protein protein were both seen in samples.</text>"
            b"<annotation><infon key=\"type\">Protein</infon><infon key=\"identifier\">P04156</infon>"
            b"<infon key=\"database\">UniProt</infon><text>Protein</text><location offset=\"0\" length=\"7\"/></annotation>"
            b"<annotation><infon key=\"type\">Protein</infon><infon key=\"identifier\">P04156</infon>"
            b"<infon key=\"database\">Local Registry</infon><text>protein</text><location offset=\"8\" length=\"7\"/></annotation>"
            b"</passage></document></collection>"
        )
        candidates, _ = _candidate_records(parse_bioc_xml(payload))
        self.assertEqual(candidates[0]["entity_ids"], ["Local_Registry:P04156", "UniProtKB:P04156"])


if __name__ == "__main__":
    unittest.main()
