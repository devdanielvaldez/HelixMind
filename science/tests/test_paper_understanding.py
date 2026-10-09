import json
import unittest

import httpx

from medai_science.paper_understanding import _sentence_spans, build_extraction_retention_receipt, parse_bioc_xml
from medai_science.literature import structure_pmc_receipt
from medai_science.sources.pmc_bioc import PMCBioCClient


BIOC = b'''<?xml version="1.0" encoding="UTF-8"?>
<collection><source>PMC</source><date>20260101</date><document>
<id>PMC123456</id><infon key="license">CC BY 4.0</infon>
<passage><infon key="section_type">TITLE</infon><offset>0</offset><text>Example biomedical paper</text></passage>
<passage><infon key="section_type">ABSTRACT</infon><offset>24</offset><text>Abstract summary text.</text></passage>
<passage><infon key="section_type">METHODS</infon><offset>47</offset><text>We analyzed 100 samples using assay X.</text></passage>
<passage><infon key="section_type">RESULTS</infon><offset>87</offset><text>APOE was associated with outcome Y in this cohort. The result requires independent validation.</text>
<annotation id="1"><infon key="type">Gene</infon><infon key="identifier">NCBI:348</infon><infon key="database">NCBI Gene</infon><text>APOE</text><location offset="87" length="4"/></annotation></passage>
<passage><infon key="section_type">LIMITATIONS</infon><offset>182</offset><text>The sample size was modest.</text></passage>
</document></collection>'''


class PaperUnderstandingTests(unittest.TestCase):
    def test_duplicate_or_missing_infon_keys_are_rejected(self):
        for infons in (
            b'<infon key="section_type">RESULTS</infon><infon key=" section_type ">METHODS</infon>',
            b'<infon>RESULTS</infon>',
        ):
            payload = (
                b"<collection><document><id>PMC-ambiguous</id><passage>"
                + infons
                + b"<offset>0</offset><text>A sufficiently long passage for stable parsing.</text>"
                + b"</passage></document></collection>"
            )
            with self.subTest(infons=infons), self.assertRaisesRegex(ValueError, "infon"):
                parse_bioc_xml(payload)

    def test_unscoped_database_local_identifiers_are_not_coalesced(self):
        result = parse_bioc_xml(
            b"<collection><document><id>PMC-unscoped</id><passage><offset>0</offset>"
            b"<text>Protein X and Protein Y were measured.</text>"
            b"<annotation><infon key=\"type\">Protein</infon><infon key=\"identifier\">17</infon>"
            b"<text>Protein X</text><location offset=\"0\" length=\"9\"/></annotation>"
            b"<annotation><infon key=\"type\">Protein</infon><infon key=\"identifier\">17</infon>"
            b"<text>Protein Y</text><location offset=\"14\" length=\"9\"/></annotation>"
            b"</passage></document></collection>"
        )
        entities = result["normalized_entities"]
        self.assertEqual(len(entities), 2)
        self.assertTrue(all(entity["database"] is None for entity in entities))
        self.assertTrue(all(entity["identity_status"] == "unresolved_namespace" for entity in entities))

    def test_section_mapping_uses_recognized_heading_when_bioc_type_is_coarse(self):
        payload = (
            b"<collection><document><id>PMC777</id>"
            b"<passage><infon key=\"section_type\">BODY</infon>"
            b"<infon key=\"section\">2. Experimental Procedures</infon>"
            b"<offset>0</offset><text>We analyzed the cohort using a prespecified assay.</text></passage>"
            b"<passage><infon key=\"section_type\">BODY</infon>"
            b"<infon key=\"section\">Findings</infon><offset>50</offset>"
            b"<text>The measured outcome changed after treatment in the sampled cohort.</text></passage>"
            b"</document></collection>"
        )
        result = parse_bioc_xml(payload)
        self.assertEqual(result["methods"], "We analyzed the cohort using a prespecified assay.")
        self.assertEqual(result["results"], "The measured outcome changed after treatment in the sampled cohort.")
        self.assertEqual(result["sections"][0]["section"], "methods")
        self.assertEqual(result["sections"][0]["source_section"], "2. Experimental Procedures")

    def test_common_method_and_result_heading_aliases_keep_source_labels(self):
        for heading, expected in (
            ("Methodology", "methods"), ("Method", "methods"), ("Patients and Methods", "methods"),
            ("Materials & Methods", "methods"), ("Methods and Material", "methods"),
            ("Statistical Analysis", "methods"), ("Statistical Methods", "methods"),
            ("Study Design and Participants", "methods"), ("Participants and Study Design", "methods"),
            ("Study Design and Setting", "methods"), ("Sample Collection", "methods"),
            ("Study Methods", "methods"),
            ("Experimental Methods", "methods"), ("Experimental Procedure", "methods"),
            ("Study Participants", "population"), ("Study Subjects", "population"),
            ("Patient Population", "population"),
            ("Outcome", "results"), ("Outcomes", "results"), ("Finding", "results"),
        ):
            with self.subTest(heading=heading):
                payload = (
                    "<collection><document><id>PMC999</id><passage>"
                    "<infon key='section_type'>BODY</infon>"
                    f"<infon key='section'>{heading.replace('&', '&amp;')}</infon>"
                    "<offset>0</offset><text>A sufficiently long passage for stable parsing.</text>"
                    "</passage></document></collection>"
                )
                parsed = parse_bioc_xml(payload)
                self.assertEqual(parsed["sections"][0]["section"], expected)
                self.assertEqual(parsed["sections"][0]["source_section"], heading)

    def test_unrecognized_section_label_remains_explicit(self):
        result = parse_bioc_xml(
            b"<collection><document><id>PMC778</id><passage>"
            b"<infon key=\"section_type\">BODY</infon>"
            b"<infon key=\"section\">Emerging questions</infon>"
            b"<text>Several questions remain for future investigation.</text>"
            b"</passage></document></collection>"
        )
        self.assertEqual(result["sections"][0]["section"], "emerging_questions")
        self.assertEqual(result["sections"][0]["source_section"], "Emerging questions")

    def test_fallback_sentence_candidates_do_not_split_common_biomedical_abbreviations(self):
        text = "Dr. Smith measured TREM2 levels in microglia, e.g. after stimulation. The effect persisted across independent cohorts."
        spans = _sentence_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0][1], "Dr. Smith measured TREM2 levels in microglia, e.g. after stimulation.")
        self.assertEqual(spans[1][0], text.index("The effect"))
        cited = "Smith et al. observed the response across multiple independent cohorts. The result was reproducible in both groups."
        self.assertEqual(len(_sentence_spans(cited)), 2)

    def test_extracts_sections_normalized_entities_and_unclassified_result_candidates(self):
        result = parse_bioc_xml(BIOC, source_url="https://example.org/PMC123456")
        self.assertEqual(result["document_id"], "PMC123456")
        self.assertEqual(result["source_license"], "CC BY 4.0")
        self.assertEqual(result["methods"], "We analyzed 100 samples using assay X.")
        self.assertEqual(result["limitations"], "The sample size was modest.")
        gene = result["normalized_entities"][0]
        self.assertEqual((gene["identifier"], gene["entity_type"]), ("NCBI:348", "Gene"))
        self.assertEqual(gene["mentions"][0]["text"], "APOE")
        self.assertEqual(len(result["result_sentence_candidates"]), 2)
        self.assertEqual({item["polarity"] for item in result["result_sentence_candidates"]}, {"unclassified"})
        sentence = result["result_sentence_candidates"][0]
        result_text = "APOE was associated with outcome Y in this cohort. The result requires independent validation."
        self.assertEqual(sentence["source_offset"], 87 + result_text.index("APOE"))
        self.assertEqual(sentence["source_length"], len(sentence["text"]))
        self.assertEqual(result["result_sentence_candidates"][1]["source_offset"], 87 + result_text.index("The result"))

    def test_entity_grouping_keeps_database_namespaces_distinct(self):
        payload = (
            b"<collection><document><id>PMC7799</id>"
            b"<passage><offset>0</offset><text>APOE and APOE were observed in the cohort.</text>"
            b"<annotation><infon key=\"type\">Gene</infon><infon key=\"identifier\">123</infon>"
            b"<infon key=\"database\">NCBI Gene</infon><text>APOE</text>"
            b"<location offset=\"0\" length=\"4\"/></annotation>"
            b"<annotation><infon key=\"type\">Gene</infon><infon key=\"identifier\">123</infon>"
            b"<infon key=\"database\">Other Registry</infon><text>APOE</text>"
            b"<location offset=\"9\" length=\"4\"/></annotation>"
            b"</passage></document></collection>"
        )
        result = parse_bioc_xml(payload)
        self.assertEqual(result["parser_version"], "0.2.11")
        entities = result["normalized_entities"]
        self.assertEqual(len(entities), 2)
        self.assertEqual({entity["database"] for entity in entities}, {"NCBI Gene", "Other Registry"})
        self.assertEqual([entity["mentions"][0]["text"] for entity in entities], ["APOE", "APOE"])

    def test_entity_grouping_normalizes_database_label_whitespace_and_case(self):
        payload = (
            b"<collection><document><id>PMC7800</id><passage><offset>0</offset>"
            b"<text>APOE was measured. APOE was measured.</text>"
            b"<annotation><infon key=\"type\">Gene</infon><infon key=\"identifier\">NCBI:348</infon>"
            b"<infon key=\"database\">NCBI Gene</infon><text>APOE</text>"
            b"<location offset=\"0\" length=\"4\"/></annotation>"
            b"<annotation><infon key=\"type\">Gene</infon><infon key=\"identifier\">NCBI:348</infon>"
            b"<infon key=\"database\">  ncbi   gene </infon><text>APOE</text>"
            b"<location offset=\"19\" length=\"4\"/></annotation>"
            b"</passage></document></collection>"
        )
        entities = parse_bioc_xml(payload)["normalized_entities"]
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0]["mentions"]), 2)

    def test_repeated_sentence_text_in_one_passage_has_distinct_source_offsets_and_ids(self):
        original = b"APOE was associated with outcome Y in this cohort. The result requires independent validation."
        repeated = b"Repeated result candidate has enough characters. Repeated result candidate has enough characters."
        result = parse_bioc_xml(BIOC.replace(original, repeated))
        candidates = result["result_sentence_candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["text"], candidates[1]["text"])
        self.assertEqual(candidates[0]["source_offset"], 87)
        self.assertEqual(candidates[1]["source_offset"], 87 + len(candidates[0]["text"]) + 1)
        self.assertNotEqual(candidates[0]["candidate_id"], candidates[1]["candidate_id"])

    def test_parses_bioc_sentence_elements_and_sentence_scoped_entity_offsets(self):
        sentence = (
            b"<collection><document><id>PMC999</id><passage>"
            b"<infon key=\"section_type\">RESULTS</infon><offset>100</offset>"
            b"<sentence><offset>100</offset><text>Protein X increased in the study and met the preset threshold.</text>"
            b"<annotation id=\"1\"><infon key=\"type\">Protein</infon>"
            b"<infon key=\"identifier\">P00001</infon><infon key=\"database\">UniProt</infon>"
            b"<text>Protein X</text><location offset=\"100\" length=\"9\"/></annotation>"
            b"</sentence></passage></document></collection>"
        )
        result = parse_bioc_xml(sentence)
        candidate = result["result_sentence_candidates"][0]
        self.assertEqual(candidate["source_offset"], 100)
        self.assertEqual(candidate["sentence_index"], 0)
        self.assertEqual(result["normalized_entities"][0]["mentions"][0]["text"], "Protein X")
        malformed_offset = parse_bioc_xml(sentence.replace(b"<offset>100</offset>", b"<offset>bad</offset>"))
        self.assertIsNone(malformed_offset["result_sentence_candidates"][0]["source_offset"])

    def test_rejects_entity_spans_outside_the_source_sentence(self):
        sentence = (
            b"<collection><document><id>PMC999</id><passage>"
            b"<infon key=\"section_type\">RESULTS</infon><offset>100</offset>"
            b"<sentence><offset>100</offset><text>Protein X increased in the study and met the preset threshold.</text>"
            b"<annotation id=\"1\"><infon key=\"type\">Protein</infon>"
            b"<infon key=\"identifier\">P00001</infon><text>Protein X</text>"
            b"<location offset=\"999\" length=\"9\"/></annotation>"
            b"</sentence></passage></document></collection>"
        )
        result = parse_bioc_xml(sentence)
        self.assertEqual(result["normalized_entities"][0]["mentions"], [])
        self.assertEqual(result["rejected_entity_location_count"], 1)

    def test_rejects_entity_location_when_source_text_disagrees_with_annotation_mention(self):
        payload = BIOC.replace(b'<text>APOE</text><location offset="87" length="4"/>',
                                b'<text>APP</text><location offset="87" length="4"/>')
        result = parse_bioc_xml(payload)
        self.assertEqual(result["parser_version"], "0.2.11")
        self.assertEqual(result["normalized_entities"][0]["identifier"], "NCBI:348")
        self.assertEqual(result["normalized_entities"][0]["mentions"], [])
        self.assertEqual(result["rejected_entity_location_count"], 1)

    def test_retains_bounded_discontinuous_entity_locations(self):
        payload = (
            b"<collection><document><id>PMC779</id><passage><offset>0</offset>"
            b"<text>APOE interacts with protein X in cells.</text>"
            b"<annotation><infon key=\"type\">Protein</infon>"
            b"<infon key=\"identifier\">P00001</infon><text>APOE protein</text>"
            b"<location offset=\"0\" length=\"4\"/>"
            b"<location offset=\"20\" length=\"7\"/></annotation>"
            b"</passage></document></collection>"
        )
        result = parse_bioc_xml(payload)
        mentions = result["normalized_entities"][0]["mentions"]
        self.assertEqual([mention["text"] for mention in mentions], ["APOE", "protein"])
        self.assertEqual(result["rejected_entity_location_count"], 0)

    def test_rejects_entity_locations_when_source_base_offset_is_missing(self):
        sentence = (
            b"<collection><document><id>PMC999</id><passage>"
            b"<infon key=\"section_type\">RESULTS</infon>"
            b"<sentence><text>Protein X increased in the study and met the preset threshold.</text>"
            b"<annotation id=\"1\"><infon key=\"type\">Protein</infon>"
            b"<infon key=\"identifier\">P00001</infon><text>Protein X</text>"
            b"<location offset=\"100\" length=\"9\"/></annotation>"
            b"</sentence></passage></document></collection>"
        )
        result = parse_bioc_xml(sentence)
        self.assertEqual(result["normalized_entities"][0]["mentions"], [])
        self.assertEqual(result["rejected_entity_location_count"], 1)

    def test_rejects_negative_and_empty_entity_locations(self):
        for offset, length in (("-1", "9"), ("100", "0")):
            payload = BIOC.replace(b'offset="87" length="4"', f'offset="{offset}" length="{length}"'.encode())
            with self.subTest(offset=offset, length=length):
                result = parse_bioc_xml(payload)
                self.assertEqual(result["normalized_entities"][0]["mentions"], [])
                self.assertEqual(result["rejected_entity_location_count"], 1)

    def test_rejects_dtd_bad_xml_wrong_root_and_oversized_input(self):
        utf16_internal_entity = (
            '<?xml version="1.0" encoding="UTF-16"?>'
            '<!DOCTYPE collection [<!ENTITY e "expanded">]>'
            '<collection><document><id>PMC999</id><passage><text>&e;</text></passage></document></collection>'
        ).encode("utf-16")
        for payload in (b"<!DOCTYPE x [<!ENTITY e 'boom'>]><collection/>", utf16_internal_entity, b"<collection>", b"<notBioc/>"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_bioc_xml(payload)
        with self.assertRaises(ValueError):
            parse_bioc_xml(b" " * 8_000_001)

    def test_groups_objective_population_dataset_discussion_and_conclusion_verbatim(self):
        payload = (
            b"<collection><document><id>PMC880</id>"
            b"<passage><infon key=\"section_type\">OBJECTIVE</infon><offset>0</offset>"
            b"<text>We aimed to characterize the sampled cohort.</text></passage>"
            b"<passage><infon key=\"section_type\">STUDY POPULATION</infon><offset>47</offset>"
            b"<text>Participants were recruited across three sites.</text></passage>"
            b"<passage><infon key=\"section_type\">DATA AVAILABILITY</infon><offset>97</offset>"
            b"<text>Processed data are available in repository X.</text></passage>"
            b"<passage><infon key=\"section_type\">DISCUSSION</infon><offset>148</offset>"
            b"<text>These findings may reflect the selected population.</text></passage>"
            b"<passage><infon key=\"section_type\">CONCLUSIONS</infon><offset>204</offset>"
            b"<text>Further validation is required.</text></passage>"
            b"</document></collection>"
        )
        result = parse_bioc_xml(payload)
        self.assertEqual(result["objectives"], "We aimed to characterize the sampled cohort.")
        self.assertEqual(result["population_sections"], "Participants were recruited across three sites.")
        self.assertEqual(result["dataset_sections"], "Processed data are available in repository X.")
        self.assertEqual(result["discussion"], "These findings may reflect the selected population.")
        self.assertEqual(result["conclusions"], "Further validation is required.")
        self.assertEqual(result["sections"][1]["source_section"], "STUDY POPULATION")
        self.assertIn("verbatim source content", " ".join(result["extraction_limitations"]))

    def test_retention_receipt_hashes_all_full_text_groups_without_mutating_transient_record(self):
        extra = (
            b'<passage><infon key="section_type">OBJECTIVE</infon><offset>210</offset>'
            b"<text>Private objective passage that must not persist.</text></passage>"
            b'<passage><infon key="section_type">STUDY POPULATION</infon><offset>260</offset>'
            b"<text>Private population passage that must not persist.</text></passage>"
            b'<passage><infon key="section_type">DATA AVAILABILITY</infon><offset>310</offset>'
            b"<text>Private dataset passage that must not persist.</text></passage>"
            b"</document>"
        )
        parsed = parse_bioc_xml(BIOC.replace(b"</document></collection>", extra + b"</collection>"))
        receipt = build_extraction_retention_receipt(parsed)
        encoded = json.dumps(receipt, ensure_ascii=False)
        for text in (
            parsed["abstract"], parsed["methods"], parsed["results"], parsed["limitations"],
            parsed["sections"][1]["text"], parsed["result_sentence_candidates"][0]["text"],
            parsed["objectives"], parsed["population_sections"], parsed["dataset_sections"],
        ):
            self.assertNotIn(text, encoded)
        self.assertEqual(parsed["normalized_entities"][0]["mentions"][0]["text"], "APOE")
        self.assertEqual(len(receipt["sections"]), len(parsed["sections"]))
        self.assertEqual(len(receipt["result_sentence_candidates"]), len(parsed["result_sentence_candidates"]))
        self.assertTrue(all("text_sha256" in row for row in receipt["sections"]))
        self.assertTrue(all("text_sha256" in row for row in receipt["result_sentence_candidates"]))

    def test_sanitized_mcp_receipt_remains_sufficient_for_worker_structural_receipt(self):
        parsed = parse_bioc_xml(BIOC)
        retained = build_extraction_retention_receipt(parsed)
        worker_receipt = structure_pmc_receipt(
            retained,
            pmid="123456",
            pmc_id="PMC123456",
            source_artifact_id="paper-artifact",
            source_artifact_sha256="a" * 64,
            extraction_artifact_id="extraction-artifact",
        )
        self.assertEqual(worker_receipt["result_sentence_candidate_count"], 2)
        self.assertEqual(worker_receipt["normalized_entities"][0]["identifier"], "NCBI:348")
        self.assertNotIn("APOE was associated", json.dumps(worker_receipt))

    def test_limits_pmc_identifiers(self):
        self.assertEqual(PMCBioCClient.validate_identifier("pmc123"), "PMC123")
        self.assertEqual(PMCBioCClient.validate_identifier("PMID:123"), "123")
        for value in ("PMC0", "PMC123/../../x", "https://example.org/PMC123", "0", "123"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PMCBioCClient.validate_identifier(value)


class PMCBioCTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_official_bioc_endpoint_and_parses_returned_document(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/PMC123456/unicode")
            return httpx.Response(200, content=BIOC, headers={"content-type": "application/xml"})

        factory = lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
        client = PMCBioCClient(client_factory=factory)
        result = await client.fetch("PMC123456")
        self.assertEqual(result["document_id"], "PMC123456")
