import unittest

import httpx

from medai_science.paper_understanding import _sentence_spans, parse_bioc_xml
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
