import unittest

from medai_science.literature import exact_mygene_identifier_matches, exact_uniprot_gene_matches, omit_pubmed_abstract_text, pubchem_cid_candidates_from_extraction_receipts, pubchem_cids_from_extraction_receipts, resolved_gene_symbols, source_entity_node, structure_pmc_receipt, structure_pubmed_abstract


class PubMedAbstractStructureTests(unittest.TestCase):
    def test_compound_discovery_uses_only_source_normalized_pubchem_cids(self):
        artifacts = [{
            "id": "extract-1",
            "kind": "paper_extraction_receipt",
            "payload": {"normalized_entities": [
                {"database": "PubChem", "entity_type": "Chemical", "identifier": "CID:2244"},
                {"database": "PubChem", "entity_type": "Drug", "identifier": "2244"},
                {"database": "PubChem", "entity_type": "Chemical", "identifier": "aspirin"},
                {"database": "MeSH", "entity_type": "Chemical", "identifier": "2244"},
                    {"database": "PubChem", "entity_type": "Gene", "identifier": "42"},
                    {"database": "PubChem", "entity_type": "Compound", "identifier": "3152"},
            ]},
        }]
        self.assertEqual(pubchem_cids_from_extraction_receipts(artifacts), [{"cid": "2244", "source_artifact_id": "extract-1"}, {"cid": "3152", "source_artifact_id": "extract-1"}])
        self.assertEqual(pubchem_cids_from_extraction_receipts(artifacts, exclude={"2244"}), [{"cid": "3152", "source_artifact_id": "extract-1"}])
        self.assertEqual(pubchem_cid_candidates_from_extraction_receipts(artifacts, limit=2)[1], False)
        self.assertEqual(pubchem_cid_candidates_from_extraction_receipts(artifacts, limit=1)[1], True)

    def test_preserves_provider_section_labels_and_does_not_infer_claims_or_entities(self):
        paper = {
            "provenance": {"source": "PubMed", "source_id": "12345", "title": "Example"},
            "identifiers": {"doi": "10.1234/example", "pmc": "PMC999", "secret": "discard"},
            "abstract_sections": [
                {"source_label": "OBJECTIVE", "text": "Study objective."},
                {"source_label": "Methods", "text": "A randomized study."},
                {"source_label": "Results", "text": "An observed result."},
                {"source_label": "Unmapped provider label", "text": "Remain unknown."},
                {"source_label": None, "text": "Keep this unclassified."},
            ],
        }
        result = structure_pubmed_abstract(paper, "ART-1", "a" * 64)
        self.assertEqual([item["normalized_section"] for item in result["sections"]], ["objective", "methods", "results", "unclassified", "unclassified"])
        self.assertEqual(result["identifiers"], {"doi": "10.1234/example", "pmc": "PMC999"})
        self.assertEqual(result["publication_metadata"]["publication_types"], [])
        self.assertEqual(result["publication_metadata"]["mesh_headings"], [])
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["source_artifact_sha256"], "a" * 64)
        self.assertNotIn("text", result["sections"][0])
        self.assertEqual(result["sections"][0]["text_length"], len("Study objective."))

    def test_unlabeled_abstract_stays_unclassified(self):
        result = structure_pubmed_abstract({"provenance": {"source": "PubMed", "source_id": "7"}, "abstract": "Findings support an idea."}, "ART-2", "b" * 64)
        self.assertEqual(result["sections"][0]["normalized_section"], "unclassified")
        self.assertEqual(result["sections"][0]["text_length"], len("Findings support an idea."))

    def test_abstract_content_is_omitted_but_digest_and_section_labels_are_retained(self):
        text = "Copyrighted abstract sentence."
        paper = {"provenance": {"source": "PubMed", "source_id": "77"}, "abstract": text, "abstract_sections": [{"source_label": "RESULTS", "text": text}], "identifiers": {}}
        result = omit_pubmed_abstract_text(paper)
        self.assertEqual(result["abstract"], "")
        self.assertEqual(result["abstract_sections"], [])
        self.assertNotIn(text, str(result))
        self.assertEqual(result["abstract_length"], len(text))
        structured = structure_pubmed_abstract(result, "ART-77", "a" * 64)
        self.assertEqual(structured["sections"][0]["normalized_section"], "results")
        self.assertNotIn("text", structured["sections"][0])

    def test_rejects_non_pubmed_or_invalid_source_identity(self):
        with self.assertRaisesRegex(ValueError, "not a PubMed"):
            structure_pubmed_abstract({"provenance": {"source": "Crossref", "source_id": "10.1/example"}}, "ART", "c" * 64)
        with self.assertRaisesRegex(ValueError, "PMID"):
            structure_pubmed_abstract({"provenance": {"source": "PubMed", "source_id": "PMC12"}}, "ART", "c" * 64)

    def test_pmc_receipt_keeps_structure_and_identifiers_without_article_text(self):
        parsed = {
            "source_sha256": "d" * 64,
            "source_license": "CC BY 4.0",
            "sections": [{"passage_index": 0, "section": "results", "source_section": "RESULTS", "offset": 10, "text": "Do not retain this passage."}],
            "normalized_entities": [{"identifier": "NCBI:348", "entity_type": "Gene", "database": "NCBI Gene", "mentions": [{"text": "APOE"}]}],
            "result_sentence_candidates": [{"text": "Do not retain this sentence."}],
            "extraction_limitations": ["candidate text is unvalidated"],
        }
        receipt = structure_pmc_receipt(parsed, pmid="12345", pmc_id="PMC765", source_artifact_id="ART-1", source_artifact_sha256="a" * 64, extraction_artifact_id="ART-2")
        serialized = str(receipt)
        self.assertNotIn("Do not retain", serialized)
        self.assertNotIn("APOE", serialized)
        self.assertEqual(receipt["result_sentence_candidate_count"], 1)
        self.assertEqual(receipt["normalized_entities"][0]["identifier"], "NCBI:348")
        self.assertEqual(receipt["source_artifact_sha256"], "a" * 64)

    def test_source_entity_node_requires_provider_ids_and_keeps_receipt_provenance(self):
        node = source_entity_node({"identifier": "NCBI:348", "entity_type": "Gene", "database": "NCBI Gene", "mentions": [{"text": "APOE"}]}, receipt_id="ART-1", receipt_sha256="b" * 64, pmc_id="PMC765")
        self.assertEqual(node["id"], "NCBIGene:348")
        self.assertEqual(node["kind"], "Gene")
        self.assertEqual(node["label"], "NCBI:348")
        self.assertEqual(node["metadata"]["source_artifact_id"], "ART-1")
        self.assertIsNone(source_entity_node({"entity_type": "Gene", "mentions": [{"text": "APOE"}]}, receipt_id="ART", receipt_sha256="x", pmc_id="PMC1"))

    def test_mygene_resolution_accepts_only_exact_entrez_ids(self):
        results = [
            {"source_id": "348", "record": {"entrezgene": 348, "symbol": "APOE"}},
            {"source_id": "999", "record": {"entrezgene": 999, "symbol": "OTHER"}},
            {"source_id": "missing", "record": {"symbol": "APOE"}},
        ]
        matches = exact_mygene_identifier_matches(results, "348")
        self.assertEqual([row["record"]["symbol"] for row in matches], ["APOE"])
        self.assertEqual(exact_mygene_identifier_matches(results, "not-an-id"), [])

    def test_pathway_inputs_require_exact_mygene_artifact_and_safe_symbol(self):
        artifacts = [{"id": "ART-1", "sha256": "a" * 64, "source": "MyGene.info", "payload": {
            "source": "MyGene.info exact NCBI Gene identifier resolution", "ncbi_gene_id": "348", "matches": [
                {"record": {"entrezgene": 348, "symbol": "APOE"}},
                {"record": {"entrezgene": 999, "symbol": "WRONG"}},
                {"record": {"entrezgene": 348, "symbol": "not a safe symbol"}},
            ],
        }}]
        result = resolved_gene_symbols(artifacts, 10)
        self.assertEqual(result, [{"entrezgene": "348", "symbol": "APOE", "source_artifact_id": "ART-1", "source_artifact_sha256": "a" * 64}])

    def test_protein_candidates_require_exact_gene_and_human_taxonomy(self):
        results = [
            {"source_id": "P02649", "record": {"primaryAccession": "P02649", "organism": {"taxonId": 9606}, "genes": [{"geneName": {"value": "APOE"}}]}},
            {"source_id": "P02649", "organism_id": 9606, "record": {"primaryAccession": "P02649", "gene_names": "APOE APOE3", "organism_name": "Homo sapiens"}},
            {"source_id": "Q1", "record": {"primaryAccession": "P12345", "organism": {"taxonId": 10090}, "genes": [{"geneName": {"value": "APOE"}}]}},
            {"source_id": "Q2", "record": {"primaryAccession": "Q11111", "organism": {"taxonId": 9606}, "genes": [{"geneName": {"value": "APOA"}}]}},
        ]
        matches = exact_uniprot_gene_matches(results, "APOE")
        self.assertEqual([row["accession"] for row in matches], ["P02649"])


if __name__ == "__main__":
    unittest.main()
