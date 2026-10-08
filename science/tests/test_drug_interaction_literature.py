import unittest

from medai_science.sources.drug_interaction_literature import build_interaction_evidence_query, build_interaction_literature_query, candidate_retention_receipt, characterize_interaction_candidate, retain_candidate_paper


class DrugInteractionLiteratureTests(unittest.TestCase):
    def test_query_uses_two_quoted_title_abstract_phrases(self):
        query = build_interaction_literature_query("acetylsalicylic acid", "warfarin sodium")
        self.assertEqual(query, '"acetylsalicylic acid"[Title/Abstract] AND "warfarin sodium"[Title/Abstract]')

    def test_query_cannot_be_rewritten_by_pubmed_operators_or_identifiers(self):
        query = build_interaction_literature_query('aspirin OR cancer[All Fields]', "warfarin")
        self.assertEqual(query, '"aspirin OR cancer All Fields"[Title/Abstract] AND "warfarin"[Title/Abstract]')
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            build_interaction_literature_query("patient MRN 123456789", "warfarin")
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_interaction_literature_query("aspirin", "ASPIRIN")

    def test_evidence_query_bounds_resolved_synonyms_and_requires_pair_plus_pkpd_terms(self):
        query = build_interaction_evidence_query(
            ["warfarin", "warfarin sodium", "warfarin", "ignored 4", "ignored 5", "ignored 6", "ignored 7"],
            ["fluconazole", "diflucan"],
        )
        self.assertIn('"warfarin sodium"[Title/Abstract]', query)
        self.assertIn('"diflucan"[Title/Abstract]', query)
        self.assertIn("pharmacokinetic*[Title/Abstract]", query)
        self.assertIn('"drug interaction"[Title/Abstract]', query)
        self.assertNotIn('"ignored 7"', query)
        with self.assertRaisesRegex(ValueError, "searchable name"):
            build_interaction_evidence_query([], ["warfarin"])

    def test_candidate_characterization_is_a_retrieval_signal_not_an_evidence_grade(self):
        result = characterize_interaction_candidate({
            "title": "Pharmacokinetic interaction study",
            "abstract": "Co-administration changed AUC and maximum concentration.",
            "publication_types": ["Clinical Trial", "Randomized Controlled Trial", "Editorial"],
            "mesh_headings": [{"descriptor": "Drug Interactions", "descriptor_ui": "D004347"}],
        })
        self.assertEqual(result["publication_type_signals"], ["clinical trial", "randomized controlled trial"])
        self.assertIn("auc", result["pkpd_or_outcome_terms"])
        self.assertIn("maximum concentration", result["pkpd_or_outcome_terms"])
        self.assertTrue(result["review_required"])
        self.assertIn("do not establish", result["interpretation_limit"])

    def test_retained_candidate_is_pmid_linked_and_bounded(self):
        result = retain_candidate_paper({
            "provenance": {"source_id": "12345678", "title": "Interaction study"},
            "abstract": "A" * 31_000,
            "authors": ["A Author"],
            "publication_date": "2024",
            "publication_types": ["Clinical Trial"],
            "publication_type_ids": ["D016430"],
            "mesh_headings": [{"descriptor": "Drug Interactions", "descriptor_ui": "D004347", "qualifiers": []}],
            "journal": "Journal",
            "identifiers": {"doi": "10.1000/example", "pmc": "PMC123", "ignored": "secret"},
        })
        self.assertEqual(result["url"], "https://pubmed.ncbi.nlm.nih.gov/12345678/")
        self.assertEqual(len(result["abstract"]), 30_000)
        self.assertEqual(result["identifiers"], {"doi": "10.1000/example", "pmc": "PMC123"})
        self.assertEqual(result["publication_type_ids"], ["D016430"])
        self.assertEqual(result["mesh_headings"][0]["descriptor_ui"], "D004347")
        self.assertIn("does not establish", result["interpretation_limit"])
        result["unreviewed_free_text"] = "do not persist"
        receipt = candidate_retention_receipt(result)
        self.assertNotIn("abstract", receipt)
        self.assertNotIn("unreviewed_free_text", receipt)
        self.assertEqual(receipt["abstract_retention"]["characters"], 30_000)
        self.assertEqual(len(receipt["abstract_retention"]["sha256"]), 64)
        self.assertFalse(receipt["abstract_retention"]["retained"])
        self.assertEqual(receipt["abstract_retention"]["license_status"], "undetermined")
        with self.assertRaisesRegex(ValueError, "numeric PMID"):
            retain_candidate_paper({"provenance": {"source_id": "NCT123"}})


if __name__ == "__main__":
    unittest.main()
