from __future__ import annotations

import unittest

from medai_science.contradiction_search import build_contradiction_query, select_contradiction_claims


def evidence(evidence_id, source_id, claim, polarity="supports"):
    return {"id": evidence_id, "source": "PubMed", "source_id": source_id, "claim": claim, "polarity": polarity}


class ContradictionSearchTests(unittest.TestCase):
    def test_selects_deterministically_and_spreads_across_source_records(self):
        rows = [
            evidence("E4", "PMID4", "TREM2 microglia pathway effect"),
            evidence("E2", "PMID1", "APOE lipid metabolism association"),
            evidence("E1", "PMID1", "APOE lipid metabolism association"),
            evidence("E3", "PMID3", "APP amyloid clearance relationship"),
            evidence("E5", "PMID5", "not a supporting record", "context"),
            {"id": "E6", "source": "PubMed", "source_id": "PMID6", "claim": "missing polarity"},
        ]
        selected = select_contradiction_claims(rows, limit=3)
        self.assertEqual([row["source_id"] for row in selected], ["PMID1", "PMID3", "PMID4"])
        self.assertEqual(select_contradiction_claims(list(reversed(rows)), limit=3), selected)

    def test_query_is_safe_bounded_and_exactly_reproducible(self):
        claim = "APOE-associated lipid metabolism may alter microglial TREM2 response. " + ("(untrusted ' OR ) " * 80)
        query = build_contradiction_query(claim)
        self.assertLessEqual(len(query.encode("utf-8")), 256)
        self.assertTrue(query.startswith("(") and query.endswith('"no association" OR null)'))
        self.assertNotIn("untrusted", query)
        self.assertNotIn("'", query)

    def test_rejects_patient_identifiers_and_claims_without_searchable_terms(self):
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            build_contradiction_query("MRN: ABC123; APOE and disease")
        with self.assertRaisesRegex(ValueError, "no searchable"):
            build_contradiction_query("This could have more within which")


if __name__ == "__main__":
    unittest.main()
