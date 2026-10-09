from __future__ import annotations

import asyncio
import copy
import unittest

import httpx

from medai_science.sources.intact import IntActClient, build_retention_payload
from medai_science.protein_interactions import reconcile_interactions


def _mitab() -> str:
    columns = [
        "uniprotkb:P05067", "uniprotkb:Q9Y6K9", "-", "-", "gene name:APP", "gene name:ABC1",
        "psi-mi:" + "MI:0018", "author:Example et al.", "pubmed:12345678",
        "taxid:9606(Homo sapiens)", "taxid:9606(Homo sapiens)", "psi-mi:MI:0915(physical association)",
        "psi-mi:MI:0469(intact)", "intact:EBI-123456", "intact-miscore:0.72",
    ]
    return "\t".join(columns)


class IntActTests(unittest.TestCase):
    def test_reconciliation_rejects_missing_or_malformed_source_row_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact MITAB row"):
            reconcile_interactions([{"interactor_a": "uniprotkb:P05067", "interactor_b": "uniprotkb:Q9Y6K9"}])
        with self.assertRaisesRegex(ValueError, "exact MITAB row"):
            reconcile_interactions([{"source_record_sha256": "not-a-digest"}])
        with self.assertRaisesRegex(ValueError, "nonempty valid MITAB"):
            reconcile_interactions([{"raw_mitab": ""}])

    def test_parses_taxa_references_method_confidence_and_raw_record(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "www.ebi.ac.uk")
            self.assertIn("id:P05067", request.url.path)
            self.assertEqual(request.url.params["format"], "tab")
            return httpx.Response(200, text=_mitab() + "\n")

        result = asyncio.run(IntActClient(transport=httpx.MockTransport(handler)).interactions("p05067"))
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["pubmed_ids"], "pubmed:12345678")
        self.assertEqual(result["records"][0]["interaction_ids"], "intact:EBI-123456")
        self.assertEqual(result["records"][0]["organism_a"], "taxid:9606(Homo sapiens)")
        self.assertEqual(len(result["response_sha256"]), 64)
        self.assertEqual(result["records"][0]["confidence"], "intact-miscore:0.72")
        self.assertEqual(result["reconciliation"]["interaction_count"], 1)
        self.assertEqual(result["reconciliation"]["interactions"][0]["unique_publication_count"], 1)
        self.assertEqual(result["retrieval_coverage"]["source_rows_seen"], 1)
        self.assertFalse(result["retrieval_coverage"]["may_be_truncated_at_requested_limit"])

    def test_retention_drops_verbatim_mitab_and_aliases_but_keeps_row_hash(self) -> None:
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=_mitab() + "\n")))
        result = asyncio.run(client.interactions("P05067"))
        retained = build_retention_payload(result)
        row = retained["records"][0]
        self.assertNotIn("raw_mitab", row)
        self.assertNotIn("aliases_a", row)
        self.assertEqual(row["source_record_sha256"], result["reconciliation"]["interactions"][0]["publication_evidence"][0]["source_record_hashes"][0])
        self.assertEqual(retained["response_sha256"], result["response_sha256"])
        self.assertIn("not been independently adjudicated", retained["retention_policy"])

    def test_retention_rejects_inconsistent_source_identity_and_reconciliation(self) -> None:
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=_mitab() + "\n")))
        result = asyncio.run(client.interactions("P05067"))
        bad_url = copy.deepcopy(result)
        bad_url["source_url"] = "https://example.invalid/"
        with self.assertRaisesRegex(ValueError, "source URL"):
            build_retention_payload(bad_url)
        bad_reconciliation = copy.deepcopy(result)
        bad_reconciliation["reconciliation"]["interaction_count"] = 0
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            build_retention_payload(bad_reconciliation)
        bad_coverage = copy.deepcopy(result)
        bad_coverage["retrieval_coverage"]["retained_rows"] = True
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_retention_payload(bad_coverage)

    def test_reconciles_unordered_pairs_by_stable_ids_taxon_and_publication_without_scoring(self) -> None:
        original = _mitab().split("\t")
        swapped = list(original)
        for left, right in ((0, 1), (2, 3), (4, 5), (9, 10)):
            swapped[left], swapped[right] = original[right], original[left]
        swapped[6] = "psi-mi:MI:0019( coimmunoprecipitation )"
        swapped[8] = "pubmed:87654321"
        duplicate = "\t".join(original)
        rows = "\n".join((_mitab(), "\t".join(swapped), duplicate)) + "\n"
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=rows)))
        result = asyncio.run(client.interactions("P05067"))
        reconciliation = result["reconciliation"]
        self.assertEqual(reconciliation["input_record_count"], 3)
        self.assertEqual(reconciliation["interaction_count"], 1)
        interaction = reconciliation["interactions"][0]
        self.assertEqual(interaction["record_count"], 3)
        self.assertEqual(interaction["exact_duplicate_row_count"], 1)
        self.assertEqual(interaction["unique_intact_record_ids"], ["EBI-123456"])
        self.assertEqual(interaction["unique_publications"], ["12345678", "87654321"])
        self.assertTrue(interaction["annotation_heterogeneity"]["multiple_detection_methods"])
        self.assertEqual(interaction["raw_source_confidence_values"], ["intact-miscore:0.72"])
        self.assertIn("not calibrated probabilities", interaction["confidence_interpretation"])
        self.assertEqual(len(interaction["publication_evidence"]), 2)
        self.assertEqual(reconciliation["source_identity_conflict_count"], 0)

    def test_flags_source_identifier_reused_for_a_different_pair(self) -> None:
        first = _mitab().split("\t")
        second = list(first)
        second[1] = "uniprotkb:Q8N158"
        text = "\n".join(("\t".join(first), "\t".join(second)))
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=text)))
        result = asyncio.run(client.interactions("P05067"))["reconciliation"]
        self.assertEqual(result["interaction_count"], 2)
        self.assertEqual(result["source_identity_conflict_count"], 1)
        conflict = result["source_identity_conflicts"][0]
        self.assertEqual(conflict["intact_record_id"], "EBI-123456")
        self.assertEqual(len(conflict["interaction_ids"]), 2)
        self.assertIn("review source records", conflict["interpretation"])

    def test_unresolved_participants_are_reported_without_name_based_merging(self) -> None:
        row = _mitab().split("\t")
        row[0] = "intact:EBI-123|gene name:APP"
        result = asyncio.run(IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="\t".join(row)))).interactions("P05067"))
        self.assertEqual(result["reconciliation"]["interaction_count"], 0)
        self.assertEqual(result["reconciliation"]["unresolved_record_count"], 1)

    def test_ambiguous_preferred_participant_ids_are_not_lexically_resolved(self) -> None:
        row = _mitab().split("\t")
        row[0] = "uniprotkb:Q9Y6K9|uniprotkb:P05067|ensembl:ENSP00000354587"
        result = asyncio.run(IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="\t".join(row)))).interactions("P05067"))
        reconciliation = result["reconciliation"]
        self.assertEqual(reconciliation["interaction_count"], 0)
        self.assertEqual(reconciliation["unresolved_record_count"], 1)
        unresolved = reconciliation["unresolved_records"][0]
        self.assertEqual(unresolved["reason"], "ambiguous_preferred_stable_participant_identifiers")
        self.assertEqual(unresolved["ambiguous_interactor_candidates"], {"interactor_a": ["uniprotkb:P05067", "uniprotkb:Q9Y6K9"]})
        retained = build_retention_payload(result)
        self.assertEqual(retained["reconciliation"]["unresolved_records"], reconciliation["unresolved_records"])

    def test_reconciliation_keeps_biological_roles_attached_when_participants_are_swapped(self) -> None:
        first = _mitab().split("\t") + [
            "-", "psi-mi:MI:0684(ancillary)", "psi-mi:MI:0499(unspecified role)",
        ]
        swapped = list(first)
        for left, right in ((0, 1), (2, 3), (4, 5), (9, 10), (16, 17)):
            swapped[left], swapped[right] = first[right], first[left]
        inverse_roles = list(first)
        inverse_roles[16], inverse_roles[17] = first[17], first[16]
        text = "\n".join("\t".join(row) for row in (first, swapped, inverse_roles))
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=text)))
        result = asyncio.run(client.interactions("P05067"))["reconciliation"]
        self.assertEqual(result["interaction_count"], 2)
        self.assertEqual(sorted(item["record_count"] for item in result["interactions"]), [1, 2])
        paired = next(item for item in result["interactions"] if item["record_count"] == 2)
        self.assertEqual([participant["biological_roles"] for participant in paired["participants"]], [["MI:0684"], ["MI:0499"]])

    def test_reconciles_homomer_records_when_role_annotated_endpoints_are_swapped(self) -> None:
        first = _mitab().split("\t") + ["-", "psi-mi:MI:0684(ancillary)", "psi-mi:MI:0499(unspecified role)"]
        first[1] = first[0]
        swapped = list(first)
        swapped[16], swapped[17] = first[17], first[16]
        rows = "\n".join(("\t".join(first), "\t".join(swapped)))
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=rows)))
        result = asyncio.run(client.interactions("P05067"))["reconciliation"]
        self.assertEqual(result["interaction_count"], 1)
        interaction = result["interactions"][0]
        self.assertEqual(interaction["record_count"], 2)
        self.assertEqual([participant["biological_roles"] for participant in interaction["participants"]], [["MI:0499"], ["MI:0684"]])

    def test_species_filter_does_not_silently_claim_cross_species_record_as_human(self) -> None:
        row = _mitab().replace("taxid:9606(Homo sapiens)", "taxid:10090(Mus musculus)", 1)
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=row)))
        result = asyncio.run(client.interactions("P05067"))
        self.assertEqual(result["records"], [])
        self.assertEqual(result["retrieval_coverage"]["rows_filtered_by_organism"], 1)
        self.assertEqual(result["retrieval_coverage"]["source_rows_seen"], 1)

    def test_marks_possible_truncation_and_rejects_invalid_utf8(self) -> None:
        one_row = _mitab() + "\n"
        client = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=one_row)))
        result = asyncio.run(client.interactions("P05067", limit=1))
        self.assertTrue(result["retrieval_coverage"]["may_be_truncated_at_requested_limit"])
        self.assertEqual(result["retrieval_coverage"]["retained_rows"], 1)
        malformed = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"\xff")))
        with self.assertRaisesRegex(ValueError, "invalid UTF-8"):
            asyncio.run(malformed.interactions("P05067"))

    def test_rejects_invalid_accession_limit_and_malformed_mitab(self) -> None:
        with self.assertRaisesRegex(ValueError, "UniProt accession"):
            asyncio.run(IntActClient().interactions("P05067/path"))
        with self.assertRaisesRegex(ValueError, "limit"):
            asyncio.run(IntActClient().interactions("P05067", 501))
        malformed = IntActClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text="a\tb\tc")))
        with self.assertRaisesRegex(ValueError, "malformed MITAB"):
            asyncio.run(malformed.interactions("P05067"))


if __name__ == "__main__":
    unittest.main()
