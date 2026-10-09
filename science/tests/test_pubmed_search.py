from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET

from medai_science.sources.pubmed_search import add_fetch_coverage, parse_esearch_response, parse_related_publications, validate_pmid_snapshot


class PubMedSearchCoverageTests(unittest.TestCase):
    def test_watch_snapshot_is_bounded_unique_ascii_and_normalized(self):
        self.assertEqual(validate_pmid_snapshot(None), [])
        self.assertEqual(validate_pmid_snapshot(["001", "2"]), ["001", "2"])
        for invalid in (["1", "1"], ["١"], ["1" * 21], [str(value) for value in range(201)]):
            with self.subTest(invalid=invalid[:2]), self.assertRaises(ValueError):
                validate_pmid_snapshot(invalid)

    def test_reports_total_hits_and_bounded_search_truncation(self):
        raw = json.dumps({"esearchresult": {"count": "248", "idlist": ["101", "102"]}}).encode()
        identifiers, coverage = parse_esearch_response(raw, 2)
        self.assertEqual(identifiers, ["101", "102"])
        self.assertEqual(coverage["total_hits"], 248)
        self.assertTrue(coverage["search_truncated"])

    def test_separates_search_truncation_from_records_omitted_by_efetch(self):
        base = {"total_hits": 2, "requested_limit": 2, "returned_pmids": 2, "search_truncated": False}
        coverage = add_fetch_coverage(base, ["101", "102"], ["101"])
        self.assertEqual(coverage["missing_record_pmids"], ["102"])
        self.assertTrue(coverage["fetch_incomplete"])
        self.assertTrue(coverage["retrieval_incomplete"])

    def test_rejects_malformed_counts_ids_and_unrequested_efetch_records(self):
        for count, identifiers in (("x", []), ("1", ["101", "102"]), ("2", ["١٢٣"]), ("1", ["101", "101"])):
            raw = json.dumps({"esearchresult": {"count": count, "idlist": identifiers}}).encode()
            with self.subTest(count=count, identifiers=identifiers), self.assertRaises(ValueError):
                parse_esearch_response(raw, 2)
        with self.assertRaisesRegex(ValueError, "absent from the ESearch"):
            add_fetch_coverage({"search_truncated": False}, ["101"], ["102"])

    def test_rejects_duplicate_keys_and_nonstandard_numbers_in_provider_json(self):
        for raw in (
            b'{"esearchresult":{"count":"1","count":"2","idlist":["101"]}}',
            b'{"esearchresult":{"count":"1","idlist":["101"],"idlist":[]}}',
            b'{"esearchresult":{"count":"1","idlist":["101"],"score":NaN}}',
            b'{"esearchresult":{"count":"1","idlist":["101"],"score":Infinity}}',
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "invalid JSON"):
                parse_esearch_response(raw, 2)

    def test_retains_bounded_deduplicated_related_pubmed_links_only_for_valid_pmids(self):
        article = ET.fromstring("""<PubmedArticle><MedlineCitation><CommentsCorrectionsList>
          <CommentsCorrections RefType="RetractionIn"><PMID>456</PMID></CommentsCorrections>
          <CommentsCorrections RefType="RetractionIn"><PMID>456</PMID></CommentsCorrections>
          <CommentsCorrections RefType="ErratumIn"><PMID>789</PMID></CommentsCorrections>
          <CommentsCorrections RefType="RetractionOf"><PMID>١٢٣</PMID></CommentsCorrections>
          <CommentsCorrections RefType="bad value"><PMID>123</PMID></CommentsCorrections>
        </CommentsCorrectionsList></MedlineCitation></PubmedArticle>""")
        self.assertEqual(parse_related_publications(article), [
            {"ref_type": "RetractionIn", "pmid": "456"},
            {"ref_type": "ErratumIn", "pmid": "789"},
        ])

    def test_caps_related_pubmed_link_count(self):
        relations = "".join(f'<CommentsCorrections RefType="CommentIn"><PMID>{index + 1}</PMID></CommentsCorrections>' for index in range(150))
        article = ET.fromstring(f"<PubmedArticle><MedlineCitation><CommentsCorrectionsList>{relations}</CommentsCorrectionsList></MedlineCitation></PubmedArticle>")
        self.assertEqual(len(parse_related_publications(article)), 100)


if __name__ == "__main__":
    unittest.main()
