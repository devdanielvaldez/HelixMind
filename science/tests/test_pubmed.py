import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from medai_science.sources.pubmed import MAX_FETCH_RESPONSE_BYTES, MAX_SEARCH_RESPONSE_BYTES, PubMedClient


class OversizedStream(httpx.AsyncByteStream):
    def __init__(self, size):
        self.size = size

    async def __aiter__(self):
        yield b"x" * self.size

    async def aclose(self):
        return None


class PubMedStructuredMetadataTests(unittest.TestCase):
    def test_pubmed_fetch_ids_require_unique_ascii_strings_before_network_dispatch(self):
        client = PubMedClient("research@example.org")
        invalid_values = (["١٢٣"], [12345], ["12345", "12345"], "12345", [True])
        for operation in (client.fetch, client.fetch_bibliographic):
            for values in invalid_values:
                with self.subTest(operation=operation.__name__, values=values), self.assertRaises(ValueError):
                    asyncio.run(operation(values))

    def test_pubmed_search_limits_reject_booleans_and_noninteger_values(self):
        client = PubMedClient("research@example.org")
        with self.assertRaisesRegex(ValueError, "limit"):
            asyncio.run(client.search_with_coverage("APOE", True))
        with self.assertRaisesRegex(ValueError, "window_days"):
            asyncio.run(client.search_recent_ids("APOE", True))

    def test_all_pubmed_operations_fail_closed_on_provider_redirects(self):
        calls = []

        def redirect(request):
            calls.append(request.url.host)
            return httpx.Response(302, headers={"Location": "https://untrusted.example/redirect"})

        async def exercise():
            client = PubMedClient("research@example.org", transport=httpx.MockTransport(redirect))
            for operation in (
                lambda: client.search_with_coverage("APOE", 1),
                lambda: client.search_recent_ids("APOE", 1, 1),
                lambda: client.fetch(["12345"]),
                lambda: client.fetch_bibliographic(["12345"]),
            ):
                with self.assertRaises(httpx.HTTPStatusError):
                    await operation()

        asyncio.run(exercise())
        self.assertTrue(calls)
        self.assertEqual(set(calls), {"eutils.ncbi.nlm.nih.gov"})

    def test_search_cache_reuses_only_esearch_and_fetches_records_fresh(self):
        calls = []

        def respond(request):
            endpoint = request.url.path.rsplit("/", 1)[-1]
            calls.append(endpoint)
            if endpoint == "esearch.fcgi":
                return httpx.Response(200, json={"esearchresult": {"count": "4", "idlist": ["12345"]}})
            return httpx.Response(200, content=b"<PubmedArticleSet></PubmedArticleSet>")

        async def search_and_fetch(cache_path):
            client = PubMedClient(
                "research@example.org", api_key="test-key", transport=httpx.MockTransport(respond),
                search_cache_path=str(cache_path),
            )
            first = await client.search_with_coverage("APOE[Title/Abstract]", 1)
            second = await client.search_with_coverage("APOE[Title/Abstract]", 1)
            fetched = await client.fetch(second[0])
            return first, second, fetched

        with tempfile.TemporaryDirectory() as directory:
            first, second, fetched = asyncio.run(search_and_fetch(Path(directory) / "search.sqlite"))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1]["total_hits"], second[1]["total_hits"])
        self.assertEqual(second[1]["cache"], "hit")
        self.assertEqual(fetched, [])
        self.assertEqual(calls, ["esearch.fcgi", "efetch.fcgi"])

    def test_search_stops_streaming_when_response_exceeds_byte_limit(self):
        async def search():
            transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=OversizedStream(MAX_SEARCH_RESPONSE_BYTES + 1)))
            return await PubMedClient("research@example.org", transport=transport).search("Alzheimer", 1)

        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(search())

    def test_fetch_rejects_oversized_xml_even_without_content_length(self):
        async def fetch():
            transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=OversizedStream(MAX_FETCH_RESPONSE_BYTES + 1)))
            return await PubMedClient("research@example.org", transport=transport).fetch(["12345"])

        with self.assertRaisesRegex(ValueError, "response exceeds"):
            asyncio.run(fetch())

    def test_recent_search_uses_overlapping_pubmed_entry_date_window(self):
        captured = {}

        def respond(request):
            captured.update(dict(request.url.params))
            return httpx.Response(200, json={"esearchresult": {"count": "205", "idlist": [str(i) for i in range(1, 201)]}})

        async def search():
            client = PubMedClient("research@example.org", transport=httpx.MockTransport(respond))
            return await client.search_recent_ids("Alzheimer disease", 14, 200)

        count, ids = asyncio.run(search())
        self.assertEqual((count, len(ids)), (205, 200))
        self.assertEqual(captured["reldate"], "14")
        self.assertEqual(captured["datetype"], "edat")
        self.assertEqual(captured["sort"], "pub_date")
        self.assertEqual(captured["retmax"], "200")

    def test_recent_search_bounds_window_and_result_limit(self):
        client = PubMedClient("research@example.org")
        with self.assertRaises(ValueError):
            asyncio.run(client.search_recent_ids("public research", 366))
        with self.assertRaises(ValueError):
            asyncio.run(client.search_recent_ids("public research", 2, 201))

    def test_fetch_preserves_abstract_section_labels_and_crossrefable_identifiers(self):
        xml = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><Journal><ISSN IssnType="Electronic">1234-5678</ISSN><JournalIssue CitedMedium="Internet"><Volume>12</Volume><Issue>3</Issue><PubDate><Year>2024</Year></PubDate></JournalIssue><Title>Journal of Example Science</Title><ISOAbbreviation>J Ex Sci</ISOAbbreviation></Journal><ArticleTitle>Structured abstract test</ArticleTitle><Abstract><AbstractText Label="OBJECTIVE" NlmCategory="OBJECTIVE">Test objective.</AbstractText><AbstractText Label="METHODS" NlmCategory="METHODS">Methods details.</AbstractText><AbstractText Label="RESULTS" NlmCategory="RESULTS">Observed findings.</AbstractText><AbstractText>Unlabeled text.</AbstractText></Abstract><AuthorList><Author><LastName>Researcher</LastName><ForeName>Ada</ForeName></Author></AuthorList><Pagination><MedlinePgn>101-110</MedlinePgn></Pagination><Language>eng</Language><PublicationTypeList><PublicationType UI="D016428">Journal Article</PublicationType><PublicationType UI="D016454">Case Reports</PublicationType></PublicationTypeList></Article><CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><PMID Version="1">54321</PMID></CommentsCorrections><CommentsCorrections RefType="ErratumIn"><PMID Version="1">67890</PMID></CommentsCorrections><CommentsCorrections RefType="RetractionIn"><PMID>54321</PMID></CommentsCorrections><CommentsCorrections RefType="RetractionIn"><PMID>not-a-pmid</PMID></CommentsCorrections></CommentsCorrectionsList><MeshHeadingList><MeshHeading><DescriptorName UI="D000544">Alzheimer Disease</DescriptorName><QualifierName UI="Q000235">drug therapy</QualifierName></MeshHeading></MeshHeadingList></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1234/test</ArticleId><ArticleId IdType="pmc">PMC7654321</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'''

        async def load():
            client = PubMedClient("research@example.org", transport=httpx.MockTransport(lambda _: httpx.Response(200, content=xml)))
            return await client.fetch(["12345"])

        papers = asyncio.run(load())
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual([section.source_label for section in paper.abstract_sections], ["OBJECTIVE", "METHODS", "RESULTS", None])
        self.assertEqual(paper.identifiers, {"doi": "10.1234/test", "pmc": "PMC7654321"})
        self.assertIn("Methods details.", paper.abstract)
        self.assertEqual(paper.provenance.source_id, "12345")
        self.assertEqual(paper.publication_types, ["Journal Article", "Case Reports"])
        self.assertEqual(paper.publication_type_ids, ["D016428", "D016454"])
        self.assertEqual(paper.related_publications, [{"ref_type": "RetractionIn", "pmid": "54321"}, {"ref_type": "ErratumIn", "pmid": "67890"}])
        self.assertEqual(paper.mesh_headings[0]["descriptor_ui"], "D000544")
        self.assertEqual(paper.mesh_headings[0]["qualifiers"][0]["ui"], "Q000235")
        self.assertEqual(paper.journal, "Journal of Example Science")
        self.assertEqual(paper.journal_abbreviation, "J Ex Sci")
        self.assertEqual(paper.journal_issns, [{"value": "1234-5678", "type": "Electronic"}])
        self.assertEqual((paper.volume, paper.issue, paper.pages, paper.language), ("12", "3", "101-110", "eng"))

    def test_bibliographic_fetch_uses_esummary_and_omits_abstracts(self):
        captured = {}
        xml = b'''<eSummaryResult><DocSum><Id>12345</Id><Item Name="Title" Type="String">A research title</Item><Item Name="PubDate" Type="Date">2026 Jan 2</Item><Item Name="FullJournalName" Type="String">Journal of Tests</Item><Item Name="ArticleIds" Type="List"><Item Name="pubmed" Type="String">12345</Item><Item Name="doi" Type="String">10.1234/example</Item><Item Name="pmc" Type="String">PMC123456</Item></Item></DocSum></eSummaryResult>'''

        def respond(request):
            captured["path"] = request.url.path
            return httpx.Response(200, content=xml)

        async def load():
            client = PubMedClient("research@example.org", transport=httpx.MockTransport(respond))
            return await client.fetch_bibliographic(["12345"])

        records = asyncio.run(load())
        self.assertTrue(captured["path"].endswith("esummary.fcgi"))
        self.assertEqual(records, [{"pmid": "12345", "title": "A research title", "publication_date": "2026 Jan 2", "journal": "Journal of Tests", "doi": "10.1234/example", "pmcid": "PMC123456"}])


if __name__ == "__main__":
    unittest.main()
