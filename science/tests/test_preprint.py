import unittest

import httpx

from medai_science.sources.preprint import PreprintClient


class PreprintClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_doi_metadata_lookup_is_bounded_and_hash_linked(self):
        async def handler(request):
            self.assertEqual(request.url.host, "api.medrxiv.org")
            self.assertEqual(request.url.path, "/details/medrxiv/10.1101/2020.09.09.20191205/na/json")
            return httpx.Response(200, json={"messages": [{"status": "ok"}], "collection": [{
                "doi": "10.1101/2020.09.09.20191205",
                "title": "Example preprint",
                "authors": "A Author; B Author",
                "date": "2020-09-10",
                "version": "2",
                "type": "new-version",
                "license": "cc_by_nc_nd",
                "category": "neurology",
                "published": "10.1000/journal.123",
                "jatsxml": "https://example.invalid/full.xml",
                "abstract": "Must not be retained",
            }]})

        client = PreprintClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.get_by_doi("medrxiv", "10.1101/2020.09.09.20191205")

        record = result["records"][0]
        self.assertEqual(record["published"], "10.1000/journal.123")
        self.assertEqual(record["license"], "cc_by_nc_nd")
        self.assertFalse(record["abstract_retained"])
        self.assertNotIn("abstract", record)
        self.assertEqual(len(record["source_record_sha256"]), 64)
        self.assertEqual(result["returned_count"], 1)

    async def test_bounded_search_pages_date_slice_and_does_not_retain_abstract(self):
        requests = []

        async def handler(request):
            requests.append(request)
            self.assertEqual(request.url.host, "api.biorxiv.org")
            self.assertEqual(request.url.params["category"], "neuroscience")
            if request.url.path.endswith("/0/json"):
                return httpx.Response(200, json={"collection": [
                    {"doi": f"10.1101/unused.{index}", "title": "No match", "abstract": "unrelated material"}
                    for index in range(30)
                ]})
            return httpx.Response(200, json={"collection": [{
                "doi": "10.1101/match.1",
                "title": "TREM2 and microglia",
                "abstract": "Transient evidence about microglia activation",
                "date": "2026-01-02",
                "license": "cc_by",
            }]})

        client = PreprintClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.search("biorxiv", "TREM2 microglia", "2026-01-01", "2026-01-30", 10, "neuroscience", 2)

        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0].url.path.endswith("/0/json"))
        self.assertTrue(requests[1].url.path.endswith("/30/json"))
        self.assertEqual(result["scanned_count"], 31)
        self.assertTrue(result["scan_exhausted"])
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(result["records"][0]["lexical_screen"]["matched_terms"], ["microglia", "trem2"])
        self.assertNotIn("abstract", result["records"][0])
        self.assertFalse(result["abstracts_retained"])

    async def test_rejects_unsupported_server_and_non_preprint_doi(self):
        client = PreprintClient()
        with self.assertRaisesRegex(ValueError, "server"):
            await client.get_by_doi("arxiv", "10.1101/2020.09.09.20191205")
        with self.assertRaisesRegex(ValueError, "DOI"):
            await client.get_by_doi("biorxiv", "10.1000/not-a-preprint")

    async def test_search_rejects_patient_identifiers_and_unbounded_intervals(self):
        client = PreprintClient()
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            await client.search("biorxiv", "my patient TREM2", "2026-01-01", "2026-01-02")
        with self.assertRaisesRegex(ValueError, "at most 93 days"):
            await client.search("biorxiv", "TREM2", "2026-01-01", "2026-06-01")


if __name__ == "__main__":
    unittest.main()
