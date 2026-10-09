from __future__ import annotations

import asyncio
import unittest

import httpx

from medai_science.sources.pubmed import PubMedClient


class PubMedClientLoopLifecycleTests(unittest.TestCase):
    def test_client_can_be_created_after_a_closed_loop_and_reused_on_new_loops(self):
        asyncio.run(asyncio.sleep(0))
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, content=b'{"esearchresult":{"count":"1","idlist":["123"]}}')

        client = PubMedClient("research@example.org", transport=httpx.MockTransport(handler))

        async def search():
            return await client.search_recent_ids("APOE", window_days=7, limit=1)

        first = asyncio.run(search())
        second = asyncio.run(search())
        self.assertEqual(first, (1, ["123"]))
        self.assertEqual(second, first)
        self.assertEqual(calls, 2)


class PubMedRecentSearchTests(unittest.IsolatedAsyncioTestCase):

    async def test_edat_search_uses_strict_bounded_esearch_contract(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/entrez/eutils/esearch.fcgi")
            self.assertEqual(request.url.params["datetype"], "edat")
            self.assertEqual(request.url.params["reldate"], "30")
            self.assertEqual(request.url.params["retmax"], "2")
            return httpx.Response(200, content=b'{"esearchresult":{"count":"241","idlist":["101","202"]}}')

        client = PubMedClient("research@example.org", transport=httpx.MockTransport(handler))
        total, identifiers = await client.search_recent_ids("APOE microglia", window_days=30, limit=2)
        self.assertEqual(total, 241)
        self.assertEqual(identifiers, ["101", "202"])

    async def test_edat_search_rejects_duplicate_keys_nonstandard_numbers_and_overflow(self):
        invalid_responses = (
            b'{"esearchresult":{"count":"1","count":"2","idlist":["1"]}}',
            b'{"esearchresult":{"count":"1","idlist":["1"],"extra":NaN}}',
            b'{"esearchresult":{"count":"1","idlist":["1"],"extra":Infinity}}',
            b'{"esearchresult":{"count":"999999999999999999999","idlist":["1"]}}',
            b'{"esearchresult":{"count":"1","idlist":["1","1"]}}',
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw):
                client = PubMedClient("research@example.org", transport=httpx.MockTransport(lambda _: httpx.Response(200, content=raw)))
                with self.assertRaises(ValueError):
                    await client.search_recent_ids("APOE", window_days=7, limit=2)


if __name__ == "__main__":
    unittest.main()
