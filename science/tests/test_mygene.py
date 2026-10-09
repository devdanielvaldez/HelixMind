from __future__ import annotations

import unittest

import httpx

from medai_science.sources.mygene import MyGeneClient


class MyGeneTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_preserves_exact_stable_record_ids(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "mygene.info")
            self.assertEqual(request.url.params["q"], "APOE")
            self.assertEqual(request.url.params["species"], "human")
            self.assertEqual(request.url.params["size"], "2")
            return httpx.Response(200, json={"total": 1, "hits": [{"_id": "348", "symbol": "APOE", "taxid": 9606}]})

        result = await MyGeneClient(transport=httpx.MockTransport(handler)).search("APOE", limit=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_id"], "348")
        self.assertEqual(result[0]["record"]["symbol"], "APOE")

    async def test_rejects_ambiguous_json_and_malformed_hit_collections(self):
        for payload, message in (
            (b'{"hits":[],"hits":[]}', "duplicate JSON object key"),
            (b'{"hits":[{"_id":348,"score":NaN}]}', "non-standard JSON number"),
            (b'{"hits":null}', "hit collection"),
            (b'{"hits":[null]}', "malformed hit"),
            (b'{"hits":[{"symbol":"APOE"}]}', "stable source ID"),
        ):
            client = MyGeneClient(transport=httpx.MockTransport(lambda _, body=payload: httpx.Response(200, content=body)))
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                await client.search("APOE", limit=2)

    async def test_rejects_boolean_limits_and_non_string_species(self):
        client = MyGeneClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"hits": []})))
        with self.assertRaisesRegex(ValueError, "limit"):
            await client.search("APOE", limit=True)
        with self.assertRaisesRegex(ValueError, "species"):
            await client.search("APOE", species=["human"])


if __name__ == "__main__":
    unittest.main()
