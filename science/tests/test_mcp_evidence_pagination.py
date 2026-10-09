from __future__ import annotations

import unittest

import httpx

from medai_science.api_pagination import read_evidence_pages


class EvidencePageReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_reads_complete_session_bound_snapshot_across_cursors(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            cursor = request.url.params.get("after_id")
            if cursor is None:
                return httpx.Response(200, json={
                    "session_id": "RES-1",
                    "items": [{"id": "EVD-1", "session_id": "RES-1"}],
                    "has_more": True,
                    "next_cursor": "EVD-1",
                    "snapshot_rowid": 7,
                })
            return httpx.Response(200, json={
                "session_id": "RES-1",
                "items": [{"id": "EVD-2", "session_id": "RES-1"}],
                "has_more": False,
                "next_cursor": None,
                "snapshot_rowid": 7,
            })

        async with httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(handler)) as client:
            items = await read_evidence_pages(client, "RES-1")

        self.assertEqual([item["id"] for item in items], ["EVD-1", "EVD-2"])
        self.assertEqual(requests[0].url.params["limit"], "500")
        self.assertEqual(requests[1].url.params["after_id"], "EVD-1")
        self.assertEqual(requests[1].url.params["snapshot_rowid"], "7")

    async def test_rejects_nonadvancing_cursor_and_cross_session_rows(self):
        cases = (
            {"session_id": "RES-1", "items": [], "has_more": True, "next_cursor": "EVD-1", "snapshot_rowid": 1},
            {"session_id": "RES-1", "items": [{"id": "EVD-1", "session_id": "RES-2"}], "has_more": False, "next_cursor": None, "snapshot_rowid": 1},
        )
        for payload in cases:
            with self.subTest(payload=payload):
                async with httpx.AsyncClient(
                    base_url="http://api",
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
                ) as client:
                    with self.assertRaisesRegex(ValueError, "evidence page"):
                        await read_evidence_pages(client, "RES-1")

    async def test_rejects_snapshot_watermark_changes_between_pages(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("after_id") is None:
                return httpx.Response(200, json={
                    "session_id": "RES-1",
                    "items": [{"id": "EVD-1", "session_id": "RES-1"}],
                    "has_more": True,
                    "next_cursor": "EVD-1",
                    "snapshot_rowid": 7,
                })
            return httpx.Response(200, json={
                "session_id": "RES-1",
                "items": [],
                "has_more": False,
                "next_cursor": None,
                "snapshot_rowid": 8,
            })

        async with httpx.AsyncClient(base_url="http://api", transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(ValueError, "snapshot watermark"):
                await read_evidence_pages(client, "RES-1")

    async def test_rejects_duplicate_or_out_of_order_ids_within_a_page(self):
        for ids in (("EVD-1", "EVD-1"), ("EVD-2", "EVD-1")):
            with self.subTest(ids=ids):
                payload = {
                    "session_id": "RES-1",
                    "items": [{"id": item_id, "session_id": "RES-1"} for item_id in ids],
                    "has_more": False,
                    "next_cursor": None,
                    "snapshot_rowid": 7,
                }
                async with httpx.AsyncClient(
                    base_url="http://api",
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
                ) as client:
                    with self.assertRaisesRegex(ValueError, "strictly increasing ID order"):
                        await read_evidence_pages(client, "RES-1")


if __name__ == "__main__":
    unittest.main()
