from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from medai_science.sources.gencc import GenCCClient, _REQUIRED_COLUMNS, _validate_export


def export_bytes() -> bytes:
    header = sorted(_REQUIRED_COLUMNS | {"disease_original_curie", "disease_original_title", "sub_notes"})
    rows = [
        {
            "sgc_id": "SGC-10001", "version_number": "2", "gene_curie": "HGNC:7", "gene_symbol": "A2M",
            "disease_curie": "MONDO:0004975", "disease_title": "Alzheimer disease",
            "disease_original_curie": "MONDO:0004975", "disease_original_title": "Alzheimer disease",
            "classification_curie": "GENCC:100004", "classification_title": "Limited", "moi_curie": "HP:0000006",
            "moi_title": "Autosomal dominant", "submitter_curie": "GENCC:000102", "submitter_title": "ClinGen",
            "sub_date": "2024-01-02", "sub_public_report_url": "https://example.org/report/1",
            "sub_assertion_criteria_url": "https://example.org/criteria/1", "sub_pmids": "12345, 67890",
            "disease_original_curie": "OMIM:999999", "disease_original_title": "Source term",
            "sub_notes": "A tab\tand newline\ninside a quoted TSV field",
        },
        {
            "sgc_id": "SGC-10002", "version_number": "1", "gene_curie": "HGNC:99", "gene_symbol": "OTHER",
            "disease_curie": "MONDO:0000001", "disease_title": "Other", "classification_curie": "GENCC:100001",
            "classification_title": "Definitive", "moi_curie": "HP:0000007", "moi_title": "Autosomal recessive",
            "submitter_curie": "GENCC:000100", "submitter_title": "Submitter",
            "sub_date": "2023-01-01", "sub_public_report_url": "", "sub_assertion_criteria_url": "",
            "sub_pmids": "", "disease_original_curie": "", "disease_original_title": "", "sub_notes": "",
        },
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=header, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class GenCCClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_downloads_new_format_conditional_cache_and_exact_gene_query(self):
        payload = export_bytes()
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(200, content=payload, headers={"ETag": '"abc"', "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"})

        with tempfile.TemporaryDirectory(prefix="medai-gencc-") as temp:
            client = GenCCClient(temp, transport=httpx.MockTransport(handler))
            result = await client.search_gene("hgnc:7")
            self.assertEqual(result["total_matching_assertions"], 1)
            self.assertEqual(result["assertions"][0]["classification"], "Limited")
            self.assertEqual(result["assertions"][0]["pubmed_ids"], ["12345", "67890"])
            self.assertEqual(result["assertions"][0]["disease_original_curie"], "OMIM:999999")
            self.assertEqual(result["data_license"], "CC0-1.0")
            self.assertEqual(len(requests), 1)
            self.assertEqual((Path(temp) / "submissions.tsv").stat().st_mode & 0o777, 0o600)
            self.assertEqual(Path(temp).stat().st_mode & 0o777, 0o700)
            # A second query uses the validated snapshot and does not consume download quota.
            disease = await client.search_disease("MONDO:0004975")
            self.assertEqual(disease["total_matching_assertions"], 1)
            self.assertEqual(len(requests), 1)

    async def test_refresh_uses_etag_and_last_modified_after_daily_interval(self):
        payload = export_bytes()
        requests = []

        async def handler(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(200, content=payload, headers={"ETag": '"stable"', "Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"})
            self.assertEqual(request.headers["if-none-match"], '"stable"')
            self.assertEqual(request.headers["if-modified-since"], "Wed, 01 Jan 2025 00:00:00 GMT")
            return httpx.Response(304)

        with tempfile.TemporaryDirectory(prefix="medai-gencc-") as temp:
            client = GenCCClient(temp, transport=httpx.MockTransport(handler))
            await client.refresh_if_due()
            metadata_path = Path(temp) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["last_checked_at"] = "2020-01-01T00:00:00Z"
            metadata_path.write_text(json.dumps(metadata))
            refresh_state = Path(temp) / "refresh-state.json"
            refresh_state.write_text(json.dumps({"last_attempt_at": "2020-01-01T00:00:00Z", "error": None}))
            result = await client.refresh_if_due()
            self.assertEqual(result["cache_status"], "not_modified")
            self.assertEqual(len(requests), 2)

    async def test_failed_refresh_returns_stale_snapshot_but_bad_first_download_fails(self):
        payload = export_bytes()

        async def good(request):
            return httpx.Response(200, content=payload)

        async def broken(request):
            return httpx.Response(503)

        with tempfile.TemporaryDirectory(prefix="medai-gencc-") as temp:
            cached = GenCCClient(temp, transport=httpx.MockTransport(good))
            await cached.refresh_if_due()
            metadata_path = Path(temp) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["last_checked_at"] = "2020-01-01T00:00:00Z"
            metadata_path.write_text(json.dumps(metadata))
            refresh_state = Path(temp) / "refresh-state.json"
            refresh_state.write_text(json.dumps({"last_attempt_at": "2020-01-01T00:00:00Z", "error": None}))
            stale = await GenCCClient(temp, transport=httpx.MockTransport(broken)).refresh_if_due()
            self.assertEqual(stale["cache_status"], "stale_on_refresh_error")
            requests = []

            async def no_request(request):
                requests.append(request)
                raise AssertionError("GenCC refresh must be throttled after a failed attempt")

            throttled = await GenCCClient(temp, transport=httpx.MockTransport(no_request)).refresh_if_due()
            self.assertEqual(throttled["cache_status"], "stale_on_refresh_error")
            self.assertEqual(requests, [])
            with self.assertRaises(httpx.HTTPStatusError):
                await GenCCClient(Path(temp) / "empty", transport=httpx.MockTransport(broken)).search_gene("A2M")

    async def test_rejects_legacy_schema_bad_identifiers_and_oversized_results(self):
        legacy = b"uuid\tgene_symbol\nlegacy\tA2M\n"
        with self.assertRaisesRegex(ValueError, "new-format"):
            _validate_export(legacy)
        client = GenCCClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=export_bytes())))
        with self.assertRaisesRegex(ValueError, "exact gene symbol"):
            await client.search_gene("Alzheimer disease")
        with self.assertRaisesRegex(ValueError, "MONDO CURIE"):
            await client.search_disease("Alzheimer disease")


if __name__ == "__main__":
    unittest.main()
