from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from medai_science.sources.clingen_gene_validity import ClinGenGeneValidityClient, SOURCE_URL, _REQUIRED, _parse_export


def export() -> bytes:
    stream = io.StringIO(newline="")
    stream.write('"CLINGEN GENE DISEASE VALIDITY CURATIONS","","","","","","","","",""\n')
    stream.write('"FILE CREATED: 2026-10-07","","","","","","","","",""\n')
    stream.write('"WEBPAGE: https://search.clinicalgenome.org/kb/gene-validity","","","","","","","","",""\n')
    stream.write(','.join('"' + '+' * len(column) + '"' for column in _REQUIRED) + "\n")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_REQUIRED)
    writer.writerow(["+" * len(column) for column in _REQUIRED])
    writer.writerow(["SLC25A4", "HGNC:10990", "mitochondrial DNA depletion syndrome 12B", "MONDO:0014175", "AR", "SOP5", "Definitive", "https://search.clinicalgenome.org/kb/gene-validity/CCID:006168", "2017-08-30T00:00:00.000Z", "Hypertrophic Cardiomyopathy"])
    writer.writerow(["SLC25A4", "HGNC:10990", "Other mitochondrial disorder", "MONDO:0000001", "AD", "SOP6", "Disputed", "https://search.clinicalgenome.org/kb/gene-validity/CCID:006169", "2018-08-30T00:00:00.000Z", "Expert Panel"])
    return stream.getvalue().encode()


class ClinGenGeneValidityTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_official_preamble_and_exact_hgnc_mondo_queries(self):
        payload = export()
        requests = []

        async def handler(request):
            requests.append(request)
            self.assertEqual(request.url, httpx.URL(SOURCE_URL))
            return httpx.Response(200, content=payload, headers={"ETag": '"clingen-1"'})

        with tempfile.TemporaryDirectory() as temp:
            client = ClinGenGeneValidityClient(temp, transport=httpx.MockTransport(handler))
            gene = await client.search_gene("hgnc:10990")
            disease = await client.search_disease("MONDO:0014175")
            self.assertEqual(gene["source_file_date"], "2026-10-07")
            self.assertEqual(gene["total_matching_curations"], 2)
            self.assertEqual(gene["curations"][0]["classification"], "Definitive")
            self.assertEqual(gene["curations"][0]["expert_panel"], "Hypertrophic Cardiomyopathy")
            self.assertEqual(disease["total_matching_curations"], 1)
            self.assertEqual(len(requests), 1)
            self.assertEqual((Path(temp) / "gene-validity.csv").stat().st_mode & 0o777, 0o600)
            self.assertEqual(Path(temp).stat().st_mode & 0o777, 0o700)

    async def test_conditional_not_modified_and_stale_on_provider_error(self):
        payload = export()
        requests = []

        async def initial(request):
            requests.append(request)
            return httpx.Response(200, content=payload, headers={"ETag": '"clingen-1"'})

        with tempfile.TemporaryDirectory() as temp:
            client = ClinGenGeneValidityClient(temp, transport=httpx.MockTransport(initial))
            await client.refresh_if_due()
            metadata_path = Path(temp) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["checked_at"] = "2020-01-01T00:00:00Z"
            metadata_path.write_text(json.dumps(metadata))
            state_path = Path(temp) / "refresh-state.json"
            state_path.write_text(json.dumps({"last_attempt_at": "2020-01-01T00:00:00Z", "error": None}))

            async def not_modified(request):
                requests.append(request)
                self.assertEqual(request.headers.get("if-none-match"), '"clingen-1"')
                return httpx.Response(304)

            refreshed = await ClinGenGeneValidityClient(temp, transport=httpx.MockTransport(not_modified)).refresh_if_due()
            self.assertEqual(refreshed["cache_status"], "not_modified")
            self.assertEqual(len(requests), 2)

            metadata = json.loads(metadata_path.read_text())
            metadata["checked_at"] = "2020-01-01T00:00:00Z"
            metadata_path.write_text(json.dumps(metadata))
            state_path.write_text(json.dumps({"last_attempt_at": "2020-01-01T00:00:00Z", "error": None}))

            async def fail(request):
                return httpx.Response(503)

            stale = await ClinGenGeneValidityClient(temp, transport=httpx.MockTransport(fail)).refresh_if_due()
            self.assertEqual(stale["cache_status"], "stale_on_refresh_error")

            async def should_not_be_called(request):
                raise AssertionError("failed refresh must be throttled for 24 hours")

            throttled = await ClinGenGeneValidityClient(temp, transport=httpx.MockTransport(should_not_be_called)).refresh_if_due()
            self.assertEqual(throttled["cache_status"], "stale_on_refresh_error")

    async def test_rejects_schema_identifiers_and_limit_errors(self):
        with self.assertRaisesRegex(ValueError, "published gene-disease validity format"):
            _parse_export(b"symbol,id\nX,HGNC:1\n")
        client = ClinGenGeneValidityClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=export())))
        with self.assertRaisesRegex(ValueError, "exact gene symbol"):
            await client.search_gene("Alzheimer disease")
        with self.assertRaisesRegex(ValueError, "exact MONDO"):
            await client.search_disease("Alzheimer")
        with self.assertRaisesRegex(ValueError, "limit"):
            await client.search_gene("SLC25A4", 101)


if __name__ == "__main__":
    unittest.main()
