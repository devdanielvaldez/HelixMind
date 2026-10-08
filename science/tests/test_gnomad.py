import json
import unittest

import httpx

from medai_science.sources.gnomad import GnomADClient


class GnomADTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.previous_interval = GnomADClient.min_interval_seconds
        GnomADClient.min_interval_seconds = 0

    async def asyncTearDown(self):
        GnomADClient.min_interval_seconds = self.previous_interval

    async def test_exact_grch38_lookup_retains_validated_frequency_and_provenance(self):
        async def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["variables"], {"variantId": "19-44908822-T-C", "datasetId": "gnomad_r4"})
            self.assertIn("$datasetId: DatasetId!", body["query"])
            self.assertIn("variant_id genome { ac an af } exome { ac an af }", body["query"])
            return httpx.Response(200, json={"data": {"variant": {
                "variant_id": "19-44908822-T-C",
                "genome": {"ac": 10, "an": 1000, "af": 0.01},
                "exome": {"ac": 20, "an": 2000, "af": 0.01},
            }}})

        result = await GnomADClient(transport=httpx.MockTransport(handler)).lookup("chr19-44908822-t-c")
        self.assertTrue(result["found"])
        self.assertEqual(result["genome_frequency"], {"ac": 10, "an": 1000, "af": 0.01})
        self.assertEqual(result["exome_frequency"], {"ac": 20, "an": 2000, "af": 0.01})
        self.assertEqual(len(result["source_record_sha256"]), 64)
        self.assertIn("do not establish", result["interpretation_limit"])
        self.assertIn("does not verify the reference allele or left-align indels", result["interpretation_limit"])

    async def test_rejects_inconsistent_exome_counts_without_merging_callsets(self):
        client = GnomADClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"variant": {
            "variant_id": "1-2-A-G",
            "genome": {"ac": 1, "an": 10, "af": 0.1},
            "exome": {"ac": 1, "an": 10, "af": 0.5},
        }}})))
        with self.assertRaisesRegex(ValueError, "exome allele frequency does not agree"):
            await client.lookup("1-2-A-G")

    async def test_zero_allele_number_keeps_frequency_undefined(self):
        client = GnomADClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"variant": {
            "variant_id": "1-2-A-G",
            "genome": {"ac": 0, "an": 0, "af": None},
            "exome": {"ac": 1, "an": 10, "af": 0.1},
        }}})))
        result = await client.lookup("1-2-A-G")
        self.assertEqual(result["genome_frequency"], {"ac": 0, "an": 0, "af": None})

        malformed = GnomADClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"variant": {
            "variant_id": "1-2-A-G",
            "genome": {"ac": 0, "an": 0, "af": 0.25},
            "exome": None,
        }}})))
        with self.assertRaisesRegex(ValueError, "does not agree with counts"):
            await malformed.lookup("1-2-A-G")

    async def test_rejects_wrong_assembly_and_nonmatching_provider_row(self):
        client = GnomADClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"variant": {"variant_id": "1-2-A-G", "genome": {"ac": 1, "an": 10, "af": 0.1}}}})))
        with self.assertRaisesRegex(ValueError, "requires GRCh38"):
            await client.lookup("1-2-A-G", "GRCh37")
        with self.assertRaisesRegex(ValueError, "does not match"):
            await client.lookup("1-3-A-G")

    async def test_distinguishes_explicit_not_found_from_missing_graphql_field(self):
        explicit_not_found = GnomADClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": {"variant": None}})
        ))
        missing_field = GnomADClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": {}})
        ))

        result = await explicit_not_found.lookup("1-2-A-G")
        self.assertFalse(result["found"])
        with self.assertRaisesRegex(ValueError, "missing the variant field"):
            await missing_field.lookup("1-2-A-G")

    def test_variant_identifier_is_validated_and_canonicalized(self):
        for value in ("1-0-A-G", "23-10-A-G", "1-10-A-A", "1 -10-A-G", "1-10-A-G\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GnomADClient.parse_variant_id(value)
        self.assertEqual(GnomADClient.parse_variant_id("chrX-100-G-A"), "X-100-G-A")
        self.assertEqual(GnomADClient.parse_variant_id("1-10-ac-g"), "1-10-AC-G")


if __name__ == "__main__":
    unittest.main()
