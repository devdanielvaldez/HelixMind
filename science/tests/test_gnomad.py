import json
import unittest

import httpx

from medai_science.sources.gnomad import GnomADClient, validate_gnomad_schema_contract


def graphql_ref(name, kind="SCALAR", non_null=False):
    ref = {"kind": kind, "name": name, "ofType": None}
    return {"kind": "NON_NULL", "name": None, "ofType": ref} if non_null else ref


def documented_schema_introspection():
    def field(name, result_type, args=None):
        return {"name": name, "type": result_type, "args": args or []}

    return {
        "query": {"fields": [field("variant", graphql_ref("VariantDetails", "OBJECT"), [
            {"name": "variantId", "type": graphql_ref("String", non_null=True)},
            {"name": "dataset", "type": graphql_ref("DatasetId", "ENUM", non_null=True)},
        ])]},
        "variantDetails": {"fields": [
            field("variant_id", graphql_ref("String")),
            field("genome", graphql_ref("VariantSequencingTypeData", "OBJECT")),
            field("exome", graphql_ref("VariantSequencingTypeData", "OBJECT")),
        ]},
        "sequencingData": {"fields": [
            field("ac", graphql_ref("Int")),
            field("an", graphql_ref("Int")),
            field("af", graphql_ref("Float")),
        ]},
        "datasetId": {"enumValues": [{"name": "gnomad_r4"}, {"name": "gnomad_r3"}]},
    }


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

    async def test_live_schema_verifier_queries_and_checks_consumed_contract(self):
        async def handler(request):
            body = json.loads(request.content)
            self.assertIn('__type(name: "VariantDetails")', body["query"])
            return httpx.Response(200, json={"data": documented_schema_introspection()})

        result = await GnomADClient(transport=httpx.MockTransport(handler)).verify_schema_contract()
        self.assertTrue(result["validated"])
        self.assertEqual(result["dataset"], "gnomad_r4")
        self.assertIn("genome.af", result["fields"])
        self.assertEqual(len(result["schema_sha256"]), 64)
        self.assertEqual(len(result["introspection_query_sha256"]), 64)

    def test_schema_contract_rejects_nonfinite_values_even_in_unused_schema_fields(self):
        data = documented_schema_introspection()
        data["unused"] = float("nan")
        with self.assertRaisesRegex(ValueError, "not strict JSON"):
            validate_gnomad_schema_contract(data)

    def test_schema_contract_rejects_missing_frequency_field_or_dataset(self):
        data = documented_schema_introspection()
        del data["sequencingData"]["fields"][2]
        with self.assertRaisesRegex(ValueError, "af must be Float"):
            validate_gnomad_schema_contract(data)
        data = documented_schema_introspection()
        data["datasetId"]["enumValues"] = [{"name": "gnomad_r3"}]
        with self.assertRaisesRegex(ValueError, "no longer includes gnomad_r4"):
            validate_gnomad_schema_contract(data)

    def test_schema_contract_rejects_duplicate_or_malformed_variant_arguments(self):
        data = documented_schema_introspection()
        variant = data["query"]["fields"][0]
        variant["args"].append(dict(variant["args"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate argument names"):
            validate_gnomad_schema_contract(data)
        data = documented_schema_introspection()
        data["query"]["fields"][0]["args"].append({"name": 1, "type": graphql_ref("String")})
        with self.assertRaisesRegex(ValueError, "malformed argument records"):
            validate_gnomad_schema_contract(data)

    def test_schema_contract_rejects_duplicate_or_malformed_dataset_enum_rows(self):
        data = documented_schema_introspection()
        data["datasetId"]["enumValues"].append({"name": "gnomad_r4"})
        with self.assertRaisesRegex(ValueError, "duplicate values"):
            validate_gnomad_schema_contract(data)
        data = documented_schema_introspection()
        data["datasetId"]["enumValues"].append({"name": None})
        with self.assertRaisesRegex(ValueError, "enum values are malformed"):
            validate_gnomad_schema_contract(data)

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

    async def test_rejects_nonstandard_json_numbers_even_outside_selected_fields(self):
        malformed_lookup = GnomADClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b'{"data":{"variant":null},"ignored":NaN}')
        ))
        malformed_schema = GnomADClient(transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b'{"data":{"query":{"fields":[]}},"ignored":Infinity}')
        ))
        with self.assertRaisesRegex(ValueError, "invalid or duplicate-key JSON"):
            await malformed_lookup.lookup("1-2-A-G")
        with self.assertRaisesRegex(ValueError, "invalid or duplicate-key JSON"):
            await malformed_schema.verify_schema_contract()

    def test_variant_identifier_is_validated_and_canonicalized(self):
        for value in ("1-0-A-G", "23-10-A-G", "1-10-A-A", "1 -10-A-G", "1-10-A-G\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GnomADClient.parse_variant_id(value)
        self.assertEqual(GnomADClient.parse_variant_id("chrX-100-G-A"), "X-100-G-A")
        self.assertEqual(GnomADClient.parse_variant_id("1-10-ac-g"), "1-10-AC-G")


if __name__ == "__main__":
    unittest.main()
