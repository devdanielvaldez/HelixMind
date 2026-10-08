from __future__ import annotations

import json
import unittest

import httpx

from medai_science.sources.ensembl_vep import EnsemblVEPClient


def report(variant_count: int = 1) -> dict:
    variants = []
    for index in range(variant_count):
        variants.append({
            "record_number": index + 1,
            "contig": "1",
            "position_1_based": 100 + index,
            "normalized_position_1_based": 100 + index,
            "reference_allele": "G",
            "alternate_allele": "A",
            "normalized_reference_allele": "G",
            "normalized_alternate_allele": "A",
            "sequence_allele_supported": True,
            "reference_validated": True,
        })
    return {"genome_build": "GRCh38", "reference_validated": True, "reference_fasta_sha256": "a" * 64, "variants": variants}


def vep_record(input_value: str, **changes) -> dict:
    return {
        "input": input_value,
        "assembly_name": "GRCh38",
        "seq_region_name": "1",
        "start": 100,
        "end": 100,
        "strand": 1,
        "allele_string": "G/A",
        "most_severe_consequence": "missense_variant",
        "transcript_consequences": [{"transcript_id": "ENST00000000001", "gene_id": "ENSG00000000001", "consequence_terms": ["missense_variant"], "amino_acids": "G/D", "codons": "Ggt/Gat"}],
        **changes,
    }


class EnsemblVEPTests(unittest.IsolatedAsyncioTestCase):
    def test_query_meter_counts_only_supported_reference_checked_alleles(self):
        source = report(3)
        source["variants"][1]["reference_validated"] = False
        source["variants"][2]["sequence_allele_supported"] = False
        self.assertEqual(EnsemblVEPClient.queryable_count(source), 1)
        with self.assertRaisesRegex(ValueError, "reference-validated"):
            EnsemblVEPClient.queryable_count({**source, "reference_validated": False})

    async def test_annotates_only_reference_validated_variants_and_retains_provenance(self):
        async def handler(request):
            self.assertEqual(request.url.host, "rest.ensembl.org")
            body = json.loads(request.content)
            self.assertEqual(body["variants"], ["1 100 . G A . . ."])
            return httpx.Response(200, json=[vep_record(body["variants"][0])], headers={"X-Ens-Release": "115"})

        client = EnsemblVEPClient(transport=httpx.MockTransport(handler))
        result = await client.annotate_report(report())
        self.assertEqual(result["ensembl_release"], "115")
        self.assertEqual(result["queried_variant_alleles"], 1)
        self.assertEqual(result["results"][0]["most_severe_consequence"], "missense_variant")
        self.assertEqual(result["results"][0]["transcript_consequences"][0]["amino_acids"], "G/D")
        self.assertEqual(len(result["results"][0]["source_record_sha256"]), 64)
        self.assertIn("not pathogenicity", result["interpretation_limit"])

    async def test_uses_build_specific_host_and_batches_with_bounded_inputs(self):
        source = report(201)
        source["genome_build"] = "GRCh37"
        requests = []

        async def handler(request):
            requests.append(request)
            submitted = json.loads(request.content)["variants"]
            return httpx.Response(200, json=[vep_record(item, assembly_name="GRCh37", start=int(item.split()[1]), end=int(item.split()[1])) for item in submitted])

        result = await EnsemblVEPClient(transport=httpx.MockTransport(handler)).annotate_report(source)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].url.host, "grch37.rest.ensembl.org")
        self.assertEqual(len(json.loads(requests[0].content)["variants"]), 200)
        self.assertEqual(result["queried_variant_alleles"], 201)

    async def test_rejects_unvalidated_mismatched_or_malformed_source_results(self):
        invalid = report()
        invalid["reference_validated"] = False
        with self.assertRaisesRegex(ValueError, "reference-validated"):
            await EnsemblVEPClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[]))).annotate_report(invalid)

        async def mismatched_input(request):
            return httpx.Response(200, json=[vep_record("1 999 . G A . . .")])

        with self.assertRaisesRegex(ValueError, "exact submitted variant"):
            await EnsemblVEPClient(transport=httpx.MockTransport(mismatched_input)).annotate_report(report())

        async def mismatched_assembly(request):
            body = json.loads(request.content)
            return httpx.Response(200, json=[vep_record(body["variants"][0], assembly_name="GRCh37")])

        with self.assertRaisesRegex(ValueError, "assembly or contig"):
            await EnsemblVEPClient(transport=httpx.MockTransport(mismatched_assembly)).annotate_report(report())

    async def test_skips_unsupported_variant_classes_without_sending_them(self):
        source = report()
        source["variants"][0]["sequence_allele_supported"] = False
        client = EnsemblVEPClient(transport=httpx.MockTransport(lambda request: self.fail("unsupported variant must not be sent")))
        result = await client.annotate_report(source)
        self.assertEqual(result["queried_variant_alleles"], 0)
        self.assertEqual(result["skipped_unsupported_or_unvalidated"], 1)


if __name__ == "__main__":
    unittest.main()
