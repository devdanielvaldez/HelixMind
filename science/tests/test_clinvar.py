import unittest
from pathlib import Path

import httpx

from medai_science.sources.clinvar import ClinVarClient


class ClinVarTests(unittest.IsolatedAsyncioTestCase):
    def test_excludes_clinvar_curated_composite_cftr_haplotype(self):
        fixture = Path(__file__).parent / "fixtures" / "clinvar_cftr_haplotype_vcv.xml"
        locations = ClinVarClient._parse_vcv_locations(fixture.read_bytes())

        # ClinVar VCV000242535 explicitly labels c.1210-12T[5] as part of a
        # two-variant CFTR haplotype. It must not be reconciled as a singleton.
        self.assertEqual(locations, {})

    async def test_search_retains_bounded_source_linked_variant_summaries(self):
        requests = []

        async def handler(request):
            requests.append(request)
            if request.url.path.endswith("esearch.fcgi"):
                return httpx.Response(200, json={"esearchresult": {"count": "2", "idlist": ["123", "456"]}})
            if request.url.path.endswith("esummary.fcgi"):
                return httpx.Response(200, json={"result": {
                    "123": {"uid": "123", "accession": "VCV000000123.1", "title": "NM_000000.1(GENE):c.1A>G", "clinical_significance": {"description": "Uncertain significance"}, "genes": [{"symbol": "GENE", "GeneID": "1"}], "variation_type": "single nucleotide variant", "last_updated": "2026-01-01", "unused_large_field": "not retained"},
                    "456": {"uid": "456", "accession": "VCV000000456.2", "title": "Second record", "clinical_significance": {"description": "Conflicting classifications"}},
                }})
            return httpx.Response(200, text=(
                '<VariationArchiveSet><VariationArchive VariationID="123"><ClassifiedRecord>'
                '<SimpleAllele><Location><SequenceLocation Assembly="GRCh38" Chr="1" start="10" stop="10" Accession="NC_000001.11" referenceAllele="A" alternateAllele="G"/>'
                '<SequenceLocation Assembly="GRCh37" Chr="1" start="20" stop="20" Accession="NC_000001.10" referenceAllele="A" alternateAllele="G"/></Location></SimpleAllele>'
                '</ClassifiedRecord></VariationArchive><VariationArchive VariationID="456"><ClassifiedRecord>'
                '<Haplotype><SimpleAllele><Location><SequenceLocation Assembly="GRCh38" Chr="1" start="11" stop="11" referenceAllele="C" alternateAllele="T"/></Location></SimpleAllele></Haplotype>'
                '</ClassifiedRecord></VariationArchive></VariationArchiveSet>'
            ))

        client = ClinVarClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.search("rs429358", 2)

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].url.params["db"], "clinvar")
        self.assertEqual(requests[1].url.params["id"], "123,456")
        self.assertEqual(requests[2].url.params["rettype"], "vcv")
        self.assertEqual(result["total_matches_reported"], 2)
        self.assertEqual(result["records"][0]["accession"], "VCV000000123.1")
        self.assertEqual(result["records"][0]["genes"][0]["symbol"], "GENE")
        self.assertNotIn("unused_large_field", result["records"][0])
        self.assertEqual(len(result["records"][0]["source_record_sha256"]), 64)
        self.assertEqual(result["records"][0]["assembly_locations"][0]["assembly"], "GRCh38")
        self.assertIn("HGVS-right-shifted", result["records"][0]["assembly_locations"][0]["source_coordinate_convention"])
        self.assertEqual(len(result["records"][0]["assembly_locations"]), 2)
        self.assertEqual(result["records"][1]["assembly_locations"], [])
        self.assertIn("not independently re-evaluated", result["records"][0]["classification_semantics"])

    async def test_rejects_patient_identifiers_and_oversized_limits(self):
        client = ClinVarClient()
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            await client.search("my patient rs429358")
        with self.assertRaisesRegex(ValueError, "limit"):
            await client.search("rs429358", 21)

    async def test_does_not_forward_api_key_through_a_redirect(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "https://attacker.example/collect"})

        client = ClinVarClient(
            lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
            api_key="clinvar-secret",
            email="researcher@example.org",
        )
        with self.assertRaises(httpx.HTTPStatusError):
            await client.search("rs429358", 1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, "eutils.ncbi.nlm.nih.gov")
        self.assertEqual(requests[0].url.params["api_key"], "clinvar-secret")


if __name__ == "__main__":
    unittest.main()
