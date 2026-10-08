import json
import unittest

import httpx

from medai_science.sources.orphadata import OrphadataClient


def _payload():
    return {
        "data": {
            "__licence": {"identifier": "CC-BY-4.0"},
            "results": {
                "ORPHAcode": 61,
                "Preferred term": "Alpha-mannosidosis",
                "Typology": "Disease",
                "DisorderGroup": "Disorder",
                "Date": "2024-12-03",
                "OrphanetURL": "https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Expert=61",
                "DisorderGeneAssociation": [{
                    "DisorderGeneAssociationStatus": "Validated",
                    "DisorderGeneAssociationType": "Disease-causing germline mutation(s) in",
                    "SourceOfValidation": "22587682[PMID]",
                    "Gene": {
                        "Symbol": "MAN2B1",
                        "name": "mannosidase alpha class 2B member 1",
                        "GeneType": "gene with protein product",
                        "Synonym": ["LAMAN", "MANB"],
                        "Locus": [{"GeneLocus": "7q31-q32"}],
                        "ExternalReference": [
                            {"Source": "HGNC", "Reference": "6826"},
                            {"Source": "UniProtKB/SwissProt", "Reference": "O00754"},
                        ],
                    },
                }],
            },
        },
        "datasetCategory": {"name": "Rare diseases and associated genes", "lang": "en"},
    }


class OrphadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_orphacode_retains_typed_association_and_provider_provenance(self):
        payload = _payload()

        def handler(request):
            self.assertEqual(request.url.path, "/rd-associated-genes/orphacodes/61")
            return httpx.Response(200, json=payload)

        result = await OrphadataClient(httpx.MockTransport(handler)).disease_genes("000061")
        self.assertEqual(result["disease"]["preferred_term"], "Alpha-mannosidosis")
        self.assertEqual(result["records"][0]["gene_symbol"], "MAN2B1")
        self.assertEqual(result["records"][0]["association_type"], "Disease-causing germline mutation(s) in")
        self.assertEqual(result["records"][0]["association_status"], "Validated")
        self.assertEqual(result["records"][0]["gene_external_references"][0], {"source": "HGNC", "reference": "6826"})
        self.assertEqual(result["attribution"], "Orphadata, CC BY 4.0")
        self.assertEqual(len(result["source_response_sha256"]), 64)
        self.assertEqual(len(result["records"][0]["record_sha256"]), 64)
        self.assertIn("not aggregated into probabilities", result["interpretation_limit"])

    async def test_exact_code_mismatch_is_not_returned_as_a_match(self):
        payload = _payload()
        payload["data"]["results"]["ORPHAcode"] = 62
        result = await OrphadataClient(httpx.MockTransport(lambda _: httpx.Response(200, json=payload))).disease_genes(61)
        self.assertFalse(result["found"])
        self.assertEqual(result["records"], [])

    async def test_rejects_invalid_orphacode_without_network_access(self):
        for value in (0, -1, "ORPHA:61", "61/../62", True, 1234567):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await OrphadataClient().disease_genes(value)

    async def test_rejects_malformed_association_collection(self):
        payload = _payload()
        payload["data"]["results"]["DisorderGeneAssociation"] = "malformed"
        client = OrphadataClient(httpx.MockTransport(lambda _: httpx.Response(200, content=json.dumps(payload))))
        with self.assertRaisesRegex(ValueError, "association list is malformed"):
            await client.disease_genes(61)


if __name__ == "__main__":
    unittest.main()
