import unittest

import httpx

from medai_science.sources.gene_disease import GeneDiseaseClient, reconcile_gene_disease_sources


class GeneDiseaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_opentargets_query_normalizes_target_evidence(self):
        async def handler(request):
            self.assertEqual(request.url.host, "api.platform.opentargets.org")
            body = __import__("json").loads(request.content)
            self.assertEqual(body["variables"], {"id": "EFO_0000249", "size": 4})
            return httpx.Response(200, json={"data": {"disease": {"id": "EFO_0000249", "name": "Alzheimer disease", "associatedTargets": {"count": 1, "rows": [{"score": 0.73, "target": {"id": "ENSG00000130203", "approvedSymbol": "APOE"}}]}}}})

        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.opentargets("EFO_0000249", 4)
        self.assertEqual(result["records"][0]["gene_id"], "ENSG00000130203")
        self.assertEqual(result["records"][0]["score"], 0.73)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(len(result["response_sha256"]), 64)

    async def test_opentargets_rejects_disease_identity_mismatch_and_nonstandard_numeric_data(self):
        wrong_identity = {"data": {"disease": {"id": "EFO_0000555", "associatedTargets": {"count": 0, "rows": []}}}}
        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wrong_identity)), **kwargs))
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            await client.opentargets("EFO_0000249")

        response = b'{"data":{"disease":{"id":"EFO_0000249","associatedTargets":{"count":1,"rows":[{"score":NaN,"target":{"id":"ENSG00000130203"}}]}}}}'
        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response)), **kwargs))
        with self.assertRaisesRegex(ValueError, "non-standard JSON"):
            await client.opentargets("EFO_0000249")

        duplicate_key = b'{"data":{"disease":{"id":"EFO_0000249","id":"EFO_0000555","associatedTargets":{"count":0,"rows":[]}}}}'
        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=duplicate_key)), **kwargs))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            await client.opentargets("EFO_0000249")

    async def test_opentargets_marks_bounded_association_results_as_truncated(self):
        payload = {"data": {"disease": {"id": "EFO_0000249", "associatedTargets": {"count": 12, "rows": [{"score": 0.5, "target": {"id": "ENSG00000130203", "approvedSymbol": "APOE"}}]}}}}
        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)), **kwargs))
        result = await client.opentargets("EFO_0000249", limit=1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["reported_count"], 12)

    async def test_gwas_trait_scope_and_gene_mapping(self):
        async def handler(request):
            self.assertEqual(request.url.params["show_child_traits"], "false")
            return httpx.Response(200, json={"_embedded": {"associations": [{"associationId": "A1", "mappedGenes": [{"ensemblId": "ENSG00000130203", "geneName": "APOE"}, {"geneName": "TREM2"}, "APP", "ENSG00000141510"], "pubmedId": 123}]}})

        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.gwas_catalog("EFO_0000249")
        self.assertEqual(result["records"][0]["gene_id"], "ENSG00000130203")
        self.assertEqual(result["records"][0]["pubmed_id"], 123)
        self.assertEqual(result["records"][1]["gene_id"], None)
        self.assertEqual(result["records"][1]["gene_mapping_candidate"], "TREM2")
        self.assertEqual(result["records"][2]["gene_id"], None)
        self.assertEqual(result["records"][2]["gene_mapping_candidate"], "APP")
        self.assertEqual(result["records"][3]["gene_id"], "ENSG00000141510")
        self.assertIsNone(result["records"][3]["gene_mapping_candidate"])
        with self.assertRaises(ValueError):
            await client.gwas_catalog("MONDO_0004979")

    async def test_gwas_rejects_missing_or_malformed_collections_instead_of_returning_no_hits(self):
        for payload in (
            {"_embedded": {}},
            {"unrelated": []},
            {"associations": [{"associationId": "A1"}]},
            {"associations": [{"mappedGenes": None}]},
            {"associations": [{"mappedGenes": 42}]},
        ):
            client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(lambda _, body=payload: httpx.Response(200, json=body)), **kwargs))
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "association|mapped-gene"):
                await client.gwas_catalog("EFO_0000249")

    async def test_monarch_keeps_native_provenance_and_filters_to_exact_direct_human_gene_assertions(self):
        async def handler(request):
            self.assertEqual(request.url.host, "api.monarchinitiative.org")
            self.assertEqual(request.url.path, "/v3/api/association")
            self.assertEqual(request.url.params.get("object"), "MONDO:0004979")
            self.assertEqual(request.url.params.get("subject_taxon"), "NCBITaxon:9606")
            self.assertEqual(request.url.params.get("direct"), "true")
            self.assertEqual(request.url.params.get_list("category"), [
                "biolink:CausalGeneToDiseaseAssociation",
                "biolink:CorrelatedGeneToDiseaseAssociation",
            ])
            valid = {"subject": "HGNC:117", "subject_label": "APOE", "subject_category": "biolink:Gene", "subject_taxon": "NCBITaxon:9606", "object": "MONDO:0004979", "object_category": "biolink:Disease", "category": "biolink:CausalGeneToDiseaseAssociation", "predicate": "biolink:causes", "primary_knowledge_source": ["infores:clingen"], "aggregator_knowledge_source": ["infores:monarchinitiative"], "publications": ["PMID:123"], "evidence_count": 1}
            invalid = {**valid, "subject": "HGNC:999", "object": "MONDO:0000001"}
            return httpx.Response(200, json={"total": 2, "items": [valid, invalid]})

        client = GeneDiseaseClient(lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
        result = await client.monarch_gene_disease("MONDO:0004979", 10)
        self.assertEqual(result["reported_count"], 2)
        self.assertEqual(len(result["records"]), 1)
        record = result["records"][0]
        self.assertEqual(record["gene_id"], "HGNC:117")
        self.assertEqual(record["primary_knowledge_source"], ["infores:clingen"])
        self.assertEqual(record["aggregator_knowledge_source"], ["infores:monarchinitiative"])
        self.assertEqual(record["publications"], ["PMID:123"])
        self.assertEqual(len(result["response_sha256"]), 64)
        self.assertEqual(result["raw_response"]["total"], 2)

    def test_reconciliation_keeps_source_scores_separate(self):
        result = reconcile_gene_disease_sources("EFO_0000249", [
            {"source": "Open Targets", "records": [{"gene_id": "ENSG00000130203", "symbol": "APOE", "score": 0.9}]},
            {"source": "GWAS Catalog", "records": [{"gene_id": "ENSG00000130203", "symbol": "APOE", "association_id": "X"}, {"gene_id": "HGNC:117", "symbol": "APOE"}, {"gene_id": None, "gene_mapping_candidate": "APOE"}]},
        ])
        gene = result["genes"][0]
        self.assertEqual(gene["distinct_source_count"], 2)
        ot_evidence = next(row for row in gene["source_evidence"] if row["source"] == "Open Targets")
        self.assertEqual(ot_evidence["score"], 0.9)
        self.assertNotIn("confidence", gene)
        self.assertEqual(len(result["genes"]), 2)
        self.assertEqual({gene["gene_id"] for gene in result["genes"]}, {"ENSG00000130203", "HGNC:117"})


if __name__ == "__main__":
    unittest.main()
