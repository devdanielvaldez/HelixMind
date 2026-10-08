from __future__ import annotations

import unittest

import httpx

from medai_science.sources.panelapp import PanelAppClient


def panel_document(panel_id: int = 214, version: str = "2.1") -> dict:
    return {
        "id": panel_id,
        "name": "Hereditary condition panel",
        "disease_group": "Neurology",
        "disease_sub_group": "Neurogenetics",
        "status": "public",
        "version": version,
        "version_created": "2026-01-01T00:00:00Z",
        "relevant_disorders": ["Example disorder"],
        "types": [{"name": "Rare Disease", "slug": "rare-disease"}],
        "genes": [
            {
                "entity_type": "gene",
                "entity_name": "GENE1",
                "gene_data": {"hgnc_id": "HGNC:1", "ensembl_genes": ["ENSG00000000001"]},
                "confidence_level": "3",
                "mode_of_inheritance": "BIALLELIC, autosomal or pseudoautosomal",
                "mode_of_pathogenicity": "Loss-of-function variants",
                "phenotypes": ["Phenotype A"],
                "evidence": ["Case-control evidence"],
                "publications": ["PMID:12345"],
                "transcript": ["ENST00000000001"],
            },
            {"entity_type": "Region", "entity_name": "ISCA-1", "confidence_level": "3"},
        ],
    }


class PanelAppTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_stable_panel_id_and_preserves_curated_values_and_provenance(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/api/v1/panels/214/")
            self.assertEqual(request.url.params["version"], "2.1")
            self.assertEqual(request.url.params["format"], "json")
            return httpx.Response(200, json=panel_document())

        panel = await PanelAppClient(transport=httpx.MockTransport(handler)).get_panel(214, "2.1")
        self.assertEqual(panel["version"], "2.1")
        self.assertEqual(panel["gene_entity_count"], 1)
        self.assertEqual(panel["non_gene_entity_count"], 1)
        self.assertEqual(panel["genes"][0]["confidence_level_source_value"], "3")
        self.assertEqual(panel["genes"][0]["mode_of_inheritance"], "BIALLELIC, autosomal or pseudoautosomal")
        self.assertEqual(panel["genes"][0]["hgnc_id"], "HGNC:1")
        self.assertEqual(len(panel["source_record_sha256"]), 64)
        self.assertIn("not combined", panel["interpretation_limit"])

    async def test_not_found_is_explicit_and_identity_or_version_mismatches_fail_closed(self):
        missing = await PanelAppClient(transport=httpx.MockTransport(lambda request: httpx.Response(404))).get_panel(999)
        self.assertFalse(missing["found"])

        async def wrong_id(request):
            return httpx.Response(200, json=panel_document(panel_id=215))

        with self.assertRaisesRegex(ValueError, "panel ID does not match"):
            await PanelAppClient(transport=httpx.MockTransport(wrong_id)).get_panel(214)

        async def wrong_version(request):
            return httpx.Response(200, json=panel_document(version="2.2"))

        with self.assertRaisesRegex(ValueError, "different version"):
            await PanelAppClient(transport=httpx.MockTransport(wrong_version)).get_panel(214, "2.1")

    async def test_rejects_bad_ids_versions_and_oversized_or_incomplete_payloads(self):
        client = PanelAppClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=panel_document())))
        for panel_id, version in ((True, None), (0, None), (214, "latest")):
            with self.subTest(panel_id=panel_id, version=version), self.assertRaises(ValueError):
                await client.get_panel(panel_id, version)

        async def missing_genes(request):
            document = panel_document()
            del document["genes"]
            return httpx.Response(200, json=document)

        with self.assertRaisesRegex(ValueError, "missing its gene list"):
            await PanelAppClient(transport=httpx.MockTransport(missing_genes)).get_panel(214)


if __name__ == "__main__":
    unittest.main()
