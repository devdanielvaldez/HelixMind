from __future__ import annotations

import asyncio
import unittest

import httpx

from medai_science.sources.openfda_labels import OpenFDALabelClient


def _record(name: str, set_id: str, text: str) -> dict:
    unii = "ASPIRIN001" if name == "aspirin" else "WARFARIN01"
    rxcui = "1191" if name == "aspirin" else "11289"
    return {
        "set_id": set_id,
        "version": "3",
        "effective_time": "20240101",
        "drug_interactions": [text],
        "openfda": {"generic_name": [name], "brand_name": [name.title()], "rxcui": [rxcui], "unii": [unii]},
    }


class _RxNormStub:
    async def resolve_ingredient(self, name: str) -> dict:
        value = {
            "aspirin": {"rxcui": "1191", "names": ["aspirin"], "unii_codes": ["ASPIRIN001"]},
            "warfarin": {"rxcui": "11289", "names": ["warfarin"], "unii_codes": ["WARFARIN01"]},
        }.get(name)
        return {"query": name, "resolution_status": "resolved_unique_ingredient" if value else "unresolved", "ingredients": [value] if value else [], "source": "RxNorm API"}


class OpenFDALabelTests(unittest.TestCase):
    def test_retrieves_versioned_label_sections_and_marks_reciprocal_mentions(self):
        def handler(request: httpx.Request) -> httpx.Response:
            search = request.url.params["search"]
            if "ASPIRIN001" in search:
                rows = [_record("aspirin", "set-a", "Concomitant use with warfarin can increase bleeding risk.")]
            else:
                rows = [_record("warfarin", "set-b", "Aspirin may increase bleeding. See clinical pharmacology.")]
            return httpx.Response(200, json={"meta": {"results": {"total": len(rows)}}, "results": rows})

        result = asyncio.run(OpenFDALabelClient(httpx.MockTransport(handler), rxnorm_client=_RxNormStub()).investigate_pair("aspirin", "warfarin"))
        self.assertEqual(result["label_mention_screen"]["classification"], "both_label_sets_mention_the_other_query_name")
        aspirin_label = result["labels"]["aspirin"][0]
        self.assertEqual(aspirin_label["set_id"], "set-a")
        self.assertEqual(aspirin_label["effective_time"], "20240101")
        self.assertEqual(len(aspirin_label["record_sha256"]), 64)
        self.assertEqual(len(aspirin_label["section_sha256"]), 64)
        self.assertTrue(result["research_only"])
        self.assertIn("not proof", result["label_mention_screen"]["interpretation"])

    def test_asymmetric_mention_is_review_lead_not_contradiction(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if "ASPIRIN001" in request.url.params["search"]:
                rows = [_record("aspirin", "set-a", "Drug interactions have not been studied.")]
            else:
                rows = [_record("warfarin", "set-b", "Avoid concomitant aspirin use.")]
            return httpx.Response(200, json={"meta": {"results": {"total": 1}}, "results": rows})

        result = asyncio.run(OpenFDALabelClient(httpx.MockTransport(handler), rxnorm_client=_RxNormStub()).investigate_pair("aspirin", "warfarin"))
        screen = result["label_mention_screen"]
        self.assertEqual(screen["classification"], "asymmetric_label_mention_requires_review")
        self.assertIn("not a contradiction", screen["interpretation"])

    def test_not_found_and_query_safety(self):
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

        result = asyncio.run(OpenFDALabelClient(httpx.MockTransport(handler), rxnorm_client=_RxNormStub()).investigate_pair("aspirin", "warfarin"))
        self.assertEqual(result["search_limits"]["aspirin"]["provider_total"], 0)
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            OpenFDALabelClient.validate_name("patient id: 123456")
        with self.assertRaisesRegex(ValueError, "distinct"):
            asyncio.run(OpenFDALabelClient(rxnorm_client=_RxNormStub()).investigate_pair("aspirin", "ASPIRIN"))

    def test_ambiguous_rxnorm_result_stops_before_openfda(self):
        class Ambiguous:
            async def resolve_ingredient(self, name: str) -> dict:
                return {"query": name, "resolution_status": "ambiguous_multiple_ingredients", "ingredients": [], "source": "RxNorm API"}

        def unexpected(_: httpx.Request) -> httpx.Response:
            self.fail("openFDA must not be queried after ambiguous RxNorm resolution")

        result = asyncio.run(OpenFDALabelClient(httpx.MockTransport(unexpected), rxnorm_client=Ambiguous()).investigate_pair("aspirin", "warfarin"))
        self.assertEqual(result["label_mention_screen"]["classification"], "unresolved_or_ambiguous_drug_identity")


if __name__ == "__main__":
    unittest.main()
