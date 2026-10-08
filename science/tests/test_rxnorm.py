from __future__ import annotations

import asyncio
import unittest

import httpx

from medai_science.sources.rxnorm import RxNormClient


class RxNormTests(unittest.TestCase):
    def test_resolves_a_unique_ingredient_without_approximate_search_and_retains_unii(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/rxcui.json"):
                self.assertEqual(request.url.params["search"], "2")
                self.assertEqual(request.url.params["allsrc"], "0")
                return httpx.Response(200, json={"idGroup": {"rxnormId": ["1191"]}})
            if request.url.path.endswith("/related.json"):
                return httpx.Response(200, json={"relatedGroup": {"conceptGroup": [{"tty": "IN", "conceptProperties": [{"rxcui": "1191", "name": "aspirin", "tty": "IN"}]}]}})
            return httpx.Response(200, json={"propConceptGroup": {"propConcept": [{"propName": "RxNorm Name", "propValue": "aspirin"}, {"propName": "UNII_CODE", "propValue": "R16CO5Y76E"}]}})

        result = asyncio.run(RxNormClient(httpx.MockTransport(handler)).resolve_ingredient("aspirin"))
        self.assertEqual(result["resolution_status"], "resolved_unique_ingredient")
        self.assertEqual(result["ingredients"][0]["rxcui"], "1191")
        self.assertEqual(result["ingredients"][0]["unii_codes"], ["R16CO5Y76E"])
        self.assertIn("approximate matching was not used", result["search_precision"])
        self.assertEqual(len(result["ingredients"][0]["record_sha256"]), 64)

    def test_multiple_ingredient_concepts_remain_ambiguous(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/rxcui.json"):
                return httpx.Response(200, json={"idGroup": {"rxnormId": ["111", "222"]}})
            if request.url.path.endswith("/related.json"):
                identifier = request.url.path.split("/")[-2]
                ingredient = "101" if identifier == "111" else "202"
                return httpx.Response(200, json={"relatedGroup": {"conceptGroup": [{"tty": "IN", "conceptProperties": [{"rxcui": ingredient, "tty": "IN"}]}]}})
            return httpx.Response(200, json={"propConceptGroup": {"propConcept": []}})

        result = asyncio.run(RxNormClient(httpx.MockTransport(handler)).resolve_ingredient("combination drug"))
        self.assertEqual(result["resolution_status"], "ambiguous_multiple_ingredients")
        self.assertEqual(len(result["ingredients"]), 2)

    def test_rejects_patient_identifiers(self):
        with self.assertRaisesRegex(ValueError, "patient-identifying"):
            asyncio.run(RxNormClient().resolve_ingredient("patient id: 123456"))


if __name__ == "__main__":
    unittest.main()
