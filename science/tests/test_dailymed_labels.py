from __future__ import annotations

import asyncio
from io import BytesIO
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

import httpx

from medai_science.sources.dailymed_labels import DailyMedLabelClient, compare_label_distributions


SET_A = "11111111-1111-4111-8111-111111111111"
SET_B = "22222222-2222-4222-8222-222222222222"


def _spl(set_id: str, version: str, text: str) -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <id root="{set_id}-version-{version}" />
      <setId root="{set_id}" />
      <versionNumber value="{version}" />
      <component><structuredBody><component><section>
        <code code="34066-1" codeSystem="2.16.840.1.113883.6.1" displayName="DRUG INTERACTIONS" />
        <title>7 DRUG INTERACTIONS</title>
        <text><paragraph>{text}</paragraph></text>
      </section></component></structuredBody></component>
    </document>'''.encode()


def _archive(xml_bytes: bytes) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("label.xml", xml_bytes)
    return output.getvalue()


class DailyMedLabelTests(unittest.TestCase):
    def test_retrieves_bounded_versioned_interaction_sections_by_exact_rxcui(self):
        calls: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url)
            if request.url.path.endswith("spls.json"):
                self.assertEqual(request.url.params["rxcui"], "1191")
                self.assertEqual(request.url.params["pagesize"], "3")
                return httpx.Response(200, json={"metadata": {"total_elements": "4"}, "data": [
                    {"setid": SET_A, "spl_version": "9", "published_date": "Jan 03, 2025", "title": "ASPIRIN TABLET"},
                    {"setid": SET_B, "spl_version": "2", "published_date": "Feb 03, 2024", "title": "ASPIRIN TABLET"},
                ]})
            if request.url.path.endswith("/history.json"):
                self.assertRegex(request.url.path, rf"/dailymed/services/v2/spls/{SET_A if SET_A in request.url.path else SET_B}/history.json$")
                self.assertEqual(request.url.params["pagesize"], "100")
                return httpx.Response(200, json={"metadata": {"total_elements": "2"}, "data": {"history": [{"spl_version": "8", "published_date": "Jan 01, 2024"}, {"spl_version": "7", "published_date": "Jan 01, 2023"}]}})
            if request.url.path.endswith("getFile.cfm"):
                version = request.url.params["version"]
                self.assertIn(request.url.params["setid"], {SET_A, SET_B})
                self.assertEqual(request.url.params["type"], "zip")
                return httpx.Response(200, content=_archive(_spl(request.url.params["setid"], version, f"Historical version {version}.")))
            if request.url.path.endswith(SET_A + ".xml"):
                return httpx.Response(200, content=_spl(SET_A, "9", "Concomitant use with warfarin can increase bleeding."))
            if request.url.path.endswith(SET_B + ".xml"):
                return httpx.Response(200, content=_spl(SET_B, "2", "Interaction section from an older label."))
            self.fail(f"unexpected DailyMed request: {request.url}")

        result = asyncio.run(DailyMedLabelClient(httpx.MockTransport(handler)).retrieve_ingredient({"rxcui": "1191", "names": ["aspirin"]}))
        labels = result["labels"]
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0]["spl_version"], "9")
        self.assertEqual(labels[0]["interaction_section_code"], "34066-1")
        self.assertIn("warfarin", labels[0]["drug_interactions"])
        self.assertEqual(len(labels[0]["spl_sha256"]), 64)
        self.assertEqual(len(labels[0]["section_sha256"]), 64)
        self.assertEqual(result["search_limits"]["provider_total"], 4)
        self.assertEqual([row["spl_version"] for row in labels[0]["historical_versions"]["records"]], ["8", "7"])
        self.assertEqual(labels[0]["historical_versions"]["records"][0]["drug_interactions"], "Historical version 8.")
        self.assertEqual(len(labels[0]["historical_versions"]["records"][0]["spl_sha256"]), 64)
        self.assertEqual(len([url for url in calls if url.path.endswith(".xml")]), 2)
        self.assertTrue(result["research_only"])

    def test_missing_interaction_section_is_not_reported_as_no_interaction(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("spls.json"):
                return httpx.Response(200, json={"data": [{"setid": SET_A, "spl_version": "1"}]})
            if request.url.path.endswith("/history.json"):
                return httpx.Response(200, json={"data": {"history": []}})
            return httpx.Response(200, content=f'<document xmlns="urn:hl7-org:v3"><title>Drug label {SET_A}</title></document>'.encode())

        result = asyncio.run(DailyMedLabelClient(httpx.MockTransport(handler)).retrieve_ingredient({"rxcui": "1191", "names": ["aspirin"]}))
        self.assertEqual(result["labels"][0]["drug_interactions"], "")
        self.assertIsNone(result["labels"][0]["section_sha256"])

    def test_rejects_invalid_identity_and_xml_entities(self):
        with self.assertRaisesRegex(ValueError, "validated RxNorm"):
            asyncio.run(DailyMedLabelClient().retrieve_ingredient({"rxcui": "../../spls", "names": []}))
        with self.assertRaisesRegex(ValueError, "entity declaration"):
            DailyMedLabelClient._interaction_section(b'<!DOCTYPE x [<!ENTITY e "boom">]><document/>')

    def test_rejects_malformed_and_oversized_historical_archives(self):
        async def retrieve(content: bytes) -> bytes:
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, content=content)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
                return await DailyMedLabelClient()._historical_xml(client, SET_A, "2")

        with self.assertRaisesRegex(ValueError, "malformed historical"):
            asyncio.run(retrieve(b"not a zip archive"))
        oversized = _archive(b"<document>" + b"0" * (8 * 1024 * 1024 + 1) + b"</document>")
        with self.assertRaisesRegex(ValueError, "unsupported file layout or size"):
            asyncio.run(retrieve(oversized))

    def test_pair_records_shared_label_origin_limit(self):
        def handler(request: httpx.Request) -> httpx.Response:
            rxcui = request.url.params.get("rxcui")
            set_id = SET_A if rxcui == "1191" else SET_B
            if request.url.path.endswith("spls.json"):
                return httpx.Response(200, json={"data": [{"setid": set_id, "spl_version": "1"}]})
            if request.url.path.endswith("/history.json"):
                return httpx.Response(200, json={"data": {"history": []}})
            return httpx.Response(200, content=_spl(set_id, "1", "No interaction information."))

        client = DailyMedLabelClient(httpx.MockTransport(handler))
        identities = [
            {"resolution_status": "resolved_unique_ingredient", "ingredients": [{"rxcui": "1191", "names": ["aspirin"]}]},
            {"resolution_status": "resolved_unique_ingredient", "ingredients": [{"rxcui": "11289", "names": ["warfarin"]}]},
        ]
        result = asyncio.run(client.investigate_pair("aspirin", "warfarin", identities))
        self.assertIn("not independent clinical evidence", result["interpretation"])
        self.assertFalse(result["clinical_advice"])

    def test_distribution_comparison_is_bound_to_shared_setid_and_not_evidence_grading(self):
        openfda = {
            "drug_queries": ["aspirin", "warfarin"],
            "ingredient_resolutions": [
                {"ingredients": [{"rxcui": "1191", "names": ["aspirin"]}]},
                {"ingredients": [{"rxcui": "11289", "names": ["warfarin"]}]},
            ],
            "labels": {
                "aspirin": [{"set_id": SET_A, "version": "8", "drug_interactions": "Concomitant use with warfarin can increase bleeding."}],
                "warfarin": [],
            },
        }
        dailymed = {"labels": {"aspirin": [{"set_id": SET_A, "spl_version": "9", "published_date": "Jan 03, 2025", "section_sha256": "b" * 64, "drug_interactions": "Warfarin can increase bleeding risk.", "historical_versions": {"records": [{"spl_version": "8", "published_date": "Jan 03, 2024", "section_sha256": "a" * 64, "history_record_sha256": "c" * 64, "spl_sha256": "d" * 64}]}}], "warfarin": []}}
        result = compare_label_distributions(openfda, dailymed)
        self.assertEqual(result["status"], "shared_label_distributions_compared")
        comparison = result["shared_spl_records"][0]
        self.assertEqual(comparison["set_id"], SET_A)
        self.assertFalse(comparison["version_match"])
        self.assertFalse(result["independent_clinical_evidence"])
        self.assertFalse(result["clinical_advice"])
        self.assertIn("not a clinical contradiction", result["interpretation"])
        self.assertEqual(result["historical_version_diffs"][0]["historical_version"], "8")
        self.assertFalse(result["historical_version_diffs"][0]["interaction_section_hash_match"])


if __name__ == "__main__":
    unittest.main()
