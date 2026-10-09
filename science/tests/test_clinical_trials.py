from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
from typing import Any
import unittest

import httpx

from medai_science.sources.clinical_trials import ClinicalTrialsClient, compare_study_snapshots, latest_retained_study_snapshot, normalize_study


def _study() -> dict[str, Any]:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT01234567",
                "briefTitle": "A study of a research intervention",
                "officialTitle": "A randomized study",
                "orgStudyIdInfo": {"id": "ORG-7"},
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "whyStopped": None,
                "studyStartDateStruct": {"date": "2020-01", "type": "ACTUAL"},
                "primaryCompletionDateStruct": {"date": "2021-02", "type": "ACTUAL"},
                "completionDateStruct": {"date": "2021-04", "type": "ACTUAL"},
                "studyFirstSubmitDate": "2019-11-01",
                "studyFirstPostDateStruct": {"date": "2019-11-04", "type": "ACTUAL"},
                "studyLastUpdateSubmitDate": "2022-01-03",
                "studyLastUpdatePostDateStruct": {"date": "2022-01-07", "type": "ACTUAL"},
                "studyLastVerifiedDate": "2021-12",
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
                "designInfo": {
                    "allocation": "RANDOMIZED",
                    "interventionModel": "PARALLEL",
                    "primaryPurpose": "TREATMENT",
                    "maskingInfo": {"masking": "DOUBLE", "whoMasked": ["PARTICIPANT", "INVESTIGATOR"]},
                },
                "enrollmentInfo": {"count": 120, "type": "ACTUAL"},
            },
            "eligibilityModule": {
                "eligibilityCriteria": "Inclusion: adults. Exclusion: prior study participation.",
                "healthyVolunteers": "NO",
                "sex": "ALL",
                "minimumAge": "18 Years",
                "maximumAge": "75 Years",
                "stdAges": ["ADULT", "OLDER_ADULT"],
                "studyPopulation": "Adults at participating sites.",
                "samplingMethod": "NON_PROBABILITY_SAMPLE",
            },
            "conditionsModule": {"conditions": ["Example disease"]},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "Example agent", "description": "Study intervention", "armGroupLabels": ["A"], "otherNames": ["EX-1"]}]
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Primary endpoint", "timeFrame": "12 weeks"}],
                "secondaryOutcomes": [{"measure": "Secondary endpoint"}],
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Example sponsor", "class": "INDUSTRY"},
                "collaborators": [{"name": "University", "class": "OTHER"}],
            },
            "contactsLocationsModule": {"locations": [{"facility": "Site A", "city": "Boston", "country": "United States", "status": "COMPLETED"}]},
        },
        "hasResults": True,
        "resultsSection": {"participantFlowModule": {}},
    }


class ClinicalTrialsTests(unittest.TestCase):
    def test_normalizes_trial_lifecycle_and_keeps_source_record(self) -> None:
        raw = _study()
        normalized = normalize_study(raw, "2026-10-07T12:00:00Z")
        self.assertEqual(normalized["nct_id"], "NCT01234567")
        self.assertEqual(normalized["lifecycle"]["overall_status"], "COMPLETED")
        self.assertEqual(normalized["lifecycle"]["last_update_submitted"], "2022-01-03")
        self.assertEqual(normalized["lifecycle"]["last_update_posted"]["date"], "2022-01-07")
        self.assertFalse(normalized["lifecycle"]["record_history_included"])
        self.assertEqual(normalized["lifecycle"]["record_history_url"], "https://clinicaltrials.gov/study/NCT01234567?tab=history")
        self.assertEqual(normalized["lifecycle"]["record_history_access"], "official_web_interface")
        self.assertEqual(normalized["lifecycle"]["record_history_api_backfill"], "unavailable_via_documented_v2_adapter")
        self.assertEqual(normalized["design"]["enrollment"], {"count": 120, "type": "ACTUAL"})
        self.assertEqual(normalized["eligibility"]["criteria_text"], raw["protocolSection"]["eligibilityModule"]["eligibilityCriteria"])
        self.assertFalse(normalized["eligibility"]["parsed_for_individual_suitability"])
        self.assertEqual(normalized["record"], raw)
        self.assertEqual(len(normalized["provenance"]["record_snapshot_sha256"]), 64)
        self.assertEqual(normalized["provenance"]["record_url"], "https://clinicaltrials.gov/study/NCT01234567")

    def test_search_and_get_study_use_the_public_v2_api(self) -> None:
        raw = _study()
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(str(request.url))
            if request.url.path.endswith("/NCT01234567"):
                return httpx.Response(200, json=raw)
            return httpx.Response(200, json={"studies": [raw]})

        client = ClinicalTrialsClient(transport=httpx.MockTransport(handler))
        searched = asyncio.run(client.search("example disease", limit=1))
        fetched = asyncio.run(client.get_study("NCT01234567"))
        self.assertEqual(searched[0]["nct_id"], "NCT01234567")
        self.assertEqual(fetched["record"], raw)
        self.assertEqual(len(paths), 2)
        self.assertIn("pageSize=1", paths[0])
        self.assertTrue(paths[1].endswith("/NCT01234567?format=json"))

    def test_rejects_identifying_queries_invalid_ids_and_malformed_records(self) -> None:
        client = ClinicalTrialsClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={})))
        with self.assertRaisesRegex(ValueError, "personal or patient-identifying"):
            asyncio.run(client.search("patient MRN: ABC123"))
        with self.assertRaisesRegex(ValueError, "NCT followed by eight digits"):
            asyncio.run(client.get_study("NCT123"))
        with self.assertRaisesRegex(ValueError, "valid NCT identifier"):
            normalize_study({"protocolSection": {"identificationModule": {"nctId": "not-an-id"}}})

    def test_search_rejects_missing_or_malformed_studies_collection(self) -> None:
        for body in (b'{"totalCount":0}', b'{"studies":[null]}'):
            client = ClinicalTrialsClient(transport=httpx.MockTransport(lambda _, value=body: httpx.Response(200, content=value)))
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "studies response|object collection"):
                asyncio.run(client.search("example disease"))

    def test_responses_reject_duplicate_json_keys_and_nonstandard_numbers(self) -> None:
        for body, expected in (
            (b'{"studies":[],"studies":[]}', "duplicate JSON key"),
            (b'{"studies":[],"score":NaN}', "non-standard JSON numeric constant"),
        ):
            client = ClinicalTrialsClient(transport=httpx.MockTransport(lambda _, value=body: httpx.Response(200, content=value)))
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, expected):
                asyncio.run(client.search("example disease"))

    def test_get_study_rejects_a_valid_record_for_another_nct_identifier(self) -> None:
        client = ClinicalTrialsClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_study())))
        with self.assertRaisesRegex(ValueError, "does not match the requested study"):
            asyncio.run(client.get_study("NCT99999999"))

    def test_normalization_rejects_malformed_present_module_shapes(self) -> None:
        raw = _study()
        raw["protocolSection"]["statusModule"] = []
        with self.assertRaisesRegex(ValueError, "malformed object field"):
            normalize_study(raw)

    def test_compares_registry_snapshots_without_interpreting_changes(self) -> None:
        earlier = _study()
        later = deepcopy(earlier)
        later["protocolSection"]["statusModule"]["overallStatus"] = "TERMINATED"
        later["protocolSection"]["statusModule"]["whyStopped"] = "Sponsor decision"
        later["protocolSection"]["eligibilityModule"]["maximumAge"] = "70 Years"
        change = compare_study_snapshots(normalize_study(earlier), normalize_study(later))
        self.assertTrue(change["changed"])
        self.assertEqual(change["previous_status"], "COMPLETED")
        self.assertEqual(change["current_status"], "TERMINATED")
        self.assertIn("protocolSection.statusModule.overallStatus", change["changed_field_paths"])
        self.assertIn("protocolSection.eligibilityModule.maximumAge", change["changed_field_paths"])
        self.assertEqual(change["scientific_or_safety_interpretation"], "not_performed")

    def test_comparison_rejects_a_tampered_normalized_snapshot(self) -> None:
        original = normalize_study(_study())
        changed = deepcopy(original)
        changed["lifecycle"]["overall_status"] = "TERMINATED"
        with self.assertRaisesRegex(ValueError, "do not match its retained source record"):
            compare_study_snapshots(original, changed)

    def test_selects_latest_same_session_artifact_and_ignores_other_trials(self) -> None:
        earlier = normalize_study(_study())
        later_raw = deepcopy(_study())
        later_raw["protocolSection"]["statusModule"]["overallStatus"] = "TERMINATED"
        later = normalize_study(later_raw)
        def artifact(kind: str, nct_id: str, recorded_at: int, snapshot: dict[str, Any]) -> dict[str, Any]:
            payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return {
                "id": f"artifact-{recorded_at}", "kind": kind, "source": "ClinicalTrials.gov",
                "source_id": nct_id, "recorded_at": recorded_at, "payload": payload,
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }

        selected = latest_retained_study_snapshot([
            artifact("clinical_trial_snapshot", "NCT99999999", 99, earlier),
            artifact("clinical_trial_snapshot", "NCT01234567", 1, earlier),
            artifact("clinical_trial", "NCT01234567", 2, later),
        ], "NCT01234567")
        self.assertEqual(selected["lifecycle"]["overall_status"], "TERMINATED")
        self.assertIsNone(latest_retained_study_snapshot([], "NCT01234567"))

    def test_latest_snapshot_rejects_corrupt_artifact_payload_or_source_record_digest(self) -> None:
        snapshot = normalize_study(_study())
        payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row = {
            "id": "artifact-1", "kind": "clinical_trial_snapshot", "source": "ClinicalTrials.gov",
            "source_id": "NCT01234567", "recorded_at": 1, "payload": payload,
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        row["payload"] = payload + " "
        with self.assertRaisesRegex(ValueError, "payload digest"):
            latest_retained_study_snapshot([row], "NCT01234567")
        row["payload"] = payload
        tampered = deepcopy(snapshot)
        tampered["lifecycle"]["overall_status"] = "TERMINATED"
        row["payload"] = json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row["sha256"] = hashlib.sha256(row["payload"].encode("utf-8")).hexdigest()
        with self.assertRaisesRegex(ValueError, "do not match its retained source record"):
            latest_retained_study_snapshot([row], "NCT01234567")


if __name__ == "__main__":
    unittest.main()
