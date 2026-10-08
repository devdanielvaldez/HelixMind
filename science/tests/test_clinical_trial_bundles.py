from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from medai_science.reproducibility import (
    create_clinical_trial_bundle,
    verify_analysis_bundle,
    verify_clinical_trial_bundle,
)
from medai_science.sources.clinical_trials import normalize_study


def trial_snapshot() -> dict:
    source_record = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT01234567",
                "briefTitle": "A registry snapshot reproducibility fixture",
            },
            "statusModule": {"overallStatus": "RECRUITING"},
        },
        "hasResults": False,
    }
    return normalize_study(source_record, "2026-10-08T12:00:00Z")


class ClinicalTrialBundleTests(unittest.TestCase):
    def test_bundle_verifies_and_replays_without_network(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-trial-bundle-") as temp:
            bundle = Path(temp) / "trial"
            created = create_clinical_trial_bundle(bundle, trial_snapshot())
            self.assertFalse(created["network_accessed_during_replay"])
            self.assertTrue(verify_clinical_trial_bundle(bundle)["valid"])
            replay = verify_analysis_bundle(bundle, replay=True)
            self.assertTrue(replay["replayed"])
            self.assertFalse(replay["network_accessed"])

    def test_creation_rejects_normalized_fields_not_derived_from_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-trial-bundle-invalid-") as temp:
            snapshot = trial_snapshot()
            snapshot["lifecycle"]["overall_status"] = "COMPLETED"
            with self.assertRaisesRegex(ValueError, "do not match"):
                create_clinical_trial_bundle(Path(temp) / "invalid", snapshot)

    def test_tampered_snapshot_fails_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-trial-bundle-tamper-") as temp:
            bundle = Path(temp) / "trial"
            create_clinical_trial_bundle(bundle, trial_snapshot())
            report_path = bundle / "trial.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["snapshot"]["lifecycle"]["overall_status"] = "COMPLETED"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity check failed|report hash mismatch"):
                verify_clinical_trial_bundle(bundle, replay=True)


if __name__ == "__main__":
    unittest.main()
