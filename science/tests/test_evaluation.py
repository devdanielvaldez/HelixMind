from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medai_science.evaluation import calibrate_capability_reuse_threshold, evaluate_adjudicated_cases, load_pinned_capability_calibration


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_double_adjudicated_v2_reports_primary_agreement_and_third_party_resolution(self):
        benchmark = {
            "schema_version": "2",
            "benchmark_id": "double-review-fixture-v1",
            "labeling_protocol": "Two blinded independent reviewers; a distinct third reviewer resolves disagreement.",
            "cases": [
                {"case_id": "agree-1", "evidence_sha256": "a" * 64, "evidence_type": "paper", "reviewers": [{"reviewer_id": "r1", "role": "reviewer", "label": "supports"}, {"reviewer_id": "r2", "role": "reviewer", "label": "supports"}], "predicted": "supports", "confidence": 0.9},
                {"case_id": "agree-2", "evidence_sha256": "b" * 64, "evidence_type": "paper", "reviewers": [{"reviewer_id": "r1", "role": "reviewer", "label": "contradicts"}, {"reviewer_id": "r2", "role": "reviewer", "label": "contradicts"}], "predicted": "supports", "confidence": 0.6},
                {"case_id": "resolve-1", "evidence_sha256": "c" * 64, "evidence_type": "abstract", "reviewers": [{"reviewer_id": "r1", "role": "reviewer", "label": "supports"}, {"reviewer_id": "r2", "role": "reviewer", "label": "contradicts"}, {"reviewer_id": "r3", "role": "adjudicator", "label": "context"}], "predicted": "context", "confidence": 0.8},
            ],
        }
        result = evaluate_adjudicated_cases(benchmark)
        expected_digest = hashlib.sha256(json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        quality = result["adjudication_quality"]
        self.assertEqual(result["overall"]["n"], 3)
        self.assertEqual(result["overall"]["accuracy"], 2 / 3)
        self.assertEqual(quality["primary_reviewers"]["raw_agreement"], 2 / 3)
        self.assertEqual(quality["primary_reviewers_by_evidence_type"]["paper"]["n"], 2)
        self.assertEqual(quality["primary_reviewers_by_evidence_type"]["abstract"]["n"], 1)
        self.assertEqual(quality["disagreement_count"], 1)
        self.assertEqual(quality["third_party_adjudication_count"], 1)
        self.assertEqual(quality["source_anchored_case_count"], 3)
        self.assertEqual(result["benchmark_sha256"], expected_digest)
        self.assertIn("not establish truth", quality["interpretation_limit"])

    def test_double_adjudication_requires_distinct_third_reviewer_for_disagreement(self):
        benchmark = {
            "schema_version": "2", "benchmark_id": "invalid-review-fixture", "labeling_protocol": "Independent review.",
            "cases": [{"case_id": "case-1", "evidence_sha256": "d" * 64, "evidence_type": "paper", "reviewers": [
                {"reviewer_id": "r1", "role": "reviewer", "label": "supports"},
                {"reviewer_id": "r2", "role": "reviewer", "label": "contradicts"},
            ], "predicted": "supports", "confidence": 0.5}],
        }
        with self.assertRaisesRegex(ValueError, "third-party adjudicator"):
            evaluate_adjudicated_cases(benchmark)

    @staticmethod
    def capability_benchmark():
        rows = []
        for index in range(600):
            reusable = index % 2 == 0
            rows.append({
                "case_id": f"pair-{index:04d}",
                "task_sha256": hashlib.sha256(f"task-{index}".encode()).hexdigest(),
                "capability_id": "biomedical.protein.structure",
                "capability_version": "2.1.0",
                "expected_reuse": reusable,
                "semantic_similarity": 0.9 if reusable else 0.4,
                "contract_compatible": True,
                "permissions_compatible": True,
                "lifecycle_approved": True,
                "runtime_active": True,
            })
        return {
            "schema_version": "1",
            "benchmark_id": "synthetic-calibration-logic-only",
            "adjudication_protocol": "Synthetic labels used only to exercise calibration code paths; not scientific evidence.",
            "embedding_model_id": "fixture-embedder-v1",
            "pairs": rows,
        }

    def test_reports_classification_and_correctness_calibration_per_evidence_type(self):
        benchmark = {
            "schema_version": "1",
            "benchmark_id": "fixture-drug-interaction-screen-v1",
            "labeling_protocol": "Two independent reviewers adjudicated the fixture; disagreements resolved by a third.",
            "cases": [
                {"case_id": "a", "evidence_type": "label", "expected": "supports", "predicted": "supports", "confidence": 0.9},
                {"case_id": "b", "evidence_type": "label", "expected": "contradicts", "predicted": "supports", "confidence": 0.6},
                {"case_id": "c", "evidence_type": "paper", "expected": "supports", "predicted": "__abstain__", "confidence": 0.4},
                {"case_id": "d", "evidence_type": "paper", "expected": "contradicts", "predicted": "contradicts", "confidence": 0.8},
            ],
        }
        result = evaluate_adjudicated_cases(benchmark, bins=5)
        self.assertEqual(result["overall"]["n"], 4)
        self.assertEqual(result["overall"]["accuracy"], 0.5)
        self.assertEqual(result["overall"]["abstention_rate"], 0.25)
        self.assertAlmostEqual(result["overall"]["correctness_brier_score"], 0.1425)
        self.assertEqual(result["by_evidence_type"]["label"]["n"], 2)
        self.assertEqual(result["by_evidence_type"]["paper"]["confusion_matrix"]["supports"]["__abstain__"], 1)
        self.assertEqual(len(result["benchmark_sha256"]), 64)
        self.assertIn("not probability", result["overall"]["confidence_semantics"])

    def test_rejects_missing_adjudication_protocol_duplicate_ids_and_invalid_confidence(self):
        case = {"case_id": "one", "evidence_type": "paper", "expected": "supports", "predicted": "supports", "confidence": 0.5}
        base = {"schema_version": "1", "benchmark_id": "test", "labeling_protocol": "Independent adjudication.", "cases": [case]}
        with self.assertRaisesRegex(ValueError, "adjudication protocol"):
            evaluate_adjudicated_cases({**base, "labeling_protocol": ""})
        with self.assertRaisesRegex(ValueError, "unique"):
            evaluate_adjudicated_cases({**base, "cases": [case, case]})
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_adjudicated_cases({**base, "cases": [{**case, "confidence": float("nan")}]})

    def test_calibrates_on_fit_pairs_then_requires_holdout_and_pinned_report_digest(self):
        benchmark = self.capability_benchmark()
        result = calibrate_capability_reuse_threshold(benchmark)
        self.assertEqual(result["selected_threshold"], 0.9)
        self.assertEqual(result["status"], "holdout_passed")
        self.assertGreaterEqual(result["holdout_metrics"]["n_predicted_reuse"], 20)
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "calibration.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
            loaded = load_pinned_capability_calibration(report_path, report_sha256, expected_embedding_model_id="fixture-embedder-v1")
            self.assertEqual(loaded["threshold"], 0.9)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_pinned_capability_calibration(report_path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_pinned_capability_calibration(report_path, report_sha256, expected_embedding_model_id="different-model")

    def test_uncalibrated_or_incompatible_capability_pairs_cannot_authorize_threshold(self):
        benchmark = self.capability_benchmark()
        benchmark["pairs"] = [{**row, "semantic_similarity": 0.7} for row in benchmark["pairs"]]
        with self.assertRaisesRegex(ValueError, "no semantic threshold"):
            calibrate_capability_reuse_threshold(benchmark)
        benchmark = self.capability_benchmark()
        benchmark["pairs"] = [{**row, "runtime_active": False} for row in benchmark["pairs"]]
        with self.assertRaisesRegex(ValueError, "at least 10 positive"):
            calibrate_capability_reuse_threshold(benchmark)


if __name__ == "__main__":
    unittest.main()
