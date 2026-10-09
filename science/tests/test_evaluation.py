from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from medai_science.evaluation import _wilson_interval, _wilson_precision_lower_bound, calibrate_capability_reuse_threshold, evaluate_adjudicated_cases, evaluate_drug_pair_benchmark, load_pinned_capability_calibration, write_private_report_no_overwrite


class BenchmarkEvaluationTests(unittest.TestCase):
    def test_wilson_intervals_cover_small_and_boundary_samples(self):
        interval = _wilson_interval(1, 2)
        self.assertIsNotNone(interval)
        self.assertAlmostEqual(interval["lower"], 0.09453120573423074)
        self.assertAlmostEqual(interval["upper"], 0.9054687942657693)
        all_successes = _wilson_interval(2, 2)
        self.assertGreater(all_successes["lower"], 0.3)
        self.assertEqual(all_successes["upper"], 1.0)
        self.assertIsNone(_wilson_interval(0, 0))
        with self.assertRaisesRegex(ValueError, "integers"):
            _wilson_interval(True, 2)
        with self.assertRaisesRegex(ValueError, "0 <= successes"):
            _wilson_interval(3, 2)

    def test_categorical_metrics_include_nominal_binomial_uncertainty(self):
        result = evaluate_drug_pair_benchmark(self.drug_pair_benchmark())
        metrics = result["retrieval"]["overall"]
        self.assertEqual(metrics["accuracy_wilson_95_ci"], _wilson_interval(3, 5))
        retrieved = metrics["per_label"]["retrieved"]
        self.assertEqual(retrieved["precision_wilson_95_ci"], _wilson_interval(2, 3))
        self.assertEqual(retrieved["recall_wilson_95_ci"], _wilson_interval(2, 3))
        self.assertIn("assume independent cases", metrics["interval_interpretation"])

    @staticmethod
    def drug_pair_benchmark():
        protocol = "Synthetic evaluator fixture only. Two blinded independent primary reviewers; distinct third reviewer resolves disagreements."

        def row(case_id, evidence_type, digest, expected, predicted, confidence=0.8):
            reviewers = [
                {"reviewer_id": "r1", "role": "reviewer", "label": expected},
                {"reviewer_id": "r2", "role": "reviewer", "label": expected},
            ]
            return {"case_id": case_id, "evidence_sha256": digest, "evidence_type": evidence_type,
                    "reviewers": reviewers, "predicted": predicted, "confidence": confidence}

        retrieval_cases = [
            row("label-1", "versioned_label", "a" * 64, "retrieved", "retrieved"),
            row("label-2", "versioned_label", "b" * 64, "retrieved", "not_retrieved", 0.6),
            row("pubmed-1", "pubmed_record", "c" * 64, "retrieved", "retrieved"),
            row("pubmed-2", "pubmed_record", "d" * 64, "not_retrieved", "not_retrieved"),
            row("pmc-1", "pmc_full_text", "e" * 64, "not_retrieved", "retrieved", 0.7),
        ]
        conflict_cases = [
            row("label-1", "versioned_label", "a" * 64, "conflict", "conflict"),
            row("pubmed-1", "pubmed_record", "c" * 64, "no_conflict", "conflict", 0.55),
        ]

        def stage(cases):
            return {"schema_version": "2", "benchmark_id": "synthetic-drug-pair-evaluator-fixture",
                    "labeling_protocol": protocol, "cases": cases}

        return {"schema_version": "1", "benchmark_id": "synthetic-drug-pair-evaluator-fixture",
                "labeling_protocol": protocol, "retrieval": stage(retrieval_cases),
                "conflict_screen": stage(conflict_cases)}

    def test_drug_pair_benchmark_separates_retrieval_and_conditional_conflict_metrics(self):
        result = evaluate_drug_pair_benchmark(self.drug_pair_benchmark(), bins=5)
        self.assertEqual(result["retrieval"]["overall"]["n"], 5)
        self.assertAlmostEqual(result["retrieval"]["overall"]["per_label"]["retrieved"]["recall"], 2 / 3)
        self.assertEqual(result["retrieval"]["by_evidence_type"]["versioned_label"]["n"], 2)
        self.assertEqual(result["retrieval"]["by_evidence_type"]["pubmed_record"]["per_label"]["retrieved"]["precision"], 1.0)
        self.assertEqual(result["conflict_screen"]["overall"]["n"], 2)
        self.assertEqual(result["conflict_screen"]["by_evidence_type"]["versioned_label"]["n"], 1)
        self.assertIn("conditional on retrieval", result["conflict_screen_population"])
        self.assertEqual(len(result["benchmark_sha256"]), 64)
        self.assertIn("Fixture benchmarks validate evaluator logic only", result["interpretation_limit"])

    def test_drug_pair_benchmark_rejects_unretrieved_or_mismatched_source_screen_cases(self):
        benchmark = self.drug_pair_benchmark()
        benchmark["conflict_screen"]["cases"][0]["evidence_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "exact source digest"):
            evaluate_drug_pair_benchmark(benchmark)
        benchmark = self.drug_pair_benchmark()
        benchmark["conflict_screen"]["cases"].append({**benchmark["retrieval"]["cases"][1], "reviewers": [
            {"reviewer_id": "r1", "role": "reviewer", "label": "conflict"},
            {"reviewer_id": "r2", "role": "reviewer", "label": "conflict"}], "predicted": "conflict"})
        with self.assertRaisesRegex(ValueError, "exact source digest"):
            evaluate_drug_pair_benchmark(benchmark)

    def test_drug_pair_benchmark_requires_distinct_adjudication_and_task_scales(self):
        benchmark = self.drug_pair_benchmark()
        case = benchmark["retrieval"]["cases"][0]
        case["reviewers"][1]["label"] = "not_retrieved"
        with self.assertRaisesRegex(ValueError, "third-party adjudicator"):
            evaluate_drug_pair_benchmark(benchmark)
        benchmark = self.drug_pair_benchmark()
        benchmark["conflict_screen"]["cases"][0]["reviewers"][0]["label"] = "retrieved"
        with self.assertRaisesRegex(ValueError, "task-specific scale"):
            evaluate_drug_pair_benchmark(benchmark)

    def test_drug_pair_benchmark_rejects_non_scalar_labels_as_validation_errors(self):
        mutations = (
            ("evidence_type", ["versioned_label"], "unsupported evidence_type"),
            ("predicted", ["retrieved"], "prediction is outside"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                benchmark = self.drug_pair_benchmark()
                benchmark["retrieval"]["cases"][0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_drug_pair_benchmark(benchmark)

        benchmark = self.drug_pair_benchmark()
        benchmark["conflict_screen"]["cases"][0]["reviewers"][0]["label"] = ["conflict"]
        with self.assertRaisesRegex(ValueError, "label outside its task-specific scale"):
            evaluate_drug_pair_benchmark(benchmark)

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

    def test_double_adjudicated_v2_rejects_repeated_source_digest(self):
        benchmark = {
            "schema_version": "2", "benchmark_id": "duplicate-source-fixture", "labeling_protocol": "Two primary reviewers.",
            "cases": [
                {"case_id": "case-one", "evidence_sha256": "e" * 64, "evidence_type": "paper", "reviewers": [
                    {"reviewer_id": "r1", "role": "reviewer", "label": "supports"},
                    {"reviewer_id": "r2", "role": "reviewer", "label": "supports"},
                ], "predicted": "supports", "confidence": 0.8},
                {"case_id": "case-two", "evidence_sha256": "e" * 64, "evidence_type": "paper", "reviewers": [
                    {"reviewer_id": "r1", "role": "reviewer", "label": "supports"},
                    {"reviewer_id": "r2", "role": "reviewer", "label": "supports"},
                ], "predicted": "supports", "confidence": 0.8},
            ],
        }
        with self.assertRaisesRegex(ValueError, "unique to avoid repeated-source"):
            evaluate_adjudicated_cases(benchmark)

    def test_cohen_kappa_is_grouped_by_reviewer_pair_and_independent_of_input_order(self):
        protocol = "Two blinded primary reviewers label each source; a distinct third reviewer resolves disagreement."

        def case(case_id, digest, reviewers, prediction):
            return {"case_id": case_id, "evidence_sha256": digest * 64, "evidence_type": "paper",
                    "reviewers": reviewers, "predicted": prediction, "confidence": 0.8}

        def reviewers(first, first_label, second, second_label):
            return [
                {"reviewer_id": first, "role": "reviewer", "label": first_label},
                {"reviewer_id": second, "role": "reviewer", "label": second_label},
            ]
        benchmark = {"schema_version": "2", "benchmark_id": "multiple-review-panels", "labeling_protocol": protocol,
                     "cases": [
                         case("ab-1", "a", reviewers("reviewer-a", "supports", "reviewer-b", "supports"), "supports"),
                         case("ab-2", "b", reviewers("reviewer-b", "contradicts", "reviewer-a", "contradicts"), "contradicts"),
                         case("cd-1", "c", reviewers("reviewer-c", "supports", "reviewer-d", "contradicts") + [{"reviewer_id": "reviewer-x", "role": "adjudicator", "label": "supports"}], "supports"),
                         case("cd-2", "d", reviewers("reviewer-c", "contradicts", "reviewer-d", "supports") + [{"reviewer_id": "reviewer-y", "role": "adjudicator", "label": "contradicts"}], "contradicts"),
                     ]}
        quality = evaluate_adjudicated_cases(benchmark)["adjudication_quality"]["primary_reviewers"]
        self.assertEqual(quality["raw_agreement"], 0.5)
        self.assertIsNone(quality["cohen_kappa"])
        self.assertEqual(quality["cohen_kappa_by_reviewer_pair"]["reviewer-a|reviewer-b"], {"n": 2, "cohen_kappa": 1.0})
        self.assertEqual(quality["cohen_kappa_by_reviewer_pair"]["reviewer-c|reviewer-d"], {"n": 2, "cohen_kappa": -1.0})

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
        self.assertGreaterEqual(result["holdout_metrics"]["precision_lower_95_one_sided_wilson"], 0.95)
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

    def test_perfect_but_small_holdout_does_not_claim_high_precision_certainty(self):
        self.assertLess(_wilson_precision_lower_bound(20, 20), 0.95)
        self.assertGreaterEqual(_wilson_precision_lower_bound(120, 120), 0.95)
        with self.assertRaisesRegex(ValueError, "positive trials"):
            _wilson_precision_lower_bound(0, 0)

    def test_calibration_report_is_private_atomic_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "nested" / "calibration.json"
            digest = write_private_report_no_overwrite(target, '{"threshold":0.9}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"threshold":0.9}\n')
            self.assertEqual(digest, hashlib.sha256(target.read_bytes()).hexdigest())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_private_report_no_overwrite(target, '{"threshold":0.5}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"threshold":0.9}\n')

    def test_calibration_report_rejects_symlink_outputs(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks are unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / "protected.json"
            protected.write_text("keep", encoding="utf-8")
            link = root / "calibration.json"
            link.symlink_to(protected)
            with self.assertRaisesRegex(ValueError, "cannot be a symlink"):
                write_private_report_no_overwrite(link, "replace")
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
