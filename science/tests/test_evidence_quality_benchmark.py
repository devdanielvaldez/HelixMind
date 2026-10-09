from __future__ import annotations

import unittest

from medai_science.evaluation import _wilson_interval, evaluate_adjudicated_cases, evaluate_ordinal_quality_cases
from medai_science.quality import QUALITY_DOMAINS


def quality_benchmark() -> dict:
    scales = {
        domain: {"ordered_levels": ["low", "moderate", "high"], "non_ordinal_labels": ["not_assessable"]}
        for domain in QUALITY_DOMAINS
    }
    cases = []
    for index in range(3):
        ratings_a = {domain: "low" for domain in QUALITY_DOMAINS}
        ratings_b = dict(ratings_a)
        adjudicated = dict(ratings_a)
        if index == 1:
            ratings_a["directness"] = "high"
            ratings_b["directness"] = "high"
            adjudicated["directness"] = "high"
            ratings_b["risk_of_bias"] = "high"
            adjudicated["risk_of_bias"] = "moderate"
        if index == 2:
            ratings_a["directness"] = "not_assessable"
            ratings_b["directness"] = "not_assessable"
            adjudicated["directness"] = "not_assessable"
        reviewers = [
            {"reviewer_id": "reviewer-a", "role": "reviewer", "ratings": ratings_a},
            {"reviewer_id": "reviewer-b", "role": "reviewer", "ratings": ratings_b},
        ]
        if index == 1:
            reviewers.append({"reviewer_id": "adjudicator-c", "role": "adjudicator", "ratings": adjudicated})
        predictions = {
            domain: {"label": "low", "confidence": 0.9}
            for domain in QUALITY_DOMAINS
        }
        if index == 1:
            predictions["directness"] = {"label": "high", "confidence": 0.9}
            predictions["risk_of_bias"] = {"label": "moderate", "confidence": 0.8}
        if index == 2:
            predictions["precision"] = {"label": "__abstain__", "confidence": None}
            predictions["directness"] = {"label": "not_assessable", "confidence": 0.9}
        cases.append({
            "case_id": f"case-{index + 1}",
            "evidence_sha256": f"{index + 1:064x}",
            "evidence_type": "paper" if index < 2 else "abstract",
            "reviewers": reviewers,
            "predictions": predictions,
        })
    return {
        "schema_version": "3",
        "benchmark_id": "quality-screen-v1",
        "labeling_protocol": "Synthetic test labels only; ordered levels are fixture values and do not represent a scientific rubric.",
        "domain_scales": scales,
        "cases": cases,
    }


class EvidenceQualityBenchmarkTests(unittest.TestCase):
    def test_weighted_kappa_is_per_reviewer_pair_and_order_independent(self) -> None:
        benchmark = quality_benchmark()
        benchmark["cases"][1]["reviewers"][:2] = list(reversed(benchmark["cases"][1]["reviewers"][:2]))
        for source_index, suffix in ((0, "4"), (1, "5")):
            original = benchmark["cases"][source_index]
            copied = {
                **original,
                "case_id": f"panel-cd-{suffix}",
                "evidence_sha256": f"{int(suffix):064x}",
                "reviewers": [
                    {**original["reviewers"][0], "reviewer_id": "reviewer-c"},
                    {**original["reviewers"][1], "reviewer_id": "reviewer-d"},
                ],
            }
            if len(original["reviewers"]) == 3:
                copied["reviewers"].append({**original["reviewers"][2], "reviewer_id": "adjudicator-e"})
            benchmark["cases"].append(copied)

        report = evaluate_ordinal_quality_cases(benchmark)
        agreement = report["overall_by_domain"]["directness"]["primary_reviewer_agreement"]
        self.assertIsNone(agreement["quadratic_weighted_kappa_on_ordinal_pairs"])
        self.assertEqual(agreement["quadratic_weighted_kappa_by_reviewer_pair"]["reviewer-a|reviewer-b"]["quadratic_weighted_kappa_on_ordinal_pairs"], 1.0)
        self.assertEqual(agreement["quadratic_weighted_kappa_by_reviewer_pair"]["reviewer-c|reviewer-d"]["quadratic_weighted_kappa_on_ordinal_pairs"], 1.0)

    def test_reports_resolved_ordinal_metrics_and_abstention_by_domain(self) -> None:
        benchmark = quality_benchmark()
        report = evaluate_adjudicated_cases(benchmark)
        self.assertEqual(report["schema_version"], "3")
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["adjudicated_disagreements_by_domain"]["risk_of_bias"], 1)
        self.assertEqual(report["adjudicated_disagreements_by_domain"]["directness"], 0)
        self.assertEqual(report["overall_by_domain"]["risk_of_bias"]["prediction"]["exact_accuracy_all_cases"], 1)
        self.assertEqual(report["overall_by_domain"]["risk_of_bias"]["prediction"]["exact_accuracy_all_cases_wilson_95_ci"], _wilson_interval(3, 3))
        self.assertEqual(
            report["overall_by_domain"]["risk_of_bias"]["prediction"]["quadratic_weighted_kappa_on_ordinal_pairs"],
            1,
        )
        self.assertAlmostEqual(report["overall_by_domain"]["precision"]["prediction"]["coverage"], 2 / 3)
        self.assertIn("assume independent cases", report["overall_by_domain"]["precision"]["interval_interpretation"])
        self.assertAlmostEqual(report["overall_by_domain"]["directness"]["primary_reviewer_agreement"]["ordinal_pair_coverage"], 2 / 3)
        self.assertEqual(report["overall_by_domain"]["directness"]["prediction"]["mean_absolute_level_error_on_ordinal_pairs"], 0)
        self.assertEqual(report["overall_by_domain"]["directness"]["prediction"]["exact_accuracy_all_cases"], 1)
        self.assertEqual(report["overall_by_domain"]["directness"]["prediction"]["quadratic_weighted_kappa_on_ordinal_pairs"], 1)
        self.assertEqual(report["overall_by_domain"]["directness"]["prediction"]["exact_correctness_calibration_bins"][0]["count"], 3)
        self.assertEqual(report["by_evidence_type"]["paper"]["risk_of_bias"]["n"], 2)
        self.assertIn("not a probability that a biomedical claim is true", report["confidence_semantics"])
        self.assertEqual(report, evaluate_ordinal_quality_cases(benchmark))

    def test_all_abstentions_have_no_covered_accuracy_interval(self) -> None:
        benchmark = quality_benchmark()
        for row in benchmark["cases"]:
            row["predictions"] = {
                domain: {"label": "__abstain__", "confidence": None}
                for domain in QUALITY_DOMAINS
            }
        report = evaluate_ordinal_quality_cases(benchmark)
        prediction = report["overall_by_domain"]["risk_of_bias"]["prediction"]
        self.assertEqual(prediction["exact_accuracy_all_cases"], 0)
        self.assertIsNotNone(prediction["exact_accuracy_all_cases_wilson_95_ci"])
        self.assertIsNone(prediction["exact_accuracy_covered"])
        self.assertIsNone(prediction["exact_accuracy_covered_wilson_95_ci"])
        self.assertIsNone(prediction["within_one_level_accuracy_wilson_95_ci"])

    def test_requires_distinct_adjudication_for_every_primary_disagreement(self) -> None:
        benchmark = quality_benchmark()
        benchmark["cases"][1]["reviewers"].pop()
        with self.assertRaisesRegex(ValueError, "require a distinct third-party"):
            evaluate_ordinal_quality_cases(benchmark)

    def test_rejects_invalid_scales_labels_and_non_finite_confidence(self) -> None:
        benchmark = quality_benchmark()
        benchmark["domain_scales"]["precision"]["ordered_levels"] = ["low", "low"]
        with self.assertRaisesRegex(ValueError, "unique labels"):
            evaluate_ordinal_quality_cases(benchmark)
        benchmark = quality_benchmark()
        benchmark["cases"][0]["predictions"]["precision"]["confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_ordinal_quality_cases(benchmark)
        benchmark = quality_benchmark()
        benchmark["cases"][0]["evidence_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluate_ordinal_quality_cases(benchmark)

    def test_rejects_duplicate_source_digests_to_prevent_case_pseudoreplication(self) -> None:
        benchmark = quality_benchmark()
        duplicate = dict(benchmark["cases"][0])
        duplicate["case_id"] = "duplicate-case"
        benchmark["cases"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "evidence digests must be unique"):
            evaluate_ordinal_quality_cases(benchmark)


if __name__ == "__main__":
    unittest.main()
