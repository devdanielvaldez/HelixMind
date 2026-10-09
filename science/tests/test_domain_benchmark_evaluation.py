from __future__ import annotations

import unittest

from medai_science.evaluation import (
    MAX_BENCHMARK_EVIDENCE_STRATA,
    MAX_BENCHMARK_LABELS_PER_DOMAIN,
    evaluate_biomedical_domain_benchmark,
)


def case(case_id: str, label: str, predicted: str | None = None):
    predicted = predicted or label
    return {
        "case_id": case_id,
        "evidence_sha256": ("a" if case_id.endswith("1") else "b") * 64,
        "evidence_type": "source_record",
        "reviewers": [
            {"reviewer_id": "reviewer-a", "role": "reviewer", "label": label},
            {"reviewer_id": "reviewer-b", "role": "reviewer", "label": label},
        ],
        "predicted": predicted,
        "confidence": 0.9 if predicted == label else 0.6,
    }


def benchmark():
    return {
        "schema_version": "1",
        "benchmark_id": "medai-domain-fixture-v1",
        "labeling_protocol": "Synthetic fixture for evaluator tests; source hashes model pinned evidence.",
        "domains": {
            "genomics": [case("geno-1", "match"), case("geno-2", "mismatch", "match")],
            "proteomics": [case("protein-1", "observed")],
            "pharmacology": [case("pharma-1", "interaction")],
            "literature_extraction": [case("literature-1", "present")],
        },
    }


class BiomedicalDomainBenchmarkTests(unittest.TestCase):
    def test_reports_domain_specific_metrics_without_cross_domain_pooling(self):
        report = evaluate_biomedical_domain_benchmark(benchmark(), bins=5)
        self.assertTrue(report["complete_domain_coverage"])
        self.assertEqual(report["total_cases"], 5)
        self.assertEqual(report["domains"]["genomics"]["overall"]["n"], 2)
        self.assertEqual(report["domains"]["proteomics"]["overall"]["n"], 1)
        self.assertEqual(report["domains"]["genomics"]["overall"]["accuracy"], 0.5)
        self.assertNotIn("overall", report)
        self.assertEqual(len(report["benchmark_sha256"]), 64)
        self.assertTrue(report["adjudication_integrity"]["case_schema_validated"])
        self.assertFalse(report["adjudication_integrity"]["evidence_source_content_verified"])
        self.assertFalse(report["adjudication_integrity"]["reviewer_identity_or_independence_verified"])
        self.assertFalse(report["adjudication_integrity"]["adjudication_protocol_externally_audited"])

    def test_empty_domains_are_explicitly_incomplete(self):
        document = benchmark()
        document["domains"]["literature_extraction"] = []
        report = evaluate_biomedical_domain_benchmark(document)
        self.assertFalse(report["complete_domain_coverage"])
        self.assertEqual(report["missing_domains"], ["literature_extraction"])
        self.assertEqual(report["domains"]["literature_extraction"]["status"], "not_evaluated")

    def test_rejects_duplicate_case_ids_across_domains_and_missing_domain(self):
        document = benchmark()
        document["domains"]["proteomics"][0]["case_id"] = "geno-1"
        with self.assertRaisesRegex(ValueError, "globally unique"):
            evaluate_biomedical_domain_benchmark(document)
        document = benchmark()
        del document["domains"]["genomics"]
        with self.assertRaisesRegex(ValueError, "must declare"):
            evaluate_biomedical_domain_benchmark(document)

    def test_rejects_a_vacuous_benchmark_with_no_cases(self):
        document = benchmark()
        document["domains"] = {domain: [] for domain in document["domains"]}
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_biomedical_domain_benchmark(document)

    def test_bounds_domain_label_cardinality_before_building_confusion_matrix(self):
        document = benchmark()
        document["domains"]["genomics"] = [
            case(f"geno-{index}", f"label-{index}")
            for index in range(MAX_BENCHMARK_LABELS_PER_DOMAIN + 1)
        ]
        with self.assertRaisesRegex(ValueError, "distinct labels"):
            evaluate_biomedical_domain_benchmark(document)

    def test_bounds_evidence_type_strata_per_domain(self):
        document = benchmark()
        document["domains"]["genomics"] = [
            {**case(f"geno-{index}", "same"), "evidence_type": f"source-{index}"}
            for index in range(MAX_BENCHMARK_EVIDENCE_STRATA + 1)
        ]
        with self.assertRaisesRegex(ValueError, "evidence-type strata"):
            evaluate_biomedical_domain_benchmark(document)


if __name__ == "__main__":
    unittest.main()
