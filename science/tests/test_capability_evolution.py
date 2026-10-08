import unittest

from medai_science.capability_evolution import compare_capability_versions, run_paired_capability_benchmark


def comparison(cases=None):
    return {
        "schema_version": "1",
        "capability_id": "genomics.variant_filter",
        "baseline_version": "1.0.0",
        "candidate_version": "1.1.0",
        "benchmark_id": "vcf-regression-2026-01",
        "benchmark_sha256": "a" * 64,
        "cases": cases if cases is not None else [
            {
                "case_id": f"case-{index}",
                "baseline_passed": index < 8,
                "candidate_passed": index < 9,
                "baseline_latency_ms": 100 + index,
                "candidate_latency_ms": 80 + index,
            }
            for index in range(10)
        ],
    }


class CapabilityEvolutionTests(unittest.TestCase):
    def test_reports_paired_performance_and_never_changes_lifecycle(self):
        report = compare_capability_versions(comparison())
        self.assertEqual(report["n"], 10)
        self.assertEqual(report["paired"]["candidate_only_passes"], 1)
        self.assertEqual(report["paired"]["baseline_only_passes"], 0)
        self.assertEqual(report["recommendation"], "candidate_improves_pass_rate_review_before_promotion")
        self.assertFalse(report["lifecycle_mutation"])
        self.assertEqual(len(report["comparison_sha256"]), 64)
        self.assertEqual(report["paired_outcome_exact_sign_test"]["two_sided_p_value"], 1.0)
        self.assertEqual(report["paired_outcome_exact_sign_test"]["discordant_pairs"], 1)

    def test_exact_paired_outcome_test_reports_strength_and_no_information(self):
        cases = comparison()["cases"]
        for index in range(10):
            cases[index]["baseline_passed"] = False
            cases[index]["candidate_passed"] = True
        report = compare_capability_versions(comparison(cases))
        self.assertEqual(report["paired_outcome_exact_sign_test"]["two_sided_p_value"], 2 / 1024)
        cases = comparison()["cases"]
        for row in cases:
            row["candidate_passed"] = row["baseline_passed"]
        report = compare_capability_versions(comparison(cases))
        self.assertIsNone(report["paired_outcome_exact_sign_test"]["two_sided_p_value"])

    def test_recommends_hold_when_candidate_regresses_any_paired_case(self):
        rows = comparison()["cases"]
        rows[0]["candidate_passed"] = False
        report = compare_capability_versions(comparison(rows))
        self.assertEqual(report["recommendation"], "retain_baseline_and_review_regressions")

    def test_requires_same_shaped_bounded_and_finite_paired_records(self):
        invalid = comparison()
        invalid["cases"][0]["candidate_latency_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            compare_capability_versions(invalid)
        invalid = comparison()
        invalid["cases"][1]["case_id"] = invalid["cases"][0]["case_id"]
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            compare_capability_versions(invalid)
        with self.assertRaisesRegex(ValueError, "10..100000"):
            compare_capability_versions(comparison([]))

    def test_runs_both_versions_on_identical_cases_and_binds_source_and_execution_hashes(self):
        class FakeSandbox:
            image = "python@sha256:" + "b" * 64

            def execute(self, source, input_data, test_cases):
                candidate = "candidate" in source
                rows = []
                for index, case in enumerate(test_cases):
                    passed = case["expected"].get("value") == case["input"].get("value") or candidate
                    rows.append({"index": index, "passed": passed, "wall_time_microseconds": 50 if candidate else 100})
                return {"validation": {"cases": rows}, "cpu_time_microseconds": 900 if candidate else 1_000}

        benchmark = {
            "schema_version": "1", "capability_id": "genomics.variant_filter",
            "baseline_version": "1.0.0", "candidate_version": "1.1.0",
            "benchmark_id": "frozen-vcf-corpus", "cases": [
                {"case_id": f"case-{index}", "input": {"value": index}, "expected": {"value": index if index < 9 else -1}}
                for index in range(10)
            ],
        }
        report = run_paired_capability_benchmark(
            FakeSandbox(), "def run(input):\n    return {'value': input['value']}\n",
            "# candidate\ndef run(input):\n    return {'value': input['value']}\n", benchmark,
            repeats=3,
        )
        self.assertEqual(report["benchmark_sha256"], __import__("hashlib").sha256(__import__("json").dumps(benchmark, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
        self.assertTrue(report["repeat_consistency"]["candidate_all_repeats_passed"])
        self.assertEqual(report["execution"]["repeats"], 3)
        self.assertEqual(report["execution"]["execution_order"], ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"])
        self.assertEqual(report["execution"]["candidate_cpu_microseconds"], [900, 900, 900])
        self.assertEqual(len(report["execution"]["baseline_source_sha256"]), 64)
        self.assertEqual(len(report["execution_receipt_sha256"]), 64)

    def test_paired_runner_requires_measured_case_timings(self):
        class TiminglessSandbox:
            image = "python@sha256:" + "b" * 64

            def execute(self, source, input_data, test_cases):
                return {"validation": {"cases": [{"index": index, "passed": True} for index, _ in enumerate(test_cases)]}, "cpu_time_microseconds": 1}

        benchmark = {
            "schema_version": "1", "capability_id": "genomics.variant_filter",
            "baseline_version": "1.0.0", "candidate_version": "1.1.0",
            "benchmark_id": "frozen-vcf-corpus", "cases": [
                {"case_id": f"case-{index}", "input": {}, "expected": {}} for index in range(10)
            ],
        }
        source = "def run(input):\n    return {}\n"
        with self.assertRaisesRegex(ValueError, "measured per-case wall time"):
            run_paired_capability_benchmark(TiminglessSandbox(), source, source + "\n", benchmark, repeats=3)

    def test_paired_runner_accepts_r_sources_with_the_r_validator(self):
        class FakeRSandbox:
            image = "r-base@sha256:" + "c" * 64
            language = "r"

            def execute(self, source, input_data, test_cases):
                return {
                    "validation": {"cases": [{"index": index, "passed": True, "wall_time_microseconds": 20} for index, _ in enumerate(test_cases)]},
                    "cpu_time_microseconds": 50,
                }

        benchmark = {
            "schema_version": "1", "capability_id": "genomics.variant_filter",
            "baseline_version": "1.0.0", "candidate_version": "1.1.0",
            "benchmark_id": "frozen-vcf-corpus", "cases": [
                {"case_id": f"case-{index}", "input": {}, "expected": {}} for index in range(10)
            ],
        }
        source = "run <- function(input) list()\n"
        report = run_paired_capability_benchmark(FakeRSandbox(), source, source + "# candidate\n", benchmark, repeats=3)
        self.assertEqual(report["execution"]["language"], "r")


if __name__ == "__main__":
    unittest.main()
