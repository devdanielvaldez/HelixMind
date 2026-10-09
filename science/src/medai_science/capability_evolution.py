"""Paired, descriptive comparison of versioned capability benchmark runs."""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from fractions import Fraction
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIN_CASES = 10


def _evolution_review(report: dict[str, Any], *, repeated_and_stable: bool, candidate_all_passed: bool) -> dict[str, Any]:
    """Translate measured comparison into a review lead without changing lifecycle state."""
    recommendation = report["recommendation"]
    if not repeated_and_stable:
        action = "repeat_or_stabilize_benchmark"
        eligible = False
    elif not candidate_all_passed:
        action = "retain_baseline_and_review_candidate_failures"
        eligible = False
    elif recommendation in {
        "candidate_improves_pass_rate_review_before_promotion",
        "candidate_is_faster_without_observed_pass_loss_review_before_promotion",
    }:
        action = "candidate_ready_for_human_promotion_review"
        eligible = True
    elif recommendation in {
        "retain_baseline_and_review_regressions",
        "retain_baseline_pending_latency_review",
    }:
        action = "retain_baseline_pending_human_review"
        eligible = False
    else:
        action = "no_lifecycle_change_recommended"
        eligible = False
    return {
        "action": action,
        "eligible_for_promotion_review": eligible,
        "comparison_sha256": report["comparison_sha256"],
        "benchmark_sha256": report["benchmark_sha256"],
        "baseline_version": report["baseline_version"],
        "candidate_version": report["candidate_version"],
        "requires_platform_admin_decision": True,
        "lifecycle_mutation": False,
        "interpretation_limit": "This execution-performance lead is bound to the paired benchmark report; it does not establish scientific validity, authorize production, or change either version's lifecycle state.",
    }


def _paired_exact_outcome_p_value(candidate_only_passes: int, baseline_only_passes: int) -> float | None:
    """Two-sided exact sign test over discordant paired pass/fail outcomes."""
    discordant = candidate_only_passes + baseline_only_passes
    if discordant == 0:
        return None
    tail_count = min(candidate_only_passes, baseline_only_passes)
    term = 1
    tail = term
    for index in range(1, tail_count + 1):
        term = term * (discordant - index + 1) // index
        tail += term
    return float(min(Fraction(1), Fraction(2 * tail, 1 << discordant)))


def _one_sided_sign_test_p_value(candidate_wins: int, baseline_wins: int) -> float | None:
    """Exact one-sided sign test over non-tied paired measurements."""
    discordant = candidate_wins + baseline_wins
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, index) for index in range(candidate_wins, discordant + 1))
    return float(Fraction(tail, 1 << discordant))


def compare_capability_versions(document: dict[str, Any]) -> dict[str, Any]:
    """Compare two versions on identical benchmark cases without changing lifecycle state."""
    required = {
        "schema_version", "capability_id", "baseline_version", "candidate_version",
        "benchmark_id", "benchmark_sha256", "cases",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("comparison requires the exact schema_version, capability_id, baseline_version, candidate_version, benchmark_id, benchmark_sha256, and cases fields")
    if document["schema_version"] != "1":
        raise ValueError("schema_version must be 1")
    for field in ("capability_id", "baseline_version", "candidate_version", "benchmark_id"):
        if not isinstance(document[field], str) or not _IDENTIFIER.fullmatch(document[field]):
            raise ValueError(f"{field} must be a bounded identifier")
    if document["baseline_version"] == document["candidate_version"]:
        raise ValueError("baseline_version and candidate_version must differ")
    if not isinstance(document["benchmark_sha256"], str) or not _SHA256.fullmatch(document["benchmark_sha256"]):
        raise ValueError("benchmark_sha256 must be a lowercase SHA-256 digest")
    cases = document["cases"]
    if not isinstance(cases, list) or not _MIN_CASES <= len(cases) <= 100_000:
        raise ValueError(f"comparison requires {_MIN_CASES}..100000 paired cases")

    seen: set[str] = set()
    baseline_passes = candidate_passes = 0
    baseline_latency: list[float] = []
    candidate_latency: list[float] = []
    improved = regressed = unchanged = 0
    for index, row in enumerate(cases):
        fields = {"case_id", "baseline_passed", "candidate_passed", "baseline_latency_ms", "candidate_latency_ms"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"case {index} has missing or unsupported fields")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not _IDENTIFIER.fullmatch(case_id) or case_id in seen:
            raise ValueError(f"case {index} has an invalid or duplicate case_id")
        seen.add(case_id)
        for field in ("baseline_passed", "candidate_passed"):
            if not isinstance(row[field], bool):
                raise ValueError(f"case {index} {field} must be boolean")
        for field, target in (("baseline_latency_ms", baseline_latency), ("candidate_latency_ms", candidate_latency)):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"case {index} {field} must be finite and non-negative")
            target.append(float(value))
        baseline_passes += int(row["baseline_passed"])
        candidate_passes += int(row["candidate_passed"])
        if row["candidate_passed"] and not row["baseline_passed"]:
            improved += 1
        elif row["baseline_passed"] and not row["candidate_passed"]:
            regressed += 1
        else:
            unchanged += 1

    total = len(cases)
    baseline_rate = baseline_passes / total
    candidate_rate = candidate_passes / total
    baseline_median = statistics.median(baseline_latency)
    candidate_median = statistics.median(candidate_latency)
    latency_change = (candidate_median - baseline_median) / baseline_median if baseline_median else None
    outcome_p_value = _paired_exact_outcome_p_value(improved, regressed)
    latency_wins = sum(candidate_latency[index] < baseline_latency[index] for index in range(total))
    latency_losses = sum(candidate_latency[index] > baseline_latency[index] for index in range(total))
    latency_p_value = _one_sided_sign_test_p_value(latency_wins, latency_losses)
    latency_regression_p_value = _one_sided_sign_test_p_value(latency_losses, latency_wins)
    if candidate_passes < baseline_passes or regressed:
        recommendation = "retain_baseline_and_review_regressions"
    elif candidate_passes > baseline_passes and outcome_p_value is not None and outcome_p_value <= 0.05:
        recommendation = "candidate_improves_pass_rate_review_before_promotion"
    elif latency_change is not None and latency_change <= -0.05 and latency_p_value is not None and latency_p_value <= 0.05:
        recommendation = "candidate_is_faster_without_observed_pass_loss_review_before_promotion"
    elif candidate_passes > baseline_passes:
        recommendation = "candidate_pass_rate_trend_requires_more_paired_cases"
    elif latency_change is not None and latency_change <= -0.05:
        recommendation = "candidate_latency_trend_requires_more_paired_cases"
    elif latency_change is not None and latency_change >= 0.05 and latency_regression_p_value is not None and latency_regression_p_value <= 0.05:
        recommendation = "retain_baseline_pending_latency_review"
    else:
        recommendation = "no_material_observed_difference"

    comparison = {
        key: document[key]
        for key in ("schema_version", "capability_id", "baseline_version", "candidate_version", "benchmark_id", "benchmark_sha256")
    }
    comparison["cases"] = cases
    digest = hashlib.sha256(json.dumps(comparison, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": "1",
        "comparison_sha256": digest,
        "capability_id": document["capability_id"],
        "baseline_version": document["baseline_version"],
        "candidate_version": document["candidate_version"],
        "benchmark_id": document["benchmark_id"],
        "benchmark_sha256": document["benchmark_sha256"],
        "n": total,
        "baseline": {"passed": baseline_passes, "pass_rate": baseline_rate, "median_latency_ms": baseline_median},
        "candidate": {"passed": candidate_passes, "pass_rate": candidate_rate, "median_latency_ms": candidate_median},
        "paired": {"candidate_only_passes": improved, "baseline_only_passes": regressed, "same_outcome": unchanged},
        "paired_outcome_exact_sign_test": {
            "two_sided_p_value": outcome_p_value,
            "discordant_pairs": improved + regressed,
            "method": "two-sided exact binomial sign test over candidate-only versus baseline-only passing cases",
            "interpretation_limit": "This descriptive paired test assumes independent benchmark cases; correlated cases, benchmark selection, and external validity are not assessed.",
        },
        "paired_latency_exact_sign_test": {
            "candidate_faster_cases": latency_wins,
            "baseline_faster_cases": latency_losses,
            "ties": total - latency_wins - latency_losses,
            "one_sided_p_value_candidate_faster": latency_p_value,
            "one_sided_p_value_baseline_faster": latency_regression_p_value,
            "method": "one-sided exact binomial sign test over per-case median latency differences",
            "interpretation_limit": "This descriptive test assumes independent benchmark cases; case dependence, benchmark selection, and external validity are not assessed.",
        },
        "pass_rate_delta": candidate_rate - baseline_rate,
        "median_latency_relative_change": latency_change,
        "recommendation": recommendation,
        "lifecycle_mutation": False,
        "interpretation_limit": "Paired benchmark observations support a review recommendation only. They do not establish scientific validity outside this exact corpus, independent adjudication, statistical significance, or permission to promote/deprecate a version.",
    }


def run_paired_capability_benchmark(
    sandbox: Any,
    baseline_source: str,
    candidate_source: str,
    benchmark: dict[str, Any],
    *,
    repeats: int = 5,
    language: str | None = None,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute both sources in the same pinned sandbox over identical benchmark cases."""
    required = {"schema_version", "capability_id", "baseline_version", "candidate_version", "benchmark_id", "cases"}
    if not isinstance(benchmark, dict) or set(benchmark) != required or benchmark.get("schema_version") != "1":
        raise ValueError("benchmark requires schema_version 1, capability_id, baseline_version, candidate_version, benchmark_id, and cases")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 3 <= repeats <= 15:
        raise ValueError("repeats must be 3..15")
    cases = benchmark["cases"]
    if not isinstance(cases, list) or not _MIN_CASES <= len(cases) <= 100:
        raise ValueError(f"benchmark requires {_MIN_CASES}..100 cases")
    for field in ("capability_id", "baseline_version", "candidate_version", "benchmark_id"):
        if not isinstance(benchmark[field], str) or not _IDENTIFIER.fullmatch(benchmark[field]):
            raise ValueError(f"{field} must be a bounded identifier")
    if benchmark["baseline_version"] == benchmark["candidate_version"]:
        raise ValueError("baseline_version and candidate_version must differ")
    canonical_benchmark = json.dumps(benchmark, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    benchmark_sha256 = hashlib.sha256(canonical_benchmark).hexdigest()
    case_ids: list[str] = []
    normalized_cases: list[dict[str, Any]] = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != {"case_id", "input", "expected"}:
            raise ValueError(f"benchmark case {index} requires only case_id, input, and expected")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not _IDENTIFIER.fullmatch(case_id) or case_id in case_ids:
            raise ValueError(f"benchmark case {index} has an invalid or duplicate case_id")
        if not isinstance(row["input"], dict) or not isinstance(row["expected"], dict):
            raise ValueError(f"benchmark case {index} input and expected must be JSON objects")
        case_ids.append(case_id)
        normalized_cases.append(row)

    from .sandbox import validate_capability_run_request, validate_python_source
    from .r_sandbox import validate_r_source
    validate_capability_run_request(
        normalized_cases[0]["input"],
        [{"input": row["input"], "expected": row["expected"]} for row in normalized_cases],
        contract,
    )
    language = language or getattr(sandbox, "language", "python")
    if language not in {"python", "r"}:
        raise ValueError("paired benchmark language must be python or r")
    validator = validate_r_source if language == "r" else validate_python_source
    baseline_receipt = validator(baseline_source)
    candidate_receipt = validator(candidate_source)
    results: dict[str, list[dict[str, Any]]] = {"baseline": [], "candidate": []}
    sources = {"baseline": baseline_source, "candidate": candidate_source}
    execution_order = []
    for repeat in range(repeats):
        order = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
        for key in order:
            execution_order.append(key)
            execute_kwargs = {"contract": contract} if contract is not None else {}
            execution = sandbox.execute(sources[key], normalized_cases[0]["input"], [{"input": row["input"], "expected": row["expected"]} for row in normalized_cases], **execute_kwargs)
            validation = execution.get("validation")
            observed = validation.get("cases") if isinstance(validation, dict) else None
            if not isinstance(observed, list) or len(observed) != len(normalized_cases):
                raise ValueError(f"{key} sandbox run {repeat + 1} returned an incomplete case receipt")
            rows = []
            for index, result in enumerate(observed):
                if not isinstance(result, dict) or result.get("index") != index or not isinstance(result.get("passed"), bool):
                    raise ValueError(f"{key} sandbox run {repeat + 1} returned an invalid case receipt")
                latency = result.get("wall_time_microseconds")
                if isinstance(latency, bool) or not isinstance(latency, int) or latency < 0:
                    raise ValueError(f"{key} sandbox run {repeat + 1} lacks measured per-case wall time")
                rows.append({"case_id": case_ids[index], "passed": result["passed"], "wall_time_microseconds": latency})
            cpu_time = execution.get("cpu_time_microseconds")
            if isinstance(cpu_time, bool) or not isinstance(cpu_time, int) or cpu_time < 0:
                raise ValueError(f"{key} sandbox run {repeat + 1} lacks measured process CPU time")
            results[key].append({"cases": rows, "cpu_time_microseconds": cpu_time})

    paired_cases = []
    for index, case_id in enumerate(case_ids):
        baseline_runs = [run["cases"][index] for run in results["baseline"]]
        candidate_runs = [run["cases"][index] for run in results["candidate"]]
        paired_cases.append({
            "case_id": case_id,
            "baseline_passed": all(row["passed"] for row in baseline_runs),
            "candidate_passed": all(row["passed"] for row in candidate_runs),
            "baseline_latency_ms": statistics.median(row["wall_time_microseconds"] for row in baseline_runs) / 1000,
            "candidate_latency_ms": statistics.median(row["wall_time_microseconds"] for row in candidate_runs) / 1000,
        })
    report = compare_capability_versions({
        "schema_version": "1",
        "capability_id": benchmark["capability_id"],
        "baseline_version": benchmark["baseline_version"],
        "candidate_version": benchmark["candidate_version"],
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_sha256": benchmark_sha256,
        "cases": paired_cases,
    })
    execution_record = {
        "baseline_source_sha256": baseline_receipt["source_sha256"],
        "candidate_source_sha256": candidate_receipt["source_sha256"],
        "sandbox_image": sandbox.image,
        "repeats": repeats,
        "repeat_results": results,
        "execution_order": execution_order,
        "language": language,
        "cpu_measurement": f"per-process {language.upper()} CPU includes loading, primary invocation, and all benchmark cases; per-case latency is measured inside the runner using a monotonic wall clock",
    }
    report["execution"] = {key: value for key, value in execution_record.items() if key != "repeat_results"}
    report["execution_receipt_sha256"] = hashlib.sha256(json.dumps(execution_record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    report["execution"]["execution_order"] = execution_order
    report["execution"]["baseline_cpu_microseconds"] = [run["cpu_time_microseconds"] for run in results["baseline"]]
    report["execution"]["candidate_cpu_microseconds"] = [run["cpu_time_microseconds"] for run in results["candidate"]]
    report["repeat_consistency"] = {
        "baseline_all_repeats_passed": all(all(row["passed"] for row in run["cases"]) for run in results["baseline"]),
        "candidate_all_repeats_passed": all(all(row["passed"] for row in run["cases"]) for run in results["candidate"]),
    }
    unstable = {
        key: [
            case_ids[index]
            for index in range(len(case_ids))
            if len({run["cases"][index]["passed"] for run in results[key]}) > 1
        ]
        for key in ("baseline", "candidate")
    }
    report["repeat_stability"] = {
        "baseline_unstable_case_ids": unstable["baseline"],
        "candidate_unstable_case_ids": unstable["candidate"],
        "all_outcomes_stable": not unstable["baseline"] and not unstable["candidate"],
        "interpretation_limit": "Any case whose pass/fail outcome changed across repeated sandbox executions makes a lifecycle performance recommendation inconclusive.",
    }
    if not report["repeat_stability"]["all_outcomes_stable"]:
        report["paired_recommendation"] = report["recommendation"]
        report["recommendation"] = "inconclusive_due_to_repeat_instability"
    candidate_all_passed = report["repeat_consistency"]["candidate_all_repeats_passed"]
    report["evolution_review"] = _evolution_review(
        report,
        repeated_and_stable=report["repeat_stability"]["all_outcomes_stable"],
        candidate_all_passed=candidate_all_passed,
    )
    return report
