"""Reproducible evaluation for independently adjudicated categorical research outputs."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from medai_science.quality import QUALITY_DOMAINS


MAX_CASES = 100_000
MAX_LABELS = 100
MAX_ECE_BINS = 100
MIN_CAPABILITY_PAIRS = 50
MIN_CAPABILITY_HOLDOUT_PREDICTIONS = 20
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _quadratic_weighted_kappa(left: list[str], right: list[str], levels: list[str]) -> float | None:
    if not left or len(left) != len(right) or len(levels) < 2:
        return None
    positions = {label: index for index, label in enumerate(levels)}
    size = len(levels)
    observed_disagreement = sum(
        ((positions[a] - positions[b]) / (size - 1)) ** 2
        for a, b in zip(left, right)
    ) / len(left)
    left_counts = [sum(label == level for label in left) for level in levels]
    right_counts = [sum(label == level for label in right) for level in levels]
    expected_disagreement = sum(
        left_counts[i] * right_counts[j] / (len(left) ** 2)
        * ((i - j) / (size - 1)) ** 2
        for i in range(size)
        for j in range(size)
    )
    if expected_disagreement == 0:
        return None
    return 1 - observed_disagreement / expected_disagreement


def _summarize_ordinal_domain(
    cases: list[dict[str, Any]], domain: str, scale: dict[str, list[str]], bins: int
) -> dict[str, Any]:
    levels = scale["ordered_levels"]
    expected = [row["expected"][domain] for row in cases]
    reviewer_a = [row["reviewers"][0][domain] for row in cases]
    reviewer_b = [row["reviewers"][1][domain] for row in cases]
    predictions = [row["predictions"][domain]["label"] for row in cases]
    covered = [index for index, label in enumerate(predictions) if label != "__abstain__"]
    positions = {label: index for index, label in enumerate(levels)}
    exact = sum(predictions[index] == expected[index] for index in covered)
    reviewer_ordinal = [index for index, (left, right) in enumerate(zip(reviewer_a, reviewer_b)) if left in positions and right in positions]
    prediction_ordinal = [index for index in covered if predictions[index] in positions and expected[index] in positions]
    absolute_errors = [abs(positions[predictions[index]] - positions[expected[index]]) for index in prediction_ordinal]
    n = len(cases)
    calibration = []
    ece = 0.0
    brier = 0.0
    for index in covered:
        confidence = cases[index]["predictions"][domain]["confidence"]
        brier += (confidence - float(predictions[index] == expected[index])) ** 2
    if covered:
        brier /= len(covered)
        for bin_index in range(bins):
            lower, upper = bin_index / bins, (bin_index + 1) / bins
            members = [
                index for index in covered
                if lower <= cases[index]["predictions"][domain]["confidence"] < upper
                or (bin_index == bins - 1 and cases[index]["predictions"][domain]["confidence"] == 1.0)
            ]
            if not members:
                continue
            mean_confidence = sum(cases[index]["predictions"][domain]["confidence"] for index in members) / len(members)
            observed_accuracy = sum(predictions[index] == expected[index] for index in members) / len(members)
            gap = abs(mean_confidence - observed_accuracy)
            ece += len(members) / len(covered) * gap
            calibration.append({"lower": lower, "upper": upper, "count": len(members), "mean_confidence": mean_confidence, "observed_exact_accuracy": observed_accuracy, "absolute_gap": gap})
    return {
        "n": n,
        "scale": scale,
        "primary_reviewer_agreement": {
            "raw_agreement": sum(a == b for a, b in zip(reviewer_a, reviewer_b)) / n,
            "ordinal_pair_coverage": len(reviewer_ordinal) / n,
            "quadratic_weighted_kappa_on_ordinal_pairs": _quadratic_weighted_kappa([reviewer_a[index] for index in reviewer_ordinal], [reviewer_b[index] for index in reviewer_ordinal], levels),
        },
        "prediction": {
            "coverage": len(covered) / n,
            "abstentions": n - len(covered),
            "exact_accuracy_all_cases": exact / n,
            "exact_accuracy_covered": exact / len(covered) if covered else None,
            "ordinal_comparison_coverage_of_predictions": len(prediction_ordinal) / len(covered) if covered else None,
            "within_one_level_accuracy_on_ordinal_pairs": sum(error <= 1 for error in absolute_errors) / len(prediction_ordinal) if prediction_ordinal else None,
            "mean_absolute_level_error_on_ordinal_pairs": sum(absolute_errors) / len(prediction_ordinal) if prediction_ordinal else None,
            "quadratic_weighted_kappa_on_ordinal_pairs": _quadratic_weighted_kappa([expected[index] for index in prediction_ordinal], [predictions[index] for index in prediction_ordinal], levels),
            "exact_correctness_brier_score_covered": brier if covered else None,
            "exact_correctness_expected_calibration_error_covered": ece if covered else None,
            "exact_correctness_calibration_bins": calibration,
        },
        "interpretation_limit": "Ordinal levels and their ordering come from the supplied protocol. Agreement, distance, and prediction-confidence metrics describe only this adjudicated dataset; they do not establish an evidence grade, clinical validity, or generalization.",
    }


def evaluate_ordinal_quality_cases(document: dict[str, Any], *, bins: int = 10) -> dict[str, Any]:
    """Evaluate blinded multi-domain ordinal quality ratings against third-party resolutions."""
    allowed_top = {"schema_version", "benchmark_id", "labeling_protocol", "domain_scales", "cases"}
    if not isinstance(document, dict) or set(document) != allowed_top:
        raise ValueError("quality benchmark requires the exact schema_version, benchmark_id, labeling_protocol, domain_scales, and cases fields")
    if document.get("schema_version") != "3":
        raise ValueError("ordinal quality benchmark schema_version must be 3")
    for field in ("benchmark_id", "labeling_protocol"):
        if not isinstance(document.get(field), str) or not document[field].strip() or len(document[field]) > 4000:
            raise ValueError(f"quality benchmark requires a non-empty bounded {field}")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= MAX_ECE_BINS:
        raise ValueError("calibration bins must be 2..100")
    scales = document["domain_scales"]
    if not isinstance(scales, dict) or set(scales) != set(QUALITY_DOMAINS):
        raise ValueError("quality benchmark must define ordered scales for all registered quality domains")
    normalized_scales: dict[str, dict[str, list[str]]] = {}
    for domain in QUALITY_DOMAINS:
        scale = scales[domain]
        if not isinstance(scale, dict) or set(scale) != {"ordered_levels", "non_ordinal_labels"}:
            raise ValueError(f"quality benchmark scale {domain} must separate ordered levels from non-ordinal labels")
        levels = scale["ordered_levels"]
        nominal = scale["non_ordinal_labels"]
        if not isinstance(levels, list) or not 2 <= len(levels) <= 7 or any(not isinstance(level, str) or not _LABEL.fullmatch(level) for level in levels) or len(set(levels)) != len(levels):
            raise ValueError(f"quality benchmark ordered scale {domain} must contain 2..7 unique labels")
        if not isinstance(nominal, list) or len(nominal) > 10 or any(not isinstance(label, str) or not _LABEL.fullmatch(label) for label in nominal) or len(set(nominal)) != len(nominal) or set(levels) & set(nominal):
            raise ValueError(f"quality benchmark non-ordinal labels for {domain} must be unique and separate from ordered levels")
        normalized_scales[domain] = {"ordered_levels": levels, "non_ordinal_labels": nominal}
    source_cases = document["cases"]
    if not isinstance(source_cases, list) or not source_cases or len(source_cases) > MAX_CASES:
        raise ValueError(f"quality benchmark must contain 1..{MAX_CASES} source-anchored cases")
    normalized_cases = []
    seen: set[str] = set()
    adjudicated_domains = {domain: 0 for domain in QUALITY_DOMAINS}
    for case_index, row in enumerate(source_cases):
        fields = {"case_id", "evidence_sha256", "evidence_type", "reviewers", "predictions"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"quality case {case_index} has missing or unsupported fields")
        for field in ("case_id", "evidence_type"):
            if not isinstance(row[field], str) or not _LABEL.fullmatch(row[field]):
                raise ValueError(f"quality case {case_index} has an invalid {field}")
        if row["case_id"] in seen:
            raise ValueError("quality benchmark case IDs must be unique")
        seen.add(row["case_id"])
        if not isinstance(row["evidence_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["evidence_sha256"]):
            raise ValueError(f"quality case {case_index} evidence_sha256 must be a lowercase SHA-256 digest")
        reviewers = row["reviewers"]
        if not isinstance(reviewers, list) or len(reviewers) not in {2, 3}:
            raise ValueError(f"quality case {case_index} requires two primary reviewers and an optional adjudicator")
        primary_ratings = []
        reviewer_ids = []
        for reviewer_index, reviewer in enumerate(reviewers):
            if not isinstance(reviewer, dict) or set(reviewer) != {"reviewer_id", "role", "ratings"}:
                raise ValueError(f"quality case {case_index} reviewer {reviewer_index} is malformed")
            reviewer_id = reviewer["reviewer_id"]
            expected_role = "reviewer" if reviewer_index < 2 else "adjudicator"
            if not isinstance(reviewer_id, str) or not _LABEL.fullmatch(reviewer_id) or reviewer.get("role") != expected_role:
                raise ValueError(f"quality case {case_index} reviewer identity or role is invalid")
            if reviewer_id in reviewer_ids:
                raise ValueError(f"quality case {case_index} reviewer identities must be distinct")
            reviewer_ids.append(reviewer_id)
            ratings = reviewer["ratings"]
            if not isinstance(ratings, dict) or set(ratings) != set(QUALITY_DOMAINS):
                raise ValueError(f"quality case {case_index} reviewer ratings must cover every registered domain")
            for domain in QUALITY_DOMAINS:
                allowed_labels = normalized_scales[domain]["ordered_levels"] + normalized_scales[domain]["non_ordinal_labels"]
                if ratings[domain] not in allowed_labels:
                    raise ValueError(f"quality case {case_index} has a label outside the declared {domain} scale")
            if reviewer_index < 2:
                primary_ratings.append(ratings)
        disagreements = {domain for domain in QUALITY_DOMAINS if primary_ratings[0][domain] != primary_ratings[1][domain]}
        if disagreements and len(reviewers) != 3:
            raise ValueError(f"quality case {case_index} disagreements require a distinct third-party adjudicator")
        if not disagreements and len(reviewers) != 2:
            raise ValueError(f"quality case {case_index} must omit an adjudicator when primary reviewers agree")
        adjudication = reviewers[2]["ratings"] if disagreements else {}
        expected_ratings = {}
        for domain in QUALITY_DOMAINS:
            if domain in disagreements:
                expected_ratings[domain] = adjudication[domain]
                adjudicated_domains[domain] += 1
            else:
                expected_ratings[domain] = primary_ratings[0][domain]
        predictions = row["predictions"]
        if not isinstance(predictions, dict) or set(predictions) != set(QUALITY_DOMAINS):
            raise ValueError(f"quality case {case_index} predictions must cover every registered domain")
        normalized_predictions = {}
        for domain in QUALITY_DOMAINS:
            prediction = predictions[domain]
            if not isinstance(prediction, dict) or set(prediction) != {"label", "confidence"}:
                raise ValueError(f"quality case {case_index} prediction for {domain} is malformed")
            label = prediction["label"]
            confidence = prediction["confidence"]
            allowed_labels = normalized_scales[domain]["ordered_levels"] + normalized_scales[domain]["non_ordinal_labels"]
            if not isinstance(label, str) or (label != "__abstain__" and label not in allowed_labels):
                raise ValueError(f"quality case {case_index} prediction for {domain} has an invalid label")
            if label == "__abstain__":
                if confidence is not None:
                    raise ValueError(f"quality case {case_index} abstention confidence must be null")
            elif isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"quality case {case_index} confidence for {domain} must be finite and within 0..1")
            normalized_predictions[domain] = {"label": label, "confidence": float(confidence) if confidence is not None else None}
        normalized_cases.append({
            "case_id": row["case_id"], "evidence_type": row["evidence_type"],
            "expected": expected_ratings,
            "reviewers": [primary_ratings[0], primary_ratings[1]],
            "predictions": normalized_predictions,
        })
    benchmark_sha256 = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in normalized_cases:
        by_type[case["evidence_type"]].append(case)
    domain_results = {
        domain: _summarize_ordinal_domain(normalized_cases, domain, normalized_scales[domain], bins)
        for domain in QUALITY_DOMAINS
    }
    by_type_results = {
        evidence_type: {
            domain: _summarize_ordinal_domain(rows, domain, normalized_scales[domain], bins)
            for domain in QUALITY_DOMAINS
        }
        for evidence_type, rows in sorted(by_type.items())
    }
    return {
        "schema_version": "3",
        "benchmark_id": document["benchmark_id"],
        "benchmark_sha256": benchmark_sha256,
        "labeling_protocol": document["labeling_protocol"],
        "domain_scales": normalized_scales,
        "cases": len(normalized_cases),
        "adjudicated_disagreements_by_domain": adjudicated_domains,
        "overall_by_domain": domain_results,
        "by_evidence_type": by_type_results,
        "confidence_semantics": "Per-domain confidence is the predicted probability that the exact ordinal label matches the resolved reviewer label; it is not a probability that a biomedical claim is true.",
        "interpretation_limit": "This evaluator measures predictions against the supplied source-digest-bound, double-reviewed, adjudicated benchmark. The supplied protocol defines rating levels and case selection; this tool does not grade evidence, validate reviewer independence, or establish clinical validity or external generalization.",
    }


def _summarize(cases: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    expected_labels = sorted({row["expected"] for row in cases})
    labels = sorted({row["expected"] for row in cases} | {row["predicted"] for row in cases if row["predicted"] != "__abstain__"})
    per_label = {}
    for label in labels:
        tp = sum(row["expected"] == label and row["predicted"] == label for row in cases)
        fp = sum(row["expected"] != label and row["predicted"] == label for row in cases)
        fn = sum(row["expected"] == label and row["predicted"] != label for row in cases)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}
    accuracy = sum(row["expected"] == row["predicted"] for row in cases) / len(cases)
    macro_f1 = sum(row["f1"] for row in per_label.values()) / len(per_label) if per_label else 0.0
    brier = sum((row["confidence"] - float(row["expected"] == row["predicted"])) ** 2 for row in cases) / len(cases)
    bin_rows = []
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in cases if lower <= row["confidence"] < upper or (index == bins - 1 and row["confidence"] == 1.0)]
        if not members:
            continue
        mean_confidence = sum(row["confidence"] for row in members) / len(members)
        observed_accuracy = sum(row["expected"] == row["predicted"] for row in members) / len(members)
        gap = abs(mean_confidence - observed_accuracy)
        ece += len(members) / len(cases) * gap
        bin_rows.append({"lower": lower, "upper": upper, "count": len(members), "mean_confidence": mean_confidence, "observed_accuracy": observed_accuracy, "absolute_gap": gap})
    confusion = {expected: {predicted: sum(row["expected"] == expected and row["predicted"] == predicted for row in cases) for predicted in [*labels, "__abstain__"]} for expected in expected_labels}
    return {
        "n": len(cases),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "abstention_rate": sum(row["predicted"] == "__abstain__" for row in cases) / len(cases),
        "confidence_semantics": "confidence that the categorical prediction is correct; not probability that a biomedical claim is true",
        "correctness_brier_score": brier,
        "correctness_expected_calibration_error": ece,
        "correctness_calibration_bins": bin_rows,
    }


def _cohen_agreement(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Report agreement between the two blinded primary reviewers."""
    if not cases:
        return {"n": 0, "raw_agreement": None, "cohen_kappa": None}
    labels = sorted({label for row in cases for label in row["reviewer_labels"]})
    total = len(cases)
    observed = sum(row["reviewer_labels"][0] == row["reviewer_labels"][1] for row in cases) / total
    expected = sum(
        (sum(label == row["reviewer_labels"][0] for row in cases) / total)
        * (sum(label == row["reviewer_labels"][1] for row in cases) / total)
        for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {"n": total, "raw_agreement": observed, "cohen_kappa": kappa}


def _resolve_double_adjudicated_v2(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate blinded primary labels and require a distinct third reviewer to resolve disagreement."""
    allowed = {"schema_version", "benchmark_id", "labeling_protocol", "cases"}
    if set(document) != allowed:
        raise ValueError("double-adjudicated benchmark requires exact schema_version, benchmark_id, labeling_protocol, and cases fields")
    if document.get("schema_version") != "2":
        raise ValueError("double-adjudicated benchmark schema_version must be 2")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise ValueError("benchmark must contain 1..100000 adjudicated cases")
    normalized = []
    seen_cases: set[str] = set()
    resolved_count = 0
    for index, row in enumerate(cases):
        fields = {"case_id", "evidence_sha256", "evidence_type", "reviewers", "predicted", "confidence"}
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"benchmark case {index} has missing or unsupported fields")
        for field in ("case_id", "evidence_type"):
            if not isinstance(row[field], str) or not _LABEL.fullmatch(row[field]):
                raise ValueError(f"benchmark case {index} has an invalid {field}")
        if row["case_id"] in seen_cases:
            raise ValueError("benchmark case IDs must be unique")
        seen_cases.add(row["case_id"])
        evidence_sha256 = row["evidence_sha256"]
        if not isinstance(evidence_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
            raise ValueError(f"benchmark case {index} evidence_sha256 must be a lowercase SHA-256 digest")
        reviewers = row["reviewers"]
        if not isinstance(reviewers, list) or len(reviewers) not in {2, 3}:
            raise ValueError(f"benchmark case {index} must contain two primary reviewers and an optional adjudicator")
        if any(not isinstance(item, dict) or set(item) != {"reviewer_id", "role", "label"} for item in reviewers):
            raise ValueError(f"benchmark case {index} has a malformed reviewer record")
        if any(
            not isinstance(item["reviewer_id"], str) or not _LABEL.fullmatch(item["reviewer_id"])
            or not isinstance(item["role"], str) or item["role"] not in {"reviewer", "adjudicator"}
            or not isinstance(item["label"], str) or not _LABEL.fullmatch(item["label"])
            for item in reviewers
        ):
            raise ValueError(f"benchmark case {index} has an invalid reviewer ID, role, or label")
        primary = reviewers[:2]
        if any(item["role"] != "reviewer" for item in primary) or primary[0]["reviewer_id"] == primary[1]["reviewer_id"]:
            raise ValueError(f"benchmark case {index} requires two distinct primary reviewers")
        disagreement = primary[0]["label"] != primary[1]["label"]
        if disagreement:
            if len(reviewers) != 3 or reviewers[2]["role"] != "adjudicator" or reviewers[2]["reviewer_id"] in {item["reviewer_id"] for item in primary}:
                raise ValueError(f"benchmark case {index} disagreement requires a distinct third-party adjudicator")
            expected = reviewers[2]["label"]
            resolved_count += 1
        else:
            if len(reviewers) != 2:
                raise ValueError(f"benchmark case {index} must omit an adjudicator when primary reviewers agree")
            expected = primary[0]["label"]
        predicted = row["predicted"]
        if not isinstance(predicted, str) or (predicted != "__abstain__" and not _LABEL.fullmatch(predicted)):
            raise ValueError(f"benchmark case {index} has an invalid predicted label")
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"benchmark case {index} confidence must be finite and within 0..1")
        normalized.append({
            "case_id": row["case_id"], "evidence_type": row["evidence_type"], "expected": expected,
            "predicted": predicted, "confidence": float(confidence),
            "reviewer_labels": [primary[0]["label"], primary[1]["label"]],
        })
    resolved = {"schema_version": "1", "benchmark_id": document["benchmark_id"], "labeling_protocol": document["labeling_protocol"], "cases": [
        {key: row[key] for key in ("case_id", "evidence_type", "expected", "predicted", "confidence")} for row in normalized
    ]}
    review_summary = {
        "schema_version": "2",
        "primary_reviewers": _cohen_agreement(normalized),
        "primary_reviewers_by_evidence_type": {
            kind: _cohen_agreement([row for row in normalized if row["evidence_type"] == kind])
            for kind in sorted({row["evidence_type"] for row in normalized})
        },
        "disagreement_count": resolved_count,
        "third_party_adjudication_count": resolved_count,
        "source_anchored_case_count": len(normalized),
        "interpretation_limit": "Reviewer agreement describes agreement on this source-anchored dataset only; it does not establish truth, evidence quality, or reviewer independence beyond the documented protocol.",
    }
    return resolved, review_summary


def evaluate_adjudicated_cases(document: dict[str, Any], *, bins: int = 10) -> dict[str, Any]:
    """Evaluate predictions against frozen labels, stratifying all metrics by evidence type."""
    if isinstance(document, dict) and document.get("schema_version") == "3":
        return evaluate_ordinal_quality_cases(document, bins=bins)
    review_summary = None
    input_sha256 = None
    if isinstance(document, dict) and document.get("schema_version") == "2":
        input_sha256 = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        document, review_summary = _resolve_double_adjudicated_v2(document)
    if not isinstance(document, dict) or set(document) - {"schema_version", "benchmark_id", "labeling_protocol", "cases"}:
        raise ValueError("benchmark document contains unsupported top-level fields")
    if document.get("schema_version") != "1" or not isinstance(document.get("benchmark_id"), str) or not document["benchmark_id"].strip():
        raise ValueError("benchmark requires schema_version 1 and a benchmark_id")
    if not isinstance(document.get("labeling_protocol"), str) or not document["labeling_protocol"].strip():
        raise ValueError("benchmark must document its independent adjudication protocol")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= MAX_ECE_BINS:
        raise ValueError("calibration bins must be 2..100")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise ValueError("benchmark must contain 1..100000 adjudicated cases")
    normalized = []
    seen = set()
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or set(row) != {"case_id", "evidence_type", "expected", "predicted", "confidence"}:
            raise ValueError(f"benchmark case {index} has missing or unsupported fields")
        for field in ("case_id", "evidence_type", "expected"):
            value = row[field]
            if not isinstance(value, str) or not _LABEL.fullmatch(value):
                raise ValueError(f"benchmark case {index} has an invalid {field}")
        if not isinstance(row["predicted"], str) or (row["predicted"] != "__abstain__" and not _LABEL.fullmatch(row["predicted"])):
            raise ValueError(f"benchmark case {index} has an invalid predicted label")
        if row["case_id"] in seen:
            raise ValueError("benchmark case IDs must be unique")
        seen.add(row["case_id"])
        confidence = row["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"benchmark case {index} confidence must be finite and within 0..1")
        normalized.append({**row, "confidence": float(confidence)})
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_type[row["evidence_type"]].append(row)
    result = {
        "schema_version": "1",
        "benchmark_id": document["benchmark_id"],
        "benchmark_sha256": input_sha256 or hashlib.sha256(canonical).hexdigest(),
        "labeling_protocol": document["labeling_protocol"],
        "overall": _summarize(normalized, bins),
        "by_evidence_type": {kind: _summarize(rows, bins) for kind, rows in sorted(by_type.items())},
        "interpretation_limit": "Metrics describe agreement with the supplied adjudicated labels on this benchmark only. They do not establish clinical validity, source quality, or generalization beyond these cases.",
    }
    if review_summary is not None:
        result["adjudication_quality"] = review_summary
    return result


def calibrate_capability_reuse_threshold(
    document: dict[str, Any], *, minimum_precision: float = 0.95
) -> dict[str, Any]:
    """Fit a semantic cutoff on adjudicated capability pairs and evaluate it on a deterministic holdout."""
    allowed_top = {"schema_version", "benchmark_id", "adjudication_protocol", "embedding_model_id", "pairs"}
    if not isinstance(document, dict) or set(document) != allowed_top:
        raise ValueError("capability benchmark must contain the exact schema_version, benchmark_id, adjudication_protocol, embedding_model_id, and pairs fields")
    if document.get("schema_version") != "1":
        raise ValueError("capability benchmark schema_version must be 1")
    for field in ("benchmark_id", "adjudication_protocol", "embedding_model_id"):
        if not isinstance(document.get(field), str) or not document[field].strip() or len(document[field]) > 2000:
            raise ValueError(f"capability benchmark requires a non-empty {field}")
    if isinstance(minimum_precision, bool) or not isinstance(minimum_precision, (int, float)) or not math.isfinite(minimum_precision) or not 0.5 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be finite and in (0.5, 1]")
    pairs = document.get("pairs")
    fields = {"case_id", "task_sha256", "capability_id", "capability_version", "expected_reuse", "semantic_similarity", "contract_compatible", "permissions_compatible", "lifecycle_approved", "runtime_active"}
    if not isinstance(pairs, list) or len(pairs) < MIN_CAPABILITY_PAIRS or len(pairs) > MAX_CASES:
        raise ValueError(f"capability benchmark must contain {MIN_CAPABILITY_PAIRS}..{MAX_CASES} adjudicated pairs")
    normalized = []
    seen_cases: set[str] = set()
    seen_pairs: set[tuple[str, str, str]] = set()
    for index, row in enumerate(pairs):
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"capability pair {index} has missing or unsupported fields")
        for field in ("case_id", "capability_id", "capability_version"):
            value = row[field]
            if not isinstance(value, str) or not _LABEL.fullmatch(value):
                raise ValueError(f"capability pair {index} has an invalid {field}")
        task_digest = row["task_sha256"]
        if not isinstance(task_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", task_digest):
            raise ValueError(f"capability pair {index} task_sha256 must be a lowercase SHA-256 digest")
        if row["case_id"] in seen_cases:
            raise ValueError("capability benchmark case IDs must be unique")
        seen_cases.add(row["case_id"])
        identity = (task_digest, row["capability_id"], row["capability_version"])
        if identity in seen_pairs:
            raise ValueError("capability task/capability pairs must be unique")
        seen_pairs.add(identity)
        if type(row["expected_reuse"]) is not bool:
            raise ValueError(f"capability pair {index} expected_reuse must be boolean")
        score = row["semantic_similarity"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"capability pair {index} semantic_similarity must be finite and within 0..1")
        for gate in ("contract_compatible", "permissions_compatible", "lifecycle_approved", "runtime_active"):
            if type(row[gate]) is not bool:
                raise ValueError(f"capability pair {index} {gate} must be boolean")
        normalized.append({**row, "semantic_similarity": float(score)})

    eligible = [row for row in normalized if all(row[gate] for gate in ("contract_compatible", "permissions_compatible", "lifecycle_approved", "runtime_active"))]
    fit = [row for row in eligible if hashlib.sha256(row["task_sha256"].encode()).digest()[0] % 5 != 0]
    holdout = [row for row in eligible if hashlib.sha256(row["task_sha256"].encode()).digest()[0] % 5 == 0]
    fit_positive = sum(row["expected_reuse"] for row in fit)
    fit_negative = len(fit) - fit_positive
    holdout_positive = sum(row["expected_reuse"] for row in holdout)
    holdout_negative = len(holdout) - holdout_positive
    if fit_positive < 10 or fit_negative < 10 or holdout_positive < 5 or holdout_negative < 5:
        raise ValueError("capability benchmark needs at least 10 positive and 10 negative fit pairs and 5 of each in holdout after contract, permission, lifecycle, and runtime gates")

    candidates = sorted({row["semantic_similarity"] for row in fit})
    best = None
    for threshold in candidates:
        selected = [row for row in fit if row["semantic_similarity"] >= threshold]
        if not selected:
            continue
        true_positive = sum(row["expected_reuse"] for row in selected)
        precision = true_positive / len(selected)
        if precision < minimum_precision:
            continue
        recall = true_positive / fit_positive
        candidate = {"threshold": threshold, "precision": precision, "recall": recall, "n": len(selected), "true_positive": true_positive, "false_positive": len(selected) - true_positive}
        if best is None or (candidate["recall"], candidate["precision"], candidate["threshold"]) > (best["recall"], best["precision"], best["threshold"]):
            best = candidate
    if best is None:
        raise ValueError("no semantic threshold reaches the requested fit-set precision; capability reuse remains disabled")

    selected_holdout = [row for row in holdout if row["semantic_similarity"] >= best["threshold"]]
    holdout_tp = sum(row["expected_reuse"] for row in selected_holdout)
    holdout_precision = holdout_tp / len(selected_holdout) if selected_holdout else 0.0
    holdout_recall = holdout_tp / holdout_positive
    benchmark_sha256 = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    passes = len(selected_holdout) >= MIN_CAPABILITY_HOLDOUT_PREDICTIONS and holdout_precision >= minimum_precision
    return {
        "schema_version": "1",
        "benchmark_id": document["benchmark_id"],
        "benchmark_sha256": benchmark_sha256,
        "adjudication_protocol": document["adjudication_protocol"],
        "embedding_model_id": document["embedding_model_id"],
        "split_strategy": "grouped by task_sha256; SHA-256(task_sha256) first byte modulo 5 equals 0 for holdout",
        "minimum_precision": float(minimum_precision),
        "selected_threshold": best["threshold"],
        "fit_metrics": best,
        "holdout_metrics": {"n_eligible": len(holdout), "n_predicted_reuse": len(selected_holdout), "true_positive": holdout_tp, "false_positive": len(selected_holdout) - holdout_tp, "expected_positive": holdout_positive, "expected_negative": holdout_negative, "precision": holdout_precision, "recall": holdout_recall},
        "eligible_pair_count": len(eligible),
        "fit_pair_count": len(fit),
        "holdout_pair_count": len(holdout),
        "reuse_enabled": passes,
        "status": "holdout_passed" if passes else "holdout_failed",
        "interpretation_limit": "The threshold is evaluated only for this adjudicated task-capability population and embedding model. It does not establish biomedical validity. Contract, permission, lifecycle, and live route gates remain mandatory; deployment requires pinning this exact report digest.",
    }


def load_pinned_capability_calibration(
    report_path: str | Path,
    expected_sha256: str,
    *,
    expected_embedding_model_id: str | None = None,
) -> dict[str, Any]:
    """Load only a holdout-passing threshold whose exact report digest is operator-pinned."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("capability calibration requires a pinned lowercase SHA-256 digest")
    path = Path(report_path)
    if path.stat().st_size > 1_000_000:
        raise ValueError("capability calibration report exceeds 1 MiB")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected_sha256):
        raise ValueError("capability calibration report digest does not match the operator-pinned value")
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("capability calibration report is not valid JSON") from error
    if not isinstance(report, dict):
        raise ValueError("capability calibration report must be a JSON object")
    threshold = report.get("selected_threshold")
    if (
        report.get("schema_version") != "1"
        or report.get("status") != "holdout_passed"
        or report.get("reuse_enabled") is not True
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or not isinstance(report.get("embedding_model_id"), str)
        or not isinstance(report.get("benchmark_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", report["benchmark_sha256"])
    ):
        raise ValueError("capability calibration report did not pass its holdout or has an invalid policy")
    if expected_embedding_model_id is not None and report["embedding_model_id"] != expected_embedding_model_id:
        raise ValueError("capability calibration embedding model does not match the active model")
    return {
        "threshold": float(threshold),
        "report_sha256": actual,
        "benchmark_sha256": report["benchmark_sha256"],
        "benchmark_id": report.get("benchmark_id"),
        "embedding_model_id": report["embedding_model_id"],
        "status": report["status"],
    }
