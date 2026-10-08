"""Source-linked candidate alignment for machine-labeled PMC result sentences.

This module identifies review leads from exact shared BioC entity identifiers and
matching result categories. It does not establish semantic equivalence or
scientific contradiction.
"""
from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict
from typing import Any


MAX_ANNOTATION_ARTIFACTS = 250
MAX_ANNOTATIONS_PER_ARTIFACT = 200
MAX_TOTAL_ANNOTATIONS = 50_000
MAX_LINKS = 1000
MAX_COMPARISONS = 100_000
_STUDY_SYSTEMS = {"human", "animal", "cell", "computational", "other", "unclear"}
_PAPER_CONTEXT_ENUMS = {
    "study_design": {"randomized_trial", "nonrandomized_intervention", "observational", "case_report_series", "systematic_review_meta_analysis", "animal_experiment", "cell_tissue_experiment", "computational", "mixed", "unclear"},
    "population": {"patients", "healthy_volunteers", "general_population", "animals", "cell_tissue", "data_only", "mixed", "unclear"},
    "method_family": {"clinical", "cohort_or_case_control", "molecular_assay", "animal_model", "cell_model", "computational_analysis", "literature_synthesis", "mixed", "unclear"},
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")


def decode_verified_annotation_artifact(artifact: dict[str, Any], session_id: str) -> dict[str, Any]:
    """Verify the exact retained artifact envelope before consuming model labels."""
    if not isinstance(artifact, dict) or artifact.get("session_id") != session_id:
        raise ValueError("annotation artifact must belong to the requested research session")
    if artifact.get("kind") != "paper_result_annotations" or artifact.get("source") != "PMC BioC local-model result annotation":
        raise ValueError("artifact is not a retained PMC result-annotation artifact")
    payload_text = artifact.get("payload")
    if not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > 1_048_576:
        raise ValueError("annotation artifact payload is malformed or exceeds 1 MiB")
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("annotation artifact has no valid payload digest")
    if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != digest:
        raise ValueError("annotation artifact payload hash does not match its retained bytes")
    try:
        payload = json.loads(
            payload_text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("annotation artifact payload is not strict, unambiguous JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("annotation artifact payload must be a JSON object")
    if not isinstance(payload.get("document_id"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_sha256", ""))):
        raise ValueError("annotation artifact lacks its stable document and source hashes")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or len(annotations) > MAX_ANNOTATIONS_PER_ARTIFACT:
        raise ValueError("annotation artifact has a missing or oversized annotation list")
    artifact["decoded_payload"] = payload
    return artifact


def validate_semantic_claim_pair(annotation_a: dict[str, Any], annotation_b: dict[str, Any]) -> list[str]:
    """Require source annotations to share a normalized entity and result class."""
    result_kinds = {"association", "intervention", "expression", "clinical", "animal_model", "cell_model", "computational", "null_result", "other", "insufficient_context"}
    directions = {"increase", "decrease", "mixed", "no_detected_effect", "not_applicable", "unclear"}
    for annotation in (annotation_a, annotation_b):
        if not isinstance(annotation, dict):
            raise ValueError("semantic comparison requires two structured source annotations")
        if annotation.get("result_kind") not in result_kinds or annotation.get("direction") not in directions:
            raise ValueError("semantic comparison annotations contain unsupported result labels")
        entity_ids = annotation.get("entity_ids")
        if not isinstance(entity_ids, list) or any(not isinstance(item, str) or not item for item in entity_ids):
            raise ValueError("semantic comparison annotations require normalized source entity IDs")
    if annotation_a["result_kind"] != annotation_b["result_kind"]:
        raise ValueError("semantic comparison requires the same source-annotated result category")
    shared = sorted(set(annotation_a["entity_ids"]) & set(annotation_b["entity_ids"]))
    if not shared:
        raise ValueError("semantic comparison requires at least one exact shared source entity ID")
    return shared


def build_claim_alignment_candidates(artifacts: list[dict[str, Any]], *, session_id: str) -> dict[str, Any]:
    """Link cross-paper result labels by exact entities and compatible categories.

    Links retain stable artifact/document/candidate identifiers and source hashes,
    not full-text sentences. Directional disagreement is only a review lead.
    """
    if not isinstance(artifacts, list) or not isinstance(session_id, str) or not session_id:
        raise ValueError("claim annotation artifacts require an array and exact research session ID")
    rows = sorted(
        (row for row in artifacts if isinstance(row, dict)),
        key=lambda row: str(row.get("id", "")),
    )
    truncated_artifacts = len(rows) > MAX_ANNOTATION_ARTIFACTS
    rows = rows[:MAX_ANNOTATION_ARTIFACTS]
    groups: dict[tuple[str, str], dict[str, dict[str, list[dict[str, Any]]]]] = defaultdict(lambda: {"increase": defaultdict(list), "decrease": defaultdict(list)})
    annotations_considered = 0
    truncated_annotations = False
    invalid_artifacts = 0
    for artifact in rows:
        if annotations_considered >= MAX_TOTAL_ANNOTATIONS:
            truncated_annotations = True
            break
        try:
            verified = decode_verified_annotation_artifact(artifact, session_id)
        except ValueError:
            invalid_artifacts += 1
            continue
        payload = verified["decoded_payload"]
        document_id = payload.get("document_id")
        source_hash = payload.get("source_sha256")
        annotations = payload.get("annotations")
        anchors = payload.get("source_anchors")
        text_hashes = payload.get("candidate_text_hashes")
        paper_context = payload.get("paper_context")
        if not isinstance(paper_context, dict):
            paper_context = {}
        paper_context = {
            field: paper_context.get(field) if isinstance(paper_context.get(field), str) and paper_context.get(field) in allowed else "unclear"
            for field, allowed in _PAPER_CONTEXT_ENUMS.items()
        }
        paper_context_conflict = payload.get("paper_context_batch_conflict") is True
        if not isinstance(document_id, str) or not document_id or not isinstance(source_hash, str) or len(source_hash) != 64:
            continue
        if not isinstance(annotations, list):
            continue
        if len(annotations) > MAX_ANNOTATIONS_PER_ARTIFACT:
            truncated_annotations = True
        for annotation in annotations[:MAX_ANNOTATIONS_PER_ARTIFACT]:
            if annotations_considered >= MAX_TOTAL_ANNOTATIONS:
                truncated_annotations = True
                break
            annotations_considered += 1
            if not isinstance(annotation, dict):
                continue
            candidate_id = annotation.get("candidate_id")
            result_kind = annotation.get("result_kind")
            direction = annotation.get("direction")
            entity_ids = annotation.get("entity_ids")
            if not isinstance(candidate_id, str) or direction not in {"increase", "decrease"} or not isinstance(result_kind, str):
                continue
            if not isinstance(entity_ids, list):
                continue
            study_system = annotation.get("study_system")
            if not isinstance(study_system, str) or study_system not in _STUDY_SYSTEMS:
                study_system = "unclear"
            source_anchor = anchors.get(candidate_id) if isinstance(anchors, dict) else None
            text_hash = text_hashes.get(candidate_id) if isinstance(text_hashes, dict) else None
            if not isinstance(source_anchor, dict) or not isinstance(text_hash, str) or len(text_hash) != 64:
                continue
            candidate = {
                "artifact_id": artifact.get("id"),
                "source_id": artifact.get("source_id"),
                "document_id": document_id,
                "source_sha256": source_hash,
                "candidate_id": candidate_id,
                "candidate_text_sha256": text_hash,
                "passage_index": source_anchor.get("passage_index"),
                "sentence_index": source_anchor.get("sentence_index"),
                "source_offset": source_anchor.get("source_offset"),
                "source_local_offset": source_anchor.get("source_local_offset"),
                "source_length": source_anchor.get("source_length"),
                "result_kind": result_kind,
                "direction": direction,
                "study_system": study_system,
                "paper_context": paper_context,
                "paper_context_batch_conflict": paper_context_conflict,
            }
            for entity_id in sorted(set(item for item in entity_ids if isinstance(item, str) and item)):
                groups[(entity_id, result_kind)][direction][document_id].append(candidate)

    links_by_key: dict[
        tuple[tuple[str, str], tuple[str, str]], dict[str, Any]
    ] = {}
    comparisons = 0
    truncated_links = False
    truncated_comparisons = False
    for (entity_id, result_kind), directions in sorted(groups.items()):
        increase_docs = directions["increase"]
        decrease_docs = directions["decrease"]
        for increase_document in sorted(increase_docs):
            for decrease_document in sorted(decrease_docs):
                if increase_document == decrease_document:
                    continue
                for increase in sorted(increase_docs[increase_document], key=lambda item: item["candidate_id"]):
                    for decrease in sorted(decrease_docs[decrease_document], key=lambda item: item["candidate_id"]):
                        if comparisons >= MAX_COMPARISONS:
                            truncated_comparisons = True
                            break
                        comparisons += 1
                        first, second = sorted((increase, decrease), key=lambda item: (item["document_id"], item["candidate_id"]))
                        key = (
                            (first["document_id"], first["candidate_id"]),
                            (second["document_id"], second["candidate_id"]),
                        )
                        if key not in links_by_key:
                            if len(links_by_key) >= MAX_LINKS:
                                truncated_links = True
                                break
                            links_by_key[key] = {
                                "relation": "potential_directional_disagreement",
                                "shared_entity_id": entity_id,
                                "shared_entity_ids": [entity_id],
                                "result_kind": result_kind,
                                "sources": [first, second],
                                "study_context_comparison": _compare_study_context(first, second),
                                "review_status": "unreviewed_candidate",
                                "interpretation": "Opposite model-labeled directions for a shared exact entity and result category; this is not semantic claim alignment or a confirmed contradiction.",
                            }
                        elif entity_id not in links_by_key[key]["shared_entity_ids"]:
                            links_by_key[key]["shared_entity_ids"].append(entity_id)
                    if truncated_links or truncated_comparisons:
                        break
                if truncated_links or truncated_comparisons:
                    break
            if truncated_links or truncated_comparisons:
                break
        if truncated_links or truncated_comparisons:
            break

    return {
        "schema_version": "1",
        "method": "exact normalized entity ID + exact result category + opposite categorical direction",
        "links": [links_by_key[key] for key in sorted(links_by_key)],
        "annotation_artifacts_considered": len(rows),
        "invalid_annotation_artifact_count": invalid_artifacts,
        "candidate_pair_comparisons": comparisons,
        "annotations_considered": annotations_considered,
        "truncated": truncated_artifacts or truncated_annotations or truncated_links or truncated_comparisons,
        "needs_review": invalid_artifacts > 0 or truncated_artifacts or truncated_annotations or truncated_links or truncated_comparisons,
        "interpretation": "Candidate retrieval only from same-session hash-verified annotation artifacts. Invalid artifacts are excluded and require review. It does not align claim meaning, statistical uncertainty, or causal validity. Categorical study-system and paper-context comparisons are unvalidated model labels for human review, not evidence of contradiction or scientific comparability.",
    }


def _compare_study_context(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare categorical local-model context while refusing semantic comparability claims."""
    first_system = first.get("study_system", "unclear")
    second_system = second.get("study_system", "unclear")
    def relation(left: Any, right: Any, unknown: set[str]) -> str:
        if not isinstance(left, str) or not isinstance(right, str) or left in unknown or right in unknown:
            return "unassessed"
        return "same_category_label" if left == right else "different_category_labels"

    first_context = first.get("paper_context") if isinstance(first.get("paper_context"), dict) else {}
    second_context = second.get("paper_context") if isinstance(second.get("paper_context"), dict) else {}
    dimensions = {
        field: {
            "source_labels": [first_context.get(field, "unclear"), second_context.get(field, "unclear")],
            "relation": relation(first_context.get(field), second_context.get(field), {"unclear"}),
        }
        for field in _PAPER_CONTEXT_ENUMS
    }
    return {
        "study_system": {
            "source_labels": [first_system, second_system],
            "relation": relation(first_system, second_system, {"unclear", "other"}),
        },
        "paper_context_dimensions": dimensions,
        "batch_label_conflict": first.get("paper_context_batch_conflict") is True or second.get("paper_context_batch_conflict") is True,
        "review_required": True,
        "interpretation_limit": "All study-system, study-design, population, and method-family labels are unvalidated local-model screens. Equal categorical labels do not establish semantic or scientific comparability; differing labels are review context only.",
    }
