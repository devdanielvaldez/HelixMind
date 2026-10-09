"""Traceable graph-path candidates and bounded, transparent criticism."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any

MAX_PROPOSAL_CANDIDATE_BYTES = 96 * 1024
MAX_PROPOSAL_CANDIDATES_BYTES = 3 * 1024 * 1024
MAX_PROPOSAL_EVIDENCE_IDS = 1024


def _edge_evidence_ids(edge: dict[str, Any]) -> list[str]:
    raw = edge.get("evidence_ids", "[]")
    values = json.loads(raw) if isinstance(raw, str) else raw
    return [str(value) for value in values] if isinstance(values, list) else []


def _validate_proposal_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("hypothesis proposal contains a malformed candidate")
    if candidate.get("classification") != "CANDIDATE RESEARCH HYPOTHESIS":
        raise ValueError("hypothesis candidate must remain explicitly classified as a research hypothesis")
    statement = candidate.get("statement")
    if not isinstance(statement, str) or not statement.strip() or len(statement) > 2000:
        raise ValueError("hypothesis candidate lacks a bounded research statement")
    path = candidate.get("path")
    path_evidence = candidate.get("path_evidence")
    if not isinstance(path, list) or not 2 <= len(path) <= 4 or not isinstance(path_evidence, list) or len(path_evidence) != len(path):
        raise ValueError("hypothesis candidate requires two to four path edges and matching evidence receipts")
    node_sequence: list[str] = []
    all_edge_evidence: list[str] = []
    for index, (edge, receipt) in enumerate(zip(path, path_evidence)):
        if not isinstance(edge, dict) or not isinstance(receipt, dict):
            raise ValueError("hypothesis path edge and evidence receipt must be objects")
        source, relation, target = (edge.get(key) for key in ("source", "relation", "target"))
        if any(not isinstance(value, str) or not value or len(value) > 256 for value in (source, relation, target)):
            raise ValueError("hypothesis path edge has invalid identifiers")
        if index and source != node_sequence[-1]:
            raise ValueError("hypothesis path edges are not connected")
        if not index:
            node_sequence.append(source)
        if target in node_sequence:
            raise ValueError("hypothesis path must be simple and cannot repeat a node")
        node_sequence.append(target)
        if any(receipt.get(key) != edge[key] for key in ("source", "relation", "target")):
            raise ValueError("hypothesis edge evidence receipt refers to a different path edge")
        evidence_ids = receipt.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) > MAX_PROPOSAL_EVIDENCE_IDS:
            raise ValueError("hypothesis path edge has no bounded evidence identifiers")
        if any(not isinstance(item, str) or not item or len(item) > 256 for item in evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("hypothesis path edge evidence identifiers are invalid or duplicated")
        counts = receipt.get("polarity_counts")
        if not isinstance(counts, dict) or any(
            isinstance(counts.get(polarity), bool)
            or not isinstance(counts.get(polarity), int)
            or counts[polarity] < 0
            for polarity in ("supports", "contradicts", "context")
        ) or counts["supports"] < 1:
            raise ValueError("hypothesis path edge requires valid source polarity counts and supporting evidence")
        if sum(counts[key] for key in ("supports", "contradicts", "context")) != len(evidence_ids):
            raise ValueError("hypothesis edge polarity counts do not match its evidence identifiers")
        interpretation = receipt.get("interpretation")
        if not isinstance(interpretation, str) or not interpretation or len(interpretation) > 500:
            raise ValueError("hypothesis edge receipt lacks its interpretation limit")
        all_edge_evidence.extend(evidence_ids)
    expected_evidence_ids = list(dict.fromkeys(all_edge_evidence))
    if candidate.get("evidence_ids") != expected_evidence_ids:
        raise ValueError("hypothesis evidence identifiers do not match the ordered edge receipts")
    dimensions = candidate.get("dimensions")
    if not isinstance(dimensions, dict) or dimensions.get("novelty") is not None:
        raise ValueError("hypothesis dimensions must retain unscored novelty explicitly")
    for metric in ("supporting_records", "contradictory_records", "context_records"):
        value = dimensions.get(metric)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= len(expected_evidence_ids):
            raise ValueError("hypothesis evidence counts are invalid")
    if sum(dimensions[metric] for metric in ("supporting_records", "contradictory_records", "context_records")) != len(expected_evidence_ids):
        raise ValueError("hypothesis evidence dimensions do not match its unique evidence identifiers")
    quality_count = dimensions.get("quality_annotated_records")
    if quality_count is None and "quality_annotated_records" not in dimensions:
        # Read retained pre-denominator artifacts using their original all-record mean contract.
        quality_count = len(expected_evidence_ids) if dimensions.get("mean_record_quality_annotation") is not None else 0
    if isinstance(quality_count, bool) or not isinstance(quality_count, int) or not 0 <= quality_count <= len(expected_evidence_ids):
        raise ValueError("hypothesis quality-annotation count is invalid")
    quality_mean = dimensions.get("mean_record_quality_annotation")
    if quality_count == 0:
        if quality_mean is not None:
            raise ValueError("hypothesis quality mean must be unavailable when no records are annotated")
    elif isinstance(quality_mean, bool) or not isinstance(quality_mean, (int, float)) or not math.isfinite(quality_mean) or not 0 <= quality_mean <= 1:
        raise ValueError("hypothesis quality mean is invalid")
    confidence_count = dimensions.get("edge_confidence_annotated_count")
    if confidence_count is None and "edge_confidence_annotated_count" not in dimensions:
        # Retain compatibility with proposal artifacts written before annotation denominators existed.
        confidence_count = len(path) if dimensions.get("mean_edge_confidence_annotation") is not None else 0
    if isinstance(confidence_count, bool) or not isinstance(confidence_count, int) or not 0 <= confidence_count <= len(path):
        raise ValueError("hypothesis edge-confidence annotation count is invalid")
    confidence_mean = dimensions.get("mean_edge_confidence_annotation")
    if confidence_count == 0:
        if confidence_mean is not None:
            raise ValueError("hypothesis edge-confidence mean must be unavailable when no edges are annotated")
    elif isinstance(confidence_mean, bool) or not isinstance(confidence_mean, (int, float)) or not math.isfinite(confidence_mean) or not 0 <= confidence_mean <= 1:
        raise ValueError("hypothesis edge-confidence mean is invalid")
    value = dimensions.get("contradictory_share_of_linked_records")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("hypothesis descriptive dimensions must be finite annotations from zero to one")
    labels = candidate.get("entity_labels")
    if labels is not None:
        if (not isinstance(labels, list) or len(labels) != len(node_sequence)
                or any(not isinstance(label, str) or not label.strip() or len(label) > 256 for label in labels)):
            raise ValueError("hypothesis entity labels do not match its path nodes")
        expected_statement = f"{labels[0]} may be connected to {labels[-1]} through {' → '.join(labels[1:-1])}."
        if statement != expected_statement:
            raise ValueError("hypothesis statement does not match its ordered path labels")


def validate_hypothesis_proposal_artifact(artifact: Any, session_id: str) -> dict[str, Any]:
    """Verify retained proposal identity and content before scientific review."""
    if not isinstance(artifact, dict) or artifact.get("session_id") != session_id:
        raise ValueError("hypothesis proposal artifact is outside the requested research session")
    if artifact.get("kind") != "hypothesis_candidates":
        raise ValueError("artifact is not a retained hypothesis proposal")
    payload_text = artifact.get("payload")
    digest = artifact.get("sha256")
    if not isinstance(payload_text, str) or len(payload_text.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("hypothesis proposal payload is missing or exceeds 4 MiB")
    if not isinstance(digest, str) or hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != digest:
        raise ValueError("hypothesis proposal payload hash does not match its retained digest")

    class DuplicateProposalKey(ValueError):
        pass

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateProposalKey("hypothesis proposal payload contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        proposal = json.loads(
            payload_text,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except DuplicateProposalKey:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("hypothesis proposal payload is not strict JSON") from error
    if not isinstance(proposal, dict) or not isinstance(proposal.get("candidates"), list) or len(proposal["candidates"]) > 200:
        raise ValueError("hypothesis proposal payload has an invalid candidate list")
    candidate_count = proposal.get("candidate_count")
    returned_count = proposal.get("returned_count")
    truncated = proposal.get("truncated")
    # Legacy two-edge artifacts predate the explicit traversal-bound field.
    traversal_truncated = proposal.get("traversal_truncated", False)
    if (
        isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0
        or isinstance(returned_count, bool) or not isinstance(returned_count, int) or returned_count != len(proposal["candidates"])
        or candidate_count < returned_count
        or not isinstance(truncated, bool) or not isinstance(traversal_truncated, bool)
        or (not truncated and (traversal_truncated or candidate_count != returned_count))
    ):
        raise ValueError("hypothesis proposal counts and truncation flags are inconsistent")
    size_fields = ("candidate_limit_omitted_count", "candidate_byte_omitted_count", "candidate_payload_bytes")
    present_size_fields = [field in proposal for field in size_fields]
    if any(present_size_fields):
        if not all(present_size_fields):
            raise ValueError("hypothesis proposal byte-bound receipt is incomplete")
        size_values = [proposal[field] for field in size_fields]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in size_values):
            raise ValueError("hypothesis proposal byte-bound counters are invalid")
        actual_candidate_bytes = sum(
            len(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
            for row in proposal["candidates"]
        )
        if (
            proposal["candidate_payload_bytes"] != actual_candidate_bytes
            or actual_candidate_bytes > MAX_PROPOSAL_CANDIDATES_BYTES
            or candidate_count != returned_count + proposal["candidate_limit_omitted_count"] + proposal["candidate_byte_omitted_count"]
            or truncated != (
                proposal["candidate_limit_omitted_count"] > 0
                or proposal["candidate_byte_omitted_count"] > 0
                or traversal_truncated
            )
        ):
            raise ValueError("hypothesis proposal byte-bound receipt does not match its candidate payload")
    candidate_ids = [
        row.get("id") if isinstance(row, dict) else None
        for row in proposal["candidates"]
    ]
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in candidate_ids):
        raise ValueError("hypothesis proposal contains a candidate without a stable ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("hypothesis proposal contains duplicate candidate IDs")
    for candidate in proposal["candidates"]:
        candidate_bytes = len(json.dumps(candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        if candidate_bytes > MAX_PROPOSAL_CANDIDATE_BYTES:
            raise ValueError("hypothesis candidate exceeds its retained byte bound")
        _validate_proposal_candidate(candidate)
    return proposal


def propose_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int = 50,
    max_edges: int = 4,
) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise ValueError("candidate limit must be 1..200")
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or not 2 <= max_edges <= 4:
        raise ValueError("maximum hypothesis path length must be 2..4 edges")
    node_by_id = {node["id"]: node for node in nodes}
    evidence_by_id = {item["id"]: item for item in evidence}
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        ids = _edge_evidence_ids(edge)
        if edge.get("source") in node_by_id and edge.get("target") in node_by_id and ids and all(item in evidence_by_id for item in ids):
            adjacency[edge["source"]].append({**edge, "_evidence_ids": ids})

    for rows in adjacency.values():
        rows.sort(key=lambda edge: (str(edge.get("target")), str(edge.get("relation")), tuple(edge["_evidence_ids"])))

    candidates: dict[str, dict[str, Any]] = {}
    seen_candidate_ids: set[str] = set()
    candidate_count = 0
    eligible_candidate_count = 0
    candidate_byte_omitted_count = 0
    traversed_paths = 0
    traversal_truncated = False
    max_traversed_paths = 50_000

    def candidate_sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
        return (row["dimensions"]["contradictory_share_of_linked_records"], -row["dimensions"]["supporting_records"], row["id"])

    def retain_path(source_id: str, path_edges: list[dict[str, Any]]) -> None:
        nonlocal candidate_count, eligible_candidate_count, candidate_byte_omitted_count
        node_ids = [source_id] + [edge["target"] for edge in path_edges]
        path = [
            {"source": edge["source"], "relation": edge["relation"], "target": edge["target"]}
            for edge in path_edges
        ]
        canonical = json.dumps(path, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        identifier = "HYP-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
        if identifier in seen_candidate_ids:
            return
        seen_candidate_ids.add(identifier)
        candidate_count += 1
        seen_evidence_ids: set[str] = set()
        unique_evidence_count = 0
        for edge in path_edges:
            if len(edge["_evidence_ids"]) > MAX_PROPOSAL_EVIDENCE_IDS:
                candidate_byte_omitted_count += 1
                return
            for evidence_id in edge["_evidence_ids"]:
                if evidence_id not in seen_evidence_ids:
                    seen_evidence_ids.add(evidence_id)
                    unique_evidence_count += 1
                    if unique_evidence_count > MAX_PROPOSAL_EVIDENCE_IDS:
                        candidate_byte_omitted_count += 1
                        return
        linked_ids = list(dict.fromkeys(item for edge in path_edges for item in edge["_evidence_ids"]))
        records = [evidence_by_id[item] for item in linked_ids]
        support = [record for record in records if record.get("polarity") == "supports"]
        contradiction = [record for record in records if record.get("polarity") == "contradicts"]
        context = [record for record in records if record.get("polarity") == "context"]
        source, target = node_by_id[node_ids[0]], node_by_id[node_ids[-1]]
        edge_confidences = []
        for edge in path_edges:
            confidence = edge.get("confidence")
            if confidence is None:
                continue
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("hypothesis edge confidence annotations must be finite values from zero to one or null")
            edge_confidences.append(float(confidence))
        all_quality = []
        for record in records:
            quality = record.get("quality")
            if quality is None:
                continue
            if (isinstance(quality, bool) or not isinstance(quality, (int, float))
                    or not math.isfinite(quality) or not 0 <= quality <= 1):
                raise ValueError("hypothesis evidence quality annotations must be finite values from zero to one or null")
            all_quality.append(float(quality))
        path_evidence = []
        for edge in path_edges:
            edge_records = [evidence_by_id[item] for item in edge["_evidence_ids"]]
            path_evidence.append({
                **{key: edge[key] for key in ("source", "relation", "target")},
                "evidence_ids": edge["_evidence_ids"],
                "polarity_counts": {polarity: sum(record.get("polarity") == polarity for record in edge_records) for polarity in ("supports", "contradicts", "context")},
                "interpretation": "Source-assigned polarity counts for this edge only; they are not independently validated or calibrated.",
            })
        labels = [str(node_by_id[item]["label"]) for item in node_ids]
        intermediates = labels[1:-1]
        route = " → ".join(intermediates)
        candidate = {
            "id": identifier,
            "classification": "CANDIDATE RESEARCH HYPOTHESIS",
            "statement": f"{source['label']} may be connected to {target['label']} through {route}.",
            "entity_labels": labels,
            "path": path,
            "path_evidence": path_evidence,
            "dimensions": {
                "supporting_records": len(support),
                "contradictory_records": len(contradiction),
                "context_records": len(context),
                "mean_record_quality_annotation": round(sum(all_quality) / len(all_quality), 4) if all_quality else None,
                "quality_annotated_records": len(all_quality),
                "mean_edge_confidence_annotation": round(sum(edge_confidences) / len(edge_confidences), 4) if edge_confidences else None,
                "edge_confidence_annotated_count": len(edge_confidences),
                "contradictory_share_of_linked_records": round(len(contradiction) / len(records), 4) if records else 0,
                "novelty": None,
            },
            "evidence_ids": linked_ids,
            "limitations": [
                "These are descriptive annotations, not calibrated probabilities.",
                "A graph path does not establish a direct or causal biological mechanism.",
                "Novelty is unscored because no complete literature comparator is available.",
            ],
        }
        encoded_size = len(json.dumps(candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        if encoded_size > MAX_PROPOSAL_CANDIDATE_BYTES:
            candidate_byte_omitted_count += 1
            return
        eligible_candidate_count += 1
        candidates[identifier] = candidate
        if len(candidates) > limit:
            worst_id = max(candidates, key=lambda key: candidate_sort_key(candidates[key]))
            del candidates[worst_id]

    for source_id in sorted(adjacency):
        def walk(current_id: str, path_edges: list[dict[str, Any]], visited: set[str]) -> None:
            nonlocal traversed_paths, traversal_truncated
            if traversal_truncated:
                return
            for edge in adjacency.get(current_id, []):
                target_id = edge["target"]
                if target_id in visited or not any(
                    evidence_by_id[item].get("polarity") == "supports" for item in edge["_evidence_ids"]
                ):
                    continue
                next_path = [*path_edges, edge]
                next_visited = {*visited, target_id}
                if len(next_path) >= 2:
                    traversed_paths += 1
                    if traversed_paths > max_traversed_paths:
                        traversal_truncated = True
                        return
                    retain_path(source_id, next_path)
                if len(next_path) < max_edges:
                    walk(target_id, next_path, next_visited)
                if traversal_truncated:
                    return

        walk(source_id, [], {source_id})
        if traversal_truncated:
            break
    ordered = sorted(candidates.values(), key=candidate_sort_key)
    bounded_candidates = []
    bounded_candidate_bytes = 0
    for candidate in ordered:
        encoded_size = len(json.dumps(candidate, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
        if bounded_candidate_bytes + encoded_size > MAX_PROPOSAL_CANDIDATES_BYTES:
            candidate_byte_omitted_count += 1
            continue
        bounded_candidates.append(candidate)
        bounded_candidate_bytes += encoded_size
    candidate_limit_omitted_count = eligible_candidate_count - len(ordered)
    return {
        "method": f"bounded directed simple-path enumeration from 2 to {max_edges} edges with session-retained evidence",
        "candidate_count": candidate_count,
        "returned_count": len(bounded_candidates),
        "candidates": bounded_candidates,
        "candidate_limit_omitted_count": candidate_limit_omitted_count,
        "candidate_byte_omitted_count": candidate_byte_omitted_count,
        "candidate_payload_bytes": bounded_candidate_bytes,
        "truncated": candidate_limit_omitted_count > 0 or candidate_byte_omitted_count > 0 or traversal_truncated,
        "traversal_truncated": traversal_truncated,
        "traversed_path_count": min(traversed_paths, max_traversed_paths),
        "interpretation": "Candidate generation for scientist review; no candidate is a confirmed discovery.",
    }


def critique_candidate(
    candidate: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    lead_limit: int = 20,
) -> dict[str, Any]:
    if not 1 <= lead_limit <= 100:
        raise ValueError("critic lead limit must be 1..100")
    evidence_by_id = {item["id"]: item for item in evidence}
    linked = [evidence_by_id[item] for item in candidate.get("evidence_ids", []) if item in evidence_by_id]
    contrary = [item for item in linked if item.get("polarity") == "contradicts"]
    path = candidate.get("path", [])
    path_ids = {item.get("source") for item in path} | {item.get("target") for item in path}
    labels = {node["id"]: node.get("label", "") for node in nodes if node.get("id") in path_ids}
    terms = {token.lower() for label in labels.values() for token in re.findall(r"[A-Za-z0-9-]{3,}", label)}
    linked_ids = set(candidate.get("evidence_ids", []))
    refutation_leads = []
    for item in evidence:
        if item["id"] in linked_ids or item.get("polarity") != "contradicts":
            continue
        claim_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9-]{3,}", item.get("claim", ""))}
        overlap = sorted(terms & claim_terms)
        if overlap:
            refutation_leads.append({"evidence_id": item["id"], "claim": item.get("claim", ""), "matched_terms": overlap, "quality_annotation": item.get("quality"), "match_type": "lexical review lead; semantic relevance not established"})
    alternatives = []
    alternative_route_checks = 0
    alternative_routes_truncated = False
    if 2 <= len(path) <= 4:
        source_id, target_id = path[0].get("source"), path[-1].get("target")
        primary_signature = tuple((item.get("source"), item.get("relation"), item.get("target")) for item in path)
        route_adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
        for edge in edges:
            edge_ids = _edge_evidence_ids(edge)
            if (
                edge.get("source") in node_ids
                and edge.get("target") in node_ids
                and edge_ids
                and all(item in evidence_by_id for item in edge_ids)
            ):
                route_adjacency[edge["source"]].append({**edge, "_evidence_ids": edge_ids})
        for rows in route_adjacency.values():
            rows.sort(key=lambda edge: (str(edge.get("target")), str(edge.get("relation")), tuple(edge["_evidence_ids"])))

        max_route_checks = 50_000
        primary_nodes = {source_id}

        def record_alternative(route: list[dict[str, Any]]) -> None:
            linked_route_ids = list(dict.fromkeys(item for edge in route for item in edge["_evidence_ids"]))
            records = [evidence_by_id[item] for item in linked_route_ids]
            route_rows = [
                {"source": edge["source"], "relation": edge["relation"], "target": edge["target"], "evidence_ids": edge["_evidence_ids"]}
                for edge in route
            ]
            signature = tuple((item["source"], item["relation"], item["target"]) for item in route_rows)
            if signature == primary_signature:
                return
            alternatives.append({
                "intermediate": [edge["target"] for edge in route[:-1]],
                "path": route_rows,
                "evidence_ids": linked_route_ids,
                "evidence_polarity_counts": {polarity: sum(row.get("polarity") == polarity for row in records) for polarity in ("supports", "contradicts", "context")},
                "interpretation": "Alternative evidence-linked graph route; source polarity labels are reported descriptively and do not independently refute the candidate.",
            })

        def find_alternatives(current_id: str, route: list[dict[str, Any]], visited: set[str]) -> None:
            nonlocal alternative_route_checks
            if alternative_route_checks >= max_route_checks or len(alternatives) >= lead_limit:
                return
            if len(route) == len(path):
                if current_id == target_id:
                    alternative_route_checks += 1
                    record_alternative(route)
                return
            for edge in route_adjacency.get(current_id, []):
                next_id = edge["target"]
                if next_id in visited:
                    continue
                alternative_route_checks += 1
                if alternative_route_checks > max_route_checks:
                    return
                find_alternatives(next_id, [*route, edge], {*visited, next_id})
                if len(alternatives) >= lead_limit or alternative_route_checks >= max_route_checks:
                    return

        if isinstance(source_id, str) and isinstance(target_id, str):
            find_alternatives(source_id, [], primary_nodes)
            alternative_routes_truncated = len(alternatives) >= lead_limit or alternative_route_checks >= max_route_checks
    alternatives.sort(key=lambda row: (str(row["intermediate"]), tuple(row["evidence_ids"])))
    return {
        "hypothesis_id": candidate.get("id"),
        "classification": "SCIENTIFIC CRITIC REVIEW",
        "linked_contradictory_evidence": [{"evidence_id": item["id"], "claim": item.get("claim"), "quality_annotation": item.get("quality")} for item in contrary],
        "lexical_refutation_leads": refutation_leads[:lead_limit],
        "alternative_graph_routes": alternatives[:lead_limit],
        "alternative_graph_routes_truncated": alternative_routes_truncated,
        "alternative_graph_route_checks": min(alternative_route_checks, 50_000),
        "major_weaknesses": [
            "Lexical overlap is only a search lead and requires scientific relevance review.",
            "Missing contradictory records cannot be interpreted as evidence of absence.",
            "Edge confidence and evidence quality are manually supplied annotations and are not calibrated.",
            "Population, model system, methods, and study limitations have not been reconciled.",
        ],
        "review_status": "needs_review_due_to_alternative_route_limit" if alternative_routes_truncated else "requires_human_scientific_review",
    }


def link_contradiction_search_candidates(
    candidate: dict[str, Any],
    artifacts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach PubMed retrieval candidates only to the exact supporting evidence they searched."""
    evidence_by_id = {row.get("id"): row for row in evidence if isinstance(row, dict)}
    candidate_evidence_ids = set(candidate.get("evidence_ids", []))
    linked = []
    seen: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "contradiction_search_candidate":
            raise ValueError("critic source must be a contradiction-search candidate artifact")
        artifact_id = artifact.get("id")
        payload_text = artifact.get("payload")
        digest = artifact.get("sha256")
        if not isinstance(artifact_id, str) or not isinstance(payload_text, str) or not isinstance(digest, str):
            raise ValueError("contradiction-search artifact is missing its identity or retained payload")
        if hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != digest:
            raise ValueError("contradiction-search artifact payload hash does not match")
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as error:
            raise ValueError("contradiction-search artifact payload is not valid JSON") from error
        if not isinstance(payload, dict) or payload.get("interpretation") != "Retrieved as a contradiction-search candidate only; no contradictory relation or evidence polarity was inferred.":
            raise ValueError("artifact does not preserve contradiction-search candidate semantics")
        target_id = payload.get("target_evidence_id")
        paper = payload.get("paper")
        source_id = artifact.get("source_id")
        if not isinstance(target_id, str) or target_id not in candidate_evidence_ids:
            continue
        target = evidence_by_id.get(target_id)
        if not isinstance(target, dict) or target.get("polarity") != "supports":
            continue
        provenance = paper.get("provenance") if isinstance(paper, dict) else None
        if (
            artifact.get("source") != "PubMed"
            or not isinstance(source_id, str)
            or not isinstance(provenance, dict)
            or provenance.get("source") != "PubMed"
            or str(provenance.get("source_id")) != source_id
        ):
            raise ValueError("contradiction-search paper source identity does not match its artifact")
        key = (artifact_id, target_id)
        if key in seen:
            continue
        seen.add(key)
        linked.append({
            "candidate_artifact_id": artifact_id,
            "candidate_artifact_sha256": digest,
            "target_evidence_id": target_id,
            "source": "PubMed",
            "source_id": source_id,
            "title": str(paper.get("title", ""))[:500],
            "publication_date": paper.get("publication_date"),
            "journal": str(paper.get("journal", ""))[:200],
            "doi": paper.get("doi"),
            "relationship": "retrieved_for_linked_supporting_claim_unassessed",
            "review_status": "requires_human_scientific_review",
            "interpretation": "This record was retrieved by a query targeting the linked supporting claim. Its results have not been assessed as supporting or contradicting the hypothesis.",
        })
    return sorted(linked, key=lambda row: (row["target_evidence_id"], row["source_id"], row["candidate_artifact_id"]))
