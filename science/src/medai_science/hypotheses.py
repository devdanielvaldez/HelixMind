"""Traceable graph-path candidates and bounded, transparent criticism."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any


def _edge_evidence_ids(edge: dict[str, Any]) -> list[str]:
    raw = edge.get("evidence_ids", "[]")
    values = json.loads(raw) if isinstance(raw, str) else raw
    return [str(value) for value in values] if isinstance(values, list) else []


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
    candidate_ids = [
        row.get("id") if isinstance(row, dict) else None
        for row in proposal["candidates"]
    ]
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in candidate_ids):
        raise ValueError("hypothesis proposal contains a candidate without a stable ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("hypothesis proposal contains duplicate candidate IDs")
    return proposal


def propose_candidates(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int = 50,
) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise ValueError("candidate limit must be 1..200")
    node_by_id = {node["id"]: node for node in nodes}
    evidence_by_id = {item["id"]: item for item in evidence}
    adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        ids = _edge_evidence_ids(edge)
        if edge.get("source") in node_by_id and edge.get("target") in node_by_id and ids and all(item in evidence_by_id for item in ids):
            adjacency[edge["source"]].append({**edge, "_evidence_ids": ids})

    candidates: dict[str, dict[str, Any]] = {}
    for source_id, first_edges in adjacency.items():
        for first in first_edges:
            middle_id = first["target"]
            for second in adjacency.get(middle_id, []):
                target_id = second["target"]
                if target_id in {source_id, middle_id}:
                    continue
                linked_ids = list(dict.fromkeys(first["_evidence_ids"] + second["_evidence_ids"]))
                records = [evidence_by_id[item] for item in linked_ids]
                first_records = [evidence_by_id[item] for item in first["_evidence_ids"]]
                second_records = [evidence_by_id[item] for item in second["_evidence_ids"]]
                if not any(record.get("polarity") == "supports" for record in first_records) or not any(
                    record.get("polarity") == "supports" for record in second_records
                ):
                    continue
                support = [record for record in records if record.get("polarity") == "supports"]
                contradiction = [record for record in records if record.get("polarity") == "contradicts"]
                context = [record for record in records if record.get("polarity") == "context"]
                if not support:
                    continue
                source, middle, target = (node_by_id[item] for item in (source_id, middle_id, target_id))
                canonical = "|".join((source_id, first["relation"], middle_id, second["relation"], target_id))
                identifier = "HYP-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]
                confidence_annotations = [float(first.get("confidence", 0)), float(second.get("confidence", 0))]
                all_quality = [float(record.get("quality", 0)) for record in records]
                path_evidence = []
                for edge in (first, second):
                    edge_records = [evidence_by_id[item] for item in edge["_evidence_ids"]]
                    path_evidence.append({
                        "source": edge["source"],
                        "relation": edge["relation"],
                        "target": edge["target"],
                        "evidence_ids": edge["_evidence_ids"],
                        "polarity_counts": {polarity: sum(record.get("polarity") == polarity for record in edge_records) for polarity in ("supports", "contradicts", "context")},
                        "interpretation": "Source-assigned polarity counts for this edge only; they are not independently validated or calibrated.",
                    })
                candidates[identifier] = {
                    "id": identifier,
                    "classification": "CANDIDATE RESEARCH HYPOTHESIS",
                    "statement": f"{source['label']} may be connected to {target['label']} through {middle['label']}.",
                    "path": [
                        {"source": source_id, "relation": first["relation"], "target": middle_id},
                        {"source": middle_id, "relation": second["relation"], "target": target_id},
                    ],
                    "path_evidence": path_evidence,
                    "dimensions": {
                        "supporting_records": len(support),
                        "contradictory_records": len(contradiction),
                        "context_records": len(context),
                        "mean_record_quality_annotation": round(sum(all_quality) / len(all_quality), 4) if all_quality else None,
                        "mean_edge_confidence_annotation": round(sum(confidence_annotations) / len(confidence_annotations), 4),
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
    ordered = sorted(candidates.values(), key=lambda row: (row["dimensions"]["contradictory_share_of_linked_records"], -row["dimensions"]["supporting_records"], row["id"]))
    return {
        "method": "two-edge directed graph path enumeration with session-retained evidence",
        "candidate_count": len(ordered),
        "returned_count": len(ordered[:limit]),
        "candidates": ordered[:limit],
        "truncated": len(ordered) > limit,
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
    if len(path) == 2:
        source_id, target_id = path[0].get("source"), path[1].get("target")
        primary_middle = path[0].get("target")
        for first in edges:
            if first.get("source") != source_id or first.get("target") == primary_middle:
                continue
            for second in edges:
                if second.get("source") == first.get("target") and second.get("target") == target_id:
                    first_ids = _edge_evidence_ids(first)
                    second_ids = _edge_evidence_ids(second)
                    linked_ids = list(dict.fromkeys(first_ids + second_ids))
                    if not first_ids or not second_ids or any(item not in evidence_by_id for item in linked_ids):
                        continue
                    records = [evidence_by_id[item] for item in linked_ids]
                    alternatives.append({
                        "intermediate": first.get("target"),
                        "path": [
                            {"source": first.get("source"), "relation": first.get("relation"), "target": first.get("target"), "evidence_ids": first_ids},
                            {"source": second.get("source"), "relation": second.get("relation"), "target": second.get("target"), "evidence_ids": second_ids},
                        ],
                        "evidence_ids": linked_ids,
                        "evidence_polarity_counts": {polarity: sum(row.get("polarity") == polarity for row in records) for polarity in ("supports", "contradicts", "context")},
                        "interpretation": "Alternative evidence-linked graph route; source polarity labels are reported descriptively and do not independently refute the candidate.",
                    })
    alternatives.sort(key=lambda row: (str(row["intermediate"]), tuple(row["evidence_ids"])))
    return {
        "hypothesis_id": candidate.get("id"),
        "classification": "SCIENTIFIC CRITIC REVIEW",
        "linked_contradictory_evidence": [{"evidence_id": item["id"], "claim": item.get("claim"), "quality_annotation": item.get("quality")} for item in contrary],
        "lexical_refutation_leads": refutation_leads[:lead_limit],
        "alternative_graph_routes": alternatives[:lead_limit],
        "major_weaknesses": [
            "Lexical overlap is only a search lead and requires scientific relevance review.",
            "Missing contradictory records cannot be interpreted as evidence of absence.",
            "Edge confidence and evidence quality are manually supplied annotations and are not calibrated.",
            "Population, model system, methods, and study limitations have not been reconciled.",
        ],
        "review_status": "requires_human_scientific_review",
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
