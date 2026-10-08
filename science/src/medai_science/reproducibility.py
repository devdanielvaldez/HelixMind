"""Build hash-linked, text-minimized research snapshots and readable audit reports."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import urllib.parse
from collections import Counter
from typing import Any
from pathlib import Path

from medai_science.hypotheses import validate_hypothesis_proposal_artifact
from medai_science.sources.safety import ensure_public_research_query

MAX_BUNDLE_JSON_BYTES = 16 * 1024 * 1024
MAX_REPORT_GRAPH_ASSERTIONS = 25


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"bundle JSON contains duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"bundle JSON contains non-standard numeric constant {value}")


def _read_bundle_json(path: Path, label: str) -> Any:
    """Read bounded, unambiguous JSON from a private replay bundle."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size > MAX_BUNDLE_JSON_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_BUNDLE_JSON_BYTES}-byte limit")
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_BUNDLE_JSON_BYTES + 1)
        if len(raw) > MAX_BUNDLE_JSON_BYTES:
            raise ValueError(f"{label} exceeds the {MAX_BUNDLE_JSON_BYTES}-byte limit")
        return json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} is invalid JSON") from error


def _canonical_hash(value: Any) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("reproducibility values must be finite, canonical JSON data") from error
    return hashlib.sha256(raw).hexdigest()


def _report_label(value: Any, maximum: int = 160) -> str:
    """Render a bounded, privacy-screened graph value without Markdown controls."""
    if not isinstance(value, str) or not value.strip():
        return "[unlabeled]"
    normalized = " ".join(value.split())[:maximum]
    try:
        ensure_public_research_query(normalized)
    except ValueError:
        return "[omitted by privacy screen]"
    return normalized.translate(str.maketrans({"`": "'", "[": "(", "]": ")", "<": "(", ">": ")", "*": "", "_": "", "#": "", "~": "", "|": ""}))


def _report_code(value: Any, maximum: int = 128) -> str:
    """Preserve bounded trace identifiers that are safe inside Markdown code spans."""
    if isinstance(value, str) and 1 <= len(value) <= maximum and re.fullmatch(r"[A-Za-z0-9._:/()+_\-]+", value):
        return value
    return "identifier omitted"


def _verified_artifact_payload(item: dict[str, Any], session_id: str) -> dict[str, Any]:
    payload_text, digest = item.get("payload"), item.get("sha256")
    if (item.get("session_id") != session_id or not isinstance(payload_text, str)
            or len(payload_text.encode("utf-8")) > 4 * 1024 * 1024
            or not isinstance(digest, str)
            or hashlib.sha256(payload_text.encode("utf-8")).hexdigest() != digest):
        raise ValueError("research report source artifact is outside the session or fails payload integrity checks")
    try:
        payload = json.loads(payload_text, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
    except (ValueError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("research report source artifact payload is not strict JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("research report source artifact payload must be an object")
    return payload


def _evidence_reference(record: dict[str, Any]) -> str:
    identifier = _report_code(record.get("id"), 96)
    source = _report_label(record.get("source"), 64)
    source_id = _report_code(record.get("source_id") or "", 96)
    polarity = _report_label(record.get("polarity"), 32)
    return f"`{identifier}` ({source}{': ' + source_id if source_id != 'identifier omitted' else ''}; {polarity})"


def _hypothesis_report_sections(artifacts: list[dict[str, Any]], session_id: str, evidence_records: list[dict[str, Any]]) -> list[str]:
    evidence_by_id = {item["id"]: item for item in evidence_records if isinstance(item.get("id"), str)}
    evidence_ids = set(evidence_by_id)
    proposals: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    invalid_proposals = 0
    for artifact in artifacts:
        if artifact.get("kind") != "hypothesis_candidates":
            continue
        try:
            proposal = validate_hypothesis_proposal_artifact(artifact, session_id)
            if any(not isinstance(candidate.get("evidence_ids"), list) or set(candidate["evidence_ids"]) - evidence_ids for candidate in proposal["candidates"]):
                raise ValueError("candidate cites evidence absent from the session snapshot")
            proposals[artifact["id"]] = (artifact, proposal)
        except (ValueError, TypeError):
            invalid_proposals += 1

    criticism_by_proposal: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    invalid_criticisms = 0
    for artifact in artifacts:
        if artifact.get("kind") != "hypothesis_criticism":
            continue
        try:
            payload = _verified_artifact_payload(artifact, session_id)
        except (ValueError, TypeError):
            invalid_criticisms += 1
            continue
        proposal_id = payload.get("proposal_artifact_id")
        proposal_entry = proposals.get(proposal_id) if isinstance(proposal_id, str) else None
        if proposal_entry is None:
            invalid_criticisms += 1
            continue
        proposal_artifact, proposal = proposal_entry
        if payload.get("proposal_artifact_sha256") != proposal_artifact.get("sha256"):
            invalid_criticisms += 1
            continue
        candidate_by_id = {row["id"]: row for row in proposal["candidates"]}
        reviews = payload.get("reviews")
        if not isinstance(reviews, list) or len(reviews) > 200:
            continue
        review_by_id = {}
        reviews_valid = True
        for review in reviews:
            if (not isinstance(review, dict) or review.get("hypothesis_id") not in candidate_by_id
                    or review["hypothesis_id"] in review_by_id
                    or review.get("review_status") != "requires_human_scientific_review"):
                reviews_valid = False
                break
            for field in ("linked_contradictory_evidence", "lexical_refutation_leads"):
                values = review.get(field, [])
                if not isinstance(values, list) or any(
                    not isinstance(row, dict) or not isinstance(row.get("evidence_id"), str)
                    or row["evidence_id"] not in evidence_ids for row in values
                ):
                    reviews_valid = False
                    break
            if not reviews_valid:
                break
            routes = review.get("alternative_graph_routes", [])
            if not isinstance(routes, list) or any(
                not isinstance(route, dict) or not isinstance(route.get("evidence_ids"), list)
                or any(not isinstance(item, str) for item in route["evidence_ids"])
                or set(route["evidence_ids"]) - evidence_ids for route in routes
            ):
                reviews_valid = False
                break
            retrieved = review.get("retrieved_contradiction_candidates", [])
            if not isinstance(retrieved, list) or any(
                not isinstance(row, dict) or not isinstance(row.get("target_evidence_id"), str)
                or row["target_evidence_id"] not in evidence_ids
                for row in retrieved
            ):
                reviews_valid = False
                break
            review_by_id[review["hypothesis_id"]] = review
        criticism_valid = reviews_valid and len(review_by_id) == len(reviews)
        if not criticism_valid:
            review_by_id = {}
            invalid_criticisms += 1
        criticism_by_proposal[proposal_id] = (artifact, {"reviews": review_by_id, "valid": criticism_valid})

    lines = ["", "## Candidate hypotheses and criticism"]
    if invalid_proposals:
        lines.append(f"- {invalid_proposals} proposal artifact(s) failed integrity or evidence-link checks and were omitted; review is required.")
    if invalid_criticisms:
        lines.append(f"- {invalid_criticisms} criticism artifact(s) failed integrity or evidence-link checks and were omitted; review is required.")
    if not proposals:
        lines.append("- No retained candidate-hypothesis proposal is available in this session snapshot.")
        return lines
    entries = sorted(proposals.items(), key=lambda item: (item[1][0].get("recorded_at", 0) if isinstance(item[1][0].get("recorded_at", 0), int) else 0, item[0]))
    proposal_id, (proposal_artifact, proposal) = entries[-1]
    candidates = proposal["candidates"]
    lines.append(f"- Proposal `{proposal_id}` retained {len(candidates)} candidate(s); the proposal artifact is `{proposal_artifact['sha256']}`.")
    if proposal.get("truncated"):
        lines.append("- The candidate proposal was truncated by its configured limit.")
    criticism_entry = criticism_by_proposal.get(proposal_id)
    reviews = criticism_entry[1]["reviews"] if criticism_entry and criticism_entry[1]["valid"] else {}
    if criticism_entry and not criticism_entry[1]["valid"]:
        lines.append("- A linked criticism artifact failed consistency checks and was excluded; human review is required.")
    for candidate in candidates[:20]:
        candidate_id = candidate["id"]
        candidate_label = _report_label(candidate_id, 128)
        dimensions = candidate.get("dimensions", {})
        if not isinstance(dimensions, dict):
            dimensions = {}
        statement = _report_label(candidate.get("statement"), 320)
        novelty_text = "unscored"
        def count_text(value: Any) -> str:
            return str(value) if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000 else "unavailable"
        candidate_evidence = candidate.get("evidence_ids", [])
        evidence_refs = [_evidence_reference(evidence_by_id[item]) for item in candidate_evidence[:5]]
        if len(candidate_evidence) > len(evidence_refs):
            evidence_refs.append(f"+{len(candidate_evidence) - len(evidence_refs)} more")
        line = (
            f"- **Candidate `{candidate_label}`:** {statement} Supporting records: "
            f"{count_text(dimensions.get('supporting_records'))}; contradictory records: "
            f"{count_text(dimensions.get('contradictory_records'))}; context records: "
            f"{count_text(dimensions.get('context_records'))}; novelty: {novelty_text}. Evidence: "
            f"{', '.join(evidence_refs) if evidence_refs else 'no verified evidence IDs'}."
        )
        review = reviews.get(candidate_id)
        if review is None:
            line += " Criticism: no integrity-verified review is linked."
        else:
            critic_evidence_ids = sorted({
                row["evidence_id"]
                for field in ("linked_contradictory_evidence", "lexical_refutation_leads")
                for row in review.get(field, [])
            })
            critic_refs = [_evidence_reference(evidence_by_id[item]) for item in critic_evidence_ids[:5]]
            if len(critic_evidence_ids) > len(critic_refs):
                critic_refs.append(f"+{len(critic_evidence_ids) - len(critic_refs)} more")
            line += (
                f" Criticism review `{review.get('review_status')}`: "
                f"{len(review.get('linked_contradictory_evidence', []))} linked contradictory record(s), "
                f"{len(review.get('lexical_refutation_leads', []))} lexical lead(s), "
                f"{len(review.get('alternative_graph_routes', []))} alternative route(s), and "
                f"{len(review.get('retrieved_contradiction_candidates', []))} retrieved candidate(s). "
                f"Critic evidence: {', '.join(critic_refs) if critic_refs else 'none linked'}."
            )
        lines.append(line)
    if len(candidates) > 20:
        lines.append(f"- {len(candidates) - 20} additional candidate(s) omitted from this bounded report section.")
    lines.append("- These are candidate research hypotheses and review leads, not confirmed findings; evidence dimensions are descriptive and uncalibrated.")
    return lines


def _literature_report_section(artifacts: list[dict[str, Any]], session_id: str) -> list[str]:
    by_pmid: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    invalid_records = 0
    duplicate_pmids = 0
    for artifact in artifacts:
        if artifact.get("kind") != "paper" or artifact.get("source") != "PubMed":
            continue
        try:
            paper = _verified_artifact_payload(artifact, session_id)
        except (ValueError, TypeError):
            invalid_records += 1
            continue
        provenance = paper.get("provenance")
        pmid = provenance.get("source_id") if isinstance(provenance, dict) and provenance.get("source") == "PubMed" else None
        if not isinstance(pmid, str) or not re.fullmatch(r"[1-9][0-9]{0,11}", pmid) or artifact.get("source_id") != pmid:
            invalid_records += 1
            continue
        if pmid in by_pmid:
            if artifact["sha256"] != by_pmid[pmid][0]["sha256"]:
                duplicate_pmids += 1
            continue
        by_pmid[pmid] = (artifact, paper)

    lines = ["", "## Captured PubMed bibliography"]
    if invalid_records:
        lines.append(f"- {invalid_records} paper artifact(s) failed source identity or payload integrity checks and were omitted.")
    if duplicate_pmids:
        lines.append(f"- {duplicate_pmids} PMID(s) had conflicting retained payloads; the first artifact was kept and human review is required.")
    if not by_pmid:
        lines.append("- No integrity-verified PubMed bibliography records were captured.")
    else:
        records = sorted(by_pmid.items(), key=lambda item: (int(item[0]), item[1][0]["id"]))
        lines.append(f"- {len(records)} unique PubMed record(s) were captured; this section is not a relevance ranking or study-quality assessment.")
        for pmid, (artifact, paper) in records[:30]:
            title = _report_label(paper.get("title"), 240)
            title_text = title if title != "[unlabeled]" else "[title unavailable]"
            date = paper.get("publication_date")
            year = date[:4] if isinstance(date, str) and re.match(r"^(?:18|19|20|21)[0-9]{2}(?:[-/]|$)", date) else None
            identifiers = paper.get("identifiers")
            doi = identifiers.get("doi") if isinstance(identifiers, dict) else None
            doi_text = ""
            if isinstance(doi, str) and re.fullmatch(r"10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+", doi, flags=re.IGNORECASE):
                safe_doi = urllib.parse.quote(doi, safe="/().;:-_")
                doi_text = f"; DOI: [{_report_label(doi, 255)}](https://doi.org/{safe_doi})"
            year_text = f" ({year})" if year else ""
            lines.append(f"- [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/){year_text}: {title_text}{doi_text}; artifact `{artifact['id']}` (`{artifact['sha256']}`).")
        if len(records) > 30:
            lines.append(f"- {len(records) - 30} additional PubMed record(s) omitted from this bounded report section.")
    lines.append("- Abstract text is omitted; titles and identifiers are bibliography metadata only, and do not establish evidence quality or support for a claim.")
    return lines


def _require_unique_ids(rows: list[Any], label: str) -> set[str]:
    identifiers: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"reproducibility {label} contains a malformed record")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"reproducibility {label} requires a stable identifier")
        if identifier in identifiers:
            raise ValueError(f"reproducibility {label} contains duplicate identity {identifier}")
        identifiers.add(identifier)
    return identifiers


def build_reproducibility_bundle(session: dict[str, Any], plan: dict[str, Any], artifacts: list[dict[str, Any]], graph: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(session, dict) or not isinstance(plan, dict) or not isinstance(graph, dict):
        raise ValueError("session, plan, and graph must be objects")
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("research session must have an identifier")
    tasks = plan.get("tasks", [])
    if not isinstance(tasks, list) or not isinstance(artifacts, list) or not isinstance(evidence, list):
        raise ValueError("plan, artifacts, and evidence must be lists or objects")
    _require_unique_ids(tasks, "task list")
    if plan.get("session_id") not in (None, session_id):
        raise ValueError("reproducibility plan belongs to another research session")
    artifact_receipts = []
    _require_unique_ids(artifacts, "artifact list")
    for item in artifacts:
        if item.get("session_id") != session_id:
            raise ValueError("reproducibility inputs cannot include cross-session artifacts")
        if not isinstance(item.get("id"), str) or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in item["sha256"]):
            raise ValueError("every artifact receipt requires a stable id and SHA-256 hash")
        artifact_receipts.append({key: item.get(key) for key in ("id", "task_id", "kind", "source", "source_id", "sha256", "recorded_at")})
    graph_nodes = graph.get("nodes", [])
    graph_edges = graph.get("edges", [])
    if graph.get("session_id") not in (None, session_id):
        raise ValueError("graph snapshot belongs to another research session")
    if not isinstance(graph_nodes, list) or not isinstance(graph_edges, list):
        raise ValueError("graph snapshot must contain node and edge arrays")
    _require_unique_ids(graph_nodes, "graph nodes")
    _require_unique_ids(evidence, "evidence list")
    node_ids = {node.get("id") for node in graph_nodes if isinstance(node, dict)}
    evidence_ids = {item.get("id") for item in evidence if isinstance(item, dict)}
    artifact_ids = {item.get("id") for item in artifact_receipts if isinstance(item, dict)}
    graph_receipt = {
        "nodes": [{key: node.get(key) for key in ("id", "kind", "label")} for node in graph_nodes],
        "edges": [],
    }
    edge_ids: set[tuple[str, str, str]] = set()
    for edge in graph_edges:
        if not isinstance(edge, dict):
            raise ValueError("graph snapshot contains a malformed edge")
        edge_identity = (edge.get("source"), edge.get("relation"), edge.get("target"))
        if any(not isinstance(value, str) or not value for value in edge_identity):
            raise ValueError("graph edge snapshot has no stable source, relation, and target")
        if edge_identity in edge_ids:
            raise ValueError("graph snapshot contains duplicate edge identity")
        edge_ids.add(edge_identity)
        try:
            linked = json.loads(edge.get("evidence_ids", "[]")) if isinstance(edge.get("evidence_ids"), str) else edge.get("evidence_ids", [])
            linked_artifacts = json.loads(edge.get("artifact_ids", "[]")) if isinstance(edge.get("artifact_ids"), str) else edge.get("artifact_ids", [])
        except json.JSONDecodeError as error:
            raise ValueError("graph edge contains malformed evidence or artifact identifiers") from error
        if (edge.get("source") not in node_ids or edge.get("target") not in node_ids
                or not isinstance(linked, list) or any(not isinstance(item, str) or item not in evidence_ids for item in linked)
                or len(linked) != len(set(linked))
                or not isinstance(linked_artifacts, list) or any(not isinstance(item, str) or item not in artifact_ids for item in linked_artifacts)
                or len(linked_artifacts) != len(set(linked_artifacts))):
            raise ValueError("graph snapshot contains a dangling node, evidence, or artifact reference")
        graph_receipt["edges"].append({"source": edge.get("source"), "relation": edge.get("relation"), "target": edge.get("target"), "evidence_ids": linked, "artifact_ids": linked_artifacts})
    evidence_receipts = [{key: item.get(key) for key in ("id", "source", "source_id", "polarity", "quality")} for item in evidence]
    if any(item.get("session_id") != session_id for item in evidence):
        raise ValueError("reproducibility inputs cannot include cross-session evidence")
    task_receipts = [{"id": task.get("id"), "status": task.get("status"), "attempts": task.get("attempts"), "output_sha256": _canonical_hash(task.get("output")) if task.get("output") is not None else None, "error_sha256": _canonical_hash(task.get("error")) if task.get("error") is not None else None} for task in tasks if isinstance(task, dict)]
    nodes_by_id = {node["id"]: node for node in graph_receipt["nodes"]}
    evidence_by_id = {item["id"]: item for item in evidence_receipts}
    report_assertions = []
    for edge in sorted(graph_receipt["edges"], key=lambda row: (row["source"], row["relation"], row["target"])):
        if not edge["evidence_ids"]:
            continue
        source_node, target_node = nodes_by_id[edge["source"]], nodes_by_id[edge["target"]]
        source_label = _report_label(source_node.get("label"))
        target_label = _report_label(target_node.get("label"))
        source_kind = _report_label(source_node.get("kind"), 64)
        target_kind = _report_label(target_node.get("kind"), 64)
        relation = _report_code(edge["relation"], 96)
        refs = []
        for evidence_id in edge["evidence_ids"][:5]:
            record = evidence_by_id[evidence_id]
            source = _report_label(record.get("source"), 64)
            source_id = _report_code(record.get("source_id") or "", 96)
            polarity = _report_label(record.get("polarity"), 32)
            ref = f"{_report_code(record['id'], 96)} ({source}{': ' + source_id if source_id != 'identifier omitted' else ''}; {polarity})"
            refs.append(f"`{ref}`")
        remaining_refs = len(edge["evidence_ids"]) - len(refs)
        refs_text = ", ".join(refs) + (f", +{remaining_refs} more" if remaining_refs else "")
        report_assertions.append(
            f"- {source_label} ({source_kind}) — `{relation}` → {target_label} ({target_kind}); recorded evidence: {refs_text}."
        )
        if len(report_assertions) == MAX_REPORT_GRAPH_ASSERTIONS:
            break
    snapshot = {
        "schema_version": "1",
        "session": {key: session.get(key) for key in ("id", "objective", "status", "created_at", "depth", "budget", "credits_used")},
        "tasks": task_receipts,
        "artifacts": artifact_receipts,
        "evidence_annotations": evidence_receipts,
        "graph": graph_receipt,
        "limitations": [
            "This bundle contains provenance and structural snapshots, not complete source datasets or article text.",
            "Evidence polarity and quality are retained annotations and are not independently calibrated.",
            "The bundle hash verifies the captured bundle bytes; it does not establish scientific correctness.",
        ],
    }
    snapshot["bundle_sha256"] = _canonical_hash(snapshot)
    report_lines = [
        "# Biomedical computational research report",
        "",
        f"Session: `{session_id}`",
        f"Objective: {session.get('objective', '')}",
        f"Status: **{session.get('status', 'unknown')}** | Depth: {session.get('depth', 'unknown')} | Credits: {session.get('credits_used', 0)}/{session.get('budget', 0)}",
        "",
        "## Research record",
        f"- Tasks recorded: {len(task_receipts)}",
        f"- Source artifacts: {len(artifact_receipts)}",
        f"- Evidence annotations: {len(evidence_receipts)}",
        f"- Graph nodes and edges: {len(graph_receipt['nodes'])} / {len(graph_receipt['edges'])}",
        f"- Reproducibility bundle SHA-256: `{snapshot['bundle_sha256']}`",
        "",
        "## Task outcomes",
    ]
    report_lines.extend(f"- `{item['id']}`: {item['status']} (attempts: {item['attempts']})" for item in task_receipts)
    report_lines.extend([
        "",
        "## Source-linked graph statements",
    ])
    if report_assertions:
        report_lines.extend(report_assertions)
        if len(report_assertions) == MAX_REPORT_GRAPH_ASSERTIONS:
            report_lines.append(f"- Further source-linked graph statements may exist; display is limited to {MAX_REPORT_GRAPH_ASSERTIONS}.")
    else:
        report_lines.append("- No evidence-linked graph statements were retained in this session snapshot.")
    report_lines.extend(_literature_report_section(artifacts, session_id))
    report_lines.extend(_hypothesis_report_sections(artifacts, session_id, evidence))
    report_lines.extend([
        "",
        "These are recorded graph relationships with source-linked evidence annotations, not claims of causality or independently confirmed findings. Polarity and quality are reported as stored; quality annotations are not calibrated.",
        "",
        "## Interpretation limits",
        "- This computational research record is not a clinical recommendation.",
        "- Machine-generated annotations and hypotheses require human scientific review and independent evaluation.",
        "- Source records must be retrieved again under the providers' current access and reuse terms for full replay.",
    ])
    return {**snapshot, "report_markdown": "\n".join(report_lines)}


def compare_reproducibility_bundles(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Report hash-verified structural changes between two captures of one research session."""
    def validate(bundle: Any, label: str) -> dict[str, Any]:
        if not isinstance(bundle, dict):
            raise ValueError(f"{label} bundle must be an object")
        digest = bundle.get("bundle_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{label} bundle hash is missing or malformed")
        snapshot = {key: value for key, value in bundle.items() if key not in {"bundle_sha256", "report_markdown"}}
        if _canonical_hash(snapshot) != digest:
            raise ValueError(f"{label} bundle hash verification failed")
        session = bundle.get("session")
        if not isinstance(session, dict) or not isinstance(session.get("id"), str) or not session["id"]:
            raise ValueError(f"{label} bundle session identity is malformed")
        return bundle

    before = validate(baseline, "baseline")
    after = validate(current, "current")
    session_id = before["session"]["id"]
    if after["session"]["id"] != session_id:
        raise ValueError("change comparison requires bundles from the same research session")

    def keyed(rows: Any, key_for: Any, name: str) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError(f"{name} snapshot must be an array")
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{name} snapshot contains a malformed record")
            key = key_for(row)
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} snapshot record has no stable identity")
            if key in result:
                raise ValueError(f"{name} snapshot contains duplicate identity {key}")
            result[key] = row
        return result

    def diff(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        shared = left.keys() & right.keys()
        return {
            "added": sorted(right.keys() - left.keys()),
            "removed": sorted(left.keys() - right.keys()),
            "modified": sorted(key for key in shared if _canonical_hash(left[key]) != _canonical_hash(right[key])),
        }

    def graph_edge_key(row: dict[str, Any]) -> str:
        fields = [row.get("source"), row.get("relation"), row.get("target")]
        if any(not isinstance(value, str) or not value for value in fields):
            raise ValueError("graph edge snapshot has no stable source, relation, and target")
        return json.dumps(fields, separators=(",", ":"))

    left_graph, right_graph = before.get("graph"), after.get("graph")
    if not isinstance(left_graph, dict) or not isinstance(right_graph, dict):
        raise ValueError("bundle graph snapshots must be objects")
    changes = {
        "session_metadata": {"modified": sorted(key for key in ("objective", "status", "created_at", "depth", "budget", "credits_used") if before["session"].get(key) != after["session"].get(key))},
        "tasks": diff(keyed(before.get("tasks"), lambda row: row.get("id"), "task"), keyed(after.get("tasks"), lambda row: row.get("id"), "task")),
        "artifacts": diff(keyed(before.get("artifacts"), lambda row: row.get("id"), "artifact"), keyed(after.get("artifacts"), lambda row: row.get("id"), "artifact")),
        "evidence_annotations": diff(keyed(before.get("evidence_annotations"), lambda row: row.get("id"), "evidence"), keyed(after.get("evidence_annotations"), lambda row: row.get("id"), "evidence")),
        "graph_nodes": diff(keyed(left_graph.get("nodes"), lambda row: row.get("id"), "graph node"), keyed(right_graph.get("nodes"), lambda row: row.get("id"), "graph node")),
        "graph_edges": diff(
            keyed(left_graph.get("edges"), graph_edge_key, "graph edge"),
            keyed(right_graph.get("edges"), graph_edge_key, "graph edge"),
        ),
    }
    has_changes = any(any(values for values in category.values()) for category in changes.values())
    return {
        "schema_version": "1",
        "session_id": session_id,
        "baseline_bundle_sha256": before["bundle_sha256"],
        "current_bundle_sha256": after["bundle_sha256"],
        "review_required": has_changes,
        "changes": changes,
        "interpretation_limit": "This comparison detects structural changes in retained research records. The unkeyed bundle hashes verify internal consistency, not producer identity. The comparison does not establish why evidence changed, whether a result is scientifically correct, or whether a clinical conclusion should change.",
    }


def parse_research_bundle_artifact(artifact: Any, expected_session_id: str) -> dict[str, Any]:
    """Verify and decode a session-scoped API artifact containing a research bundle."""
    if not isinstance(artifact, dict):
        raise ValueError("research bundle API response must be an object")
    if artifact.get("kind") != "research_reproducibility_bundle":
        raise ValueError("artifact is not a research reproducibility bundle")
    if artifact.get("session_id") != expected_session_id:
        raise ValueError("research bundle artifact belongs to a different session")
    payload, digest = artifact.get("payload"), artifact.get("sha256")
    if not isinstance(payload, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("research bundle artifact payload or digest is malformed")
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
        raise ValueError("research bundle artifact payload hash verification failed")
    try:
        bundle = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("research bundle artifact payload is not valid JSON") from error
    if not isinstance(bundle, dict) or not isinstance(bundle.get("session"), dict) or bundle["session"].get("id") != expected_session_id:
        raise ValueError("research bundle payload belongs to a different session")
    return bundle


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_bundle_file(bundle: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("bundle contains an invalid artifact path")
    candidate = bundle / relative
    current = bundle
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("bundle artifact paths must not traverse symbolic links")
    try:
        candidate.resolve(strict=True).relative_to(bundle.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ValueError("bundle artifact path is missing or escapes its directory") from error
    if not candidate.is_file():
        raise ValueError("bundle artifact path must identify a regular file")
    return candidate


def _verify_python_runtime(manifest: dict[str, Any]) -> None:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("python") != os.sys.version.split()[0]:
        raise ValueError("installed Python runtime differs from the captured bundle runtime")


def _write_private(path: Path, content: bytes) -> None:
    if path.suffix.casefold() == ".json":
        if len(content) > MAX_BUNDLE_JSON_BYTES:
            raise ValueError(f"bundle JSON exceeds the {MAX_BUNDLE_JSON_BYTES}-byte limit")
        try:
            json.loads(content, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("bundle writer received invalid or ambiguous JSON") from error
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(content)
    path.chmod(0o600)


def _validate_vep_cross_validation_receipt(report: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Bind a retained Ensembl VEP receipt to the exact reference-validated local report."""
    from medai_science.sources.ensembl_vep import EnsemblVEPClient

    if not isinstance(receipt, dict) or receipt.get("source") != "Ensembl Variant Effect Predictor REST API":
        raise ValueError("external cross-validation must be a retained Ensembl VEP result")
    build = report.get("genome_build")
    expected_url = EnsemblVEPClient.endpoints.get(build)
    if expected_url is None or receipt.get("assembly") != build or receipt.get("source_url") not in {expected_url + "/vep/homo_sapiens/region"}:
        raise ValueError("Ensembl VEP source URL or assembly does not match the local genome build")
    report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if receipt.get("input_report_sha256") != hashlib.sha256(report_bytes).hexdigest() or receipt.get("reference_fasta_sha256") != report.get("reference_fasta_sha256"):
        raise ValueError("Ensembl VEP receipt does not bind the exact local report and reference")
    rows = receipt.get("results")
    queryable = EnsemblVEPClient.queryable_count(report)
    skipped = receipt.get("skipped_unsupported_or_unvalidated")
    if not isinstance(rows, list) or len(rows) != queryable or receipt.get("queried_variant_alleles") != queryable:
        raise ValueError("Ensembl VEP result count does not match reference-validated local alleles")
    if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped != len(report["variants"]) - queryable:
        raise ValueError("Ensembl VEP skipped-allele count does not match the local report")
    expected_inputs: Counter[str] = Counter()
    local_variants_by_input: dict[str, list[dict[str, Any]]] = {}
    for variant in report["variants"]:
        if not isinstance(variant, dict) or variant.get("sequence_allele_supported") is not True or variant.get("reference_validated") is not True:
            continue
        contig = variant.get("contig")
        position = variant.get("normalized_position_1_based", variant.get("position_1_based"))
        ref = variant.get("normalized_reference_allele", variant.get("reference_allele"))
        alt = variant.get("normalized_alternate_allele", variant.get("alternate_allele"))
        if not isinstance(contig, str) or isinstance(position, bool) or not isinstance(position, int) or not all(isinstance(allele, str) for allele in (ref, alt)):
            raise ValueError("local VEP-queryable allele identity is malformed")
        input_value = f"{contig.removeprefix('chr')} {position} . {ref} {alt} . . ."
        expected_inputs[input_value] += 1
        local_variants_by_input.setdefault(input_value, []).append(variant)
    observed_inputs: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Ensembl VEP receipt contains a malformed result")
        input_value = row.get("input")
        source_hash = row.get("source_record_sha256")
        if not isinstance(input_value, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+ [1-9][0-9]* \. [ACGTN]+ [ACGTN]+ \. \. \." , input_value):
            raise ValueError("Ensembl VEP receipt contains a malformed input allele")
        if input_value not in expected_inputs or observed_inputs[input_value] >= expected_inputs[input_value]:
            raise ValueError("Ensembl VEP receipt contains an unrequested or excess duplicate allele")
        local_variant = local_variants_by_input[input_value][observed_inputs[input_value]]
        expected_local_identity = {
            "local_record_number": local_variant.get("record_number"),
            "local_contig": local_variant.get("contig"),
            "local_position_1_based": local_variant.get("normalized_position_1_based", local_variant.get("position_1_based")),
            "local_reference_allele": local_variant.get("normalized_reference_allele", local_variant.get("reference_allele")),
            "local_alternate_allele": local_variant.get("normalized_alternate_allele", local_variant.get("alternate_allele")),
            "seq_region_name": input_value.split(" ", 1)[0],
        }
        if any(row.get(key) != value for key, value in expected_local_identity.items()):
            raise ValueError("Ensembl VEP result local allele metadata does not match the bound report allele")
        if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            raise ValueError("Ensembl VEP result is missing its source-record digest")
        if not isinstance(row.get("transcript_consequences"), list) or row.get("vep_assembly") != build:
            raise ValueError("Ensembl VEP result assembly or transcript consequences are malformed")
        observed_inputs[input_value] += 1
    if observed_inputs != expected_inputs:
        raise ValueError("Ensembl VEP receipt does not cover every queryable local allele")


def create_genomics_bundle(
    bundle_dir: str | Path,
    report: dict[str, Any],
    vcf_path: str | Path,
    genome_build: str,
    max_variants: int,
    reference_fasta: str | Path | None = None,
    transcript_gff3: str | Path | None = None,
    external_vep: dict[str, Any] | None = None,
    clinvar_records: list[dict[str, Any]] | None = None,
    gnomad_frequency_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture bounded local VCF analysis inputs and exact module hashes for private replay."""
    from medai_science.genomics import analyze_vcf

    target = Path(bundle_dir).expanduser().resolve()
    vcf = Path(vcf_path).expanduser().resolve()
    reference = Path(reference_fasta).expanduser().resolve() if reference_fasta else None
    annotation = Path(transcript_gff3).expanduser().resolve() if transcript_gff3 else None
    if target.exists():
        raise ValueError("genomics bundle directory already exists")
    if not vcf.is_file() or (reference is not None and not reference.is_file()) or (annotation is not None and not annotation.is_file()):
        raise ValueError("all genomic inputs must be regular files")
    if annotation is not None and reference is None:
        raise ValueError("transcript annotation requires a local reference FASTA")
    if not isinstance(report, dict) or report.get("genome_build") != genome_build:
        raise ValueError("analysis report genome build does not match requested bundle parameters")
    if external_vep is not None:
        _validate_vep_cross_validation_receipt(report, external_vep)
    clinvar_reconciliation = None
    source_reconciliation = None
    gnomad_frequency_reconciliation = None
    if gnomad_frequency_receipts is not None:
        if reference is None:
            raise ValueError("gnomAD frequency bundles require the exact local reference FASTA")
        from medai_science.variant_source_reconciliation import reconcile_gnomad_frequency
        gnomad_frequency_reconciliation = reconcile_gnomad_frequency(report, gnomad_frequency_receipts)
    if clinvar_records is not None:
        if reference is None:
            raise ValueError("ClinVar reconciliation bundles require the exact local reference FASTA")
        from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
        clinvar_reconciliation = reconcile_clinvar_variants(report, clinvar_records, reference, genome_build)
        if external_vep is not None:
            from medai_science.variant_source_reconciliation import reconcile_variant_sources
            source_reconciliation = reconcile_variant_sources(report, clinvar_reconciliation, external_vep)
    target.mkdir(parents=True, mode=0o700)
    try:
        input_rows = [("inputs/cohort.vcf", vcf)]
        if reference is not None:
            input_rows.append(("inputs/reference.fa", reference))
        if annotation is not None:
            input_rows.append(("inputs/annotation.gff3", annotation))
        artifacts = []
        for relative, source in input_rows:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
            artifacts.append({"path": relative, "size": destination.stat().st_size, "sha256": _sha256(destination)})
        report_path = target / "report.json"
        report_bytes = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _write_private(report_path, report_bytes)
        artifacts.append({"path": "report.json", "size": report_path.stat().st_size, "sha256": _sha256(report_path)})
        external_validation_path = None
        if external_vep is not None:
            external_validation_path = "external/ensembl-vep.json"
            validation_path = target / external_validation_path
            vep_bytes = (json.dumps(external_vep, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if len(vep_bytes) > 8_000_000:
                raise ValueError("Ensembl VEP cross-validation receipt exceeds the 8 MB bundle limit")
            _write_private(validation_path, vep_bytes)
            artifacts.append({"path": external_validation_path, "size": validation_path.stat().st_size, "sha256": _sha256(validation_path)})
        external_clinvar_path = None
        external_clinvar_reconciliation_path = None
        external_source_reconciliation_path = None
        if clinvar_records is not None and clinvar_reconciliation is not None:
            external_clinvar_path = "external/clinvar-records.json"
            clinvar_records_path = target / external_clinvar_path
            records_bytes = (json.dumps(clinvar_records, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if len(records_bytes) > 8_000_000:
                raise ValueError("ClinVar source records exceed the 8 MB bundle limit")
            _write_private(clinvar_records_path, records_bytes)
            artifacts.append({"path": external_clinvar_path, "size": clinvar_records_path.stat().st_size, "sha256": _sha256(clinvar_records_path)})
            external_clinvar_reconciliation_path = "external/clinvar-reconciliation.json"
            clinvar_path = target / external_clinvar_reconciliation_path
            clinvar_bytes = (json.dumps(clinvar_reconciliation, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            _write_private(clinvar_path, clinvar_bytes)
            artifacts.append({"path": external_clinvar_reconciliation_path, "size": clinvar_path.stat().st_size, "sha256": _sha256(clinvar_path)})
            if source_reconciliation is not None:
                external_source_reconciliation_path = "external/source-reconciliation.json"
                source_path = target / external_source_reconciliation_path
                source_bytes = (json.dumps(source_reconciliation, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
                _write_private(source_path, source_bytes)
                artifacts.append({"path": external_source_reconciliation_path, "size": source_path.stat().st_size, "sha256": _sha256(source_path)})
        external_gnomad_receipts_path = None
        external_gnomad_reconciliation_path = None
        if gnomad_frequency_receipts is not None and gnomad_frequency_reconciliation is not None:
            external_gnomad_receipts_path = "external/gnomad-frequency-receipts.json"
            receipts_path = target / external_gnomad_receipts_path
            receipts_bytes = (json.dumps(gnomad_frequency_receipts, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            if len(receipts_bytes) > 8_000_000:
                raise ValueError("gnomAD frequency receipts exceed the 8 MB bundle limit")
            _write_private(receipts_path, receipts_bytes)
            artifacts.append({"path": external_gnomad_receipts_path, "size": receipts_path.stat().st_size, "sha256": _sha256(receipts_path)})
            external_gnomad_reconciliation_path = "external/gnomad-frequency-reconciliation.json"
            reconciliation_path = target / external_gnomad_reconciliation_path
            reconciliation_bytes = (json.dumps(gnomad_frequency_reconciliation, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            _write_private(reconciliation_path, reconciliation_bytes)
            artifacts.append({"path": external_gnomad_reconciliation_path, "size": reconciliation_path.stat().st_size, "sha256": _sha256(reconciliation_path)})
        source_root = Path(__file__).resolve().parent
        module_rows = []
        module_names = ["__init__.py", "genomics.py", "transcripts.py"]
        if external_vep is not None:
            module_names.extend(["reproducibility.py", "sources/ensembl_vep.py"])
        if clinvar_records is not None:
            module_names.extend(["reproducibility.py", "clinvar_reconciliation.py"])
            if external_vep is not None:
                module_names.append("variant_source_reconciliation.py")
        if gnomad_frequency_receipts is not None:
            module_names.extend(["reproducibility.py", "variant_source_reconciliation.py", "sources/gnomad.py", "sources/bounded_http.py"])
        module_names = list(dict.fromkeys(module_names))
        for module_name in module_names:
            source = source_root / module_name
            relative = "code/medai_science/" + module_name
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(source, destination)
            destination.chmod(0o600)
            digest = _sha256(destination)
            artifacts.append({"path": relative, "size": destination.stat().st_size, "sha256": digest})
            module_key = "medai_science" if module_name == "__init__.py" else "medai_science." + module_name.removesuffix(".py").replace("/", ".")
            module_rows.append({"module": module_key, "path": relative, "sha256": digest})
        manifest = {
            "schema_version": "1",
            "created_at_unix": int(__import__("time").time()),
            "analysis": {"workflow": "genomics.vcf", "genome_build": genome_build, "max_variants": max_variants, "reference_fasta": "inputs/reference.fa" if reference else None, "transcript_gff3": "inputs/annotation.gff3" if annotation else None, "external_vep": external_validation_path, "external_clinvar_records": external_clinvar_path, "external_clinvar_reconciliation": external_clinvar_reconciliation_path, "external_source_reconciliation": external_source_reconciliation_path, "external_gnomad_frequency_receipts": external_gnomad_receipts_path, "external_gnomad_frequency_reconciliation": external_gnomad_reconciliation_path},
            "network_accessed": False,
            "runtime": {"python": os.sys.version.split()[0]},
            "scientific_modules": module_rows,
            "artifacts": artifacts,
            "report_sha256": _sha256(report_path),
        }
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "external_vep_captured": external_vep is not None, "external_clinvar_captured": clinvar_records is not None, "cross_source_reconciliation_captured": source_reconciliation is not None, "external_gnomad_frequency_captured": gnomad_frequency_receipts is not None, "network_accessed": False}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def create_sequence_bundle(
    bundle_dir: str | Path,
    report: dict[str, Any],
    input_path: str | Path,
    sequence_format: str,
    max_records: int,
    molecule_type: str = "nucleotide",
) -> dict[str, Any]:
    """Capture FASTA/FASTQ input and exact parser code for private deterministic replay."""
    from medai_science.genomics import inspect_fasta, inspect_fastq

    target = Path(bundle_dir).expanduser().resolve()
    source = Path(input_path).expanduser().resolve()
    normalized_format = sequence_format.casefold()
    if normalized_format not in {"fasta", "fastq"}:
        raise ValueError("sequence bundle format must be fasta or fastq")
    if not source.is_file() or source.stat().st_size > 2 * 1024 * 1024 * 1024:
        raise ValueError("sequence input must be a regular file no larger than 2 GiB")
    if not isinstance(max_records, int) or isinstance(max_records, bool) or not 1 <= max_records <= 2_000_000:
        raise ValueError("max_records must be 1..2000000")
    if normalized_format == "fasta" and molecule_type not in {"nucleotide", "protein"}:
        raise ValueError("molecule_type must be nucleotide or protein")
    if target.exists():
        raise ValueError("sequence bundle directory already exists")
    expected_analysis = inspect_fasta if normalized_format == "fasta" else inspect_fastq
    expected_report = expected_analysis(source, molecule_type, max_records) if normalized_format == "fasta" else expected_analysis(source, max_records)
    if not isinstance(report, dict) or report != expected_report:
        raise ValueError("sequence analysis report does not match the supplied input and parser parameters")
    target.mkdir(parents=True, mode=0o700)
    try:
        with source.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        suffix = ".fasta" if normalized_format == "fasta" else ".fastq"
        relative_input = "inputs/sequences" + suffix + (".gz" if compressed else "")
        destination = target / relative_input
        destination.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(source, destination)
        destination.chmod(0o600)
        if _sha256(destination) != report.get("input_sha256"):
            raise ValueError("sequence input changed between analysis and bundle capture")
        artifacts = [{"path": relative_input, "size": destination.stat().st_size, "sha256": _sha256(destination)}]
        report_path = target / "report.json"
        _write_private(report_path, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        artifacts.append({"path": "report.json", "size": report_path.stat().st_size, "sha256": _sha256(report_path)})
        source_root = Path(__file__).resolve().parent
        module_rows = []
        for module_name in ("__init__.py", "genomics.py"):
            original = source_root / module_name
            relative = "code/medai_science/" + module_name
            captured = target / relative
            captured.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copyfile(original, captured)
            captured.chmod(0o600)
            digest = _sha256(captured)
            artifacts.append({"path": relative, "size": captured.stat().st_size, "sha256": digest})
            module_key = "medai_science" if module_name == "__init__.py" else "medai_science.genomics"
            module_rows.append({"module": module_key, "path": relative, "sha256": digest})
        manifest = {
            "schema_version": "1",
            "created_at_unix": int(__import__("time").time()),
            "analysis": {"workflow": "sequence." + normalized_format, "input": relative_input, "max_records": max_records, "molecule_type": molecule_type if normalized_format == "fasta" else None},
            "network_accessed": False,
            "runtime": {"python": os.sys.version.split()[0]},
            "scientific_modules": module_rows,
            "artifacts": artifacts,
            "report_sha256": _sha256(report_path),
        }
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed": False}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def create_compound_bundle(bundle_dir: str | Path, match: dict[str, Any]) -> dict[str, Any]:
    """Capture one exact PubChem identity and replayable local RDKit descriptors."""
    from medai_science.sources.pubchem import PubChemClient

    if not isinstance(match, dict) or match.get("source") != "PubChem PUG REST":
        raise ValueError("compound bundle requires one normalized PubChem source record")
    if not isinstance(match.get("cid"), int) or isinstance(match.get("cid"), bool) or match["cid"] < 1:
        raise ValueError("compound bundle requires a positive PubChem CID")
    smiles = match.get("smiles")
    if not isinstance(smiles, str) or not smiles or len(smiles) > 4096:
        raise ValueError("compound bundle requires bounded PubChem SMILES")
    if not isinstance(match.get("inchikey"), str) or not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", match["inchikey"]):
        raise ValueError("compound bundle requires a standard InChIKey")
    if not isinstance(match.get("record_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", match["record_sha256"]):
        raise ValueError("compound bundle requires a PubChem source-record digest")
    rdkit = match.get("rdkit")
    if not isinstance(rdkit, dict) or rdkit.get("status") != "computed" or rdkit.get("software") != "RDKit":
        raise ValueError("compound bundle requires locally computed RDKit descriptors")
    if not isinstance(rdkit.get("version"), str) or not rdkit["version"] or rdkit.get("input_smiles_sha256") != hashlib.sha256(smiles.encode()).hexdigest():
        raise ValueError("RDKit descriptor version or input SMILES digest is invalid")
    descriptors = rdkit.get("descriptors")
    if not isinstance(descriptors, dict) or not descriptors:
        raise ValueError("compound bundle requires a descriptor object")
    for name, value in descriptors.items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("compound bundle descriptor values must be named finite numbers")
    if PubChemClient._rdkit_descriptors(smiles) != rdkit:
        raise ValueError("compound report RDKit descriptors do not match the supplied PubChem SMILES")
    target = Path(bundle_dir).expanduser().resolve()
    if target.exists():
        raise ValueError("compound bundle directory already exists")
    record = {
        "cid": match["cid"],
        "title": str(match.get("title", ""))[:1000],
        "molecular_formula": match.get("molecular_formula"),
        "molecular_weight_g_mol": match.get("molecular_weight_g_mol"),
        "exact_mass": match.get("exact_mass"),
        "xlogp": match.get("xlogp"),
        "smiles": smiles,
        "inchi": match.get("inchi"),
        "inchikey": match["inchikey"],
        "rdkit": rdkit,
        "source": "PubChem PUG REST",
        "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{match['cid']}",
        "source_record_sha256": match["record_sha256"],
        "retrieved_at": match.get("retrieved_at"),
        "interpretation_limit": "Local RDKit descriptors reproduce the captured calculation only. They do not predict target engagement, biological activity, exposure, safety, or efficacy.",
    }
    target.mkdir(parents=True, mode=0o700)
    try:
        report_path = target / "compound.json"
        _write_private(report_path, (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        artifacts = [{"path": "compound.json", "size": report_path.stat().st_size, "sha256": _sha256(report_path)}]
        source = Path(__file__).resolve().parent / "sources" / "pubchem.py"
        captured = target / "code" / "medai_science" / "sources" / "pubchem.py"
        captured.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(source, captured)
        captured.chmod(0o600)
        code_digest = _sha256(captured)
        artifacts.append({"path": "code/medai_science/sources/pubchem.py", "size": captured.stat().st_size, "sha256": code_digest})
        manifest = {
            "schema_version": "1",
            "created_at_unix": int(__import__("time").time()),
            "analysis": {"workflow": "compound.rdkit_descriptors", "cid": match["cid"], "input_smiles_sha256": rdkit["input_smiles_sha256"], "rdkit_version": rdkit["version"]},
            "network_accessed_during_replay": False,
            "runtime": {"python": __import__("sys").version.split()[0]},
            "scientific_modules": [{"module": "medai_science.sources.pubchem", "path": "code/medai_science/sources/pubchem.py", "sha256": code_digest}],
            "artifacts": artifacts,
            "report_sha256": _sha256(report_path),
        }
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed_during_replay": False}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def verify_genomics_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Verify private bundle integrity; optionally replay only with identical installed analysis code."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("genomics bundle manifest is missing")
    try:
        manifest = _read_bundle_json(manifest_path, "genomics bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("genomics bundle manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("genomics bundle manifest must be an object")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("genomics bundle artifact manifest is empty")
    verified = []
    for item in artifacts:
        relative = item.get("path") if isinstance(item, dict) else None
        path = _safe_bundle_file(bundle, relative)
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError(f"genomics bundle artifact hash or size mismatch: {relative}")
        verified.append(relative)
    if _sha256(_safe_bundle_file(bundle, "report.json")) != manifest.get("report_sha256"):
        raise ValueError("genomics bundle report hash mismatch")
    settings = manifest.get("analysis", {})
    if not isinstance(settings, dict):
        raise ValueError("genomics bundle analysis settings are malformed")
    captured_report = _read_bundle_json(_safe_bundle_file(bundle, "report.json"), "genomics report")
    external_path = settings.get("external_vep")
    has_external_validation = external_path is not None
    if has_external_validation:
        vep_path = _safe_bundle_file(bundle, external_path)
        try:
            vep_receipt = _read_bundle_json(vep_path, "Ensembl VEP receipt")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("captured Ensembl VEP receipt is malformed") from error
        _validate_vep_cross_validation_receipt(captured_report, vep_receipt)
    clinvar_records_path = settings.get("external_clinvar_records")
    has_clinvar = clinvar_records_path is not None
    clinvar_reconciliation_path = settings.get("external_clinvar_reconciliation")
    source_reconciliation_path = settings.get("external_source_reconciliation")
    if has_clinvar != (clinvar_reconciliation_path is not None):
        raise ValueError("captured ClinVar source records and reconciliation paths must appear together")
    if source_reconciliation_path is not None and not (has_clinvar and has_external_validation):
        raise ValueError("cross-source reconciliation requires captured ClinVar and Ensembl VEP receipts")
    gnomad_receipts_path = settings.get("external_gnomad_frequency_receipts")
    gnomad_reconciliation_path = settings.get("external_gnomad_frequency_reconciliation")
    has_gnomad_frequency = gnomad_receipts_path is not None
    if has_gnomad_frequency != (gnomad_reconciliation_path is not None):
        raise ValueError("captured gnomAD frequency receipts and reconciliation paths must appear together")
    if has_gnomad_frequency and not settings.get("reference_fasta"):
        raise ValueError("gnomAD frequency bundle replay requires its captured reference FASTA")
    clinvar_records = None
    clinvar_receipt = None
    source_receipt = None
    if has_clinvar:
        if not settings.get("reference_fasta"):
            raise ValueError("ClinVar bundle replay requires its captured reference FASTA")
        try:
            clinvar_records = _read_bundle_json(_safe_bundle_file(bundle, clinvar_records_path), "ClinVar records")
            clinvar_receipt = _read_bundle_json(_safe_bundle_file(bundle, clinvar_reconciliation_path), "ClinVar reconciliation receipt")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("captured ClinVar bundle records or reconciliation are malformed") from error
        from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
        recomputed_clinvar = reconcile_clinvar_variants(captured_report, clinvar_records, _safe_bundle_file(bundle, settings["reference_fasta"]), settings["genome_build"])
        if recomputed_clinvar != clinvar_receipt:
            raise ValueError("recomputed ClinVar reconciliation differs from the captured receipt")
    if source_reconciliation_path is not None:
        try:
            source_receipt = _read_bundle_json(_safe_bundle_file(bundle, source_reconciliation_path), "cross-source reconciliation receipt")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("captured cross-source reconciliation is malformed") from error
        from medai_science.variant_source_reconciliation import reconcile_variant_sources
        recomputed_sources = reconcile_variant_sources(captured_report, clinvar_receipt, vep_receipt)
        if recomputed_sources != source_receipt:
            raise ValueError("recomputed cross-source reconciliation differs from the captured receipt")
    gnomad_receipts = None
    gnomad_reconciliation = None
    if has_gnomad_frequency:
        try:
            gnomad_receipts = _read_bundle_json(_safe_bundle_file(bundle, gnomad_receipts_path), "gnomAD frequency receipts")
            gnomad_reconciliation = _read_bundle_json(_safe_bundle_file(bundle, gnomad_reconciliation_path), "gnomAD frequency reconciliation")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("captured gnomAD bundle receipts or reconciliation are malformed") from error
        from medai_science.variant_source_reconciliation import reconcile_gnomad_frequency
        recomputed_gnomad = reconcile_gnomad_frequency(captured_report, gnomad_receipts)
        if recomputed_gnomad != gnomad_reconciliation:
            raise ValueError("recomputed gnomAD frequency reconciliation differs from the captured receipt")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "external_vep_receipt_verified": has_external_validation, "external_clinvar_receipt_verified": has_clinvar, "cross_source_reconciliation_verified": source_reconciliation_path is not None, "external_gnomad_frequency_verified": has_gnomad_frequency, "external_source_requeried": False}
    _verify_python_runtime(manifest)
    from medai_science.genomics import analyze_vcf
    import medai_science.genomics as genomics_module
    import medai_science.transcripts as transcripts_module
    import medai_science as package_module
    modules = {"medai_science": package_module, "medai_science.genomics": genomics_module, "medai_science.transcripts": transcripts_module}
    if has_external_validation:
        modules["medai_science.reproducibility"] = __import__("medai_science.reproducibility", fromlist=["verify_genomics_bundle"])
        modules["medai_science.sources.ensembl_vep"] = __import__("medai_science.sources.ensembl_vep", fromlist=["EnsemblVEPClient"])
    if has_clinvar:
        modules["medai_science.clinvar_reconciliation"] = __import__("medai_science.clinvar_reconciliation", fromlist=["reconcile_clinvar_variants"])
    if source_reconciliation_path is not None:
        modules["medai_science.variant_source_reconciliation"] = __import__("medai_science.variant_source_reconciliation", fromlist=["reconcile_variant_sources"])
    if has_gnomad_frequency:
        modules["medai_science.reproducibility"] = __import__("medai_science.reproducibility", fromlist=["verify_genomics_bundle"])
        modules["medai_science.variant_source_reconciliation"] = __import__("medai_science.variant_source_reconciliation", fromlist=["reconcile_gnomad_frequency"])
        modules["medai_science.sources.gnomad"] = __import__("medai_science.sources.gnomad", fromlist=["GnomADClient"])
        modules["medai_science.sources.bounded_http"] = __import__("medai_science.sources.bounded_http", fromlist=["read_bounded_response"])
    for row in manifest.get("scientific_modules", []):
        module = modules.get(row.get("module"))
        if module is None or _sha256(Path(module.__file__).resolve()) != row.get("sha256"):
            raise ValueError(f"installed scientific module {row.get('module')} differs from the captured version")
    vcf = _safe_bundle_file(bundle, "inputs/cohort.vcf")
    reference = _safe_bundle_file(bundle, settings["reference_fasta"]) if settings.get("reference_fasta") else None
    annotation = _safe_bundle_file(bundle, settings["transcript_gff3"]) if settings.get("transcript_gff3") else None
    replay_report = analyze_vcf(vcf, settings["genome_build"], int(settings["max_variants"]), reference, annotation)
    if replay_report != captured_report:
        raise ValueError("replayed genomic report differs from the captured report")
    if has_external_validation:
        _validate_vep_cross_validation_receipt(replay_report, vep_receipt)
    if has_clinvar:
        from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
        replay_clinvar = reconcile_clinvar_variants(replay_report, clinvar_records, reference, settings["genome_build"])
        if replay_clinvar != clinvar_receipt:
            raise ValueError("replayed ClinVar reconciliation differs from the captured receipt")
    if source_reconciliation_path is not None:
        from medai_science.variant_source_reconciliation import reconcile_variant_sources
        replay_sources = reconcile_variant_sources(replay_report, clinvar_receipt, vep_receipt)
        if replay_sources != source_receipt:
            raise ValueError("replayed cross-source reconciliation differs from the captured receipt")
    if has_gnomad_frequency:
        from medai_science.variant_source_reconciliation import reconcile_gnomad_frequency
        replay_gnomad = reconcile_gnomad_frequency(replay_report, gnomad_receipts)
        if replay_gnomad != gnomad_reconciliation:
            raise ValueError("replayed gnomAD frequency reconciliation differs from the captured receipt")
    return {"valid": True, "replayed": True, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "external_vep_receipt_verified": has_external_validation, "external_clinvar_receipt_verified": has_clinvar, "cross_source_reconciliation_verified": source_reconciliation_path is not None, "external_gnomad_frequency_verified": has_gnomad_frequency, "external_source_requeried": False}


def verify_sequence_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Verify sequence bundle integrity and optionally rerun matching installed parser code."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("sequence bundle manifest is missing")
    try:
        manifest = _read_bundle_json(manifest_path, "sequence bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("sequence bundle manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ValueError("sequence bundle manifest must be an object")
    settings = manifest.get("analysis", {})
    if not isinstance(settings, dict):
        raise ValueError("sequence bundle analysis settings are malformed")
    workflow = settings.get("workflow")
    if workflow not in {"sequence.fasta", "sequence.fastq"}:
        raise ValueError("bundle does not describe a supported FASTA/FASTQ workflow")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("sequence bundle artifact manifest is empty")
    verified = []
    for item in artifacts:
        relative = item.get("path") if isinstance(item, dict) else None
        path = _safe_bundle_file(bundle, relative)
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError(f"sequence bundle artifact hash or size mismatch: {relative}")
        verified.append(relative)
    report_path = _safe_bundle_file(bundle, "report.json")
    if _sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("sequence bundle report hash mismatch")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path)}
    _verify_python_runtime(manifest)
    import medai_science as package_module
    import medai_science.genomics as genomics_module

    modules = {"medai_science": package_module, "medai_science.genomics": genomics_module}
    for row in manifest.get("scientific_modules", []):
        module = modules.get(row.get("module"))
        if module is None or _sha256(Path(module.__file__).resolve()) != row.get("sha256"):
            raise ValueError(f"installed scientific module {row.get('module')} differs from the captured version")
    input_relative = settings.get("input")
    if not isinstance(input_relative, str) or Path(input_relative).is_absolute() or ".." in Path(input_relative).parts:
        raise ValueError("sequence bundle input path is invalid")
    input_file = _safe_bundle_file(bundle, input_relative)
    if workflow == "sequence.fasta":
        from medai_science.genomics import inspect_fasta
        report = inspect_fasta(input_file, settings.get("molecule_type"), int(settings.get("max_records")))
    else:
        from medai_science.genomics import inspect_fastq
        report = inspect_fastq(input_file, int(settings.get("max_records")))
    captured_report = _read_bundle_json(report_path, "sequence report")
    if report != captured_report:
        raise ValueError("replayed sequence report differs from the captured report")
    return {"valid": True, "replayed": True, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"]}


def verify_compound_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Verify a private compound bundle and optionally recompute RDKit descriptors offline."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("compound bundle manifest is missing")
    try:
        manifest = _read_bundle_json(manifest_path, "compound bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("compound bundle manifest is invalid") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("analysis"), dict):
        raise ValueError("compound bundle manifest is malformed")
    settings = manifest["analysis"]
    if settings.get("workflow") != "compound.rdkit_descriptors" or manifest.get("network_accessed_during_replay") is not False:
        raise ValueError("bundle does not describe an offline compound-descriptor workflow")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("compound bundle artifact manifest is empty")
    verified = []
    for item in artifacts:
        relative = item.get("path") if isinstance(item, dict) else None
        path = _safe_bundle_file(bundle, relative)
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError(f"compound bundle artifact hash or size mismatch: {relative}")
        verified.append(relative)
    report_path = _safe_bundle_file(bundle, "compound.json")
    if _sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("compound bundle report hash mismatch")
    try:
        report = _read_bundle_json(report_path, "compound report")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("compound bundle report is invalid") from error
    if not isinstance(report, dict) or report.get("source") != "PubChem PUG REST":
        raise ValueError("compound bundle report is malformed")
    cid = settings.get("cid")
    if not isinstance(cid, int) or isinstance(cid, bool) or cid < 1 or report.get("cid") != cid:
        raise ValueError("compound bundle CID does not match the captured source record")
    if report.get("source_url") != f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}":
        raise ValueError("compound bundle source URL does not match its PubChem CID")
    if not isinstance(report.get("inchikey"), str) or not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", report["inchikey"]):
        raise ValueError("compound bundle source record lacks a standard InChIKey")
    if not isinstance(report.get("source_record_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", report["source_record_sha256"]):
        raise ValueError("compound bundle source record digest is malformed")
    smiles, rdkit = report.get("smiles"), report.get("rdkit")
    if not isinstance(smiles, str) or not isinstance(rdkit, dict) or rdkit.get("status") != "computed":
        raise ValueError("compound bundle lacks its captured SMILES or computed descriptors")
    if hashlib.sha256(smiles.encode()).hexdigest() != settings.get("input_smiles_sha256") or rdkit.get("input_smiles_sha256") != settings.get("input_smiles_sha256"):
        raise ValueError("compound bundle SMILES digest does not match the analysis manifest")
    if rdkit.get("version") != settings.get("rdkit_version"):
        raise ValueError("compound bundle RDKit version differs from the captured analysis metadata")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path)}
    _verify_python_runtime(manifest)
    try:
        from rdkit import rdBase
    except ImportError as error:
        raise ValueError("RDKit is required to replay this compound bundle") from error
    if rdBase.rdkitVersion != settings["rdkit_version"]:
        raise ValueError("installed RDKit version differs from the captured compound analysis")
    import medai_science.sources.pubchem as pubchem_module

    captured_code = _safe_bundle_file(bundle, "code/medai_science/sources/pubchem.py")
    if _sha256(Path(pubchem_module.__file__).resolve()) != _sha256(captured_code):
        raise ValueError("installed PubChem descriptor implementation differs from the captured bundle code")
    from medai_science.sources.pubchem import PubChemClient
    replayed = PubChemClient._rdkit_descriptors(smiles)
    if replayed != rdkit:
        raise ValueError("replayed RDKit descriptors differ from the captured report")
    return {"valid": True, "replayed": True, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"]}


def create_protein_bundle(bundle_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Capture an attributed UniProtKB record and its deterministic features for offline replay."""
    from medai_science.sources.uniprot import UniProtClient, _entry_features

    if not isinstance(result, dict) or result.get("source") != "UniProtKB" or result.get("license") != "CC BY 4.0":
        raise ValueError("protein bundle requires an attributed UniProtKB record under CC BY 4.0")
    accession = UniProtClient.validate_accession(result.get("accession"))
    record = result.get("record")
    if not isinstance(record, dict) or record.get("primaryAccession", "").upper() != accession.split("-")[0]:
        raise ValueError("protein bundle record accession does not match the requested accession")
    canonical_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(canonical_record) > 8 * 1024 * 1024:
        raise ValueError("UniProtKB source record exceeds the 8 MiB bundle limit")
    record_digest = hashlib.sha256(canonical_record).hexdigest()
    if result.get("record_sha256") != record_digest:
        raise ValueError("UniProtKB source-record digest does not match the retained record")
    if result.get("entry_url") != f"https://www.uniprot.org/uniprotkb/{accession}":
        raise ValueError("UniProtKB entry URL does not match the accession")
    domains, structures = _entry_features(record)
    if result.get("domains_and_regions") != domains or result.get("experimental_structure_cross_references") != structures:
        raise ValueError("reported protein features do not match the source record")
    target = Path(bundle_dir).expanduser().resolve()
    if target.exists():
        raise ValueError("protein bundle directory already exists")
    target.mkdir(parents=True, mode=0o700)
    try:
        report = {
            "source": "UniProtKB", "accession": accession, "entry_url": result["entry_url"],
            "retrieved_at": result.get("retrieved_at"), "license": "CC BY 4.0",
            "attribution": "UniProt Consortium", "record_sha256": record_digest,
            "record": record, "domains_and_regions": domains,
            "experimental_structure_cross_references": structures,
            "structure_cross_reference_limit": result.get("structure_cross_reference_limit"),
            "interpretation_limit": "This offline replay reproduces feature extraction from the captured UniProtKB record. It does not independently validate domain annotations, PDB mappings, structure quality, or biological function.",
        }
        report_path = target / "protein.json"
        _write_private(report_path, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        artifacts = [{"path": "protein.json", "size": report_path.stat().st_size, "sha256": _sha256(report_path)}]
        source = Path(__file__).resolve().parent / "sources" / "uniprot.py"
        captured = target / "code" / "medai_science" / "sources" / "uniprot.py"
        captured.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(source, captured)
        captured.chmod(0o600)
        code_digest = _sha256(captured)
        artifacts.append({"path": "code/medai_science/sources/uniprot.py", "size": captured.stat().st_size, "sha256": code_digest})
        manifest = {
            "schema_version": "1", "created_at_unix": int(__import__("time").time()),
            "analysis": {"workflow": "protein.uniprot_features", "accession": accession, "source_record_sha256": record_digest},
            "source_license": "CC BY 4.0", "source_attribution": "UniProt Consortium",
            "network_accessed_during_replay": False,
            "runtime": {"python": os.sys.version.split()[0]},
            "scientific_modules": [{"module": "medai_science.sources.uniprot", "path": "code/medai_science/sources/uniprot.py", "sha256": code_digest}],
            "artifacts": artifacts, "report_sha256": _sha256(report_path),
        }
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed_during_replay": False}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def verify_protein_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Verify a UniProtKB snapshot and optionally replay feature extraction offline."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("protein bundle manifest is missing")
    try:
        manifest = _read_bundle_json(manifest_path, "protein bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("protein bundle manifest is invalid") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("analysis"), dict):
        raise ValueError("protein bundle manifest is malformed")
    settings = manifest["analysis"]
    if settings.get("workflow") != "protein.uniprot_features" or manifest.get("network_accessed_during_replay") is not False or manifest.get("source_license") != "CC BY 4.0":
        raise ValueError("bundle does not describe an attributed offline UniProtKB feature workflow")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts if isinstance(item, dict)} != {"protein.json", "code/medai_science/sources/uniprot.py"}:
        raise ValueError("protein bundle artifact manifest is empty")
    verified = []
    for item in artifacts:
        relative = item.get("path") if isinstance(item, dict) else None
        path = _safe_bundle_file(bundle, relative)
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError(f"protein bundle artifact hash or size mismatch: {relative}")
        verified.append(relative)
    report_path = _safe_bundle_file(bundle, "protein.json")
    if _sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("protein bundle report hash mismatch")
    try:
        report = _read_bundle_json(report_path, "protein report")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("protein bundle report is invalid") from error
    from medai_science.sources.uniprot import UniProtClient, _entry_features
    if not isinstance(report, dict) or report.get("source") != "UniProtKB" or report.get("license") != "CC BY 4.0":
        raise ValueError("protein bundle report lacks source attribution or reuse license")
    accession = UniProtClient.validate_accession(settings.get("accession"))
    record = report.get("record")
    if not isinstance(record, dict) or report.get("accession") != accession or report.get("entry_url") != f"https://www.uniprot.org/uniprotkb/{accession}" or record.get("primaryAccession", "").upper() != accession.split("-")[0]:
        raise ValueError("protein bundle accession does not match its captured source record")
    canonical_record = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(canonical_record) > 8 * 1024 * 1024:
        raise ValueError("UniProtKB source record exceeds the 8 MiB bundle limit")
    digest = hashlib.sha256(canonical_record).hexdigest()
    if digest != settings.get("source_record_sha256") or digest != report.get("record_sha256"):
        raise ValueError("protein bundle source-record digest mismatch")
    if report.get("attribution") != "UniProt Consortium":
        raise ValueError("protein bundle is missing UniProt Consortium attribution")
    domains, structures = _entry_features(record)
    if domains != report.get("domains_and_regions") or structures != report.get("experimental_structure_cross_references"):
        raise ValueError("captured UniProtKB features do not match the source record")
    modules = manifest.get("scientific_modules")
    if not isinstance(modules, list) or len(modules) != 1 or modules[0].get("module") != "medai_science.sources.uniprot" or modules[0].get("path") != "code/medai_science/sources/uniprot.py":
        raise ValueError("protein bundle scientific module manifest is malformed")
    captured_code = _safe_bundle_file(bundle, "code/medai_science/sources/uniprot.py")
    if _sha256(captured_code) != modules[0].get("sha256"):
        raise ValueError("captured UniProtKB feature extractor hash does not match its module manifest")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path)}
    _verify_python_runtime(manifest)
    import medai_science.sources.uniprot as uniprot_module
    if _sha256(Path(uniprot_module.__file__).resolve()) != _sha256(captured_code):
        raise ValueError("installed UniProtKB feature extractor differs from the captured bundle code")
    return {"valid": True, "replayed": True, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"]}


def create_pathway_bundle(bundle_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Capture a bounded Reactome analysis response for offline integrity replay."""
    from medai_science.sources.reactome import ReactomeClient

    if not isinstance(result, dict) or result.get("source") != "Reactome Analysis Service":
        raise ValueError("pathway bundle requires a Reactome Analysis Service result")
    parameters = result.get("analysis_parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"pageSize", "sortBy", "order", "species"} or parameters.get("sortBy") != "ENTITIES_PVALUE" or parameters.get("order") != "ASC" or parameters.get("species") != result.get("species"):
        raise ValueError("Reactome analysis parameters are missing or unsupported")
    genes = ReactomeClient.validate(result.get("input_identifiers"), result.get("species"), parameters.get("pageSize"))
    input_digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    request_digest = _canonical_hash({"endpoint": ReactomeClient.endpoint, "parameters": parameters, "identifiers": genes})
    if result.get("input_sha256") != input_digest or result.get("request_sha256") != request_digest or not isinstance(result.get("result"), dict):
        raise ValueError("Reactome result does not match its exact input identifiers")
    target = Path(bundle_dir).expanduser().resolve()
    if target.exists():
        raise ValueError("pathway bundle directory already exists")
    target.mkdir(parents=True, mode=0o700)
    try:
        captured = {key: result.get(key) for key in ("source", "source_url", "retrieved_at", "species", "input_identifiers", "input_sha256", "analysis_parameters", "request_sha256", "result", "interpretation_limit")}
        raw = (json.dumps(captured, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Reactome result exceeds the 8 MiB bundle limit")
        report = target / "pathways.json"
        _write_private(report, raw)
        artifacts = [{"path": "pathways.json", "size": report.stat().st_size, "sha256": _sha256(report)}]
        source = Path(__file__).resolve().parent / "sources" / "reactome.py"
        code = target / "code" / "medai_science" / "sources" / "reactome.py"
        code.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(source, code)
        code.chmod(0o600)
        code_hash = _sha256(code)
        artifacts.append({"path": "code/medai_science/sources/reactome.py", "size": code.stat().st_size, "sha256": code_hash})
        manifest = {"schema_version": "1", "created_at_unix": int(__import__("time").time()), "analysis": {"workflow": "pathways.reactome_overrepresentation", "species": result["species"], "input_sha256": input_digest, "request_sha256": request_digest, "parameters": parameters}, "network_accessed_during_replay": False, "runtime": {"python": os.sys.version.split()[0]}, "scientific_modules": [{"module": "medai_science.sources.reactome", "path": "code/medai_science/sources/reactome.py", "sha256": code_hash}], "artifacts": artifacts, "report_sha256": _sha256(report), "interpretation_limit": "Replay verifies the captured provider response and its exact input/query-parameter binding offline; it does not rerun Reactome or authenticate provider authorship."}
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"]}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def verify_pathway_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Check a captured Reactome response and exact input binding without network access."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    try:
        manifest = _read_bundle_json(manifest_path, "pathway bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pathway bundle manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or manifest.get("analysis", {}).get("workflow") != "pathways.reactome_overrepresentation" or manifest.get("network_accessed_during_replay") is not False:
        raise ValueError("bundle does not describe an offline Reactome pathway workflow")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts if isinstance(item, dict)} != {"pathways.json", "code/medai_science/sources/reactome.py"}:
        raise ValueError("pathway bundle artifact manifest is malformed")
    for item in artifacts:
        path = _safe_bundle_file(bundle, item.get("path"))
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError("pathway bundle artifact integrity check failed")
    report_path = _safe_bundle_file(bundle, "pathways.json")
    if _sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("pathway bundle report hash mismatch")
    report = _read_bundle_json(report_path, "pathway report")
    from medai_science.sources.reactome import ReactomeClient
    parameters = report.get("analysis_parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"pageSize", "sortBy", "order", "species"} or parameters.get("sortBy") != "ENTITIES_PVALUE" or parameters.get("order") != "ASC" or parameters.get("species") != report.get("species"):
        raise ValueError("captured Reactome analysis parameters are malformed")
    genes = ReactomeClient.validate(report.get("input_identifiers"), report.get("species"), parameters.get("pageSize"))
    digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    request_digest = _canonical_hash({"endpoint": ReactomeClient.endpoint, "parameters": parameters, "identifiers": genes})
    if report.get("source") != "Reactome Analysis Service" or report.get("source_url") != ReactomeClient.endpoint or report.get("input_sha256") != digest or digest != manifest["analysis"].get("input_sha256") or report.get("request_sha256") != request_digest or request_digest != manifest["analysis"].get("request_sha256") or parameters != manifest["analysis"].get("parameters") or not isinstance(report.get("result"), dict):
        raise ValueError("captured Reactome response is not bound to its exact validated input")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(artifacts), "bundle_sha256": _sha256(manifest_path)}
    _verify_python_runtime(manifest)
    import medai_science.sources.reactome as reactome_module
    code = _safe_bundle_file(bundle, "code/medai_science/sources/reactome.py")
    if _sha256(Path(reactome_module.__file__).resolve()) != _sha256(code):
        raise ValueError("installed Reactome adapter differs from captured bundle code")
    return {"valid": True, "replayed": True, "artifact_count": len(artifacts), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed": False}


def create_clinical_trial_bundle(bundle_dir: str | Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Capture a public ClinicalTrials.gov v2 snapshot and its normalized view for offline replay."""
    from medai_science.sources.clinical_trials import ClinicalTrialsClient, _snapshot_hash, normalize_study

    if not isinstance(snapshot, dict) or snapshot.get("source") != "ClinicalTrials.gov":
        raise ValueError("clinical trial bundle requires a normalized ClinicalTrials.gov snapshot")
    nct_id = snapshot.get("nct_id")
    if not isinstance(nct_id, str) or not re.fullmatch(r"NCT\d{8}", nct_id):
        raise ValueError("clinical trial bundle requires a valid NCT identifier")
    retrieved_at = snapshot.get("provenance", {}).get("retrieved_at") if isinstance(snapshot.get("provenance"), dict) else None
    record = snapshot.get("record")
    if not isinstance(retrieved_at, str) or not retrieved_at or not isinstance(record, dict):
        raise ValueError("clinical trial bundle requires the retained source record and retrieval time")
    source_digest = _snapshot_hash(record)
    if snapshot.get("provenance", {}).get("record_snapshot_sha256") != source_digest:
        raise ValueError("clinical trial bundle source-record digest does not match the retained record")
    normalized = normalize_study(record, retrieved_at)
    if _canonical_hash(normalized) != _canonical_hash(snapshot):
        raise ValueError("clinical trial snapshot fields do not match the retained source record")
    endpoint = f"{ClinicalTrialsClient.endpoint}/{nct_id}"
    if snapshot.get("provenance", {}).get("api_endpoint") != ClinicalTrialsClient.endpoint or snapshot.get("url") != f"https://clinicaltrials.gov/study/{nct_id}":
        raise ValueError("clinical trial snapshot endpoint or record URL is inconsistent")
    target = Path(bundle_dir).expanduser().resolve()
    if target.exists():
        raise ValueError("clinical trial bundle directory already exists")
    target.mkdir(parents=True, mode=0o700)
    try:
        report = target / "trial.json"
        raw = (json.dumps({"snapshot": snapshot}, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("clinical trial snapshot bundle exceeds 16 MiB")
        _write_private(report, raw)
        artifacts = [{"path": "trial.json", "size": report.stat().st_size, "sha256": _sha256(report)}]
        source = Path(__file__).resolve().parent / "sources" / "clinical_trials.py"
        captured = target / "code" / "medai_science" / "sources" / "clinical_trials.py"
        captured.parent.mkdir(parents=True, mode=0o700)
        shutil.copyfile(source, captured)
        captured.chmod(0o600)
        code_digest = _sha256(captured)
        artifacts.append({"path": "code/medai_science/sources/clinical_trials.py", "size": captured.stat().st_size, "sha256": code_digest})
        manifest = {
            "schema_version": "1",
            "created_at_unix": int(__import__("time").time()),
            "analysis": {"workflow": "trials.clinicaltrials_v2_snapshot", "nct_id": nct_id, "source_record_sha256": source_digest, "retrieved_at": retrieved_at, "request_url": f"{endpoint}?format=json"},
            "network_accessed_during_replay": False,
            "runtime": {"python": os.sys.version.split()[0]},
            "scientific_modules": [{"module": "medai_science.sources.clinical_trials", "path": "code/medai_science/sources/clinical_trials.py", "sha256": code_digest}],
            "artifacts": artifacts,
            "report_sha256": _sha256(report),
            "interpretation_limit": "Offline replay recomputes source-field normalization from the captured public registry record. It does not authenticate ClinicalTrials.gov authorship, retrieve record history, assess study quality, determine participant eligibility, or provide clinical advice.",
        }
        manifest_path = target / "manifest.json"
        _write_private(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        target.chmod(0o700)
        for directory in sorted((path for path in target.rglob("*") if path.is_dir()), reverse=True):
            directory.chmod(0o700)
        return {"bundle_dir": str(target), "artifact_count": len(artifacts), "manifest_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed_during_replay": False}
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def verify_clinical_trial_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Verify a captured ClinicalTrials.gov source record and replay its normalization offline."""
    bundle = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("clinical trial bundle manifest is missing")
    try:
        manifest = _read_bundle_json(manifest_path, "ClinicalTrials.gov bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("clinical trial bundle manifest is invalid") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("analysis"), dict) or manifest["analysis"].get("workflow") != "trials.clinicaltrials_v2_snapshot" or manifest.get("network_accessed_during_replay") is not False:
        raise ValueError("bundle does not describe an offline ClinicalTrials.gov snapshot workflow")
    artifacts = manifest.get("artifacts")
    expected_paths = {"trial.json", "code/medai_science/sources/clinical_trials.py"}
    if not isinstance(artifacts, list) or {item.get("path") for item in artifacts if isinstance(item, dict)} != expected_paths:
        raise ValueError("clinical trial bundle artifact manifest is malformed")
    verified = []
    for item in artifacts:
        path = _safe_bundle_file(bundle, item.get("path") if isinstance(item, dict) else None)
        if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
            raise ValueError(f"clinical trial bundle artifact integrity check failed: {item.get('path')}")
        verified.append(item["path"])
    report_path = _safe_bundle_file(bundle, "trial.json")
    if _sha256(report_path) != manifest.get("report_sha256"):
        raise ValueError("clinical trial bundle report hash mismatch")
    try:
        report = _read_bundle_json(report_path, "ClinicalTrials.gov bundle report")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("clinical trial bundle report is invalid") from error
    snapshot = report.get("snapshot") if isinstance(report, dict) else None
    analysis = manifest["analysis"]
    if not isinstance(snapshot, dict) or snapshot.get("nct_id") != analysis.get("nct_id"):
        raise ValueError("clinical trial bundle report does not match its NCT identifier")
    from medai_science.sources.clinical_trials import ClinicalTrialsClient, _snapshot_hash, normalize_study
    nct_id = snapshot.get("nct_id")
    record = snapshot.get("record")
    provenance = snapshot.get("provenance")
    retrieved_at = analysis.get("retrieved_at")
    if not isinstance(nct_id, str) or not re.fullmatch(r"NCT\d{8}", nct_id) or not isinstance(record, dict) or not isinstance(provenance, dict):
        raise ValueError("clinical trial bundle source snapshot is malformed")
    source_digest = _snapshot_hash(record)
    if source_digest != analysis.get("source_record_sha256") or source_digest != provenance.get("record_snapshot_sha256"):
        raise ValueError("clinical trial bundle source-record digest mismatch")
    if provenance.get("api_endpoint") != ClinicalTrialsClient.endpoint or provenance.get("record_url") != f"https://clinicaltrials.gov/study/{nct_id}" or snapshot.get("url") != provenance.get("record_url"):
        raise ValueError("clinical trial bundle source URL or endpoint is inconsistent")
    if provenance.get("retrieved_at") != retrieved_at or analysis.get("request_url") != f"{ClinicalTrialsClient.endpoint}/{nct_id}?format=json":
        raise ValueError("clinical trial bundle request provenance is inconsistent")
    recomputed = normalize_study(record, retrieved_at)
    if _canonical_hash(recomputed) != _canonical_hash(snapshot):
        raise ValueError("captured trial normalization does not match its source record")
    modules = manifest.get("scientific_modules")
    if not isinstance(modules, list) or len(modules) != 1 or modules[0].get("module") != "medai_science.sources.clinical_trials" or modules[0].get("path") != "code/medai_science/sources/clinical_trials.py":
        raise ValueError("clinical trial bundle scientific module manifest is malformed")
    captured_code = _safe_bundle_file(bundle, "code/medai_science/sources/clinical_trials.py")
    if _sha256(captured_code) != modules[0].get("sha256"):
        raise ValueError("captured clinical trial adapter hash does not match its module manifest")
    if not replay:
        return {"valid": True, "replayed": False, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path)}
    _verify_python_runtime(manifest)
    import medai_science.sources.clinical_trials as trials_module
    if _sha256(Path(trials_module.__file__).resolve()) != _sha256(captured_code):
        raise ValueError("installed ClinicalTrials.gov adapter differs from the captured bundle code")
    return {"valid": True, "replayed": True, "artifact_count": len(verified), "bundle_sha256": _sha256(manifest_path), "report_sha256": manifest["report_sha256"], "network_accessed": False}


def verify_analysis_bundle(bundle_dir: str | Path, replay: bool = False) -> dict[str, Any]:
    """Dispatch integrity/replay verification to the workflow named in its manifest."""
    manifest_path = Path(bundle_dir).expanduser().resolve() / "manifest.json"
    try:
        manifest = _read_bundle_json(manifest_path, "analysis bundle manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("analysis bundle manifest is missing or invalid") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("analysis"), dict):
        raise ValueError("analysis bundle manifest is malformed")
    workflow = manifest["analysis"].get("workflow")
    if workflow in {"sequence.fasta", "sequence.fastq"}:
        return verify_sequence_bundle(bundle_dir, replay)
    if workflow == "compound.rdkit_descriptors":
        return verify_compound_bundle(bundle_dir, replay)
    if workflow == "protein.uniprot_features":
        return verify_protein_bundle(bundle_dir, replay)
    if workflow == "pathways.reactome_overrepresentation":
        return verify_pathway_bundle(bundle_dir, replay)
    if workflow == "trials.clinicaltrials_v2_snapshot":
        return verify_clinical_trial_bundle(bundle_dir, replay)
    if workflow == "genomics.vcf":
        return verify_genomics_bundle(bundle_dir, replay)
    raise ValueError(f"analysis bundle uses an unsupported workflow: {workflow!r}")
