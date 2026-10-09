"""Build source searches for graph-path hypotheses without scoring novelty."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from medai_science.sources.safety import ensure_public_research_query


_STATEMENT = re.compile(r"^(.+?) may be connected to (.+?) through (.+?)\.$")


def build_hypothesis_literature_query(candidate: dict[str, Any], max_bytes: int = 256) -> str:
    """Search PubMed for every named entity on one deterministic graph path."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
        raise ValueError("hypothesis candidate requires a stable ID")
    statement = candidate.get("statement")
    if not isinstance(statement, str):
        raise ValueError("hypothesis candidate requires a deterministic graph-path statement")
    raw_entities = candidate.get("entity_labels")
    if raw_entities is None:
        # Accept retained pre-multi-hop proposals during migration.
        match = _STATEMENT.fullmatch(statement.strip())
        if match is None:
            raise ValueError("hypothesis statement does not match the supported graph-path form")
        raw_entities = [match.group(1), *re.split(r"\s*→\s*", match.group(3)), match.group(2)]
    if not isinstance(raw_entities, list) or not 3 <= len(raw_entities) <= 5 or any(not isinstance(item, str) for item in raw_entities):
        raise ValueError("hypothesis search requires three to five path entity labels")
    expected_statement = f"{raw_entities[0]} may be connected to {raw_entities[-1]} through {' → '.join(raw_entities[1:-1])}."
    if statement.strip() != expected_statement:
        raise ValueError("hypothesis statement does not match its ordered graph-path entities")
    phrases = []
    for raw in raw_entities:
        phrase = " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", raw))
        if not phrase or len(phrase) > 80:
            raise ValueError("hypothesis entity label is empty or exceeds 80 normalized characters")
        ensure_public_research_query(phrase)
        phrases.append(phrase)
    if len({phrase.casefold() for phrase in phrases}) != len(phrases):
        raise ValueError("hypothesis search requires distinct path entity labels")
    query = " AND ".join(f'"{phrase}"[All Fields]' for phrase in phrases)
    if len(query.encode("utf-8")) > max_bytes:
        raise ValueError("hypothesis entity query exceeds the 256-byte search limit")
    return ensure_public_research_query(query)


def build_hypothesis_literature_receipt(
    candidate: dict[str, Any], query: str, search: dict[str, Any], paper_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind bounded bibliographic leads and provider coverage to one candidate."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip() or len(candidate["id"]) > 128:
        raise ValueError("hypothesis candidate requires a stable ID")
    expected_query = build_hypothesis_literature_query(candidate)
    query = ensure_public_research_query(query)
    if query != expected_query:
        raise ValueError("literature query does not match the exact hypothesis path")
    if not isinstance(search, dict) or not isinstance(paper_artifacts, list) or len(paper_artifacts) > 20:
        raise ValueError("PubMed search result or stored artifact list is malformed")
    coverage = search.get("coverage") if isinstance(search.get("coverage"), dict) else {}
    coverage_artifact_id = search.get("coverage_artifact_id")
    papers = search.get("results")
    if not isinstance(papers, list):
        raise ValueError("PubMed search result records are malformed")
    if len(papers) != len(paper_artifacts):
        raise ValueError("PubMed papers and stored candidate receipts are misaligned")
    fetched_pmids = []
    for paper in papers:
        provenance = paper.get("provenance") if isinstance(paper, dict) else None
        pmid = provenance.get("source_id") if isinstance(provenance, dict) else None
        if not isinstance(pmid, str):
            raise ValueError("PubMed fetched record lacks an exact PMID")
        fetched_pmids.append(pmid)
    coverage_status = validate_hypothesis_search_coverage(coverage, fetched_pmids)
    if not isinstance(coverage_artifact_id, str) or not coverage_artifact_id or len(coverage_artifact_id) > 256:
        raise ValueError("PubMed search coverage requires its exact stored artifact ID")
    leads = []
    for paper, artifact in zip(papers, paper_artifacts):
        if not isinstance(paper, dict) or not isinstance(artifact, dict):
            raise ValueError("PubMed lead or stored artifact receipt is malformed")
        provenance = paper.get("provenance") if isinstance(paper, dict) else None
        pmid = provenance.get("source_id") if isinstance(provenance, dict) else None
        artifact_id = artifact.get("id")
        record_sha256 = artifact.get("record_sha256")
        if not isinstance(pmid, str) or not pmid.isdigit() or not isinstance(artifact_id, str) or not artifact_id or len(artifact_id) > 256:
            raise ValueError("PubMed lead lacks an exact PMID or stored artifact receipt")
        expected_record = bounded_paper_record(paper)
        if not isinstance(record_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", record_sha256) or record_sha256 != expected_record["record_sha256"]:
            raise ValueError("PubMed paper digest does not match its stored candidate artifact receipt")
        leads.append({"pmid": pmid, "artifact_id": artifact_id, "record_sha256": record_sha256})
    return {
        "schema_version": 1,
        "hypothesis_id": candidate["id"],
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "provider": "PubMed",
        "coverage": coverage,
        "retrieval_coverage": coverage_status,
        "coverage_artifact_id": coverage_artifact_id,
        "returned_record_count": len(papers),
        "candidate_leads": leads,
        "novelty_assessment": None,
        "review_status": "requires_human_relevance_review",
        "interpretation_limit": "This bounded exact-entity PubMed search is a literature discovery lead, not a source-complete search or semantic relevance judgment. The presence or absence of retrieved records does not establish novelty; novelty remains unscored.",
    }


def validate_hypothesis_search_coverage(coverage: dict[str, Any], fetched_pmids: list[str]) -> dict[str, Any]:
    """Check count/omission arithmetic before calling an exact query retrieval complete."""
    integer_fields = ("total_hits", "requested_limit", "returned_pmids", "fetched_records")
    for field in integer_fields:
        value = coverage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"PubMed coverage {field} must be a non-negative integer")
    total, limit, returned, fetched = (coverage[field] for field in integer_fields)
    if (
        not isinstance(fetched_pmids, list)
        or any(not isinstance(pmid, str) or not pmid.isascii() or not pmid.isdecimal() or not 1 <= len(pmid) <= 20 for pmid in fetched_pmids)
        or len(fetched_pmids) != len(set(fetched_pmids))
    ):
        raise ValueError("PubMed fetched records must have unique exact PMID identifiers")
    if not 1 <= limit <= 20 or returned > limit or returned > total:
        raise ValueError("PubMed coverage counts exceed the bounded hypothesis-search request")
    if fetched != len(fetched_pmids) or fetched > returned:
        raise ValueError("PubMed fetched-record count does not match retained paper records")
    missing = coverage.get("missing_record_pmids")
    if (
        not isinstance(missing, list)
        or len(missing) > limit
        or any(not isinstance(pmid, str) or not pmid.isascii() or not pmid.isdecimal() or not 1 <= len(pmid) <= 20 for pmid in missing)
        or len(missing) != len(set(missing))
        or len(missing) != returned - fetched
        or set(missing) & set(fetched_pmids)
    ):
        raise ValueError("PubMed missing-record identifiers do not match fetch coverage counts")
    search_truncated = coverage.get("search_truncated")
    fetch_incomplete = coverage.get("fetch_incomplete")
    retrieval_incomplete = coverage.get("retrieval_incomplete")
    expected_truncated = total > returned
    expected_fetch_incomplete = bool(missing)
    expected_incomplete = expected_truncated or expected_fetch_incomplete
    if (
        not isinstance(search_truncated, bool)
        or search_truncated != expected_truncated
        or not isinstance(fetch_incomplete, bool)
        or fetch_incomplete != expected_fetch_incomplete
        or not isinstance(retrieval_incomplete, bool)
        or retrieval_incomplete != expected_incomplete
    ):
        raise ValueError("PubMed coverage flags conflict with provider counts or missing records")
    return {
        "status": "retrieval_incomplete" if expected_incomplete else "retrieved_page_complete",
        "search_truncated": expected_truncated,
        "fetch_incomplete": expected_fetch_incomplete,
        "source_complete": False,
        "query_strategy_complete": False,
        "novelty_assessable": False,
        "interpretation_limit": "Completeness applies only to this exact bounded PubMed query page; it does not establish query-strategy completeness, semantic relevance, or literature-wide novelty.",
    }


def bounded_paper_record(paper: dict[str, Any]) -> dict[str, Any]:
    """Retain bibliographic metadata and a digest, omitting the abstract body."""
    if not isinstance(paper, dict):
        raise ValueError("PubMed paper must be a JSON object")
    provenance = paper.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "PubMed":
        raise ValueError("PubMed paper source provenance is missing")
    source_id = str(provenance.get("source_id", ""))
    if not source_id.isdigit():
        raise ValueError("PubMed paper source PMID is invalid")
    canonical = json.dumps(paper, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "pmid": source_id,
        "title": str(provenance.get("title", ""))[:1000],
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{source_id}/",
        "journal": str(paper.get("journal", ""))[:256],
        "publication_date": str(paper.get("publication_date") or "")[:64],
        "publication_types": [str(value)[:128] for value in paper.get("publication_types", [])[:20] if isinstance(value, str)] if isinstance(paper.get("publication_types"), list) else [],
        "doi": paper.get("identifiers", {}).get("doi") if isinstance(paper.get("identifiers"), dict) else None,
        "pmcid": paper.get("identifiers", {}).get("pmc") if isinstance(paper.get("identifiers"), dict) else None,
        "record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "abstract_retained": False,
        "source": "PubMed",
        "interpretation_limit": "Bibliographic discovery lead only; semantic relevance and novelty are not assessed.",
    }
