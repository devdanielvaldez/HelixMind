"""Build source searches for graph-path hypotheses without scoring novelty."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from medai_science.sources.safety import ensure_public_research_query


_STATEMENT = re.compile(r"^(.+?) may be connected to (.+?) through (.+?)\.$")


def build_hypothesis_literature_query(candidate: dict[str, Any], max_bytes: int = 256) -> str:
    """Search PubMed for the three named entities in one deterministic graph path."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
        raise ValueError("hypothesis candidate requires a stable ID")
    statement = candidate.get("statement")
    if not isinstance(statement, str):
        raise ValueError("hypothesis candidate requires a deterministic graph-path statement")
    match = _STATEMENT.fullmatch(statement.strip())
    if match is None:
        raise ValueError("hypothesis statement does not match the supported graph-path form")
    phrases = []
    for raw in (match.group(1), match.group(3), match.group(2)):
        phrase = " ".join(re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{0,39}", raw))
        if not phrase or len(phrase) > 80:
            raise ValueError("hypothesis entity label is empty or exceeds 80 normalized characters")
        ensure_public_research_query(phrase)
        phrases.append(phrase)
    if len({phrase.casefold() for phrase in phrases}) < 3:
        raise ValueError("hypothesis search requires three distinct path entity labels")
    query = " AND ".join(f'"{phrase}"[All Fields]' for phrase in phrases)
    if len(query.encode("utf-8")) > max_bytes:
        raise ValueError("hypothesis entity query exceeds the 256-byte search limit")
    return ensure_public_research_query(query)


def build_hypothesis_literature_receipt(
    candidate: dict[str, Any], query: str, search: dict[str, Any], paper_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind bounded bibliographic leads and provider coverage to one candidate."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str):
        raise ValueError("hypothesis candidate requires a stable ID")
    query = ensure_public_research_query(query)
    coverage = search.get("coverage") if isinstance(search.get("coverage"), dict) else {}
    coverage_artifact_id = search.get("coverage_artifact_id")
    papers = search.get("results")
    if not isinstance(papers, list):
        raise ValueError("PubMed search result records are malformed")
    if len(papers) != len(paper_artifacts):
        raise ValueError("PubMed papers and stored candidate receipts are misaligned")
    leads = []
    for paper, artifact in zip(papers, paper_artifacts):
        provenance = paper.get("provenance") if isinstance(paper, dict) else None
        pmid = provenance.get("source_id") if isinstance(provenance, dict) else None
        if not isinstance(pmid, str) or not pmid.isdigit() or not isinstance(artifact, dict):
            raise ValueError("PubMed lead lacks an exact PMID or stored artifact receipt")
        leads.append({"pmid": pmid, "artifact_id": artifact.get("id"), "record_sha256": artifact.get("record_sha256")})
    return {
        "schema_version": 1,
        "hypothesis_id": candidate["id"],
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "provider": "PubMed",
        "coverage": coverage,
        "coverage_artifact_id": coverage_artifact_id,
        "returned_record_count": len(papers),
        "candidate_leads": leads,
        "novelty_assessment": None,
        "review_status": "requires_human_relevance_review",
        "interpretation_limit": "This bounded exact-entity PubMed search is a literature discovery lead, not a source-complete search or semantic relevance judgment. The presence or absence of retrieved records does not establish novelty; novelty remains unscored.",
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
