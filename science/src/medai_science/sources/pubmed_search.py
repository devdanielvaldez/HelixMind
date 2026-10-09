"""Validate PubMed ESearch result counts and report bounded retrieval coverage."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any


def _strict_json_loads(raw: bytes) -> Any:
    """Reject duplicate object keys and non-standard numbers in provider JSON."""
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"PubMed ESearch contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"PubMed ESearch contains a non-standard JSON number: {value}")

    return json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)


def _pmid(value: Any) -> bool:
    return isinstance(value, str) and value.isascii() and value.isdecimal() and 1 <= len(value) <= 20


def validate_pmid_snapshot(value: Any, *, max_items: int = 200) -> list[str]:
    """Validate a persisted PMID snapshot before using it to suppress discoveries."""
    if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= 200:
        raise ValueError("PubMed snapshot item limit must be 1..200")
    if value is None:
        return []
    if (
        not isinstance(value, list)
        or len(value) > max_items
        or any(not _pmid(pmid) for pmid in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"PubMed snapshot must contain at most {max_items} unique ASCII PMID identifiers")
    return value


def parse_related_publications(article: ET.Element) -> list[dict[str, str]]:
    """Retain bounded PubMed CommentsCorrections links without interpreting their effect."""
    related: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for relation in article.findall(".//CommentsCorrectionsList/CommentsCorrections")[:100]:
        ref_type = relation.attrib.get("RefType", "")
        pmid = (relation.findtext("./PMID") or "").strip()
        identity = (ref_type, pmid)
        if (re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", ref_type)
                and _pmid(pmid) and identity not in seen):
            related.append({"ref_type": ref_type, "pmid": pmid})
            seen.add(identity)
    return related


def parse_esearch_response(raw: bytes, requested_limit: int) -> tuple[list[str], dict[str, Any]]:
    """Parse an ESearch response without implying that a bounded result list is complete."""
    if isinstance(requested_limit, bool) or not isinstance(requested_limit, int) or not 1 <= requested_limit <= 200:
        raise ValueError("PubMed search limit must be 1..200")
    if not isinstance(raw, bytes) or not raw or len(raw) > 2_000_000:
        raise ValueError("PubMed ESearch JSON must contain 1..2000000 bytes")
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("PubMed ESearch returned invalid JSON") from error
    result = payload.get("esearchresult") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        raise ValueError("PubMed ESearch response has no result object")
    raw_count = result.get("count")
    if not isinstance(raw_count, str) or not raw_count.isascii() or not raw_count.isdecimal() or len(raw_count) > 20:
        raise ValueError("PubMed ESearch returned an invalid total count")
    total_count = int(raw_count)
    identifiers = result.get("idlist")
    if (
        not isinstance(identifiers, list)
        or len(identifiers) > requested_limit
        or any(not _pmid(value) for value in identifiers)
        or len(set(identifiers)) != len(identifiers)
        or len(identifiers) > total_count
    ):
        raise ValueError("PubMed ESearch returned an invalid or over-limit PMID list")
    return identifiers, {
        "total_hits": total_count,
        "requested_limit": requested_limit,
        "returned_pmids": len(identifiers),
        "search_truncated": total_count > len(identifiers),
        "coverage_semantics": "PubMed-reported total count versus the bounded ESearch PMID page; count does not establish relevance or completeness of the search strategy.",
    }


def add_fetch_coverage(coverage: dict[str, Any], requested_pmids: list[str], fetched_pmids: list[str]) -> dict[str, Any]:
    """Record EFetch omissions separately from ESearch truncation."""
    if (
        not isinstance(coverage, dict)
        or not isinstance(requested_pmids, list)
        or not isinstance(fetched_pmids, list)
        or any(not _pmid(value) for value in requested_pmids + fetched_pmids)
    ):
        raise ValueError("PubMed fetch coverage requires validated PMID lists")
    if len(set(requested_pmids)) != len(requested_pmids) or len(set(fetched_pmids)) != len(fetched_pmids):
        raise ValueError("PubMed fetch coverage cannot contain duplicate PMIDs")
    requested = set(requested_pmids)
    fetched = set(fetched_pmids)
    unexpected = fetched - requested
    if unexpected:
        raise ValueError("PubMed EFetch returned PMIDs absent from the ESearch result")
    missing = sorted(requested - fetched, key=int)
    return {
        **coverage,
        "fetched_records": len(fetched_pmids),
        "missing_record_pmids": missing,
        "fetch_incomplete": bool(missing),
        "retrieval_incomplete": bool(coverage.get("search_truncated") or missing),
    }
