"""Deterministic, bounded source-query construction for contradiction review leads."""
from __future__ import annotations

import re
from typing import Any

from medai_science.sources.safety import ensure_public_research_query


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{1,39}")
_STOP = {
    "about", "after", "among", "associated", "association", "because", "between", "could", "during",
    "effect", "effects", "finding", "findings", "from", "have", "into", "may", "might", "more",
    "observed", "over", "study", "that", "their", "these", "this", "through", "using", "were",
    "which", "with", "within", "without", "would",
}
_REFUTATION_TERMS = '(contradict* OR inconsistent OR "no association" OR null)'


def select_contradiction_claims(evidence: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    """Choose stable source-linked supporting claims, preferring distinct source records."""
    if not 1 <= limit <= 20:
        raise ValueError("contradiction search claim limit must be 1..20")
    candidates = []
    for row in evidence:
        if not isinstance(row, dict) or row.get("polarity") != "supports":
            continue
        claim, source, source_id, evidence_id = row.get("claim"), row.get("source"), row.get("source_id"), row.get("id")
        if not all(isinstance(value, str) and value.strip() for value in (claim, source, source_id, evidence_id)):
            continue
        candidates.append(row)
    candidates.sort(key=lambda row: (row["source"].casefold(), row["source_id"], row["id"]))
    deduplicated = []
    seen_claims: set[str] = set()
    for row in candidates:
        normalized = " ".join(row["claim"].casefold().split())
        if normalized in seen_claims:
            continue
        seen_claims.add(normalized)
        deduplicated.append(row)
    candidates = deduplicated

    selected = []
    selected_ids = set()
    source_records = set()
    for row in candidates:
        source_key = (row["source"].casefold(), row["source_id"])
        if source_key not in source_records:
            selected.append(row)
            selected_ids.add(row["id"])
            source_records.add(source_key)
            if len(selected) >= limit:
                return selected
    for row in candidates:
        if row["id"] in selected_ids:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def build_contradiction_query(claim: str, max_bytes: int = 256) -> str:
    """Build a compact PubMed query whose exact bounded text is safe to hash and submit."""
    ensure_public_research_query(claim)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 128 <= max_bytes <= 2000:
        raise ValueError("contradiction query byte limit must be 128..2000")
    tokens = []
    seen = set()
    for token in _TOKEN.findall(claim):
        normalized = token.casefold()
        if normalized in _STOP or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(token)
        if len(tokens) >= 6:
            break
    if not tokens:
        raise ValueError("supporting claim contains no searchable biomedical terms")
    query = f"({' AND '.join(tokens)}) AND {_REFUTATION_TERMS}"
    while len(query.encode("utf-8")) > max_bytes and len(tokens) > 1:
        tokens.pop()
        query = f"({' AND '.join(tokens)}) AND {_REFUTATION_TERMS}"
    if len(query.encode("utf-8")) > max_bytes:
        raise ValueError("supporting claim terms cannot fit within the configured PubMed query bound")
    return ensure_public_research_query(query)
