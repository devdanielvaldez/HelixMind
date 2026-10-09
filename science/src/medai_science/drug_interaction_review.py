"""Source-anchored candidate selection for human review of drug interaction claims."""
from __future__ import annotations

import hashlib
import re


MAX_SENTENCE_CHARS = 1200
MAX_SOURCE_SENTENCES = 100
MAX_REVIEW_PAIRS = 10
MAX_LABEL_SENTENCES = 100
_STOP_WORDS = {"about", "after", "against", "among", "also", "and", "are", "been", "before", "between", "both", "but", "can", "could", "did", "during", "each", "for", "from", "had", "has", "have", "into", "its", "may", "more", "most", "not", "one", "other", "our", "out", "over", "per", "same", "should", "since", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "under", "use", "used", "using", "was", "were", "what", "when", "where", "which", "while", "with", "within", "without", "would"}


def screen_sentence_candidates(text: str, prefix: str) -> dict:
    """Return bounded sentence spans with explicit total and truncation metadata."""
    if not isinstance(text, str) or len(text) > 32_000:
        raise ValueError("source text must be a string no longer than 32,000 characters")
    if not re.fullmatch(r"[A-Z]{2,8}", prefix):
        raise ValueError("candidate prefix must be 2..8 uppercase letters")
    spans = []
    source_sentence_count = 0
    overlong_sentence_count = 0

    def retain(start: int, end: int) -> None:
        nonlocal source_sentence_count, overlong_sentence_count
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        sentence = text[start:end]
        if not sentence:
            return
        source_sentence_count += 1
        if len(sentence) > MAX_SENTENCE_CHARS:
            overlong_sentence_count += 1
            return
        digest = hashlib.sha256(f"{start}\0{sentence}".encode("utf-8")).hexdigest()[:16]
        row = {"candidate_id": f"{prefix}-{digest}", "offset": start, "length": len(sentence), "text": sentence, "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest()}
        if len(spans) < MAX_SOURCE_SENTENCES:
            spans.append(row)

    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        retain(start, match.start())
        start = match.end()
    retain(start, len(text))
    eligible_candidate_count = source_sentence_count - overlong_sentence_count
    return {
        "candidates": spans,
        "source_sentence_count": source_sentence_count,
        "eligible_candidate_count": eligible_candidate_count,
        "retained_candidate_count": len(spans),
        "omitted_overlong_sentence_count": overlong_sentence_count,
        "omitted_by_limit_count": max(0, eligible_candidate_count - len(spans)),
        "truncated": eligible_candidate_count > len(spans) or overlong_sentence_count > 0,
    }


def sentence_candidates(text: str, prefix: str) -> list[dict]:
    """Compatibility helper returning retained exact sentence spans and hashes."""
    return screen_sentence_candidates(text, prefix)["candidates"]


def screen_drug_label_candidates(labels: dict, limit: int = MAX_LABEL_SENTENCES) -> dict:
    """Collect bounded label interaction spans and account for every omission."""
    if not isinstance(labels, dict) or len(labels) > 32:
        raise ValueError("drug label inventory must be an object with at most 32 ingredients")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LABEL_SENTENCES:
        raise ValueError(f"label candidate limit must be 1..{MAX_LABEL_SENTENCES}")
    candidates = []
    source_sentence_count = 0
    eligible_candidate_count = 0
    omitted_overlong_sentence_count = 0
    any_source_truncated = False
    for drug_name, rows in labels.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("drug_interactions"), str):
                continue
            coverage = screen_sentence_candidates(row["drug_interactions"], "LABEL")
            source_sentence_count += coverage["source_sentence_count"]
            eligible_candidate_count += coverage["eligible_candidate_count"]
            omitted_overlong_sentence_count += coverage["omitted_overlong_sentence_count"]
            any_source_truncated = any_source_truncated or coverage["truncated"]
            for span in coverage["candidates"]:
                if len(candidates) >= limit:
                    continue
                candidates.append({
                    **span,
                    "drug_name": str(drug_name)[:128],
                    "set_id": row.get("set_id"),
                    "version": row.get("version") or row.get("spl_version"),
                    "effective_time": row.get("effective_time") or row.get("published_date"),
                    "product_name": (row.get("generic_name") or row.get("brand_name") or [None])[0]
                    if isinstance(row.get("generic_name") or row.get("brand_name") or [], list) else None,
                })
    omitted_by_limit_count = max(0, eligible_candidate_count - len(candidates))
    return {
        "candidates": candidates,
        "coverage": {
            "source_sentence_count": source_sentence_count,
            "eligible_candidate_count": eligible_candidate_count,
            "retained_candidate_count": len(candidates),
            "omitted_overlong_sentence_count": omitted_overlong_sentence_count,
            "omitted_by_limit_count": omitted_by_limit_count,
            "truncated": any_source_truncated or omitted_by_limit_count > 0,
        },
    }


def rank_candidate_pairs(labels: list[dict], papers: list[dict], limit: int = 10) -> list[dict]:
    """Prioritize sentence pairs sharing lexical content; rank is only a screen."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_REVIEW_PAIRS:
        raise ValueError("review pair limit must be 1..10")
    if len(labels) > MAX_SOURCE_SENTENCES or len(papers) > MAX_SOURCE_SENTENCES:
        raise ValueError("candidate source sentence count exceeds 100")
    pairs = []
    for label in labels:
        left = {term for term in re.findall(r"[a-z0-9]+", label["text"].casefold()) if len(term) > 2 and term not in _STOP_WORDS}
        for paper in papers:
            right = {term for term in re.findall(r"[a-z0-9]+", paper["text"].casefold()) if len(term) > 2 and term not in _STOP_WORDS}
            overlap = left & right
            union = left | right
            similarity = len(overlap) / len(union) if union else 0.0
            pairs.append({"label": label, "paper": paper, "shared_terms": sorted(overlap)[:20], "lexical_overlap": round(similarity, 6)})
    pairs.sort(key=lambda item: (-item["lexical_overlap"], item["label"]["offset"], item["paper"]["offset"]))
    selected = pairs[:limit]
    for index, pair in enumerate(selected, 1):
        pair["screen_rank"] = index
    return selected
