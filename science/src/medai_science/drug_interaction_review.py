"""Source-anchored candidate selection for human review of drug interaction claims."""
from __future__ import annotations

import hashlib
import re


MAX_SENTENCE_CHARS = 1200
MAX_SOURCE_SENTENCES = 100
MAX_REVIEW_PAIRS = 10
_STOP_WORDS = {"about", "after", "against", "among", "also", "and", "are", "been", "before", "between", "both", "but", "can", "could", "did", "during", "each", "for", "from", "had", "has", "have", "into", "its", "may", "more", "most", "not", "one", "other", "our", "out", "over", "per", "same", "should", "since", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "under", "use", "used", "using", "was", "were", "what", "when", "where", "which", "while", "with", "within", "without", "would"}


def sentence_candidates(text: str, prefix: str) -> list[dict]:
    """Return exact sentence spans and hashes; overlong passages are omitted."""
    if not isinstance(text, str) or len(text) > 32_000:
        raise ValueError("source text must be a string no longer than 32,000 characters")
    if not re.fullmatch(r"[A-Z]{2,8}", prefix):
        raise ValueError("candidate prefix must be 2..8 uppercase letters")
    spans = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        end = match.start()
        value_start = start
        value_end = end
        while value_start < value_end and text[value_start].isspace():
            value_start += 1
        while value_end > value_start and text[value_end - 1].isspace():
            value_end -= 1
        sentence = text[value_start:value_end]
        if sentence and len(sentence) <= MAX_SENTENCE_CHARS:
            digest = hashlib.sha256(f"{value_start}\0{sentence}".encode("utf-8")).hexdigest()[:16]
            spans.append({"candidate_id": f"{prefix}-{digest}", "offset": value_start, "length": len(sentence), "text": sentence, "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest()})
            if len(spans) >= MAX_SOURCE_SENTENCES:
                return spans
        start = match.end()
    value_start = start
    value_end = len(text)
    while value_start < value_end and text[value_start].isspace():
        value_start += 1
    while value_end > value_start and text[value_end - 1].isspace():
        value_end -= 1
    sentence = text[value_start:value_end]
    if sentence and len(sentence) <= MAX_SENTENCE_CHARS and len(spans) < MAX_SOURCE_SENTENCES:
        digest = hashlib.sha256(f"{value_start}\0{sentence}".encode("utf-8")).hexdigest()[:16]
        spans.append({"candidate_id": f"{prefix}-{digest}", "offset": value_start, "length": len(sentence), "text": sentence, "text_sha256": hashlib.sha256(sentence.encode("utf-8")).hexdigest()})
    return spans


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
