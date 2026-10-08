"""Conservative checks before sending research text to public biomedical sources."""
import re

_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?\d[\s().-]*){10,15}\b"),
    re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)\d{2}\b"),
    re.compile(r"\b(?:MRN|medical\s+record\s+(?:number|id)|patient\s+(?:id|identifier|number|record))\s*[:#-]?\s*[A-Z0-9-]{3,}\b", re.IGNORECASE),
    re.compile(r"\b(?:my|our|this|specific)\s+patient\b", re.IGNORECASE),
)


def ensure_public_research_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("query cannot be empty")
    if len(normalized) > 2000:
        raise ValueError("public-source query cannot exceed 2000 characters")
    if any(pattern.search(normalized) for pattern in _PATTERNS):
        raise ValueError(
            "query appears to contain personal or patient-identifying information; redact it before using public sources"
        )
    return normalized
