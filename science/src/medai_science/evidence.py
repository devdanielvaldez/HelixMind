"""Transparent evidence aggregation. Scores remain descriptive, not model confidence."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from medai_science.quality import is_valid_quality_annotation

if TYPE_CHECKING:
    from medai_science.models import Claim


def summarize(claims: list[Claim]) -> dict:
    counts = Counter(c.polarity.value for c in claims)
    annotations = []
    for claim in claims:
        if not is_valid_quality_annotation(claim.evidence_quality):
            raise ValueError("claim quality annotations must be finite values in 0..1")
        annotations.append({
            "source": claim.source.source,
            "source_id": claim.source.source_id,
            "polarity": claim.polarity.value,
            "value": claim.evidence_quality,
        })
    annotations.sort(key=lambda item: (item["source"], item["source_id"], item["polarity"], item["value"]))
    limitations = sorted({lim for c in claims for lim in c.limitations})
    return {
        "claim_count": len(claims),
        "by_polarity": {polarity: counts[polarity] for polarity in ("supports", "contradicts", "context")},
        "source_quality_annotations": annotations,
        "limitations": limitations,
        "interpretation": "Quality values are retained as source/user annotations linked to their original records. They are not evidence weights, calibrated probabilities, risk-of-bias assessments, or clinical confidence.",
    }
