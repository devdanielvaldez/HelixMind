"""Source-anchored lexical leads from PMC BioC drug-pair full text."""
from __future__ import annotations

import hashlib
import re
from typing import Any


_PMCID = re.compile(r"^PMC[1-9][0-9]*$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHARMACOLOGY_TERMS = re.compile(
    r"\b(?:pharmacokinetic\w*|pharmacodynamic\w*|coadminist\w*|co-medication|"
    r"cross-?over|randomi[sz]ed|placebo|comparator|control group|dose|dosage|"
    r"mg|microgram\w*|route|oral|intraven\w*|plasma|exposure|clearance|"
    r"auc|cmax|endpoint|bioavailability|concentration|interaction)\b",
    re.IGNORECASE,
)
_REVIEW_SECTIONS = {"methods", "results", "limitations", "discussion", "conclusions"}


def validate_interaction_fulltext_binding(
    literature_screen: dict[str, Any], candidate: dict[str, Any], candidate_source_id: str,
) -> dict[str, Any]:
    """Require exact membership in the persisted resolved drug-pair candidate set."""
    if not isinstance(literature_screen, dict) or literature_screen.get("status") != "retrieved":
        raise ValueError("selected literature screen has no completed pair-specific PubMed search")
    query_sha256 = literature_screen.get("query_sha256")
    if not isinstance(query_sha256, str) or not _SHA256.fullmatch(query_sha256):
        raise ValueError("selected literature screen has no valid pair-specific query digest")
    if not isinstance(candidate, dict) or candidate.get("source") != "PubMed":
        raise ValueError("selected source record is not a PubMed candidate")
    pmid = candidate.get("pmid")
    if not isinstance(pmid, str) or not pmid.isascii() or not pmid.isdecimal() or not 1 <= len(pmid) <= 20 or pmid != candidate_source_id:
        raise ValueError("PubMed candidate identity does not match its retained artifact")
    pair_pmids = literature_screen.get("pmids")
    if not isinstance(pair_pmids, list) or pmid not in pair_pmids:
        raise ValueError("selected PubMed candidate is not linked to this exact drug-pair search")
    identifiers = candidate.get("identifiers")
    pmcid = identifiers.get("pmc") if isinstance(identifiers, dict) else None
    if not isinstance(pmcid, str) or not _PMCID.fullmatch(pmcid) or len(pmcid) > 20:
        raise ValueError("selected PubMed candidate has no valid linked PMC full-text record")
    ingredient_names = literature_screen.get("resolved_ingredient_names")
    if not isinstance(ingredient_names, list) or len(ingredient_names) != 2:
        raise ValueError("pair-specific resolved ingredient names are absent from the source search record")
    _name_patterns(ingredient_names[0], "ingredient_names_a")
    _name_patterns(ingredient_names[1], "ingredient_names_b")
    return {"pmid": pmid, "pmcid": pmcid, "ingredient_names_a": ingredient_names[0], "ingredient_names_b": ingredient_names[1]}


def rehydrate_interaction_fulltext_candidates(
    screen: dict[str, Any], article: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recheck retained sentence receipts against freshly fetched BioC text."""
    if not isinstance(screen, dict) or not isinstance(article, dict):
        raise ValueError("PMC screen and fetched article must be objects")
    source_hash = screen.get("source_sha256")
    if article.get("source") != "PMC BioC" or article.get("source_sha256") != source_hash:
        raise ValueError("refetched PMC BioC digest differs from the retained full-text screen")
    sections = article.get("sections")
    candidates = screen.get("candidates")
    if not isinstance(sections, list) or not isinstance(candidates, list) or len(candidates) > 20:
        raise ValueError("PMC screen candidates or fetched sections are malformed")
    output = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        offset, length, passage_index = (candidate.get(key) for key in ("source_offset", "source_length", "passage_index"))
        candidate_id = candidate.get("candidate_id")
        text_hash = candidate.get("text_sha256")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (offset, length, passage_index)) or offset < 0 or not 1 <= length <= 1200:
            continue
        if not isinstance(candidate_id, str) or not isinstance(text_hash, str) or not _SHA256.fullmatch(text_hash):
            continue
        section = next((row for row in sections if isinstance(row, dict) and row.get("passage_index") == passage_index and row.get("section") == candidate.get("section") and isinstance(row.get("offset"), int)), None)
        if section is None:
            continue
        local_offset = offset - section["offset"]
        source_text = section.get("text")
        sentence = source_text[local_offset:local_offset + length] if isinstance(source_text, str) and local_offset >= 0 else ""
        expected_id = hashlib.sha256(f"{source_hash}\0{offset}\0{text_hash}".encode()).hexdigest()
        if len(sentence) != length or hashlib.sha256(sentence.encode("utf-8")).hexdigest() != text_hash or candidate_id != expected_id:
            continue
        output.append({
            "candidate_id": candidate_id, "offset": offset, "length": length,
            "text": sentence, "text_sha256": text_hash, "passage_index": passage_index,
            "section": candidate.get("section"), "source_section": candidate.get("source_section"),
            "matched_ingredient_names_a": candidate.get("matched_ingredient_names_a", []),
            "matched_ingredient_names_b": candidate.get("matched_ingredient_names_b", []),
        })
    return output


def _name_patterns(values: list[str], label: str) -> list[tuple[str, re.Pattern[str]]]:
    if not isinstance(values, list) or not 1 <= len(values) <= 6:
        raise ValueError(f"{label} must contain 1..6 resolved ingredient names")
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 128:
            raise ValueError(f"{label} contains an invalid ingredient name")
        name = " ".join(value.split())
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append((name, re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)))
    return result


def build_interaction_fulltext_screen(
    article: dict[str, Any],
    *,
    pmcid: str,
    pmid: str,
    candidate_artifact_id: str,
    candidate_artifact_sha256: str,
    ingredient_names_a: list[str],
    ingredient_names_b: list[str],
    limit: int = 20,
) -> dict[str, Any]:
    """Create hash/offset review leads without persisting PMC sentence text or a verdict."""
    if not isinstance(article, dict) or article.get("source") != "PMC BioC":
        raise ValueError("full-text screen requires a parsed PMC BioC article")
    if not isinstance(pmcid, str) or not _PMCID.fullmatch(pmcid):
        raise ValueError("full-text screen requires a valid PMC identifier")
    if not isinstance(pmid, str) or not pmid.isascii() or not pmid.isdecimal() or not 1 <= len(pmid) <= 20:
        raise ValueError("full-text screen requires a numeric PMID")
    if not isinstance(candidate_artifact_id, str) or not candidate_artifact_id or len(candidate_artifact_id) > 128:
        raise ValueError("PubMed candidate artifact id is invalid")
    if not isinstance(candidate_artifact_sha256, str) or not _SHA256.fullmatch(candidate_artifact_sha256):
        raise ValueError("PubMed candidate artifact hash is invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("full-text candidate limit must be 1..20")
    names_a = _name_patterns(ingredient_names_a, "ingredient_names_a")
    names_b = _name_patterns(ingredient_names_b, "ingredient_names_b")
    sections = article.get("sections")
    if not isinstance(sections, list) or len(sections) > 30_000:
        raise ValueError("PMC BioC section list is invalid or exceeds 30000 entries")
    source_sha256 = article.get("source_sha256")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError("PMC BioC source hash is invalid")

    candidates = []
    unanchored_match_count = 0
    scan_truncated = False
    anchored_source_sentence_count = 0
    overlong_sentence_count = 0
    for index, section in enumerate(sections):
        if len(candidates) >= 5000:
            scan_truncated = True
            break
        if not isinstance(section, dict) or section.get("section") not in _REVIEW_SECTIONS:
            continue
        text = section.get("text")
        if not isinstance(text, str) or not text or len(text) > 100_000:
            continue
        offset = section.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            if any(pattern.search(text) for _, pattern in names_a + names_b):
                unanchored_match_count += 1
            continue
        # Keep exact sentence spans so later human review can recheck the
        # location against this receipt without retaining licensed full text.
        for match in re.finditer(r"[^.!?\n]+(?:[.!?]+(?=\s|$)|$)", text):
            local_start, local_end = match.span()
            sentence = text[local_start:local_end]
            trim_start = len(sentence) - len(sentence.lstrip())
            trim_end = len(sentence.rstrip())
            local_start += trim_start
            local_end = local_start + max(0, trim_end - trim_start)
            sentence = text[local_start:local_end]
            if not sentence:
                continue
            anchored_source_sentence_count += 1
            if len(sentence) > 1200:
                overlong_sentence_count += 1
                continue
            matched_a = [name for name, pattern in names_a if pattern.search(sentence)]
            matched_b = [name for name, pattern in names_b if pattern.search(sentence)]
            if not matched_a and not matched_b:
                continue
            has_pharmacology_term = bool(_PHARMACOLOGY_TERMS.search(sentence))
            if (not matched_a or not matched_b) and not has_pharmacology_term:
                continue
            source_offset = offset + local_start
            text_sha256 = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
            candidates.append({
                "candidate_id": hashlib.sha256(f"{source_sha256}\0{source_offset}\0{text_sha256}".encode()).hexdigest(),
                "section": section["section"],
                "source_section": section.get("source_section"),
                "passage_index": section.get("passage_index"),
                "sentence_index": section.get("sentence_index"),
                "source_offset": source_offset,
                "source_length": len(sentence),
                "text_sha256": text_sha256,
                "matched_ingredient_names_a": matched_a,
                "matched_ingredient_names_b": matched_b,
                "both_ingredients_in_candidate": bool(matched_a and matched_b),
                "pharmacology_term_present": has_pharmacology_term,
            })
            if len(candidates) >= 5000:
                scan_truncated = True
                break

    candidates.sort(key=lambda row: (
        -(2 * int(row["both_ingredients_in_candidate"]) + int(row["pharmacology_term_present"])),
        row["source_offset"],
        row["candidate_id"],
    ))
    return {
        "schema_version": "1",
        "status": "review_leads_available" if candidates else "no_anchored_lexical_leads",
        "source": "PMC BioC",
        "pmcid": pmcid.upper(),
        "pmid": pmid,
        "candidate_artifact_id": candidate_artifact_id,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "article_title": str(article.get("title", ""))[:1000],
        "source_sha256": source_sha256,
        "source_license": article.get("source_license"),
        "parser_version": article.get("parser_version"),
        "section_count": len(sections),
        "anchored_source_sentence_count": anchored_source_sentence_count,
        "omitted_overlong_sentence_count": overlong_sentence_count,
        "candidate_count": min(len(candidates), limit),
        "omitted_candidate_count": max(0, len(candidates) - limit),
        "candidate_scan_truncated": scan_truncated,
        "unanchored_lexical_match_count": unanchored_match_count,
        "candidates": candidates[:limit],
        "interaction_verdict": None,
        "review_required": True,
        "interpretation_limit": "Exact-name and pharmacology-term matches are unvalidated lexical review leads, not evidence that both drugs were administered, an interaction occurred, or any effect was causal. Review full text for cohort, pair exposure, comparator, dose, route, design, endpoint, direction, magnitude, limitations, and overlap with label evidence. Section labels and offsets are provider-supplied and may be coarse or absent.",
        "retention": "No PMC sentence text is included in this receipt; inspect the source offsets against the retrieved BioC document and verify article-specific reuse terms before further retention or model processing.",
    }
