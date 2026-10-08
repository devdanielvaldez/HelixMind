"""Conservative PubMed query construction for candidate drug-pair literature."""
from __future__ import annotations

import re
import hashlib

from medai_science.sources.safety import ensure_public_research_query


def build_interaction_literature_query(ingredient_a: str, ingredient_b: str) -> str:
    """Build an exact phrase screen from already resolved public ingredient names."""
    terms = []
    for value in (ingredient_a, ingredient_b):
        value = ensure_public_research_query(value.strip())
        if not value or len(value) > 128:
            raise ValueError("resolved ingredient name must contain 1..128 characters")
        # Keep names as phrases, but remove PubMed field/query operators and
        # punctuation that could change the meaning of the expression.
        phrase = " ".join(re.findall(r"[A-Za-z0-9]+(?:[-+][A-Za-z0-9]+)*", value))
        if not phrase:
            raise ValueError("resolved ingredient name has no searchable tokens")
        terms.append(phrase)
    if terms[0].casefold() == terms[1].casefold():
        raise ValueError("two distinct resolved ingredient names are required")
    return f'"{terms[0]}"[Title/Abstract] AND "{terms[1]}"[Title/Abstract]'


def build_interaction_evidence_query(names_a: list[str], names_b: list[str]) -> str:
    """Search pair co-mentions alongside explicit interaction/PK/PD study terms."""
    def group(values: list[str]) -> str:
        if not isinstance(values, list):
            raise ValueError("RxNorm ingredient names must be lists")
        phrases = []
        for value in values[:6]:
            if not isinstance(value, str):
                continue
            value = ensure_public_research_query(value.strip())
            if not value or len(value) > 128:
                continue
            phrase = " ".join(re.findall(r"[A-Za-z0-9]+(?:[-+][A-Za-z0-9]+)*", value))
            if phrase and phrase.casefold() not in {item.casefold() for item in phrases}:
                phrases.append(phrase)
        if not phrases:
            raise ValueError("each RxNorm ingredient requires at least one searchable name")
        return "(" + " OR ".join(f'"{phrase}"[Title/Abstract]' for phrase in phrases) + ")"

    left, right = group(names_a), group(names_b)
    if left.casefold() == right.casefold():
        raise ValueError("two distinct RxNorm ingredients are required")
    endpoints = (
        '"drug interaction"[Title/Abstract] OR "drug-drug interaction"[Title/Abstract] '
        'OR pharmacokinetic*[Title/Abstract] OR pharmacodynamic*[Title/Abstract] '
        'OR coadministration[Title/Abstract] OR co-medication[Title/Abstract] '
        'OR AUC[Title/Abstract] OR "maximum concentration"[Title/Abstract]'
    )
    return f"{left} AND {right} AND ({endpoints})"


def retain_candidate_paper(paper: object) -> dict:
    """Convert a PubMed model or JSON record into a compact, provenance-bearing candidate."""
    if hasattr(paper, "model_dump"):
        row = paper.model_dump(mode="json")
    elif isinstance(paper, dict):
        row = paper
    else:
        raise ValueError("PubMed candidate must be a paper model or JSON object")
    provenance = row.get("provenance", {}) if isinstance(row, dict) else {}
    if hasattr(provenance, "model_dump"):
        provenance = provenance.model_dump(mode="json")
    if not isinstance(provenance, dict):
        raise ValueError("PubMed candidate lacks source provenance")
    pmid = str(provenance.get("source_id", ""))
    if not pmid.isdigit():
        raise ValueError("PubMed candidate lacks a numeric PMID")
    identifiers = row.get("identifiers", {})
    if not isinstance(identifiers, dict):
        identifiers = {}
    pmcid = identifiers.get("pmc")
    pmcid = pmcid.upper() if isinstance(pmcid, str) and re.fullmatch(r"PMC[1-9][0-9]*", pmcid, re.IGNORECASE) else None
    return {
        "pmid": pmid,
        "title": str(provenance.get("title", ""))[:4000],
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "abstract": str(row.get("abstract", ""))[:30_000],
        "authors": [str(item)[:512] for item in row.get("authors", [])[:50] if isinstance(item, str)] if isinstance(row.get("authors"), list) else [],
        "publication_date": row.get("publication_date") if isinstance(row.get("publication_date"), str) else None,
        "publication_types": [str(item)[:256] for item in row.get("publication_types", [])[:30] if isinstance(item, str)] if isinstance(row.get("publication_types"), list) else [],
        "publication_type_ids": [str(item)[:32] for item in row.get("publication_type_ids", [])[:30] if isinstance(item, str)] if isinstance(row.get("publication_type_ids"), list) else [],
        "mesh_headings": [
            {"descriptor": str(item.get("descriptor", ""))[:256], "descriptor_ui": str(item.get("descriptor_ui", ""))[:32], "qualifiers": [
                {"name": str(qualifier.get("name", ""))[:128], "ui": str(qualifier.get("ui", ""))[:32]}
                for qualifier in item.get("qualifiers", [])[:8] if isinstance(qualifier, dict)
            ]}
            for item in row.get("mesh_headings", [])[:50] if isinstance(item, dict)
        ] if isinstance(row.get("mesh_headings"), list) else [],
        "journal": row.get("journal")[:1000] if isinstance(row.get("journal"), str) else None,
        "identifiers": {**{key: value for key, value in identifiers.items() if key == "doi" and isinstance(value, str) and len(value) <= 256}, **({"pmc": pmcid} if pmcid else {})},
        "source": "PubMed",
        "source_record_kind": "bibliographic literature candidate; query co-mentions both resolved ingredient names",
        "interpretation_limit": "Retrieval is a title/abstract co-mention screen. It does not establish a pharmacokinetic or pharmacodynamic interaction, causality, independence of underlying evidence, or clinical relevance.",
    }


def characterize_interaction_candidate(candidate: dict) -> dict:
    """Expose indexed study-type and endpoint-term signals without grading evidence."""
    publication_types = candidate.get("publication_types", [])
    allowed_types = {
        "clinical trial", "randomized controlled trial", "observational study",
        "case reports", "systematic review", "meta-analysis", "review",
        "comparative study", "multicenter study",
    }
    type_signals = sorted({value.casefold() for value in publication_types if isinstance(value, str) and value.casefold() in allowed_types}) if isinstance(publication_types, list) else []
    text = " ".join(str(candidate.get(key, "")) for key in ("title", "abstract")).casefold()
    endpoint_terms = [term for term in ("auc", "cmax", "maximum concentration", "clearance", "half-life", "pharmacokinetic", "pharmacodynamic", "inr", "bleeding", "exposure", "concentration") if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text)]
    return {
        "publication_type_signals": type_signals,
        "pkpd_or_outcome_terms": endpoint_terms,
        "mesh_headings": candidate.get("mesh_headings", []),
        "review_required": True,
        "interpretation_limit": "PubMed publication types, MeSH headings, and text terms are retrieval signals only. They do not establish that both drugs were administered, study independence, a comparator, a causal interaction, direction, magnitude, or clinical relevance.",
    }


def candidate_retention_receipt(candidate: dict) -> dict:
    """Build a compact PubMed receipt without retaining potentially copyrighted abstract text."""
    abstract = candidate.get("abstract", "")
    abstract = abstract if isinstance(abstract, str) else ""
    allowed_fields = {
        "pmid", "title", "url", "authors", "publication_date", "publication_types",
        "publication_type_ids", "mesh_headings", "journal", "identifiers", "source",
        "source_record_kind", "interpretation_limit", "evidence_screen",
    }
    receipt = {key: candidate[key] for key in allowed_fields if key in candidate}
    receipt["abstract_retention"] = {
        "retained": False,
        "sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
        "characters": len(abstract),
        "license_status": "undetermined",
        "reason": "PubMed record availability does not establish reuse rights; abstract used only during transient screening.",
    }
    return receipt
