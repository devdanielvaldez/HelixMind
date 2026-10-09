"""Validation primitives for descriptive evidence-quality annotations."""
import math
import re


_PUBMED_PUBLICATION_TYPE_DESIGNS = {
    "case reports": "case_report",
    "clinical trial": "clinical_trial",
    "clinical trial, phase i": "phase_i_clinical_trial",
    "clinical trial, phase ii": "phase_ii_clinical_trial",
    "clinical trial, phase iii": "phase_iii_clinical_trial",
    "clinical trial, phase iv": "phase_iv_clinical_trial",
    "controlled clinical trial": "controlled_clinical_trial",
    "meta-analysis": "meta_analysis",
    "observational study": "observational_study",
    "randomized controlled trial": "randomized_controlled_trial",
    "systematic review": "systematic_review",
}
_PUBMED_PUBLICATION_TYPE_UI_DESIGNS = {
    "D002363": "case_report",
    "D016430": "clinical_trial",
    "D016449": "randomized_controlled_trial",
    "D017418": "meta_analysis",
    "D017426": "phase_i_clinical_trial",
    "D017427": "phase_ii_clinical_trial",
    "D017428": "phase_iii_clinical_trial",
    "D017429": "phase_iv_clinical_trial",
    "D018848": "controlled_clinical_trial",
    "D064888": "observational_study",
    "D000078182": "systematic_review",
}
QUALITY_DOMAINS = ("risk_of_bias", "precision", "directness", "consistency", "reporting_bias")

_PUBMED_METADATA_FIELDS = {
    "title": ("provenance", "title", "text"),
    "publication_date": (None, "publication_date", "text"),
    "journal": (None, "journal", "text"),
    "authors": (None, "authors", "list"),
    "doi": ("identifiers", "doi", "text"),
    "pmcid": ("identifiers", "pmc", "text"),
    "abstract_length": (None, "abstract_length", "length"),
    "publication_types": (None, "publication_types", "list"),
    "publication_type_ids": (None, "publication_type_ids", "list"),
    "mesh_headings": (None, "mesh_headings", "list"),
}


def _pubmed_metadata_coverage(records: list[dict]) -> dict:
    """Count captured, empty, and malformed source fields without grading papers."""
    counts = {
        name: {"nonempty": 0, "empty": 0, "not_captured": 0, "malformed": 0}
        for name in _PUBMED_METADATA_FIELDS
    }
    record_count = 0
    for paper in records:
        if not isinstance(paper, dict):
            continue
        provenance = paper.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("source") != "PubMed":
            continue
        record_count += 1
        for name, (parent_key, field, kind) in _PUBMED_METADATA_FIELDS.items():
            owner = paper.get(parent_key) if parent_key else paper
            if not isinstance(owner, dict):
                counts[name]["malformed" if parent_key in paper else "not_captured"] += 1
                continue
            if field not in owner:
                counts[name]["empty" if parent_key == "identifiers" else "not_captured"] += 1
                continue
            value = owner[field]
            if kind == "text":
                if value is None or value == "":
                    state = "empty"
                elif isinstance(value, str) and len(value) <= 100_000:
                    state = "nonempty"
                else:
                    state = "malformed"
            elif kind == "list":
                if not isinstance(value, list) or len(value) > 1000:
                    state = "malformed"
                elif any(
                    not isinstance(item, dict if field == "mesh_headings" else str)
                    for item in value
                ):
                    state = "malformed"
                else:
                    state = "nonempty" if value else "empty"
            else:
                if value is None:
                    state = "empty"
                elif isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000:
                    state = "nonempty" if value > 0 else "empty"
                else:
                    state = "malformed"
            counts[name][state] += 1
    return {
        "methodology": "pubmed-source-metadata-coverage-v1",
        "records": record_count,
        "fields": counts,
        "interpretation": "Counts describe whether bounded PubMed source metadata was retained. Missing or empty fields can reflect provider indexing or retrieval; availability is not study quality, evidence strength, or paper validity.",
    }


def is_valid_quality_annotation(value: object) -> bool:
    """Accept finite numeric annotations in [0, 1], excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def classify_pubmed_design_metadata(
    publication_types: object,
    mesh_headings: object,
    publication_type_ids: object = (),
) -> dict:
    """Describe only exact PubMed publication-type signals; never grade evidence quality."""
    if not isinstance(publication_types, list) or len(publication_types) > 100:
        raise ValueError("PubMed publication types must be an array of at most 100 labels")
    if not isinstance(mesh_headings, list) or len(mesh_headings) > 200:
        raise ValueError("PubMed MeSH headings must be an array of at most 200 records")
    if any(not isinstance(value, str) or len(value) > 256 for value in publication_types):
        raise ValueError("PubMed publication-type labels must be strings no longer than 256 characters")
    if not isinstance(publication_type_ids, (list, tuple)) or len(publication_type_ids) > 100:
        raise ValueError("PubMed publication-type identifiers must be an array of at most 100 values")
    if any(
        not isinstance(value, str)
        or len(value) > 16
        or not re.fullmatch(r"D[0-9]{6,9}", value)
        for value in publication_type_ids
    ):
        raise ValueError("PubMed publication-type identifiers must be valid MeSH UI strings")
    if any(not isinstance(value, dict) for value in mesh_headings):
        raise ValueError("PubMed MeSH heading records must be objects")
    signals = []
    for label in publication_types:
        normalized_label = label.strip().casefold()
        category = _PUBMED_PUBLICATION_TYPE_DESIGNS.get(normalized_label)
        if category:
            signals.append({"category": category, "source_field": "publication_types", "source_term": label, "_normalized_source_term": normalized_label})
    for identifier in publication_type_ids:
        category = _PUBMED_PUBLICATION_TYPE_UI_DESIGNS.get(identifier)
        if category:
            signals.append({"category": category, "source_field": "publication_type_ids", "source_term": identifier, "source_url": f"https://meshb.nlm.nih.gov/record/ui?ui={identifier}", "_normalized_source_term": identifier})
    unique = {
        (item["category"], item["source_field"], item["_normalized_source_term"]): item
        for item in signals
    }
    signals = [{key: value for key, value in item.items() if key != "_normalized_source_term"} for item in unique.values()]
    descriptors = sorted({
        str(heading.get("descriptor", "")).strip()
        for heading in mesh_headings
        if isinstance(heading.get("descriptor"), str) and heading["descriptor"].strip()
    })
    return {
        "methodology": "pubmed-publication-type-exact-v2",
        "design_categories": sorted({item["category"] for item in unique.values()}),
        "publication_type_signals": sorted(signals, key=lambda item: (item["category"], item["source_term"].casefold())),
        "mesh_descriptor_context": descriptors,
        "evidence_grade": None,
        "quality_domains": {domain: "not_assessed" for domain in QUALITY_DOMAINS},
        "interpretation": "Publication-type labels are provider-supplied indexing metadata and may be missing or incomplete. MeSH descriptors are retained as topic context, not treated as study design. No risk-of-bias assessment, evidence grade, or clinical confidence is assigned.",
    }


def summarize_pubmed_design_metadata(records: object) -> dict:
    """Aggregate PubMed publication-type signals without ranking or grading records."""
    if not isinstance(records, list):
        raise ValueError("PubMed design inventory records must be an array")
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    truncated: dict[str, bool] = {}
    records_without_signal = 0
    malformed_records = 0
    malformed_related_metadata = 0
    pubmed_records = 0
    related_publication_counts: dict[str, int] = {}
    related_publication_examples: dict[str, list[str]] = {}
    related_publication_links = 0
    for paper in records:
        if not isinstance(paper, dict):
            malformed_records += 1
            continue
        provenance = paper.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("source") != "PubMed":
            continue
        pubmed_records += 1
        try:
            design = classify_pubmed_design_metadata(
                paper.get("publication_types", []),
                paper.get("mesh_headings", []),
                paper.get("publication_type_ids", []),
            )
        except ValueError:
            malformed_records += 1
            continue
        related = paper.get("related_publications", [])
        if not isinstance(related, list) or len(related) > 100 or any(
            not isinstance(link, dict)
            or not isinstance(link.get("ref_type"), str)
            or not 1 <= len(link["ref_type"]) <= 64
            or not isinstance(link.get("pmid"), str)
            or not link["pmid"].isascii()
            or not link["pmid"].isdecimal()
            or len(link["pmid"]) > 20
            for link in related
        ):
            malformed_related_metadata += 1
        else:
            for link in related:
                related_publication_links += 1
                kind = link["ref_type"]
                related_publication_counts[kind] = related_publication_counts.get(kind, 0) + 1
                examples_for_kind = related_publication_examples.setdefault(kind, [])
                if link["pmid"] not in examples_for_kind and len(examples_for_kind) < 100:
                    examples_for_kind.append(link["pmid"])
        categories = design["design_categories"]
        if not categories:
            records_without_signal += 1
        pmid = provenance.get("source_id")
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
            category_examples = examples.setdefault(category, [])
            if isinstance(pmid, str) and pmid.isdigit() and pmid not in category_examples:
                if len(category_examples) < 100:
                    category_examples.append(pmid)
                else:
                    truncated[category] = True
    return {
        "methodology": "pubmed-publication-type-exact-v2",
        "pubmed_records": pubmed_records,
        "counts_by_provider_publication_type_signal": counts,
        "example_pmids_by_signal": examples,
        "example_pmids_truncated_by_signal": truncated,
        "related_publication_links": related_publication_links,
        "counts_by_pubmed_related_publication_type": related_publication_counts,
        "example_pmids_by_related_publication_type": related_publication_examples,
        "source_metadata_coverage": _pubmed_metadata_coverage(records),
        "records_without_recognized_design_signal": records_without_signal,
        "records_with_malformed_design_metadata": malformed_records,
        "records_with_malformed_related_publication_metadata": malformed_related_metadata,
        "categories_can_overlap": True,
        "evidence_grade": None,
        "quality_domains": {domain: "not_assessed" for domain in QUALITY_DOMAINS},
        "interpretation": "Exact PubMed publication-type metadata is summarized as descriptive design signals. PubMed related-publication links are counted by source RefType and do not independently establish article validity, retraction scope, or evidence quality. Malformed related-publication metadata is reported separately from malformed design metadata. MeSH descriptors remain topic context; this inventory does not infer design from them, assign quality, or establish evidence certainty.",
    }
