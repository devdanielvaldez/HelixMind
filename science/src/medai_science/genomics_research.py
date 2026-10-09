"""Validated, privacy-minimized summaries for locally analyzed VCF reports."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUILDS = {"GRCh37", "GRCh38"}
_IMPLEMENTATION_FILES = {"__init__.py", "genomics.py", "transcripts.py"}
_SO_TERM = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SUMMARY_FIELDS = {
    "schema_version", "source", "source_vcf_sha256", "analysis_implementation_sha256",
    "genome_build", "reference_validated", "reference_fasta_sha256", "records_parsed",
    "alternate_alleles_emitted", "variant_class_counts", "validation_counts",
    "local_transcript_consequence_counts", "local_annotated_gene_identifier_count",
    "sample_identifiers_retained", "raw_variant_records_retained_in_summary",
    "remote_sources_queried", "interpretation_limit",
}
_SUMMARY_INTERPRETATION_LIMIT = (
    "This receipt summarizes a researcher-submitted local file analysis report. The worker does not receive or replay the source VCF, verify the submitted report against the file, query remote variant sources, or interpret pathogenicity/clinical significance."
)


def summarize_local_vcf_report(report: Any, *, expected_source_id: str) -> dict[str, Any]:
    """Validate a submitted local analysis receipt and return only aggregate research metadata."""
    if not isinstance(report, dict):
        raise ValueError("genomic dataset payload must be a local VCF analysis report object")
    digest = report.get("input_sha256")
    if report.get("analysis") != "local VCF parsing and validation" or not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != expected_source_id:
        raise ValueError("genomic dataset source ID must match its locally analyzed VCF SHA-256")
    build = report.get("genome_build")
    if build not in _BUILDS or report.get("genome_build_source") != "explicit_user_selection":
        raise ValueError("genomic dataset requires an explicitly selected GRCh37 or GRCh38 build")
    if report.get("sample_identifiers_retained") is not False or report.get("remote_services_contacted") is not False:
        raise ValueError("genomic report must omit sample identifiers and attest that analysis contacted no remote service")
    if report.get("clinical_interpretation") != "not_performed":
        raise ValueError("local VCF report must not contain clinical interpretation")
    implementation = report.get("analysis_implementation_sha256")
    if not isinstance(implementation, dict) or set(implementation) != _IMPLEMENTATION_FILES:
        raise ValueError("genomic report must identify the exact local analysis implementation files")
    if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in implementation.values()):
        raise ValueError("genomic report contains an invalid implementation hash")
    variants = report.get("variants")
    if not isinstance(variants, list) or len(variants) > 10_000:
        raise ValueError("genomic report must contain at most 10000 parsed alternate alleles")
    emitted = report.get("alternate_alleles_emitted")
    if isinstance(emitted, bool) or not isinstance(emitted, int) or emitted != len(variants):
        raise ValueError("genomic report alternate allele count does not match its records")
    records_parsed = report.get("records_parsed")
    unsupported_reported = report.get("unsupported_symbolic_or_breakend_alleles")
    if (isinstance(records_parsed, bool) or not isinstance(records_parsed, int)
            or not 0 <= records_parsed <= emitted
            or isinstance(unsupported_reported, bool) or not isinstance(unsupported_reported, int)
            or not 0 <= unsupported_reported <= emitted):
        raise ValueError("genomic report record and unsupported-allele counts are invalid")
    reference_validated = report.get("reference_validated")
    reference_sha256 = report.get("reference_fasta_sha256")
    if (not isinstance(reference_validated, bool)
            or (reference_validated and (not isinstance(reference_sha256, str) or not _SHA256.fullmatch(reference_sha256)))
            or (not reference_validated and reference_sha256 is not None)):
        raise ValueError("genomic report has inconsistent reference provenance")

    classes: Counter[str] = Counter()
    validation_counts = {"reference_validated": 0, "reference_unvalidated": 0, "unsupported_symbolic_or_breakend": 0}
    consequences: Counter[str] = Counter()
    annotated_gene_ids: set[str] = set()
    record_alternates: dict[int, list[int]] = {}
    consequence_entries = 0
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"genomic report variant {index} must be an object")
        if not isinstance(variant.get("sequence_allele_supported"), bool):
            raise ValueError(f"genomic report variant {index} lacks an explicit sequence-allele support state")
        record_number, alternate_index = variant.get("record_number"), variant.get("alternate_index")
        if (isinstance(record_number, bool) or not isinstance(record_number, int)
                or not 1 <= record_number <= records_parsed
                or isinstance(alternate_index, bool) or not isinstance(alternate_index, int) or alternate_index < 1):
            raise ValueError(f"genomic report variant {index} has invalid VCF record/alternate identity")
        record_alternates.setdefault(record_number, []).append(alternate_index)
        variant_class = variant.get("variant_class")
        if not isinstance(variant_class, str) or variant_class not in {"SNV", "small_indel_or_mnv", "unsupported_symbolic_or_breakend"}:
            raise ValueError(f"genomic report variant {index} has an unsupported variant class")
        classes[variant_class] += 1
        if variant["sequence_allele_supported"] != (variant_class != "unsupported_symbolic_or_breakend"):
            raise ValueError(f"genomic report variant {index} has inconsistent sequence support and variant class")
        variant_reference_validated = variant.get("reference_validated", False)
        if not isinstance(variant_reference_validated, bool) or variant_reference_validated != reference_validated:
            raise ValueError(f"genomic report variant {index} has inconsistent reference-validation state")
        if variant_reference_validated:
            validation_counts["reference_validated"] += 1
        else:
            validation_counts["reference_unvalidated"] += 1
        if variant_class == "unsupported_symbolic_or_breakend":
            validation_counts["unsupported_symbolic_or_breakend"] += 1
        transcript_consequences = variant.get("transcript_consequences", [])
        if not isinstance(transcript_consequences, list) or len(transcript_consequences) > 1000:
            raise ValueError(f"genomic report variant {index} transcript consequences must be a list")
        consequence_entries += len(transcript_consequences)
        if consequence_entries > 100_000:
            raise ValueError("genomic report exceeds 100000 transcript-consequence annotations")
        for transcript in transcript_consequences:
            if not isinstance(transcript, dict):
                raise ValueError(f"genomic report variant {index} has a malformed transcript consequence")
            gene_id = transcript.get("gene_id")
            if gene_id is not None:
                if not isinstance(gene_id, str) or not gene_id.strip() or len(gene_id) > 128:
                    raise ValueError(f"genomic report variant {index} has an invalid annotated gene identifier")
                annotated_gene_ids.add(gene_id.strip())
            terms = transcript.get("consequences", [])
            if (not isinstance(terms, list) or len(terms) > 64
                    or any(not isinstance(term, str) or not _SO_TERM.fullmatch(term) for term in terms)):
                raise ValueError(f"genomic report variant {index} has malformed transcript consequence terms")
            consequences.update(terms)

    if set(record_alternates) != set(range(1, records_parsed + 1)):
        raise ValueError("genomic report record identities do not match its parsed-record count")
    for indexes in record_alternates.values():
        if sorted(indexes) != list(range(1, len(indexes) + 1)):
            raise ValueError("genomic report alternate indexes must be contiguous and unique per record")
    if classes["unsupported_symbolic_or_breakend"] != unsupported_reported:
        raise ValueError("genomic report unsupported-allele count does not match its variant rows")
    return {
        "schema_version": "1",
        "source": "MedAI local VCF analysis report",
        "source_vcf_sha256": digest,
        "analysis_implementation_sha256": implementation,
        "genome_build": build,
        "reference_validated": reference_validated,
        "reference_fasta_sha256": reference_sha256,
        "records_parsed": records_parsed,
        "alternate_alleles_emitted": emitted,
        "variant_class_counts": dict(sorted(classes.items())),
        "validation_counts": validation_counts,
        "local_transcript_consequence_counts": dict(sorted(consequences.items())),
        "local_annotated_gene_identifier_count": len(annotated_gene_ids),
        "sample_identifiers_retained": False,
        "raw_variant_records_retained_in_summary": False,
        "remote_sources_queried": False,
        "interpretation_limit": "This receipt summarizes a researcher-submitted local file analysis report. The worker does not receive or replay the source VCF, verify the submitted report against the file, query remote variant sources, or interpret pathogenicity/clinical significance.",
    }


def validate_local_vcf_summary(summary: Any, *, expected_source_id: str) -> dict[str, Any]:
    """Validate a privacy-minimized receipt before it is accepted by the research worker."""
    if not isinstance(summary, dict):
        raise ValueError("genomic dataset payload must be an aggregate VCF receipt")
    if "variants" in summary or "genotype_call_counts" in summary or "sample_ids" in summary:
        raise ValueError("genomic receipt must not contain raw variant, genotype, or sample records")
    if set(summary) != _SUMMARY_FIELDS:
        raise ValueError("genomic dataset receipt fields do not match the exact privacy-minimized schema")
    if summary.get("schema_version") != "1" or summary.get("source") != "MedAI local VCF analysis report":
        raise ValueError("genomic dataset is not a supported local VCF aggregate receipt")
    digest = summary.get("source_vcf_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != expected_source_id:
        raise ValueError("genomic dataset source ID must match its locally analyzed VCF SHA-256")
    if summary.get("genome_build") not in _BUILDS:
        raise ValueError("genomic receipt requires a GRCh37 or GRCh38 build")
    if summary.get("sample_identifiers_retained") is not False or summary.get("raw_variant_records_retained_in_summary") is not False or summary.get("remote_sources_queried") is not False:
        raise ValueError("genomic receipt must attest that identifiers and raw records were omitted and no remote sources were queried")
    implementation = summary.get("analysis_implementation_sha256")
    if not isinstance(implementation, dict) or set(implementation) != _IMPLEMENTATION_FILES or any(not isinstance(v, str) or not _SHA256.fullmatch(v) for v in implementation.values()):
        raise ValueError("genomic receipt has invalid implementation provenance")
    reference_validated = summary.get("reference_validated")
    reference_sha256 = summary.get("reference_fasta_sha256")
    if (not isinstance(reference_validated, bool)
            or (reference_validated and (not isinstance(reference_sha256, str) or not _SHA256.fullmatch(reference_sha256)))
            or (not reference_validated and reference_sha256 is not None)):
        raise ValueError("genomic receipt has inconsistent reference provenance")
    for key in ("records_parsed", "alternate_alleles_emitted", "local_annotated_gene_identifier_count"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2_000_000:
            raise ValueError(f"genomic receipt field {key} is outside its supported bounds")
    for key in ("variant_class_counts", "validation_counts", "local_transcript_consequence_counts"):
        counts = summary.get(key)
        if not isinstance(counts, dict) or len(counts) > 128 or any(not isinstance(name, str) or not name or len(name) > 128 or isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 2_000_000 for name, count in counts.items()):
            raise ValueError(f"genomic receipt field {key} must contain bounded aggregate counts")
    emitted = summary["alternate_alleles_emitted"]
    records_parsed = summary["records_parsed"]
    if records_parsed > emitted or (emitted == 0) != (records_parsed == 0):
        raise ValueError("genomic receipt record and alternate-allele counts are inconsistent")
    classes = summary["variant_class_counts"]
    validation = summary["validation_counts"]
    if (set(classes) - {"SNV", "small_indel_or_mnv", "unsupported_symbolic_or_breakend"}
            or set(validation) != {"reference_validated", "reference_unvalidated", "unsupported_symbolic_or_breakend"}
            or sum(classes.values()) != emitted
            or classes.get("unsupported_symbolic_or_breakend", 0) != validation["unsupported_symbolic_or_breakend"]
            or validation["reference_validated"] + validation["reference_unvalidated"] != emitted
            or validation["reference_validated"] != (emitted if reference_validated else 0)
            or validation["reference_unvalidated"] != (0 if reference_validated else emitted)):
        raise ValueError("genomic receipt aggregate counts are internally inconsistent")
    consequences = summary["local_transcript_consequence_counts"]
    if any(not _SO_TERM.fullmatch(term) for term in consequences) or sum(consequences.values()) > 6_400_000:
        raise ValueError("genomic receipt transcript-consequence counts are malformed or exceed the annotation bound")
    if (summary["local_annotated_gene_identifier_count"] > 100_000
            or summary["local_annotated_gene_identifier_count"] > emitted * 1000):
        raise ValueError("genomic receipt annotated-gene count exceeds the transcript bound")
    if summary.get("interpretation_limit") != _SUMMARY_INTERPRETATION_LIMIT:
        raise ValueError("genomic receipt interpretation limit does not match its provenance contract")
    return summary
