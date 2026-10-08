"""Join source-specific ClinVar and VEP receipts to exact local normalized alleles."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from medai_science.sources.gnomad import GnomADClient, validate_callset_frequency


_HEX = re.compile(r"[a-f0-9]{64}\Z")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def reconcile_gnomad_frequency(
    genomic_report: dict[str, Any], frequency_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join gnomAD population observations only to exact local GRCh38-normalized alleles."""
    if not isinstance(genomic_report, dict) or genomic_report.get("reference_validated") is not True:
        raise ValueError("gnomAD reconciliation requires a reference-validated local genomic report")
    if genomic_report.get("genome_build") != "GRCh38":
        raise ValueError("gnomAD v4 frequency receipts require a GRCh38 local report")
    reference_digest = genomic_report.get("reference_fasta_sha256")
    local_vcf_digest = genomic_report.get("input_sha256")
    variants = genomic_report.get("variants")
    if (not isinstance(reference_digest, str) or not _HEX.fullmatch(reference_digest)
            or not isinstance(local_vcf_digest, str) or not _HEX.fullmatch(local_vcf_digest)
            or not isinstance(variants, list) or len(variants) > 1000):
        raise ValueError("local genomic report identity or variant list is malformed")
    if not isinstance(frequency_receipts, list) or len(frequency_receipts) > len(variants):
        raise ValueError("gnomAD receipt list is malformed or exceeds the local variant count")

    local_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in variants:
        if not isinstance(row, dict):
            raise ValueError("local report contains a malformed variant row")
        if row.get("sequence_allele_supported") is not True:
            continue
        if row.get("reference_validated") is not True:
            raise ValueError("sequence variants in the local report must be reference-validated")
        contig = row.get("contig")
        position = row.get("normalized_position_1_based")
        ref, alt = row.get("normalized_reference_allele"), row.get("normalized_alternate_allele")
        if (not isinstance(contig, str) or isinstance(position, bool) or not isinstance(position, int)
                or not isinstance(ref, str) or not isinstance(alt, str)):
            raise ValueError("local report contains a malformed normalized allele")
        try:
            variant_id = GnomADClient.parse_variant_id(f"{contig.removeprefix('chr')}-{position}-{ref}-{alt}")
        except ValueError:
            # Mitochondrial, structural, symbolic, and otherwise unsupported local variants stay unmatched.
            continue
        local_by_id.setdefault(variant_id, []).append(row)

    receipts_by_id: dict[str, dict[str, Any]] = {}
    for receipt in frequency_receipts:
        if not isinstance(receipt, dict):
            raise ValueError("gnomAD receipt must be an object")
        variant_id = receipt.get("variant_id")
        if (receipt.get("source") != "gnomAD Browser GraphQL API"
                or receipt.get("source_url") != "https://gnomad.broadinstitute.org/"
                or receipt.get("assembly") != "GRCh38"
                or receipt.get("dataset") != "gnomAD v4 (gnomad_r4)"
                or not isinstance(receipt.get("source_record_sha256"), str)
                or not _HEX.fullmatch(receipt["source_record_sha256"])
                or not isinstance(receipt.get("retrieved_at"), str)
                or len(receipt["retrieved_at"]) > 32
                or not isinstance(receipt.get("found"), bool)):
            raise ValueError("gnomAD receipt source identity, digest, or metadata is inconsistent")
        try:
            canonical_id = GnomADClient.parse_variant_id(variant_id)
        except ValueError as error:
            raise ValueError("gnomAD receipt variant identity is malformed") from error
        if canonical_id != variant_id or variant_id not in local_by_id:
            raise ValueError("gnomAD receipt does not match an exact local normalized allele")
        if variant_id in receipts_by_id:
            raise ValueError("duplicate gnomAD receipt for one normalized allele")
        frequency = receipt.get("genome_frequency")
        exome_frequency = receipt.get("exome_frequency")
        if receipt["found"]:
            frequency = validate_callset_frequency(frequency, "genome")
            exome_frequency = validate_callset_frequency(exome_frequency, "exome")
            if frequency is None and exome_frequency is None:
                raise ValueError("found gnomAD variant must include at least one callset frequency")
        elif frequency is not None or exome_frequency is not None:
            raise ValueError("not-found gnomAD result must not include callset frequency fields")
        expected_source_digest = _canonical_digest({
            "variant": {"variant_id": variant_id, "genome": frequency, "exome": exome_frequency} if receipt["found"] else None,
        })
        if receipt["source_record_sha256"] != expected_source_digest:
            raise ValueError("gnomAD receipt source-record digest does not match its retained observation")
        receipts_by_id[variant_id] = receipt

    joined = []
    for variant_id, receipt in sorted(receipts_by_id.items()):
        for row in local_by_id[variant_id]:
            joined.append({
                "record_number": row.get("record_number"),
                "alternate_index": row.get("alternate_index"),
                "variant_id": variant_id,
                "gnomad": {
                    "found": receipt["found"],
                    "dataset": receipt["dataset"],
                    "genome_frequency": receipt["genome_frequency"],
                    "exome_frequency": receipt["exome_frequency"],
                    "source_record_sha256": receipt["source_record_sha256"],
                    "retrieved_at": receipt["retrieved_at"],
                },
            })
    report_digest = _canonical_digest(genomic_report)
    return {
        "schema_version": "1",
        "source": "gnomAD Browser GraphQL API",
        "dataset": "gnomAD v4 (gnomad_r4)",
        "assembly": "GRCh38",
        "reference_fasta_sha256": reference_digest,
        "local_vcf_sha256": local_vcf_digest,
        "local_report_sha256": report_digest,
        "receipt_count": len(receipts_by_id),
        "matched_local_allele_count": len(joined),
        "variants": joined,
        "source_authentication_limit": "This local join checks receipt structure and exact allele identity but cannot authenticate caller-supplied JSON or the original HTTPS response. Preserve the source adapter artifact through the authenticated session artifact store when provenance authentication is required.",
        "interpretation_limit": "gnomAD allele counts and frequencies are population-level observations in the named dataset. They do not establish an individual's genotype, pathogenicity, disease causality, or clinical risk.",
    }


def reconcile_variant_sources(
    genomic_report: dict[str, Any],
    clinvar_reconciliation: dict[str, Any],
    vep_annotation: dict[str, Any],
) -> dict[str, Any]:
    """Combine ClinVar identity and VEP consequences only for exact local allele rows.

    The result preserves the evidence reported by each provider independently. It
    does not reconcile classification disagreement or infer pathogenicity.
    """
    if not isinstance(genomic_report, dict) or genomic_report.get("reference_validated") is not True:
        raise ValueError("source reconciliation requires a reference-validated local genomic report")
    build = genomic_report.get("genome_build")
    reference_digest = genomic_report.get("reference_fasta_sha256")
    local_vcf_digest = genomic_report.get("input_sha256")
    variants = genomic_report.get("variants")
    if build not in {"GRCh37", "GRCh38"} or not isinstance(reference_digest, str) or not _HEX.fullmatch(reference_digest):
        raise ValueError("local genomic report is missing a supported build or reference digest")
    if not isinstance(local_vcf_digest, str) or not _HEX.fullmatch(local_vcf_digest) or not isinstance(variants, list) or len(variants) > 1000:
        raise ValueError("local genomic report identity or variant list is malformed")
    local_by_allele: dict[tuple[int, int], dict[str, Any]] = {}
    for row in variants:
        if not isinstance(row, dict):
            raise ValueError("local genomic report contains a malformed variant row")
        record_number = row.get("record_number")
        alternate_index = row.get("alternate_index")
        if (isinstance(record_number, bool) or not isinstance(record_number, int) or record_number < 1
                or isinstance(alternate_index, bool) or not isinstance(alternate_index, int) or alternate_index < 1
                or (record_number, alternate_index) in local_by_allele):
            raise ValueError("local genomic report requires unique positive VCF record/alternate-index pairs")
        local_by_allele[(record_number, alternate_index)] = row

    if not isinstance(clinvar_reconciliation, dict) or clinvar_reconciliation.get("genome_build") != build:
        raise ValueError("ClinVar reconciliation build does not match the local report")
    if clinvar_reconciliation.get("reference_fasta_sha256") != reference_digest or clinvar_reconciliation.get("local_vcf_sha256") != local_vcf_digest:
        raise ValueError("ClinVar reconciliation is not bound to the local reference and VCF")
    clinvar_rows = clinvar_reconciliation.get("variants")
    if not isinstance(clinvar_rows, list) or len(clinvar_rows) != len(variants):
        raise ValueError("ClinVar reconciliation variant count does not match the local report")

    report_digest = _canonical_digest(genomic_report)
    endpoint = "https://grch37.rest.ensembl.org" if build == "GRCh37" else "https://rest.ensembl.org"
    if not isinstance(vep_annotation, dict) or vep_annotation.get("source") != "Ensembl Variant Effect Predictor REST API":
        raise ValueError("VEP annotation source identity is missing or unsupported")
    if vep_annotation.get("assembly") != build or vep_annotation.get("source_url") != endpoint + "/vep/homo_sapiens/region":
        raise ValueError("VEP annotation assembly or endpoint does not match the local report")
    if vep_annotation.get("reference_fasta_sha256") != reference_digest or vep_annotation.get("input_report_sha256") != report_digest:
        raise ValueError("VEP annotation is not bound to the exact local report and reference")
    vep_rows = vep_annotation.get("results")
    if not isinstance(vep_rows, list) or len(vep_rows) > 1000:
        raise ValueError("VEP annotation result list is malformed or exceeds the reconciliation bound")

    clinvar_by_record: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(clinvar_rows):
        local = variants[index]
        if (not isinstance(row, dict) or row.get("record_number") != local.get("record_number")
                or row.get("alternate_index") != local.get("alternate_index")):
            raise ValueError("ClinVar reconciliation rows are not aligned to exact local record/allele identities")
        if row.get("status") not in {"exact_normalized_allele_match", "no_exact_match", "needs_review"}:
            raise ValueError("ClinVar reconciliation contains an unsupported match status")
        records = row.get("clinvar_records")
        if not isinstance(records, list) or len(records) > 1000:
            raise ValueError("ClinVar reconciliation source records are malformed")
        if (row.get("status") == "exact_normalized_allele_match") != bool(records):
            raise ValueError("ClinVar reconciliation status does not match its source record list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("ClinVar source record must be an object")
            variation_id, accession = record.get("variation_id"), record.get("accession")
            if (
                not isinstance(variation_id, str) or not variation_id.isdigit()
                or not isinstance(accession, str) or not re.fullmatch(r"VCV[0-9]{9}\.[0-9]+", accession)
                or record.get("source") != "NCBI ClinVar E-utilities"
                or record.get("source_url") != f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
                or not isinstance(record.get("source_record_sha256"), str) or not _HEX.fullmatch(record["source_record_sha256"])
                or not isinstance(record.get("location_source_sha256"), str) or not _HEX.fullmatch(record["location_source_sha256"])
            ):
                raise ValueError("ClinVar reconciliation has inconsistent source identity or digest metadata")
        clinvar_by_record[(row["record_number"], row["alternate_index"])] = row

    vep_by_record: dict[tuple[int, int], dict[str, Any]] = {}
    for row in vep_rows:
        if not isinstance(row, dict):
            raise ValueError("VEP result row must be an object")
        record_number = row.get("local_record_number")
        alternate_index = row.get("local_alternate_index")
        allele_key = (record_number, alternate_index)
        if (isinstance(record_number, bool) or not isinstance(record_number, int)
                or isinstance(alternate_index, bool) or not isinstance(alternate_index, int)
                or allele_key not in local_by_allele or allele_key in vep_by_record):
            raise ValueError("VEP rows must identify unique local record/alternate-index pairs")
        source_digest = row.get("source_record_sha256")
        if not isinstance(source_digest, str) or not _HEX.fullmatch(source_digest):
            raise ValueError("VEP row is missing a source-record digest")
        vep_by_record[allele_key] = row

    reconciled = []
    for index, variant in enumerate(variants):
        record_number = variant.get("record_number")
        alternate_index = variant.get("alternate_index")
        allele_key = (record_number, alternate_index)
        clinvar_row = clinvar_by_record[allele_key]
        vep_row = vep_by_record.get(allele_key)
        exact_local = all(
            clinvar_row.get(field) == variant.get(local_field)
            for field, local_field in (
                ("contig", "contig"),
                ("position_1_based", "normalized_position_1_based"),
                ("reference_allele", "normalized_reference_allele"),
                ("alternate_allele", "normalized_alternate_allele"),
            )
        )
        if not exact_local:
            raise ValueError("ClinVar reconciliation allele identity differs from its local VCF record")
        if vep_row is not None and not all(
            vep_row.get(field) == variant.get(local_field)
            for field, local_field in (
                ("local_contig", "contig"),
                ("local_position_1_based", "normalized_position_1_based"),
                ("local_reference_allele", "normalized_reference_allele"),
                ("local_alternate_allele", "normalized_alternate_allele"),
            )
        ):
            raise ValueError("VEP annotation allele identity differs from its local VCF record")
        clinvar_matches = clinvar_row.get("clinvar_records", [])
        if not isinstance(clinvar_matches, list):
            raise ValueError("ClinVar reconciliation record list is malformed")
        has_clinvar = clinvar_row.get("status") == "exact_normalized_allele_match" and bool(clinvar_matches)
        status = "matched_both_sources" if has_clinvar and vep_row else "matched_clinvar_only" if has_clinvar else "matched_vep_only" if vep_row else "no_source_match"
        reconciled.append({
            "record_number": record_number,
            "alternate_index": alternate_index,
            "allele": {
                "contig": variant.get("contig"),
                "position_1_based": variant.get("normalized_position_1_based"),
                "reference_allele": variant.get("normalized_reference_allele"),
                "alternate_allele": variant.get("normalized_alternate_allele"),
            },
            "status": status,
            "clinvar": {
                "status": clinvar_row.get("status"),
                "records": clinvar_matches,
            },
            "ensembl_vep": None if vep_row is None else {
                "source_record_sha256": vep_row["source_record_sha256"],
                "most_severe_consequence": vep_row.get("most_severe_consequence"),
                "transcript_consequences": vep_row.get("transcript_consequences", []),
            },
        })
    return {
        "schema_version": "1",
        "genome_build": build,
        "reference_fasta_sha256": reference_digest,
        "local_vcf_sha256": local_vcf_digest,
        "local_report_sha256": report_digest,
        "clinvar_reconciliation_sha256": _canonical_digest(clinvar_reconciliation),
        "vep_annotation_sha256": _canonical_digest(vep_annotation),
        "variants": reconciled,
        "source_authentication_limit": "This local join verifies receipt structure and digest bindings, not the authenticity of caller-supplied JSON artifacts or the original HTTPS sessions. Preserve and verify the source adapter artifacts through the authenticated session artifact store when provenance authentication is required.",
        "interpretation_limit": "ClinVar submitter classifications and Ensembl transcript consequences are retained as separate source observations joined only by exact reference-normalized local allele identity. This report does not adjudicate source conflicts, establish pathogenicity, or provide clinical interpretation.",
    }
