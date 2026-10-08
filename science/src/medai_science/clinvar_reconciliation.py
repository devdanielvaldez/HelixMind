"""Exact, reference-normalized matching between local VCF calls and ClinVar locations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from medai_science.genomics import IndexedFasta, _digest, _normalize_alleles


def reconcile_clinvar_variants(
    genomics_report: dict[str, Any],
    clinvar_records: list[dict[str, Any]],
    reference_fasta: str | Path,
    genome_build: str,
) -> dict[str, Any]:
    """Join only identical normalized alleles on the same declared build and contig."""
    if genome_build not in {"GRCh37", "GRCh38"}:
        raise ValueError("genome_build must be GRCh37 or GRCh38")
    if not isinstance(genomics_report, dict) or genomics_report.get("genome_build") != genome_build:
        raise ValueError("local genomic report build must exactly match the requested ClinVar assembly")
    if not genomics_report.get("reference_validated") or not isinstance(genomics_report.get("reference_fasta_sha256"), str):
        raise ValueError("local genomic report must have validated variants against a reference FASTA")
    if _digest(Path(reference_fasta)) != genomics_report["reference_fasta_sha256"]:
        raise ValueError("reference FASTA digest does not match the local genomic report")
    if not isinstance(clinvar_records, list):
        raise ValueError("ClinVar records must be an array")
    if len(clinvar_records) > 1000:
        raise ValueError("ClinVar reconciliation is limited to 1000 source records")

    reference = IndexedFasta(reference_fasta)
    try:
        if reference.sha256 != genomics_report["reference_fasta_sha256"]:
            raise ValueError("reference FASTA changed while preparing ClinVar reconciliation")
        matches: dict[tuple[str, int, str, str], list[dict[str, str]]] = {}
        unresolved: list[dict[str, str]] = []
        for record in clinvar_records:
            if not isinstance(record, dict):
                continue
            variation_id = record.get("variation_id")
            if not isinstance(variation_id, str) or not variation_id.isdigit():
                continue
            locations = record.get("assembly_locations")
            if not isinstance(locations, list):
                continue
            accession = record.get("accession")
            source_record_hash = record.get("source_record_sha256")
            location_hash = record.get("location_source_sha256")
            expected_url = f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
            actual_location_hash = hashlib.sha256(
                json.dumps(locations, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                record.get("source") != "NCBI ClinVar E-utilities"
                or record.get("source_url") != expected_url
                or not isinstance(accession, str)
                or not re.fullmatch(r"VCV[0-9]{9}\.[0-9]+", accession)
                or not isinstance(source_record_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", source_record_hash)
                or not isinstance(location_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", location_hash)
                or location_hash != actual_location_hash
            ):
                unresolved.append({"variation_id": variation_id, "reason": "ClinVar record source, accession, URL, or location digest is missing or inconsistent"})
                continue
            for location in locations:
                if not isinstance(location, dict) or location.get("assembly") != genome_build:
                    continue
                contig = location.get("contig")
                position = location.get("position_1_based")
                ref = location.get("reference_allele")
                alt = location.get("alternate_allele")
                if not isinstance(contig, str) or isinstance(position, bool) or not isinstance(position, int) or position < 1:
                    continue
                if not isinstance(ref, str) or not isinstance(alt, str) or not ref or not alt:
                    continue
                if not re.fullmatch(r"[ACGT]+", ref.upper()) or not re.fullmatch(r"[ACGT]+", alt.upper()):
                    unresolved.append({"variation_id": variation_id, "reason": "ClinVar location contains an ambiguous or unsupported sequence allele"})
                    continue
                try:
                    observed = reference.fetch(contig, position - 1, len(ref))
                    if observed != ref.upper():
                        unresolved.append({"variation_id": variation_id, "reason": "ClinVar reference allele does not match the supplied FASTA"})
                        continue
                    normalized = _normalize_alleles(reference, contig, position, ref.upper(), alt.upper())
                except ValueError as error:
                    unresolved.append({"variation_id": variation_id, "reason": str(error)[:200]})
                    continue
                if normalized[1] == normalized[2]:
                    continue
                key = (contig, normalized[0], normalized[1], normalized[2])
                source = {
                    "variation_id": variation_id,
                    "accession": accession,
                    "source": record["source"],
                    "source_url": expected_url,
                    "source_record_sha256": source_record_hash,
                    "location_source_sha256": location_hash,
                }
                matches.setdefault(key, []).append(source)

        variants = genomics_report.get("variants")
        if not isinstance(variants, list):
            raise ValueError("local genomic report does not contain a variant array")
        reconciled = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if not variant.get("reference_validated") or not variant.get("sequence_allele_supported"):
                reconciled.append({"record_number": variant.get("record_number"), "status": "needs_review", "reason": "local allele lacks reference validation or is not a supported sequence allele"})
                continue
            position = variant.get("normalized_position_1_based")
            ref = variant.get("normalized_reference_allele")
            alt = variant.get("normalized_alternate_allele")
            contig = variant.get("contig")
            key = (contig, position, ref, alt)
            sources = matches.get(key, [])
            deduped = {item["variation_id"]: item for item in sources}
            reconciled.append({
                "record_number": variant.get("record_number"),
                "alternate_index": variant.get("alternate_index"),
                "contig": contig,
                "position_1_based": position,
                "reference_allele": ref,
                "alternate_allele": alt,
                "status": "exact_normalized_allele_match" if deduped else "no_exact_match",
                "clinvar_records": [deduped[identifier] for identifier in sorted(deduped)],
            })
        reference.verify_unchanged()
        return {
            "genome_build": genome_build,
            "reference_fasta_sha256": genomics_report["reference_fasta_sha256"],
            "local_vcf_sha256": genomics_report.get("input_sha256"),
            "clinvar_records_considered": len(clinvar_records),
            "variants": reconciled,
            "unresolved_clinvar_locations": unresolved[:100],
            "source": "NCBI ClinVar E-utilities records supplied to the local reconciler",
            "source_identity_verification": "ClinVar source labels, VCV accession format, canonical location digest, and source-record digest format were checked; caller-supplied JSON is not cryptographically authenticated by this local operation.",
            "interpretation_limit": "Exact normalized allele identity on the same assembly only. ClinVar classifications remain submitter evidence and this reconciliation does not establish pathogenicity or clinical relevance.",
        }
    finally:
        reference.close()
