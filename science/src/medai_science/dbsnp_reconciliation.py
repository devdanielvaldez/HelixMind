"""Exact assembly-aware RefSNP placement joins to local normalized VCF alleles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from medai_science.genomics import IndexedFasta, _digest, _normalize_alleles
from medai_science.sources.dbsnp import DbSNPClient


_HEX = re.compile(r"[a-f0-9]{64}\Z")
_DNA = re.compile(r"[ACGT]+\Z")
_ASSEMBLY_ACCESSION = re.compile(r"(?:GCA|GCF)_[0-9]{9}\.[0-9]{1,3}\Z")
_MAX_PLACEMENT_ALLELES = 4_000
_MAX_ASSEMBLY_ACCESSIONS_PER_PLACEMENT = 100
_MAX_RECONCILIATION_BYTES = 4 * 1024 * 1024


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _assembly_matches(name: Any, genome_build: str) -> bool:
    if not isinstance(name, str):
        return False
    # dbSNP names assembly patches as GRCh37.pN / GRCh38.pN. Accept the
    # unpatched root or a numeric patch only; arbitrary prefix suffixes are
    # not evidence that a placement belongs to the requested assembly.
    return re.fullmatch(re.escape(genome_build) + r"(?:\.p[1-9][0-9]*)?", name) is not None


def _spdi_to_normalized_vcf(
    reference: IndexedFasta, contig: str, position: int, deleted: str, inserted: str,
) -> tuple[int, str, str]:
    """Convert one sequence-oriented SPDI to a reference-checked, normalized VCF allele."""
    if position < 0 or (deleted and not _DNA.fullmatch(deleted)):
        raise ValueError("dbSNP SPDI has an unsupported position or deleted sequence")
    if inserted and not _DNA.fullmatch(inserted):
        raise ValueError("dbSNP SPDI has an unsupported inserted sequence")
    if deleted == inserted:
        raise ValueError("dbSNP SPDI describes a reference allele, not a variant")
    if len(deleted) == len(inserted):
        position_1_based = position + 1
        ref, alt = deleted, inserted
    elif position > 0:
        anchor = reference.fetch(contig, position - 1, 1)
        position_1_based = position
        ref, alt = anchor + deleted, anchor + inserted
    else:
        anchor = reference.fetch(contig, len(deleted), 1)
        position_1_based = 1
        ref, alt = deleted + anchor, inserted + anchor
    if not ref or not alt or reference.fetch(contig, position_1_based - 1, len(ref)) != ref:
        raise ValueError("dbSNP SPDI deleted sequence does not match the supplied reference FASTA")
    normalized_position, normalized_ref, normalized_alt = _normalize_alleles(
        reference, contig, position_1_based, ref, alt,
    )
    return normalized_position, normalized_ref, normalized_alt


def reconcile_dbsnp_placements(
    genomic_report: dict[str, Any],
    receipt: dict[str, Any],
    reference_fasta: str | Path,
    genome_build: str,
    sequence_accession_to_contig: dict[str, str],
) -> dict[str, Any]:
    """Join dbSNP SPDI placements only after explicit build/sequence mapping and FASTA checks.

    dbSNP provides placements on sequence accessions, not local VCF contig names. A caller
    must supply an explicit RefSeq accession-to-FASTA-contig map; no chromosome-name inference
    or approximate allele match is performed.
    """
    if genome_build not in {"GRCh37", "GRCh38"}:
        raise ValueError("genome_build must be GRCh37 or GRCh38")
    if not isinstance(genomic_report, dict) or genomic_report.get("genome_build") != genome_build:
        raise ValueError("local genomic report build must exactly match the requested dbSNP assembly")
    reference_digest = genomic_report.get("reference_fasta_sha256")
    vcf_digest = genomic_report.get("input_sha256")
    variants = genomic_report.get("variants")
    emitted_count = genomic_report.get("alternate_alleles_emitted")
    parsed_count = genomic_report.get("records_parsed")
    if (
        genomic_report.get("reference_validated") is not True
        or not isinstance(reference_digest, str) or not _HEX.fullmatch(reference_digest)
        or not isinstance(vcf_digest, str) or not _HEX.fullmatch(vcf_digest)
        or not isinstance(variants, list) or len(variants) > 1000
        or isinstance(emitted_count, bool) or not isinstance(emitted_count, int) or emitted_count != len(variants)
        or isinstance(parsed_count, bool) or not isinstance(parsed_count, int)
        or parsed_count < 0 or parsed_count > emitted_count
    ):
        raise ValueError("local genomic report must have bounded, reference-validated alleles and provenance")
    if _digest(Path(reference_fasta)) != reference_digest:
        raise ValueError("reference FASTA digest does not match the local genomic report")
    if (
        not isinstance(sequence_accession_to_contig, dict) or not sequence_accession_to_contig
        or len(sequence_accession_to_contig) > 1000
        or any(not isinstance(key, str) or not re.fullmatch(r"(?:NC|NW|NT)_[0-9]+\.[0-9]+", key)
               or not isinstance(value, str) or not value or len(value) > 64
               for key, value in sequence_accession_to_contig.items())
    ):
        raise ValueError("provide a bounded explicit RefSeq sequence-accession to FASTA-contig map")

    if not isinstance(receipt, dict):
        raise ValueError("dbSNP receipt must be an object")
    rsid = receipt.get("rsid")
    record = receipt.get("source_record")
    if (
        receipt.get("source") != DbSNPClient.source
        or not isinstance(rsid, str) or DbSNPClient.normalize_rsid(rsid) != rsid
        or receipt.get("source_url") != f"https://www.ncbi.nlm.nih.gov/snp/{rsid[2:]}"
        or receipt.get("api_endpoint") != f"{DbSNPClient.endpoint}/{rsid[2:]}"
        or not isinstance(receipt.get("source_record_sha256"), str)
        or not _HEX.fullmatch(receipt["source_record_sha256"])
        or not isinstance(record, dict)
        or str(record.get("refsnp_id")) != rsid[2:]
        or receipt.get("source_record_canonical_sha256") != _canonical_digest(record)
    ):
        raise ValueError("dbSNP receipt identity, source URL, or canonical record digest is inconsistent")
    snapshot = record.get("primary_snapshot_data")
    placements = snapshot.get("placements_with_allele") if isinstance(snapshot, dict) else None
    if not isinstance(placements, list) or len(placements) > 2000:
        raise ValueError("dbSNP RefSNP record has no bounded primary placement list")
    placement_allele_count = sum(
        len(placement["alleles"])
        for placement in placements
        if isinstance(placement, dict)
        and isinstance(placement.get("alleles"), list)
        and len(placement["alleles"]) <= 256
    )
    if placement_allele_count > _MAX_PLACEMENT_ALLELES:
        raise ValueError(
            f"dbSNP placement allele collection exceeds the {_MAX_PLACEMENT_ALLELES}-allele reconciliation limit"
        )

    seen_variant_indexes: set[tuple[int, int]] = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("local genomic report contains a malformed variant row")
        record_number = variant.get("record_number")
        alternate_index = variant.get("alternate_index")
        if (
            isinstance(record_number, bool) or not isinstance(record_number, int) or not 1 <= record_number <= parsed_count
            or isinstance(alternate_index, bool) or not isinstance(alternate_index, int) or alternate_index < 1
            or (record_number, alternate_index) in seen_variant_indexes
            or not isinstance(variant.get("sequence_allele_supported"), bool)
            or not isinstance(variant.get("reference_validated"), bool)
        ):
            raise ValueError("local genomic report contains malformed or duplicate record/alternate identities")
        seen_variant_indexes.add((record_number, alternate_index))
        if not variant["sequence_allele_supported"] or not variant["reference_validated"]:
            continue
        contig = variant.get("contig")
        position = variant.get("normalized_position_1_based")
        ref, alt = variant.get("normalized_reference_allele"), variant.get("normalized_alternate_allele")
        if (
            not isinstance(contig, str) or isinstance(position, bool) or not isinstance(position, int)
            or position < 1 or not isinstance(ref, str) or not isinstance(alt, str)
            or not _DNA.fullmatch(ref) or not _DNA.fullmatch(alt)
        ):
            raise ValueError("local report contains a malformed normalized sequence allele")

    reference = IndexedFasta(reference_fasta)
    try:
        if reference.sha256 != reference_digest:
            raise ValueError("reference FASTA changed while preparing dbSNP reconciliation")
        for variant in variants:
            if not variant["sequence_allele_supported"] or not variant["reference_validated"]:
                continue
            contig = variant["contig"]
            position = variant["normalized_position_1_based"]
            ref, alt = variant["normalized_reference_allele"], variant["normalized_alternate_allele"]
            if reference.fetch(contig, position - 1, len(ref)) != ref:
                raise ValueError("local normalized VCF reference allele does not match the supplied FASTA")
            if ref == alt or _normalize_alleles(reference, contig, position, ref, alt) != (position, ref, alt):
                raise ValueError("local VCF allele is not minimally normalized against the supplied FASTA")
        source_matches: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
        unresolved: list[dict[str, str]] = []
        compatible_placements = 0
        for placement_index, placement in enumerate(placements):
            if not isinstance(placement, dict):
                unresolved.append({"placement_index": str(placement_index), "reason": "placement is not an object"})
                continue
            seq_id = placement.get("seq_id")
            annotation = placement.get("placement_annot")
            traits = annotation.get("seq_id_traits_by_assembly") if isinstance(annotation, dict) else None
            if not isinstance(seq_id, str) or not isinstance(traits, list):
                continue
            selected_traits = [trait for trait in traits if isinstance(trait, dict) and _assembly_matches(trait.get("assembly_name"), genome_build)]
            if not selected_traits:
                continue
            compatible_placements += 1
            contig = sequence_accession_to_contig.get(seq_id)
            if contig is None:
                unresolved.append({"placement_index": str(placement_index), "reason": f"no explicit FASTA contig mapping for {seq_id}"})
                continue
            alleles = placement.get("alleles")
            if not isinstance(alleles, list) or len(alleles) > 256:
                unresolved.append({"placement_index": str(placement_index), "reason": "allele list is malformed or exceeds the bound"})
                continue
            assembly_accessions = sorted({
                trait.get("assembly_accession") for trait in selected_traits
                if isinstance(trait.get("assembly_accession"), str)
                and _ASSEMBLY_ACCESSION.fullmatch(trait["assembly_accession"])
            })
            if not assembly_accessions or len(assembly_accessions) > _MAX_ASSEMBLY_ACCESSIONS_PER_PLACEMENT:
                unresolved.append({"placement_index": str(placement_index), "reason": "assembly accessions are missing, malformed, or exceed the bound"})
                continue
            for allele_index, allele_row in enumerate(alleles):
                allele = allele_row.get("allele") if isinstance(allele_row, dict) else None
                spdi = allele.get("spdi") if isinstance(allele, dict) else None
                if not isinstance(spdi, dict) or spdi.get("seq_id") != seq_id:
                    unresolved.append({"placement_index": str(placement_index), "reason": f"allele {allele_index} is not a sequence-matched SPDI"})
                    continue
                position, deleted, inserted = spdi.get("position"), spdi.get("deleted_sequence"), spdi.get("inserted_sequence")
                if isinstance(position, bool) or not isinstance(position, int) or position < 0 or not isinstance(deleted, str) or not isinstance(inserted, str):
                    unresolved.append({"placement_index": str(placement_index), "reason": f"allele {allele_index} has malformed SPDI fields"})
                    continue
                if deleted == inserted:
                    continue
                if (deleted and not _DNA.fullmatch(deleted)) or (inserted and not _DNA.fullmatch(inserted)):
                    unresolved.append({"placement_index": str(placement_index), "reason": f"allele {allele_index} contains an unsupported SPDI sequence"})
                    continue
                try:
                    normalized = _spdi_to_normalized_vcf(reference, contig, position, deleted, inserted)
                except ValueError as error:
                    unresolved.append({"placement_index": str(placement_index), "reason": str(error)[:200]})
                    continue
                key = (contig, *normalized)
                source_entry = {
                    "rsid": rsid,
                    "source": DbSNPClient.source,
                    "source_url": receipt["source_url"],
                    "source_record_sha256": receipt["source_record_sha256"],
                    "source_record_canonical_sha256": receipt["source_record_canonical_sha256"],
                    "sequence_accession": seq_id,
                    "assembly_accessions": assembly_accessions,
                    "spdi": {"position": position, "deleted_sequence": deleted, "inserted_sequence": inserted},
                    "normalized_allele": {"contig": contig, "position_1_based": normalized[0], "reference_allele": normalized[1], "alternate_allele": normalized[2]},
                }
                source_matches.setdefault(key, []).append(source_entry)
        reference.verify_unchanged()
    finally:
        reference.close()

    reconciled = []
    for variant in variants:
        record_number = variant.get("record_number") if isinstance(variant, dict) else None
        alternate_index = variant.get("alternate_index") if isinstance(variant, dict) else None
        if not isinstance(variant, dict) or variant.get("sequence_allele_supported") is not True or variant.get("reference_validated") is not True:
            reconciled.append({"record_number": record_number, "alternate_index": alternate_index, "status": "needs_review", "dbsnp_records": []})
            continue
        key = (variant["contig"], variant["normalized_position_1_based"], variant["normalized_reference_allele"], variant["normalized_alternate_allele"])
        matches = source_matches.get(key, [])
        reconciled.append({
            "record_number": record_number,
            "alternate_index": alternate_index,
            "status": "exact_normalized_allele_match" if matches else "no_exact_match",
            "dbsnp_records": matches,
        })
    result = {
        "schema_version": "1",
        "source": DbSNPClient.source,
        "rsid": rsid,
        "genome_build": genome_build,
        "input_vcf_sha256": vcf_digest,
        "reference_fasta_sha256": reference_digest,
        "source_record_canonical_sha256": receipt["source_record_canonical_sha256"],
        "explicit_sequence_accession_to_contig_map": dict(sorted(sequence_accession_to_contig.items())),
        "compatible_placement_count": compatible_placements,
        "matched_allele_count": sum(row["status"] == "exact_normalized_allele_match" for row in reconciled),
        "variants": reconciled,
        "unresolved_source_placements": unresolved[:200],
        "interpretation_limit": "Exact reference-normalized sequence-allele identity on a caller-selected assembly and explicit sequence-accession map only. A missing exact match is not evidence of absence from dbSNP. Source placements and dbSNP annotations do not establish pathogenicity, phenotype, or clinical significance; response digests do not cryptographically authenticate NCBI.",
    }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_RECONCILIATION_BYTES:
        raise ValueError("dbSNP reconciliation receipt exceeds the 4 MiB retained-artifact limit")
    return result
