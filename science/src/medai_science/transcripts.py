"""Local GFF3 transcript consequence projection against a supplied reference."""
from __future__ import annotations

import gzip
import hashlib
import re
import zlib
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote

from medai_science.genomics import IndexedFasta

_MAX_GFF_BYTES = 8 * 1024 * 1024 * 1024
_MAX_GFF_FEATURES = 8_000_000
_MAX_GFF_LINE = 2_000_000
_TRANSCRIPT_TYPES = {"mrna", "transcript", "ncrna", "lnc_rna", "lncrna", "trna", "rrna", "primary_transcript", "so:0000234"}
_UTR_TYPES = {
    "five_prime_utr": "5_prime_UTR_variant",
    "so:0000204": "5_prime_UTR_variant",
    "three_prime_utr": "3_prime_UTR_variant",
    "so:0000205": "3_prime_UTR_variant",
}
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

_CODONS = {
    "TTT":"F", "TTC":"F", "TTA":"L", "TTG":"L", "TCT":"S", "TCC":"S", "TCA":"S", "TCG":"S",
    "TAT":"Y", "TAC":"Y", "TAA":"*", "TAG":"*", "TGT":"C", "TGC":"C", "TGA":"*", "TGG":"W",
    "CTT":"L", "CTC":"L", "CTA":"L", "CTG":"L", "CCT":"P", "CCC":"P", "CCA":"P", "CCG":"P",
    "CAT":"H", "CAC":"H", "CAA":"Q", "CAG":"Q", "CGT":"R", "CGC":"R", "CGA":"R", "CGG":"R",
    "ATT":"I", "ATC":"I", "ATA":"I", "ATG":"M", "ACT":"T", "ACC":"T", "ACA":"T", "ACG":"T",
    "AAT":"N", "AAC":"N", "AAA":"K", "AAG":"K", "AGT":"S", "AGC":"S", "AGA":"R", "AGG":"R",
    "GTT":"V", "GTC":"V", "GTA":"V", "GTG":"V", "GCT":"A", "GCC":"A", "GCA":"A", "GCG":"A",
    "GAT":"D", "GAC":"D", "GAA":"E", "GAG":"E", "GGT":"G", "GGC":"G", "GGA":"G", "GGG":"G",
}


def _text(path: Path) -> TextIO:
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="")
    return path.open("rt", encoding="utf-8", errors="strict", newline="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attributes(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(";"):
        if not item:
            continue
        key, separator, content = item.partition("=")
        if not separator:
            raise ValueError(f"malformed GFF3 attribute: {item[:80]}")
        result[unquote(key)] = unquote(content)
    return result


def _reverse_complement(sequence: str) -> str:
    return sequence.upper().translate(_COMPLEMENT)[::-1]


@dataclass
class _CdsSegment:
    start: int
    end: int
    phase: int


@dataclass
class _Transcript:
    identifier: str
    contig: str
    start: int
    end: int
    strand: str
    gene_id: str | None = None
    gene_name: str | None = None
    translation_table: int = 1
    cds: list[_CdsSegment] = field(default_factory=list)
    exons: list[tuple[int, int]] = field(default_factory=list)
    utrs: list[tuple[int, int, str]] = field(default_factory=list)
    explicit_feature: bool = False

    def extend(self, contig: str, start: int, end: int, strand: str) -> None:
        if self.contig != contig or self.strand != strand:
            raise ValueError(f"transcript {self.identifier} has inconsistent contig or strand")
        self.start = min(self.start, start)
        self.end = max(self.end, end)


class _IntervalNode:
    def __init__(self, intervals: list[_Transcript]):
        midpoints = sorted((item.start + item.end) // 2 for item in intervals)
        self.center = midpoints[len(midpoints) // 2]
        crossing = [item for item in intervals if item.start <= self.center <= item.end]
        self.by_start = sorted(crossing, key=lambda item: item.start)
        self.by_end = sorted(crossing, key=lambda item: item.end, reverse=True)
        left = [item for item in intervals if item.end < self.center]
        right = [item for item in intervals if item.start > self.center]
        self.left = _IntervalNode(left) if left else None
        self.right = _IntervalNode(right) if right else None

    def query(self, start: int, end: int) -> list[_Transcript]:
        found: list[_Transcript] = []
        if end < self.center:
            for item in self.by_start:
                if item.start > end:
                    break
                found.append(item)
            if self.left:
                found.extend(self.left.query(start, end))
        elif start > self.center:
            for item in self.by_end:
                if item.end < start:
                    break
                found.append(item)
            if self.right:
                found.extend(self.right.query(start, end))
        else:
            found.extend(self.by_start)
            if self.left:
                found.extend(self.left.query(start, end))
            if self.right:
                found.extend(self.right.query(start, end))
        return found


def _parse_gff3(path: Path) -> tuple[dict[str, _Transcript], str, str | None]:
    if not path.is_file() or path.stat().st_size > _MAX_GFF_BYTES:
        raise ValueError("GFF3 annotation must be a regular file no larger than 8 GiB")
    initial_signature = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
    initial_hash = _sha256(path)
    transcripts: dict[str, _Transcript] = {}
    gene_names: dict[str, str] = {}
    transcript_parents: dict[str, str] = {}
    feature_count = 0
    decompressed_bytes = 0
    gff_version: str | None = None
    genome_build: str | None = None

    def get_transcript(identifier: str, contig: str, start: int, end: int, strand: str) -> _Transcript:
        model = transcripts.get(identifier)
        if model is None:
            model = _Transcript(identifier, contig, start, end, strand)
            transcripts[identifier] = model
        else:
            model.extend(contig, start, end, strand)
        return model

    try:
        with _text(path) as handle:
            line_number = 0
            while True:
                raw = handle.readline(_MAX_GFF_LINE + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw.encode("utf-8")) > _MAX_GFF_LINE:
                    raise ValueError(f"GFF3 line {line_number} exceeds the 2 MB safety limit")
                decompressed_bytes += len(raw.encode("utf-8"))
                if decompressed_bytes > _MAX_GFF_BYTES:
                    raise ValueError("decompressed GFF3 data exceeds the 8 GiB safety limit")
                line = raw.rstrip("\r\n")
                if line == "##FASTA":
                    break
                if not line:
                    continue
                if line.startswith("##gff-version"):
                    if line_number != 1 or gff_version is not None:
                        raise ValueError("GFF3 version declaration must appear once on the first line")
                    gff_version = line.partition(" ")[2].strip()
                    if gff_version != "3":
                        raise ValueError("only GFF3 version 3 annotations are supported")
                    continue
                if line.startswith("##genome-build "):
                    genome_build = line.partition(" ")[2].strip()
                    continue
                if line.startswith("#"):
                    continue
                columns = line.split("\t")
                if len(columns) != 9:
                    raise ValueError(f"GFF3 line {line_number} must have exactly nine tab-separated columns")
                contig, _source, feature, start_text, end_text, _score, strand, phase_text, attributes_text = columns
                feature_count += 1
                if feature_count > _MAX_GFF_FEATURES:
                    raise ValueError(f"GFF3 exceeds the {_MAX_GFF_FEATURES}-feature safety limit")
                if not re.fullmatch(r"[1-9]\d*", start_text) or not re.fullmatch(r"[1-9]\d*", end_text):
                    raise ValueError(f"GFF3 feature coordinates are invalid at line {line_number}")
                start, end = int(start_text), int(end_text)
                if end < start or strand not in {"+", "-"}:
                    raise ValueError(f"GFF3 feature interval or strand is invalid at line {line_number}")
                attrs = _attributes(attributes_text)
                lowered = feature.lower()
                if lowered == "gene":
                    if attrs.get("ID"):
                        gene_names[attrs["ID"]] = attrs.get("Name", attrs.get("gene", attrs["ID"]))
                    continue
                if lowered in _TRANSCRIPT_TYPES:
                    identifier = attrs.get("ID")
                    if not identifier:
                        raise ValueError(f"GFF3 transcript at line {line_number} has no ID attribute")
                    model = get_transcript(identifier, contig, start, end, strand)
                    model.explicit_feature = True
                    parents = attrs.get("Parent", "").split(",")
                    if parents and parents[0]:
                        transcript_parents[identifier] = parents[0]
                        model.gene_id = parents[0]
                    model.gene_name = attrs.get("gene_name", attrs.get("gene", model.gene_name))
                    table = attrs.get("transl_table")
                    if table:
                        model.translation_table = int(table)
                    continue
                parents = [item for item in attrs.get("Parent", "").split(",") if item]
                if lowered == "cds":
                    if phase_text not in {"0", "1", "2"}:
                        raise ValueError(f"GFF3 CDS phase must be 0, 1, or 2 at line {line_number}")
                    if not parents:
                        raise ValueError(f"GFF3 CDS at line {line_number} has no transcript Parent")
                    table_text = attrs.get("transl_table")
                    for parent in parents:
                        model = get_transcript(parent, contig, start, end, strand)
                        model.cds.append(_CdsSegment(start, end, int(phase_text)))
                        if table_text:
                            table = int(table_text)
                            if model.translation_table not in {1, table}:
                                raise ValueError(f"transcript {parent} has conflicting translation tables")
                            model.translation_table = table
                elif lowered == "exon":
                    for parent in parents:
                        get_transcript(parent, contig, start, end, strand).exons.append((start, end))
                elif lowered in _UTR_TYPES:
                    for parent in parents:
                        get_transcript(parent, contig, start, end, strand).utrs.append((start, end, _UTR_TYPES[lowered]))
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise ValueError(f"could not read GFF3 annotation: {error}") from error

    if gff_version is None:
        raise ValueError("GFF3 version directive was not found")
    for transcript_id, model in transcripts.items():
        if model.gene_id and not model.gene_name:
            model.gene_name = gene_names.get(model.gene_id)
        model.exons = sorted(set(model.exons))
        model.utrs = sorted(set(model.utrs))
    signature = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
    final_hash = _sha256(path)
    if signature != initial_signature or initial_hash != final_hash:
        raise ValueError("GFF3 annotation changed while being parsed")
    return transcripts, initial_hash, genome_build


def _transcript_cds(model: _Transcript, reference: IndexedFasta) -> tuple[str, bool, int]:
    segments = sorted(model.cds, key=lambda item: (item.start, item.end), reverse=model.strand == "-")
    sequence_parts: list[str] = []
    cumulative = 0
    phase_consistent = True
    for index, segment in enumerate(segments):
        if index:
            expected = (3 - cumulative % 3) % 3
            if segment.phase != expected:
                phase_consistent = False
        genomic = reference.fetch(model.contig, segment.start - 1, segment.end - segment.start + 1)
        if model.strand == "-":
            genomic = _reverse_complement(genomic)
        sequence_parts.append(genomic)
        cumulative += len(genomic)
    first_phase = segments[0].phase if segments else 0
    return "".join(sequence_parts), phase_consistent, first_phase


def _translate(sequence: str, table: int) -> str:
    codons = dict(_CODONS)
    if table == 2:
        codons.update({"ATA": "M", "TGA": "W", "AGA": "*", "AGG": "*"})
    if table not in {1, 2}:
        raise ValueError(f"unsupported NCBI translation table {table}")
    return "".join(codons.get(sequence[index:index + 3], "X") for index in range(0, len(sequence) - 2, 3))


def _splice_site_consequences(model: _Transcript, start: int, end: int) -> list[str]:
    """Find canonical splice bases and nearby splice-region bases at exon junctions."""
    effects: set[str] = set()

    def overlaps(interval_start: int, interval_end: int) -> bool:
        return start <= interval_end and interval_start <= end

    for exon_start, exon_end in model.exons:
        if model.strand == "+":
            if exon_start > model.start:
                if overlaps(exon_start - 2, exon_start - 1):
                    effects.add("splice_acceptor_variant")
                if overlaps(exon_start - 8, exon_start - 3) or overlaps(exon_start, exon_start + 2):
                    effects.add("splice_region_variant")
            if exon_end < model.end:
                if overlaps(exon_end + 1, exon_end + 2):
                    effects.add("splice_donor_variant")
                if overlaps(exon_end - 2, exon_end) or overlaps(exon_end + 3, exon_end + 8):
                    effects.add("splice_region_variant")
        else:
            if exon_start > model.start:
                if overlaps(exon_start - 2, exon_start - 1):
                    effects.add("splice_donor_variant")
                if overlaps(exon_start - 8, exon_start - 3) or overlaps(exon_start, exon_start + 2):
                    effects.add("splice_region_variant")
            if exon_end < model.end:
                if overlaps(exon_end + 1, exon_end + 2):
                    effects.add("splice_acceptor_variant")
                if overlaps(exon_end - 2, exon_end) or overlaps(exon_end + 3, exon_end + 8):
                    effects.add("splice_region_variant")
    order = {"splice_acceptor_variant": 0, "splice_donor_variant": 1, "splice_region_variant": 2}
    return sorted(effects, key=order.__getitem__)


def _noncoding_region_consequence(model: _Transcript, start: int, end: int) -> str:
    """Classify non-translated transcript overlap only from explicit local structure."""
    explicit_utr = next(
        (term for utr_start, utr_end, term in model.utrs if utr_start <= end and start <= utr_end),
        None,
    )
    if explicit_utr:
        return explicit_utr
    exon_overlap = any(exon_start <= end and start <= exon_end for exon_start, exon_end in model.exons)
    if exon_overlap:
        if not model.cds:
            return "non_coding_transcript_exon_variant"
        coding_start = min(segment.start for segment in model.cds)
        coding_end = max(segment.end for segment in model.cds)
        if model.strand == "+":
            if end < coding_start:
                return "5_prime_UTR_variant"
            if start > coding_end:
                return "3_prime_UTR_variant"
        else:
            if start > coding_end:
                return "5_prime_UTR_variant"
            if end < coding_start:
                return "3_prime_UTR_variant"
        # The GFF3 says this is exonic but does not identify a UTR and the span
        # does not sit wholly outside the translated CDS bounds.
        return "transcript_variant"
    if model.exons:
        return "intron_variant"
    ordered_cds = sorted(model.cds, key=lambda segment: (segment.start, segment.end))
    if any(left.end < start and end < right.start for left, right in zip(ordered_cds, ordered_cds[1:])):
        return "intron_variant"
    # Transcript bounds alone do not distinguish introns from unannotated UTRs.
    return "transcript_variant"


def _protein_effect(
    model: _Transcript,
    cds_sequence: str,
    phase: int,
    cds_start: int,
    reference_allele: str,
    alternate_allele: str,
) -> dict[str, Any]:
    alternate_cds = cds_sequence[:cds_start] + alternate_allele + cds_sequence[cds_start + len(reference_allele):]
    reference_protein = _translate(cds_sequence[phase:], model.translation_table)
    alternate_protein = _translate(alternate_cds[phase:], model.translation_table)
    protein_offset = max(0, (cds_start - phase) // 3)
    first_difference = next((i for i, (left, right) in enumerate(zip(reference_protein, alternate_protein)) if left != right), None)
    if first_difference is None and len(reference_protein) != len(alternate_protein):
        first_difference = min(len(reference_protein), len(alternate_protein))
    if first_difference is None:
        amino_acid = reference_protein[protein_offset:protein_offset + 1]
        return {"consequences": ["synonymous_variant"], "protein_position": protein_offset + 1, "reference_amino_acids": amino_acid, "alternate_amino_acids": amino_acid}

    ref_aa = reference_protein[first_difference:first_difference + 1]
    alt_aa = alternate_protein[first_difference:first_difference + 1]
    length_delta = len(alternate_allele) - len(reference_allele)
    if cds_start <= phase + 2 and reference_protein.startswith("M") and not alternate_protein.startswith("M"):
        consequences = ["start_lost"]
    elif length_delta % 3:
        consequences = ["frameshift_variant"]
    elif length_delta > 0:
        consequences = ["inframe_insertion"]
    elif length_delta < 0:
        consequences = ["inframe_deletion"]
    elif alt_aa == "*" and ref_aa != "*":
        consequences = ["stop_gained"]
    elif ref_aa == "*" and alt_aa != "*":
        consequences = ["stop_lost"]
    elif alt_aa == ref_aa:
        consequences = ["synonymous_variant"]
    else:
        consequences = ["missense_variant"]
    return {
        "consequences": consequences,
        "protein_position": first_difference + 1,
        "reference_amino_acids": ref_aa,
        "alternate_amino_acids": alt_aa,
    }


def annotate_coding_variants(
    variants: list[dict[str, Any]],
    gff3_path: str | Path,
    reference: IndexedFasta,
    genome_build: str,
) -> dict[str, Any]:
    """Project reference-validated VCF alleles onto local GFF3 transcripts."""
    annotation_path = Path(gff3_path).expanduser()
    transcripts, annotation_hash, annotation_build = _parse_gff3(annotation_path)
    if annotation_build:
        from medai_science.genomics import _reference_build
        parsed_build = _reference_build(annotation_build)
        if parsed_build and parsed_build != genome_build:
            raise ValueError(f"selected genome build {genome_build} conflicts with GFF3 directive {annotation_build}")

    by_contig: dict[str, list[_Transcript]] = {}
    for model in transcripts.values():
        by_contig.setdefault(model.contig, []).append(model)
    indexes = {
        contig: _IntervalNode(sorted(models, key=lambda item: (item.start, item.end)))
        for contig, models in by_contig.items() if models
    }
    cds_cache: dict[str, tuple[str, bool, int]] = {}
    annotated = 0
    for variant in variants:
        variant["transcript_consequences"] = []
        if not variant.get("sequence_allele_supported") or not variant.get("reference_validated"):
            variant["transcript_annotation_status"] = "requires_reference_validated_sequence_allele"
            continue
        position = int(variant.get("normalized_position_1_based", variant["position_1_based"]))
        ref = str(variant.get("normalized_reference_allele", variant["reference_allele"]))
        alt = str(variant.get("normalized_alternate_allele", variant["alternate_allele"]))
        if "N" in ref or "N" in alt:
            variant["transcript_annotation_status"] = "ambiguous_sequence_allele_not_translated"
            continue
        start, end = position, position + len(ref) - 1
        interval_index = indexes.get(variant["contig"])
        matches = interval_index.query(start, end) if interval_index else []
        for model in matches:
            splice_effects = _splice_site_consequences(model, start, end)
            if not model.cds:
                term = splice_effects or [_noncoding_region_consequence(model, start, end)]
                variant["transcript_consequences"].append({"transcript_id": model.identifier, "gene_id": model.gene_id, "gene_name": model.gene_name, "consequences": term, "protein_effect_evaluated": False})
                continue
            cache = cds_cache.get(model.identifier)
            if cache is None:
                cache = _transcript_cds(model, reference)
                cds_cache[model.identifier] = cache
            cds, phase_consistent, first_phase = cache
            ordered_segments = sorted(model.cds, key=lambda item: (item.start, item.end), reverse=model.strand == "-")
            mapped_indices: list[int] = []
            offset = 0
            for segment in ordered_segments:
                low, high = max(start, segment.start), min(end, segment.end)
                if low <= high:
                    if model.strand == "+":
                        mapped_indices.extend(offset + (coordinate - segment.start) for coordinate in range(low, high + 1))
                    else:
                        mapped_indices.extend(offset + (segment.end - coordinate) for coordinate in range(high, low - 1, -1))
                offset += segment.end - segment.start + 1
            coding_positions = len(mapped_indices) == len(ref) and sorted(mapped_indices) == list(range(min(mapped_indices), min(mapped_indices) + len(ref))) if mapped_indices else False
            if not coding_positions:
                overlaps_cds = any(segment.start <= end and start <= segment.end for segment in model.cds)
                consequences = splice_effects or (["coding_sequence_variant"] if overlaps_cds else [_noncoding_region_consequence(model, start, end)])
                variant["transcript_consequences"].append({"transcript_id": model.identifier, "gene_id": model.gene_id, "gene_name": model.gene_name, "consequences": consequences, "protein_effect_evaluated": False, "limitation": "variant crosses a CDS boundary or includes non-CDS bases" if overlaps_cds else None})
                continue
            cds_start = min(mapped_indices)
            transcript_ref = cds[cds_start:cds_start + len(ref)]
            oriented_ref = ref if model.strand == "+" else _reverse_complement(ref)
            oriented_alt = alt if model.strand == "+" else _reverse_complement(alt)
            if transcript_ref != oriented_ref:
                variant["transcript_consequences"].append({"transcript_id": model.identifier, "gene_id": model.gene_id, "gene_name": model.gene_name, "consequences": ["reference_transcript_mismatch"], "protein_effect_evaluated": False})
                continue
            if not phase_consistent:
                consequences = {"consequences": ["coding_sequence_variant"], "protein_effect_evaluated": False, "limitation": "GFF3 CDS phase chain is inconsistent"}
            elif model.translation_table not in {1, 2}:
                consequences = {"consequences": ["coding_sequence_variant"], "protein_effect_evaluated": False, "limitation": f"translation table {model.translation_table} is unsupported"}
            elif cds_start < first_phase:
                consequences = {"consequences": ["coding_sequence_variant"], "protein_effect_evaluated": False, "limitation": "variant falls in the partial leading CDS phase"}
            else:
                effect = _protein_effect(model, cds, first_phase, cds_start, oriented_ref, oriented_alt)
                consequences = {**effect, "protein_effect_evaluated": True}
            if "splice_region_variant" in splice_effects:
                terms = consequences.get("consequences", [])
                if "splice_region_variant" not in terms:
                    consequences["consequences"] = [*terms, "splice_region_variant"]
            consequences.update({"transcript_id": model.identifier, "gene_id": model.gene_id, "gene_name": model.gene_name, "cds_position": f"{cds_start + 1}-{cds_start + len(ref)}", "strand": model.strand, "translation_table": model.translation_table})
            variant["transcript_consequences"].append(consequences)
        if variant["transcript_consequences"]:
            variant["transcript_annotation_status"] = "annotated_from_local_gff3"
            annotated += 1
        else:
            variant["transcript_annotation_status"] = "no_transcript_overlap_in_supplied_annotation"

    return {
        "source": "local GFF3 annotation",
        "gff3_sha256": annotation_hash,
        "gff3_genome_build_directive": annotation_build,
        "transcript_models_loaded": len(transcripts),
        "transcripts_with_cds": sum(bool(model.cds) for model in transcripts.values()),
        "variants_with_transcript_overlap": annotated,
        "splice_site_definition": "Canonical donor/acceptor terms use the two intronic bases at an explicitly annotated exon junction. splice_region_variant uses the adjacent 1-3 exonic or 3-8 intronic bases. Strand determines transcript orientation; this is positional annotation, not a splice-effect prediction.",
        "method": "local transcript coordinates and direct reference sequence translation; no pathogenicity scoring",
    }
