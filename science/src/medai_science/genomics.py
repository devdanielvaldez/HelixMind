"""Local, provenance-preserving VCF inspection; no remote annotation calls."""
from __future__ import annotations

import gzip
import hashlib
import math
import re
import zlib
from pathlib import Path
from typing import Any, TextIO

_SEQUENCE_ALLELE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)
_NUCLEOTIDE = re.compile(r"^[ACGTRYSWKMBDHVNU]+$", re.IGNORECASE)
_PROTEIN = re.compile(r"^[ABCDEFGHIKLMNPQRSTVWXYZJUO*-]+$", re.IGNORECASE)
_AMBIGUOUS_NUCLEOTIDES = frozenset("RYSWKMBDHVN")
_MAX_VARIANTS = 2_000_000
_MAX_GENOTYPE_CALLS = 5_000_000
_MAX_LINE_BYTES = 2_000_000
_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def _open_text(path: Path) -> TextIO:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="")
    return path.open("rt", encoding="utf-8", errors="strict", newline="")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_build(reference: str) -> str | None:
    normalized = reference.lower()
    if re.search(r"(?:^|[^a-z0-9])(?:grch38|hg38)(?:$|[^a-z0-9])", normalized):
        return "GRCh38"
    if re.search(r"(?:^|[^a-z0-9])(?:grch37|hg19)(?:$|[^a-z0-9])", normalized):
        return "GRCh37"
    return None


class IndexedFasta:
    """Create a transient faidx-compatible index for a regular, wrapped FASTA."""

    max_bytes = 8 * 1024 * 1024 * 1024

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not self.path.is_file() or self.path.stat().st_size > self.max_bytes:
            raise ValueError("reference FASTA must be an uncompressed regular file no larger than 8 GiB")
        with self.path.open("rb") as probe:
            if probe.read(2) == b"\x1f\x8b":
                raise ValueError("reference FASTA must be uncompressed; use a local .fa/.fasta file")
        self.contigs: dict[str, tuple[int, int, int, int]] = {}
        self._handle = None
        digest = hashlib.sha256()
        signature = self._signature()
        try:
            with self.path.open("rb") as handle:
                name: str | None = None
                length = 0
                sequence_offset = 0
                line_bases = 0
                line_width = 0
                previous_line_bases = 0
                while True:
                    offset = handle.tell()
                    raw = handle.readline(_MAX_LINE_BYTES + 1)
                    if not raw:
                        break
                    digest.update(raw)
                    if len(raw) > _MAX_LINE_BYTES:
                        raise ValueError("reference FASTA line exceeds the 2 MB safety limit")
                    line = raw.rstrip(b"\r\n")
                    if line.startswith(b">"):
                        if name is not None:
                            self._store_contig(name, length, sequence_offset, line_bases, line_width)
                        header = line[1:].strip().split(maxsplit=1)
                        if not header:
                            raise ValueError("reference FASTA contains an empty contig identifier")
                        try:
                            name = header[0].decode("utf-8", "strict")
                        except UnicodeError as error:
                            raise ValueError("reference FASTA contig identifiers must be UTF-8") from error
                        if not name:
                            raise ValueError("reference FASTA contains an empty contig identifier")
                        length = 0
                        sequence_offset = handle.tell()
                        line_bases = 0
                        line_width = 0
                        previous_line_bases = 0
                        continue
                    if name is None:
                        raise ValueError("reference FASTA sequence appeared before its first header")
                    if not line or any(base not in b"ACGTNacgtn" for base in line):
                        raise ValueError(f"reference FASTA has an empty line or unsupported base near byte {offset}")
                    if line_bases == 0:
                        line_bases = len(line)
                        line_width = len(raw)
                    else:
                        if previous_line_bases != line_bases:
                            raise ValueError(f"reference contig {name} has sequence lines after a short terminal line")
                        if len(line) > line_bases:
                            raise ValueError(f"reference contig {name} has inconsistent line wrapping")
                        if len(line) == line_bases and len(raw) != line_width and len(raw) != line_bases:
                            raise ValueError(f"reference contig {name} has inconsistent line endings or wrapping")
                    previous_line_bases = len(line)
                    length += len(line)
                if name is not None:
                    self._store_contig(name, length, sequence_offset, line_bases, line_width)
            if not self.contigs:
                raise ValueError("reference FASTA contains no contigs")
            if self._signature() != signature:
                raise ValueError("reference FASTA changed while its index was being built")
            self.sha256 = digest.hexdigest()
            self._handle = self.path.open("rb")
            self._signature_at_index = signature
        except Exception:
            self.close()
            raise

    def _signature(self) -> tuple[int, int, int, int]:
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _store_contig(self, name: str, length: int, offset: int, line_bases: int, line_width: int) -> None:
        if not length or not line_bases or not line_width:
            raise ValueError(f"reference contig {name} has no sequence")
        if name in self.contigs:
            raise ValueError(f"duplicate reference contig identifier: {name}")
        self.contigs[name] = (length, offset, line_bases, line_width)

    def fetch(self, contig: str, start_zero_based: int, length: int) -> str:
        if self._handle is None:
            raise ValueError("reference FASTA is closed")
        try:
            contig_length, offset, line_bases, line_width = self.contigs[contig]
        except KeyError as error:
            raise ValueError(f"VCF contig {contig!r} is absent from the reference FASTA") from error
        if start_zero_based < 0 or length < 0 or start_zero_based + length > contig_length:
            raise ValueError(f"reference interval is outside contig {contig}")
        result = bytearray()
        cursor = start_zero_based
        remaining = length
        while remaining:
            line_index, column = divmod(cursor, line_bases)
            count = min(remaining, line_bases - column)
            self._handle.seek(offset + line_index * line_width + column)
            chunk = self._handle.read(count)
            if len(chunk) != count or any(base not in b"ACGTNacgtn" for base in chunk):
                raise ValueError(f"reference FASTA index is inconsistent near {contig}:{cursor + 1}")
            result.extend(chunk.upper())
            cursor += count
            remaining -= count
        return result.decode("ascii")

    def verify_unchanged(self) -> None:
        if self._signature() != self._signature_at_index or _digest(self.path) != self.sha256:
            raise ValueError("reference FASTA changed during VCF analysis")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


def _trim_alleles(position: int, reference: str, alternate: str) -> tuple[int, str, str]:
    while len(reference) > 1 and len(alternate) > 1 and reference[-1] == alternate[-1]:
        reference, alternate = reference[:-1], alternate[:-1]
    while len(reference) > 1 and len(alternate) > 1 and reference[0] == alternate[0]:
        reference, alternate = reference[1:], alternate[1:]
        position += 1
    return position, reference, alternate


def _normalize_alleles(reference_fasta: IndexedFasta, contig: str, position: int, reference: str, alternate: str) -> tuple[int, str, str]:
    position, reference, alternate = _trim_alleles(position, reference, alternate)
    if len(reference) == len(alternate) or min(len(reference), len(alternate)) > 1:
        return position, reference, alternate
    while position > 1 and reference[-1] == alternate[-1]:
        previous = reference_fasta.fetch(contig, position - 2, 1)
        # Unknown reference symbols are not evidence that an indel can be
        # shifted through that base. Keep normalization within verified DNA.
        if not re.fullmatch(r"[ACGT]", previous):
            break
        reference = previous + reference
        alternate = previous + alternate
        position -= 1
        while len(reference) > 1 and len(alternate) > 1 and reference[-1] == alternate[-1]:
            reference, alternate = reference[:-1], alternate[:-1]
        position, reference, alternate = _trim_alleles(position, reference, alternate)
    return position, reference, alternate


def _genotype_counts(format_value: str, sample_values: list[str], alternate_index: int, alternate_count: int) -> dict[str, int] | None:
    keys = format_value.split(":") if format_value else []
    if "GT" not in keys:
        return None
    genotype_position = keys.index("GT")
    counts = {
        "diploid_alt_dosage_0": 0,
        "diploid_alt_dosage_1": 0,
        "diploid_alt_dosage_2": 0,
        "haploid_alt_dosage_0": 0,
        "haploid_alt_dosage_1": 0,
        "other_or_ploidy": 0,
        "missing": 0,
    }
    for sample in sample_values:
        fields = sample.split(":")
        gt = fields[genotype_position] if genotype_position < len(fields) else "."
        if "/" in gt and "|" in gt:
            raise ValueError(f"genotype cannot mix phased and unphased separators: {gt}")
        alleles = re.split(r"[/|]", gt)
        if not alleles or any(value == "." for value in alleles):
            counts["missing"] += 1
            continue
        if any(not value.isascii() or not value.isdecimal() or int(value) > alternate_count for value in alleles):
            raise ValueError(f"invalid genotype allele index: {gt}")
        dosage = sum(int(value) == alternate_index for value in alleles)
        if len(alleles) == 1:
            counts[f"haploid_alt_dosage_{dosage}"] += 1
        elif len(alleles) == 2:
            counts[f"diploid_alt_dosage_{dosage}"] += 1
        else:
            counts["other_or_ploidy"] += 1
    return counts


def analyze_vcf(
    path: str | Path,
    genome_build: str,
    max_variants: int = _MAX_VARIANTS,
    reference_fasta: str | Path | None = None,
    transcript_gff3: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a VCF and optionally verify/normalize alleles against local FASTA."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError("VCF input must be an existing regular file")
    if source.stat().st_size > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("VCF input exceeds the 2 GiB file-size safety limit")
    if genome_build not in {"GRCh37", "GRCh38"}:
        raise ValueError("genome_build is required and must be GRCh37 or GRCh38")
    if transcript_gff3 is not None and reference_fasta is None:
        raise ValueError("transcript consequence mapping requires a local --reference-fasta")
    if not 1 <= max_variants <= _MAX_VARIANTS:
        raise ValueError(f"max_variants must be 1..{_MAX_VARIANTS}")

    reference = IndexedFasta(reference_fasta) if reference_fasta is not None else None
    initial_digest = _digest(source)
    variants: list[dict[str, Any]] = []
    header_columns: list[str] | None = None
    file_format: str | None = None
    declared_reference: str | None = None
    total_records = 0
    total_alternate_alleles = 0
    total_genotype_calls = 0
    total_bytes = 0
    unsupported_records = 0
    samples_count = 0
    try:
        with _open_text(source) as handle:
            line_number = 0
            while True:
                raw_line = handle.readline(_MAX_LINE_BYTES + 1)
                if not raw_line:
                    break
                line_number += 1
                raw_size = len(raw_line.encode("utf-8"))
                total_bytes += raw_size
                if raw_size > _MAX_LINE_BYTES:
                    raise ValueError(f"line {line_number} exceeds the {_MAX_LINE_BYTES}-byte safety limit")
                if total_bytes > _MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("decompressed VCF data exceeds the 2 GiB safety limit")
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if header_columns is not None and line.startswith("##"):
                    raise ValueError(f"VCF metadata appears after the #CHROM header at line {line_number}")
                if line.startswith("##fileformat="):
                    if line_number != 1 or file_format is not None:
                        raise ValueError("VCF fileformat declaration must be the first line and appear once")
                    file_format = line.partition("=")[2]
                    if not re.fullmatch(r"VCFv4\.\d+", file_format):
                        raise ValueError("unsupported or malformed VCF fileformat declaration")
                    continue
                if line.startswith("##reference="):
                    declared_reference = line.partition("=")[2].strip()
                    continue
                if line.startswith("#CHROM\t"):
                    if file_format is None:
                        raise ValueError("VCF is missing its required first-line ##fileformat declaration")
                    if header_columns is not None:
                        raise ValueError("VCF contains more than one #CHROM header")
                    header_columns = line.split("\t")
                    if len(header_columns) < 8 or header_columns[:8] != ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]:
                        raise ValueError("VCF #CHROM header does not match the required eight columns")
                    if len(header_columns) == 9:
                        raise ValueError("VCF FORMAT column requires at least one sample column")
                    if len(header_columns) >= 10 and header_columns[8] != "FORMAT":
                        raise ValueError("ninth VCF column must be FORMAT when samples are present")
                    sample_names = header_columns[9:] if len(header_columns) >= 10 else []
                    if any(not sample for sample in sample_names) or len(sample_names) != len(set(sample_names)):
                        raise ValueError("VCF sample column names must be non-empty and unique")
                    samples_count = max(0, len(header_columns) - 9)
                    continue
                if line.startswith("#"):
                    if not line.startswith("##"):
                        raise ValueError(f"unexpected VCF header line at line {line_number}")
                    continue
                if header_columns is None:
                    raise ValueError("VCF data appeared before the #CHROM header")
                if total_records >= max_variants:
                    raise ValueError(f"VCF exceeds the configured maximum of {max_variants} records")
                columns = line.split("\t")
                if len(columns) != len(header_columns):
                    raise ValueError(f"VCF line {line_number} has {len(columns)} columns; expected {len(header_columns)}")
                chrom, pos_text, _identifier, ref, alt_text, qual_text, filter_value, info_text = columns[:8]
                if not chrom or len(chrom) > 128 or any(char.isspace() for char in chrom):
                    raise ValueError(f"invalid contig at line {line_number}")
                if not pos_text.isascii() or not pos_text.isdecimal():
                    raise ValueError(f"invalid VCF position at line {line_number}")
                try:
                    position = int(pos_text)
                except ValueError as error:
                    raise ValueError(f"invalid VCF position at line {line_number}") from error
                if position < 1:
                    raise ValueError(f"VCF position must be positive at line {line_number}")
                if not _SEQUENCE_ALLELE.fullmatch(ref):
                    raise ValueError(f"invalid reference allele at line {line_number}")
                if reference is not None:
                    observed_reference = reference.fetch(chrom, position - 1, len(ref))
                    if observed_reference != ref.upper():
                        raise ValueError(
                            f"VCF REF {ref.upper()} does not match reference {observed_reference} at {chrom}:{position}"
                        )
                alternate_count = alt_text.count(",") + 1
                if total_alternate_alleles + alternate_count > max_variants:
                    raise ValueError(f"VCF exceeds the configured maximum of {max_variants} alternate alleles")
                alternates = alt_text.split(",")
                if not alternates or any(not alt for alt in alternates):
                    raise ValueError(f"empty alternate allele at line {line_number}")
                quality = None if qual_text == "." else float(qual_text)
                if quality is not None and (not math.isfinite(quality) or quality < 0):
                    raise ValueError(f"invalid QUAL value at line {line_number}")
                passed_filter = filter_value in {"PASS", "."}
                info_fields = sorted({item.split("=", 1)[0] for item in info_text.split(";") if item and item != "."})
                format_value = columns[8] if len(columns) > 8 else ""
                sample_values = columns[9:] if len(columns) > 9 else []
                if sample_values and "GT" in format_value.split(":"):
                    record_genotype_calls = len(sample_values) * alternate_count
                    if total_genotype_calls + record_genotype_calls > _MAX_GENOTYPE_CALLS:
                        raise ValueError(f"VCF genotype calls exceed the {_MAX_GENOTYPE_CALLS} safety limit")
                    total_genotype_calls += record_genotype_calls
                total_records += 1
                total_alternate_alleles += len(alternates)
                for alternate_index, alternate in enumerate(alternates, 1):
                    is_sequence_variant = bool(_SEQUENCE_ALLELE.fullmatch(alternate))
                    if not is_sequence_variant:
                        unsupported_records += 1
                    variant = {
                        "record_number": total_records,
                        "alternate_index": alternate_index,
                        "contig": chrom,
                        "position_1_based": position,
                        "reference_allele": ref.upper(),
                        "alternate_allele": alternate.upper(),
                        "variant_class": "SNV" if is_sequence_variant and len(ref) == len(alternate) == 1 else "small_indel_or_mnv" if is_sequence_variant else "unsupported_symbolic_or_breakend",
                        "sequence_allele_supported": is_sequence_variant,
                        "quality": quality,
                        "filter": filter_value,
                        "passes_vcf_filter": passed_filter,
                        "info_field_names": info_fields,
                    }
                    if reference is not None:
                        if is_sequence_variant:
                            normalized_position, normalized_ref, normalized_alt = _normalize_alleles(
                                reference, chrom, position, ref.upper(), alternate.upper()
                            )
                            normalized_reference = reference.fetch(chrom, normalized_position - 1, len(normalized_ref))
                            if normalized_reference != normalized_ref:
                                raise ValueError(
                                    f"normalized REF does not match reference at {chrom}:{normalized_position}"
                                )
                            variant.update({
                                "reference_validated": True,
                                "normalized_position_1_based": normalized_position,
                                "normalized_reference_allele": normalized_ref,
                                "normalized_alternate_allele": normalized_alt,
                                "normalization_status": (
                                    "reference_checked_left_aligned_minimal"
                                    if len(normalized_ref) != len(normalized_alt) and min(len(normalized_ref), len(normalized_alt)) == 1
                                    else "reference_checked_minimal_complex_allele_not_left_aligned"
                                    if len(normalized_ref) != len(normalized_alt)
                                    else "reference_checked_minimal"
                                ),
                            })
                        else:
                            variant["reference_validated"] = True
                            variant["normalization_status"] = "symbolic_or_breakend_not_normalized"
                    if sample_values:
                        variant["genotype_call_counts"] = _genotype_counts(format_value, sample_values, alternate_index, len(alternates))
                    variants.append(variant)
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise ValueError(f"could not read VCF input: {error}") from error

    if header_columns is None:
        raise ValueError("VCF #CHROM header was not found")
    if file_format is None:
        raise ValueError("VCF fileformat declaration was not found on the first line")
    declared_build = _reference_build(declared_reference or "")
    if declared_build and declared_build != genome_build:
        raise ValueError(f"requested genome build {genome_build} conflicts with VCF reference {declared_reference}")
    if reference is not None:
        reference.verify_unchanged()
        reference_contigs = len(reference.contigs)
        reference_sha256 = reference.sha256
        reference.close()
        normalization_limitations = []
    else:
        reference_contigs = None
        reference_sha256 = None
        normalization_limitations = [
            "No reference FASTA was supplied; REF alleles were not verified and variants were not normalized."
        ]
    final_digest = _digest(source)
    if final_digest != initial_digest:
        raise ValueError("VCF input changed while it was being analyzed; retry with a stable local file")
    report = {
        "analysis": "local VCF parsing and validation",
        "file_format": file_format,
        "input_sha256": initial_digest,
        "genome_build": genome_build,
        "genome_build_source": "explicit_user_selection",
        "analysis_implementation_sha256": {
            "__init__.py": _digest(Path(__file__).resolve().with_name("__init__.py")),
            "genomics.py": _digest(Path(__file__).resolve()),
            "transcripts.py": _digest(Path(__file__).resolve().with_name("transcripts.py")),
        },
        "reference_fasta_sha256": reference_sha256,
        "reference_contigs_indexed": reference_contigs,
        "reference_validated": reference is not None,
        "declared_reference": declared_reference,
        "declared_reference_match": declared_build == genome_build if declared_build else None,
        "sample_count": samples_count,
        "sample_identifiers_retained": False,
        "genotype_count_semantics": (
            "Per-alternate allele counts; haploid and diploid dosages are reported separately, missing calls are separate, and ploidy above two is grouped as other_or_ploidy."
            if samples_count else None
        ),
        "records_parsed": total_records,
        "alternate_alleles_emitted": len(variants),
        "genotype_calls_examined": total_genotype_calls,
        "unsupported_symbolic_or_breakend_alleles": unsupported_records,
        "remote_services_contacted": False,
        "clinical_interpretation": "not_performed",
        "variants": variants,
        "limitations": [
            *normalization_limitations,
            *(["Transcript consequence mapping was performed against the supplied local GFF3; no pathogenicity classification was performed."] if transcript_gff3 else ["Transcript consequence mapping and pathogenicity classification are not performed."]),
            "Symbolic structural variants and breakends are reported as unsupported; complex unequal-length alleles are not left-aligned.",
            "This output is a file-structure analysis, not clinical interpretation or pathogenicity classification.",
        ],
    }
    if transcript_gff3:
        annotation_reference = IndexedFasta(reference_fasta)
        try:
            if annotation_reference.sha256 != reference_sha256:
                raise ValueError("reference FASTA changed between VCF validation and transcript annotation")
            from medai_science.transcripts import annotate_coding_variants
            report["transcript_annotation"] = annotate_coding_variants(
                variants, transcript_gff3, annotation_reference, genome_build
            )
        finally:
            annotation_reference.close()
    return report


def _sequence_summary(input_sha256: str, format_name: str, records: int, lengths: list[int], bases: dict[str, int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    total_bases = sum(lengths)
    result = {
        "analysis": f"local {format_name} structural validation",
        "input_sha256": input_sha256,
        "records": records,
        "total_sequence_symbols": total_bases,
        "minimum_record_length": min(lengths) if lengths else 0,
        "maximum_record_length": max(lengths) if lengths else 0,
        "mean_record_length": round(total_bases / records, 3) if records else 0,
        "sequence_symbol_counts": dict(sorted(bases.items())),
        "record_identifiers_retained": False,
        "sequence_content_retained": False,
        "remote_services_contacted": False,
    }
    if extra:
        result.update(extra)
    return result


def _nucleotide_composition(bases: dict[str, int], total_symbols: int) -> dict[str, Any]:
    """Report IUPAC ambiguity separately from GC measured over all sequence symbols."""
    ambiguous = {symbol: bases[symbol] for symbol in sorted(_AMBIGUOUS_NUCLEOTIDES) if bases.get(symbol, 0)}
    ambiguous_count = sum(ambiguous.values())
    canonical_count = sum(bases.get(symbol, 0) for symbol in "ACGTU")
    gc_count = bases.get("G", 0) + bases.get("C", 0)
    return {
        "gc_fraction": round(gc_count / total_symbols, 6) if total_symbols else None,
        "gc_fraction_unambiguous": round(gc_count / canonical_count, 6) if canonical_count else None,
        "gc_fraction_denominator": "all_sequence_symbols_including_IUPAC_ambiguity",
        "gc_fraction_unambiguous_denominator": "A_C_G_T_U_symbols_only",
        "ambiguous_symbol_counts": ambiguous,
        "ambiguous_symbol_count": ambiguous_count,
        "ambiguous_symbol_fraction": round(ambiguous_count / total_symbols, 6) if total_symbols else None,
    }


def inspect_fasta(path: str | Path, molecule_type: str = "nucleotide", max_records: int = _MAX_VARIANTS) -> dict[str, Any]:
    """Validate FASTA records and compute local aggregate sequence statistics."""
    source = Path(path).expanduser()
    if not source.is_file() or source.stat().st_size > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("FASTA input must be a regular file no larger than 2 GiB")
    if molecule_type not in {"nucleotide", "protein"}:
        raise ValueError("molecule_type must be nucleotide or protein")
    if not 1 <= max_records <= _MAX_VARIANTS:
        raise ValueError(f"max_records must be 1..{_MAX_VARIANTS}")
    alphabet = _NUCLEOTIDE if molecule_type == "nucleotide" else _PROTEIN
    canonical_protein_symbols = set("ACDEFGHIKLMNPQRSTVWY")
    records = 0
    total_bytes = 0
    current_length = 0
    lengths: list[int] = []
    bases: dict[str, int] = {}
    header_seen = False
    initial_digest = _digest(source)
    try:
        with _open_text(source) as handle:
            line_number = 0
            while True:
                raw = handle.readline(_MAX_LINE_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                raw_size = len(raw.encode("utf-8"))
                total_bytes += raw_size
                if raw_size > _MAX_LINE_BYTES or total_bytes > _MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("FASTA exceeds a line or 2 GiB decompressed-size safety limit")
                line = raw.rstrip("\r\n")
                if line.startswith(">"):
                    if not line[1:].strip():
                        raise ValueError(f"empty FASTA identifier at line {line_number}")
                    if header_seen:
                        if current_length == 0:
                            raise ValueError(f"empty FASTA sequence before line {line_number}")
                        lengths.append(current_length)
                    records += 1
                    if records > max_records:
                        raise ValueError(f"FASTA exceeds the configured maximum of {max_records} records")
                    current_length = 0
                    header_seen = True
                    continue
                if not line.strip():
                    continue
                if not header_seen:
                    raise ValueError(f"FASTA sequence appeared before its header at line {line_number}")
                sequence = "".join(line.split()).upper()
                if not sequence or not alphabet.fullmatch(sequence):
                    raise ValueError(f"invalid {molecule_type} sequence symbols at line {line_number}")
                current_length += len(sequence)
                for char in sequence:
                    key = char if molecule_type == "nucleotide" or char in canonical_protein_symbols else "ambiguous_or_other"
                    bases[key] = bases.get(key, 0) + 1
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise ValueError(f"could not read FASTA input: {error}") from error
    if not header_seen:
        raise ValueError("FASTA contains no records")
    if current_length == 0:
        raise ValueError("last FASTA record has an empty sequence")
    lengths.append(current_length)
    final_digest = _digest(source)
    if final_digest != initial_digest:
        raise ValueError("FASTA input changed while it was being analyzed")
    extra: dict[str, Any] = {"molecule_type": molecule_type}
    if molecule_type == "nucleotide":
        extra.update(_nucleotide_composition(bases, sum(lengths)))
    return _sequence_summary(initial_digest, "FASTA", records, lengths, bases, extra)


def inspect_fastq(path: str | Path, max_records: int = _MAX_VARIANTS) -> dict[str, Any]:
    """Validate wrapped FASTQ records, Phred+33 bytes, and sequence/quality lengths."""
    source = Path(path).expanduser()
    if not source.is_file() or source.stat().st_size > _MAX_UNCOMPRESSED_BYTES:
        raise ValueError("FASTQ input must be a regular file no larger than 2 GiB")
    if not 1 <= max_records <= _MAX_VARIANTS:
        raise ValueError(f"max_records must be 1..{_MAX_VARIANTS}")
    records = 0
    total_bytes = 0
    lengths: list[int] = []
    bases: dict[str, int] = {}
    quality_count = 0
    quality_sum = 0
    quality_min: int | None = None
    quality_max: int | None = None
    initial_digest = _digest(source)

    def read_line(handle: TextIO, line_number: int) -> tuple[str | None, int]:
        nonlocal total_bytes
        raw = handle.readline(_MAX_LINE_BYTES + 1)
        if raw == "":
            return None, line_number
        line_number += 1
        raw_size = len(raw.encode("utf-8"))
        total_bytes += raw_size
        if raw_size > _MAX_LINE_BYTES or total_bytes > _MAX_UNCOMPRESSED_BYTES:
            raise ValueError("FASTQ exceeds a line or 2 GiB decompressed-size safety limit")
        return raw.rstrip("\r\n"), line_number

    try:
        with _open_text(source) as handle:
            line_number = 0
            while True:
                header, line_number = read_line(handle, line_number)
                if header is None:
                    break
                if not header.startswith("@") or not header[1:].strip():
                    raise ValueError(f"expected a FASTQ @header at line {line_number}")
                if records >= max_records:
                    raise ValueError(f"FASTQ exceeds the configured maximum of {max_records} records")
                identifier = header[1:].split(maxsplit=1)[0]
                sequence_length = 0
                sequence_counts: dict[str, int] = {}
                while True:
                    line, line_number = read_line(handle, line_number)
                    if line is None:
                        raise ValueError(f"FASTQ record {records + 1} ended before its + separator")
                    if line.startswith("+"):
                        repeated = line[1:].split(maxsplit=1)[0] if line[1:].strip() else ""
                        if repeated and repeated != identifier:
                            raise ValueError(f"FASTQ + identifier does not match @header at line {line_number}")
                        break
                    sequence = line.upper()
                    if not sequence or not _NUCLEOTIDE.fullmatch(sequence):
                        raise ValueError(f"invalid nucleotide symbols in FASTQ record {records + 1}")
                    sequence_length += len(sequence)
                    for char in sequence:
                        sequence_counts[char] = sequence_counts.get(char, 0) + 1
                if sequence_length == 0:
                    raise ValueError(f"FASTQ record {records + 1} has an empty sequence")
                observed_quality = 0
                while observed_quality < sequence_length:
                    quality_line, line_number = read_line(handle, line_number)
                    if quality_line is None:
                        raise ValueError(f"FASTQ record {records + 1} has truncated quality values")
                    if not quality_line:
                        raise ValueError(f"FASTQ record {records + 1} has an empty quality line")
                    if any(not 33 <= ord(char) <= 126 for char in quality_line):
                        raise ValueError(f"FASTQ quality at line {line_number} is outside printable Phred+33 range")
                    remaining = sequence_length - observed_quality
                    if len(quality_line) > remaining:
                        raise ValueError(f"FASTQ quality is longer than its sequence in record {records + 1}")
                    scores = [ord(char) - 33 for char in quality_line]
                    observed_quality += len(scores)
                    quality_count += len(scores)
                    quality_sum += sum(scores)
                    if scores:
                        low, high = min(scores), max(scores)
                        quality_min = low if quality_min is None else min(quality_min, low)
                        quality_max = high if quality_max is None else max(quality_max, high)
                records += 1
                lengths.append(sequence_length)
                for char, count in sequence_counts.items():
                    bases[char] = bases.get(char, 0) + count
    except (OSError, UnicodeError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        raise ValueError(f"could not read FASTQ input: {error}") from error
    if records == 0:
        raise ValueError("FASTQ contains no records")
    final_digest = _digest(source)
    if final_digest != initial_digest:
        raise ValueError("FASTQ input changed while it was being analyzed")
    nucleotide_metrics = _nucleotide_composition(bases, sum(lengths))
    return _sequence_summary(
        initial_digest,
        "FASTQ",
        records,
        lengths,
        bases,
        {
            "molecule_type": "nucleotide",
            **nucleotide_metrics,
            "quality_encoding": "Phred+33 (required) — not inferred",
            "quality_scores_counted": quality_count,
            "mean_phred_score": round(quality_sum / quality_count, 4) if quality_count else None,
            "minimum_phred_score": quality_min,
            "maximum_phred_score": quality_max,
        },
    )
