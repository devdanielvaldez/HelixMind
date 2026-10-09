"""Bounded, source-neutral structural summaries for local PDB and mmCIF files."""
from __future__ import annotations

import hashlib
import io
import math
import re
from collections import defaultdict
from typing import Any


MAX_PDB_BYTES = 32 * 1024 * 1024
MAX_MMCIF_BYTES = 32 * 1024 * 1024
MAX_ATOMS = 400_000
_AMINO_ACIDS = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def summarize_pdb(payload: bytes, *, label: str | None = None) -> dict[str, Any]:
    """Validate fixed-column coordinate records and return hash-linked aggregates.

    Coordinates and atom records are intentionally not copied to the result.
    PDB temperature factors are reported as source values without assuming they
    are AlphaFold pLDDT or another model-confidence measure.
    """
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_PDB_BYTES:
        raise ValueError(f"PDB input must contain 1..{MAX_PDB_BYTES} bytes")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("PDB input must be ASCII text") from error
    if any((ord(character) < 32 and character not in "\r\n") or ord(character) == 127 for character in text):
        raise ValueError("PDB input contains a forbidden control character")
    if label is not None and (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", label)):
        raise ValueError("structure label must contain 1..128 safe identifier characters")

    explicit_models = False
    model = 1
    model_ids: set[int] = set()
    atoms: set[tuple[int, str, int, str, str]] = set()
    residues: dict[tuple[int, str], dict[tuple[int, str], str]] = defaultdict(dict)
    alpha_carbons: set[tuple[int, str, int, str]] = set()
    factors: list[float] = []
    atom_count = 0
    chain_ids: set[str] = set()
    discarded_alternate_atom_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if len(line) > 512:
            raise ValueError(f"PDB line {line_number} exceeds 512 characters")
        record = line[:6].strip().upper()
        if record == "MODEL":
            if not line[10:14].strip().isdigit():
                raise ValueError(f"PDB MODEL record on line {line_number} has an invalid model number")
            explicit_models = True
            model = int(line[10:14])
            if model < 1 or model in model_ids:
                raise ValueError(f"PDB MODEL record on line {line_number} has a zero or repeated model number")
            model_ids.add(model)
            continue
        if record != "ATOM":
            continue
        if len(line) < 66:
            raise ValueError(f"PDB ATOM record on line {line_number} is truncated")
        atom_name = line[12:16].strip()
        alternate = line[16].strip()
        residue_name = line[17:20].strip().upper()
        chain = line[21].strip() or "_"
        residue_number_text = line[22:26].strip()
        insertion = line[26].strip()
        if (not atom_name or len(atom_name) > 4 or (alternate and not re.fullmatch(r"[A-Za-z0-9]", alternate))
                or not re.fullmatch(r"[A-Z]{3}", residue_name)
                or not re.fullmatch(r"-?[0-9]{1,4}", residue_number_text)
                or (insertion and not re.fullmatch(r"[A-Za-z0-9]", insertion))):
            raise ValueError(f"PDB ATOM record on line {line_number} has malformed identity fields")
        try:
            residue_number = int(residue_number_text)
            coordinates = [float(line[start:start + 8]) for start in (30, 38, 46)]
            occupancy = float(line[54:60].strip() or "0")
            factor = float(line[60:66].strip() or "0")
        except ValueError as error:
            raise ValueError(f"PDB ATOM record on line {line_number} has invalid numeric fields") from error
        if any(not math.isfinite(value) for value in (*coordinates, occupancy, factor)):
            raise ValueError(f"PDB ATOM record on line {line_number} contains a non-finite number")
        if any(abs(value) > 10_000 for value in coordinates) or not 0 <= occupancy <= 1.1 or not -1 <= factor <= 1_000_000:
            raise ValueError(f"PDB ATOM record on line {line_number} has out-of-range coordinates or metadata")
        if alternate not in {"", "A"}:
            discarded_alternate_atom_count += 1
            continue

        atom_count += 1
        if atom_count > MAX_ATOMS:
            raise ValueError(f"PDB input exceeds {MAX_ATOMS} accepted atom records")
        chain_ids.add(chain)
        residue_key = (residue_number, insertion)
        residue_bucket = residues[(model, chain)]
        previous_name = residue_bucket.setdefault(residue_key, residue_name)
        if previous_name != residue_name:
            raise ValueError(f"PDB residue identity changes within model {model}, chain {chain}, position {residue_number}{insertion}")
        atom_key = (model, chain, residue_number, insertion, atom_name)
        if atom_key in atoms:
            raise ValueError(f"PDB contains a duplicate accepted atom record on line {line_number}")
        atoms.add(atom_key)
        if atom_name == "CA":
            alpha_carbons.add((model, chain, residue_number, insertion))
        factors.append(factor)

    if atom_count == 0:
        raise ValueError("PDB input contains no accepted ATOM coordinate records")
    observed_models = {key[0] for key in residues}
    if explicit_models and (model_ids != observed_models or model_ids != set(range(1, len(model_ids) + 1))):
        raise ValueError("PDB MODEL numbering is not a contiguous 1-based sequence")

    chain_summaries = []
    for (model_id, chain_id), chain_residues in sorted(residues.items()):
        sequence = "".join(_AMINO_ACIDS.get(name, "X") for _, name in sorted(chain_residues.items()))
        chain_summaries.append({
            "model": model_id,
            "chain": chain_id,
            "residue_count": len(chain_residues),
            "alpha_carbon_count": sum((model_id, chain_id, number, insertion) in alpha_carbons for number, insertion in chain_residues),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "unknown_residue_count": sequence.count("X"),
        })
    factors_sorted = sorted(factors)
    middle = len(factors_sorted) // 2
    median_factor = factors_sorted[middle] if len(factors_sorted) % 2 else (factors_sorted[middle - 1] + factors_sorted[middle]) / 2
    return {
        "schema_version": "1",
        "source": "user-supplied PDB coordinate file",
        "label": label,
        "input_bytes": len(payload),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "model_count": len({key[0] for key in residues}) if explicit_models else 1,
        "atom_count": atom_count,
        "chain_count": len(residues),
        "unique_chain_id_count": len(chain_ids),
        "discarded_alternate_atom_count": discarded_alternate_atom_count,
        "residue_count": sum(len(rows) for rows in residues.values()),
        "alpha_carbon_count": len(alpha_carbons),
        "chains": chain_summaries,
        "atom_temperature_factor": {
            "minimum": factors_sorted[0],
            "median": median_factor,
            "maximum": factors_sorted[-1],
            "interpretation": "Raw PDB temperature-factor field; it is not identified as pLDDT or calibrated confidence without source-specific provenance.",
        },
        "coordinates_retained": False,
        "interpretation_limits": [
            "This is a structural-file integrity and composition summary, not structure prediction, binding analysis, or functional interpretation.",
            "Only ATOM records and blank/A alternate locations are counted; HETATM records and other alternate conformations are omitted.",
            "Residue sequence digests include X for non-standard residue names; they do not establish protein identity or reference-sequence equivalence.",
        ],
    }


def summarize_mmcif(payload: bytes, *, label: str | None = None) -> dict[str, Any]:
    """Summarize an mmCIF atom-site loop using Biopython's CIF tokenizer."""
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_MMCIF_BYTES:
        raise ValueError(f"mmCIF input must contain 1..{MAX_MMCIF_BYTES} bytes")
    if label is not None and (not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", label)):
        raise ValueError("structure label must contain 1..128 safe identifier characters")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("mmCIF input must be valid UTF-8") from error
    if any((ord(character) < 32 and character not in "\t\r\n") or ord(character) == 127 for character in text):
        raise ValueError("mmCIF input contains a forbidden control character")
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as error:
        raise RuntimeError("install medai-science[bio] to analyze mmCIF coordinate files") from error
    try:
        data = MMCIF2Dict(io.StringIO(text))
    except (OSError, ValueError, KeyError, IndexError, TypeError, StopIteration) as error:
        raise ValueError("mmCIF input is malformed or has an invalid CIF token stream") from error

    def column(name: str, fallback: str | None = None, *, required: bool = True) -> list[str]:
        value = data.get("_atom_site." + name)
        if value is None and fallback is not None:
            value = data.get("_atom_site." + fallback)
        if value is None:
            if required:
                raise ValueError(f"mmCIF atom_site loop is missing {name}")
            return []
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not values or any(not isinstance(item, str) for item in values):
            raise ValueError(f"mmCIF atom_site column {name} is malformed")
        return values

    groups = column("group_PDB")
    count = len(groups)
    auth_chains = column("auth_asym_id", required=False)
    label_chains = column("label_asym_id", required=False)
    auth_sequences = column("auth_seq_id", required=False)
    label_sequences = column("label_seq_id", required=False)
    columns = {
        "atom": column("label_atom_id"),
        "alternate": column("label_alt_id", required=False),
        "residue": column("label_comp_id"),
        "chain": auth_chains,
        "label_chain": label_chains,
        "sequence": auth_sequences,
        "label_sequence": label_sequences,
        "insertion": column("pdbx_PDB_ins_code", required=False),
        "x": column("Cartn_x"),
        "y": column("Cartn_y"),
        "z": column("Cartn_z"),
        "occupancy": column("occupancy"),
        "factor": column("B_iso_or_equiv"),
        "model": column("pdbx_PDB_model_num", required=False),
    }
    for name, values in columns.items():
        if values and len(values) != count:
            raise ValueError(f"mmCIF atom_site column {name} does not match the row count")
        if not values:
            columns[name] = ["?"] * count
    if not auth_chains and not label_chains:
        raise ValueError("mmCIF atom_site loop is missing both author and label chain identifiers")
    if not auth_sequences and not label_sequences:
        raise ValueError("mmCIF atom_site loop is missing both author and label residue numbering")

    atom_count = 0
    model_ids: set[int] = set()
    chain_ids: set[str] = set()
    atoms: set[tuple[int, str, int, str, str]] = set()
    residues: dict[tuple[int, str], dict[tuple[int, str], str]] = defaultdict(dict)
    alpha_carbons: set[tuple[int, str, int, str]] = set()
    factors: list[float] = []
    discarded_alternates = 0
    for index, group in enumerate(groups):
        if group.upper() != "ATOM":
            continue
        atom_name = columns["atom"][index]
        alternate = columns["alternate"][index]
        residue_name = columns["residue"][index].upper()
        chain = columns["chain"][index]
        if chain in {".", "?", ""}:
            chain = columns["label_chain"][index]
        sequence_text = columns["sequence"][index]
        if sequence_text in {".", "?", ""}:
            sequence_text = columns["label_sequence"][index]
        insertion = columns["insertion"][index]
        try:
            model = int(columns["model"][index]) if columns["model"][index] not in {".", "?", ""} else 1
            residue_number = int(sequence_text)
            coordinates = [float(columns[axis][index]) for axis in ("x", "y", "z")]
            occupancy = float(columns["occupancy"][index]) if columns["occupancy"][index] not in {".", "?", ""} else 0.0
            factor = float(columns["factor"][index]) if columns["factor"][index] not in {".", "?", ""} else 0.0
        except (ValueError, OverflowError) as error:
            raise ValueError(f"mmCIF atom_site row {index + 1} has invalid numeric fields") from error
        if (model < 1 or not atom_name or len(atom_name) > 8
                or not re.fullmatch(r"[A-Z0-9]{1,3}", residue_name)
                or chain in {".", "?", ""} or len(chain) > 128
                or (alternate not in {".", "?", ""} and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", alternate))
                or (insertion not in {".", "?", ""} and not re.fullmatch(r"[A-Za-z0-9]", insertion))):
            raise ValueError(f"mmCIF atom_site row {index + 1} has malformed identity fields")
        insertion = "" if insertion in {".", "?"} else insertion
        if any(not math.isfinite(value) for value in (*coordinates, occupancy, factor)):
            raise ValueError(f"mmCIF atom_site row {index + 1} contains a non-finite number")
        if any(abs(value) > 10_000 for value in coordinates) or not 0 <= occupancy <= 1.1 or not -1 <= factor <= 1_000_000:
            raise ValueError(f"mmCIF atom_site row {index + 1} has out-of-range coordinates or metadata")
        if alternate not in {".", "?", "", "A"}:
            discarded_alternates += 1
            continue
        if model > 1000:
            raise ValueError("mmCIF model number exceeds 1000")
        model_ids.add(model)
        atom_count += 1
        if atom_count > MAX_ATOMS:
            raise ValueError(f"mmCIF input exceeds {MAX_ATOMS} accepted atom records")
        chain_ids.add(chain)
        residue_key = (residue_number, insertion)
        residue_bucket = residues[(model, chain)]
        previous = residue_bucket.setdefault(residue_key, residue_name)
        if previous != residue_name:
            raise ValueError(f"mmCIF residue identity changes in model {model}, chain {chain}, position {residue_number}{insertion}")
        atom_key = (model, chain, residue_number, insertion, atom_name)
        if atom_key in atoms:
            raise ValueError(f"mmCIF contains a duplicate accepted atom row at {index + 1}")
        atoms.add(atom_key)
        if atom_name.upper() == "CA":
            alpha_carbons.add((model, chain, residue_number, insertion))
        factors.append(factor)
    if atom_count == 0:
        raise ValueError("mmCIF atom_site loop contains no accepted ATOM coordinate rows")
    if model_ids != set(range(1, len(model_ids) + 1)):
        raise ValueError("mmCIF model numbering is not a contiguous 1-based sequence")

    chain_summaries = []
    for (model_id, chain_id), chain_residues in sorted(residues.items()):
        sequence = "".join(_AMINO_ACIDS.get(name, "X") for _, name in sorted(chain_residues.items()))
        chain_summaries.append({
            "model": model_id,
            "chain": chain_id,
            "residue_count": len(chain_residues),
            "alpha_carbon_count": sum((model_id, chain_id, number, insertion) in alpha_carbons for number, insertion in chain_residues),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "unknown_residue_count": sequence.count("X"),
        })
    factors.sort()
    middle = len(factors) // 2
    median_factor = factors[middle] if len(factors) % 2 else (factors[middle - 1] + factors[middle]) / 2
    return {
        "schema_version": "1",
        "source": "user-supplied mmCIF coordinate file",
        "label": label,
        "input_bytes": len(payload),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "model_count": len(model_ids),
        "atom_count": atom_count,
        "chain_count": len(residues),
        "unique_chain_id_count": len(chain_ids),
        "discarded_alternate_atom_count": discarded_alternates,
        "residue_count": sum(len(rows) for rows in residues.values()),
        "alpha_carbon_count": len(alpha_carbons),
        "chains": chain_summaries,
        "atom_temperature_factor": {
            "minimum": factors[0], "median": median_factor, "maximum": factors[-1],
            "interpretation": "Raw mmCIF B_iso_or_equiv values; not identified as pLDDT or calibrated confidence without source-specific provenance.",
        },
        "coordinates_retained": False,
        "parser": "Biopython MMCIF2Dict",
        "interpretation_limits": [
            "This is a structural-file integrity and composition summary, not structure prediction, binding analysis, or functional interpretation.",
            "Only ATOM records and blank/A alternate locations are counted; HETATM records and other alternate conformations are omitted.",
            "Residue sequence digests include X for non-standard residue names; they do not establish protein identity or reference-sequence equivalence.",
        ],
    }
