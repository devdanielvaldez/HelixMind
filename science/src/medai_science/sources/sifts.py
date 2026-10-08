"""Residue-level UniProt/PDB mappings parsed from PDBe SIFTS XML snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
import shlex
from urllib.parse import quote
import xml.etree.ElementTree as ET
import zlib

import httpx
from medai_science.sources.xml_safety import reject_xml_declarations

_PDB_ID = re.compile(r"^[0-9][A-Z0-9]{3}$", re.IGNORECASE)
_MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
_MAX_XML_BYTES = 64 * 1024 * 1024
_MAX_PDB_ENTRIES = 20
_MAX_PDB_IDS = 1_000
_MAX_RESIDUE_MAPPINGS = 200_000
_MAX_MMCIF_BYTES = 64 * 1024 * 1024


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_observed_mmcif(mmcif: bytes, accession: str) -> list[dict]:
    """Read residue-level SIFTS observation flags from the enriched mmCIF category."""
    try:
        text = mmcif.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("PDBe updated mmCIF is not valid UTF-8 text") from error
    lines = text.splitlines()
    result: list[dict] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().casefold() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            headers.append(lines[index].strip().split()[0].casefold())
            index += 1
        if not any(header.startswith("_pdbx_sifts_xref_db.") for header in headers):
            continue
        if not headers or any(not header.startswith("_pdbx_sifts_xref_db.") for header in headers):
            raise ValueError("PDBe updated mmCIF has an invalid SIFTS cross-reference loop")
        names = [header.partition(".")[2] for header in headers]
        required = {"entity_id", "asym_id", "seq_id", "unp_acc", "unp_num", "observed"}
        if not required.issubset(names):
            raise ValueError("PDBe updated mmCIF SIFTS loop is missing required residue or observation fields")
        buffered: list[str] = []
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if buffered:
                    raise ValueError("PDBe updated mmCIF SIFTS loop ended inside a row")
                index += 1
                break
            if stripped.casefold() in {"loop_", "stop_", "global_"} or stripped.casefold().startswith(("data_", "save_")) or (stripped.startswith("_") and not buffered):
                if buffered:
                    raise ValueError("PDBe updated mmCIF SIFTS loop ended inside a row")
                break
            if stripped.startswith(";"):
                raise ValueError("unexpected multiline value in PDBe SIFTS residue loop")
            try:
                buffered.extend(shlex.split(line, comments=True, posix=True))
            except ValueError as error:
                raise ValueError("PDBe updated mmCIF contains malformed quoted values") from error
            while len(buffered) >= len(names):
                row_values = buffered[:len(names)]
                del buffered[:len(names)]
                row = dict(zip(names, row_values))
                if row["unp_acc"].upper() != accession.upper():
                    continue
                raw_observed = row["observed"].upper()
                observed = True if raw_observed == "Y" else False if raw_observed == "N" else None
                try:
                    uniprot_number = int(row["unp_num"])
                    pdb_sequence_number = int(row["seq_id"])
                except ValueError:
                    continue
                result.append({
                    "entity_id": row["entity_id"],
                    "label_chain_id": row["asym_id"],
                    "pdb_sequence_number": pdb_sequence_number,
                    "uniprot_accession": row["unp_acc"],
                    "uniprot_residue_number": uniprot_number,
                    "observed": observed,
                    "raw_observed": row["observed"],
                })
            index += 1
        if buffered:
            raise ValueError("PDBe updated mmCIF SIFTS loop contains an incomplete row")
    return result


def _parse_sifts_xml(xml_data: bytes, pdb_id: str, accession: str) -> list[dict]:
    mappings: list[dict] = []
    reject_xml_declarations(xml_data, f"SIFTS for PDB {pdb_id}")
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as error:
        raise ValueError(f"SIFTS returned malformed XML for PDB {pdb_id}") from error
    if _local_name(root.tag) != "entry":
        raise ValueError(f"SIFTS returned an unexpected XML root for PDB {pdb_id}")

    for entity in (item for item in root.iter() if _local_name(item.tag) == "entity"):
        entity_id = entity.attrib.get("entityId")
        for residue in (item for item in entity.iter() if _local_name(item.tag) == "residue"):
            crossrefs = [item for item in residue if _local_name(item.tag) == "crossRefDb"]
            uniprot = next((item for item in crossrefs if item.attrib.get("dbSource", "").casefold() == "uniprot" and item.attrib.get("dbAccessionId", "").upper() == accession.upper()), None)
            pdb = next((item for item in crossrefs if item.attrib.get("dbSource", "").casefold() in {"pdb", "pdbe"}), None)
            if uniprot is None or pdb is None:
                continue
            uniprot_position = uniprot.attrib.get("dbResNum")
            if not uniprot_position or not uniprot_position.isdigit():
                continue
            details = [
                {"source": detail.attrib.get("dbSource"), "property": detail.attrib.get("property"), "value": "".join(detail.itertext()).strip()}
                for detail in residue if _local_name(detail.tag) == "residueDetail"
            ]
            mappings.append({
                "pdb_id": pdb_id.lower(),
                "entity_id": entity_id,
                "chain_id": pdb.attrib.get("dbChainId"),
                "uniprot_accession": accession,
                "uniprot_residue_number": int(uniprot_position),
                "uniprot_residue_name": uniprot.attrib.get("dbResName", uniprot.attrib.get("dbResMon")),
                "pdb_residue_number": pdb.attrib.get("dbResNum", residue.attrib.get("dbResNum")),
                "pdb_residue_name": pdb.attrib.get("dbResName", residue.attrib.get("dbResMon")),
                "pdb_author_residue_number": pdb.attrib.get("dbAuthorResNum"),
                "pdb_author_chain_id": pdb.attrib.get("dbAuthorChainId"),
                "pdb_coordinate_system": pdb.attrib.get("dbCoordSys"),
                "residue_details": details,
            })
    return mappings


class SIFTSClient:
    base_url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    async def map_residues(self, accession: str, pdb_ids: list[str], max_structures: int = 20) -> dict:
        if not isinstance(accession, str) or not re.fullmatch(r"[A-Z0-9]{6,10}(?:-[1-9][0-9]*)?", accession, re.IGNORECASE):
            raise ValueError("provide a UniProt accession (optionally followed by an isoform suffix)")
        accession = accession.upper()
        if not 1 <= max_structures <= _MAX_PDB_ENTRIES:
            raise ValueError(f"max_structures must be 1..{_MAX_PDB_ENTRIES}")
        if len(pdb_ids) > _MAX_PDB_IDS or any(not isinstance(pdb_id, str) or not _PDB_ID.fullmatch(pdb_id) for pdb_id in pdb_ids):
            raise ValueError(f"PDB identifiers must be four-character archive IDs; at most {_MAX_PDB_IDS} may be supplied")
        selected = list(dict.fromkeys(pdb_id.lower() for pdb_id in pdb_ids))[:max_structures]
        unique_count = len(set(pdb_id.lower() for pdb_id in pdb_ids))
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        entries, all_mappings = [], []
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            for pdb_id in selected:
                url = f"{self.base_url}/{quote(pdb_id, safe='')}.xml.gz"
                async with client.stream("GET", url, headers={"Accept": "application/gzip, application/octet-stream"}) as response:
                    if response.status_code == 404:
                        entries.append({"pdb_id": pdb_id, "status": "not_found"})
                        continue
                    response.raise_for_status()
                    compressed = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(chunk) > _MAX_COMPRESSED_BYTES - len(compressed):
                            raise ValueError(f"SIFTS XML for {pdb_id} exceeds the 8 MiB compressed safety limit")
                        compressed.extend(chunk)
                try:
                    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
                    xml_data = decoder.decompress(bytes(compressed), _MAX_XML_BYTES + 1)
                    if len(xml_data) > _MAX_XML_BYTES or decoder.unconsumed_tail:
                        raise ValueError(f"SIFTS XML for {pdb_id} exceeds the 64 MiB decompressed safety limit")
                    xml_data += decoder.flush(_MAX_XML_BYTES + 1 - len(xml_data))
                    if len(xml_data) > _MAX_XML_BYTES or not decoder.eof:
                        raise ValueError(f"SIFTS XML for {pdb_id} is oversized or truncated")
                except zlib.error as error:
                    raise ValueError(f"SIFTS returned invalid gzip data for PDB {pdb_id}") from error
                mappings = _parse_sifts_xml(xml_data, pdb_id, accession)
                if len(all_mappings) + len(mappings) > _MAX_RESIDUE_MAPPINGS:
                    raise ValueError(f"SIFTS result exceeds the {_MAX_RESIDUE_MAPPINGS}-mapping safety limit")
                digest = hashlib.sha256(xml_data).hexdigest()
                mmcif_url = f"https://www.ebi.ac.uk/pdbe/static/entry/{quote(pdb_id, safe='')}_updated.cif"
                coordinate_metadata = {"status": "not_found"}
                async with client.stream("GET", mmcif_url, headers={"Accept": "text/plain"}) as response:
                    if response.status_code != 404:
                        response.raise_for_status()
                        mmcif_data = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(chunk) > _MAX_MMCIF_BYTES - len(mmcif_data):
                                raise ValueError(f"Updated mmCIF for {pdb_id} exceeds the 64 MiB safety limit")
                            mmcif_data.extend(chunk)
                        mmcif_digest = hashlib.sha256(mmcif_data).hexdigest()
                        flags = _parse_observed_mmcif(bytes(mmcif_data), accession)
                        coordinate_metadata = {
                            "status": "available" if flags else "no_matching_sifts_rows",
                            "source_url": mmcif_url,
                            "mmcif_sha256": mmcif_digest,
                            "matching_sifts_rows": len(flags),
                        }
                        by_key: dict[tuple[str, int, int], list[bool | None]] = {}
                        for flag in flags:
                            key = (flag["entity_id"], flag["uniprot_residue_number"], flag["pdb_sequence_number"])
                            by_key.setdefault(key, []).append(flag["observed"])
                        for mapping in mappings:
                            try:
                                pdb_sequence_number = int(mapping["pdb_residue_number"])
                            except (TypeError, ValueError):
                                pdb_sequence_number = None
                            key = (str(mapping["entity_id"]), mapping["uniprot_residue_number"], pdb_sequence_number)
                            candidates = by_key.get(key, []) if pdb_sequence_number is not None else []
                            values = set(candidates)
                            if values == {True}:
                                status = "observed"
                            elif values == {False}:
                                status = "not_observed"
                            elif not candidates or values == {None}:
                                status = "unknown"
                            else:
                                status = "ambiguous"
                            mapping["coordinate_observation"] = {
                                "status": status,
                                "has_atomic_coordinates": True if status == "observed" else False if status == "not_observed" else None,
                                "mmcif_matching_rows": len(candidates),
                                "source_field": "_pdbx_sifts_xref_db.observed",
                            }
                if coordinate_metadata["status"] != "available":
                    for mapping in mappings:
                        mapping["coordinate_observation"] = {
                            "status": "unknown",
                            "has_atomic_coordinates": None,
                            "mmcif_matching_rows": 0,
                            "source_field": "_pdbx_sifts_xref_db.observed",
                        }
                entries.append({"pdb_id": pdb_id, "source_url": url, "xml_sha256": digest, "mapping_count": len(mappings), "coordinate_metadata": coordinate_metadata})
                all_mappings.extend(mappings)
        all_mappings.sort(key=lambda row: (row["pdb_id"], row["chain_id"] or "", row["uniprot_residue_number"]))
        return {
            "source": "PDBe SIFTS", "accession": accession, "retrieved_at": retrieved_at,
            "entries": entries, "residue_mappings": all_mappings,
            "coverage": {
                "mapped_uniprot_positions": len({row["uniprot_residue_number"] for row in all_mappings}),
                "requested_structure_count": len(selected),
                "available_structure_count": unique_count,
                "omitted_structure_count": max(0, unique_count - len(selected)),
                "returned_mapping_count": len(all_mappings),
                "residues_with_coordinates": sum(row.get("coordinate_observation", {}).get("status") == "observed" for row in all_mappings),
                "residues_without_coordinates": sum(row.get("coordinate_observation", {}).get("status") == "not_observed" for row in all_mappings),
                "residues_with_unknown_coordinate_status": sum(row.get("coordinate_observation", {}).get("status") in {"unknown", "ambiguous"} for row in all_mappings),
            },
            "interpretation_limits": [
                "Atomic-coordinate presence is reported only when the PDBe updated mmCIF explicitly supplies _pdbx_sifts_xref_db.observed; missing or ambiguous metadata remains unknown.",
                "Coordinate-system and author numbering fields are retained separately; do not assume numbering is contiguous or interchangeable.",
                "Only structures listed by the UniProtKB record supplied to this lookup are queried.",
            ],
        }
