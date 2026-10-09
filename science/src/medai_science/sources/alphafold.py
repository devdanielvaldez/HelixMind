"""AlphaFold DB prediction metadata adapter using current model-field names."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from medai_science.sources.bounded_http import read_bounded_response

_ACCESSION = re.compile(r"^[A-Z0-9]{6,10}(?:-[1-9][0-9]*)?$", re.IGNORECASE)
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_COORDINATE_BYTES = 32 * 1024 * 1024


def _first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _digest(records: Any) -> str:
    serialized = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_prediction(accession: str, prediction: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    model_id = _first(prediction, "modelEntityId", "entryId")
    result_accession = _first(prediction, "uniprotAccession", "accession")
    if not isinstance(model_id, str) or not model_id.startswith("AF-") or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._" for character in model_id):
        raise ValueError("AlphaFold response is missing a valid model entity identifier")
    if not isinstance(result_accession, str):
        raise ValueError("AlphaFold response is missing its UniProt accession")
    if result_accession.upper().split("-")[0] != accession.upper().split("-")[0]:
        raise ValueError("AlphaFold response accession does not match the requested UniProt accession")

    coordinates = {
        "mmcif_url": _first(prediction, "cifUrl", "mmcifUrl"),
        "pdb_url": prediction.get("pdbUrl"),
        "bcif_url": _first(prediction, "bcifUrl", "binaryCifUrl"),
        "pae_json_url": _first(prediction, "paeDocUrl", "predictedAlignedErrorUrl"),
    }
    for label, url in coordinates.items():
        if url is not None:
            parsed = urlsplit(url) if isinstance(url, str) else None
            host = (parsed.hostname or "").lower() if parsed else ""
            trusted_host = host == "alphafold.com" or host.endswith(".ebi.ac.uk")
            if not parsed or parsed.scheme != "https" or not trusted_host or parsed.username or parsed.password:
                raise ValueError(f"AlphaFold returned an invalid {label}")

    sequence_start = _first(prediction, "sequenceStart", "uniprotStart")
    sequence_end = _first(prediction, "sequenceEnd", "uniprotEnd")
    sequence = _first(prediction, "sequence", "uniprotSequence")
    return {
        "source": "AlphaFold Protein Structure Database",
        "model_id": model_id,
        "uniprot_accession": result_accession,
        "uniprot_id": prediction.get("uniprotId"),
        "protein_description": _first(prediction, "uniprotDescription", "geneDescription"),
        "gene": prediction.get("gene"),
        "organism": prediction.get("organismScientificName"),
        "taxonomy_id": prediction.get("taxId"),
        "model_version": _first(prediction, "latestVersion", "version"),
        "model_created_date": _first(prediction, "modelCreatedDate", "createdDate"),
        "sequence_range": {"start": sequence_start, "end": sequence_end},
        "sequence_length": len(sequence) if isinstance(sequence, str) else None,
        "sequence": sequence,
        "confidence_metrics": {
            "global_metric_value": prediction.get("globalMetricValue"),
            "fraction_plddt_very_low": prediction.get("fractionPlddtVeryLow"),
            "fraction_plddt_low": prediction.get("fractionPlddtLow"),
            "fraction_plddt_confident": prediction.get("fractionPlddtConfident"),
            "fraction_plddt_very_high": prediction.get("fractionPlddtVeryHigh"),
        },
        "coordinates": coordinates,
        "is_uniprot_reviewed": _first(prediction, "isUniProtReviewed", "isReviewed"),
        "is_reference_proteome": _first(prediction, "isUniProtReferenceProteome", "isReferenceProteome"),
        "model_kind": "predicted_structure",
        "experimental_structure": False,
        "interpretation_limits": [
            "pLDDT estimates local per-residue model confidence; it is not a probability of biological function.",
            "Local confidence does not by itself validate relative domain placement, binding, or interactions.",
            "This record does not provide pathogenicity, target suitability, or clinical interpretation.",
        ],
        "provenance": {
            "provider": "AlphaFold Protein Structure Database API",
            "api_endpoint": f"https://alphafold.com/api/prediction/{quote(accession, safe='-')}",
            "entry_url": f"https://alphafold.com/entry/{quote(model_id, safe='-')}",
            "retrieved_at": retrieved_at,
            "response_record_sha256": _digest(prediction),
        },
        "record": prediction,
    }


class AlphaFoldClient:
    endpoint = "https://alphafold.com/api/prediction"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate_accession(uniprot_accession: str) -> str:
        if not isinstance(uniprot_accession, str) or not _ACCESSION.fullmatch(uniprot_accession):
            raise ValueError("provide a UniProt accession (optionally followed by an isoform suffix)")
        return uniprot_accession.upper()

    async def predictions(self, uniprot_accession: str) -> list[dict[str, Any]]:
        uniprot_accession = self.validate_accession(uniprot_accession)
        endpoint = f"{self.endpoint}/{quote(uniprot_accession, safe='-')}"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", endpoint, headers={"Accept": "application/json"}) as response:
                if response.status_code == 404:
                    return []
                body = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "AlphaFold")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("AlphaFold returned invalid JSON") from error
        if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
            raise ValueError("AlphaFold returned an unexpected prediction response")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return [_normalize_prediction(uniprot_accession, item, retrieved_at) for item in payload]

    @staticmethod
    def coordinate_url(model_id: str, format: str = "pdb") -> str:
        """Build the provider's canonical coordinate URL from a validated model ID."""
        if not isinstance(model_id, str) or not re.fullmatch(r"AF-[A-Za-z0-9-]+-F[0-9]+-v[0-9]+", model_id):
            raise ValueError("model_id is not a valid AlphaFold model entity identifier")
        if format not in {"pdb", "cif"}:
            raise ValueError("coordinate format must be pdb or cif")
        return f"https://alphafold.ebi.ac.uk/files/{model_id}.{format}"

    async def coordinate_file(self, model_id: str, format: str = "pdb") -> dict[str, Any]:
        """Fetch bounded coordinates from the fixed AlphaFold host without redirects."""
        url = self.coordinate_url(model_id, format)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", url, headers={"Accept": "chemical/x-pdb" if format == "pdb" else "chemical/x-mmcif"}) as response:
                body = await read_bounded_response(response, _MAX_COORDINATE_BYTES, "AlphaFold coordinate file")
        if not body.strip():
            raise ValueError("AlphaFold returned an empty coordinate file")
        return {
            "model_id": model_id,
            "format": format,
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "sha256": hashlib.sha256(body).hexdigest(),
            "byte_length": len(body),
            "license": "CC-BY-4.0",
            "attribution": "AlphaFold Protein Structure Database; cite Jumper et al., Nature (2021) and the current AlphaFold DB release paper.",
            "content": body,
        }
