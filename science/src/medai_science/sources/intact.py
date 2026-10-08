"""IntAct PSICQUIC interactions with retained MITAB evidence and provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from urllib.parse import quote

import httpx
from medai_science.protein_interactions import reconcile_interactions
from medai_science.sources.bounded_http import read_bounded_response

_ACCESSION = re.compile(r"^[A-Z0-9]{6,10}(?:-[1-9][0-9]*)?$", re.IGNORECASE)
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class IntActClient:
    endpoint = "https://www.ebi.ac.uk/Tools/webservices/psicquic/intact/webservices/current/search"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate_accession(accession: str) -> str:
        if not isinstance(accession, str) or not _ACCESSION.fullmatch(accession):
            raise ValueError("provide a UniProt accession (optionally followed by an isoform suffix)")
        return accession.upper()

    async def interactions(self, accession: str, limit: int = 100, organism_id: int | None = 9606) -> dict:
        accession = self.validate_accession(accession)
        if not 1 <= limit <= 500:
            raise ValueError("interaction limit must be 1..500")
        if organism_id is not None and not 1 <= organism_id <= 9_999_999:
            raise ValueError("organism_id must be a positive NCBI taxonomy identifier or None")
        query = f"id:{accession}"
        url = f"{self.endpoint}/{quote(query, safe=':')}/"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", url, params={"format": "tab", "first": 0, "maxResults": limit}, headers={"Accept": "text/plain"}) as response:
                if response.status_code == 404:
                    lines = b""
                else:
                    lines = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "IntAct")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        try:
            raw = lines.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("IntAct returned invalid UTF-8 MITAB data") from error
        records = []
        source_rows_seen = 0
        rows_missing_taxonomy = 0
        rows_filtered_by_organism = 0
        for line in raw.splitlines():
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 15:
                raise ValueError("IntAct returned a malformed MITAB record")
            source_rows_seen += 1
            organism_a, organism_b = columns[9], columns[10]
            taxon_a = _taxon(organism_a)
            taxon_b = _taxon(organism_b)
            if taxon_a is None or taxon_b is None:
                rows_missing_taxonomy += 1
            if organism_id is not None and (taxon_a != organism_id or taxon_b != organism_id):
                rows_filtered_by_organism += 1
                continue
            records.append({
                "interactor_a": columns[0], "interactor_b": columns[1],
                "aliases_a": columns[4], "aliases_b": columns[5],
                "detection_method": columns[6], "publication": columns[7],
                "pubmed_ids": columns[8], "organism_a": organism_a, "organism_b": organism_b,
                "interaction_type": columns[11], "source_database": columns[12],
                "interaction_ids": columns[13], "confidence": columns[14],
                "complex_expansion": columns[15] if len(columns) > 15 else "-",
                "biological_role_a": columns[16] if len(columns) > 16 else "-",
                "biological_role_b": columns[17] if len(columns) > 17 else "-",
                "experimental_role_a": columns[18] if len(columns) > 18 else "-",
                "experimental_role_b": columns[19] if len(columns) > 19 else "-",
                "raw_mitab": line,
            })
        return {
            "source": "IntAct", "query_accession": accession, "organism_filter": organism_id,
            "retrieved_at": retrieved_at, "response_sha256": hashlib.sha256(lines).hexdigest(),
            "retrieval_coverage": {
                "requested_limit": limit,
                "source_rows_seen": source_rows_seen,
                "retained_rows": len(records),
                "rows_filtered_by_organism": rows_filtered_by_organism,
                "rows_missing_taxonomy": rows_missing_taxonomy,
                "may_be_truncated_at_requested_limit": source_rows_seen >= limit,
                "interpretation": "PSICQUIC rows are bounded by maxResults. A count equal to the requested limit may omit additional matches; this receipt does not claim search completeness.",
            },
            "records": records,
            "reconciliation": reconcile_interactions(records),
            "interpretation_limits": [
                "Records are curated or submitted interaction evidence, not proof of an interaction in every tissue, condition, or assay.",
                "MITAB confidence fields are retained as source strings and are not calibrated probabilities.",
                "No clinical or therapeutic inference is made.",
            ],
        }


def _taxon(value: str) -> int | None:
    for item in value.split("|"):
        if item.startswith("taxid:"):
            number = item.partition(":")[2].partition("(")[0]
            if number.isdigit() and len(number) <= 7:
                return int(number)
            return None
    return None
