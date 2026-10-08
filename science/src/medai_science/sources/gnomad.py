"""Exact, paced gnomAD v4 population-frequency lookup (research only)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import re
import asyncio
import time
from typing import Any

import httpx

from medai_science.sources.bounded_http import read_bounded_response


_VARIANT_ID = re.compile(r"^(?:chr)?(\d{1,2}|X|Y)-(\d+)-([ACGTN]+)-([ACGTN]+)$", re.IGNORECASE)


def validate_callset_frequency(value: Any, callset: str) -> dict[str, Any] | None:
    """Validate one source-reported exome or genome aggregate independently."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"ac", "an", "af"}:
        raise ValueError(f"gnomAD {callset} frequency record is malformed")
    ac, an, af = value["ac"], value["an"], value["af"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (ac, an)):
        raise ValueError(f"gnomAD {callset} allele counts are malformed")
    if an < ac:
        raise ValueError(f"gnomAD {callset} allele number cannot be smaller than allele count")
    if af is None:
        if an != 0 or ac != 0:
            raise ValueError(f"gnomAD {callset} allele frequency is missing despite observed alleles")
        return value
    if isinstance(af, bool) or not isinstance(af, (int, float)) or not math.isfinite(af) or not 0 <= af <= 1:
        raise ValueError(f"gnomAD {callset} allele frequency is malformed")
    if an == 0 or not math.isclose(float(af), ac / an, rel_tol=1e-5, abs_tol=1e-8):
        raise ValueError(f"gnomAD {callset} allele frequency does not agree with counts")
    return value


class GnomADClient:
    endpoint = "https://gnomad.broadinstitute.org/api"
    max_response_bytes = 1_000_000
    min_interval_seconds = 6.1
    _last_request_at = 0.0
    _pace_lock = asyncio.Lock()

    def __init__(self, client_factory=None, transport=None, timeout: float = 20):
        self.client_factory = client_factory or httpx.AsyncClient
        self.transport = transport
        self.timeout = timeout

    @classmethod
    def parse_variant_id(cls, variant_id: str) -> str:
        """Canonicalize variant-ID spelling; this does not left-align against a reference."""
        if not isinstance(variant_id, str) or len(variant_id) > 96:
            raise ValueError("gnomAD requires one normalized GRCh38 variant ID")
        match = _VARIANT_ID.fullmatch(variant_id)
        if not match:
            raise ValueError("variant ID must be contig-position-reference-alternate (GRCh38)")
        contig, position, ref, alt = match.groups()
        if contig.isdigit() and not 1 <= int(contig) <= 22:
            raise ValueError("gnomAD lookup supports autosomes 1..22, X, or Y")
        if int(position) < 1 or int(position) > 300_000_000:
            raise ValueError("variant position is outside the supported range")
        ref, alt = ref.upper(), alt.upper()
        if ref == alt or len(ref) > 50 or len(alt) > 50:
            raise ValueError("variant alleles must be distinct and at most 50 bases")
        return f"{contig.upper() if not contig.isdigit() else int(contig)}-{int(position)}-{ref}-{alt}"

    async def _pace(self) -> None:
        async with self._pace_lock:
            delay = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            type(self)._last_request_at = time.monotonic()

    async def lookup(self, variant_id: str, assembly: str = "GRCh38") -> dict[str, Any]:
        if assembly != "GRCh38":
            raise ValueError("gnomAD v4 lookup requires GRCh38 coordinates")
        normalized_id = self.parse_variant_id(variant_id)
        query = """query VariantFrequency($variantId: String!, $datasetId: DatasetId!) {
          variant(variantId: $variantId, dataset: $datasetId) {
            variant_id genome { ac an af } exome { ac an af }
          }
        }"""
        await self._pace()
        async with self.client_factory(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("POST", self.endpoint, json={"query": query, "variables": {"variantId": normalized_id, "datasetId": "gnomad_r4"}}, headers={"Accept": "application/json"}) as response:
                raw = await read_bounded_response(response, self.max_response_bytes, "gnomAD GraphQL")
                response.raise_for_status()
        try:
            payload = json.loads(raw, object_pairs_hook=self._unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("gnomAD returned invalid or duplicate-key JSON") from error
        if not isinstance(payload, dict) or payload.get("errors"):
            raise ValueError("gnomAD GraphQL returned an error")
        data = payload.get("data")
        if not isinstance(data, dict) or "variant" not in data:
            raise ValueError("gnomAD response is missing the variant field")
        row = data["variant"]
        if row is not None:
            if not isinstance(row, dict) or row.get("variant_id") != normalized_id:
                raise ValueError("gnomAD response does not match the exact requested variant")
            if set(row) != {"variant_id", "genome", "exome"}:
                raise ValueError("gnomAD response fields do not match the requested frequency selection")
            genome = validate_callset_frequency(row["genome"], "genome")
            exome = validate_callset_frequency(row["exome"], "exome")
            if genome is None and exome is None:
                raise ValueError("gnomAD variant record contains no genome or exome frequency callset")
        else:
            genome = exome = None
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        source_payload = {"variant": row}
        digest = hashlib.sha256(json.dumps(source_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return {
            "variant_id": normalized_id,
            "assembly": "GRCh38",
            "dataset": "gnomAD v4 (gnomad_r4)",
            "genome_frequency": genome,
            "exome_frequency": exome,
            "found": row is not None,
            "source": "gnomAD Browser GraphQL API",
            "source_url": "https://gnomad.broadinstitute.org/",
            "source_record_sha256": digest,
            "retrieved_at": retrieved_at,
            "interpretation_limit": "The requested variant ID is canonicalized syntactically only; this lookup does not verify the reference allele or left-align indels. Use a reference-validated local VCF analysis for representation normalization before exact reconciliation. Exome and genome allele counts are separate callsets and must not be summed as if they were disjoint cohorts. These population observations do not establish individual genotype, pathogenicity, disease causality, or clinical risk.",
        }

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
