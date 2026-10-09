"""Bounded single-RefSNP retrieval from NCBI Variation Services."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from typing import Any

import httpx

from medai_science.sources.bounded_http import event_loop_lock, read_bounded_response


_RSID = re.compile(r"^(?:rs)?([0-9]{1,12})$", re.IGNORECASE)


class DbSNPClient:
    """Fetch one bounded RefSNP JSON record without inferring an assembly match."""

    endpoint = "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp"
    source = "NCBI dbSNP Variation Services"
    max_response_bytes = 2_000_000
    min_interval_seconds = 1.1
    _last_request_at = 0.0

    def __init__(self, client_factory=None, transport=None, timeout: float = 20):
        self.client_factory = client_factory or httpx.AsyncClient
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def normalize_rsid(value: str) -> str:
        if not isinstance(value, str) or len(value) > 14:
            raise ValueError("dbSNP lookup requires one RefSNP rs identifier")
        match = _RSID.fullmatch(value.strip())
        if not match or int(match.group(1)) == 0:
            raise ValueError("dbSNP identifier must be rs followed by 1..12 nonzero-leading digits")
        return "rs" + str(int(match.group(1)))

    @classmethod
    async def _pace(cls) -> None:
        async with event_loop_lock("dbsnp"):
            delay = cls.min_interval_seconds - (time.monotonic() - cls._last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            cls._last_request_at = time.monotonic()

    async def lookup(self, rsid: str) -> dict[str, Any]:
        canonical_rsid = self.normalize_rsid(rsid)
        number = canonical_rsid[2:]
        endpoint = f"{self.endpoint}/{number}"
        await self._pace()
        async with self.client_factory(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", endpoint, headers={"Accept": "application/json"}) as response:
                raw = await read_bounded_response(response, self.max_response_bytes, "NCBI dbSNP")
        try:
            payload = json.loads(raw, object_pairs_hook=self._unique_object, parse_constant=self._reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("NCBI dbSNP returned invalid or duplicate-key JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("NCBI dbSNP RefSNP response must be a JSON object")
        returned_id = payload.get("refsnp_id")
        if isinstance(returned_id, bool) or str(returned_id) != number:
            raise ValueError("NCBI dbSNP response does not match the requested RefSNP identifier")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return {
            "rsid": canonical_rsid,
            "source": self.source,
            "source_url": f"https://www.ncbi.nlm.nih.gov/snp/{number}",
            "api_endpoint": endpoint,
            "retrieved_at": retrieved_at,
            "source_record_sha256": hashlib.sha256(raw).hexdigest(),
            "source_record_canonical_sha256": hashlib.sha256(json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "source_record": payload,
            "interpretation_limit": "This is an archived dbSNP RefSNP source record. Its placements and alleles are not assumed to use GRCh37 or GRCh38 and are not joined to local calls here; verify assembly, sequence accession, and reference-normalized alleles before reconciliation. dbSNP identity and annotations do not establish pathogenicity or clinical risk.",
        }

    @staticmethod
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")
