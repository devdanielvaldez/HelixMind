"""Rate-limited retrieval of PMC Open Access BioC documents."""
from __future__ import annotations

import asyncio
import re
import time

import httpx

from medai_science.paper_understanding import MAX_BIOC_BYTES, parse_bioc_xml
from medai_science.sources.bounded_http import read_bounded_response


class PMCBioCClient:
    endpoint = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"
    _identifier = re.compile(r"^(?:PMC[1-9][0-9]*|PMID:[1-9][0-9]{0,8})$", re.IGNORECASE)
    _shared_request_lock = asyncio.Lock()
    _shared_last_request = 0.0

    def __init__(self, timeout: float = 30.0, client_factory=httpx.AsyncClient):
        self.timeout = timeout
        self._client_factory = client_factory

    @classmethod
    def validate_identifier(cls, identifier: str) -> str:
        value = identifier.strip().upper()
        if not cls._identifier.fullmatch(value):
            raise ValueError("provide a PMC identifier (PMC123456) or an explicitly labeled PMID (PMID:123456)")
        return value.removeprefix("PMID:")

    async def fetch(self, identifier: str) -> dict:
        identifier = self.validate_identifier(identifier)
        url = f"{self.endpoint}/BioC_xml/{identifier}/unicode"
        async with self._shared_request_lock:
            delay = (1 / 3) - (time.monotonic() - type(self)._shared_last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            type(self)._shared_last_request = time.monotonic()
            async with self._client_factory(timeout=self.timeout, follow_redirects=False) as client:
                async with client.stream("GET", url, headers={"Accept": "application/xml"}) as response:
                    payload = await read_bounded_response(response, MAX_BIOC_BYTES, "PMC BioC")
        return parse_bioc_xml(payload, source_url=str(response.url))
