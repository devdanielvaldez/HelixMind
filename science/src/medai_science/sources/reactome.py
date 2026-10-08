"""Reactome Analysis Service adapter for over-representation of gene lists."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import httpx
from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query

_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

class ReactomeClient:
    endpoint = "https://reactome.org/AnalysisService/identifiers/"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 60):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate(identifiers: list[str], species: str, page_size: int) -> list[str]:
        genes = list(dict.fromkeys(item.strip() for item in identifiers if item.strip()))
        if not 1 <= len(genes) <= 5000:
            raise ValueError("provide 1..5000 non-empty gene identifiers")
        if any(len(item) > 64 or any(char in item for char in "\r\n\t") for item in genes):
            raise ValueError("gene identifiers must be single-line values up to 64 characters")
        ensure_public_research_query(" ".join(genes))
        if species not in {"Homo sapiens", "Mus musculus", "Rattus norvegicus"}:
            raise ValueError("supported Reactome species: Homo sapiens, Mus musculus, Rattus norvegicus")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be 1..100")
        return genes

    async def analyze_genes(
        self, identifiers: list[str], species: str = "Homo sapiens", page_size: int = 20
    ) -> dict:
        genes = self.validate(identifiers, species, page_size)

        input_text = "\n".join(genes)
        parameters = {"pageSize": page_size, "sortBy": "ENTITIES_PVALUE", "order": "ASC", "species": species}
        request_digest = hashlib.sha256(json.dumps({"endpoint": self.endpoint, "parameters": parameters, "identifiers": genes}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream(
                "POST", self.endpoint, params=parameters, content=input_text.encode(),
                headers={"Content-Type": "text/plain", "Accept": "application/json"},
            ) as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "Reactome")
        try:
            result = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Reactome returned invalid JSON") from error

        return {
            "source": "Reactome Analysis Service",
            "source_url": self.endpoint,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "species": species,
            "input_identifiers": genes,
            "input_sha256": hashlib.sha256(input_text.encode()).hexdigest(),
            "analysis_parameters": parameters,
            "request_sha256": request_digest,
            "result": result,
            "interpretation_limit": "Pathway over-representation is an association analysis, not causal evidence.",
        }
