"""MyGene.info v3 gene annotation search with provider provenance."""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query
from medai_science.json_input import strict_json_loads

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class MyGeneClient:
    endpoint = "https://mygene.info/v3/query"
    fields = "_id,_score,symbol,name,taxid,entrezgene,ensembl,summary,genomic_pos,alias,type_of_gene"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 20):
        self.transport = transport
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10, species: str = "human") -> list[dict]:
        query = ensure_public_research_query(query)
        if len(query) > 256:
            raise ValueError("gene query cannot exceed 256 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("gene result limit must be 1..20")
        if not isinstance(species, str) or species not in {"human", "mouse", "rat", "zebrafish", "pig", "fruitfly", "nematode", "thale-cress", "frog"}:
            raise ValueError("unsupported MyGene.info common species filter")
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", self.endpoint, params={"q": query, "species": species, "fields": self.fields, "size": limit}) as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "MyGene.info")
        try:
            payload = strict_json_loads(raw)
        except ValueError as error:
            raise ValueError(f"MyGene.info returned invalid strict JSON: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list) or len(payload["hits"]) > limit:
            raise ValueError("MyGene.info returned a malformed or over-limit hit collection")
        hits = payload["hits"]
        if any(not isinstance(hit, dict) or not isinstance(hit.get("_id"), (str, int)) or isinstance(hit.get("_id"), bool) or not str(hit["_id"]).strip() for hit in hits):
            raise ValueError("MyGene.info returned a malformed hit or missing stable source ID")
        retrieved_at = datetime.now(timezone.utc).isoformat()
        return [
            {
                "source": "MyGene.info",
                "source_id": str(hit.get("_id", "")),
                "query": query,
                "species": species,
                "retrieved_at": retrieved_at,
                "record": hit,
            }
            for hit in hits
        ]
