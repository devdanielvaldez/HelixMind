"""MyGene.info v3 gene annotation search with provider provenance."""
import json
from datetime import datetime, timezone

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query

_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class MyGeneClient:
    endpoint = "https://mygene.info/v3/query"
    fields = "_id,_score,symbol,name,taxid,entrezgene,ensembl,summary,genomic_pos,alias,type_of_gene"

    async def search(self, query: str, limit: int = 10, species: str = "human") -> list[dict]:
        query = ensure_public_research_query(query)
        if len(query) > 256:
            raise ValueError("gene query cannot exceed 256 characters")
        if not 1 <= limit <= 20:
            raise ValueError("gene result limit must be 1..20")
        if species not in {"human", "mouse", "rat", "zebrafish", "pig", "fruitfly", "nematode", "thale-cress", "frog"}:
            raise ValueError("unsupported MyGene.info common species filter")
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            async with client.stream("GET", self.endpoint, params={"q": query, "species": species, "fields": self.fields, "size": limit}) as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "MyGene.info")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("MyGene.info returned invalid JSON") from error
        retrieved_at = datetime.now(timezone.utc).isoformat()
        hits = payload.get("hits", [])
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
            if isinstance(hit, dict) and hit.get("_id") is not None
        ]
