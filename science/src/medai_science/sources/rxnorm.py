"""Exact-or-normalized RxNorm name resolution to ingredient concepts."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re

import httpx

from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import read_bounded_response

_RXCUI = re.compile(r"^[0-9]{1,12}$")
_UNII = re.compile(r"^[A-Z0-9]{10}$")
_MAX_BODY = 512 * 1024
_MAX_SEARCH_IDS = 10


class RxNormClient:
    base_url = "https://rxnav.nlm.nih.gov/REST"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 20.0):
        self.transport = transport
        self.timeout = timeout

    async def _get(self, client: httpx.AsyncClient, path: str, params: dict[str, str] | None = None) -> dict:
        async with client.stream("GET", f"{self.base_url}/{path}", params=params, headers={"Accept": "application/json", "User-Agent": "MedAI computational biomedical research"}) as response:
            if response.status_code == 404:
                return {}
            body = await read_bounded_response(response, _MAX_BODY, "RxNorm")
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("RxNorm returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("RxNorm returned an unexpected response shape")
        return payload

    async def resolve_ingredient(self, name: str) -> dict:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 128:
            raise ValueError("drug name must contain 1..128 characters")
        query = ensure_public_research_query(name.strip())
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            search = await self._get(client, "rxcui.json", {"name": query, "search": "2", "allsrc": "0"})
            raw_ids = ((search.get("idGroup") or {}).get("rxnormId") or [])
            if isinstance(raw_ids, str):
                raw_ids = [raw_ids]
            if not isinstance(raw_ids, list) or any(not isinstance(value, (str, int)) or not _RXCUI.fullmatch(str(value)) for value in raw_ids):
                raise ValueError("RxNorm returned invalid RxCUI values")
            candidate_ids = sorted(set(map(str, raw_ids)))
            if len(candidate_ids) > _MAX_SEARCH_IDS:
                return {"query": query, "source": "RxNorm", "candidate_rxcuis": candidate_ids[:_MAX_SEARCH_IDS], "too_many_matches": True, "ingredients": [], "resolution_status": "ambiguous", "research_only": True}
            related_payloads = await asyncio.gather(*[
                self._get(client, f"rxcui/{rxcui}/related.json", {"tty": "IN"}) for rxcui in candidate_ids
            ])
            ingredient_ids: set[str] = set()
            for payload in related_payloads:
                groups = ((payload.get("relatedGroup") or {}).get("conceptGroup") or [])
                if not isinstance(groups, list):
                    continue
                for group in groups:
                    if not isinstance(group, dict) or group.get("tty") != "IN":
                        continue
                    concepts = group.get("conceptProperties") or []
                    if isinstance(concepts, dict):
                        concepts = [concepts]
                    if isinstance(concepts, list):
                        ingredient_ids.update(str(item.get("rxcui")) for item in concepts if isinstance(item, dict) and _RXCUI.fullmatch(str(item.get("rxcui", ""))))
            properties = await asyncio.gather(*[
                self._get(client, f"rxcui/{rxcui}/allProperties.json", {"prop": "names attributes codes"}) for rxcui in sorted(ingredient_ids)
            ])
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        ingredients = []
        for rxcui, payload in zip(sorted(ingredient_ids), properties):
            concepts = ((payload.get("propConceptGroup") or {}).get("propConcept") or [])
            if isinstance(concepts, dict):
                concepts = [concepts]
            if not isinstance(concepts, list):
                raise ValueError("RxNorm returned malformed ingredient properties")
            values: dict[str, list[str]] = {}
            for prop in concepts:
                if isinstance(prop, dict) and isinstance(prop.get("propName"), str) and isinstance(prop.get("propValue"), str):
                    values.setdefault(prop["propName"], []).append(prop["propValue"])
            names = sorted(set(values.get("RxNorm Name", []) + values.get("RXNAV_STR", [])))
            uniis = sorted({value.upper() for value in values.get("UNII_CODE", []) if _UNII.fullmatch(value.upper())})
            ingredients.append({"rxcui": rxcui, "tty": "IN", "names": names, "unii_codes": uniis, "source": "RxNorm API", "source_url": f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/allProperties.json?prop=names+attributes+codes", "retrieved_at": retrieved_at, "record_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        status = "resolved_unique_ingredient" if len(ingredients) == 1 else "unresolved" if not ingredients else "ambiguous_multiple_ingredients"
        return {
            "query": query,
            "source": "RxNorm API",
            "search_precision": "exact_or_normalized_only; approximate matching was not used",
            "candidate_rxcuis": candidate_ids,
            "ingredients": ingredients,
            "resolution_status": status,
            "research_only": True,
            "interpretation": "Only RxNorm ingredient (TTY=IN) concepts reached through the exact-or-normalized name search are returned. Multiple ingredients remain ambiguous; names are not silently resolved to one substance.",
        }
