"""Retrieve a versioned Genomics England PanelApp gene panel by stable numeric ID."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import httpx
from medai_science.sources.bounded_http import read_bounded_response


_PANEL_VERSION = re.compile(r"^[0-9]{1,4}(?:\.[0-9]{1,4}){0,3}$")
_MAX_RESPONSE_BYTES = 8_000_000
_MAX_GENES = 10_000


def _strings(value: Any, limit: int = 30, length: int = 1000) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip()[:length] for item in value[:limit] if isinstance(item, str) and item.strip()]


class PanelAppClient:
    endpoint = "https://panelapp.genomicsengland.co.uk/api/v1/panels"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def _validate(panel_id: int, version: str | None) -> tuple[int, str | None]:
        if isinstance(panel_id, bool) or not isinstance(panel_id, int) or not 1 <= panel_id <= 10_000_000:
            raise ValueError("PanelApp panel ID must be a positive numeric identifier")
        if version is not None and (not isinstance(version, str) or not _PANEL_VERSION.fullmatch(version)):
            raise ValueError("PanelApp version must be a dotted numeric version such as 1.2")
        return panel_id, version

    async def get_panel(self, panel_id: int, version: str | None = None) -> dict[str, Any]:
        panel_id, version = self._validate(panel_id, version)
        params = {"format": "json"}
        if version is not None:
            params["version"] = version
        url = f"{self.endpoint}/{panel_id}/"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", url, params=params, headers={"Accept": "application/json"}) as response:
                if response.status_code == 404:
                    return {"source": "Genomics England PanelApp", "panel_id": panel_id, "requested_version": version, "found": False, "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}
                body = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "PanelApp")
        try:
            raw = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("PanelApp returned invalid JSON") from error
        if not isinstance(raw, dict) or raw.get("id") != panel_id:
            raise ValueError("PanelApp response panel ID does not match the requested identifier")
        resolved_version = raw.get("version")
        if not isinstance(resolved_version, str) or not _PANEL_VERSION.fullmatch(resolved_version):
            raise ValueError("PanelApp panel response has an invalid version")
        if version is not None and resolved_version != version:
            raise ValueError("PanelApp returned a different version than requested")
        genes = raw.get("genes")
        if not isinstance(genes, list) or len(genes) > _MAX_GENES:
            raise ValueError("PanelApp panel is missing its gene list or exceeds 10000 entities")
        normalized_genes = []
        for entity in genes:
            if not isinstance(entity, dict):
                continue
            entity_type = entity.get("entity_type")
            if not isinstance(entity_type, str) or entity_type.casefold() != "gene":
                continue
            gene_data = entity.get("gene_data") if isinstance(entity.get("gene_data"), dict) else {}
            confidence = entity.get("confidence_level")
            if isinstance(confidence, (dict, list)) or not isinstance(confidence, (str, int, float)):
                confidence = None
            hgnc = gene_data.get("hgnc_id")
            normalized_genes.append({
                "gene_symbol": str(entity.get("entity_name", ""))[:100],
                "hgnc_id": hgnc if isinstance(hgnc, str) and re.fullmatch(r"HGNC:[0-9]+", hgnc) else None,
                "ensembl_gene_ids": _strings(gene_data.get("ensembl_genes"), limit=20, length=40),
                "confidence_level_source_value": confidence,
                "mode_of_inheritance": str(entity.get("mode_of_inheritance", ""))[:500],
                "mode_of_pathogenicity": str(entity.get("mode_of_pathogenicity", ""))[:500] if entity.get("mode_of_pathogenicity") is not None else None,
                "phenotypes": _strings(entity.get("phenotypes"), limit=30),
                "evidence": _strings(entity.get("evidence"), limit=30),
                "publications": _strings(entity.get("publications"), limit=30, length=300),
                "transcripts": _strings(entity.get("transcript"), limit=30, length=100),
                "source_entity_type": entity_type,
            })
        canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        panel = {
            "source": "Genomics England PanelApp",
            "source_url": url + (f"?version={version}" if version else ""),
            "reuse_terms_url": "https://panelapp.genomicsengland.co.uk/",
            "panel_id": panel_id,
            "panel_name": str(raw.get("name", ""))[:500],
            "disease_group": str(raw.get("disease_group", ""))[:300] or None,
            "disease_sub_group": str(raw.get("disease_sub_group", ""))[:300] or None,
            "status": str(raw.get("status", ""))[:100] or None,
            "version": resolved_version,
            "version_created": raw.get("version_created"),
            "relevant_disorders": _strings(raw.get("relevant_disorders"), limit=50),
            "panel_types": [
                {"name": str(item.get("name", ""))[:200], "slug": str(item.get("slug", ""))[:100]}
                for item in raw.get("types", [])[:20] if isinstance(item, dict)
            ] if isinstance(raw.get("types"), list) else [],
            "genes": normalized_genes,
            "gene_entity_count": len(normalized_genes),
            "non_gene_entity_count": len(genes) - len(normalized_genes),
            "source_record_sha256": hashlib.sha256(canonical).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "interpretation_limit": "PanelApp expert-review confidence levels and inheritance annotations are reported as the source provides them. They are not combined with other source scores and do not establish a diagnosis or pathogenicity for an individual variant.",
        }
        if len(json.dumps(panel, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) > 2_000_000:
            raise ValueError("normalized PanelApp artifact exceeds the 2 MB limit")
        return panel
