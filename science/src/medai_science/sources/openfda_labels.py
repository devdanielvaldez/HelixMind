"""Bounded retrieval and cross-label screening of FDA drug-interaction sections."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re

import httpx

from medai_science.sources.rxnorm import RxNormClient
from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import read_bounded_response

_MAX_BODY = 2 * 1024 * 1024
_MAX_LABELS = 5
_MAX_SECTION_CHARS = 32_000
_MAX_NAME = 128


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _record_hash(record: dict) -> str:
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _mentions(section: str, term: str) -> bool:
    normalized_term = _normalized(term)
    if not normalized_term:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, _normalized(section)) is not None


class OpenFDALabelClient:
    """Searches public labels; returned text is source evidence, not clinical advice."""

    base_url = "https://api.fda.gov/drug/label.json"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0, rxnorm_client: RxNormClient | None = None):
        self.transport = transport
        self.timeout = timeout
        self.rxnorm_client = rxnorm_client or RxNormClient(transport=transport, timeout=timeout)

    @staticmethod
    def validate_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > _MAX_NAME:
            raise ValueError("drug name must contain 1..128 characters")
        return ensure_public_research_query(value.strip())

    async def _search(self, client: httpx.AsyncClient, identity: dict) -> tuple[list[dict], int | None]:
        ingredients = identity.get("ingredients", [])
        ingredient = ingredients[0]
        uniis = ingredient.get("unii_codes", [])
        if uniis:
            field, identifiers = "unii", uniis
        else:
            field, identifiers = "rxcui", [ingredient["rxcui"]]
        # RxNorm identifiers are restricted to validated alphanumeric codes.
        search = " OR ".join(f"openfda.{field}:{value}" for value in identifiers)
        response_payload: dict
        async with client.stream(
            "GET",
            self.base_url,
            params={"search": search, "limit": str(_MAX_LABELS)},
            headers={"Accept": "application/json", "User-Agent": "MedAI computational biomedical research"},
        ) as response:
            if response.status_code == 404:
                return [], 0
            body = await read_bounded_response(response, _MAX_BODY, "openFDA label")
        try:
            response_payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("openFDA returned invalid JSON") from error
        if not isinstance(response_payload, dict):
            raise ValueError("openFDA returned an unexpected label response")
        rows = response_payload.get("results", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("openFDA returned an invalid label record set")
        total = (response_payload.get("meta") or {}).get("results", {}).get("total")
        matched_rows = []
        wanted = set(identifiers)
        for row in rows:
            openfda = row.get("openfda") if isinstance(row.get("openfda"), dict) else {}
            values = openfda.get("unii" if field == "unii" else "rxcui", [])
            if isinstance(values, str):
                values = [values]
            if isinstance(values, list) and wanted.intersection(str(value).upper() for value in values):
                matched_rows.append(row)
        return matched_rows[:_MAX_LABELS], total if isinstance(total, int) else None

    @staticmethod
    def _label(row: dict, drug_name: str, identity: dict, retrieved_at: str) -> dict:
        openfda = row.get("openfda") if isinstance(row.get("openfda"), dict) else {}
        sections = row.get("drug_interactions", [])
        if isinstance(sections, str):
            sections = [sections]
        if not isinstance(sections, list) or any(not isinstance(item, str) for item in sections):
            raise ValueError("openFDA label contains malformed drug_interactions section")
        full_text = "\n".join(sections)
        digest = hashlib.sha256(full_text.encode()).hexdigest()
        return {
            "query_name": drug_name,
            "rxnorm_ingredient": identity["ingredients"][0],
            "set_id": row.get("set_id"),
            "version": row.get("version"),
            "effective_time": row.get("effective_time"),
            "brand_name": openfda.get("brand_name", []),
            "generic_name": openfda.get("generic_name", []),
            "manufacturer_name": openfda.get("manufacturer_name", []),
            "product_type": openfda.get("product_type", []),
            "rxcui": openfda.get("rxcui", []),
            "unii": openfda.get("unii", []),
            "ndc": openfda.get("ndc", []),
            "drug_interactions": full_text[:_MAX_SECTION_CHARS],
            "section_sha256": digest,
            "section_truncated": len(full_text) > _MAX_SECTION_CHARS,
            "record_sha256": _record_hash(row),
            "source": "openFDA drug labeling",
            "source_url": "https://api.fda.gov/drug/label.json",
            "retrieved_at": retrieved_at,
        }

    async def investigate_pair(self, drug_a: str, drug_b: str) -> dict:
        name_a = self.validate_name(drug_a)
        name_b = self.validate_name(drug_b)
        if _normalized(name_a) == _normalized(name_b):
            raise ValueError("provide two distinct generic drug names")
        identities = await asyncio.gather(
            self.rxnorm_client.resolve_ingredient(name_a),
            self.rxnorm_client.resolve_ingredient(name_b),
        )
        unresolved = [identity for identity in identities if identity.get("resolution_status") != "resolved_unique_ingredient"]
        if unresolved:
            return {
                "drug_queries": [name_a, name_b],
                "source": "RxNorm API and openFDA drug labeling",
                "ingredient_resolutions": identities,
                "labels": {name_a: [], name_b: []},
                "search_limits": {},
                "label_mention_screen": {"classification": "unresolved_or_ambiguous_drug_identity", "interpretation": "Label retrieval stopped because one or both names did not resolve to exactly one RxNorm ingredient."},
                "research_only": True,
                "clinical_advice": False,
            }
        if identities[0]["ingredients"][0]["rxcui"] == identities[1]["ingredients"][0]["rxcui"]:
            raise ValueError("the two names resolve to the same RxNorm ingredient")
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            (rows_a, total_a), (rows_b, total_b) = await asyncio.gather(
                self._search(client, identities[0]), self._search(client, identities[1])
            )
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        labels_a = [self._label(row, name_a, identities[0], retrieved_at) for row in rows_a]
        labels_b = [self._label(row, name_b, identities[1], retrieved_at) for row in rows_b]
        names_a = [name_a] + identities[0]["ingredients"][0].get("names", [])
        names_b = [name_b] + identities[1]["ingredients"][0].get("names", [])
        mentions_a = [label for label in labels_a if any(_mentions(label["drug_interactions"], term) for term in names_b)]
        mentions_b = [label for label in labels_b if any(_mentions(label["drug_interactions"], term) for term in names_a)]
        if mentions_a and mentions_b:
            screening = "both_label_sets_mention_the_other_query_name"
        elif mentions_a or mentions_b:
            screening = "asymmetric_label_mention_requires_review"
        else:
            screening = "no_exact_query_name_mention_found_in_returned_sections"
        return {
            "drug_queries": [name_a, name_b],
            "source": "RxNorm API and openFDA drug labeling",
            "ingredient_resolutions": identities,
            "labels": {name_a: labels_a, name_b: labels_b},
            "search_limits": {
                name_a: {"returned": len(labels_a), "provider_total": total_a, "truncated": isinstance(total_a, int) and total_a > _MAX_LABELS, "limit": _MAX_LABELS},
                name_b: {"returned": len(labels_b), "provider_total": total_b, "truncated": isinstance(total_b, int) and total_b > _MAX_LABELS, "limit": _MAX_LABELS},
            },
            "label_mention_screen": {
                "classification": screening,
                "drug_a_labels_mentioning_drug_b_name": [label["set_id"] for label in mentions_a],
                "drug_b_labels_mentioning_drug_a_name": [label["set_id"] for label in mentions_b],
                "interpretation": "Lexical screening of returned label sections only. Asymmetry is not a contradiction, reciprocal mention is not proof of an interaction, and no mention is not evidence of absence. Review exact products, doses, routes, populations, label dates, context, and primary evidence.",
            },
            "research_only": True,
            "clinical_advice": False,
        }
