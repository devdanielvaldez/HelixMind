"""Bounded ChEMBL pharmacology lookups linked by exact InChIKey identity."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import asyncio
from urllib.parse import urlencode

import httpx
from medai_science.sources.bounded_http import read_bounded_response

_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_CHEMBL_ID = re.compile(r"^CHEMBL[0-9]+$")
_MAX_BODY = 1024 * 1024
_MAX_ROWS = 50


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("ChEMBL returned duplicate JSON object keys")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"ChEMBL returned non-standard JSON number {value}")


class ChEMBLClient:
    base_url = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    async def _get(self, client: httpx.AsyncClient, resource: str, params: dict[str, str]) -> dict:
        url = f"{self.base_url}/{resource}.json?{urlencode(params)}"
        async with client.stream("GET", url, headers={"Accept": "application/json", "User-Agent": "MedAI computational biomedical research"}) as response:
            if response.status_code == 404:
                return {}
            body = await read_bounded_response(response, _MAX_BODY, "ChEMBL")
        try:
            value = json.loads(body, object_pairs_hook=_unique_object, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            raise ValueError("ChEMBL returned invalid, ambiguous, or non-standard JSON") from error
        if not isinstance(value, dict):
            raise ValueError("ChEMBL returned an unexpected record shape")
        return value

    @staticmethod
    def _rows(payload: dict, key: str) -> list[dict]:
        rows = payload.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"ChEMBL returned an invalid {key} record set")
        return rows[:_MAX_ROWS]

    @classmethod
    def _exact_molecules(cls, payload: dict, inchikey: str) -> list[dict]:
        exact = []
        for row in cls._rows(payload, "molecules"):
            structures = row.get("molecule_structures")
            observed_key = structures.get("standard_inchi_key") if isinstance(structures, dict) else None
            molecule_id = row.get("molecule_chembl_id")
            if isinstance(observed_key, str) and observed_key.upper() == inchikey and isinstance(molecule_id, str) and _CHEMBL_ID.fullmatch(molecule_id):
                exact.append(row)
        return exact

    @staticmethod
    def _exact_parent_ids(exact: list[dict]) -> list[str]:
        ids = set()
        for row in exact:
            hierarchy = row.get("molecule_hierarchy") or {}
            if not isinstance(hierarchy, dict):
                raise ValueError("ChEMBL returned malformed molecule hierarchy metadata")
            molecule_id = row.get("molecule_chembl_id")
            parent_id = hierarchy.get("parent_chembl_id") or molecule_id
            if not isinstance(parent_id, str):
                raise ValueError("ChEMBL returned malformed molecule hierarchy metadata")
            ids.add(parent_id)
        return sorted(ids)

    @staticmethod
    def _page_receipt(payload: dict, key: str) -> dict:
        rows = payload.get(key, [])
        meta = payload.get("page_meta") or {}
        total = meta.get("total_count") if isinstance(meta, dict) else None
        return {"returned": len(rows) if isinstance(rows, list) else 0, "provider_total": total if isinstance(total, int) else None, "truncated": isinstance(total, int) and total > _MAX_ROWS, "limit": _MAX_ROWS}

    @staticmethod
    def _receipt(source: str, resource: str, row: dict, retrieved_at: str) -> dict:
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        molecule_id = row.get("molecule_chembl_id")
        source_url = f"https://www.ebi.ac.uk/chembl/api/data/{resource}.json?molecule_chembl_id={molecule_id}" if molecule_id else f"https://www.ebi.ac.uk/chembl/api/data/{resource}.json"
        return {"record": row, "source": source, "source_url": source_url, "retrieved_at": retrieved_at, "record_sha256": hashlib.sha256(canonical.encode()).hexdigest()}

    @classmethod
    def normalize_investigation(
        cls,
        inchikey: str,
        molecule_payload: dict,
        mechanisms_payload: dict,
        indications_payload: dict,
        warnings_payload: dict,
        retrieved_at: str,
    ) -> dict:
        """Normalize retained ChEMBL responses deterministically for offline replay."""
        if not isinstance(inchikey, str) or not _INCHIKEY.fullmatch(inchikey):
            raise ValueError("ChEMBL replay requires a standard uppercase InChIKey")
        if not isinstance(retrieved_at, str) or not retrieved_at:
            raise ValueError("ChEMBL replay requires its captured retrieval timestamp")
        exact = cls._exact_molecules(molecule_payload, inchikey)
        if not exact:
            return {"inchikey": inchikey, "source": "ChEMBL", "molecule_matches": [], "mechanisms": [], "indications": [], "warnings": [], "research_only": True, "interpretation": "No exact ChEMBL molecule identity was found; no pharmacology claims were joined."}
        ids = cls._exact_parent_ids(exact)
        if any(not _CHEMBL_ID.fullmatch(value) for value in ids):
            raise ValueError("ChEMBL returned an invalid parent molecule identifier")
        mechanisms = cls._rows(mechanisms_payload, "mechanisms")
        indications = cls._rows(indications_payload, "drug_indications")
        warnings = cls._rows(warnings_payload, "drug_warnings")
        allowed_ids = set(ids)
        for key, rows in (("mechanism", mechanisms), ("drug indication", indications), ("drug warning", warnings)):
            for row in rows:
                if row.get("molecule_chembl_id") not in allowed_ids:
                    raise ValueError(f"ChEMBL {key} response did not match exact parent molecule IDs")
        return {
            "inchikey": inchikey,
            "source": "ChEMBL web services",
            "molecule_matches": [cls._receipt("ChEMBL", "molecule", row, retrieved_at) for row in exact],
            "parent_chembl_ids": ids,
            "mechanisms": [cls._receipt("ChEMBL mechanism", "mechanism", row, retrieved_at) for row in mechanisms],
            "indications": [cls._receipt("ChEMBL drug indication", "drug_indication", row, retrieved_at) for row in indications],
            "warnings": [cls._receipt("ChEMBL drug warning", "drug_warning", row, retrieved_at) for row in warnings],
            "record_limits": {
                "molecule_matches": cls._page_receipt(molecule_payload, "molecules"),
                "mechanisms": cls._page_receipt(mechanisms_payload, "mechanisms"),
                "indications": cls._page_receipt(indications_payload, "drug_indications"),
                "warnings": cls._page_receipt(warnings_payload, "drug_warnings"),
            },
            "interpretation": "Provider-reported approved-drug records linked through exact standard InChIKey and ChEMBL parent IDs. Missing records do not establish absence of a mechanism, indication, or warning; this is not a clinical recommendation.",
            "research_only": True,
        }

    async def investigate(self, inchikey: str) -> dict:
        if not isinstance(inchikey, str) or not _INCHIKEY.fullmatch(inchikey.strip().upper()):
            raise ValueError("provide a standard 27-character InChIKey from a resolved chemical identity")
        inchikey = inchikey.strip().upper()
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            payload = await self._get(client, "molecule", {"molecule_structures__standard_inchi_key__exact": inchikey, "limit": str(_MAX_ROWS), "only": "molecule_chembl_id,molecule_structures,molecule_hierarchy,pref_name,first_approval,therapeutic_flag"})
            exact = self._exact_molecules(payload, inchikey)
            exact_ids = self._exact_parent_ids(exact)
            if exact_ids and any(not _CHEMBL_ID.fullmatch(value) for value in exact_ids):
                raise ValueError("ChEMBL returned an invalid parent molecule identifier")
            params = {"molecule_chembl_id__in": ",".join(exact_ids), "limit": str(_MAX_ROWS), "offset": "0"}
            mechanisms_payload, indications_payload, warnings_payload = await asyncio.gather(
                self._get(client, "mechanism", params),
                self._get(client, "drug_indication", params),
                self._get(client, "drug_warning", params),
            ) if exact_ids else ({}, {}, {})
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return self.normalize_investigation(inchikey, payload, mechanisms_payload, indications_payload, warnings_payload, retrieved_at)

    async def compare_shared_targets(self, inchikey_a: str, inchikey_b: str) -> dict:
        """Compare source-reported ChEMBL mechanism targets for exact chemical identities."""
        if not all(isinstance(value, str) and _INCHIKEY.fullmatch(value.strip().upper()) for value in (inchikey_a, inchikey_b)):
            raise ValueError("provide two standard 27-character InChIKeys from resolved chemical identities")
        if inchikey_a.strip().upper() == inchikey_b.strip().upper():
            raise ValueError("shared-target screening requires two distinct chemical identities")
        profiles = await asyncio.gather(self.investigate(inchikey_a), self.investigate(inchikey_b))

        def target_map(profile: dict) -> dict[str, list[dict]]:
            targets: dict[str, list[dict]] = {}
            for receipt in profile["mechanisms"]:
                record = receipt["record"]
                target_id = record.get("target_chembl_id")
                if isinstance(target_id, str) and _CHEMBL_ID.fullmatch(target_id):
                    targets.setdefault(target_id, []).append(receipt)
            return targets

        targets_a, targets_b = map(target_map, profiles)
        shared = sorted(set(targets_a) & set(targets_b))
        return {
            "source": "ChEMBL web services",
            "compound_a": {"inchikey": profiles[0]["inchikey"], "parent_chembl_ids": profiles[0].get("parent_chembl_ids", []), "target_ids": sorted(targets_a)},
            "compound_b": {"inchikey": profiles[1]["inchikey"], "parent_chembl_ids": profiles[1].get("parent_chembl_ids", []), "target_ids": sorted(targets_b)},
            "shared_targets": [{"target_chembl_id": target_id, "compound_a_mechanism_records": targets_a[target_id], "compound_b_mechanism_records": targets_b[target_id]} for target_id in shared],
            "target_overlap": {"shared_count": len(shared), "interpretation": "descriptive intersection of provider-reported mechanism target identifiers"},
            "limitations": ["A shared target is a candidate pharmacology link, not evidence of a drug-drug interaction.", "No shared target does not establish absence of a pharmacologic relationship.", "This screen does not assess potency, exposure, dose, clinical outcomes, or interaction direction."],
            "research_only": True,
        }
