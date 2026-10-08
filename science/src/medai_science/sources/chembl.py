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
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("ChEMBL returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("ChEMBL returned an unexpected record shape")
        return value

    @staticmethod
    def _rows(payload: dict, key: str) -> list[dict]:
        rows = payload.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"ChEMBL returned an invalid {key} record set")
        return rows[:_MAX_ROWS]

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

    async def investigate(self, inchikey: str) -> dict:
        if not isinstance(inchikey, str) or not _INCHIKEY.fullmatch(inchikey.strip().upper()):
            raise ValueError("provide a standard 27-character InChIKey from a resolved chemical identity")
        inchikey = inchikey.strip().upper()
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            payload = await self._get(client, "molecule", {"molecule_structures__standard_inchi_key__exact": inchikey, "limit": str(_MAX_ROWS), "only": "molecule_chembl_id,molecule_structures,molecule_hierarchy,pref_name,first_approval,therapeutic_flag"})
            molecules = self._rows(payload, "molecules")
            exact = []
            for row in molecules:
                structures = row.get("molecule_structures") or {}
                if structures.get("standard_inchi_key", "").upper() == inchikey and _CHEMBL_ID.fullmatch(str(row.get("molecule_chembl_id", ""))):
                    exact.append(row)
            if not exact:
                return {"inchikey": inchikey, "source": "ChEMBL", "molecule_matches": [], "mechanisms": [], "indications": [], "warnings": [], "research_only": True, "interpretation": "No exact ChEMBL molecule identity was found; no pharmacology claims were joined."}
            ids = sorted({str((row.get("molecule_hierarchy") or {}).get("parent_chembl_id") or row["molecule_chembl_id"]) for row in exact})
            if any(not _CHEMBL_ID.fullmatch(value) for value in ids):
                raise ValueError("ChEMBL returned an invalid parent molecule identifier")
            query_ids = ",".join(ids)
            params = {"molecule_chembl_id__in": query_ids, "limit": str(_MAX_ROWS), "offset": "0"}
            mechanisms_payload, indications_payload, warnings_payload = await asyncio.gather(
                self._get(client, "mechanism", params),
                self._get(client, "drug_indication", params),
                self._get(client, "drug_warning", params),
            )
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        mechanisms = self._rows(mechanisms_payload, "mechanisms")
        indications = self._rows(indications_payload, "drug_indications")
        warnings = self._rows(warnings_payload, "drug_warnings")
        allowed_ids = set(ids)
        for key, rows in (("mechanism", mechanisms), ("drug indication", indications), ("drug warning", warnings)):
            for row in rows:
                record_id = row.get("molecule_chembl_id")
                if record_id not in allowed_ids:
                    raise ValueError(f"ChEMBL {key} response did not match exact parent molecule IDs")
        return {
            "inchikey": inchikey,
            "source": "ChEMBL web services",
            "molecule_matches": [self._receipt("ChEMBL", "molecule", row, retrieved_at) for row in exact],
            "parent_chembl_ids": ids,
            "mechanisms": [self._receipt("ChEMBL mechanism", "mechanism", row, retrieved_at) for row in mechanisms],
            "indications": [self._receipt("ChEMBL drug indication", "drug_indication", row, retrieved_at) for row in indications],
            "warnings": [self._receipt("ChEMBL drug warning", "drug_warning", row, retrieved_at) for row in warnings],
            "record_limits": {
                "molecule_matches": self._page_receipt(payload, "molecules"),
                "mechanisms": self._page_receipt(mechanisms_payload, "mechanisms"),
                "indications": self._page_receipt(indications_payload, "drug_indications"),
                "warnings": self._page_receipt(warnings_payload, "drug_warnings"),
            },
            "interpretation": "Provider-reported approved-drug records linked through exact standard InChIKey and ChEMBL parent IDs. Missing records do not establish absence of a mechanism, indication, or warning; this is not a clinical recommendation.",
            "research_only": True,
        }

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
