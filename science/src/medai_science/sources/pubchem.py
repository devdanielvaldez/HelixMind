"""Bounded PubChem PUG REST lookup with sourced fields and optional RDKit descriptors."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import re
import threading
import time
from urllib.parse import quote

import httpx

from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import read_bounded_response

_CID = re.compile(r"^[1-9][0-9]{0,11}$")
_PROPERTIES = (
    "MolecularFormula", "MolecularWeight", "SMILES", "InChI", "InChIKey",
    "IUPACName", "Title", "XLogP", "ExactMass",
)
_MAX_BODY = 512 * 1024
_MAX_NAME = 256
_MAX_MATCHES = 10


class PubChemClient:
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    _request_schedule_lock = threading.Lock()
    _next_request_at = 0.0

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    @staticmethod
    def validate_identifier(identifier: str) -> str:
        if not isinstance(identifier, str):
            raise ValueError("provide a PubChem CID or compound name")
        value = identifier.strip()
        if not value or len(value) > _MAX_NAME:
            raise ValueError("compound identifier must contain 1..256 characters")
        if _CID.fullmatch(value):
            return value
        return ensure_public_research_query(value)

    async def _json(self, client: httpx.AsyncClient, url: str) -> dict:
        # Reserve a process-wide request slot without binding an asyncio lock to
        # whichever loop happened to import this module.
        while True:
            now = time.monotonic()
            with type(self)._request_schedule_lock:
                delay = type(self)._next_request_at - now
                if delay <= 0:
                    type(self)._next_request_at = now + 0.21
                    break
            await asyncio.sleep(delay)
        async with client.stream("GET", url, headers={"Accept": "application/json", "User-Agent": "MedAI biomedical research; PubChem PUG REST client"}) as response:
            if response.status_code == 404:
                raise ValueError("PubChem did not find a matching compound")
            body = await read_bounded_response(response, _MAX_BODY, "PubChem")
        try:
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("PubChem returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("PubChem returned an unexpected record shape")
        return value

    @staticmethod
    def _rdkit_descriptors(smiles: str | None) -> dict:
        if not smiles:
            return {"status": "unavailable", "reason": "PubChem did not provide SMILES"}
        try:
            from rdkit import Chem, rdBase
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        except ImportError:
            return {"status": "not_installed", "install_extra": "pip install -e '.[chemistry]'"}
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {"status": "invalid_smiles", "source_smiles": "PubChem SMILES could not be parsed by installed RDKit"}
        return {
            "status": "computed",
            "software": "RDKit",
            "version": rdBase.rdkitVersion,
            "input_smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
            "descriptors": {
                "molecular_weight_g_mol": Descriptors.MolWt(molecule),
                "tpsa_angstrom2": rdMolDescriptors.CalcTPSA(molecule),
                "h_bond_donors": Lipinski.NumHDonors(molecule),
                "h_bond_acceptors": Lipinski.NumHAcceptors(molecule),
                "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
                "lipophilicity_logp": Descriptors.MolLogP(molecule),
            },
            "limitations": "Descriptors are computed from the supplied PubChem SMILES with this RDKit version; they are not biological activity, exposure, safety, or efficacy predictions.",
        }

    async def lookup(self, identifier: str) -> dict:
        """Resolve a CID/name to a bounded list; never silently pick among ambiguous names."""
        identifier = self.validate_identifier(identifier)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            if _CID.fullmatch(identifier):
                cids = [identifier]
            else:
                encoded = quote(identifier, safe="")
                resolution = await self._json(client, f"{self.base_url}/compound/name/{encoded}/cids/JSON")
                cids = resolution.get("IdentifierList", {}).get("CID", [])
                if not isinstance(cids, list) or not cids:
                    raise ValueError("PubChem did not resolve the compound name")
                if len(cids) > _MAX_MATCHES:
                    return {"query": identifier, "source": "PubChem", "matches": [], "too_many_matches": True, "match_limit": _MAX_MATCHES, "research_only": True}
                if any(not isinstance(cid, int) or cid < 1 for cid in cids):
                    raise ValueError("PubChem returned invalid compound identifiers")

            cid_text = ",".join(str(cid) for cid in cids)
            properties = ",".join(_PROPERTIES)
            payload = await self._json(client, f"{self.base_url}/compound/cid/{cid_text}/property/{properties}/JSON")
        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("PubChem returned no compound property records")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        normalized = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("CID"), int) or row["CID"] not in {int(cid) for cid in cids}:
                raise ValueError("PubChem property response does not match the requested CID set")
            record_hash = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            normalized.append({
                "cid": row["CID"],
                "title": row.get("Title"),
                "iupac_name": row.get("IUPACName"),
                "molecular_formula": row.get("MolecularFormula"),
                "molecular_weight_g_mol": row.get("MolecularWeight"),
                "exact_mass": row.get("ExactMass"),
                "xlogp": row.get("XLogP"),
                "smiles": row.get("SMILES"),
                "inchi": row.get("InChI"),
                "inchikey": row.get("InChIKey"),
                "rdkit": self._rdkit_descriptors(row.get("SMILES")),
                "source": "PubChem PUG REST",
                "source_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}",
                "retrieved_at": retrieved_at,
                "record_sha256": record_hash,
            })
        returned = {row["cid"] for row in normalized}
        if returned != {int(cid) for cid in cids}:
            raise ValueError("PubChem returned an incomplete compound property set")
        return {
            "query": identifier,
            "source": "PubChem PUG REST",
            "matches": normalized,
            "ambiguous": len(normalized) > 1,
            "interpretation": "Multiple chemical records matched; review CID identity before connecting this record to a biological claim." if len(normalized) > 1 else "Chemical identity record only; no target, mechanism, efficacy, or clinical safety inference was made.",
            "research_only": True,
        }
