"""Bounded retrieval of versioned DailyMed SPL interaction sections."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import re
import xml.etree.ElementTree as ET
from medai_science.sources.xml_safety import reject_xml_declarations
from zipfile import BadZipFile, ZipFile

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query

_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_MAX_BODY = 8 * 1024 * 1024
_MAX_LABELS = 3
_MAX_HISTORICAL_VERSIONS = 2
_MAX_SECTION_CHARS = 32_000
_SET_ID = re.compile(r"^[0-9a-fA-F-]{36}$")
_NS = "{urn:hl7-org:v3}"


def _record_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _section_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


class DailyMedLabelClient:
    """Reads current and bounded historical SPL labels by stable RxNorm ingredient IDs."""

    base_url = _BASE

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 30.0):
        self.transport = transport
        self.timeout = timeout

    async def _json(self, client: httpx.AsyncClient, path: str, params: dict[str, str]) -> dict:
        async with client.stream("GET", f"{self.base_url}/{path}", params=params, headers={"Accept": "application/json", "User-Agent": "MedAI computational biomedical research"}) as response:
            if response.status_code == 404:
                return {}
            chunks = await read_bounded_response(response, _MAX_BODY, "DailyMed")
        try:
            payload = json.loads(chunks)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("DailyMed returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("DailyMed returned an unexpected listing response")
        return payload

    async def _xml(self, client: httpx.AsyncClient, set_id: str) -> bytes:
        if not _SET_ID.fullmatch(set_id):
            raise ValueError("DailyMed returned an invalid SPL set ID")
        async with client.stream("GET", f"{self.base_url}/spls/{set_id}.xml", headers={"Accept": "application/xml", "User-Agent": "MedAI computational biomedical research"}) as response:
            if response.status_code == 404:
                return b""
            chunks = await read_bounded_response(response, _MAX_BODY, "DailyMed")
        return chunks

    async def _historical_xml(self, client: httpx.AsyncClient, set_id: str, version: str) -> bytes:
        if not _SET_ID.fullmatch(set_id) or not re.fullmatch(r"[0-9]{1,12}", version):
            raise ValueError("DailyMed historical SPL requires an exact set ID and numeric version")
        async with client.stream(
            "GET", "https://dailymed.nlm.nih.gov/dailymed/getFile.cfm",
            params={"type": "zip", "setid": set_id, "version": version},
            headers={"Accept": "application/zip", "User-Agent": "MedAI computational biomedical research"},
        ) as response:
            if response.status_code == 404:
                return b""
            raw = await read_bounded_response(response, _MAX_BODY, "DailyMed historical SPL archive")
        try:
            with ZipFile(BytesIO(raw)) as archive:
                files = [item for item in archive.infolist() if not item.is_dir()]
                xml_files = [item for item in files if item.filename.casefold().endswith(".xml")]
                if len(files) > 8 or len(xml_files) != 1 or xml_files[0].file_size > _MAX_BODY:
                    raise ValueError("DailyMed historical SPL archive has an unsupported file layout or size")
                xml_bytes = archive.read(xml_files[0])
        except BadZipFile as error:
            raise ValueError("DailyMed returned a malformed historical SPL archive") from error
        reject_xml_declarations(xml_bytes, "DailyMed historical SPL")
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as error:
            raise ValueError("DailyMed returned malformed historical SPL XML") from error
        set_id_element = root.find(f"{_NS}setId")
        version_element = root.find(f"{_NS}versionNumber")
        if (set_id_element is None or set_id_element.attrib.get("root", "").casefold() != set_id.casefold()
                or version_element is None or version_element.attrib.get("value") != version):
            raise ValueError("DailyMed historical SPL did not match the requested set ID and version")
        return xml_bytes

    async def _version_history(self, client: httpx.AsyncClient, set_id: str, current_version: object) -> dict:
        if not _SET_ID.fullmatch(set_id):
            raise ValueError("DailyMed returned an invalid SPL set ID")
        payload = await self._json(client, f"spls/{set_id}/history.json", {"pagesize": "100", "page": "1"})
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        rows = data.get("history", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("DailyMed returned invalid SPL version history")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        records = []
        for row in rows:
            version = row.get("spl_version")
            if not isinstance(version, str) or not re.fullmatch(r"[0-9]{1,12}", version) or str(current_version) == version:
                continue
            archive = await self._historical_xml(client, set_id, version)
            section, section_code = self._interaction_section(archive)
            records.append({
                "spl_version": version,
                "published_date": row.get("published_date"),
                "interaction_section_code": section_code,
                "drug_interactions": section[:_MAX_SECTION_CHARS],
                "section_truncated": len(section) > _MAX_SECTION_CHARS,
                "section_sha256": hashlib.sha256(section.encode()).hexdigest() if section else None,
                "spl_sha256": hashlib.sha256(archive).hexdigest() if archive else None,
                "history_record_sha256": _record_hash(row),
                "archive_retrieved": bool(archive),
                "source_url": f"https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={set_id}&version={version}",
            })
            if len(records) >= _MAX_HISTORICAL_VERSIONS:
                break
        return {
            "records": records,
            "provider_total": int(metadata["total_elements"]) if str(metadata.get("total_elements", "")).isdigit() else len(rows),
            "retrieval_limit": _MAX_HISTORICAL_VERSIONS,
            "truncated": len(rows) > len(records) or len(rows) >= 100,
            "source_url": f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}/history.json",
            "interpretation_limit": "Historical DailyMed SPLs are earlier versions of the same label record, not independent clinical evidence; the bounded history does not cover unretained versions.",
        }

    @staticmethod
    def _interaction_section(xml_bytes: bytes) -> tuple[str, str | None]:
        if not xml_bytes:
            return "", None
        reject_xml_declarations(xml_bytes, "DailyMed SPL")
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as error:
            raise ValueError("DailyMed returned malformed SPL XML") from error
        candidates = []
        for section in root.iter(f"{_NS}section"):
            title = section.find(f"{_NS}title")
            title_text = _section_text(title) if title is not None else ""
            code = section.find(f"{_NS}code")
            code_value = code.attrib.get("code") if code is not None else None
            normalized = re.sub(r"\s+", " ", title_text).strip().casefold()
            display = (code.attrib.get("displayName", "") if code is not None else "").casefold()
            if "drug interactions" in normalized or code_value == "34066-1" or "drug interactions" in display:
                body = section.find(f"{_NS}text")
                text = _section_text(body if body is not None else section)
                if text:
                    candidates.append((text, code_value))
        if len(candidates) > 1:
            # SPLs can nest subsection elements; retain only top-level section matches.
            candidates = [candidate for candidate in candidates if candidate[0]]
        if not candidates:
            return "", None
        return candidates[0]

    async def retrieve_ingredient(self, ingredient: dict) -> dict:
        if not isinstance(ingredient, dict) or not re.fullmatch(r"[0-9]{1,12}", str(ingredient.get("rxcui", ""))):
            raise ValueError("DailyMed lookup requires a validated RxNorm ingredient RxCUI")
        rxcui = str(ingredient["rxcui"])
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            listing = await self._json(client, "spls.json", {"rxcui": rxcui, "pagesize": str(_MAX_LABELS), "page": "1"})
            rows = listing.get("data", [])
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("DailyMed returned an invalid SPL listing")
            rows = rows[:_MAX_LABELS]
            retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            labels = []
            for row in rows:
                set_id = row.get("setid")
                if not isinstance(set_id, str) or not _SET_ID.fullmatch(set_id):
                    continue
                raw_xml = await self._xml(client, set_id)
                section, section_code = self._interaction_section(raw_xml)
                labels.append({
                    "set_id": set_id,
                    "spl_version": row.get("spl_version"),
                    "published_date": row.get("published_date"),
                    "title": row.get("title"),
                    "rxcui": rxcui,
                    "interaction_section_code": section_code,
                    "drug_interactions": section[:_MAX_SECTION_CHARS],
                    "section_truncated": len(section) > _MAX_SECTION_CHARS,
                    "section_sha256": hashlib.sha256(section.encode()).hexdigest() if section else None,
                    "spl_sha256": hashlib.sha256(raw_xml).hexdigest() if raw_xml else None,
                    "record_sha256": _record_hash(row),
                    "source": "DailyMed Structured Product Label",
                    "source_url": f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml",
                    "historical_versions": await self._version_history(client, set_id, row.get("spl_version")),
                    "retrieved_at": retrieved_at,
                })
        metadata = listing.get("metadata") if isinstance(listing.get("metadata"), dict) else {}
        return {
            "ingredient": {"rxcui": rxcui, "names": ingredient.get("names", [])},
            "source": "DailyMed REST API v2",
            "source_url": f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?rxcui={rxcui}",
            "history_limit_per_label": _MAX_HISTORICAL_VERSIONS,
            "labels": labels,
            "search_limits": {"returned": len(labels), "provider_total": int(metadata["total_elements"]) if str(metadata.get("total_elements", "")).isdigit() else None, "limit": _MAX_LABELS},
            "retrieved_at": retrieved_at,
            "research_only": True,
        }

    async def investigate_pair(self, drug_a: str, drug_b: str, identities: list[dict]) -> dict:
        names = [ensure_public_research_query(value.strip()) for value in (drug_a, drug_b) if isinstance(value, str) and value.strip()]
        if len(names) != 2 or any(len(value) > 128 for value in names):
            raise ValueError("provide two public drug names containing 1..128 characters")
        if len(identities) != 2:
            raise ValueError("DailyMed pair lookup requires two RxNorm identity resolutions")
        if any(item.get("resolution_status") != "resolved_unique_ingredient" for item in identities):
            return {"drug_queries": names, "source": "DailyMed REST API v2", "labels": {}, "classification": "unresolved_or_ambiguous_drug_identity", "research_only": True}
        results = [await self.retrieve_ingredient(item["ingredients"][0]) for item in identities]
        return {"drug_queries": names, "source": "DailyMed REST API v2", "labels": {names[0]: results[0]["labels"], names[1]: results[1]["labels"]}, "search_limits": {names[0]: results[0]["search_limits"], names[1]: results[1]["search_limits"]}, "interpretation": "Current and bounded historical SPL interaction sections are versions of source label records. DailyMed and openFDA may redistribute the same FDA-originated labeling; historical versions are not independent clinical evidence and source changes are review signals only.", "research_only": True, "clinical_advice": False}


def compare_label_distributions(openfda: dict, dailymed: dict) -> dict:
    """Compare shared SPL IDs and exact-name mention signals without grading interactions."""
    from medai_science.sources.openfda_labels import _mentions, _normalized

    names = openfda.get("drug_queries", [])
    identities = openfda.get("ingredient_resolutions", [])
    if len(names) != 2 or len(identities) != 2:
        return {"status": "identity_unresolved", "shared_spl_records": [], "research_only": True}
    daily_labels = dailymed.get("labels", {})
    open_labels = openfda.get("labels", {})
    comparisons = []
    historical_diffs = []
    source_signals = {}
    for index, name in enumerate(names):
        target_names = [names[1 - index], *identities[1 - index].get("ingredients", [{}])[0].get("names", [])]
        of_rows = open_labels.get(name, []) if isinstance(open_labels, dict) else []
        dm_rows = daily_labels.get(name, []) if isinstance(daily_labels, dict) else []
        for current in dm_rows:
            if not isinstance(current, dict):
                continue
            history = current.get("historical_versions", {})
            history_rows = history.get("records", []) if isinstance(history, dict) else []
            for previous in history_rows[:_MAX_HISTORICAL_VERSIONS] if isinstance(history_rows, list) else []:
                if not isinstance(previous, dict):
                    continue
                historical_diffs.append({
                    "drug_query": name,
                    "set_id": current.get("set_id"),
                    "current_version": current.get("spl_version"),
                    "current_published_date": current.get("published_date"),
                    "historical_version": previous.get("spl_version"),
                    "historical_published_date": previous.get("published_date"),
                    "interaction_section_hash_match": bool(current.get("section_sha256")) and current.get("section_sha256") == previous.get("section_sha256"),
                    "current_section_sha256": current.get("section_sha256"),
                    "historical_section_sha256": previous.get("section_sha256"),
                    "history_record_sha256": previous.get("history_record_sha256"),
                    "archive_sha256": previous.get("spl_sha256"),
                    "interpretation": "A section hash difference identifies a versioned source change for review; it does not establish clinical significance or interaction evidence.",
                })
        of_by_id = {row.get("set_id"): row for row in of_rows if isinstance(row, dict) and row.get("set_id")}
        dm_by_id = {row.get("set_id"): row for row in dm_rows if isinstance(row, dict) and row.get("set_id")}
        for set_id in sorted(set(of_by_id) & set(dm_by_id)):
            left, right = of_by_id[set_id], dm_by_id[set_id]
            left_text = _normalized(str(left.get("drug_interactions", "")))
            right_text = _normalized(str(right.get("drug_interactions", "")))
            comparisons.append({
                "drug_query": name,
                "set_id": set_id,
                "openfda_version": left.get("version"),
                "dailymed_version": right.get("spl_version"),
                "version_match": str(left.get("version")) == str(right.get("spl_version")),
                "normalized_section_text_match": bool(left_text) and left_text == right_text,
            })
        source_signals[name] = {
            "openfda_exact_name_mention": any(_mentions(str(row.get("drug_interactions", "")), term) for row in of_rows if isinstance(row, dict) for term in target_names),
            "dailymed_exact_name_mention": any(_mentions(str(row.get("drug_interactions", "")), term) for row in dm_rows if isinstance(row, dict) for term in target_names),
            "labels_returned": {"openfda": len(of_rows), "dailymed": len(dm_rows)},
        }
        # Keep a binding to the resolved query identity in the output, not only display labels.
        source_signals[name]["rxnorm_ingredient_rxcui"] = identities[index].get("ingredients", [{}])[0].get("rxcui")
    return {
        "status": "shared_label_distributions_compared" if comparisons else "no_shared_spl_ids_for_direct_comparison",
        "shared_spl_records": comparisons,
        "historical_version_diffs": historical_diffs[:_MAX_LABELS * 2],
        "exact_name_mention_signals": source_signals,
        "interpretation": "This compares two distributions of FDA-originated labeling. A mismatch is a source-version or formatting review lead, not a clinical contradiction; matching labels are not independent corroboration. Exact-name mentions and missing sections do not establish presence or absence of a drug interaction.",
        "independent_clinical_evidence": False,
        "research_only": True,
        "clinical_advice": False,
    }
