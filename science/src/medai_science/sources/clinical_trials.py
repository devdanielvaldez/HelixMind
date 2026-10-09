"""Provenance-preserving ClinicalTrials.gov v2 records; never treatment advice."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query

_NCT_ID = re.compile(r"^NCT\d{8}$")
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("ClinicalTrials.gov record contains a malformed object field")
    return value


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("ClinicalTrials.gov record contains a malformed object collection")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"ClinicalTrials.gov returned duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"ClinicalTrials.gov returned non-standard JSON numeric constant {value}")


def _load_response_json(raw: bytes, response_kind: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ClinicalTrials.gov returned invalid {response_kind} JSON") from error


def _validated_snapshot(snapshot: Any, expected_nct_id: str | None = None) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or snapshot.get("source") != "ClinicalTrials.gov":
        raise ValueError("trial snapshot has an invalid source")
    nct_id = snapshot.get("nct_id")
    record = snapshot.get("record")
    provenance = snapshot.get("provenance")
    if (not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id)
            or expected_nct_id is not None and nct_id != expected_nct_id
            or not isinstance(record, dict) or not isinstance(provenance, dict)):
        raise ValueError("trial snapshot identity or retained source record is malformed")
    retrieved_at = provenance.get("retrieved_at")
    digest = _snapshot_hash(record)
    if not isinstance(retrieved_at, str) or not retrieved_at or provenance.get("record_snapshot_sha256") != digest:
        raise ValueError("trial snapshot source-record digest or retrieval time is inconsistent")
    recomputed = normalize_study(record, retrieved_at)
    if json.dumps(recomputed, ensure_ascii=False, sort_keys=True, separators=(",", ":")) != json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")):
        raise ValueError("trial snapshot fields do not match its retained source record")
    return snapshot


def _pick(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _date(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value[key] for key in ("date", "type") if key in value}


def _snapshot_hash(study: dict[str, Any]) -> str:
    canonical = json.dumps(study, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_study(study: dict[str, Any], retrieved_at: str | None = None) -> dict[str, Any]:
    """Extract registry fields without inferring eligibility or treatment effects."""
    protocol = _mapping(study.get("protocolSection"))
    identification = _mapping(protocol.get("identificationModule"))
    nct_id = identification.get("nctId")
    if not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id):
        raise ValueError("ClinicalTrials.gov record is missing a valid NCT identifier")

    status = _mapping(protocol.get("statusModule"))
    design = _mapping(protocol.get("designModule"))
    design_info = _mapping(design.get("designInfo"))
    masking = _mapping(design_info.get("maskingInfo"))
    eligibility = _mapping(protocol.get("eligibilityModule"))
    conditions = _mapping(protocol.get("conditionsModule"))
    interventions_module = _mapping(protocol.get("armsInterventionsModule"))
    outcomes = _mapping(protocol.get("outcomesModule"))
    sponsor = _mapping(protocol.get("sponsorCollaboratorsModule"))
    sponsor_org = _mapping(sponsor.get("leadSponsor"))
    first_posted = _date(status.get("studyFirstPostDateStruct"))

    locations_module = _mapping(protocol.get("contactsLocationsModule"))
    locations = []
    for location in _rows(locations_module.get("locations")):
        locations.append({
            "facility": location.get("facility"),
            "city": location.get("city"),
            "state": location.get("state"),
            "country": location.get("country"),
            "status": location.get("status"),
        })

    primary_outcomes = _rows(outcomes.get("primaryOutcomes"))
    secondary_outcomes = _rows(outcomes.get("secondaryOutcomes"))
    other_outcomes = _rows(outcomes.get("otherOutcomes"))
    enrollment = _mapping(design.get("enrollmentInfo"))
    last_update = _date(status.get("studyLastUpdatePostDateStruct"))
    retrieved = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if retrieved.endswith("+00:00"):
        retrieved = retrieved[:-6] + "Z"

    record = {
        "source": "ClinicalTrials.gov",
        "nct_id": nct_id,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
        "identification": {
            "brief_title": identification.get("briefTitle"),
            "official_title": identification.get("officialTitle"),
            "acronym": identification.get("acronym"),
            "organization_study_id": _mapping(identification.get("orgStudyIdInfo")).get("id"),
        },
        "lifecycle": {
            "overall_status": status.get("overallStatus"),
            "why_stopped": status.get("whyStopped"),
            "start_date": _date(status.get("studyStartDateStruct")),
            "primary_completion_date": _date(status.get("primaryCompletionDateStruct")),
            "completion_date": _date(status.get("completionDateStruct")),
            "first_submitted": status.get("studyFirstSubmitDate"),
            "first_posted": first_posted,
            "last_update_submitted": status.get("studyLastUpdateSubmitDate"),
            "last_update_posted": last_update,
            "last_verified": _pick(status, "studyVerificationDate", "studyLastVerifiedDate"),
            "record_history_included": False,
            "record_history_url": f"https://clinicaltrials.gov/study/{nct_id}?tab=history",
            "record_history_access": "official_web_interface",
            "record_history_api_backfill": "unavailable_via_documented_v2_adapter",
        },
        "design": {
            "study_type": design.get("studyType"),
            "phases": design.get("phases", []),
            "allocation": design_info.get("allocation"),
            "intervention_model": design_info.get("interventionModel"),
            "primary_purpose": design_info.get("primaryPurpose"),
            "masking": masking.get("masking"),
            "who_masked": masking.get("whoMasked", []),
            "enrollment": {
                "count": _pick(enrollment, "count", "enrollmentCount"),
                "type": enrollment.get("type"),
            },
        },
        "conditions": conditions.get("conditions", []),
        "interventions": [
            {
                "type": item.get("type"),
                "name": item.get("name"),
                "description": item.get("description"),
                "arm_group_labels": item.get("armGroupLabels", []),
                "other_names": item.get("otherNames", []),
            }
            for item in _rows(interventions_module.get("interventions"))
        ],
        "eligibility": {
            "criteria_text": eligibility.get("eligibilityCriteria"),
            "healthy_volunteers": eligibility.get("healthyVolunteers"),
            "sex": eligibility.get("sex"),
            "minimum_age": eligibility.get("minimumAge"),
            "maximum_age": eligibility.get("maximumAge"),
            "standard_ages": eligibility.get("stdAges", []),
            "study_population": eligibility.get("studyPopulation"),
            "sampling_method": eligibility.get("samplingMethod"),
            "parsed_for_individual_suitability": False,
        },
        "outcomes": {
            "primary": primary_outcomes,
            "secondary": secondary_outcomes,
            "other": other_outcomes,
        },
        "sponsor": {
            "lead_name": sponsor_org.get("name"),
            "lead_class": sponsor_org.get("class"),
            "collaborators": _rows(sponsor.get("collaborators")),
        },
        "locations": locations,
        "results": {
            "has_results": study.get("hasResults"),
            "results_section_present": isinstance(study.get("resultsSection"), dict),
        },
        "provenance": {
            "provider": "ClinicalTrials.gov API v2",
            "api_endpoint": "https://clinicaltrials.gov/api/v2/studies",
            "record_url": f"https://clinicaltrials.gov/study/{nct_id}",
            "retrieved_at": retrieved,
            "record_snapshot_sha256": _snapshot_hash(study),
            "last_update_posted_date": last_update,
            "source_record_retained": True,
        },
        "record": study,
        "research_only": True,
    }
    return record


def compare_study_snapshots(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Report changed source-field paths between two retained snapshots."""
    previous = _validated_snapshot(previous)
    current = _validated_snapshot(current)
    previous_id = previous.get("nct_id")
    current_id = current.get("nct_id")
    if not previous_id or previous_id != current_id:
        raise ValueError("trial snapshots must refer to the same NCT identifier")
    previous_record = previous.get("record")
    current_record = current.get("record")
    if not isinstance(previous_record, dict) or not isinstance(current_record, dict):
        raise ValueError("both trial snapshots must retain their raw registry record")

    changed_paths: list[str] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child_path = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    changed_paths.append(child_path)
                else:
                    visit(left[key], right[key], child_path)
        elif left != right:
            changed_paths.append(path or "$")

    visit(previous_record, current_record, "")
    prior_status = _mapping(previous.get("lifecycle")).get("overall_status")
    latest_status = _mapping(current.get("lifecycle")).get("overall_status")
    return {
        "nct_id": current_id,
        "changed": bool(changed_paths),
        "changed_field_paths": changed_paths,
        "previous_snapshot_sha256": previous.get("provenance", {}).get("record_snapshot_sha256"),
        "current_snapshot_sha256": current.get("provenance", {}).get("record_snapshot_sha256"),
        "previous_status": prior_status,
        "current_status": latest_status,
        "scientific_or_safety_interpretation": "not_performed",
    }


def latest_retained_study_snapshot(artifacts: list[dict[str, Any]], nct_id: str) -> dict[str, Any] | None:
    """Return the latest same-session retained CTG record for an NCT ID."""
    if not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id):
        raise ValueError("nct_id must match NCT followed by eight digits")
    matches = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") not in {"clinical_trial_snapshot", "clinical_trial"}:
            continue
        if artifact.get("source_id") != nct_id:
            continue
        payload = artifact.get("payload")
        if (artifact.get("source") != "ClinicalTrials.gov" or not isinstance(payload, str)
                or not isinstance(artifact.get("sha256"), str)
                or hashlib.sha256(payload.encode("utf-8")).hexdigest() != artifact["sha256"]):
            raise ValueError("retained ClinicalTrials.gov artifact payload digest or source is invalid")
        try:
            decoded = json.loads(payload, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("retained ClinicalTrials.gov artifact payload is invalid JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("retained ClinicalTrials.gov artifact payload must be an object")
        _validated_snapshot(decoded, nct_id)
        recorded_at = artifact.get("recorded_at")
        artifact_id = artifact.get("id")
        if isinstance(recorded_at, bool) or not isinstance(recorded_at, int) or recorded_at < 0 or not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("retained ClinicalTrials.gov artifact ordering metadata is malformed")
        matches.append((recorded_at, artifact_id, decoded))
    return max(matches, key=lambda item: (item[0], item[1]))[2] if matches else None


class ClinicalTrialsClient:
    endpoint = "https://clinicaltrials.gov/api/v2/studies"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 20.0):
        self.transport = transport
        self.timeout = timeout

    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query = ensure_public_research_query(query)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", self.endpoint, params={"query.term": query, "pageSize": limit, "format": "json"}) as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "ClinicalTrials.gov")
        payload = _load_response_json(raw, "search")
        if not isinstance(payload, dict) or "studies" not in payload or not isinstance(payload.get("studies"), list):
            raise ValueError("ClinicalTrials.gov returned an unexpected studies response")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        return [normalize_study(study, retrieved_at) for study in _rows(payload.get("studies"))]

    async def get_study(self, nct_id: str) -> dict[str, Any]:
        if not isinstance(nct_id, str) or not _NCT_ID.fullmatch(nct_id):
            raise ValueError("nct_id must match NCT followed by eight digits")
        endpoint = f"{self.endpoint}/{nct_id}"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("GET", endpoint, params={"format": "json"}) as response:
                raw = await read_bounded_response(response, _MAX_RESPONSE_BYTES, "ClinicalTrials.gov")
        study = _load_response_json(raw, "study")
        if not isinstance(study, dict):
            raise ValueError("ClinicalTrials.gov returned an unexpected study record")
        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        normalized = normalize_study(study, retrieved_at)
        if normalized["nct_id"] != nct_id:
            raise ValueError("ClinicalTrials.gov response NCT identifier does not match the requested study")
        return normalized
