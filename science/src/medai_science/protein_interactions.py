"""Deterministic, non-probabilistic reconciliation of protein interaction records."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from typing import Any

_STABLE_ID_ORDER = ("uniprotkb", "ensembl", "ensemblgenomes", "refseq", "refseq_protein")
_UNIPROT_ACCESSION = re.compile(r"^[A-Z0-9]{6,10}(?:-[1-9][0-9]*)?$", re.IGNORECASE)
_INTACT_ID = re.compile(r"^intact:(EBI-\d+)$", re.IGNORECASE)


def _identifiers(value: str) -> list[tuple[str, str]]:
    result = []
    for token in value.split("|"):
        namespace, separator, identifier = token.strip().partition(":")
        if not separator or not identifier or identifier == "-":
            continue
        namespace = namespace.casefold()
        identifier = identifier.strip()
        if namespace == "uniprotkb" and _UNIPROT_ACCESSION.fullmatch(identifier):
            result.append((namespace, identifier.upper()))
        elif namespace in _STABLE_ID_ORDER[1:] and re.fullmatch(r"[A-Za-z0-9_.-]{3,80}", identifier):
            result.append((namespace, identifier.upper()))
    return result


def _primary_identifier(value: str) -> str | None:
    identifiers = _identifiers(value)
    if not identifiers:
        return None
    priority = {name: index for index, name in enumerate(_STABLE_ID_ORDER)}
    namespace, identifier = min(identifiers, key=lambda item: (priority[item[0]], item[1]))
    return f"{namespace}:{identifier}"


def _xrefs(value: str, namespace: str | None = None) -> list[tuple[str, str]]:
    result = []
    for token in value.split("|"):
        token = token.strip()
        match = re.match(r"^([^:]+):([^()]+?)(?:\(([^()]*)\))?$", token)
        if not match:
            continue
        current_namespace, identifier, label = match.groups()
        if namespace is None or current_namespace.casefold() == namespace.casefold():
            result.append((identifier.strip(), (label or "").strip()))
    return result


def _taxon_id(value: str) -> int | None:
    for token in value.split("|"):
        match = re.match(r"^taxid:(\d+)", token.strip(), re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _mi_terms(value: str) -> tuple[str, ...]:
    terms = []
    for identifier, _label in _xrefs(value):
        terms.append(f"MI:{identifier.split(':')[-1]}" if identifier.upper().startswith("MI:") else identifier)
    return tuple(sorted(set(terms)))


def _participant(value: str, taxon: int | None, biological_roles: str = "") -> tuple[str, int | None, tuple[str, ...]] | None:
    identifier = _primary_identifier(value)
    if identifier is None:
        return None
    return identifier, taxon, _mi_terms(biological_roles)


def _row_key(record: dict[str, Any]) -> tuple[tuple[str, int | None, tuple[str, ...]], tuple[str, int | None, tuple[str, ...]]] | None:
    first = _participant(str(record.get("interactor_a", "")), _taxon_id(str(record.get("organism_a", ""))), str(record.get("biological_role_a", "")))
    second = _participant(str(record.get("interactor_b", "")), _taxon_id(str(record.get("organism_b", ""))), str(record.get("biological_role_b", "")))
    if first is None or second is None:
        return None
    return tuple(sorted((first, second), key=lambda item: (item[0], item[1] or 0)))  # type: ignore[return-value]


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    pmids = sorted({identifier for identifier, _label in _xrefs(str(record.get("pubmed_ids", ""))) if identifier.isdigit()})
    methods = _xrefs(str(record.get("detection_method", "")))
    interaction_types = _xrefs(str(record.get("interaction_type", "")))
    intact_ids = sorted({identifier.upper() for identifier, _label in _xrefs(str(record.get("interaction_ids", "")), "intact") if re.fullmatch(r"EBI-\d+", identifier, re.IGNORECASE)})
    return {
        "pubmed_ids": pmids,
        "detection_methods": sorted({f"MI:{identifier.split(':')[-1]}" if identifier.upper().startswith("MI:") else identifier for identifier, _label in methods}),
        "detection_method_labels": sorted({label for _identifier, label in methods if label}),
        "interaction_types": sorted({f"MI:{identifier.split(':')[-1]}" if identifier.upper().startswith("MI:") else identifier for identifier, _label in interaction_types}),
        "interaction_type_labels": sorted({label for _identifier, label in interaction_types if label}),
        "biological_role_a": _mi_terms(str(record.get("biological_role_a", ""))),
        "biological_role_b": _mi_terms(str(record.get("biological_role_b", ""))),
        "experimental_roles": sorted(set(_mi_terms(str(record.get("experimental_role_a", ""))) + _mi_terms(str(record.get("experimental_role_b", ""))))),
        "complex_expansion": _mi_terms(str(record.get("complex_expansion", ""))),
        "intact_record_ids": intact_ids,
        "source_confidence_values": sorted(set(str(record.get("confidence", "")).split("|")) - {"", "-"}),
        "source_record_hash": hashlib.sha256(str(record.get("raw_mitab", "")).encode("utf-8")).hexdigest(),
    }


def reconcile_interactions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group records by unordered stable participant pair and taxon, preserving study-level details."""
    grouped: dict[tuple[tuple[str, int | None, tuple[str, ...]], tuple[str, int | None, tuple[str, ...]]], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    unresolved: list[dict[str, Any]] = []
    for record in records:
        key = _row_key(record)
        if key is None:
            unresolved.append({
                "reason": "no_supported_stable_participant_identifiers",
                "interactor_a": record.get("interactor_a"),
                "interactor_b": record.get("interactor_b"),
                "source_record_hash": hashlib.sha256(str(record.get("raw_mitab", "")).encode("utf-8")).hexdigest(),
            })
            continue
        grouped[key].append((record, _record_metadata(record)))

    interactions = []
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        publications: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"detection_methods": set(), "interaction_types": set(), "intact_record_ids": set(), "source_record_hashes": set()})
        all_methods: set[str] = set()
        all_types: set[str] = set()
        experimental_roles: set[str] = set()
        complex_expansions: set[str] = set()
        intact_ids: set[str] = set()
        exact_rows: Counter[str] = Counter()
        records_without_publication: list[str] = []
        records_without_intact_id = 0
        for record, metadata in rows:
            all_methods.update(metadata["detection_methods"])
            all_types.update(metadata["interaction_types"])
            experimental_roles.update(metadata["experimental_roles"])
            complex_expansions.update(metadata["complex_expansion"])
            intact_ids.update(metadata["intact_record_ids"])
            exact_rows[str(record.get("raw_mitab", ""))] += 1
            if not metadata["pubmed_ids"]:
                records_without_publication.append(metadata["source_record_hash"])
            if not metadata["intact_record_ids"]:
                records_without_intact_id += 1
            for pmid in metadata["pubmed_ids"]:
                publication = publications[pmid]
                publication["detection_methods"].update(metadata["detection_methods"])
                publication["interaction_types"].update(metadata["interaction_types"])
                publication["intact_record_ids"].update(metadata["intact_record_ids"])
                publication["source_record_hashes"].add(metadata["source_record_hash"])

        canonical = json.dumps(key, separators=(",", ":"))
        interactions.append({
            "interaction_id": "ppi-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
            "participants": [{"identifier": identifier, "taxonomy_id": taxon, "biological_roles": list(roles)} for identifier, taxon, roles in key],
            "record_count": len(rows),
            "exact_duplicate_row_count": sum(count - 1 for count in exact_rows.values()),
            "unique_intact_record_ids": sorted(intact_ids),
            "unique_intact_record_id_count": len(intact_ids),
            "unique_publications": sorted(publications),
            "unique_publication_count": len(publications),
            "records_without_pubmed_id": len(records_without_publication),
            "records_without_pubmed_source_hashes": sorted(records_without_publication),
            "records_without_intact_record_id": records_without_intact_id,
            "detection_methods": sorted(all_methods),
            "interaction_types": sorted(all_types),
            "experimental_roles": sorted(experimental_roles),
            "complex_expansions": sorted(complex_expansions),
            "annotation_heterogeneity": {
                "multiple_detection_methods": len(all_methods) > 1,
                "multiple_interaction_types": len(all_types) > 1,
                "multiple_experimental_roles": len(experimental_roles) > 1,
                "interpretation": "Different assay or interaction-type annotations are retained as heterogeneity; they are not automatically treated as contradictory results.",
            },
            "publication_evidence": [
                {
                    "pubmed_id": pmid,
                    "detection_methods": sorted(data["detection_methods"]),
                    "interaction_types": sorted(data["interaction_types"]),
                    "intact_record_ids": sorted(data["intact_record_ids"]),
                    "source_record_hashes": sorted(data["source_record_hashes"]),
                }
                for pmid, data in sorted(publications.items())
            ],
            "raw_source_confidence_values": sorted({value for _record, metadata in rows for value in metadata["source_confidence_values"]}),
            "confidence_interpretation": "Original source strings only; not calibrated probabilities and not aggregated into an interaction confidence.",
        })

    record_id_pairs: dict[str, set[str]] = defaultdict(set)
    for interaction in interactions:
        for record_id in interaction["unique_intact_record_ids"]:
            record_id_pairs[record_id].add(interaction["interaction_id"])
    source_identity_conflicts = [
        {
            "intact_record_id": record_id,
            "interaction_ids": sorted(interaction_ids),
            "interpretation": "One source record identifier resolves to multiple participant-pair groups; review source records before interpreting this as biology.",
        }
        for record_id, interaction_ids in sorted(record_id_pairs.items())
        if len(interaction_ids) > 1
    ]

    return {
        "interaction_count": len(interactions),
        "input_record_count": len(records),
        "reconciled_record_count": sum(len(rows) for rows in grouped.values()),
        "unresolved_record_count": len(unresolved),
        "unresolved_records": unresolved,
        "source_identity_conflict_count": len(source_identity_conflicts),
        "source_identity_conflicts": source_identity_conflicts,
        "source_identity_conflict_interpretation": "A repeated source identifier across different participant-pair groups is a source-data integrity flag; it is not a biological contradiction verdict.",
        "interactions": interactions,
        "method": "Deterministic entity, unordered-pair, and organism reconciliation within one IntAct retrieval; repeated IntAct identifiers assigned to different pair groups are reported as source-identity conflicts. Publications and assays are descriptive evidence units, not assumed independent replications.",
    }
