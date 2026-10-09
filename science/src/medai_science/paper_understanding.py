"""Bounded BioC parsing that keeps source annotations and section context explicit."""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy
import xml.etree.ElementTree as ET

from medai_science.sources.xml_safety import reject_xml_declarations


MAX_BIOC_BYTES = 8_000_000
MAX_PASSAGES = 30_000
MAX_ANNOTATIONS = 100_000
PARSER_VERSION = "0.2.11"
_FULL_TEXT_FIELDS = {
    "sections", "section_text", "objectives", "population_sections", "dataset_sections",
    "discussion", "conclusions", "methods", "results", "limitations", "abstract",
    "result_sentence_candidates",
}
_NON_TERMINAL_ABBREVIATIONS = {
    "dr.", "prof.", "mr.", "mrs.", "ms.", "fig.", "eq.", "vs.",
    "e.g.", "i.e.", "et al.", "no.", "approx.", "ref.", "refs.",
}
_SECTION_MAP = {
    "TITLE": "title", "ABSTRACT": "abstract", "INTRO": "introduction", "INTRODUCTION": "introduction",
    "BACKGROUND": "introduction", "METHOD": "methods", "METHODS": "methods", "METHODOLOGY": "methods", "MATERIAL": "methods", "MATERIALS": "methods",
    "METHODS AND MATERIALS": "methods", "MATERIALS AND METHODS": "methods", "MATERIALS & METHODS": "methods",
    "METHODS AND MATERIAL": "methods", "PATIENTS AND METHODS": "methods", "STUDY METHODS": "methods",
    "STATISTICAL ANALYSIS": "methods", "STATISTICAL ANALYSES": "methods", "STATISTICAL METHODS": "methods",
    "STUDY DESIGN AND PARTICIPANTS": "methods", "PARTICIPANTS AND STUDY DESIGN": "methods",
    "STUDY DESIGN AND SETTING": "methods", "SAMPLE COLLECTION": "methods",
    "EXPERIMENTAL METHOD": "methods", "EXPERIMENTAL METHODS": "methods",
    "RESULTS": "results", "RESULT": "results", "DISCUSS": "discussion", "DISCUSSION": "discussion",
    "CONCL": "conclusions", "CONCLUSION": "conclusions", "CONCLUSIONS": "conclusions",
    "LIMITATION": "limitations", "LIMITATIONS": "limitations", "STRENGTHS AND LIMITATIONS": "limitations",
    "SUPPLEMENTARY MATERIAL": "supplementary", "SUPPLEMENTARY INFORMATION": "supplementary",
    "SUPPLEMENTAL MATERIAL": "supplementary", "ACKNOWLEDGEMENTS": "acknowledgements",
    "ACKNOWLEDGMENTS": "acknowledgements", "REFERENCES": "references",
    "FINDING": "results", "FINDINGS": "results", "OUTCOME": "results", "OUTCOMES": "results", "EXPERIMENTAL PROCEDURE": "methods", "EXPERIMENTAL PROCEDURES": "methods",
    "EXPERIMENTAL DESIGN": "methods", "STUDY DESIGN": "methods",
    "AIM": "objective", "AIMS": "objective", "OBJECTIVE": "objective", "OBJECTIVES": "objective",
    "PURPOSE": "objective", "STUDY OBJECTIVE": "objective", "STUDY OBJECTIVES": "objective",
    "PATIENTS": "population", "PARTICIPANTS": "population", "STUDY PARTICIPANTS": "population",
    "STUDY SUBJECTS": "population", "PATIENT POPULATION": "population", "STUDY POPULATION": "population",
    "POPULATION": "population", "COHORT": "population", "STUDY COHORT": "population",
    "DATA": "dataset", "DATASET": "dataset", "DATASETS": "dataset", "DATA SOURCE": "dataset",
    "DATA SOURCES": "dataset", "DATA COLLECTION": "dataset", "DATA AVAILABILITY": "dataset",
}


def _infons(element: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in element.findall("./infon"):
        key = node.attrib.get("key", "").strip().lower()
        if not key:
            raise ValueError("BioC infon is missing its key")
        if key in values:
            raise ValueError(f"BioC element contains duplicate infon key {key!r}")
        values[key] = (node.text or "").strip()
    return values


def _section(infons: dict[str, str]) -> tuple[str, str | None]:
    """Map BioC section metadata conservatively, preferring a recognized label.

    BioC exporters sometimes set a coarse `section_type` such as BODY while
    retaining the actual heading in `section` or `title`. Keep the source label
    and choose the first recognized category; unknown labels remain explicit.
    """
    labels = [
        infons.get("section_type"),
        infons.get("section"),
        infons.get("title"),
        infons.get("type"),
    ]
    normalized: list[tuple[str, str]] = []
    for raw in labels:
        if not isinstance(raw, str) or not raw.strip():
            continue
        key = re.sub(r"\s+", " ", raw.strip().upper())
        key = re.sub(r"^(?:\d+(?:\.\d+)*[.)]?\s+)", "", key)
        normalized.append((raw, key))
    for raw, key in normalized:
        if "LIMITATION" in key:
            return "limitations", raw
        if key in _SECTION_MAP:
            return _SECTION_MAP[key], raw
    if normalized:
        # A specific free-text heading is more informative than a coarse BODY
        # category when neither label maps to one of the standard sections.
        raw, key = normalized[1] if len(normalized) > 1 else normalized[0]
        return key.lower().replace(" ", "_"), raw
    return "unclassified", None


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    """Return verbatim sentence candidates and local character offsets."""
    spans: list[tuple[int, str]] = []
    start = 0
    for separator in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text):
        end = separator.start()
        left = text[start:end].rstrip()
        token = re.search(r"(?:^|\s)([^\s]+)$", left)
        last_token = token.group(1).casefold() if token else left.casefold()
        # Biomedical prose commonly uses these periods inside names/citations.
        # A one-letter initial is also not a sentence boundary ("A. Smith").
        if (
            last_token in _NON_TERMINAL_ABBREVIATIONS
            or any(left.casefold().endswith(abbreviation) for abbreviation in _NON_TERMINAL_ABBREVIATIONS)
            or re.fullmatch("[a-z]\\.", last_token)
        ):
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 30:
            spans.append((start, text[start:end]))
        start = separator.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start >= 30:
        spans.append((start, text[start:end]))
    return spans


def parse_bioc_xml(payload: bytes | str, *, source_url: str | None = None) -> dict:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not raw or len(raw) > MAX_BIOC_BYTES:
        raise ValueError(f"BioC document must be 1..{MAX_BIOC_BYTES} bytes")
    reject_xml_declarations(raw, "BioC")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError("malformed BioC XML") from error
    documents = root.findall("./document") if root.tag == "collection" else [root] if root.tag == "document" else []
    if len(documents) != 1:
        raise ValueError("expected a BioC collection containing exactly one document")
    document = documents[0]
    doc_infons = _infons(document)
    passages = document.findall("./passage")
    if not passages or len(passages) > MAX_PASSAGES:
        raise ValueError(f"BioC document must contain 1..{MAX_PASSAGES} passages")
    annotation_count = sum(
        len(passage.findall("./annotation"))
        + sum(len(sentence.findall("./annotation")) for sentence in passage.findall("./sentence"))
        for passage in passages
    )
    if annotation_count > MAX_ANNOTATIONS:
        raise ValueError(f"BioC annotations exceed limit {MAX_ANNOTATIONS}")

    sections: list[dict] = []
    entities: dict[tuple[str, str, str, str], dict] = {}
    method_text: list[str] = []
    result_text: list[str] = []
    limitation_text: list[str] = []
    grouped_section_text: dict[str, list[str]] = {}
    claims = []
    rejected_entity_location_count = 0
    for index, passage in enumerate(passages):
        infons = _infons(passage)
        category, source_section = _section(infons)
        offset_node = passage.findtext("./offset")
        try:
            passage_offset = int(offset_node) if offset_node is not None else None
        except ValueError:
            passage_offset = None

        sentence_nodes = passage.findall("./sentence")
        units = []
        if sentence_nodes:
            for sentence_index, sentence_node in enumerate(sentence_nodes):
                text = sentence_node.findtext("./text") or ""
                offset_value = sentence_node.findtext("./offset")
                try:
                    offset = int(offset_value) if offset_value is not None else None
                except ValueError:
                    offset = None
                units.append((sentence_index, text, offset, sentence_node.findall("./annotation"), True))
        else:
            text_node = passage.find("./text")
            text = text_node.text or "" if text_node is not None else ""
            units.append((None, text, passage_offset, passage.findall("./annotation"), False))

        for sentence_index, text, unit_offset, annotations, source_sentence in units:
            if text:
                section = {
                    "passage_index": index,
                    "section": category,
                    "source_section": source_section,
                    "offset": unit_offset,
                    "text": text,
                }
                if sentence_index is not None:
                    section["sentence_index"] = sentence_index
                sections.append(section)
                grouped_section_text.setdefault(category, []).append(text)
                if category == "methods":
                    method_text.append(text)
                elif category == "results":
                    result_text.append(text)
                elif category == "limitations":
                    limitation_text.append(text)

            for annotation_index, annotation in enumerate(annotations):
                ann_infons = _infons(annotation)
                entity_type = ann_infons.get("type") or ann_infons.get("entity") or "untyped"
                identifier = ann_infons.get("identifier") or ann_infons.get("id")
                database = ann_infons.get("database") or ann_infons.get("source")
                mention = (annotation.findtext("./text") or "").strip()
                locations = []
                for location in annotation.findall("./location"):
                    try:
                        location_offset, length = int(location.attrib["offset"]), int(location.attrib["length"])
                    except (KeyError, ValueError):
                        rejected_entity_location_count += 1
                        continue
                    if location_offset < 0 or length <= 0 or length > 10_000:
                        rejected_entity_location_count += 1
                        continue
                    if unit_offset is not None:
                        local_start = location_offset - unit_offset
                        if local_start < 0 or local_start + length > len(text):
                            rejected_entity_location_count += 1
                            continue
                        observed = text[local_start:local_start + length]
                        if not observed:
                            rejected_entity_location_count += 1
                            continue
                        # For a single source span, BioC's annotation text must
                        # match that slice. Multi-location annotations can encode
                        # discontinuous mentions, so validate their bounds without
                        # comparing each fragment to the combined mention string.
                        if len(annotation.findall("./location")) == 1 and mention and observed != mention:
                            rejected_entity_location_count += 1
                            continue
                    else:
                        # Without a passage/sentence base offset, the BioC location
                        # cannot be checked against the exact text slice. Preserve
                        # the stable entity ID, but do not retain an unverified span.
                        rejected_entity_location_count += 1
                        continue
                    locations.append({"offset": location_offset, "length": length, "passage_index": index, "sentence_index": sentence_index, "section": category, "text": observed})
                if identifier:
                    # Database-local identifiers are not globally unique. Keep
                    # otherwise identical IDs from different BioC namespaces
                    # separate while tolerating harmless source-label casing
                    # and whitespace differences.
                    database_key = re.sub(r"\s+", " ", database.strip()).casefold() if database else ""
                    # Without a source namespace, a database-local identifier cannot
                    # safely be coalesced with another annotation's identifier.
                    occurrence_key = f"{index}:{sentence_index}:{annotation_index}" if not database_key else ""
                    key = (database_key, occurrence_key, identifier, entity_type)
                    entity = entities.setdefault(key, {
                        "identifier": identifier,
                        "entity_type": entity_type,
                        "database": database,
                        "identity_status": "source_normalized" if database_key else "unresolved_namespace",
                        "mentions": [],
                    })
                    for location in locations:
                        if location not in entity["mentions"]:
                            entity["mentions"].append(location)
                    if not locations and mention and not annotation.findall("./location"):
                        entity["mentions"].append({"offset": None, "length": None, "passage_index": index, "sentence_index": sentence_index, "section": category, "text": mention})

            if category == "results":
                spans = [(0, text)] if source_sentence else _sentence_spans(text)
                for local_offset, sentence_text in spans:
                    source_offset = unit_offset + local_offset if unit_offset is not None else None
                    claims.append({
                        "text": sentence_text,
                        "section": "results",
                        "passage_index": index,
                        "sentence_index": sentence_index,
                        "polarity": "unclassified",
                        "source_offset": source_offset,
                        "source_local_offset": local_offset,
                        "source_length": len(sentence_text),
                        "interpretation": "verbatim result-section sentence candidate; not independently validated",
                    })

    title = next((s["text"] for s in sections if s["section"] == "title"), "")
    abstract = "\n".join(s["text"] for s in sections if s["section"] == "abstract")
    license_name = doc_infons.get("license") or doc_infons.get("license_url") or None
    for candidate in claims:
        identity = f"{document.findtext('./id') or ''}\0{candidate['source_offset']}\0{candidate['passage_index']}\0{candidate['sentence_index']}\0{candidate['source_local_offset']}\0{candidate['text']}"
        candidate["candidate_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return {
        "parser_version": PARSER_VERSION,
        "source": "PMC BioC",
        "document_id": document.findtext("./id"),
        "source_url": source_url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "title": title,
        "abstract": abstract,
        "sections": sections,
        # Verbatim, source-labeled passages; section labels and content are
        # retained as supplied and are not independently checked.
        "section_text": {key: "\n".join(values) for key, values in sorted(grouped_section_text.items())},
        "objectives": "\n".join(grouped_section_text.get("objective", [])),
        "population_sections": "\n".join(grouped_section_text.get("population", [])),
        "dataset_sections": "\n".join(grouped_section_text.get("dataset", [])),
        "discussion": "\n".join(grouped_section_text.get("discussion", [])),
        "conclusions": "\n".join(grouped_section_text.get("conclusions", [])),
        "methods": "\n".join(method_text),
        "results": "\n".join(result_text),
        "limitations": "\n".join(limitation_text),
        "normalized_entities": sorted(entities.values(), key=lambda item: (item["entity_type"], item["identifier"], (item["database"] or "").casefold())),
        "rejected_entity_location_count": rejected_entity_location_count,
        "result_sentence_candidates": claims,
        "source_license": license_name,
        "reuse_notice": "BioC is provided for the PMC Open Access subset and author manuscript collection; check the article-specific license before downstream retention or reuse.",
        "extraction_limitations": ["Section labels depend on source annotations and may be absent or coarse.", "Grouped section text is verbatim source content and does not verify the source heading or scientific accuracy.", "Sentence candidates are not validated claims and have no inferred polarity.", "Only entity identifiers and offsets supplied by BioC are normalized; no entity prediction is performed."],
    }


def build_extraction_retention_receipt(parsed: dict) -> dict:
    """Strip full-text passages before persistence while keeping source anchors and digests."""
    if not isinstance(parsed, dict) or parsed.get("source") != "PMC BioC":
        raise ValueError("PMC retention receipt requires a parsed BioC document")
    sections = parsed.get("sections")
    candidates = parsed.get("result_sentence_candidates")
    entities = parsed.get("normalized_entities")
    if not isinstance(sections, list) or not isinstance(candidates, list) or not isinstance(entities, list):
        raise ValueError("PMC parsed record has malformed text-bearing fields")
    retained = {key: deepcopy(value) for key, value in parsed.items() if key not in _FULL_TEXT_FIELDS}
    retained["sections"] = []
    for row in sections:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("PMC parsed section is malformed")
        retained["sections"].append({
            key: row[key] for key in ("passage_index", "sentence_index", "section", "source_section", "offset")
            if key in row
        } | {"text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(), "text_length": len(row["text"])})
    retained["result_sentence_candidates"] = []
    for row in candidates:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError("PMC result candidate is malformed")
        retained["result_sentence_candidates"].append({
            key: row[key] for key in (
                "candidate_id", "section", "passage_index", "sentence_index", "polarity",
                "source_offset", "source_local_offset", "source_length",
            ) if key in row
        } | {"text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()})
    retained["normalized_entities"] = deepcopy(entities)
    for entity in retained["normalized_entities"]:
        if not isinstance(entity, dict) or not isinstance(entity.get("mentions", []), list):
            raise ValueError("PMC normalized entity is malformed")
        for mention in entity.get("mentions", []):
            if not isinstance(mention, dict):
                raise ValueError("PMC normalized entity mention is malformed")
            mention.pop("text", None)
    return retained
