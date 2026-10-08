"""Bounded BioC parsing that keeps source annotations and section context explicit."""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET

from medai_science.sources.xml_safety import reject_xml_declarations


MAX_BIOC_BYTES = 8_000_000
MAX_PASSAGES = 30_000
MAX_ANNOTATIONS = 100_000
PARSER_VERSION = "0.2.3"
_NON_TERMINAL_ABBREVIATIONS = {
    "dr.", "prof.", "mr.", "mrs.", "ms.", "fig.", "eq.", "vs.",
    "e.g.", "i.e.", "et al.", "no.", "approx.", "ref.", "refs.",
}
_SECTION_MAP = {
    "TITLE": "title", "ABSTRACT": "abstract", "INTRO": "introduction", "INTRODUCTION": "introduction",
    "BACKGROUND": "introduction", "METHODS": "methods", "MATERIALS": "methods", "METHODS AND MATERIALS": "methods", "MATERIALS AND METHODS": "methods",
    "RESULTS": "results", "RESULT": "results", "DISCUSS": "discussion", "DISCUSSION": "discussion",
    "CONCL": "conclusions", "CONCLUSION": "conclusions", "CONCLUSIONS": "conclusions",
    "LIMITATIONS": "limitations", "STRENGTHS AND LIMITATIONS": "limitations", "SUPPLEMENTARY MATERIAL": "supplementary",
}


def _infons(element: ET.Element) -> dict[str, str]:
    return {node.attrib.get("key", "").strip().lower(): (node.text or "").strip() for node in element.findall("./infon") if node.attrib.get("key")}


def _section(infons: dict[str, str]) -> str:
    raw = infons.get("section_type") or infons.get("section") or infons.get("type") or "unclassified"
    key = re.sub(r"\s+", " ", raw.strip().upper())
    if "LIMITATION" in key:
        return "limitations"
    return _SECTION_MAP.get(key, key.lower().replace(" ", "_"))


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
    entities: dict[tuple[str, str], dict] = {}
    method_text: list[str] = []
    result_text: list[str] = []
    limitation_text: list[str] = []
    claims = []
    rejected_entity_location_count = 0
    for index, passage in enumerate(passages):
        infons = _infons(passage)
        category = _section(infons)
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
                    "source_section": infons.get("section_type") or infons.get("section"),
                    "offset": unit_offset,
                    "text": text,
                }
                if sentence_index is not None:
                    section["sentence_index"] = sentence_index
                sections.append(section)
                if category == "methods":
                    method_text.append(text)
                elif category == "results":
                    result_text.append(text)
                elif category == "limitations":
                    limitation_text.append(text)

            for annotation in annotations:
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
                    else:
                        # Without a passage/sentence base offset, the BioC location
                        # cannot be checked against the exact text slice. Preserve
                        # the stable entity ID, but do not retain an unverified span.
                        rejected_entity_location_count += 1
                        continue
                    locations.append({"offset": location_offset, "length": length, "passage_index": index, "sentence_index": sentence_index, "section": category, "text": observed})
                if identifier:
                    key = (identifier, entity_type)
                    entity = entities.setdefault(key, {"identifier": identifier, "entity_type": entity_type, "database": database, "mentions": []})
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
        "methods": "\n".join(method_text),
        "results": "\n".join(result_text),
        "limitations": "\n".join(limitation_text),
        "normalized_entities": sorted(entities.values(), key=lambda item: (item["entity_type"], item["identifier"])),
        "rejected_entity_location_count": rejected_entity_location_count,
        "result_sentence_candidates": claims,
        "source_license": license_name,
        "reuse_notice": "BioC is provided for the PMC Open Access subset and author manuscript collection; check the article-specific license before downstream retention or reuse.",
        "extraction_limitations": ["Section labels depend on source annotations and may be absent or coarse.", "Sentence candidates are not validated claims and have no inferred polarity.", "Only entity identifiers and offsets supplied by BioC are normalized; no entity prediction is performed."],
    }
