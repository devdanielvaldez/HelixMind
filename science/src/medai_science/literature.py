"""Source-label-preserving PubMed abstract structuring without claim inference."""
from __future__ import annotations

import hashlib
import re
import json
from typing import Any


_SECTION_LABELS = {
    "aim": "objective",
    "aims": "objective",
    "objective": "objective",
    "objectives": "objective",
    "purpose": "objective",
    "background": "background",
    "introduction": "background",
    "methods": "methods",
    "method": "methods",
    "materials and methods": "methods",
    "patients and methods": "methods",
    "design": "methods",
    "results": "results",
    "findings": "results",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "interpretation": "conclusion",
    "importance": "background",
}


def omit_pubmed_abstract_text(paper: dict[str, Any]) -> dict[str, Any]:
    """Return a bibliographic record with abstract text replaced by bounded digests."""
    if not isinstance(paper, dict) or not isinstance(paper.get("provenance"), dict) or paper["provenance"].get("source") != "PubMed":
        raise ValueError("abstract retention policy requires a PubMed paper record")
    raw_sections = paper.get("abstract_sections")
    section_metadata = []
    if isinstance(raw_sections, list):
        for item in raw_sections[:100]:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            text = item["text"].strip()
            if not text:
                continue
            label = item.get("source_label")
            section_metadata.append({
                "source_label": label[:128] if isinstance(label, str) else None,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text_length": len(text),
            })
    abstract = paper.get("abstract") if isinstance(paper.get("abstract"), str) else ""
    if not abstract and section_metadata and isinstance(raw_sections, list):
        abstract = " ".join(item["text"].strip() for item in raw_sections if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip())
    clean = {key: value for key, value in paper.items() if key not in {"abstract", "abstract_sections"}}
    clean.update({
        "abstract": "",
        "abstract_sections": [],
        "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
        "abstract_length": len(abstract),
        "abstract_section_metadata": section_metadata,
        "abstract_text_retained": False,
    })
    return clean


def structure_pubmed_abstract(paper: dict[str, Any], source_artifact_id: str, source_artifact_sha256: str) -> dict[str, Any]:
    """Preserve provider section labels and text digests without retaining abstract text."""
    provenance = paper.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "PubMed":
        raise ValueError("paper artifact is not a PubMed record")
    pmid = provenance.get("source_id")
    if not isinstance(pmid, str) or not pmid.isdigit():
        raise ValueError("PubMed paper artifact is missing its PMID")
    raw_sections = paper.get("abstract_sections")
    sections: list[dict[str, Any]] = []
    section_metadata = paper.get("abstract_section_metadata")
    if isinstance(section_metadata, list):
        for item in section_metadata[:100]:
            if not isinstance(item, dict):
                continue
            label = item.get("source_label")
            normalized = _SECTION_LABELS.get(label.strip().casefold(), "unclassified") if isinstance(label, str) else "unclassified"
            digest = item.get("text_sha256")
            length = item.get("text_length")
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) and isinstance(length, int) and not isinstance(length, bool) and 1 <= length <= 100_000:
                sections.append({"source_label": label if isinstance(label, str) else "", "normalized_section": normalized, "text_sha256": digest, "text_length": length, "classification_source": "PubMed AbstractText Label/NlmCategory" if label else "no provider section label; left unclassified"})
    elif isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                continue
            label = item.get("source_label")
            normalized = _SECTION_LABELS.get(label.strip().casefold(), "unclassified") if isinstance(label, str) else "unclassified"
            text = item["text"].strip()
            sections.append({
                "source_label": label if isinstance(label, str) else "",
                "normalized_section": normalized,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "text_length": len(text),
                "classification_source": "PubMed AbstractText Label/NlmCategory" if label else "no provider section label; left unclassified",
            })
    if not sections and not isinstance(section_metadata, list):
        abstract = paper.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            text = abstract.strip()
            sections.append({"source_label": "", "normalized_section": "unclassified", "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "text_length": len(text), "classification_source": "PubMed abstract without retained section labels"})
    identifiers = paper.get("identifiers")
    if not isinstance(identifiers, dict):
        identifiers = {}
    normalized_ids = {key: value for key, value in identifiers.items() if key in {"doi", "pmc"} and isinstance(value, str) and value.strip()}
    return {
        "source": "PubMed abstract section structure",
        "pmid": pmid,
        "title": provenance.get("title", ""),
        "publication_metadata": {
            "publication_types": paper.get("publication_types", []),
            "publication_type_ids": paper.get("publication_type_ids", []),
            "mesh_headings": paper.get("mesh_headings", []),
            "journal": paper.get("journal"),
            "journal_abbreviation": paper.get("journal_abbreviation"),
            "journal_issns": paper.get("journal_issns", []),
            "volume": paper.get("volume"),
            "issue": paper.get("issue"),
            "pages": paper.get("pages"),
            "publication_date": paper.get("publication_date"),
            "language": paper.get("language"),
        },
        "identifiers": normalized_ids,
        "source_artifact_id": source_artifact_id,
        "source_artifact_sha256": source_artifact_sha256,
        "abstract_sha256": paper.get("abstract_sha256") if isinstance(paper.get("abstract_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", paper["abstract_sha256"]) else None,
        "abstract_length": paper.get("abstract_length") if isinstance(paper.get("abstract_length"), int) and not isinstance(paper.get("abstract_length"), bool) and 0 <= paper["abstract_length"] <= 1_000_000 else None,
        "sections": sections,
        "claims": [],
        "entities": [],
        "limitations": [
            "Only PubMed abstract section labels are structured; this is not full-text analysis.",
            "Unlabeled passages remain unclassified; no claim polarity, entity identity, support, or contradiction is inferred.",
        ],
    }


def structure_pmc_receipt(parsed: dict[str, Any], *, pmid: str, pmc_id: str, source_artifact_id: str, source_artifact_sha256: str, extraction_artifact_id: str | None) -> dict[str, Any]:
    """Retain structural PMC extraction metadata without copying article text."""
    if not pmid.isdigit() or not pmc_id.upper().startswith("PMC") or not pmc_id[3:].isdigit():
        raise ValueError("PMC receipt requires numeric PMID and PMC identifiers")
    sections = parsed.get("sections", [])
    entities = parsed.get("normalized_entities", [])
    candidates = parsed.get("result_sentence_candidates", [])
    return {
        "source": "PMC BioC structural extraction receipt",
        "pmid": pmid,
        "pmc_id": pmc_id.upper(),
        "source_artifact_id": source_artifact_id,
        "source_artifact_sha256": source_artifact_sha256,
        "paper_extraction_artifact_id": extraction_artifact_id,
        "source_sha256": parsed.get("source_sha256"),
        "source_license": parsed.get("source_license"),
        "sections": [{key: row.get(key) for key in ("passage_index", "section", "source_section", "offset")} for row in sections if isinstance(row, dict)],
        "normalized_entities": [{key: entity.get(key) for key in ("identifier", "entity_type", "database")} for entity in entities if isinstance(entity, dict)],
        "result_sentence_candidate_count": len(candidates),
        "limitations": parsed.get("extraction_limitations", []),
        "retention": "Article text and sentence candidates are not copied into this artifact.",
    }


def canonical_source_entity_id(entity: dict[str, Any]) -> str | None:
    """Return a namespace-qualified ID only for an explicit source namespace."""
    identifier = entity.get("identifier")
    entity_type = entity.get("entity_type")
    if not isinstance(identifier, str) or not identifier.strip() or not isinstance(entity_type, str) or not entity_type.strip():
        return None
    namespace = entity.get("database")
    if not isinstance(namespace, str) or not namespace.strip():
        return None
    identifier, namespace, entity_type = identifier.strip(), namespace.strip(), entity_type.strip()
    source_identifier = identifier
    node_namespace = re.sub(r"[^A-Za-z0-9._-]+", "_", namespace)
    if entity_type.casefold() == "gene" and namespace.casefold() in {"ncbi gene", "ncbi_gene", "ncbigene"}:
        match = re.fullmatch(r"(?:NCBI(?:Gene)?[:_])?([0-9]+)", identifier, flags=re.IGNORECASE)
        if match:
            node_namespace, identifier = "NCBIGene", match.group(1)
    elif entity_type.casefold() == "protein" and namespace.casefold() in {"uniprot", "uniprotkb"}:
        # BioC sources commonly label UniProtKB accessions as "UniProt";
        # converge that alias on the same stable namespace used by UniProt tools.
        if not re.fullmatch(
            r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?",
            identifier,
            flags=re.IGNORECASE,
        ):
            return None
        node_namespace, identifier = "UniProtKB", identifier.upper()
    return f"{node_namespace}:{identifier}"


def source_entity_node(entity: dict[str, Any], *, receipt_id: str, receipt_sha256: str, pmc_id: str) -> dict[str, Any] | None:
    """Map a provider-normalized stable identifier to a provenance-linked graph node."""
    node_id = canonical_source_entity_id(entity)
    if node_id is None:
        return None
    identifier = entity["identifier"].strip()
    source_identifier = identifier
    namespace = entity["database"].strip()
    entity_type = entity["entity_type"].strip()
    return {
        "id": node_id,
        "kind": entity_type,
        "label": source_identifier,
        "metadata": {"source": "PMC BioC annotation", "source_artifact_id": receipt_id, "source_artifact_sha256": receipt_sha256, "database": namespace, "identifier": source_identifier, "pmc_id": pmc_id},
    }


def pubchem_cid_candidates_from_extraction_receipts(artifacts: Any, limit: int = 20, exclude: set[str] | None = None) -> tuple[list[dict[str, str]], bool]:
    """Extract provider CIDs and report whether the unique candidate set exceeded the limit."""
    if not isinstance(artifacts, list) or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        return [], False
    found: dict[str, dict[str, str]] = {}
    truncated = False
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "paper_extraction_receipt":
            continue
        receipt_id = artifact.get("id")
        if not isinstance(receipt_id, str) or not receipt_id:
            continue
        payload = artifact.get("payload")
        try:
            payload = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            continue
        entities = payload.get("normalized_entities") if isinstance(payload, dict) else None
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            database = entity.get("database")
            entity_type = entity.get("entity_type")
            identifier = entity.get("identifier")
            if not isinstance(database, str) or database.strip().casefold() not in {"pubchem", "pubchem compound", "pubchem compound database"}:
                continue
            if not isinstance(entity_type, str) or entity_type.strip().casefold() not in {"chemical", "drug", "compound"}:
                continue
            if not isinstance(identifier, str):
                continue
            match = re.fullmatch(r"(?:CID[:_])?([1-9][0-9]{0,11})", identifier.strip(), flags=re.IGNORECASE)
            if match:
                cid = match.group(1)
                if exclude and cid in exclude:
                    continue
                found.setdefault(cid, {"cid": cid, "source_artifact_id": receipt_id})
                if len(found) > limit:
                    truncated = True
                    return list(found.values())[:limit], truncated
    return list(found.values()), truncated


def pubchem_cids_from_extraction_receipts(artifacts: Any, limit: int = 20, exclude: set[str] | None = None) -> list[dict[str, str]]:
    """Extract only provider-identified PubChem compound CIDs from retained PMC receipts."""
    return pubchem_cid_candidates_from_extraction_receipts(artifacts, limit, exclude)[0]


def exact_mygene_identifier_matches(results: Any, entrezgene_id: str) -> list[dict[str, Any]]:
    """Keep only MyGene records whose returned stable Entrez ID exactly matches."""
    if not entrezgene_id.isdigit() or not isinstance(results, list):
        return []
    matches = []
    for item in results:
        if not isinstance(item, dict):
            continue
        record = item.get("record")
        if not isinstance(record, dict) or str(record.get("entrezgene")) != entrezgene_id:
            continue
        matches.append({"source_id": item.get("source_id"), "record": record})
    return matches


def resolved_gene_symbols(artifacts: Any, limit: int) -> list[dict[str, str]]:
    """Read symbols only from source artifacts containing exact Entrez ID matches."""
    if not isinstance(artifacts, list) or not 1 <= limit <= 20:
        return []
    selected: dict[str, dict[str, str]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("source") != "MyGene.info":
            continue
        payload = artifact.get("payload")
        try:
            payload = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("source") != "MyGene.info exact NCBI Gene identifier resolution":
            continue
        gene_id = payload.get("ncbi_gene_id")
        if not isinstance(gene_id, str) or not gene_id.isdigit():
            continue
        for match in payload.get("matches", []):
            record = match.get("record") if isinstance(match, dict) else None
            symbol = record.get("symbol") if isinstance(record, dict) else None
            if not isinstance(record, dict) or str(record.get("entrezgene")) != gene_id or not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", symbol):
                continue
            selected.setdefault(gene_id, {"entrezgene": gene_id, "symbol": symbol, "source_artifact_id": str(artifact.get("id", "")), "source_artifact_sha256": str(artifact.get("sha256", ""))})
            if len(selected) >= limit:
                return [selected[key] for key in sorted(selected)]
    return [selected[key] for key in sorted(selected)]


def exact_uniprot_gene_matches(results: Any, symbol: str) -> list[dict[str, Any]]:
    """Select reviewed human UniProt search records carrying the exact gene symbol."""
    if not isinstance(results, list) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", symbol):
        return []
    matches = []
    seen_accessions: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        record = item.get("record")
        if not isinstance(record, dict):
            continue
        accession = record.get("primaryAccession") or item.get("source_id")
        organism = record.get("organism")
        taxonomy = organism.get("taxonId") if isinstance(organism, dict) else None
        if taxonomy is None:
            taxonomy = item.get("organism_id")
        genes = record.get("genes", [])
        names = [gene.get("geneName", {}).get("value") for gene in genes if isinstance(gene, dict) and isinstance(gene.get("geneName"), dict)] if isinstance(genes, list) else []
        projected_names = record.get("gene_names")
        if isinstance(projected_names, str):
            names.extend(projected_names.split())
        elif isinstance(projected_names, list):
            names.extend(name for name in projected_names if isinstance(name, str))
        if taxonomy == 9606 and isinstance(accession, str) and re.fullmatch(r"[A-Z0-9]{6,10}(?:-[0-9]+)?", accession) and accession not in seen_accessions and any(isinstance(name, str) and name.casefold() == symbol.casefold() for name in names):
            matches.append({"accession": accession, "record": record, "source_id": item.get("source_id")})
            seen_accessions.add(accession)
    return matches
