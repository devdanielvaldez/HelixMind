"""Bounded ClinVar E-utilities search with source-record provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import xml.etree.ElementTree as ET
from medai_science.sources.xml_safety import reject_xml_declarations

import httpx

from medai_science.sources.bounded_http import read_bounded_response
from medai_science.sources.safety import ensure_public_research_query


class ClinVarClient:
    endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    max_response_bytes = 2_000_000

    def __init__(self, client_factory=None, api_key: str | None = None, email: str | None = None):
        self.client_factory = client_factory or httpx.AsyncClient
        self.api_key = api_key
        self.email = email

    async def search(self, query: str, limit: int = 10) -> dict:
        query = ensure_public_research_query(query)
        if len(query) > 256:
            raise ValueError("ClinVar query cannot exceed 256 characters")
        if not 1 <= limit <= 20:
            raise ValueError("ClinVar result limit must be 1..20")
        params = {"db": "clinvar", "term": query, "retmax": limit, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        async with self.client_factory(timeout=20, follow_redirects=False) as client:
            async with client.stream("GET", f"{self.endpoint}/esearch.fcgi", params=params) as response:
                search_raw = await read_bounded_response(response, self.max_response_bytes, "ClinVar ESearch")
            try:
                search_payload = json.loads(search_raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("ClinVar ESearch returned invalid JSON") from error
            result = search_payload.get("esearchresult", {})
            identifiers = result.get("idlist", [])
            count_text = result.get("count", "0")
            if not isinstance(identifiers, list) or any(not isinstance(item, str) or not item.isdigit() for item in identifiers):
                raise ValueError("ClinVar ESearch returned malformed variation identifiers")
            try:
                total_count = max(0, int(count_text))
            except (TypeError, ValueError):
                raise ValueError("ClinVar ESearch returned malformed result count") from None
            summaries = {}
            locations_by_uid = {}
            if identifiers:
                summary_params = {"db": "clinvar", "id": ",".join(identifiers), "retmode": "json"}
                if self.api_key:
                    summary_params["api_key"] = self.api_key
                if self.email:
                    summary_params["email"] = self.email
                async with client.stream("GET", f"{self.endpoint}/esummary.fcgi", params=summary_params) as response:
                    summary_raw = await read_bounded_response(response, self.max_response_bytes, "ClinVar ESummary")
                try:
                    summary_payload = json.loads(summary_raw).get("result", {})
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("ClinVar ESummary returned invalid JSON") from error
                summaries = summary_payload if isinstance(summary_payload, dict) else {}
                fetch_params = {
                    "db": "clinvar", "id": ",".join(identifiers), "rettype": "vcv",
                    "is_variationid": "true", "from_esearch": "true",
                }
                if self.api_key:
                    fetch_params["api_key"] = self.api_key
                if self.email:
                    fetch_params["email"] = self.email
                async with client.stream("GET", f"{self.endpoint}/efetch.fcgi", params=fetch_params) as response:
                    fetch_raw = await read_bounded_response(response, self.max_response_bytes, "ClinVar EFetch")
                locations_by_uid = self._parse_vcv_locations(fetch_raw)

        retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        records = []
        for uid in identifiers:
            raw = summaries.get(uid)
            if not isinstance(raw, dict):
                continue
            selected = {
                key: raw[key]
                for key in ("uid", "accession", "title", "variation_set", "clinical_significance", "genes", "variation_type", "obj_type", "last_updated")
                if key in raw
            }
            canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            variation_id = str(raw.get("uid", uid))
            clinical_significance = raw.get("clinical_significance")
            significance = {
                key: clinical_significance[key][:500] if isinstance(clinical_significance.get(key), str) else clinical_significance[key]
                for key in ("description", "review_status", "last_evaluated")
                if isinstance(clinical_significance, dict) and isinstance(clinical_significance.get(key), (str, int, float, bool))
            }
            genes = []
            for gene in raw.get("genes", [])[:20] if isinstance(raw.get("genes"), list) else []:
                if isinstance(gene, dict):
                    genes.append({key: gene[key][:200] if isinstance(gene.get(key), str) else gene[key] for key in ("symbol", "GeneID", "geneid") if isinstance(gene.get(key), (str, int))})
            records.append({
                "variation_id": variation_id,
                "accession": raw.get("accession", "")[:40] if isinstance(raw.get("accession"), str) else None,
                "title": raw.get("title")[:1000] if isinstance(raw.get("title"), str) else None,
                "clinical_significance": significance or None,
                "genes": genes,
                "assembly_locations": locations_by_uid.get(uid, []),
                "location_source_sha256": hashlib.sha256(json.dumps(locations_by_uid.get(uid, []), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "variation_type": raw.get("variation_type")[:160] if isinstance(raw.get("variation_type"), str) else None,
                "last_updated": raw.get("last_updated") if isinstance(raw.get("last_updated"), str) else None,
                "source": "NCBI ClinVar E-utilities",
                "source_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/",
                "source_record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "retrieved_at": retrieved_at,
                "classification_semantics": "Archived submitter classifications aggregated by ClinVar; not independently re-evaluated and not a clinical recommendation.",
            })
        return {
            "query": query,
            "total_matches_reported": total_count,
            "returned_count": len(records),
            "records": records,
            "source": "NCBI ClinVar E-utilities",
            "retrieved_at": retrieved_at,
            "interpretation_limit": "ClinVar aggregates submitted variant interpretations. This lookup does not establish pathogenicity for an individual or provide medical advice.",
        }

    @staticmethod
    def _parse_vcv_locations(xml_bytes: bytes) -> dict[str, list[dict]]:
        """Extract precise small-allele locations; composite variation sets are excluded."""
        reject_xml_declarations(xml_bytes, "ClinVar EFetch")
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as error:
            raise ValueError("ClinVar EFetch returned malformed VCV XML") from error
        output: dict[str, list[dict]] = {}
        for archive in root.iter("VariationArchive"):
            uid = archive.attrib.get("VariationID")
            if not uid or not uid.isdigit():
                continue
            classified = archive.find(".//ClassifiedRecord")
            if classified is None:
                continue
            if classified.find(".//Haplotype") is not None or classified.find(".//Genotype") is not None:
                continue
            alleles = classified.findall(".//SimpleAllele")
            if len(alleles) != 1:
                continue
            for location in alleles[0].findall(".//SequenceLocation"):
                attrs = location.attrib
                required = ("Assembly", "Chr", "start", "referenceAllele")
                if not all(attrs.get(key) for key in required) or "alternateAllele" not in attrs:
                    continue
                if attrs["Assembly"] not in {"GRCh37", "GRCh38"} or not attrs["start"].isdigit():
                    continue
                reference_allele = "" if attrs["referenceAllele"] == "-" else attrs["referenceAllele"]
                alternate_allele = "" if attrs["alternateAllele"] == "-" else attrs["alternateAllele"]
                if (not reference_allele and not alternate_allele
                        or (reference_allele and not reference_allele.isalpha())
                        or (alternate_allele and not alternate_allele.isalpha())):
                    continue
                stop = attrs.get("stop")
                output.setdefault(uid, []).append({
                    "assembly": attrs["Assembly"],
                    "contig": attrs["Chr"],
                    "position_1_based": int(attrs["start"]),
                    "end_position_1_based": int(stop) if stop and stop.isdigit() else None,
                    "reference_allele": reference_allele.upper(),
                    "alternate_allele": alternate_allele.upper(),
                    "sequence_accession": attrs.get("Accession"),
                    "source_coordinate_convention": "ClinVar VCV XML SequenceLocation, 1-based and HGVS-right-shifted per NCBI documentation; normalize against the supplied reference before comparison",
                })
        return output
