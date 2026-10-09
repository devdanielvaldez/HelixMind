"""NCBI E-utilities adapter. Respects API limits and retains source identifiers."""
from __future__ import annotations
import asyncio
import os
import time
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
import httpx
from medai_science.sources.safety import ensure_public_research_query
from medai_science.sources.bounded_http import event_loop_lock, read_bounded_response
from medai_science.sources.xml_safety import reject_xml_declarations
from medai_science.sources.pubmed_search import add_fetch_coverage, parse_esearch_response, parse_related_publications, validate_pmid_snapshot
from medai_science.sources.pubmed_cache import PubMedSearchCache

if TYPE_CHECKING:
    from medai_science.models import Paper

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MAX_SEARCH_RESPONSE_BYTES = 2_000_000
MAX_FETCH_RESPONSE_BYTES = 25 * 1024 * 1024
MAX_BIBLIOGRAPHIC_RESPONSE_BYTES = 5 * 1024 * 1024

class PubMedClient:
    _last_request = 0.0
    _last_min_interval = 0.0

    def __init__(self, email: str, api_key: str | None = None, timeout: float = 20, transport: httpx.AsyncBaseTransport | None = None, search_cache_path: str | None = None):
        if not email or "@" not in email:
            raise ValueError("NCBI requests require a valid contact email")
        self.params = {"tool": "medai", "email": email}
        if api_key:
            self.params["api_key"] = api_key
        self.timeout = timeout
        self.transport = transport
        self._min_interval = 0.11 if api_key else 0.34
        cache_path = search_cache_path or os.environ.get("MEDAI_PUBMED_SEARCH_CACHE")
        self.search_cache = PubMedSearchCache(cache_path) if cache_path else None

    async def _get(self, client: httpx.AsyncClient, endpoint: str, params: dict, max_bytes: int) -> bytes:
        async with event_loop_lock("pubmed"):
            minimum_interval = max(self._min_interval, type(self)._last_min_interval)
            delay = minimum_interval - (time.monotonic() - type(self)._last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            async with client.stream("GET", BASE + endpoint, params=params, headers={"Accept-Encoding": "identity"}) as response:
                type(self)._last_request = time.monotonic()
                type(self)._last_min_interval = self._min_interval
                return await read_bounded_response(response, max_bytes, f"PubMed {endpoint}")

    async def search(self, query: str, limit: int = 20) -> list[str]:
        identifiers, _ = await self.search_with_coverage(query, limit)
        return identifiers

    async def search_with_coverage(self, query: str, limit: int = 20) -> tuple[list[str], dict]:
        """Return the bounded PMID page and the provider-reported total search count."""
        query = ensure_public_research_query(query)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ValueError("limit must be 1..200")
        if self.search_cache is not None:
            cached = self.search_cache.get(query, limit)
            if cached is not None:
                return cached
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            raw = await self._get(client, "esearch.fcgi", {**self.params, "db": "pubmed", "term": query, "retmax": limit, "retmode": "json"}, MAX_SEARCH_RESPONSE_BYTES)
            identifiers, coverage = parse_esearch_response(raw, limit)
        if self.search_cache is not None:
            self.search_cache.put(query, limit, identifiers, coverage)
        return identifiers, coverage

    async def search_recent_ids(self, query: str, window_days: int, limit: int = 200) -> tuple[int, list[str]]:
        """Return recent PMID identifiers and PubMed's total count for an overlapping EDAT window."""
        query = ensure_public_research_query(query)
        if (isinstance(window_days, bool) or not isinstance(window_days, int) or not 1 <= window_days <= 365
                or isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200):
            raise ValueError("window_days must be 1..365 and limit must be 1..200")
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            raw = await self._get(client, "esearch.fcgi", {
                **self.params, "db": "pubmed", "term": query, "retmax": limit,
                "retmode": "json", "reldate": window_days, "datetype": "edat", "sort": "pub_date",
            }, MAX_SEARCH_RESPONSE_BYTES)
        identifiers, coverage = parse_esearch_response(raw, limit)
        return coverage["total_hits"], identifiers

    async def fetch(self, identifiers: list[str]) -> list[Paper]:
        if not isinstance(identifiers, list):
            raise ValueError("fetch requires a list of unique ASCII numeric PubMed identifiers")
        if not identifiers:
            return []
        validate_pmid_snapshot(identifiers, max_items=200)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            raw = await self._get(client, "efetch.fcgi", {**self.params, "db": "pubmed", "id": ",".join(identifiers), "retmode": "xml"}, MAX_FETCH_RESPONSE_BYTES)
        reject_xml_declarations(raw, "PubMed EFetch")
        root = ET.fromstring(raw)
        papers: list[Paper] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else ""
            abstract_sections = []
            for section in article.findall(".//Abstract/AbstractText"):
                text = " ".join("".join(section.itertext()).split())
                if text:
                    abstract_sections.append({"source_label": section.attrib.get("Label") or section.attrib.get("NlmCategory"), "text": text})
            abstract = " ".join(section["text"] for section in abstract_sections)
            authors = [" ".join(filter(None, [a.findtext("./LastName"), a.findtext("./ForeName")])) for a in article.findall(".//Author")]
            date_node = article.find(".//PubDate")
            date = date_node.findtext("./Year") or date_node.findtext("./MedlineDate") if date_node is not None else None
            publication_types = [" ".join("".join(node.itertext()).split()) for node in article.findall(".//PublicationTypeList/PublicationType")]
            publication_types = [value for value in publication_types if value]
            publication_type_ids = [node.attrib["UI"] for node in article.findall(".//PublicationTypeList/PublicationType") if node.attrib.get("UI")]
            related_publications = parse_related_publications(article)
            mesh_headings = []
            for heading in article.findall(".//MeshHeadingList/MeshHeading"):
                descriptor = heading.find("./DescriptorName")
                if descriptor is None or not "".join(descriptor.itertext()).strip():
                    continue
                mesh_headings.append({
                    "descriptor": " ".join("".join(descriptor.itertext()).split()),
                    "descriptor_ui": descriptor.attrib.get("UI", ""),
                    "qualifiers": [
                        {"name": " ".join("".join(node.itertext()).split()), "ui": node.attrib.get("UI", "")}
                        for node in heading.findall("./QualifierName")
                    ],
                })
            journal_node = article.find(".//Journal")
            journal = journal_node.findtext("./Title") if journal_node is not None else None
            journal_abbreviation = journal_node.findtext("./ISOAbbreviation") if journal_node is not None else None
            journal_issns = [{"value": (node.text or "").strip(), "type": node.attrib.get("IssnType", "")} for node in (journal_node.findall("./ISSN") if journal_node is not None else []) if (node.text or "").strip()]
            issue_node = journal_node.find("./JournalIssue") if journal_node is not None else None
            volume = issue_node.findtext("./Volume") if issue_node is not None else None
            issue = issue_node.findtext("./Issue") if issue_node is not None else None
            pages = article.findtext(".//Pagination/MedlinePgn")
            language = article.findtext(".//Language")
            if pmid:
                from medai_science.models import Paper, SourceRecord

                identifiers = {}
                for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                    kind = article_id.attrib.get("IdType", "").casefold()
                    value = (article_id.text or "").strip()
                    if kind in {"doi", "pmc"} and value:
                        identifiers[kind] = value
                papers.append(Paper(provenance=SourceRecord(source="PubMed", source_id=pmid, title=title, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"), abstract=abstract, abstract_sections=abstract_sections, authors=authors, publication_date=date, publication_types=publication_types, publication_type_ids=publication_type_ids, related_publications=related_publications, mesh_headings=mesh_headings, journal=journal, journal_abbreviation=journal_abbreviation, journal_issns=journal_issns, volume=volume, issue=issue, pages=pages, language=language, identifiers=identifiers))
        return papers

    async def fetch_bibliographic(self, identifiers: list[str]) -> list[dict[str, str | None]]:
        """Fetch compact PubMed citation metadata without downloading article abstracts."""
        if not isinstance(identifiers, list):
            raise ValueError("bibliographic fetch requires a list of unique ASCII numeric PubMed identifiers")
        if not identifiers:
            return []
        validate_pmid_snapshot(identifiers, max_items=20)
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            raw = await self._get(client, "esummary.fcgi", {**self.params, "db": "pubmed", "id": ",".join(identifiers), "retmode": "xml"}, MAX_BIBLIOGRAPHIC_RESPONSE_BYTES)
        reject_xml_declarations(raw, "PubMed ESummary")
        root = ET.fromstring(raw)
        records = []
        for summary in root.findall(".//DocSum"):
            pmid = summary.findtext("./Id")
            items = {item.attrib.get("Name", ""): " ".join("".join(item.itertext()).split()) for item in summary.findall("./Item")}
            article_ids = {item.attrib.get("Name", ""): " ".join("".join(item.itertext()).split()) for item in summary.findall(".//Item[@Name]")}
            if not pmid or not pmid.isdigit():
                continue
            records.append({
                "pmid": pmid,
                "title": items.get("Title", ""),
                "publication_date": items.get("PubDate"),
                "journal": items.get("FullJournalName") or items.get("Source"),
                "doi": article_ids.get("doi"),
                "pmcid": article_ids.get("pmc"),
            })
        return records

    async def search_and_fetch(self, query: str, limit: int = 20) -> list[Paper]:
        papers, _ = await self.search_and_fetch_with_coverage(query, limit)
        return papers

    async def search_and_fetch_with_coverage(self, query: str, limit: int = 20) -> tuple[list[Paper], dict]:
        """Fetch one bounded PMID page and report search and record-fetch coverage separately."""
        identifiers, coverage = await self.search_with_coverage(query, limit)
        papers = await self.fetch(identifiers)
        fetched_ids = [paper.provenance.source_id for paper in papers]
        return papers, add_fetch_coverage(coverage, identifiers, fetched_ids)
