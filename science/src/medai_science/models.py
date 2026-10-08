from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, HttpUrl

class Polarity(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"

    def __str__(self) -> str:
        return self.value

class SourceRecord(BaseModel):
    source: str
    source_id: str
    title: str
    url: HttpUrl | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    license: str | None = None

class Claim(BaseModel):
    text: str
    polarity: Polarity
    evidence_quality: float = Field(ge=0, le=1)
    source: SourceRecord
    limitations: list[str] = Field(default_factory=list)

class AbstractSection(BaseModel):
    source_label: str | None = None
    text: str

class Paper(BaseModel):
    provenance: SourceRecord
    abstract: str = ""
    abstract_sections: list[AbstractSection] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    publication_date: str | None = None
    publication_types: list[str] = Field(default_factory=list)
    publication_type_ids: list[str] = Field(default_factory=list)
    related_publications: list[dict[str, str]] = Field(default_factory=list)
    mesh_headings: list[dict[str, Any]] = Field(default_factory=list)
    journal: str | None = None
    journal_abbreviation: str | None = None
    journal_issns: list[dict[str, str]] = Field(default_factory=list)
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    language: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    claims: list[Claim] = Field(default_factory=list)
