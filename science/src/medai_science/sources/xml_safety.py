"""Shared checks for untrusted XML returned by scientific data providers."""
from __future__ import annotations

import re


_DECLARATION = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def reject_xml_declarations(raw: bytes, source: str) -> None:
    """Require UTF-8 XML and reject DTD/entity declarations before parsing.

    Checking bytes misses declarations encoded as UTF-16/32. Decode explicitly so
    every XML ingestion path applies the same policy before ElementTree sees data.
    """
    try:
        decoded = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} XML is not valid UTF-8") from error
    if _DECLARATION.search(decoded):
        raise ValueError(f"{source} XML must not contain document type or entity declarations")
