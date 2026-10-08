"""Verify the documented gnomAD v4 GraphQL query against two provider examples.

Run from the repository root with:
    PYTHONPATH=science/src python3 science/scripts/verify_gnomad_live.py

This performs two public single-variant lookups, paced according to gnomAD's
published browser API guidance. It prints only callset counts/frequencies and
does not persist response bodies.
"""
from __future__ import annotations

import asyncio
import json

from medai_science.sources.gnomad import GnomADClient


DOCUMENTED_GRCH38_EXAMPLES = ("1-55052746-GT-G", "1-55058620-TG-T")


async def main() -> None:
    client = GnomADClient()
    results = []
    for variant_id in DOCUMENTED_GRCH38_EXAMPLES:
        result = await client.lookup(variant_id, "GRCh38")
        if not result["found"]:
            raise RuntimeError(f"documented gnomAD example was not found: {variant_id}")
        if result["genome_frequency"] is None and result["exome_frequency"] is None:
            raise RuntimeError(f"gnomAD returned no frequency callset for {variant_id}")
        results.append({
            "variant_id": result["variant_id"],
            "dataset": result["dataset"],
            "genome_frequency": result["genome_frequency"],
            "exome_frequency": result["exome_frequency"],
            "source_record_sha256": result["source_record_sha256"],
        })
    print(json.dumps({"provider": "gnomAD Browser GraphQL API", "validated_examples": results}, sort_keys=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
