from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

import httpx

from medai_science.sources.uniprot import UniProtClient
from medai_science.reproducibility import create_protein_bundle, verify_analysis_bundle


def _record() -> dict:
    return {
        "primaryAccession": "P05067",
        "features": [
            {"type": "Domain", "featureId": "PRO_000000001", "description": "E1 domain", "location": {"start": {"value": 20}, "end": {"value": 120}}},
            {"type": "Signal", "description": "Signal peptide", "location": {"start": {"value": 1}, "end": {"value": 17}}},
        ],
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "1AAP", "properties": [{"key": "Method", "value": "X-ray"}, {"key": "Resolution", "value": "2.40 A"}, {"key": "Chains", "value": "A=18-120"}]},
            {"database": "GO", "id": "GO:0000000", "properties": []},
        ],
        "sequence": {"length": 770, "value": "M" * 770},
    }


class UniProtEntryTests(unittest.TestCase):
    def test_extracts_domains_pdb_crossrefs_and_content_provenance(self) -> None:
        payload = _record()

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "rest.uniprot.org")
            self.assertEqual(request.url.path, "/uniprotkb/P05067.json")
            return httpx.Response(200, json=payload)

        result = asyncio.run(UniProtClient(transport=httpx.MockTransport(handler)).get_entry("p05067"))
        self.assertEqual(result["domains_and_regions"], [{"feature_id": "PRO_000000001", "type": "Domain", "description": "E1 domain", "start": 20, "end": 120, "sequence": None}])
        self.assertEqual(result["experimental_structure_cross_references"][0]["pdb_id"], "1AAP")
        self.assertEqual(result["experimental_structure_cross_references"][0]["resolution"], "2.40 A")
        self.assertEqual(len(result["record_sha256"]), 64)
        self.assertIn("not independently validated", result["structure_cross_reference_limit"])
        with tempfile.TemporaryDirectory(prefix="medai-uniprot-flow-") as temp:
            created = create_protein_bundle(Path(temp) / "bundle", result)
            replay = verify_analysis_bundle(Path(temp) / "bundle", replay=True)
        self.assertEqual(created["artifact_count"], 2)
        self.assertTrue(replay["valid"])
        self.assertTrue(replay["replayed"])

    def test_rejects_bad_accession_missing_record_and_mismatched_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "UniProt accession"):
            asyncio.run(UniProtClient().get_entry("https://example.org/P05067"))
        missing = UniProtClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
        with self.assertRaisesRegex(ValueError, "not found"):
            asyncio.run(missing.get_entry("P05067"))
        wrong = _record()
        wrong["primaryAccession"] = "Q9Y6K9"
        client = UniProtClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=wrong)))
        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(client.get_entry("P05067"))

    def test_search_validates_strict_json_and_every_record(self) -> None:
        async def run(body: bytes):
            client = UniProtClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))
            return await client.search("APP", limit=2)

        valid = b'{"results":[{"primaryAccession":"P05067","organism":{"taxonId":9606}}]}'
        result = asyncio.run(run(valid))
        self.assertEqual(result[0]["source_id"], "P05067")
        for body, message in (
            (b'{"results":[],"results":[]}', "duplicate JSON object key"),
            (b'{"results":[],"score":Infinity}', "non-standard JSON number"),
            (b'{"results":[null]}', "malformed search record"),
            (b'{"results":[{}]}', "missing accession"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                asyncio.run(run(body))


if __name__ == "__main__":
    unittest.main()
