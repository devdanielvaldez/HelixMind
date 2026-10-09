from __future__ import annotations

import asyncio
from typing import Any
import unittest

import httpx

from medai_science.sources.alphafold import AlphaFoldClient


def _prediction() -> dict[str, Any]:
    return {
        "modelEntityId": "AF-P05067-F1-v6",
        "uniprotAccession": "P05067",
        "uniprotId": "A4_HUMAN",
        "uniprotDescription": "Amyloid-beta precursor protein",
        "gene": "APP",
        "organismScientificName": "Homo sapiens",
        "taxId": 9606,
        "latestVersion": 6,
        "modelCreatedDate": "2025-10-21",
        "sequenceStart": 1,
        "sequenceEnd": 13,
        "sequence": "MKKLLFAIPLVVP",
        "globalMetricValue": 89.2,
        "fractionPlddtVeryLow": 0.01,
        "fractionPlddtLow": 0.08,
        "fractionPlddtConfident": 0.21,
        "fractionPlddtVeryHigh": 0.70,
        "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-v6.cif",
        "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-v6.pdb",
        "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-v6.bcif",
        "paeDocUrl": "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-predicted_aligned_error_v6.json",
        "isUniProtReviewed": True,
        "isUniProtReferenceProteome": True,
    }


class AlphaFoldTests(unittest.TestCase):
    def test_uses_current_schema_fields_and_preserves_provenance(self) -> None:
        prediction = _prediction()

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.host, "alphafold.com")
            self.assertEqual(request.url.path, "/api/prediction/P05067")
            return httpx.Response(200, json=[prediction])

        result = asyncio.run(AlphaFoldClient(transport=httpx.MockTransport(handler)).predictions("p05067"))
        model = result[0]
        self.assertEqual(model["model_id"], "AF-P05067-F1-v6")
        self.assertEqual(model["sequence_length"], len(prediction["sequence"]))
        self.assertEqual(model["confidence_metrics"]["global_metric_value"], 89.2)
        self.assertFalse(model["experimental_structure"])
        self.assertIn("domain placement", model["interpretation_limits"][1])
        self.assertEqual(model["record"], prediction)
        self.assertEqual(len(model["provenance"]["response_record_sha256"]), 64)

    def test_supports_legacy_aliases_during_provider_transition(self) -> None:
        legacy = _prediction()
        legacy["entryId"] = legacy.pop("modelEntityId")
        legacy["uniprotSequence"] = legacy.pop("sequence")
        legacy["uniprotStart"] = legacy.pop("sequenceStart")
        legacy["uniprotEnd"] = legacy.pop("sequenceEnd")
        legacy["isReviewed"] = legacy.pop("isUniProtReviewed")
        legacy["isReferenceProteome"] = legacy.pop("isUniProtReferenceProteome")
        result = asyncio.run(
            AlphaFoldClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[legacy]))).predictions("P05067")
        )
        self.assertEqual(result[0]["sequence"], legacy["uniprotSequence"])
        self.assertTrue(result[0]["is_uniprot_reviewed"])

    def test_handles_missing_models_and_rejects_bad_accession_or_mismatched_response(self) -> None:
        missing = AlphaFoldClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
        self.assertEqual(asyncio.run(missing.predictions("P05067")), [])
        with self.assertRaisesRegex(ValueError, "UniProt accession"):
            asyncio.run(missing.predictions("https://evil.example/P05067"))
        mismatch = _prediction()
        mismatch["uniprotAccession"] = "Q9Y6K9"
        client = AlphaFoldClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[mismatch])))
        with self.assertRaisesRegex(ValueError, "does not match"):
            asyncio.run(client.predictions("P05067"))

    def test_prediction_api_does_not_follow_provider_redirects(self) -> None:
        calls = []

        def redirect(request):
            calls.append(request.url.host)
            return httpx.Response(302, headers={"Location": "https://untrusted.example/redirect"})

        with self.assertRaises(httpx.HTTPStatusError):
            asyncio.run(AlphaFoldClient(transport=httpx.MockTransport(redirect)).predictions("P05067"))
        self.assertEqual(calls, ["alphafold.com"])

    def test_downloads_canonical_bounded_coordinate_file_and_hashes_bytes(self) -> None:
        body = b"HEADER    TEST\n"

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), "https://alphafold.ebi.ac.uk/files/AF-P05067-F1-v6.pdb")
            return httpx.Response(200, content=body)

        result = asyncio.run(AlphaFoldClient(transport=httpx.MockTransport(handler)).coordinate_file("AF-P05067-F1-v6"))
        self.assertEqual(result["content"], body)
        self.assertEqual(result["byte_length"], len(body))
        self.assertEqual(result["sha256"], __import__("hashlib").sha256(body).hexdigest())
        self.assertEqual(result["license"], "CC-BY-4.0")

    def test_rejects_untrusted_model_id_format_redirect_and_oversized_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "model entity"):
            AlphaFoldClient.coordinate_url("AF-P05067-F1-v6.evil")
        with self.assertRaisesRegex(ValueError, "format"):
            AlphaFoldClient.coordinate_url("AF-P05067-F1-v6", "json")
        redirect = AlphaFoldClient(transport=httpx.MockTransport(lambda _: httpx.Response(302, headers={"Location": "https://evil.example/x"})))
        with self.assertRaises(httpx.HTTPStatusError):
            asyncio.run(redirect.coordinate_file("AF-P05067-F1-v6"))
        oversized = AlphaFoldClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, headers={"Content-Length": str(33 * 1024 * 1024)})))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            asyncio.run(oversized.coordinate_file("AF-P05067-F1-v6"))


if __name__ == "__main__":
    unittest.main()
