from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from medai_science.benchmark_artifacts import (
    DOMAIN_BENCHMARK_ARTIFACT_KIND,
    DOMAIN_BENCHMARK_ARTIFACT_SOURCE,
    EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND,
    EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE,
    MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES,
    evaluate_evidence_quality_benchmark_bytes,
    load_domain_benchmark_artifact,
    submit_domain_benchmark_artifact,
    submit_evidence_quality_benchmark_artifact,
)
from medai_science.quality import QUALITY_DOMAINS


def document():
    case = {
        "case_id": "case-1",
        "evidence_sha256": "a" * 64,
        "evidence_type": "source_record",
        "reviewers": [
            {"reviewer_id": "reviewer-a", "role": "reviewer", "label": "present"},
            {"reviewer_id": "reviewer-b", "role": "reviewer", "label": "present"},
        ],
        "predicted": "present",
        "confidence": 0.9,
    }
    return {
        "schema_version": "1",
        "benchmark_id": "fixture-v1",
        "labeling_protocol": "Two independent primary reviewers; adjudicate disagreements.",
        "domains": {
            "genomics": [case],
            "proteomics": [],
            "pharmacology": [],
            "literature_extraction": [],
        },
    }


def evidence_quality_document():
    ratings = {domain: "low" for domain in QUALITY_DOMAINS}
    return {
        "schema_version": "3",
        "benchmark_id": "quality-v1",
        "labeling_protocol": "Two distinct reviewers independently rate each hash-bound source; a third reviewer resolves disagreements.",
        "domain_scales": {
            domain: {"ordered_levels": ["low", "moderate", "high"], "non_ordinal_labels": ["not_assessable"]}
            for domain in QUALITY_DOMAINS
        },
        "cases": [{
            "case_id": "case-1",
            "evidence_sha256": "a" * 64,
            "evidence_type": "paper",
            "reviewers": [
                {"reviewer_id": "reviewer-a", "role": "reviewer", "ratings": ratings},
                {"reviewer_id": "reviewer-b", "role": "reviewer", "ratings": dict(ratings)},
            ],
            "predictions": {
                domain: {"label": "low", "confidence": 0.9}
                for domain in QUALITY_DOMAINS
            },
        }],
    }
class BenchmarkArtifactTests(unittest.TestCase):
    def test_loads_and_validates_bounded_strict_input(self):
        value = document()
        raw = json.dumps(value, separators=(",", ":")).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_bytes(raw)
            loaded, report, digest = load_domain_benchmark_artifact(path)
        self.assertEqual(loaded, value)
        self.assertEqual(report["total_cases"], 1)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["missing_domains"], ["proteomics", "pharmacology", "literature_extraction"])

    def test_rejects_ambiguous_nonfinite_invalid_utf8_and_oversized_files(self):
        bad_payloads = [
            b'{"schema_version":"1","schema_version":"1"}',
            b'{"x":NaN}',
            b"\xff",
            b" " * (MAX_DOMAIN_BENCHMARK_ARTIFACT_BYTES + 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            for payload in bad_payloads:
                path.write_bytes(payload)
                with self.subTest(length=len(payload)):
                    with self.assertRaises(ValueError):
                        load_domain_benchmark_artifact(path)

    def test_rejects_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            link = Path(directory) / "link.json"
            source.write_text(json.dumps(document()), encoding="utf-8")
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                load_domain_benchmark_artifact(link)

    def test_submission_binds_exact_research_task_artifact_and_encodes_session(self):
        value = document()
        report = {"benchmark_id": "fixture-v1"}
        request = Mock(return_value={"id": "artifact-1", "sha256": "b" * 64})
        result = submit_domain_benchmark_artifact(value, report, "RES:one", request)
        request.assert_called_once_with(
            "/v1/research/RES%3Aone/artifacts",
            {
                "task_id": "benchmark.evaluate",
                "kind": DOMAIN_BENCHMARK_ARTIFACT_KIND,
                "source": DOMAIN_BENCHMARK_ARTIFACT_SOURCE,
                "source_id": "fixture-v1",
                "payload": value,
            },
        )
        self.assertEqual(result["id"], "artifact-1")

    def test_submission_rejects_invalid_session_before_api_call(self):
        request = Mock()
        with self.assertRaisesRegex(ValueError, "session ID"):
            submit_domain_benchmark_artifact(document(), {"benchmark_id": "fixture-v1"}, "../other", request)
        request.assert_not_called()

    def test_evidence_quality_input_is_strict_hash_bound_and_domain_separated(self):
        value = evidence_quality_document()
        raw = json.dumps(value, separators=(",", ":")).encode()
        loaded, report, digest = evaluate_evidence_quality_benchmark_bytes(raw)
        self.assertEqual(loaded, value)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        self.assertEqual(report["cases"], 1)
        self.assertEqual(set(report["overall_by_domain"]), set(QUALITY_DOMAINS))
        for payload in (b'{"schema_version":"3","schema_version":"3"}', b'{"value":NaN}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                evaluate_evidence_quality_benchmark_bytes(payload)
        with self.assertRaisesRegex(ValueError, "schema version 3"):
            evaluate_evidence_quality_benchmark_bytes(b'{"schema_version":"2"}')

    def test_evidence_quality_submission_uses_typed_benchmark_task(self):
        value = evidence_quality_document()
        request = Mock(return_value={"id": "quality-artifact", "sha256": "b" * 64})
        result = submit_evidence_quality_benchmark_artifact(value, {"benchmark_id": "quality-v1"}, "RES:one", request)
        request.assert_called_once_with(
            "/v1/research/RES%3Aone/artifacts",
            {
                "task_id": "benchmark.evaluate",
                "kind": EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_KIND,
                "source": EVIDENCE_QUALITY_BENCHMARK_ARTIFACT_SOURCE,
                "source_id": "quality-v1",
                "payload": value,
            },
        )
        self.assertEqual(result["id"], "quality-artifact")


if __name__ == "__main__":
    unittest.main()
