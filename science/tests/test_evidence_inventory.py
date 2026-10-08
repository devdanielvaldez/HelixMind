from __future__ import annotations

import unittest
from types import SimpleNamespace

from medai_science.evidence import summarize


class EvidenceInventoryTests(unittest.TestCase):
    def test_preserves_source_quality_annotations_without_weighted_aggregation(self) -> None:
        claims = [
            SimpleNamespace(
                text="First source claim",
                polarity=SimpleNamespace(value="supports"),
                evidence_quality=0.9,
                source=SimpleNamespace(source="PubMed", source_id="123"),
                limitations=[],
            ),
            SimpleNamespace(
                text="Second source claim",
                polarity=SimpleNamespace(value="supports"),
                evidence_quality=0.2,
                source=SimpleNamespace(source="IntAct", source_id="interaction-1"),
                limitations=[],
            ),
        ]
        result = summarize(claims)
        self.assertEqual(result["by_polarity"]["supports"], 2)
        self.assertEqual(
            result["source_quality_annotations"],
            [
                {"source": "IntAct", "source_id": "interaction-1", "polarity": "supports", "value": 0.2},
                {"source": "PubMed", "source_id": "123", "polarity": "supports", "value": 0.9},
            ],
        )
        self.assertNotIn("quality_weighted_observations", result)
        self.assertIn("not evidence weights", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
