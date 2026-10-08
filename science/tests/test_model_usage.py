from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from medai_science.model_usage import ModelUsageError, meter_model_usage


class ModelUsageTests(unittest.TestCase):
    def test_calculates_credits_from_reported_input_and_output_tokens(self):
        self.assertEqual(
            meter_model_usage({"prompt_tokens": 1200, "completion_tokens": 801, "total_tokens": 2001}, 1000),
            {"prompt_tokens": 1200, "completion_tokens": 801, "total_tokens": 2001, "tokens_per_credit": 1000, "credits": 3},
        )

    def test_accepts_total_only_and_zero_usage(self):
        self.assertEqual(meter_model_usage({"total_tokens": 1000}, 1000)["credits"], 1)
        self.assertEqual(meter_model_usage({"total_tokens": 0}, 1000)["credits"], 0)

    def test_rejects_missing_malformed_and_inconsistent_provider_usage(self):
        bad = (
            {},
            {"prompt_tokens": 2},
            {"prompt_tokens": True, "completion_tokens": 1},
            {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 4},
            {"total_tokens": -1},
        )
        for usage in bad:
            with self.subTest(usage=usage), self.assertRaises(ModelUsageError):
                meter_model_usage(usage, 1000)

    def test_uses_validated_configured_tokens_per_credit(self):
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "250"}):
            self.assertEqual(meter_model_usage({"total_tokens": 251})["credits"], 2)
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "0"}):
            with self.assertRaisesRegex(ModelUsageError, "tokens_per_credit"):
                meter_model_usage({"total_tokens": 1})


if __name__ == "__main__":
    unittest.main()
