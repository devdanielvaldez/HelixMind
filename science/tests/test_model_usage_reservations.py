import os
import unittest
from unittest.mock import patch

from medai_science.model_usage import estimate_model_reservation, meter_model_usage


class ModelUsageReservationTests(unittest.TestCase):
    def test_reservation_covers_utf8_request_bytes_output_and_protocol_overhead(self):
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "1000"}):
            estimate = estimate_model_reservation("systém", "prompt", {"type": "object"}, 512)
        expected_tokens = len("systém".encode()) + len("prompt".encode()) + len(b'{"type":"object"}') + 512 + 512
        self.assertEqual(estimate["token_ceiling"], expected_tokens)
        self.assertEqual(estimate["credits"], (expected_tokens + 999) // 1000)

    def test_actual_usage_is_rounded_with_the_same_rate(self):
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "1000"}):
            estimate = estimate_model_reservation("system", "prompt", {}, 256)
            metered = meter_model_usage({"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300})
        self.assertGreaterEqual(estimate["credits"], metered["credits"])

    def test_invalid_rate_fails_before_provider_call(self):
        with patch.dict(os.environ, {"MEDAI_MODEL_TOKENS_PER_CREDIT": "0"}):
            with self.assertRaises(ValueError):
                estimate_model_reservation("system", "prompt", {}, 256)


if __name__ == "__main__":
    unittest.main()
