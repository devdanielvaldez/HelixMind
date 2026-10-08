import unittest

from medai_science.compute_usage import meter_cpu_time


class ComputeUsageTests(unittest.TestCase):
    def test_rounds_measured_cpu_time_up_at_configured_rate(self):
        self.assertEqual(meter_cpu_time(0), {"cpu_time_microseconds": 0, "seconds_per_credit": 10.0, "credits": 0})
        self.assertEqual(meter_cpu_time(10_000_000)["credits"], 1)
        self.assertEqual(meter_cpu_time(10_000_001)["credits"], 2)
        self.assertEqual(meter_cpu_time(20_000_000, 5)["credits"], 4)

    def test_rejects_invalid_measured_cpu_or_rate(self):
        for usage in (-1, True, 1.0, "10"):
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                meter_cpu_time(usage)  # type: ignore[arg-type]
        for rate in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                meter_cpu_time(1, rate)


if __name__ == "__main__":
    unittest.main()
