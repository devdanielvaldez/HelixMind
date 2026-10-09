"""Opt-in real Docker execution for CI's immutable R sandbox image."""
from __future__ import annotations

import os
import unittest

from medai_science.r_sandbox import DockerRSandbox


class DockerRSandboxIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MEDAI_R_SANDBOX_IMAGE"), "CI has not provisioned an immutable R sandbox image")
    def test_r_capability_executes_in_real_isolated_container(self):
        image = os.environ["MEDAI_R_SANDBOX_IMAGE"]
        sandbox = DockerRSandbox(image, timeout=30)
        contract = {
            "input": {
                "type": "object",
                "required": ["left", "right"],
                "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
                "additionalProperties": False,
            },
            "output": {
                "type": "object",
                "required": ["sum"],
                "properties": {"sum": {"type": "integer"}},
                "additionalProperties": False,
            },
        }
        receipt = sandbox.execute(
            "run <- function(input) list(sum=as.integer(input$left + input$right))\n",
            {"left": 7, "right": 11},
            [
                {"input": {"left": 0, "right": 0}, "expected": {"sum": 0}},
                {"input": {"left": -4, "right": 9}, "expected": {"sum": 5}},
                {"input": {"left": 21, "right": 34}, "expected": {"sum": 55}},
            ],
            contract=contract,
        )
        self.assertEqual(receipt["result"], {"sum": 18})
        self.assertTrue(receipt["case_tests_passed"])
        self.assertEqual(receipt["validation"]["passed"], 3)
        self.assertEqual(receipt["validation"]["failed"], 0)
        self.assertEqual(receipt["image"], image)
        self.assertEqual(len(receipt["source_sha256"]), 64)
        self.assertGreaterEqual(receipt["cpu_time_microseconds"], 0)


if __name__ == "__main__":
    unittest.main()
