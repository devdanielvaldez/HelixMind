"""Opt-in real Docker/Wasmtime execution for CI's immutable sandbox image."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

from medai_science.wasm_sandbox import DockerWasmSandbox


class DockerWasmSandboxIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MEDAI_WASM_SANDBOX_IMAGE") and os.environ.get("MEDAI_WASM_FIXTURE_MODULE"),
        "CI has not provisioned the immutable Wasmtime image and compiled fixture",
    )
    def test_json_wasi_guest_executes_in_real_isolated_container(self):
        image = os.environ["MEDAI_WASM_SANDBOX_IMAGE"]
        module = Path(os.environ["MEDAI_WASM_FIXTURE_MODULE"]).read_bytes()
        sandbox = DockerWasmSandbox(image, timeout=30)
        receipt = sandbox.execute(
            module,
            {"value": 7},
            [
                {"input": {"value": 0}, "expected": {"value": 1}},
                {"input": {"value": -4}, "expected": {"value": -3}},
                {"input": {"value": 41}, "expected": {"value": 42}},
            ],
            contract={
                "input": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False},
                "output": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False},
            },
        )
        self.assertEqual(receipt["result"], {"value": 8})
        self.assertTrue(receipt["case_tests_passed"])
        self.assertEqual(receipt["validation"]["passed"], 3)
        self.assertEqual(receipt["validation"]["failed"], 0)
        self.assertEqual(receipt["image"], image)
        self.assertEqual(len(receipt["module_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
