from __future__ import annotations

import unittest

from medai_science.contract_validation import capability_io_schemas, validate_value
from medai_science.sandbox import validate_capability_run_request


CONTRACT = {
    "inputs": {"type": "object", "required": ["variants"], "additionalProperties": False, "properties": {"variants": {"type": "array", "items": {"type": "object", "required": ["chromosome"], "properties": {"chromosome": {"type": "string", "minLength": 1}}}}}},
    "outputs": {"type": "object", "required": ["count"], "additionalProperties": False, "properties": {"count": {"type": "integer", "minimum": 0}}},
}


class CapabilityContractValidationTests(unittest.TestCase):
    def test_validates_nested_input_and_output_values(self):
        input_schema, output_schema = capability_io_schemas(CONTRACT)
        validate_value({"variants": [{"chromosome": "19"}]}, input_schema)
        validate_value({"count": 1}, output_schema)

    def test_rejects_wrong_types_missing_fields_and_extras(self):
        with self.assertRaisesRegex(ValueError, "required"):
            validate_value({}, CONTRACT["inputs"])
        with self.assertRaisesRegex(ValueError, "undeclared"):
            validate_value({"variants": [], "patient": "x"}, CONTRACT["inputs"])
        with self.assertRaisesRegex(ValueError, "type"):
            validate_value({"count": True}, CONTRACT["outputs"])

    def test_sandbox_request_checks_input_and_expected_output_contract(self):
        request = validate_capability_run_request(
            {"variants": [{"chromosome": "19"}]},
            [{"input": {"variants": []}, "expected": {"count": 0}}],
            CONTRACT,
        )
        self.assertEqual(request["output_schema"], CONTRACT["outputs"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_capability_run_request(
                {"variants": []},
                [{"input": {"variants": []}, "expected": {"count": "zero"}}],
                CONTRACT,
            )


if __name__ == "__main__":
    unittest.main()
