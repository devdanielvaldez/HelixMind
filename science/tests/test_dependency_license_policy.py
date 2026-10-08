from __future__ import annotations

import unittest

from medai_science.dependency_license_policy import validate_license_inventory


class DependencyLicensePolicyTests(unittest.TestCase):
    def test_accepts_each_explicitly_allowed_label(self):
        self.assertEqual(
            validate_license_inventory([
                {"Name": "one", "License": "MIT License"},
                {"Name": "two", "License": "Apache Software License; BSD-3-Clause"},
            ]),
            [],
        )

    def test_rejects_an_unapproved_label_even_when_another_label_is_allowed(self):
        errors = validate_license_inventory([
            {"Name": "mixed", "License": "MIT License; GPL-3.0-only"},
        ])
        self.assertEqual(len(errors), 1)
        self.assertIn("GPL-3.0-only", errors[0])

    def test_rejects_unknown_missing_and_compound_labels(self):
        errors = validate_license_inventory([
            {"Name": "unknown", "License": "UNKNOWN"},
            {"Name": "missing", "License": ""},
            {"Name": "compound", "License": "MIT OR Apache-2.0"},
        ])
        self.assertEqual(len(errors), 3)

    def test_rejects_malformed_and_duplicate_package_rows(self):
        errors = validate_license_inventory([
            {"Name": "same", "License": "MIT"},
            {"Name": "same", "License": "MIT"},
            None,
        ])
        self.assertEqual(len(errors), 2)

    def test_rejects_empty_inventory(self):
        self.assertTrue(validate_license_inventory([]))


if __name__ == "__main__":
    unittest.main()
