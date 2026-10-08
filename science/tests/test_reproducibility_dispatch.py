from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from medai_science.reproducibility import _read_bundle_json, _write_private, verify_analysis_bundle


class ReproducibilityDispatchTests(unittest.TestCase):
    def test_unknown_workflow_is_rejected_instead_of_dispatched_as_genomics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-unknown-bundle-") as temp:
            bundle = Path(temp)
            (bundle / "manifest.json").write_text(
                json.dumps({"schema_version": "1", "analysis": {"workflow": "future.unregistered"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported workflow"):
                verify_analysis_bundle(bundle)

    def test_bundle_json_rejects_duplicate_keys_and_non_standard_numbers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-bundle-json-") as temp:
            source = Path(temp) / "record.json"
            source.write_text('{"result": 1, "result": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate object key"):
                _read_bundle_json(source, "test record")
            source.write_text('{"score": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard numeric constant"):
                _read_bundle_json(source, "test record")

    def test_bundle_json_enforces_size_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-bundle-json-") as temp:
            root = Path(temp)
            source = root / "record.json"
            source.write_text("{}", encoding="utf-8")
            source.write_bytes(b" " * (16 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "byte limit"):
                _read_bundle_json(source, "test record")
            oversized_report = root / "oversized.json"
            with self.assertRaisesRegex(ValueError, "bundle JSON exceeds"):
                _write_private(oversized_report, b" " * (16 * 1024 * 1024 + 1))
            self.assertFalse(oversized_report.exists())
            with self.assertRaisesRegex(ValueError, "non-standard numeric constant"):
                _write_private(oversized_report, b'{"score": Infinity}')
            self.assertFalse(oversized_report.exists())
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            source.unlink()
            source.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                _read_bundle_json(source, "test record")


if __name__ == "__main__":
    unittest.main()
