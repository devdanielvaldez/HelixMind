from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from medai_science.json_input import read_strict_json, strict_json_loads


class StrictResearchJSONInputTests(unittest.TestCase):
    def test_accepts_valid_utf8_and_rejects_duplicate_keys_nonfinite_and_non_utf8(self):
        self.assertEqual(strict_json_loads(b'{"allele":"A>G","position":17}'), {"allele": "A>G", "position": 17})
        for payload, message in (
            (b'{"assembly":"GRCh38","assembly":"GRCh37"}', "duplicate JSON object key"),
            (b'{"frequency":NaN}', "non-standard JSON number"),
            (b'{"label":"\xff"}', "strict UTF-8"),
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, message):
                strict_json_loads(payload)

    def test_reads_bounded_regular_files_and_fails_closed_on_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            path.write_bytes(b'{"records":[]}')
            self.assertEqual(read_strict_json(path, 64, "ClinVar records"), {"records": []})
            path.write_bytes(b" " * 65)
            with self.assertRaisesRegex(ValueError, "exceeds the 64-byte limit"):
                read_strict_json(path, 64, "ClinVar records")

    def test_rejects_non_regular_inputs_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "regular file"):
                read_strict_json(folder, 1024, "receipt")
            path = Path(folder) / "receipt.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                read_strict_json(path, True, "receipt")
            if hasattr(os, "mkfifo"):
                fifo = Path(folder) / "receipt.pipe"
                os.mkfifo(fifo)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    read_strict_json(fifo, 1024, "receipt")


if __name__ == "__main__":
    unittest.main()
