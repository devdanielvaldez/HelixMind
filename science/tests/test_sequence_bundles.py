from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from medai_science.genomics import inspect_fasta, inspect_fastq
from medai_science.reproducibility import create_sequence_bundle, verify_analysis_bundle, verify_sequence_bundle


class SequenceBundleTests(unittest.TestCase):
    def test_compressed_fasta_bundle_replays_exact_parser_report_privately(self):
        with tempfile.TemporaryDirectory(prefix="medai-sequence-bundle-") as temp:
            root = Path(temp)
            source = root / "input.fa.gz"
            source.write_bytes(gzip.compress(b">private-record\nACGT\nTGCA\n"))
            report = inspect_fasta(source, "nucleotide", 10)
            bundle = root / "bundle"
            created = create_sequence_bundle(bundle, report, source, "fasta", 10, "nucleotide")
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            captured_report = (bundle / "report.json").read_text(encoding="utf-8")
            self.assertEqual(created["artifact_count"], 4)
            self.assertEqual(manifest["analysis"]["workflow"], "sequence.fasta")
            self.assertFalse(manifest["network_accessed"])
            self.assertNotIn("private-record", captured_report)
            self.assertEqual((bundle / manifest["analysis"]["input"]).read_bytes(), source.read_bytes())
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual((bundle / manifest["analysis"]["input"]).stat().st_mode & 0o777, 0o600)
            self.assertTrue(verify_analysis_bundle(bundle)["valid"])
            replay = verify_analysis_bundle(bundle, replay=True)
            self.assertTrue(replay["replayed"])

    def test_fastq_bundle_replays_quality_summary_and_detects_tampering(self):
        with tempfile.TemporaryDirectory(prefix="medai-sequence-bundle-") as temp:
            root = Path(temp)
            source = root / "reads.fastq"
            source.write_text("@read1\nACGT\n+\nIIII\n@read2\nGG\n+\n!!\n", encoding="ascii")
            report = inspect_fastq(source, 20)
            bundle = root / "bundle"
            create_sequence_bundle(bundle, report, source, "fastq", 20)
            self.assertEqual(verify_sequence_bundle(bundle, replay=True)["report_sha256"], json.loads((bundle / "manifest.json").read_text())["report_sha256"])
            relative = json.loads((bundle / "manifest.json").read_text())["analysis"]["input"]
            (bundle / relative).write_text("tampered", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                verify_sequence_bundle(bundle)

    def test_bundle_requires_matching_analysis_parameters_and_existing_destination_absent(self):
        with tempfile.TemporaryDirectory(prefix="medai-sequence-bundle-") as temp:
            root = Path(temp)
            source = root / "sequence.fa"
            source.write_text(">record\nACGT\n", encoding="ascii")
            report = inspect_fasta(source, "nucleotide", 10)
            with self.assertRaisesRegex(ValueError, "does not match"):
                create_sequence_bundle(root / "wrong", report, source, "fasta", 10, "protein")
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                create_sequence_bundle(existing, report, source, "fasta", 10, "nucleotide")

    def test_verifier_rejects_bundle_symlinks_before_following_them(self):
        with tempfile.TemporaryDirectory(prefix="medai-sequence-bundle-") as temp:
            root = Path(temp)
            source = root / "input.fa"
            source.write_text(">record\nACGT\n", encoding="ascii")
            report = inspect_fasta(source, "nucleotide", 10)
            bundle = root / "bundle"
            create_sequence_bundle(bundle, report, source, "fasta", 10)
            relative = json.loads((bundle / "manifest.json").read_text())["analysis"]["input"]
            captured = bundle / relative
            captured.unlink()
            try:
                captured.symlink_to(source)
            except OSError as error:
                self.skipTest(f"symlinks are unavailable in this test environment: {error}")
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                verify_sequence_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
