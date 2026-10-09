from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from medai_science.genomics import analyze_vcf, inspect_fasta, inspect_fastq
from medai_science.transcripts import _parse_gff3


_CORRUPT_DEFLATE = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + b"\xff" * 32


class CorruptCompressedSequenceTests(unittest.TestCase):
    def test_corrupt_deflate_is_reported_as_a_controlled_input_error(self):
        analyzers = (
            ("calls.vcf.gz", analyze_vcf, ("GRCh38",)),
            ("sequence.fa.gz", inspect_fasta, ("nucleotide",)),
            ("reads.fastq.gz", inspect_fastq, ()),
            ("transcripts.gff3.gz", _parse_gff3, ()),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename, analyzer, extra_args in analyzers:
                path = root / filename
                path.write_bytes(_CORRUPT_DEFLATE)
                with self.subTest(filename=filename), self.assertRaisesRegex(ValueError, "could not read"):
                    analyzer(path, *extra_args)


if __name__ == "__main__":
    unittest.main()
