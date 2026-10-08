from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from medai_science.genomics import analyze_vcf


class VCFGenotypePloidyTests(unittest.TestCase):
    def test_counts_haploid_and_diploid_dosage_separately_and_keeps_polyploid_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
                "\tdiploid_ref\tdiploid_het\tdiploid_alt\thaploid_ref\thaploid_alt\tpolyploid\tmissing\n"
                "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0|1\t1/1\t0\t1\t1/1/1\t./.\n",
                encoding="ascii",
            )

            report = analyze_vcf(path, "GRCh38")

        counts = report["variants"][0]["genotype_call_counts"]
        self.assertEqual(counts, {
            "diploid_alt_dosage_0": 1,
            "diploid_alt_dosage_1": 1,
            "diploid_alt_dosage_2": 1,
            "haploid_alt_dosage_0": 1,
            "haploid_alt_dosage_1": 1,
            "other_or_ploidy": 1,
            "missing": 1,
        })
        self.assertIn("haploid and diploid dosages are reported separately", report["genotype_count_semantics"])
        self.assertFalse(report["sample_identifiers_retained"])

    def test_requires_fileformat_declaration_before_column_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-fileformat.vcf"
            path.write_text(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "missing its required first-line"):
                analyze_vcf(path, "GRCh38")

    def test_rejects_metadata_after_column_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "late-metadata.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "##reference=GRCh38\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "metadata appears after the #CHROM header"):
                analyze_vcf(path, "GRCh38")

    def test_rejects_genotypes_that_mix_phased_and_unphased_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed-phase.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
                "1\t10\t.\tA\tC,G\t.\tPASS\t.\tGT\t0/1|2\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "cannot mix phased and unphased separators"):
                analyze_vcf(path, "GRCh38")


if __name__ == "__main__":
    unittest.main()
