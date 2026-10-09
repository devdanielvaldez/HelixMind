from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medai_science.genomics import analyze_vcf


class VCFGenotypePloidyTests(unittest.TestCase):
    def test_multiallelic_records_count_toward_expanded_variant_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many-alternates.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t10\t.\tA\tC,G,T\t.\tPASS\t.\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "maximum of 2 alternate alleles"):
                analyze_vcf(path, "GRCh38", max_variants=2)

    def test_genotype_work_is_bounded_after_multiallelic_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genotype-work.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
                "1\t10\t.\tA\tC,G,T\t.\tPASS\t.\tGT\t0/1\n",
                encoding="ascii",
            )
            with patch("medai_science.genomics._MAX_GENOTYPE_CALLS", 2):
                with self.assertRaisesRegex(ValueError, "genotype calls exceed"):
                    analyze_vcf(path, "GRCh38")

    def test_vcf_enforces_record_limit_before_parsing_next_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "limited.vcf"
            path.write_text(
                "##fileformat=VCFv4.3\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t10\t.\tA\tC\t.\tPASS\t.\n"
                "this second record is deliberately malformed\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "maximum of 1 records"):
                analyze_vcf(path, "GRCh38", max_variants=1)

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

    def test_rejects_non_ascii_decimal_digits_in_positions_and_genotypes(self):
        records = (
            ("١٠", "0/1", "invalid VCF position"),
            ("10", "٠/١", "invalid genotype allele index"),
        )
        for position, genotype, expected in records:
            with self.subTest(position=position, genotype=genotype), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "non-ascii-digit.vcf"
                path.write_text(
                    "##fileformat=VCFv4.3\n"
                    "##reference=GRCh38\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
                    f"1\t{position}\t.\tA\tC\t.\tPASS\t.\tGT\t{genotype}\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    analyze_vcf(path, "GRCh38")


if __name__ == "__main__":
    unittest.main()
