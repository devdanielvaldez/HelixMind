import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
from medai_science.genomics import analyze_vcf


def clinvar_record(variation_id, locations, accession=None):
    return {
        "variation_id": str(variation_id),
        "accession": accession or f"VCV{int(variation_id):09d}.1",
        "source": "NCBI ClinVar E-utilities",
        "source_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/",
        "source_record_sha256": hashlib.sha256(f"record:{variation_id}".encode()).hexdigest(),
        "location_source_sha256": hashlib.sha256(json.dumps(locations, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "assembly_locations": locations,
    }


class ClinVarReconciliationTests(unittest.TestCase):
    def test_clinvar_prrt2_homopolymer_duplication_matches_left_shifted_vcf(self):
        # ClinVar VCV000065758.157 reports NM_145239.3:c.649dupC /
        # NC_000016.10:g.29813703dup and canonical SPDI
        # NC_000016.10:29813694:CCCCCCCCC:CCCCCCCCCC. The SPDI allele therefore
        # anchors a nine-C run at 0-based offset 29,813,694. ClinVar documents
        # that XML coordinates follow HGVS (right-shifted) while VCF uses
        # left-shifted representation:
        # https://www.ncbi.nlm.nih.gov/clinvar/variation/VCV000065758.157/
        # https://www.ncbi.nlm.nih.gov/clinvar/docs/ftp_primer/
        # The published left-normalized representation is chr16:29813694G>GC
        # (https://www.mediasphera.ru/issues/zhurnal-nevrologii-i-psikhiatrii-im-s-s-korsakova/2022/5/1199772982022051128).
        # ClinVar independently reports the anchor base as NC_000016.10:g.29813694G>C
        # with canonical SPDI NC_000016.10:29813693:G:C (VCV000206687.64):
        # https://www.ncbi.nlm.nih.gov/clinvar/variation/206687/
        # Match that anchor and the ClinVar repeat tract; the remaining flanks
        # in this generated FASTA are synthetic and do not stand in for GRCh38.
        position_1_based = 29_813_695
        rightmost_repeat_position = position_1_based + 8
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-prrt2-repeat-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"

            prefix_length = position_1_based - 1
            full_lines, remainder = divmod(prefix_length, 60)
            with reference.open("w", encoding="ascii") as handle:
                handle.write(">16\n")
                line = "A" * 60 + "\n"
                while full_lines:
                    count = min(full_lines, 16_384)
                    handle.write(line * count)
                    full_lines -= count
                suffix = "A" * (remainder - 1) + "G" + "C" * 9 + "A" * 200
                for offset in range(0, len(suffix), 60):
                    handle.write(suffix[offset:offset + 60] + "\n")

            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "16\t29813694\t.\tG\tC\t.\tPASS\t.\n"
                f"16\t{position_1_based}\t.\tC\tCC\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            location = {
                "assembly": "GRCh38",
                "contig": "16",
                "position_1_based": rightmost_repeat_position,
                "reference_allele": "C",
                "alternate_allele": "CC",
            }
            insertion_source = clinvar_record("65758", [location], accession="VCV000065758.157")
            anchor_snv = clinvar_record("206687", [{
                "assembly": "GRCh38",
                "contig": "16",
                "position_1_based": 29_813_694,
                "reference_allele": "G",
                "alternate_allele": "C",
            }], accession="VCV000206687.64")
            result = reconcile_clinvar_variants(report, [insertion_source, anchor_snv], reference, "GRCh38")

        by_allele = {row["alternate_allele"]: row for row in result["variants"]}
        insertion = by_allele["GC"]
        self.assertEqual(insertion["status"], "exact_normalized_allele_match")
        self.assertEqual(insertion["position_1_based"], position_1_based - 1)
        self.assertEqual(insertion["reference_allele"], "G")
        self.assertEqual(insertion["clinvar_records"][0]["accession"], "VCV000065758.157")
        anchor = by_allele["C"]
        self.assertEqual(anchor["status"], "exact_normalized_allele_match")
        self.assertEqual(anchor["position_1_based"], 29_813_694)
        self.assertEqual(anchor["reference_allele"], "G")
        self.assertEqual(anchor["clinvar_records"][0]["accession"], "VCV000206687.64")

    def test_independently_curated_clinvar_mitochondrial_snv_matches_exact_local_vcf_allele(self):
        # NCBI ClinVar VCV000009589.86 reports NC_012920.1:m.3243A>G at MT:3243
        # on both assemblies: https://www.ncbi.nlm.nih.gov/clinvar/variation/9589/
        # This minimal reference fixture exercises allele matching;
        # it is not a substitute for the complete NC_012920.1 reference sequence.
        locations = [{
            "assembly": "GRCh38",
            "contig": "MT",
            "position_1_based": 3243,
            "reference_allele": "A",
            "alternate_allele": "G",
        }]
        source = clinvar_record("9589", locations, accession="VCV000009589.86")
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-mt-curated-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"
            reference.write_text(">MT\n" + "N" * 3242 + "A" + "N" * 20 + "\n", encoding="ascii")
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "MT\t3243\t.\tA\tG\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            result = reconcile_clinvar_variants(report, [source], reference, "GRCh38")

        self.assertEqual(result["variants"][0]["status"], "exact_normalized_allele_match")
        self.assertEqual(result["variants"][0]["clinvar_records"][0]["variation_id"], "9589")
        self.assertEqual(result["variants"][0]["clinvar_records"][0]["accession"], "VCV000009589.86")
        self.assertEqual(result["variants"][0]["position_1_based"], 3243)

    def test_right_shifted_repeat_indel_matches_after_reference_normalization(self):
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-repeat-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"
            reference.write_text(">1\nCAAAAAG\n", encoding="ascii")
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t5\t.\tA\tAA\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            source = clinvar_record("42", [{"assembly": "GRCh38", "contig": "1", "position_1_based": 6, "reference_allele": "A", "alternate_allele": "AA"}])
            result = reconcile_clinvar_variants(report, [source], reference, "GRCh38")

        self.assertEqual(result["variants"][0]["position_1_based"], 1)
        self.assertEqual(result["variants"][0]["reference_allele"], "C")
        self.assertEqual(result["variants"][0]["alternate_allele"], "CA")
        self.assertEqual(result["variants"][0]["status"], "exact_normalized_allele_match")
        source_receipt = result["variants"][0]["clinvar_records"][0]
        self.assertEqual(source_receipt["source_url"], "https://www.ncbi.nlm.nih.gov/clinvar/variation/42/")
        self.assertEqual(source_receipt["source"], "NCBI ClinVar E-utilities")
        self.assertIn("not cryptographically authenticated", result["source_identity_verification"])

    def test_matches_only_exact_normalized_alleles_on_the_declared_assembly(self):
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-reconcile-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"
            reference.write_text(">1\nATATAT\n", encoding="ascii")
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t2\t.\tT\tC\t.\tPASS\t.\n"
                "1\t4\t.\tT\tG\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            source = clinvar_record("12345", [{"assembly": "GRCh38", "contig": "1", "position_1_based": 2, "reference_allele": "T", "alternate_allele": "C"}])
            malformed_source = clinvar_record("67890", [{"assembly": "GRCh38", "contig": "1", "position_1_based": 2, "reference_allele": "A", "alternate_allele": "C"}])
            result = reconcile_clinvar_variants(report, [source, malformed_source], reference, "GRCh38")
            wrong_assembly = clinvar_record("12345", [{**source["assembly_locations"][0], "assembly": "GRCh37"}])
            mismatched_build_result = reconcile_clinvar_variants(report, [wrong_assembly], reference, "GRCh38")

        self.assertEqual(result["variants"][0]["status"], "exact_normalized_allele_match")
        self.assertEqual(result["variants"][0]["clinvar_records"][0]["variation_id"], "12345")
        self.assertEqual(result["variants"][1]["status"], "no_exact_match")
        self.assertEqual(result["unresolved_clinvar_locations"][0]["reason"], "ClinVar reference allele does not match the supplied FASTA")
        self.assertEqual(mismatched_build_result["variants"][0]["status"], "no_exact_match")

    def test_rejects_build_or_reference_digest_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-reconcile-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"
            reference.write_text(">1\nATATAT\n", encoding="ascii")
            vcf.write_text("##fileformat=VCFv4.3\n##reference=GRCh38\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n1\t2\t.\tT\tC\t.\tPASS\t.\n", encoding="ascii")
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            with self.assertRaisesRegex(ValueError, "build"):
                reconcile_clinvar_variants(report, [], reference, "GRCh37")
            reference.write_text(">1\nAAAAAA\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "digest"):
                reconcile_clinvar_variants(report, [], reference, "GRCh38")

    def test_requires_consistent_clinvar_identity_and_location_digest(self):
        with tempfile.TemporaryDirectory(prefix="medai-clinvar-source-identity-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "variants.vcf"
            reference.write_text(">1\nATATAT\n", encoding="ascii")
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t2\t.\tT\tC\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            locations = [{"assembly": "GRCh38", "contig": "1", "position_1_based": 2, "reference_allele": "T", "alternate_allele": "C"}]
            tampered = clinvar_record("76543", locations)
            tampered["assembly_locations"] = [{**locations[0], "alternate_allele": "G"}]
            result = reconcile_clinvar_variants(report, [tampered], reference, "GRCh38")
            bad_url = clinvar_record("87654", locations)
            bad_url["source_url"] = "https://example.invalid/variation/87654/"
            bad_identity = reconcile_clinvar_variants(report, [bad_url], reference, "GRCh38")

        self.assertEqual(result["variants"][0]["status"], "no_exact_match")
        self.assertEqual(result["unresolved_clinvar_locations"][0]["reason"], "ClinVar record source, accession, URL, or location digest is missing or inconsistent")
        self.assertEqual(bad_identity["unresolved_clinvar_locations"][0]["variation_id"], "87654")


if __name__ == "__main__":
    unittest.main()
