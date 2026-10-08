import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medai_science.clinvar_reconciliation import reconcile_clinvar_variants
from medai_science.genomics import analyze_vcf
from medai_science.variant_source_reconciliation import reconcile_gnomad_frequency, reconcile_variant_sources


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class VariantSourceReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.reference = root / "reference.fa"
        self.vcf = root / "variants.vcf"
        self.reference.write_text(">1\nATATAT\n", encoding="ascii")
        self.vcf.write_text(
            "##fileformat=VCFv4.3\n##reference=GRCh38\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t2\t.\tT\tC\t.\tPASS\t.\n",
            encoding="ascii",
        )
        self.report = analyze_vcf(self.vcf, "GRCh38", 10, self.reference)
        location = [{"assembly": "GRCh38", "contig": "1", "position_1_based": 2, "reference_allele": "T", "alternate_allele": "C"}]
        self.clinvar = reconcile_clinvar_variants(self.report, [{
            "variation_id": "123", "accession": "VCV000000123.1",
            "source": "NCBI ClinVar E-utilities", "source_url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/123/",
            "source_record_sha256": "a" * 64,
            "location_source_sha256": hashlib.sha256(json.dumps(location, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "assembly_locations": location,
        }], self.reference, "GRCh38")
        self.vep = {
            "source": "Ensembl Variant Effect Predictor REST API",
            "source_url": "https://rest.ensembl.org/vep/homo_sapiens/region",
            "assembly": "GRCh38", "reference_fasta_sha256": self.report["reference_fasta_sha256"],
            "input_report_sha256": digest(self.report),
            "results": [{
                "local_record_number": 1, "local_alternate_index": 1, "local_contig": "1", "local_position_1_based": 2,
                "local_reference_allele": "T", "local_alternate_allele": "C",
                "source_record_sha256": "b" * 64,
                "most_severe_consequence": "missense_variant",
                "transcript_consequences": [{"transcript_id": "ENST00000000001", "consequence_terms": ["missense_variant"]}],
            }],
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_joins_source_records_by_exact_local_allele_and_preserves_sources(self):
        result = reconcile_variant_sources(self.report, self.clinvar, self.vep)
        row = result["variants"][0]
        self.assertEqual(row["status"], "matched_both_sources")
        self.assertEqual(row["clinvar"]["records"][0]["accession"], "VCV000000123.1")
        self.assertEqual(row["ensembl_vep"]["most_severe_consequence"], "missense_variant")
        self.assertIn("does not adjudicate source conflicts", result["interpretation_limit"])

    def test_joins_gnomad_frequency_only_to_exact_reference_normalized_grch38_allele(self):
        receipt = {
            "variant_id": "1-2-T-C", "assembly": "GRCh38", "dataset": "gnomAD v4 (gnomad_r4)",
            "genome_frequency": {"ac": 10, "an": 1000, "af": 0.01}, "exome_frequency": {"ac": 20, "an": 2000, "af": 0.01}, "found": True,
            "source": "gnomAD Browser GraphQL API", "source_url": "https://gnomad.broadinstitute.org/",
            "source_record_sha256": digest({"variant": {"variant_id": "1-2-T-C", "genome": {"ac": 10, "an": 1000, "af": 0.01}, "exome": {"ac": 20, "an": 2000, "af": 0.01}}}),
            "retrieved_at": "2026-10-08T12:00:00Z",
        }
        result = reconcile_gnomad_frequency(self.report, [receipt])
        row = result["variants"][0]
        self.assertEqual(row["variant_id"], "1-2-T-C")
        self.assertEqual(row["gnomad"]["genome_frequency"]["af"], 0.01)
        self.assertEqual(row["gnomad"]["exome_frequency"]["af"], 0.01)
        self.assertIn("do not establish", result["interpretation_limit"])
        with self.assertRaisesRegex(ValueError, "exact local normalized allele"):
            reconcile_gnomad_frequency(self.report, [{**receipt, "variant_id": "1-3-T-C"}])
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            reconcile_gnomad_frequency(self.report, [{**receipt, "source_record_sha256": "f" * 64}])

    def test_rejects_gnomad_count_frequency_disagreement_and_wrong_build(self):
        receipt = {
            "variant_id": "1-2-T-C", "assembly": "GRCh38", "dataset": "gnomAD v4 (gnomad_r4)",
            "genome_frequency": {"ac": 10, "an": 1000, "af": 0.5}, "exome_frequency": None, "found": True,
            "source": "gnomAD Browser GraphQL API", "source_url": "https://gnomad.broadinstitute.org/",
            "source_record_sha256": digest({"variant": {"variant_id": "1-2-T-C", "genome": {"ac": 10, "an": 1000, "af": 0.5}, "exome": None}}),
            "retrieved_at": "2026-10-08T12:00:00Z",
        }
        with self.assertRaisesRegex(ValueError, "does not agree"):
            reconcile_gnomad_frequency(self.report, [receipt])
        with self.assertRaisesRegex(ValueError, "require a GRCh38"):
            reconcile_gnomad_frequency({**self.report, "genome_build": "GRCh37"}, [])

    def test_reconciliation_preserves_undefined_frequency_when_allele_number_is_zero(self):
        genome = {"ac": 0, "an": 0, "af": None}
        exome = {"ac": 2, "an": 100, "af": 0.02}
        receipt = {
            "variant_id": "1-2-T-C", "assembly": "GRCh38", "dataset": "gnomAD v4 (gnomad_r4)",
            "genome_frequency": genome, "exome_frequency": exome, "found": True,
            "source": "gnomAD Browser GraphQL API", "source_url": "https://gnomad.broadinstitute.org/",
            "source_record_sha256": digest({"variant": {"variant_id": "1-2-T-C", "genome": genome, "exome": exome}}),
            "retrieved_at": "2026-10-08T12:00:00Z",
        }
        result = reconcile_gnomad_frequency(self.report, [receipt])
        row = result["variants"][0]["gnomad"]
        self.assertIsNone(row["genome_frequency"]["af"])
        self.assertEqual(row["exome_frequency"]["af"], 0.02)

    def test_rejects_stale_reference_or_local_report_binding(self):
        with self.assertRaisesRegex(ValueError, "reference and VCF"):
            reconcile_variant_sources(self.report, {**self.clinvar, "reference_fasta_sha256": "0" * 64}, self.vep)
        with self.assertRaisesRegex(ValueError, "exact local report"):
            reconcile_variant_sources(self.report, self.clinvar, {**self.vep, "input_report_sha256": "0" * 64})

    def test_rejects_source_allele_mismatch_instead_of_creating_a_join(self):
        changed = {**self.vep, "results": [{**self.vep["results"][0], "local_alternate_allele": "G"}]}
        with self.assertRaisesRegex(ValueError, "allele identity"):
            reconcile_variant_sources(self.report, self.clinvar, changed)

    def test_rejects_inconsistent_clinvar_source_identity(self):
        row = self.clinvar["variants"][0]
        bad_record = {**row["clinvar_records"][0], "source_url": "https://example.invalid/"}
        bad_rows = [{**row, "clinvar_records": [bad_record]}]
        with self.assertRaisesRegex(ValueError, "source identity"):
            reconcile_variant_sources(self.report, {**self.clinvar, "variants": bad_rows}, self.vep)

    def test_multiallelic_site_keeps_each_alternate_allele_separate(self):
        self.vcf.write_text(
            "##fileformat=VCFv4.3\n##reference=GRCh38\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t2\t.\tT\tC,G\t.\tPASS\t.\n",
            encoding="ascii",
        )
        report = analyze_vcf(self.vcf, "GRCh38", 10, self.reference)
        location = [{"assembly": "GRCh38", "contig": "1", "position_1_based": 2, "reference_allele": "T", "alternate_allele": "C"}]
        clinvar = reconcile_clinvar_variants(report, [{
            "variation_id": "123", "accession": "VCV000000123.1",
            "source": "NCBI ClinVar E-utilities", "source_url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/123/",
            "source_record_sha256": "a" * 64,
            "location_source_sha256": hashlib.sha256(json.dumps(location, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "assembly_locations": location,
        }], self.reference, "GRCh38")
        vep = {
            "source": "Ensembl Variant Effect Predictor REST API",
            "source_url": "https://rest.ensembl.org/vep/homo_sapiens/region",
            "assembly": "GRCh38", "reference_fasta_sha256": report["reference_fasta_sha256"],
            "input_report_sha256": digest(report),
            "results": [
                {"local_record_number": 1, "local_alternate_index": index,
                 "local_contig": "1", "local_position_1_based": 2, "local_reference_allele": "T",
                 "local_alternate_allele": alternate, "source_record_sha256": str(index) * 64,
                 "most_severe_consequence": consequence, "transcript_consequences": []}
                for index, (alternate, consequence) in enumerate((("C", "missense_variant"), ("G", "synonymous_variant")), 1)
            ],
        }
        result = reconcile_variant_sources(report, clinvar, vep)
        self.assertEqual([row["alternate_index"] for row in result["variants"]], [1, 2])
        self.assertEqual([row["status"] for row in result["variants"]], ["matched_both_sources", "matched_vep_only"])


if __name__ == "__main__":
    unittest.main()
