import unittest

from medai_science.genomics_research import summarize_local_vcf_report, validate_local_vcf_summary


def report():
    digest = "a" * 64
    return {
        "analysis": "local VCF parsing and validation",
        "input_sha256": digest,
        "genome_build": "GRCh38",
        "genome_build_source": "explicit_user_selection",
        "sample_identifiers_retained": False,
        "remote_services_contacted": False,
        "clinical_interpretation": "not_performed",
        "analysis_implementation_sha256": {"genomics.py": "b" * 64},
        "reference_validated": True,
        "reference_fasta_sha256": "c" * 64,
        "records_parsed": 1,
        "alternate_alleles_emitted": 1,
        "variants": [{
            "record_number": 1,
            "variant_class": "SNV",
            "sequence_allele_supported": True,
            "reference_validated": True,
            "genotype_call_counts": {"samples": 2},
            "transcript_consequences": [{"gene_id": "ENSG000001", "consequences": ["missense_variant"]}],
        }],
    }


class GenomicsResearchTests(unittest.TestCase):
    def test_summarizes_valid_local_report_without_copying_variant_or_genotype_records(self):
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        self.assertEqual(summary["genome_build"], "GRCh38")
        self.assertEqual(summary["variant_class_counts"], {"SNV": 1})
        self.assertEqual(summary["local_transcript_consequence_counts"], {"missense_variant": 1})
        self.assertEqual(summary["local_annotated_gene_identifier_count"], 1)
        self.assertFalse(summary["raw_variant_records_retained_in_summary"])
        self.assertNotIn("genotype_call_counts", summary)
        validated = validate_local_vcf_summary(summary, expected_source_id="a" * 64)
        self.assertEqual(validated, summary)

    def test_rejects_forged_identity_missing_privacy_attestation_and_inconsistent_counts(self):
        with self.assertRaisesRegex(ValueError, "source ID"):
            summarize_local_vcf_report(report(), expected_source_id="d" * 64)
        malformed = report()
        malformed["sample_identifiers_retained"] = True
        with self.assertRaisesRegex(ValueError, "omit sample identifiers"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)
        malformed = report()
        malformed["alternate_alleles_emitted"] = 0
        with self.assertRaisesRegex(ValueError, "count does not match"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)

    def test_worker_receipt_rejects_raw_variant_and_genotype_records(self):
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        summary["variants"] = [{"contig": "1", "position_1_based": 10, "reference_allele": "A", "alternate_allele": "G"}]
        with self.assertRaisesRegex(ValueError, "must not contain raw variant"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)


if __name__ == "__main__":
    unittest.main()
