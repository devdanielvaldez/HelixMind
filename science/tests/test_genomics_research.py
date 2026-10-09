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
        "analysis_implementation_sha256": {"__init__.py": "b" * 64, "genomics.py": "b" * 64, "transcripts.py": "b" * 64},
        "reference_validated": True,
        "reference_fasta_sha256": "c" * 64,
        "records_parsed": 1,
        "alternate_alleles_emitted": 1,
        "unsupported_symbolic_or_breakend_alleles": 0,
        "variants": [{
            "record_number": 1,
            "alternate_index": 1,
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

    def test_rejects_inconsistent_reference_state_and_unexpected_implementation_files(self):
        malformed = report()
        malformed["reference_validated"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent reference provenance"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)
        malformed = report()
        malformed["analysis_implementation_sha256"]["arbitrary-file.txt"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "exact local analysis implementation files"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)

    def test_rejects_inconsistent_record_alternate_and_variant_class_metadata(self):
        malformed = report()
        malformed["variants"][0]["alternate_index"] = 2
        with self.assertRaisesRegex(ValueError, "alternate indexes"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)
        malformed = report()
        malformed["variants"][0]["sequence_allele_supported"] = False
        with self.assertRaisesRegex(ValueError, "sequence support and variant class"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)
        malformed = report()
        malformed["unsupported_symbolic_or_breakend_alleles"] = 1
        with self.assertRaisesRegex(ValueError, "unsupported-allele count"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)

    def test_rejects_unbounded_or_malformed_transcript_annotations(self):
        malformed = report()
        malformed["variants"][0]["transcript_consequences"][0]["gene_id"] = "G" * 129
        with self.assertRaisesRegex(ValueError, "gene identifier"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)
        malformed = report()
        malformed["variants"][0]["transcript_consequences"][0]["consequences"] = ["missense variant"]
        with self.assertRaisesRegex(ValueError, "consequence terms"):
            summarize_local_vcf_report(malformed, expected_source_id="a" * 64)

    def test_worker_receipt_rejects_raw_variant_and_genotype_records(self):
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        summary["variants"] = [{"contig": "1", "position_1_based": 10, "reference_allele": "A", "alternate_allele": "G"}]
        with self.assertRaisesRegex(ValueError, "must not contain raw variant"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)

    def test_worker_receipt_rejects_extra_fields_and_inconsistent_aggregate_counts(self):
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        summary["sample_name"] = "SAMPLE-PRIVATE"
        with self.assertRaisesRegex(ValueError, "exact privacy-minimized schema"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        summary["validation_counts"]["reference_unvalidated"] = 1
        with self.assertRaisesRegex(ValueError, "internally inconsistent"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)
        summary = summarize_local_vcf_report(report(), expected_source_id="a" * 64)
        summary["interpretation_limit"] = "This sample has a harmful variant."
        with self.assertRaisesRegex(ValueError, "interpretation limit"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)

    def test_worker_receipt_rejects_reference_digest_without_reference_validation(self):
        local = report()
        local["reference_validated"] = False
        local["reference_fasta_sha256"] = None
        local["variants"][0].pop("reference_validated")
        summary = summarize_local_vcf_report(local, expected_source_id="a" * 64)
        summary["reference_fasta_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "reference provenance"):
            validate_local_vcf_summary(summary, expected_source_id="a" * 64)


if __name__ == "__main__":
    unittest.main()
