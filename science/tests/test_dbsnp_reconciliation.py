import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from medai_science.dbsnp_reconciliation import reconcile_dbsnp_placements
from medai_science.genomics import analyze_vcf


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DbSNPReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="medai-dbsnp-reconcile-")
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference.fa"
        self.vcf = self.root / "variants.vcf"
        self.reference.write_text(">1\nCAAAAG\n", encoding="ascii")
        self.vcf.write_text(
            "##fileformat=VCFv4.3\n##reference=GRCh38\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t3\t.\tA\tAA\t.\tPASS\t.\n"
            "1\t6\t.\tG\tT\t.\tPASS\t.\n",
            encoding="ascii",
        )
        self.report = analyze_vcf(self.vcf, "GRCh38", 10, self.reference)
        self.record = {
            "refsnp_id": 328,
            "primary_snapshot_data": {
                "placements_with_allele": [
                    {
                        "seq_id": "NC_000001.11",
                        "is_ptlp": True,
                        "placement_annot": {"seq_id_traits_by_assembly": [{"assembly_name": "GRCh38.p14", "assembly_accession": "GCF_000001405.40"}]},
                        "alleles": [
                            {"allele": {"spdi": {"seq_id": "NC_000001.11", "position": 1, "deleted_sequence": "A", "inserted_sequence": "A"}}},
                            {"allele": {"spdi": {"seq_id": "NC_000001.11", "position": 1, "deleted_sequence": "", "inserted_sequence": "A"}}},
                            {"allele": {"spdi": {"seq_id": "NC_000001.11", "position": 5, "deleted_sequence": "G", "inserted_sequence": "T"}}},
                        ],
                    }
                ]
            },
        }
        self.receipt = {
            "rsid": "rs328",
            "source": "NCBI dbSNP Variation Services",
            "source_url": "https://www.ncbi.nlm.nih.gov/snp/328",
            "api_endpoint": "https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/328",
            "source_record_sha256": "a" * 64,
            "source_record_canonical_sha256": canonical_digest(self.record),
            "source_record": self.record,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def reconcile(self, receipt=None, build="GRCh38", sequence_map=None):
        return reconcile_dbsnp_placements(
            self.report, receipt or self.receipt, self.reference, build,
            sequence_map or {"NC_000001.11": "1"},
        )

    def test_joins_reference_checked_snv_and_repeat_insertion_after_normalization(self):
        result = self.reconcile()
        rows = result["variants"]
        self.assertEqual([row["status"] for row in rows], ["exact_normalized_allele_match", "exact_normalized_allele_match"])
        self.assertEqual(rows[0]["dbsnp_records"][0]["normalized_allele"], {
            "contig": "1", "position_1_based": 1, "reference_allele": "C", "alternate_allele": "CA",
        })
        self.assertEqual(rows[0]["dbsnp_records"][0]["assembly_accessions"], ["GCF_000001405.40"])
        self.assertEqual(rows[1]["dbsnp_records"][0]["spdi"]["position"], 5)
        self.assertIn("do not establish pathogenicity", result["interpretation_limit"])

    def test_requires_digest_bound_record_build_and_explicit_accession_mapping(self):
        with self.assertRaisesRegex(ValueError, "canonical record digest"):
            self.reconcile({**self.receipt, "source_record_canonical_sha256": "b" * 64})
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.reconcile(build="GRCh37")
        result = self.reconcile(sequence_map={"NC_000002.12": "1"})
        self.assertEqual(result["matched_allele_count"], 0)
        self.assertTrue(any("no explicit FASTA contig mapping" in item["reason"] for item in result["unresolved_source_placements"]))

    def test_assembly_patch_match_requires_an_exact_grch_patch_form(self):
        for assembly_name in ("GRCh38.draft", "GRCh38.p14.preview", "GRCh380.p14", "GRCh38.p0"):
            with self.subTest(assembly_name=assembly_name):
                altered = json.loads(json.dumps(self.receipt))
                traits = altered["source_record"]["primary_snapshot_data"]["placements_with_allele"][0]["placement_annot"]["seq_id_traits_by_assembly"]
                traits[0]["assembly_name"] = assembly_name
                altered["source_record_canonical_sha256"] = canonical_digest(altered["source_record"])
                result = self.reconcile(altered)
                self.assertEqual(result["compatible_placement_count"], 0)
                self.assertEqual(result["matched_allele_count"], 0)

        unpatched = json.loads(json.dumps(self.receipt))
        traits = unpatched["source_record"]["primary_snapshot_data"]["placements_with_allele"][0]["placement_annot"]["seq_id_traits_by_assembly"]
        traits[0]["assembly_name"] = "GRCh38"
        unpatched["source_record_canonical_sha256"] = canonical_digest(unpatched["source_record"])
        self.assertEqual(self.reconcile(unpatched)["matched_allele_count"], 2)

    def test_rejects_reference_mismatch_without_fabricating_a_match(self):
        bad = json.loads(json.dumps(self.receipt))
        bad["source_record"]["primary_snapshot_data"]["placements_with_allele"][0]["alleles"][2]["allele"]["spdi"]["deleted_sequence"] = "A"
        bad["source_record_canonical_sha256"] = canonical_digest(bad["source_record"])
        result = self.reconcile(bad)
        self.assertEqual(result["variants"][1]["status"], "no_exact_match")
        self.assertTrue(any("does not match the supplied reference FASTA" in item["reason"] for item in result["unresolved_source_placements"]))

    def test_rechecks_reported_local_allele_normalization_before_joining(self):
        tampered = json.loads(json.dumps(self.report))
        tampered["variants"][0]["normalized_position_1_based"] = 2
        tampered["variants"][0]["normalized_reference_allele"] = "A"
        tampered["variants"][0]["normalized_alternate_allele"] = "AA"
        with self.assertRaisesRegex(ValueError, "not minimally normalized"):
            self.reconcile_report(tampered)

    def test_rejects_duplicate_local_record_alternate_identity(self):
        tampered = json.loads(json.dumps(self.report))
        tampered["variants"][1]["record_number"] = tampered["variants"][0]["record_number"]
        tampered["variants"][1]["alternate_index"] = tampered["variants"][0]["alternate_index"]
        with self.assertRaisesRegex(ValueError, "duplicate record/alternate identities"):
            self.reconcile_report(tampered)

    def test_rejects_aggregate_source_alleles_above_reconciliation_bound(self):
        altered = json.loads(json.dumps(self.receipt))
        placement = altered["source_record"]["primary_snapshot_data"]["placements_with_allele"][0]
        placement["alleles"] = [placement["alleles"][0]] * 256
        altered["source_record"]["primary_snapshot_data"]["placements_with_allele"] = [placement] * 16
        altered["source_record_canonical_sha256"] = canonical_digest(altered["source_record"])
        with self.assertRaisesRegex(ValueError, "4000-allele reconciliation limit"):
            self.reconcile(altered)

    def test_rejects_noncanonical_assembly_accessions_as_unresolved(self):
        altered = json.loads(json.dumps(self.receipt))
        traits = altered["source_record"]["primary_snapshot_data"]["placements_with_allele"][0]["placement_annot"]["seq_id_traits_by_assembly"]
        traits[0]["assembly_accession"] = "attacker-controlled-" + ("x" * 100_000)
        altered["source_record_canonical_sha256"] = canonical_digest(altered["source_record"])
        result = self.reconcile(altered)
        self.assertEqual(result["matched_allele_count"], 0)
        self.assertTrue(any("assembly accessions are missing, malformed" in item["reason"] for item in result["unresolved_source_placements"]))

    def reconcile_report(self, report):
        return reconcile_dbsnp_placements(
            report, self.receipt, self.reference, "GRCh38", {"NC_000001.11": "1"},
        )


if __name__ == "__main__":
    unittest.main()
