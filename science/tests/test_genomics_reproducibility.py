from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from medai_science.genomics import analyze_vcf
from medai_science.reproducibility import create_genomics_bundle, verify_genomics_bundle
from medai_science.sources.ensembl_vep import EnsemblVEPClient
from medai_science.genomics import IndexedFasta
from medai_science.transcripts import annotate_coding_variants


def _vep_receipt_for_local_report(report: dict) -> dict:
    report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    rows = []
    for variant in report["variants"]:
        if variant.get("sequence_allele_supported") is not True or variant.get("reference_validated") is not True:
            continue
        contig = variant["contig"].removeprefix("chr")
        position = variant.get("normalized_position_1_based", variant["position_1_based"])
        ref = variant.get("normalized_reference_allele", variant["reference_allele"])
        alt = variant.get("normalized_alternate_allele", variant["alternate_allele"])
        rows.append({
            "input": f"{contig} {position} . {ref} {alt} . . .",
            "local_record_number": variant["record_number"],
            "local_alternate_index": variant["alternate_index"],
            "local_contig": variant["contig"],
            "local_position_1_based": position,
            "local_reference_allele": ref,
            "local_alternate_allele": alt,
            "vep_assembly": report["genome_build"],
            "seq_region_name": contig,
            "transcript_consequences": [],
            "source_record_sha256": "c" * 64,
        })
    endpoint = EnsemblVEPClient.endpoints[report["genome_build"]]
    return {
        "source": "Ensembl Variant Effect Predictor REST API",
        "source_url": endpoint + "/vep/homo_sapiens/region",
        "assembly": report["genome_build"],
        "input_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "reference_fasta_sha256": report["reference_fasta_sha256"],
        "queried_variant_alleles": len(rows),
        "skipped_unsupported_or_unvalidated": len(report["variants"]) - len(rows),
        "results": rows,
    }


def _ensembl_vep_reference_fixture(directory: Path) -> tuple[Path, Path, Path]:
    fixture = json.loads((Path(__file__).parent / "fixtures/ensembl_vep_negative_strand.json").read_text(encoding="utf-8"))
    codons = ["GCT"] * 268
    codons[112] = "TTG"  # Ensembl VEP: c.337, p.Leu113
    codons[267] = "ATG"  # Ensembl VEP: c.803, p.Met268
    cds = "".join(codons)
    complement = str.maketrans("ACGT", "TGCA")
    genomic = cds.translate(complement)[::-1]
    reference = directory / "vep-reference.fa"
    annotation = directory / "vep-transcript.gff3"
    vcf = directory / "vep-cases.vcf"
    reference.write_text(f">1\n{genomic}\n", encoding="ascii")
    annotation.write_text(
        "##gff-version 3\n"
        "##genome-build GRCh38\n"
        f"1\tfixture\tgene\t1\t{len(genomic)}\t.\t-\t.\tID={fixture['gene_id']};Name=AGT\n"
        f"1\tfixture\tmRNA\t1\t{len(genomic)}\t.\t-\t.\tID={fixture['transcript_id']};Parent={fixture['gene_id']}\n"
        f"1\tfixture\tCDS\t1\t{len(genomic)}\t.\t-\t0\tParent={fixture['transcript_id']}\n",
        encoding="ascii",
    )
    alleles_by_site: dict[int, dict[str, object]] = {}
    for case in fixture["cases"]:
        site = int(case["local_fixture_position"])
        entry = alleles_by_site.setdefault(site, {"identifier": case["name"].split("_alt_", 1)[0], "reference": case["local_reference"], "alternates": []})
        if entry["reference"] != case["local_reference"]:
            raise AssertionError("fixture alleles at one site must share a reference allele")
        entry["alternates"].append(case["local_alternate"])
    variants = "".join(
        f"1\t{site}\t{entry['identifier']}\t{entry['reference']}\t{','.join(entry['alternates'])}\t.\tPASS\t.\n"
        for site, entry in sorted(alleles_by_site.items())
    )
    vcf.write_text(
        "##fileformat=VCFv4.3\n##reference=GRCh38\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + variants,
        encoding="ascii",
    )
    return vcf, reference, annotation


def _inputs(directory: Path) -> tuple[Path, Path, Path]:
    reference = directory / "reference.fa"
    annotation = directory / "transcripts.gff3"
    vcf = directory / "cohort.vcf"
    reference.write_text(">plus\nATGGAATTT\n>minus\nAAATTCCAT\n", encoding="ascii")
    annotation.write_text(
        "##gff-version 3\n"
        "##genome-build GRCh38\n"
        "plus\ttest\tgene\t1\t9\t.\t+\t.\tID=g1;Name=GENE1\n"
        "plus\ttest\tmRNA\t1\t9\t.\t+\t.\tID=t1;Parent=g1\n"
        "plus\ttest\tCDS\t1\t9\t.\t+\t0\tParent=t1\n"
        "minus\ttest\tgene\t1\t9\t.\t-\t.\tID=g2;Name=GENE2\n"
        "minus\ttest\tmRNA\t1\t9\t.\t-\t.\tID=t2;Parent=g2\n"
        "minus\ttest\tCDS\t1\t9\t.\t-\t0\tParent=t2\n",
        encoding="ascii",
    )
    vcf.write_text(
        "##fileformat=VCFv4.3\n"
        "##reference=GRCh38\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "plus\t6\t.\tA\tG\t.\tPASS\t.\n"
        "minus\t4\t.\tT\tC\t.\tPASS\t.\n",
        encoding="ascii",
    )
    return vcf, reference, annotation


class GenomicsReproducibilityTests(unittest.TestCase):
    def test_frameshift_matches_independent_ensembl_vep_documentation_fixture(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/ensembl_vep_frameshift.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="medai-vep-frameshift-") as temp:
            root = Path(temp)
            prefix = "ATG" + "GCT" * 312
            cds = prefix + "AGC" + "GCT" * 20
            reference = root / "reference.fa"
            annotation = root / "transcripts.gff3"
            vcf = root / "case.vcf"
            reference.write_text(f">3\n{cds}\n", encoding="ascii")
            annotation.write_text(
                "##gff-version 3\n##genome-build GRCh38\n"
                f"3\tfixture\tgene\t1\t{len(cds)}\t.\t+\t.\tID={fixture['gene_id']}\n"
                f"3\tfixture\tmRNA\t1\t{len(cds)}\t.\t+\t.\tID={fixture['transcript_id']};Parent={fixture['gene_id']}\n"
                f"3\tfixture\tCDS\t1\t{len(cds)}\t.\t+\t0\tParent={fixture['transcript_id']}\n",
                encoding="ascii",
            )
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "3\t940\tvep_frameshift\tA\tAG\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference, annotation)
        consequence = report["variants"][0]["transcript_consequences"][0]
        self.assertEqual(consequence["transcript_id"], fixture["transcript_id"])
        self.assertEqual(consequence["gene_id"], fixture["gene_id"])
        self.assertEqual(consequence["strand"], "+" if fixture["strand"] == 1 else "-")
        self.assertEqual(consequence["consequences"], [fixture["source_consequences"][0]])
        self.assertEqual(consequence["cds_position"], "940-940")
        self.assertEqual(consequence["protein_position"], fixture["source_protein_position"])
        self.assertEqual(consequence["reference_amino_acids"] + "/" + consequence["alternate_amino_acids"] + "X", fixture["source_amino_acids"])

    def test_vep_cross_validation_receipt_is_bound_to_replayable_local_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-vep-bundle-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            vep_receipt = _vep_receipt_for_local_report(report)
            bundle = root / "with-vep"
            created = create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation, vep_receipt)
            self.assertEqual(created["artifact_count"], 10)
            checked = verify_genomics_bundle(bundle, replay=True)
            self.assertTrue(checked["external_vep_receipt_verified"])
            self.assertFalse(checked["external_source_requeried"])

            wrong_binding = {**vep_receipt, "input_report_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "does not bind the exact local report"):
                create_genomics_bundle(root / "invalid", report, vcf, "GRCh38", 100, reference, annotation, wrong_binding)
            wrong_allele = json.loads(json.dumps(vep_receipt))
            wrong_allele["results"][0]["local_position_1_based"] += 1
            with self.assertRaisesRegex(ValueError, "local allele metadata"):
                create_genomics_bundle(root / "wrong-allele", report, vcf, "GRCh38", 100, reference, annotation, wrong_allele)

    def test_clinvar_vep_cross_source_reconciliation_is_recomputed_during_bundle_replay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-cross-source-bundle-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            vep_receipt = _vep_receipt_for_local_report(report)
            locations = [{"assembly": "GRCh38", "contig": "plus", "position_1_based": 6, "reference_allele": "A", "alternate_allele": "G"}]
            clinvar_record = {
                "variation_id": "123", "accession": "VCV000000123.1",
                "source": "NCBI ClinVar E-utilities", "source_url": "https://www.ncbi.nlm.nih.gov/clinvar/variation/123/",
                "source_record_sha256": "a" * 64,
                "location_source_sha256": hashlib.sha256(json.dumps(locations, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "assembly_locations": locations,
            }
            bundle = root / "with-cross-source"
            created = create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation, vep_receipt, [clinvar_record])
            self.assertTrue(created["cross_source_reconciliation_captured"])
            checked = verify_genomics_bundle(bundle, replay=True)
            self.assertTrue(checked["external_clinvar_receipt_verified"])
            self.assertTrue(checked["cross_source_reconciliation_verified"])
            self.assertFalse(checked["external_source_requeried"])

    def test_gnomad_frequency_receipt_is_reconciled_and_replayed_offline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-gnomad-bundle-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            vcf = root / "cohort.vcf"
            reference.write_text(">1\nATATAT\n", encoding="ascii")
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "1\t2\t.\tT\tC\t.\tPASS\t.\n",
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", 10, reference)
            frequency = {"ac": 10, "an": 1000, "af": 0.01}
            source_row = {"variant_id": "1-2-T-C", "genome": frequency, "exome": None}
            source_digest = hashlib.sha256(json.dumps({"variant": source_row}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            gnomad_receipt = {
                "variant_id": "1-2-T-C", "assembly": "GRCh38", "dataset": "gnomAD v4 (gnomad_r4)",
                "genome_frequency": frequency, "exome_frequency": None, "found": True, "source": "gnomAD Browser GraphQL API",
                "source_url": "https://gnomad.broadinstitute.org/", "source_record_sha256": source_digest,
                "retrieved_at": "2026-10-08T12:00:00Z",
            }
            bundle = root / "with-gnomad-frequency"
            created = create_genomics_bundle(bundle, report, vcf, "GRCh38", 10, reference, gnomad_frequency_receipts=[gnomad_receipt])
            self.assertTrue(created["external_gnomad_frequency_captured"])
            checked = verify_genomics_bundle(bundle, replay=True)
            self.assertTrue(checked["external_gnomad_frequency_verified"])
            self.assertFalse(checked["external_source_requeried"])

    def test_positive_strand_coding_consequence_classes_on_a_synthetic_transcript(self) -> None:
        cases = [
            ("start_lost", "1", "A", "C", "start_lost", "M", "L"),
            ("stop_gained", "4", "G", "T", "stop_gained", "E", "*"),
            ("frameshift", "3", "G", "GA", "frameshift_variant", "E", "R"),
            ("inframe_insertion", "3", "G", "GCCC", "inframe_insertion", "E", "P"),
            ("inframe_deletion", "3", "GGAA", "G", "inframe_deletion", "E", "F"),
            ("missense", "4", "G", "A", "missense_variant", "E", "K"),
            ("synonymous", "6", "A", "G", "synonymous_variant", "E", "E"),
        ]
        with tempfile.TemporaryDirectory(prefix="medai-consequence-classes-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            annotation = root / "transcripts.gff3"
            vcf = root / "cases.vcf"
            reference.write_text(">1\nATGGAATTT\n", encoding="ascii")
            annotation.write_text(
                "##gff-version 3\n##genome-build GRCh38\n"
                "1\ttest\tgene\t1\t9\t.\t+\t.\tID=g1;Name=GENE1\n"
                "1\ttest\tmRNA\t1\t9\t.\t+\t.\tID=t1;Parent=g1\n"
                "1\ttest\tCDS\t1\t9\t.\t+\t0\tParent=t1\n",
                encoding="ascii",
            )
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                + "".join(f"1\t{pos}\t{name}\t{ref}\t{alt}\t.\tPASS\t.\n" for name, pos, ref, alt, *_ in cases),
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", len(cases), reference, annotation)

        self.assertEqual(len(report["variants"]), len(cases))
        for case, row in zip(cases, report["variants"]):
            with self.subTest(name=case[0]):
                consequence = row["transcript_consequences"][0]
                self.assertEqual(consequence["consequences"], [case[4]])
                self.assertEqual(consequence["reference_amino_acids"], case[5])
                self.assertEqual(consequence["alternate_amino_acids"], case[6])
                self.assertTrue(consequence["protein_effect_evaluated"])

    def test_negative_strand_coding_consequence_classes_on_a_synthetic_transcript(self) -> None:
        cases = [
            ("start_lost", 1, "A", "C", "start_lost", "M", "L"),
            ("stop_gained", 4, "G", "T", "stop_gained", "E", "*"),
            ("frameshift", 3, "G", "GA", "frameshift_variant", "E", "R"),
            ("inframe_insertion", 3, "G", "GCCC", "inframe_insertion", "E", "P"),
            ("inframe_deletion", 3, "GGAA", "G", "inframe_deletion", "E", "F"),
            ("missense", 4, "G", "A", "missense_variant", "E", "K"),
            ("synonymous", 6, "A", "G", "synonymous_variant", "E", "E"),
        ]
        complement = str.maketrans("ACGT", "TGCA")

        def reverse_complement(sequence: str) -> str:
            return sequence.translate(complement)[::-1]

        coding_sequence = "ATGGAATTT"
        genomic_sequence = reverse_complement(coding_sequence)
        with tempfile.TemporaryDirectory(prefix="medai-negative-consequences-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            annotation = root / "transcripts.gff3"
            vcf = root / "cases.vcf"
            reference.write_text(f">1\n{genomic_sequence}\n", encoding="ascii")
            annotation.write_text(
                "##gff-version 3\n##genome-build GRCh38\n"
                "1\ttest\tgene\t1\t9\t.\t-\t.\tID=g1;Name=GENE1\n"
                "1\ttest\tmRNA\t1\t9\t.\t-\t.\tID=t1;Parent=g1\n"
                "1\ttest\tCDS\t1\t9\t.\t-\t0\tParent=t1\n",
                encoding="ascii",
            )
            rows = []
            for name, coding_position, coding_ref, coding_alt, *_ in cases:
                mutated_cds = (
                    coding_sequence[: coding_position - 1]
                    + coding_alt
                    + coding_sequence[coding_position - 1 + len(coding_ref) :]
                )
                mutated_genome = reverse_complement(mutated_cds)
                prefix = 0
                while prefix < min(len(genomic_sequence), len(mutated_genome)) and genomic_sequence[prefix] == mutated_genome[prefix]:
                    prefix += 1
                suffix = 0
                while (
                    suffix < min(len(genomic_sequence) - prefix, len(mutated_genome) - prefix)
                    and genomic_sequence[len(genomic_sequence) - suffix - 1]
                    == mutated_genome[len(mutated_genome) - suffix - 1]
                ):
                    suffix += 1
                reference_end = len(genomic_sequence) - suffix
                alternate_end = len(mutated_genome) - suffix
                if prefix:
                    variant_start = prefix - 1
                    reference_allele = genomic_sequence[variant_start:reference_end]
                    alternate_allele = mutated_genome[variant_start:alternate_end]
                else:
                    variant_start = prefix
                    reference_allele = genomic_sequence[prefix:reference_end] + genomic_sequence[reference_end]
                    alternate_allele = mutated_genome[prefix:alternate_end] + mutated_genome[alternate_end]
                rows.append(
                    f"1\t{variant_start + 1}\t{name}\t{reference_allele}\t"
                    f"{alternate_allele}\t.\tPASS\t.\n"
                )
            vcf.write_text(
                "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "".join(rows),
                encoding="ascii",
            )
            report = analyze_vcf(vcf, "GRCh38", len(cases), reference, annotation)

        self.assertEqual(len(report["variants"]), len(cases))
        for case, row in zip(cases, report["variants"]):
            with self.subTest(name=case[0]):
                consequence = row["transcript_consequences"][0]
                self.assertEqual(consequence["strand"], "-")
                self.assertEqual(consequence["consequences"], [case[4]])
                self.assertEqual(consequence["reference_amino_acids"], case[5])
                self.assertEqual(consequence["alternate_amino_acids"], case[6])
                self.assertTrue(consequence["protein_effect_evaluated"])

    def test_local_transcript_annotation_identifies_canonical_splice_sites_on_both_strands(self) -> None:
        cases = (
            ("+", 6, ["splice_donor_variant"]),
            ("+", 49, ["splice_acceptor_variant"]),
            ("-", 6, ["splice_acceptor_variant"]),
            ("-", 19, ["splice_donor_variant"]),
            ("+", 10, ["splice_region_variant"]),
            ("-", 30, ["splice_region_variant"]),
            ("+", 4, ["coding_plus_splice_region"]),
            ("-", 4, ["coding_plus_splice_region"]),
            ("+", 36, ["intron_variant"]),
            ("-", 36, ["intron_variant"]),
        )
        with tempfile.TemporaryDirectory(prefix="medai-splice-sites-") as temp:
            root = Path(temp)
            reference = root / "reference.fa"
            reference.write_text(">1\n" + "A" * 60 + "\n", encoding="ascii")
            for index, (strand, position, expected) in enumerate(cases):
                annotation = root / f"transcript-{index}.gff3"
                vcf = root / f"case-{index}.vcf"
                annotation.write_text(
                    "##gff-version 3\n##genome-build GRCh38\n"
                    f"1\ttest\tgene\t1\t60\t.\t{strand}\t.\tID=g{index};Name=GENE{index}\n"
                    f"1\ttest\tmRNA\t1\t60\t.\t{strand}\t.\tID=t{index};Parent=g{index}\n"
                    + "".join(
                        f"1\ttest\texon\t{start}\t{end}\t.\t{strand}\t.\tParent=t{index}\n"
                        for start, end in ((1, 5), (20, 25), (50, 60))
                    )
                    + "".join(
                        f"1\ttest\tCDS\t{start}\t{end}\t.\t{strand}\t{phase}\tParent=t{index}\n"
                        for start, end, phase in (
                            ((1, 5, 0), (20, 25, 1), (50, 60, 1))
                            if strand == "+"
                            else ((1, 5, 1), (20, 25, 1), (50, 60, 0))
                        )
                    ),
                    encoding="ascii",
                )
                vcf.write_text(
                    "##fileformat=VCFv4.3\n##reference=GRCh38\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                    f"1\t{position}\tsplice\tA\tG\t.\tPASS\t.\n",
                    encoding="ascii",
                )
                report = analyze_vcf(vcf, "GRCh38", 10, reference, annotation)
                consequence = report["variants"][0]["transcript_consequences"][0]
                with self.subTest(strand=strand, position=position):
                    if expected == ["coding_plus_splice_region"]:
                        self.assertIn("splice_region_variant", consequence["consequences"])
                        self.assertTrue(consequence["protein_effect_evaluated"])
                    else:
                        self.assertEqual(consequence["consequences"], expected)
                        self.assertFalse(consequence["protein_effect_evaluated"])

    def test_transcript_projection_does_not_infer_splicing_or_introns_from_cds_overlap_alone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-transcript-limits-") as temp:
            root = Path(temp)
            reference_path = root / "reference.fa"
            annotation = root / "transcripts.gff3"
            reference_path.write_text(">1\n" + "A" * 60 + "\n", encoding="ascii")
            annotation.write_text(
                "##gff-version 3\n##genome-build GRCh38\n"
                "1\ttest\tmRNA\t1\t60\t.\t+\t.\tID=protein_tx\n"
                "1\ttest\texon\t1\t60\t.\t+\t.\tParent=protein_tx\n"
                "1\ttest\tfive_prime_UTR\t1\t10\t.\t+\t.\tParent=protein_tx\n"
                "1\ttest\tCDS\t11\t60\t.\t+\t0\tParent=protein_tx\n"
                "1\ttest\tmRNA\t1\t60\t.\t+\t.\tID=unknown_structure_tx\n",
                encoding="ascii",
            )
            variants = [
                {"sequence_allele_supported": True, "reference_validated": True, "contig": "1", "position_1_based": 8, "reference_allele": "AAAAA", "alternate_allele": "A"},
                {"sequence_allele_supported": True, "reference_validated": True, "contig": "1", "position_1_based": 30, "reference_allele": "A", "alternate_allele": "G"},
                {"sequence_allele_supported": True, "reference_validated": True, "contig": "1", "position_1_based": 8, "reference_allele": "A", "alternate_allele": "G"},
            ]
            reference = IndexedFasta(reference_path)
            try:
                annotate_coding_variants(variants, annotation, reference, "GRCh38")
            finally:
                reference.close()

        coding_overlap = variants[0]["transcript_consequences"][0]
        self.assertEqual(coding_overlap["consequences"], ["coding_sequence_variant"])
        self.assertNotIn("splice_region_variant", coding_overlap["consequences"])
        self.assertFalse(coding_overlap["protein_effect_evaluated"])
        unknown_structure = next(row for row in variants[1]["transcript_consequences"] if row["transcript_id"] == "unknown_structure_tx")
        self.assertEqual(unknown_structure["consequences"], ["transcript_variant"])
        self.assertNotIn("intron_variant", unknown_structure["consequences"])
        explicit_utr = next(row for row in variants[2]["transcript_consequences"] if row["transcript_id"] == "protein_tx")
        self.assertEqual(explicit_utr["consequences"], ["5_prime_UTR_variant"])

    def test_negative_strand_coding_consequences_match_independent_ensembl_vep_fixtures(self) -> None:
        fixture = json.loads((Path(__file__).parent / "fixtures/ensembl_vep_negative_strand.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="medai-vep-fixture-") as temp:
            vcf, reference, annotation = _ensembl_vep_reference_fixture(Path(temp))
            report = analyze_vcf(vcf, fixture["assembly"], 10, reference, annotation)
            bundle = Path(temp) / "vep-bundle"
            create_genomics_bundle(bundle, report, vcf, fixture["assembly"], 10, reference, annotation)
            replay = verify_genomics_bundle(bundle, replay=True)
            self.assertTrue(replay["valid"])
            self.assertTrue(replay["replayed"])
        self.assertEqual(len(report["variants"]), len(fixture["cases"]))
        for case, row in zip(fixture["cases"], report["variants"]):
            consequence = row["transcript_consequences"][0]
            self.assertEqual(consequence["transcript_id"], fixture["transcript_id"])
            self.assertEqual(consequence["gene_id"], fixture["gene_id"])
            self.assertEqual(consequence["strand"], "-")
            self.assertEqual(consequence["cds_position"], f"{case['source_cds_position']}-{case['source_cds_position']}")
            self.assertEqual(consequence["protein_position"], case["source_protein_position"])
            self.assertEqual(consequence["consequences"], [case["expected_consequence"]])
            self.assertEqual(consequence["reference_amino_acids"], case["expected_reference_amino_acid"])
            self.assertEqual(consequence["alternate_amino_acids"], case["expected_alternate_amino_acid"])

    def test_bundle_replays_exact_local_analysis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-bundle-test-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            bundle = root / "bundle"

            created = create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation)
            self.assertEqual(created["artifact_count"], 7)
            self.assertEqual((bundle / "inputs/cohort.vcf").read_bytes(), vcf.read_bytes())
            self.assertEqual((bundle / "inputs/reference.fa").read_bytes(), reference.read_bytes())
            self.assertEqual((bundle / "inputs/annotation.gff3").read_bytes(), annotation.read_bytes())
            self.assertFalse(json.loads((bundle / "manifest.json").read_text())["network_accessed"])
            self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
            self.assertEqual((bundle / "inputs/cohort.vcf").stat().st_mode & 0o777, 0o600)

            integrity = verify_genomics_bundle(bundle)
            replay = verify_genomics_bundle(bundle, replay=True)
            self.assertTrue(integrity["valid"])
            self.assertFalse(integrity["replayed"])
            self.assertTrue(replay["replayed"])
            consequences = [item["consequences"] for row in report["variants"] for item in row["transcript_consequences"]]
            self.assertEqual(consequences, [["synonymous_variant"], ["synonymous_variant"]])

    def test_bundle_detects_changed_input_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-bundle-test-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            bundle = root / "bundle"
            create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation)

            with self.assertRaisesRegex(ValueError, "already exists"):
                create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation)
            (bundle / "inputs/cohort.vcf").write_text("tampered", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_genomics_bundle(bundle)

    def test_replay_rejects_changed_installed_science_code(self) -> None:
        import medai_science.reproducibility as reproducibility
        from medai_science import genomics

        with tempfile.TemporaryDirectory(prefix="medai-bundle-test-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            bundle = root / "bundle"
            create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation)
            real_hash = reproducibility._sha256
            changed_source = Path(genomics.__file__).resolve()

            def changed_hash(path: str | Path) -> str:
                if Path(path).resolve() == changed_source:
                    return "0" * 64
                return real_hash(Path(path))

            reproducibility._sha256 = changed_hash
            try:
                with self.assertRaisesRegex(ValueError, "differs from the captured version"):
                    verify_genomics_bundle(bundle, replay=True)
            finally:
                reproducibility._sha256 = real_hash

    def test_replay_rejects_analysis_paths_that_escape_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="medai-bundle-test-") as temp:
            root = Path(temp)
            vcf, reference, annotation = _inputs(root)
            report = analyze_vcf(vcf, "GRCh38", 100, reference, annotation)
            bundle = root / "bundle"
            create_genomics_bundle(bundle, report, vcf, "GRCh38", 100, reference, annotation)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["analysis"]["reference_fasta"] = "../../outside.fa"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid artifact path"):
                verify_genomics_bundle(bundle, replay=True)


if __name__ == "__main__":
    unittest.main()
