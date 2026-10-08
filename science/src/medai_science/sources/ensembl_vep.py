"""Bounded Ensembl VEP consequence lookup for reference-validated local variants."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

import httpx
from medai_science.sources.bounded_http import read_bounded_response


_CONTIG = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
_ALLELE = re.compile(r"^[ACGTN]+$")
_CONSEQUENCE_FIELDS = (
    "transcript_id", "gene_id", "gene_symbol", "biotype", "strand", "consequence_terms",
    "cds_start", "cds_end", "protein_start", "protein_end", "amino_acids", "codons",
)


class EnsemblVEPClient:
    """Query the official GRCh37/GRCh38 Ensembl REST VEP regional endpoint."""

    endpoints = {
        "GRCh37": "https://grch37.rest.ensembl.org",
        "GRCh38": "https://rest.ensembl.org",
    }
    max_variants = 200
    max_response_bytes = 8_000_000
    max_total_response_bytes = 24_000_000
    max_result_bytes = 2_000_000

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 45.0):
        self.transport = transport
        self.timeout = timeout

    @classmethod
    def queryable_count(cls, report: dict[str, Any]) -> int:
        if not isinstance(report, dict) or not isinstance(report.get("variants"), list):
            raise ValueError("Ensembl VEP requires a local genomic report with variants")
        if report.get("genome_build") not in cls.endpoints or report.get("reference_validated") is not True:
            raise ValueError("VEP cross-validation requires a supported build and reference-validated local variants")
        if not isinstance(report.get("reference_fasta_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", report["reference_fasta_sha256"]):
            raise ValueError("local report is missing its reference FASTA digest")
        if len(report["variants"]) > 1000:
            raise ValueError("VEP annotation is limited to 1000 local variant alleles per request")
        return sum(
            isinstance(variant, dict)
            and variant.get("sequence_allele_supported") is True
            and variant.get("reference_validated") is True
            for variant in report["variants"]
        )

    async def annotate_report(self, report: dict[str, Any], genome_build: str | None = None) -> dict[str, Any]:
        if not isinstance(report, dict) or not isinstance(report.get("variants"), list):
            raise ValueError("Ensembl VEP requires a local genomic report with variants")
        build = genome_build or report.get("genome_build")
        if build not in self.endpoints or report.get("genome_build") != build:
            raise ValueError("VEP assembly must exactly match the local report's GRCh37/GRCh38 build")
        if report.get("reference_validated") is not True:
            raise ValueError("VEP cross-validation requires reference-validated local variants")
        if not isinstance(report.get("reference_fasta_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", report["reference_fasta_sha256"]):
            raise ValueError("local report is missing its reference FASTA digest")
        variants = report["variants"]
        if len(variants) > 1000:
            raise ValueError("VEP annotation is limited to 1000 local variant alleles per request")
        submitted: list[tuple[str, dict[str, Any]]] = []
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict) or variant.get("sequence_allele_supported") is not True or variant.get("reference_validated") is not True:
                continue
            contig = variant.get("contig")
            position = variant.get("normalized_position_1_based", variant.get("position_1_based"))
            ref = variant.get("normalized_reference_allele", variant.get("reference_allele"))
            alt = variant.get("normalized_alternate_allele", variant.get("alternate_allele"))
            if not isinstance(contig, str) or not _CONTIG.fullmatch(contig.removeprefix("chr")):
                raise ValueError(f"local variant {index} has an unsupported contig identifier")
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                raise ValueError(f"local variant {index} has an invalid position")
            if not isinstance(ref, str) or not _ALLELE.fullmatch(ref) or not isinstance(alt, str) or not _ALLELE.fullmatch(alt) or ref == alt:
                raise ValueError(f"local variant {index} has unsupported reference/alternate alleles")
            # VEP regional input follows VCF-like whitespace-separated fields; isolate each ALT allele.
            request_input = f"{contig.removeprefix('chr')} {position} . {ref} {alt} . . ."
            submitted.append((request_input, variant))
        if not submitted:
            report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            return {
                "source": "Ensembl Variant Effect Predictor REST API",
                "source_url": self.endpoints[build] + "/vep/homo_sapiens/region",
                "assembly": build,
                "ensembl_release": None,
                "input_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "reference_fasta_sha256": report["reference_fasta_sha256"],
                "queried_variant_alleles": 0,
                "results": [],
                "skipped_unsupported_or_unvalidated": len(variants),
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "attribution": "Ensembl Variant Effect Predictor; Ensembl REST API data are provided under the applicable Ensembl reuse terms.",
                "interpretation_limit": "VEP transcript annotations describe consequences under the selected Ensembl assembly and transcript set; they are not pathogenicity, disease-causality, or clinical interpretations.",
            }

        records: list[dict[str, Any]] = []
        server_release: str | None = None
        total_response_bytes = 0
        endpoint = self.endpoints[build] + "/vep/homo_sapiens/region"
        async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout, follow_redirects=False) as client:
            for start in range(0, len(submitted), self.max_variants):
                batch = submitted[start : start + self.max_variants]
                async with client.stream("POST", endpoint, json={"variants": [item[0] for item in batch]}, headers={"Accept": "application/json"}) as response:
                    body = await read_bounded_response(response, self.max_response_bytes, "Ensembl VEP")
                    response_headers = response.headers
                total_response_bytes += len(body)
                if total_response_bytes > self.max_total_response_bytes:
                    raise ValueError("Ensembl VEP responses exceed the 24 MB per-operation limit")
                try:
                    payload = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError("Ensembl VEP returned invalid JSON") from error
                if not isinstance(payload, list) or len(payload) != len(batch):
                    raise ValueError("Ensembl VEP response count does not match submitted variants")
                if server_release is None:
                    server_release = response_headers.get("x-ens-release") or response_headers.get("x-ens-version")
                for (request_input, local_variant), result in zip(batch, payload):
                    if not isinstance(result, dict) or result.get("input") != request_input:
                        raise ValueError("Ensembl VEP response does not bind to its exact submitted variant")
                    allele_string = result.get("allele_string")
                    if not isinstance(allele_string, str) or "/" not in allele_string or local_variant.get("normalized_alternate_allele", local_variant.get("alternate_allele")) not in allele_string.split("/"):
                        raise ValueError("Ensembl VEP response lacks its allele identity")
                    transcript_rows = result.get("transcript_consequences", [])
                    if not isinstance(transcript_rows, list) or len(transcript_rows) > 100_000:
                        raise ValueError("Ensembl VEP transcript consequence list is malformed or oversized")
                    selected_rows = []
                    for transcript in transcript_rows:
                        if not isinstance(transcript, dict):
                            continue
                        selected_rows.append({key: transcript[key] for key in _CONSEQUENCE_FIELDS if key in transcript})
                    normalized = {
                        "input": request_input,
                        "local_record_number": local_variant.get("record_number"),
                        "local_alternate_index": local_variant.get("alternate_index"),
                        "local_contig": local_variant.get("contig"),
                        "local_position_1_based": local_variant.get("normalized_position_1_based", local_variant.get("position_1_based")),
                        "local_reference_allele": local_variant.get("normalized_reference_allele", local_variant.get("reference_allele")),
                        "local_alternate_allele": local_variant.get("normalized_alternate_allele", local_variant.get("alternate_allele")),
                        "vep_assembly": result.get("assembly_name"),
                        "seq_region_name": result.get("seq_region_name"),
                        "start": result.get("start"),
                        "end": result.get("end"),
                        "strand": result.get("strand"),
                        "allele_string": allele_string,
                        "most_severe_consequence": result.get("most_severe_consequence"),
                        "transcript_consequences": selected_rows,
                        "source_record_sha256": hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    }
                    if normalized["vep_assembly"] != build or normalized["seq_region_name"] != request_input.split(" ", 1)[0]:
                        raise ValueError("Ensembl VEP response assembly or contig does not match the request")
                    records.append(normalized)
        report_bytes = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result = {
            "source": "Ensembl Variant Effect Predictor REST API",
            "source_url": endpoint,
            "assembly": build,
            "ensembl_release": server_release,
            "input_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "reference_fasta_sha256": report["reference_fasta_sha256"],
            "queried_variant_alleles": len(submitted),
            "skipped_unsupported_or_unvalidated": len(variants) - len(submitted),
            "results": records,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "attribution": "Ensembl Variant Effect Predictor; Ensembl REST API data are provided under the applicable Ensembl reuse terms.",
            "interpretation_limit": "VEP transcript annotations describe consequences under the selected Ensembl assembly and transcript set; they are not pathogenicity, disease-causality, or clinical interpretations.",
        }
        result_bytes = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(result_bytes) > self.max_result_bytes:
            raise ValueError("normalized Ensembl VEP artifact exceeds the 2 MB retention limit")
        return result
