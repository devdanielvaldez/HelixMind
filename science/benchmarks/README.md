# Adjudicated benchmark input

Run `medai-science benchmark-evaluate <file.json> [--bins N]` on a frozen adjudicated dataset. The JSON format is:

```json
{
  "schema_version": "1",
  "benchmark_id": "drug-interaction-review-v1",
  "labeling_protocol": "Two independent reviewers; third reviewer resolved disagreements.",
  "cases": [
    {
      "case_id": "case-0001",
      "evidence_type": "versioned_label",
      "expected": "supports",
      "predicted": "supports",
      "confidence": 0.82
    }
  ]
}
```

Expected labels must come from a documented independent adjudication process. Predictions use the same categorical vocabulary; use `__abstain__` for an explicit non-answer. The evaluator reports accuracy, macro F1, per-label precision/recall/F1, confusion matrices, and abstention rate overall and separately for each evidence type. It also reports two-sided 95% Wilson score intervals for accuracy, precision, and recall proportions; a metric with a zero denominator has a null interval. These nominal intervals assume independent binomial cases, which may not hold when cases share papers, datasets, or study cohorts. They describe sampling uncertainty for the supplied benchmark and do not correct case-selection bias or establish generalization ([NIST guidance on Wilson proportion intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)). The confidence field means the system's probability that its categorical prediction matches the adjudicated label. Correctness Brier score and equal-width expected calibration error measure that narrow quantity; they do not measure whether a biomedical claim is true or clinically valid.

Inputs are bounded to 100,000 cases and 20 MiB at the CLI. Every case ID must be unique. The output includes a canonical SHA-256 of the benchmark document so the dataset version is auditable. This repository currently supplies the evaluator and logic fixtures, not an independently adjudicated biomedical benchmark corpus.

For evidence-screening benchmarks, schema version `2` records the provenance digest for each reviewed evidence item and preserves two blinded primary-review labels. If the primary reviewers disagree, a third, distinct adjudicator is required; agreement cases must not include an adjudicator record. The evaluator derives the expected label from these records and evaluates system predictions against the consensus/resolved labels. Raw primary agreement is aggregated across cases. Cohen's kappa is computed only within a stable reviewer pair; if a benchmark uses several reviewer pairs, kappas are reported separately per pair and no misleading pooled kappa is emitted. Primary labels are aligned by pseudonymous reviewer ID, so JSON reviewer order does not change the agreement result. Example case:

Each source-digest-bound schema-v2 case must have a unique `evidence_sha256`. Repeating the same source under multiple case IDs would pseudoreplicate observations and is rejected. Schema-v3 evidence-quality cases likewise require one case per exact source digest, with all quality domains in that case. This does not detect distinct papers that share a cohort; the report retains the independence and clustering limitation.

```json
{
  "case_id": "case-0001",
  "evidence_sha256": "<64 lowercase hex characters>",
  "evidence_type": "paper_claim",
  "reviewers": [
    {"reviewer_id": "reviewer-a", "role": "reviewer", "label": "supports"},
    {"reviewer_id": "reviewer-b", "role": "reviewer", "label": "contradicts"},
    {"reviewer_id": "adjudicator-c", "role": "adjudicator", "label": "context"}
  ],
  "predicted": "context",
  "confidence": 0.82
}
```

The evidence digest binds the annotation to a source artifact without embedding article text. Reviewer IDs should be pseudonymous and stable within a benchmark. The evaluator checks the recorded review structure; it cannot independently verify reviewer identity, blindness, or the source artifact itself. Existing schema version `1` datasets remain supported, but do not produce inter-reviewer agreement metrics.

Schema version `3` evaluates multidomain ordinal evidence-quality annotations. Each domain declares `ordered_levels` and separate `non_ordinal_labels` (for example, `not_assessable`); nominal states are excluded from ordinal distance and weighted kappa. Two distinct primary reviewers are required for each source-digest-bound case, and a distinct adjudicator is required exactly when any domain disagrees. Predictions may abstain with `{"label":"__abstain__","confidence":null}`. The report is stratified by domain and evidence type and includes reviewer agreement, coverage, ordinal error and weighted kappa on comparable ordinal labels, plus exact-label confidence calibration. Weighted kappas are calculated separately for each stable primary-reviewer pair; when there are multiple pairs, the pooled kappa is null while raw agreement and coverage remain aggregated. Reviewer labels are aligned by pseudonymous reviewer ID, regardless of case-array order. Confidence means probability of an exact rating match to the resolved annotation, not probability that a scientific claim is true. The protocol supplies the ordinal meanings and order; the evaluator does not create an evidence grade or verify reviewer independence. This repository has synthetic schema-3 fixtures only; an independently curated and adjudicated biomedical corpus remains an open task.

## Four-domain benchmark input

The CLI can validate and attach an input using `medai-science biomedical-domain-benchmark-submit RES-... benchmark.json`; set `MEDAI_API_URL` and `MEDAI_API_BEARER_TOKEN` for a tenant-owned research session. The optional AgentMesh `benchmark.evaluate` task reads one same-session artifact named `biomedical_domain_benchmark_input`; its `source` must be `researcher-supplied adjudicated biomedical benchmark`. The MCP evaluator verifies the API-retained payload digest before parsing. The UTF-8 JSON payload is limited to 4 MiB and has this top-level shape:

```json
{
  "schema_version": "1",
  "benchmark_id": "biomedical-domain-v1",
  "labeling_protocol": "Two blinded primary reviewers; an independent adjudicator resolved disagreements.",
  "domains": {
    "genomics": [],
    "proteomics": [],
    "pharmacology": [],
    "literature_extraction": []
  }
}
```

Each nonempty domain array contains cases using the source-digest-bound schema-version-2 case format above (`case_id`, `evidence_sha256`, `evidence_type`, two primary `reviewers`, `predicted`, and `confidence`; disagreement requires a third adjudicator). Case IDs are unique across all four arrays. Per-domain label cardinality is capped at 32, evidence-type strata at 16, and total cases at 100,000. Empty arrays are retained as `not_evaluated`, never as evidence of zero errors. The report explicitly says that source bytes, reviewer identity/independence, and protocol execution were not externally verified; a syntactically valid digest is not proof that a source was checked. Upload the JSON as the payload of a session artifact with the required kind, task ID, and source; then select the optional `benchmark.evaluate` task in the research plan. The worker evaluates only when exactly one matching input exists; place separate benchmark runs in separate research sessions. The output report is a second retained artifact bound to the exact input artifact ID and payload digest. No independently curated production corpus is included here, so synthetic cases remain logic fixtures only.

The same optional task accepts a schema-v3 evidence-quality corpus submitted by `medai-science evidence-quality-benchmark-submit RES-... quality-benchmark.json`. It is stored as `evidence_quality_benchmark_input` with source `researcher-supplied double-reviewed evidence-quality benchmark`, then evaluated by the `benchmark_analyst` through the `biomedical.evidence-quality-benchmark-evaluate` contract. Reports preserve per-domain and per-evidence-type agreement, ordinal prediction, abstention, and exact-label calibration, bound to the API-retained input artifact digest. If both evaluator input types are attached, the worker requests review rather than choosing one. The evaluator validates reviewer records and digest formatting but cannot authenticate the reviewed evidence bytes, reviewer independence, or the protocol. Independent corpus curation remains required.
