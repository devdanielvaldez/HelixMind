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

Expected labels must come from a documented independent adjudication process. Predictions use the same categorical vocabulary; use `__abstain__` for an explicit non-answer. The evaluator reports accuracy, macro F1, per-label precision/recall/F1, confusion matrices, and abstention rate overall and separately for each evidence type. The confidence field means the system's probability that its categorical prediction matches the adjudicated label. Correctness Brier score and equal-width expected calibration error measure that narrow quantity; they do not measure whether a biomedical claim is true or clinically valid.

Inputs are bounded to 100,000 cases and 20 MiB at the CLI. Every case ID must be unique. The output includes a canonical SHA-256 of the benchmark document so the dataset version is auditable. This repository currently supplies the evaluator and logic fixtures, not an independently adjudicated biomedical benchmark corpus.

For evidence-screening benchmarks, schema version `2` records the provenance digest for each reviewed evidence item and preserves two blinded primary-review labels. If the primary reviewers disagree, a third, distinct adjudicator is required; agreement cases must not include an adjudicator record. The evaluator derives the expected label from these records, reports raw primary agreement and Cohen's kappa, and then evaluates system predictions against the consensus/resolved labels. Example case:

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

Schema version `3` evaluates multidomain ordinal evidence-quality annotations. Each domain declares `ordered_levels` and separate `non_ordinal_labels` (for example, `not_assessable`); nominal states are excluded from ordinal distance and weighted kappa. Two distinct primary reviewers are required for each source-digest-bound case, and a distinct adjudicator is required exactly when any domain disagrees. Predictions may abstain with `{"label":"__abstain__","confidence":null}`. The report is stratified by domain and evidence type and includes reviewer agreement, coverage, ordinal error and weighted kappa on comparable ordinal labels, plus exact-label confidence calibration. Confidence means probability of an exact rating match to the resolved annotation, not probability that a scientific claim is true. The protocol supplies the ordinal meanings and order; the evaluator does not create an evidence grade or verify reviewer independence. This repository has synthetic schema-3 fixtures only; an independently curated and adjudicated biomedical corpus remains an open task.
