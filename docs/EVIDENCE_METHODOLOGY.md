# Evidence and study metadata methodology

## Current inventory

The `evidence.grade` worker stage is a descriptive inventory. It summarizes existing evidence polarity and quality annotations, and adds a separate study metadata inventory from PubMed publication-type labels and exact Publication Type MeSH UIs.

| Exact PubMed publication type | Inventory category |
| --- | --- |
| `Case Reports` | `case_report` |
| `Clinical Trial` | `clinical_trial` |
| `Clinical Trial, Phase I` | `phase_i_clinical_trial` |
| `Clinical Trial, Phase II` | `phase_ii_clinical_trial` |
| `Clinical Trial, Phase III` | `phase_iii_clinical_trial` |
| `Clinical Trial, Phase IV` | `phase_iv_clinical_trial` |
| `Controlled Clinical Trial` | `controlled_clinical_trial` |
| `Meta-Analysis` | `meta_analysis` |
| `Observational Study` | `observational_study` |
| `Randomized Controlled Trial` | `randomized_controlled_trial` |
| `Systematic Review` | `systematic_review` |

Matching trims outer whitespace and ignores case. It does not infer study design from a title, abstract, MeSH descriptor, or a broader publication type such as `Journal Article`. Original PubMed publication types and MeSH descriptor identifiers remain available in the source artifact. MeSH descriptors are treated as topic context because they do not by themselves establish the design of the reported study.

For the following Publication Type UIs, the inventory can also recognize the exact identifier supplied in PubMed's publication-type field. The ID signal remains distinct from the display-label signal and links to its NLM record:

| NLM Publication Type UI | Inventory category |
| --- | --- |
| [D002363](https://meshb.nlm.nih.gov/record/ui?ui=D002363) | `case_report` |
| [D016430](https://meshb.nlm.nih.gov/record/ui?ui=D016430) | `clinical_trial` |
| [D016449](https://meshb.nlm.nih.gov/record/ui?ui=D016449) | `randomized_controlled_trial` |
| [D017418](https://meshb.nlm.nih.gov/record/ui?ui=D017418) | `meta_analysis` |
| [D017426](https://meshb.nlm.nih.gov/record/ui?ui=D017426) | `phase_i_clinical_trial` |
| [D017427](https://meshb.nlm.nih.gov/record/ui?ui=D017427) | `phase_ii_clinical_trial` |
| [D017428](https://meshb.nlm.nih.gov/record/ui?ui=D017428) | `phase_iii_clinical_trial` |
| [D017429](https://meshb.nlm.nih.gov/record/ui?ui=D017429) | `phase_iv_clinical_trial` |
| [D018848](https://meshb.nlm.nih.gov/record/ui?ui=D018848) | `controlled_clinical_trial` |
| [D064888](https://meshb.nlm.nih.gov/record/ui?ui=D064888) | `observational_study` |
| [D000078182](https://meshb.nlm.nih.gov/record/ui?ui=D000078182) | `systematic_review` |

Only these exact IDs are mapped. For example, the phase III topic descriptor ([D017326](https://meshb.nlm.nih.gov/record/ui?ui=D017326)) is not treated as a phase III study Publication Type. NLM distinguishes Publication Types, which describe indexed publication characteristics, from topic headings; this mapping preserves that distinction ([NLM Publication Characteristics](https://www.nlm.nih.gov/mesh/pubtypes.html)).

The inventory counts a record once per recognized category. Categories can overlap. It preserves up to 100 example PMIDs per category and reports whether that example list was truncated; category counts are not truncated. An unrecognized or absent publication type remains unclassified.

The same inventory reports retained PubMed metadata coverage for title, publication date, journal, authors, DOI, PMCID, abstract length, publication types, Publication Type UIs, and MeSH headings. Each field is counted as nonempty, empty, not captured, or malformed for the session's PubMed records. Abstract content is not retained; only bounded length and digest metadata are used. These counts describe the captured source record and do not infer completeness at PubMed, article quality, or evidence strength. A missing identifier inside a captured identifier map is counted as empty; a missing identifier map is counted as not captured.

## Related-publication links

PubMed EFetch `CommentsCorrections` entries are retained as bounded `{ref_type, pmid}` pairs when the provider supplies a recognized-shaped relationship type and a numeric PMID. Duplicate pairs and malformed identifiers are omitted. The evidence inventory counts these exact provider `RefType` values, and the literature workspace links to the referenced PubMed records. This includes relationship types for errata, expressions of concern, and retractions; a link is a source navigation signal, not an automatic determination that an article is invalid or that a notice applies to every result in it. Researchers must inspect the linked records and their scope. NLM documents these citation relationships in its [PubMed data-provider guidance](https://www.ncbi.nlm.nih.gov/books/NBK3828/) and lists the human-readable relationship labels in [PubMed Help](https://pubmed.ncbi.nlm.nih.gov/help/).

## Quality and certainty limits

This inventory does not assign an evidence grade, rank designs, estimate certainty, or assess risk of bias, precision, directness, consistency, or reporting bias. Those fields are explicitly `not_assessed`. A provider publication type is indexing metadata and may be absent or incomplete. Existing numeric quality annotations are retained per source record without summation, weighting, or cross-source calibration; they are not calibrated probabilities.

The NLM definition of `Randomized Controlled Trial` describes random treatment assignment alongside concurrent test and control groups and follow-up ([MeSH publication type](https://www.ncbi.nlm.nih.gov/mesh/68016449)). MedAI records the exact provider label and does not independently inspect the paper to verify that definition. Publication type indexing guidance is available in the [NLM PubMed data-provider documentation](https://www.ncbi.nlm.nih.gov/sites/books/NBK3828/).

## Evaluation status

The categorical benchmark evaluator reports two-sided 95% Wilson score intervals for accuracy, per-label precision, and recall; ordinal evidence-quality reports include intervals for exact agreement/accuracy and within-one-level accuracy. A zero-denominator metric has no interval. Source-digest-bound schema-v2 and schema-v3 cases require unique evidence digests; repeated copies of the same source cannot inflate the case count, and multiple quality domains for one source belong in its single schema-v3 case. These are nominal binomial intervals and assume independent cases; distinct papers from shared cohorts or other clustering can still violate that assumption. They quantify sampling uncertainty in the supplied benchmark and do not correct selection bias or establish generalization ([NIST explanation of Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)).

The exact-label and exact-UI mappings have unit tests for recognized signals, case/whitespace handling, identifier boundaries, topic-versus-publication-type distinction, deduplication, unsupported labels, malformed metadata, overlap, and the no-grade policy. They have not been independently validated against a human-adjudicated corpus. This implementation is therefore a metadata inventory, not evidence grading or a validated study-design classifier.
# Hypothesis literature coverage

Hypothesis literature receipts validate that PubMed's reported total, requested page size, returned identifiers, fetched records, missing records, and truncation flags agree. A `retrieved_page_complete` result means only that the bounded page for that exact query was fetched without reported omissions. It does not mean PubMed or the query strategy is complete, that returned records are relevant, or that a hypothesis is novel. Novelty stays unscored until source-complete literature coverage and independently validated calibration are available.
