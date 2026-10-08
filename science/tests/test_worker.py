import asyncio
import unittest

from types import SimpleNamespace
from unittest.mock import patch

from medai_science.cancellation import ResearchSessionCancelled
from medai_science.worker import ArtifactCollectionLimitError, ResearchWorker, _collect_artifact_pages, _literature_watch_pmids, _register_association_source_graph, _register_gene_disease_curation_graph, _register_orphadata_gene_graph, _resolve_objective_disease_identifiers, _retrieve_gene_disease_curations, _retrieve_objective_disease_evidence, _retrieve_source_identified_compounds, _search_hypothesis_literature, _specialist_meta, _tool_value
from medai_science.scientific_agents import agent_may_call


class WorkerLeaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_ols_disease_resolution_uses_only_literal_preferred_labels_not_synonyms_or_rank(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if arguments["ontology"] == "efo":
                return {"artifact_id": "efo-artifact", "records": [{"obo_id": "EFO_0000249", "ontology_prefix": "EFO", "label": "Alzheimer disease"}, {"obo_id": "EFO_0000555", "ontology_prefix": "EFO", "label": "Alzheimer's disease"}]}
            return {"artifact_id": "mondo-artifact", "records": [{"obo_id": "MONDO:0004979", "ontology_prefix": "MONDO", "label": "Alzheimer disease", "synonyms": ["Alzheimer's disease"]}]}

        artifacts = []
        result = await _resolve_objective_disease_identifiers(call, "RES-1", "genes.curate", "Investigate Alzheimer disease", ["Alzheimer disease"], asyncio.Event(), artifacts)
        self.assertEqual(result["identifiers"], ["EFO_0000249", "MONDO:0004979"])
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(set(artifacts), {"efo-artifact", "mondo-artifact"})
        self.assertTrue(all(arguments["ontology"] in {"efo", "mondo"} for _, arguments in calls))

    async def test_association_graph_keeps_native_scores_and_unresolved_gene_mappings_separate(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            return {"saved": True}

        result = await _register_association_source_graph(call, "RES-1", "genes.curate", {
            "source": "NHGRI-EBI GWAS Catalog", "disease_id": "EFO_0000249", "records": [{"gene_id": None, "gene_mapping_candidate": "BRCA1", "symbol": "BRCA1", "association_id": "GCST001", "study_id": "GCST001", "evidence_type": "curated_gwas_variant_mapped_gene"}],
        }, "ART-GWAS")
        self.assertEqual(result, {"edges_added": 2, "warnings": []})
        entities = [row for name, row in calls if name == "medai_graph_add_entity"]
        edges = [row for name, row in calls if name == "medai_graph_link_evidence"]
        mapping = next(row for row in entities if row["entity_type"] == "GeneMappingCandidate")
        self.assertEqual(mapping["metadata"]["stable_gene_identifier"], False)
        self.assertEqual(mapping["metadata"]["mapping_candidate"], "BRCA1")
        self.assertTrue(all(row["artifact_ids"] == ["ART-GWAS"] for row in edges))

    async def test_monarch_assertion_graph_retains_aggregator_provenance_and_artifact(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            return {"saved": True}

        result = await _register_association_source_graph(call, "RES-1", "genes.curate", {
            "source": "Monarch Initiative Knowledge Graph", "disease_id": "MONDO:0004979", "records": [{
                "gene_id": "HGNC:117", "symbol": "APOE", "association_id": "MONDO:association/1",
                "association_category": "biolink:CausalGeneToDiseaseAssociation", "primary_knowledge_source": ["infores:clingen"],
                "aggregator_knowledge_source": ["infores:monarchinitiative"], "publications": ["PMID:123"], "evidence_count": 1,
                "evidence_type": "monarch_knowledge_graph_gene_disease_assertion",
            }],
        }, "ART-MONARCH")
        self.assertEqual(result, {"edges_added": 2, "warnings": []})
        assertion = next(arguments for name, arguments in calls if name == "medai_graph_add_entity" and arguments["entity_type"] == "GeneDiseaseAssociationRecord")
        self.assertEqual(assertion["metadata"]["primary_knowledge_source"], ["infores:clingen"])
        self.assertEqual(assertion["metadata"]["aggregator_knowledge_source"], ["infores:monarchinitiative"])
        edges = [arguments for name, arguments in calls if name == "medai_graph_link_evidence"]
        self.assertTrue(all(edge["artifact_ids"] == ["ART-MONARCH"] for edge in edges))

    async def test_gene_disease_graph_preserves_source_assertions_and_exact_artifact_provenance(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            return {"saved": True}

        digest = "a" * 64
        result = await _register_gene_disease_curation_graph(call, "RES-1", "genes.curate", "GenCC", {
            "assertions": [{"gene_curie": "HGNC:3236", "gene_symbol": "HFE", "disease_curie": "MONDO:0005148", "disease_title": "hemochromatosis", "classification": "Definitive", "record_sha256": digest}],
        }, "ART-1")
        self.assertEqual(result, {"edges_added": 2, "warnings": []})
        entities = [arguments for name, arguments in calls if name == "medai_graph_add_entity"]
        edges = [arguments for name, arguments in calls if name == "medai_graph_link_evidence"]
        self.assertEqual({row["entity_id"] for row in entities}, {"HGNC:3236", "MONDO:0005148", f"GenCCAssertion:{digest}"})
        self.assertEqual({row["relation"] for row in edges}, {"SOURCE_ASSERTION_NAMES_GENE", "SOURCE_ASSERTION_NAMES_DISEASE"})
        self.assertTrue(all(row["artifact_ids"] == ["ART-1"] and row["evidence_ids"] == [] for row in edges))

    async def test_gene_disease_graph_skips_rows_without_exact_stable_identifiers(self):
        calls = []

        async def call(name, arguments):
            calls.append(name)

        result = await _register_gene_disease_curation_graph(call, "RES-1", "genes.curate", "ClinGen", {
            "curations": [{"gene_hgnc_id": "HGNC:3236", "disease_mondo_id": "", "record_sha256": "b" * 64}],
        }, "ART-2")
        self.assertEqual(result["edges_added"], 0)
        self.assertTrue(result["warnings"])
        self.assertEqual(calls, [])

    async def test_orphadata_graph_requires_exact_hgnc_and_preserves_native_assertion_fields(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            return {"saved": True}

        result = await _register_orphadata_gene_graph(call, "RES-1", "genes.curate", {
            "disease": {"orphacode": "61", "preferred_term": "Alpha-mannosidosis"},
            "dataset_release_date": "2024-12-03",
            "records": [
                {"gene_symbol": "MAN2B1", "gene_external_references": [{"source": "HGNC", "reference": "6826"}, {"source": "UniProtKB/SwissProt", "reference": "O00754"}], "association_type": "Disease-causing germline mutation(s) in", "association_status": "Validated", "source_of_validation": "22587682[PMID]", "record_sha256": "c" * 64},
                {"gene_symbol": "UNKNOWN", "gene_external_references": [{"source": "HGNC", "reference": "bad"}], "association_type": "Candidate gene tested in", "association_status": "Not validated", "record_sha256": "d" * 64},
            ],
        }, "ORPHA-ART")
        self.assertEqual(result["edges_added"], 2)
        self.assertEqual(len(result["warnings"]), 1)
        entities = [arguments for name, arguments in calls if name == "medai_graph_add_entity"]
        edges = [arguments for name, arguments in calls if name == "medai_graph_link_evidence"]
        self.assertEqual({row["entity_id"] for row in entities}, {"Disease:ORPHA:61", "HGNC:6826", f"OrphadataGeneDiseaseAssertion:{'c' * 64}"})
        assertion = next(row for row in entities if row["entity_type"] == "GeneDiseaseAssertion")
        self.assertEqual(assertion["metadata"]["association_status"], "Validated")
        self.assertEqual(assertion["metadata"]["association_type"], "Disease-causing germline mutation(s) in")
        self.assertEqual(assertion["metadata"]["source_of_validation"], "22587682[PMID]")
        self.assertTrue(all(row["artifact_ids"] == ["ORPHA-ART"] for row in edges))

    async def test_source_identified_compounds_resolve_through_agentmesh_and_checkpoint(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_compounds_lookup_pubchem":
                return {"artifact_id": f"pubchem-{arguments['identifier']}", "matches": []}
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            raise AssertionError(name)

        receipts = [{"id": "extract-1", "kind": "paper_extraction_receipt", "payload": {"normalized_entities": [
            {"database": "PubChem", "entity_type": "Chemical", "identifier": "CID:2244"},
            {"database": "PubChem", "entity_type": "Chemical", "identifier": "aspirin"},
        ]}}]
        completed, artifact_ids, checkpoint = set(), [], {}
        result = await _retrieve_source_identified_compounds(
            call, "RES-1", "drugs.investigate", "worker", receipts, [], 2,
            asyncio.Event(), completed, artifact_ids, checkpoint,
        )
        self.assertEqual([name for name, _ in calls if name == "medai_compounds_lookup_pubchem"], ["medai_compounds_lookup_pubchem"])
        self.assertEqual(calls[0][1]["identifier"], "2244")
        self.assertEqual(result["discovered"][0]["extraction_receipt_id"], "extract-1")
        self.assertIn("pubchem-cid:2244", completed)
        self.assertIn("pubchem-2244", artifact_ids)
        self.assertEqual(checkpoint["stage"], "source_identified_compounds_resolved")

    async def test_hypothesis_literature_search_is_budgeted_source_linked_and_never_scores_novelty(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_research_get_budget":
                return {"remaining": 13}
            if name == "medai_papers_search_pubmed":
                return {"results": [{"provenance": {"source": "PubMed", "source_id": "12345", "title": "Candidate paper"}, "abstract": "Do not copy this abstract.", "identifiers": {}}], "coverage": {"total_hits": 3, "search_truncated": False}, "coverage_artifact_id": "coverage-1"}
            if name == "medai_research_store_artifact":
                return {"id": f"artifact-{sum(call_name == name for call_name, _ in calls)}"}
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            raise AssertionError(name)

        proposal = {"artifact_id": "proposal-1", "candidate_count": 2, "candidates": [
            {"id": "HYP-1", "statement": "APOE may be connected to Alzheimer disease through TREM2."},
            {"id": "HYP-2", "statement": "APP may be connected to amyloid beta through PSEN1."},
        ]}
        artifacts, checkpoint, completed = [], {}, set()
        receipt = await _search_hypothesis_literature(call, "RES-1", "hypotheses.propose", "worker", proposal, 2, asyncio.Event(), completed, artifacts, checkpoint)
        self.assertEqual(receipt["searched_candidate_count"], 2)
        self.assertEqual(receipt["omitted_candidate_count"], 0)
        self.assertIsNone(receipt["novelty_assessment"])
        self.assertEqual(len([name for name, _ in calls if name == "medai_papers_search_pubmed"]), 2)
        paper_receipt = next(arguments["payload"] for name, arguments in calls if name == "medai_research_store_artifact" and arguments["kind"] == "hypothesis_literature_candidate")
        self.assertNotIn("abstract", paper_receipt)
        self.assertIn("does not establish semantic relevance", receipt["interpretation_limit"])
        resumed = await _search_hypothesis_literature(call, "RES-1", "hypotheses.propose", "worker", proposal, 2, asyncio.Event(), completed, artifacts, checkpoint)
        self.assertEqual(resumed["searched_candidate_count"], 2)
        self.assertEqual(len([name for name, _ in calls if name == "medai_papers_search_pubmed"]), 2)
        changed = {**proposal, "candidates": [dict(proposal["candidates"][0], statement="APOE may be connected to Alzheimer disease through TREM3.")]}
        await _search_hypothesis_literature(call, "RES-1", "hypotheses.propose", "worker", changed, 2, asyncio.Event(), completed, artifacts, checkpoint)
        self.assertEqual(len([name for name, _ in calls if name == "medai_papers_search_pubmed"]), 3)

    async def test_objective_disease_associations_resolve_labels_and_resume_by_objective_and_plan(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            if name == "medai_ontology_search":
                return {"artifact_id": f"ols-{arguments['ontology']}", "records": []}
            if name == "medai_ontology_cross_mapping_review":
                return {"artifact_id": f"oxo-{arguments['source_identifier']}", "mappings": [], "mapping_interpretation": "Candidate mappings only; human review required."}
            if name in {"medai_graph_add_entity", "medai_graph_link_evidence"}:
                return {"saved": True}
            if name in {"medai_gene_disease_clingen_disease", "medai_gene_disease_gencc_disease"}:
                digest = ("a" if "clingen" in name else "b") * 64
                if "clingen" in name:
                    return {"source": "ClinGen Gene-Disease Validity", "query": {"value": arguments["disease_curie"]}, "artifact_id": "clingen", "curations": [{"gene_hgnc_id": "HGNC:3236", "gene_symbol": "HFE", "disease_mondo_id": arguments["disease_curie"], "disease_label": "hemochromatosis", "classification": "Definitive", "record_sha256": digest}]}
                return {"source": "Gene Curation Coalition (GenCC)", "query": {"disease_curie": arguments["disease_curie"]}, "artifact_id": "gencc", "assertions": [{"gene_curie": "HGNC:3236", "gene_symbol": "HFE", "disease_curie": arguments["disease_curie"], "disease_title": "hemochromatosis", "classification": "Definitive", "record_sha256": digest}]}
            if arguments["disease_id"].startswith("MONDO:"):
                return {"sources": [{"source": "Monarch Initiative Knowledge Graph", "disease_id": arguments["disease_id"], "records": [{"gene_id": "HGNC:117", "symbol": "APOE", "association_id": "monarch-1", "evidence_type": "monarch_knowledge_graph_gene_disease_assertion"}]}], "source_errors": [], "artifact_ids": ["monarch"], "reconciliation": {"genes": []}}
            return {"reconciliation": {"disease_id": arguments["disease_id"], "genes": []}, "sources": [
                {"source": "Open Targets Platform", "disease_id": arguments["disease_id"], "disease_name": "Alzheimer disease", "records": [{"gene_id": "ENSG00000141510", "symbol": "TP53", "score": 0.63, "evidence_type": "integrated_target_disease_association"}]},
                {"source": "NHGRI-EBI GWAS Catalog", "disease_id": arguments["disease_id"], "records": [{"gene_id": "BRCA1", "symbol": "BRCA1", "association_id": "GCST001", "evidence_type": "curated_gwas_variant_mapped_gene"}]},
            ], "source_errors": [], "artifact_ids": ["open-targets", "gwas", "reconciliation-1"]}

        cancellation = asyncio.Event()
        completed, artifacts, checkpoint = set(), [], {}
        missing = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Investigate Alzheimer disease", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(missing["status"], "skipped")
        self.assertEqual(len([row for row in calls if row[0] == "medai_ontology_search"]), 2)
        planned_missing = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Investigate Alzheimer disease", cancellation, completed, artifacts, checkpoint, ["Alzheimer disease"])
        self.assertEqual(planned_missing["status"], "skipped")
        ontology_call_count = len([row for row in calls if row[0] == "medai_ontology_search"])
        self.assertEqual(ontology_call_count, 4)
        await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Investigate Alzheimer disease", cancellation, completed, artifacts, checkpoint, ["Alzheimer disease"])
        self.assertEqual(len([row for row in calls if row[0] == "medai_ontology_search"]), ontology_call_count)
        ambiguous = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "EFO_0000249 versus EFO_0000511", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(ambiguous["status"], "skipped")
        mixed = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "EFO_0000249 (MONDO:0004979)", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(mixed["status"], "skipped")
        mondo = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "MONDO:0004979", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(mondo["status"], "succeeded")
        self.assertEqual(mondo["disease_id"], "MONDO:0004979")
        self.assertEqual(mondo["ontology_cross_mapping_review"]["status"], "no_direct_candidate_found")
        self.assertIn("oxo-MONDO:0004979", mondo["artifact_ids"])
        self.assertEqual(len(mondo["source_results"]), 3)
        self.assertTrue(all(row[1]["disease_curie"] == "MONDO:0004979" for row in calls if row[0].endswith("_disease")))
        first = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Alzheimer disease EFO_0000249", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(first["graph_edges_added"], 4)
        query_calls = [row for row in calls if row[0] == "medai_gene_disease_evidence"]
        self.assertEqual(len(query_calls), 2)
        self.assertEqual({row[1]["disease_id"] for row in query_calls}, {"EFO_0000249", "MONDO:0004979"})
        resumed = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Alzheimer disease EFO_0000249", cancellation, completed, artifacts, checkpoint)
        self.assertEqual(resumed, first)
        self.assertEqual(len([row for row in calls if row[0] == "medai_gene_disease_evidence"]), 2)

    async def test_direct_ontology_mapping_candidates_are_retained_for_researcher_review(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_ontology_cross_mapping_review":
                return {"artifact_id": "oxo-review-1", "mappings": [{"target_curie": "MONDO:0004979", "distance": 1}], "mapping_interpretation": "Candidate mapping; not equivalence."}
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            return {"reconciliation": {"genes": []}, "sources": [], "artifact_ids": ["efo-source"], "source_errors": []}

        result = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "EFO_0000249", asyncio.Event(), set(), [], {})
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["ontology_cross_mapping_review"]["status"], "pending_researcher_review")
        self.assertIn("oxo-review-1", result["artifact_ids"])
        mapping_call = next(arguments for name, arguments in calls if name == "medai_ontology_cross_mapping_review")
        self.assertEqual(mapping_call["target_prefixes"], ["MONDO"])

    async def test_truncated_ontology_mapping_search_never_reports_no_candidate(self):
        async def call(name, _arguments):
            if name == "medai_ontology_cross_mapping_review":
                return {"artifact_id": "oxo-truncated", "mappings": [], "provider_mapping_count": 50, "truncated": True, "duplicate_mapping_rows": 0, "mapping_interpretation": "Truncated candidate search."}
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            return {"reconciliation": {"genes": []}, "sources": [], "artifact_ids": [], "source_errors": []}

        result = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "EFO_0000249", asyncio.Event(), set(), [], {})
        mapping = result["ontology_cross_mapping_review"]
        self.assertEqual(mapping["status"], "pending_researcher_review")
        self.assertTrue(mapping["truncated"])
        self.assertEqual(mapping["provider_mapping_count"], 50)

    async def test_explicit_orphacode_routes_to_orphadata_and_resumes_from_checkpoint(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            if name in {"medai_graph_add_entity", "medai_graph_link_evidence"}:
                return {"saved": True}
            if name == "medai_gene_disease_orphadata":
                return {"source": "Orphadata / Orphanet", "found": True, "artifact_id": "orpha-61", "disease": {"orphacode": "61", "preferred_term": "Alpha-mannosidosis"}, "records": [{"gene_symbol": "MAN2B1", "gene_external_references": [{"source": "HGNC", "reference": "6826"}], "association_type": "Disease-causing germline mutation(s) in", "association_status": "Validated", "record_sha256": "e" * 64}], "record_count": 1, "interpretation_limit": "Source-reported category only."}
            raise AssertionError(f"unexpected tool call: {name}")

        completed, artifacts, checkpoint = set(), [], {}
        result = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Alpha-mannosidosis ORPHA:000061", asyncio.Event(), completed, artifacts, checkpoint)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["disease_id"], "ORPHA:61")
        self.assertEqual(result["artifact_ids"], ["orpha-61"])
        self.assertEqual(result["graph_edges_added"], 2)
        self.assertEqual(next(args["orphacode"] for name, args in calls if name == "medai_gene_disease_orphadata"), "61")
        resumed = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "Alpha-mannosidosis ORPHA:000061", asyncio.Event(), completed, artifacts, checkpoint)
        self.assertEqual(resumed, result)
        self.assertEqual(len([name for name, _ in calls if name == "medai_gene_disease_orphadata"]), 1)

    async def test_partial_disease_source_failure_is_retained_for_review(self):
        async def call(name, _arguments):
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            return {"reconciliation": {"genes": []}, "source_errors": [{"source": "GWAS Catalog", "error": "TimeoutException"}], "artifact_ids": ["open-targets", "reconciliation"]}

        result = await _retrieve_objective_disease_evidence(call, "RES-1", "genes.curate", "worker", "EFO_0000249", asyncio.Event(), set(), [], {})
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["source_errors"][0]["source"], "GWAS Catalog")

    async def test_specialist_actor_has_distinct_worker_identity_and_rejects_unknown_roles(self):
        token = "x" * 32
        worker = ResearchWorker("http://localhost:8080/mcp", worker_id="worker", agent_id="literature_librarian", agent_api_token=token)
        self.assertEqual(worker.agent_id, "literature_librarian")
        self.assertTrue(worker.worker_id.startswith("worker.literature_librarian."))
        self.assertLessEqual(len(worker.worker_id), 128)
        with self.assertRaisesRegex(ValueError, "unknown scientific agent"):
            ResearchWorker("http://localhost:8080/mcp", worker_id="worker", agent_id="unregistered")
        with self.assertRaisesRegex(ValueError, "require MEDAI_SPECIALIST_API_TOKEN"):
            ResearchWorker("http://localhost:8080/mcp", worker_id="worker", agent_id="literature_librarian")
        self.assertEqual(_specialist_meta("literature_librarian", token), {
            "io.medai/specialist-api-bearer": token,
            "io.medai/scientific-agent": "literature_librarian",
        })
        self.assertTrue(agent_may_call("pharmacology_analyst", "medai_compounds_lookup_pubchem"))
        self.assertFalse(agent_may_call("entity_curator", "medai_compounds_lookup_pubchem"))

    async def test_literature_watch_snapshot_decodes_api_json_before_mcp_call(self):
        self.assertEqual(_literature_watch_pmids('["101","202"]'), ["101", "202"])
        self.assertEqual(_literature_watch_pmids(None), [])
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            _literature_watch_pmids("not-json")
        with self.assertRaisesRegex(ValueError, "snapshot is invalid"):
            _literature_watch_pmids(["123", "bad"])

    async def test_gene_curation_stage_resumes_exact_mygene_symbols_and_keeps_sources_separate(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if name == "medai_research_save_checkpoint":
                return {"saved": True}
            source_count = 4 if "clingen" in name else 9
            digest = ("c" if "clingen" in name else "d") * 64
            record = {"gene_hgnc_id": "HGNC:3236", "gene_symbol": "HFE", "disease_mondo_id": "MONDO:0005148", "classification": "Definitive", "record_sha256": digest} if "clingen" in name else {"gene_curie": "HGNC:3236", "gene_symbol": "HFE", "disease_curie": "MONDO:0005148", "classification": "Definitive", "record_sha256": digest}
            return {"total_matching_curations" if "clingen" in name else "total_matching_assertions": source_count, "artifact_id": f"artifact-{len(calls)}", "curations" if "clingen" in name else "assertions": [record]}

        genes = [{"entrezgene": "123", "symbol": "GENE1"}]
        cancellation = asyncio.Event()
        completed = set()
        artifact_ids = []
        checkpoint = {}
        first = await _retrieve_gene_disease_curations(call, "RES-1", "genes.curate", "worker-1", genes, cancellation, completed, artifact_ids, checkpoint)
        self.assertEqual(first["source_assertions"], {"ClinGen": 4, "GenCC": 9})
        self.assertEqual(first["source_queries"], {"ClinGen": 1, "GenCC": 1})
        self.assertEqual({name for name, _ in calls if name.startswith("medai_gene_disease_")}, {"medai_gene_disease_clingen_gene", "medai_gene_disease_gencc_gene"})
        self.assertEqual(completed, {"clingen-curation:123", "gencc-curation:123"})
        call_count = len(calls)
        resumed = await _retrieve_gene_disease_curations(call, "RES-1", "genes.curate", "worker-1", genes, cancellation, completed, artifact_ids, checkpoint)
        self.assertEqual(len(calls), call_count)
        self.assertEqual(resumed["source_queries"], {"ClinGen": 0, "GenCC": 0})

    async def test_artifact_reader_follows_bounded_stable_cursors(self):
        calls = []

        async def call(name, arguments):
            calls.append((name, arguments))
            if len(calls) == 1:
                return {"items": [{"id": f"item-{index:03d}", "recorded_at": 123} for index in range(200)], "next_cursor": {"recorded_at": 123, "id": "item-199"}}
            return {"items": [{"id": "item-200", "recorded_at": 123}], "next_cursor": None}

        result = await _collect_artifact_pages(call, "RES-1", "paper")
        self.assertEqual(len(result["items"]), 201)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["pages_read"], 2)
        self.assertEqual(calls[0][1], {"session_id": "RES-1", "limit": 200, "kind": "paper"})
        self.assertEqual(calls[1][1]["after_recorded_at"], 123)
        self.assertEqual(calls[1][1]["after_id"], "item-199")

    async def test_artifact_reader_rejects_malformed_continuation(self):
        async def call(_name, _arguments):
            return {"items": [], "next_cursor": {"recorded_at": "bad", "id": "item"}}

        with self.assertRaisesRegex(ValueError, "invalid continuation cursor"):
            await _collect_artifact_pages(call, "RES-1")

    async def test_artifact_reader_rejects_a_cursor_that_does_not_match_the_last_row(self):
        async def call(_name, _arguments):
            return {"items": [{"id": "last", "recorded_at": 4}], "next_cursor": {"recorded_at": 4, "id": "earlier"}}

        with self.assertRaisesRegex(ValueError, "did not advance"):
            await _collect_artifact_pages(call, "RES-1")

    async def test_artifact_reader_observes_cancellation_between_pages(self):
        cancellation = asyncio.Event()
        calls = 0

        async def call(_name, _arguments):
            nonlocal calls
            calls += 1
            cancellation.set()
            return {"items": [{"id": "first", "recorded_at": 1}], "next_cursor": {"recorded_at": 1, "id": "first"}}

        with self.assertRaises(ResearchSessionCancelled):
            await _collect_artifact_pages(call, "RES-1", cancellation=cancellation)
        self.assertEqual(calls, 1)

    async def test_artifact_reader_fails_closed_when_serialized_byte_limit_is_exceeded(self):
        async def call(_name, _arguments):
            return {"items": [{"id": "large"}], "next_cursor": None}

        with patch("medai_science.worker._json_bytes", return_value=b"x" * (8 * 1024 * 1024 + 1)):
            with self.assertRaises(ArtifactCollectionLimitError):
                await _collect_artifact_pages(call, "RES-1")

    async def test_agentmesh_cancellation_error_is_classified_for_active_task(self):
        result = SimpleNamespace(
            isError=True,
            content=[SimpleNamespace(text="research session cancellation requested; provider operation cancelled")],
        )
        with self.assertRaises(ResearchSessionCancelled):
            _tool_value(result)

    async def test_immediate_cancellation_stops_heartbeat_without_renewing(self):
        worker = ResearchWorker("http://127.0.0.1:8080/mcp")
        calls = []

        async def call(name, arguments):
            calls.append(name)
            return {"cancellation_requested": True}

        cancellation = asyncio.Event()
        lease_lost = []
        await worker._maintain_lease(call, "RES-1", "task-1", cancellation, lease_lost)
        self.assertTrue(cancellation.is_set())
        self.assertEqual(calls, ["medai_research_get_control"])
        self.assertEqual(lease_lost, [])

    async def test_control_or_renewal_failure_marks_lease_lost(self):
        worker = ResearchWorker("http://127.0.0.1:8080/mcp")
        calls = []

        async def call(name, arguments):
            calls.append(name)
            if name == "medai_research_get_control":
                return {"cancellation_requested": False}
            raise RuntimeError("gateway unavailable")

        cancellation = asyncio.Event()
        lease_lost = []
        await worker._maintain_lease(call, "RES-1", "task-1", cancellation, lease_lost)
        self.assertEqual(calls, ["medai_research_get_control", "medai_research_renew_task_lease"])
        self.assertIn("gateway unavailable", lease_lost[0])


if __name__ == "__main__":
    unittest.main()
