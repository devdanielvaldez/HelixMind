"""Guard expensive AgentMesh scientific operations against unmetered regressions."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


_SERVER = Path(__file__).parents[1] / "src/medai_science/mcp_server.py"

# Directly metered tool handlers. The two bulk-curation helpers conditionally
# charge before a network refresh and serve local cached searches without charge.
_DIRECTLY_METERED = {
    "plan_objective_with_model",
    "search_capabilities_semantic",
    "generate_python_capability_proposal",
    "generate_r_capability_proposal",
    "validate_python_capability_proposal",
    "validate_r_capability_proposal",
    "execute_generated_python_capability",
    "benchmark_generated_capability_versions",
    "skills_learn_research_sequence",
    "search_pubmed",
    "resolve_crossref_paper",
    "resolve_crossref_reference",
    "lookup_preprint",
    "search_preprints",
    "ontology_search",
    "ontology_resolve_identifier",
    "ontology_cross_mapping_review",
    "search_clinical_trials",
    "get_clinical_trial",
    "refresh_and_compare_clinical_trial",
    "get_alphafold_structure",
    "analyze_alphafold_coordinates",
    "get_uniprot_protein_record",
    "map_experimental_protein_residues",
    "search_intact_protein_interactions",
    "search_genes_mygene",
    "search_variants_clinvar",
    "lookup_variant_dbsnp",
    "lookup_variants_gnomad_frequency",
    "annotate_variants_ensembl_vep",
    "search_proteins_uniprot",
    "analyze_reactome_pathways",
    "lookup_compound_pubchem",
    "drug_profile_chembl",
    "drug_shared_targets_chembl",
    "drug_interaction_labels_openfda",
    "drug_interaction_cross_source_review",
    "screen_drug_interaction_pmc_fulltext",
    "drug_interaction_review_sources",
    "gene_disease_evidence",
    "gene_disease_panelapp",
    "gene_disease_orphadata",
    "understand_pmc_paper",
    "align_pmc_result_claims",
    "propose_research_hypotheses",
    "criticize_research_hypotheses",
    "evaluate_biomedical_domain_benchmark_in_session",
    "evaluate_evidence_quality_benchmark_in_session",
}

_CONDITIONALLY_METERED_HELPERS = {"_gencc_query", "_clingen_gene_validity_query"}
_CONDITIONALLY_METERED_TOOLS = {
    "gene_disease_gencc_gene": "_gencc_query",
    "gene_disease_gencc_disease": "_gencc_query",
    "gene_disease_clingen_gene": "_clingen_gene_validity_query",
    "gene_disease_clingen_disease": "_clingen_gene_validity_query",
}

_EXTERNAL_SOURCE_CLIENTS = {
    "PubMedClient", "CrossrefClient", "PreprintClient", "OntologyClient",
    "ClinicalTrialsClient", "AlphaFoldClient", "UniProtClient", "SIFTSClient",
    "IntActClient", "MyGeneClient", "ClinVarClient", "GnomADClient",
    "EnsemblVEPClient", "ReactomeClient", "PubChemClient", "ChEMBLClient",
    "OpenFDALabelClient", "DailyMedClient", "PMCBioCClient", "PanelAppClient",
    "OrphadataClient", "OpenTargetsClient", "GWASCatalogClient", "DbSNPClient",
}


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            names.add(item.func.id if isinstance(item.func, ast.Name) else item.func.attr if isinstance(item.func, ast.Attribute) else "")
    return names


def _attributes(node: ast.AST) -> set[str]:
    return {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)}


class MCPBudgetCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(_SERVER.read_text(encoding="utf-8"))
        cls.functions = {
            node.name: node
            for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_direct_scientific_mcp_operations_have_charge_paths(self) -> None:
        missing = sorted(name for name in _DIRECTLY_METERED if name not in self.functions)
        self.assertEqual(missing, [], "metering inventory names unknown MCP handlers")
        unmetered = sorted(
            name for name in _DIRECTLY_METERED
            if not (_calls(self.functions[name]) & {"_charge", "_charge_model_usage"})
        )
        self.assertEqual(unmetered, [], "scientific MCP operations must contain explicit session charge paths")

    def test_cached_curated_sources_meter_only_conditional_refreshes(self) -> None:
        for name in _CONDITIONALLY_METERED_HELPERS:
            self.assertIn(name, self.functions)
            calls = _calls(self.functions[name])
            self.assertIn("_charge", calls, f"{name} must meter refreshes")
            self.assertIn("refresh_due", _attributes(self.functions[name]), f"{name} must distinguish cache reads from refreshes")
        for name, helper in _CONDITIONALLY_METERED_TOOLS.items():
            self.assertIn(helper, _calls(self.functions[name]), f"{name} must use its metered cache/refresh helper")

    def test_scheduled_external_refreshes_use_durable_budget_routes(self) -> None:
        for name in ("refresh_literature_watch", "refresh_clinical_trial_watch"):
            source = ast.get_source_segment(_SERVER.read_text(encoding="utf-8"), self.functions[name]) or ""
            self.assertIn("/budget/consume", source, f"{name} must use the durable research-budget endpoint")

    def test_pubmed_watch_snapshot_validation_precedes_charges_and_provider_calls(self) -> None:
        handler = self.functions["refresh_literature_watch"]
        validation = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_pmid_snapshot"
        ]
        charges = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "post"
        ]
        searches = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "search_recent_ids"
        ]
        self.assertEqual(len(validation), 2)
        self.assertTrue(charges)
        self.assertEqual(len(searches), 1)
        self.assertLess(max(node.lineno for node in validation), min(node.lineno for node in charges))
        self.assertLess(max(node.lineno for node in validation), searches[0].lineno)

    def test_pmc_revalidation_is_charged_before_fetching_source_text(self) -> None:
        handler = self.functions["drug_interaction_review_sources"]
        charges = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_charge"
            and len(node.value.args) >= 4
            and isinstance(node.value.args[3], ast.Constant)
            and node.value.args[3].value == "PMC BioC source revalidation for selected drug-interaction claim screening"
        ]
        fetches = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fetch"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "PMCBioCClient"
        ]
        self.assertEqual(len(charges), 1)
        self.assertEqual(len(fetches), 1)
        self.assertLess(charges[0].lineno, fetches[0].lineno, "charge for refetch must occur before network retrieval")

    def test_crossref_reference_search_is_hash_bound_charged_and_never_auto_indexed(self) -> None:
        handler = self.functions["resolve_crossref_reference"]
        source = ast.get_source_segment(_SERVER.read_text(encoding="utf-8"), handler) or ""
        self.assertIn("retained_records_sha256", source)
        self.assertIn("reference_sha256", source)
        self.assertIn("unverified_candidates", source)
        self.assertIn("crossref_reference_resolution_candidates", source)
        self.assertNotIn("citation-links", source)
        charge = source.index("await _charge(")
        provider_call = source.index("client.search_bibliographic(")
        self.assertLess(charge, provider_call, "candidate lookup must be charged before Crossref is called")

    def test_external_scientific_calls_follow_a_budget_charge(self) -> None:
        violations = []
        for name in _DIRECTLY_METERED:
            handler = self.functions[name]
            charge_lines = [
                node.lineno for node in ast.walk(handler)
                if isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"_charge", "_charge_model_usage"}
            ]
            source_calls = [
                node for node in ast.walk(handler)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id in _EXTERNAL_SOURCE_CLIENTS
            ]
            for call in source_calls:
                if not any(line < call.lineno for line in charge_lines):
                    client = call.func.value.func.id
                    violations.append(f"{name}: {client}.{call.func.attr} at line {call.lineno}")
        self.assertEqual(violations, [], "provider calls must be preceded by a session budget charge")

    def test_every_handler_constructing_a_scientific_source_client_is_metered(self) -> None:
        scheduled = {"refresh_literature_watch", "refresh_clinical_trial_watch"}
        discovered = {
            name
            for name, handler in self.functions.items()
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _EXTERNAL_SOURCE_CLIENTS
                for node in ast.walk(handler)
            )
        }
        declared = _DIRECTLY_METERED | scheduled
        self.assertEqual(
            sorted(discovered - declared),
            [],
            "new scientific provider handlers must be added to explicit budget coverage",
        )
        violations = []
        for name in sorted(discovered):
            handler = self.functions[name]
            if name in scheduled:
                source = ast.get_source_segment(_SERVER.read_text(encoding="utf-8"), handler) or ""
                if "/budget/consume" not in source:
                    violations.append(f"{name}: missing durable session budget charge")
                continue
            charge_lines = [
                node.lineno for node in ast.walk(handler)
                if isinstance(node, ast.Await)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"_charge", "_charge_model_usage"}
            ]
            if not charge_lines:
                violations.append(f"{name}: missing explicit session budget charge")
                continue
            source_calls = [
                node for node in ast.walk(handler)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id in _EXTERNAL_SOURCE_CLIENTS
            ]
            for call in source_calls:
                if not any(line < call.lineno for line in charge_lines):
                    violations.append(f"{name}: provider call at line {call.lineno} precedes its charge")
        self.assertEqual(violations, [], "all discovered scientific source handlers must be metered")

    def test_intact_artifacts_use_license_cautious_retention_payload(self) -> None:
        handler = self.functions["search_intact_protein_interactions"]
        self.assertIn("build_intact_retention_payload", _calls(handler))

    def test_pmc_extraction_uses_text_stripping_projection_before_artifact_persistence(self) -> None:
        handler = self.functions["understand_pmc_paper"]
        projections = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_extraction_retention_receipt"
        ]
        persists = [
            node for node in ast.walk(handler)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "post"
        ]
        self.assertEqual(len(projections), 1)
        self.assertTrue(persists)
        self.assertLess(projections[0].lineno, min(node.lineno for node in persists))
        paper_values = [
            value
            for node in ast.walk(handler)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
            for key, value in zip(node.value.keys, node.value.values)
            if isinstance(key, ast.Constant) and key.value == "paper"
        ]
        self.assertEqual(len(paper_values), 1)
        self.assertIsInstance(paper_values[0], ast.Name)
        self.assertEqual(paper_values[0].id, "retained", "MCP must not return transient PMC full text to its caller")

    def test_citation_graph_read_is_session_scoped_and_has_agentmesh_contract(self) -> None:
        handler = self.functions.get("get_citation_network")
        self.assertIsNotNone(handler)
        calls = _calls(handler)
        self.assertIn("get", calls)
        self.assertIn("_api_headers", calls)
        source = ast.get_source_segment(_SERVER.read_text(encoding="utf-8"), handler) or ""
        self.assertIn("/v1/research/{session_id}/citation-links", source)
        contract = Path(__file__).parents[2] / "integrations/agentmesh/capabilities/biomedical.paper-citation-network.yaml"
        contract_text = contract.read_text(encoding="utf-8")
        self.assertIn("biomedical.paper-citation-network", contract_text)
        self.assertIn("medai_papers_citation_network", contract_text)

    def test_alphafold_coordinate_analysis_is_metered_bound_and_registered(self) -> None:
        handler = self.functions["analyze_alphafold_coordinates"]
        charge_lines = [
            node.lineno for node in ast.walk(handler)
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "_charge"
        ]
        prediction_calls = [
            node.lineno for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "predictions"
        ]
        coordinate_calls = [
            node.lineno for node in ast.walk(handler)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "coordinate_file"
        ]
        self.assertEqual(len(charge_lines), 1)
        self.assertEqual(len(prediction_calls), 1)
        self.assertEqual(len(coordinate_calls), 1)
        self.assertLess(charge_lines[0], prediction_calls[0])
        self.assertLess(prediction_calls[0], coordinate_calls[0])
        source = ast.get_source_segment(_SERVER.read_text(encoding="utf-8"), handler) or ""
        self.assertIn('prediction["model_id"] == model_id', source)
        contract = Path(__file__).parents[2] / "integrations/agentmesh/capabilities/biomedical.protein-structure-alphafold-analyze.yaml"
        contract_text = contract.read_text(encoding="utf-8")
        self.assertIn("biomedical.protein-structure-alphafold-analyze", contract_text)
        self.assertIn("mcp://medai/medai_proteins_analyze_alphafold_coordinates", contract_text)


if __name__ == "__main__":
    unittest.main()
