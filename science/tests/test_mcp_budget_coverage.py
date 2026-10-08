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
    "lookup_preprint",
    "search_preprints",
    "ontology_search",
    "ontology_resolve_identifier",
    "ontology_cross_mapping_review",
    "search_clinical_trials",
    "get_clinical_trial",
    "refresh_and_compare_clinical_trial",
    "get_alphafold_structure",
    "get_uniprot_protein_record",
    "map_experimental_protein_residues",
    "search_intact_protein_interactions",
    "search_genes_mygene",
    "search_variants_clinvar",
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
}

_CONDITIONALLY_METERED_HELPERS = {"_gencc_query", "_clingen_gene_validity_query"}
_CONDITIONALLY_METERED_TOOLS = {
    "gene_disease_gencc_gene": "_gencc_query",
    "gene_disease_gencc_disease": "_gencc_query",
    "gene_disease_clingen_gene": "_clingen_gene_validity_query",
    "gene_disease_clingen_disease": "_clingen_gene_validity_query",
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


if __name__ == "__main__":
    unittest.main()
