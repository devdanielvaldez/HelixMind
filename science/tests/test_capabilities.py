import tempfile
import unittest
import json
from pathlib import Path

from medai_science.capabilities import Decision, LocalSentenceTransformer, SemanticCapabilityIndex, apply_authoritative_lifecycle, canonical_amcp_registry_package, evaluate_contract_fit, hash_local_embedding_model, load_amcp_catalog, runtime_capability_statuses


class FakeEmbedder:
    model_id = "fake-v1"
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        value = text.lower()
        if "structure" in value or "fold" in value:
            return [3.0, 0.0]
        if "literature" in value or "paper" in value:
            return [0.0, 2.0]
        return [0.0, 0.1]


class DimensionChangingEmbedder:
    model_id = "dimension-changing-v1"

    def embed(self, text):
        if "protein" in text.lower():
            return [1.0, 0.0]
        return [1.0, 0.0, 0.0]


def package(status="verified", permission="biomedical.proteins.read"):
    return {
        "capability": {"id": "biomedical.protein.structure", "version": "2.1.0", "status": status, "intent": "Retrieve a protein's predicted structure"},
        "contract": {
            "inputs": {"type": "object", "required": ["accession"], "properties": {"accession": {"type": "string"}}},
            "outputs": {"type": "object", "required": ["structure", "artifact_id"], "properties": {"structure": {"type": "object"}, "artifact_id": {"type": "string"}}},
            "effects": ["read_only", "network_egress"],
        },
        "authority": {"permissions": [permission]},
    }


def requirement():
    return {
        "inputs": {"type": "object", "required": ["accession"], "properties": {"accession": {"type": "string"}}},
        "outputs": {"type": "object", "required": ["structure"], "properties": {"structure": {"type": "object"}}},
    }


class SemanticCapabilityTests(unittest.TestCase):
    def test_contract_fit_rejects_unprovided_required_inputs_and_optional_required_outputs(self):
        manifest = package()
        manifest["contract"]["inputs"] = {
            "type": "object",
            "required": ["accession", "organism"],
            "properties": {
                "accession": {"type": "string"},
                "organism": {"type": "string"},
            },
        }
        manifest["contract"]["outputs"] = {
            "type": "object",
            "required": ["artifact_id"],
            "properties": {
                "structure": {"type": "object"},
                "artifact_id": {"type": "string"},
            },
        }

        result = evaluate_contract_fit(requirement(), manifest)

        self.assertFalse(result["compatible"])
        self.assertEqual(
            result["reasons"],
            [
                "input schema does not accept all required inputs",
                "output schema does not provide all required outputs",
            ],
        )

    def test_contract_fit_accepts_guaranteed_required_inputs_and_outputs(self):
        result = evaluate_contract_fit(requirement(), package())

        self.assertTrue(result["compatible"])

    def test_contract_fit_checks_additional_properties_and_array_item_schemas(self):
        manifest = package()
        manifest["contract"]["inputs"]["additionalProperties"] = False
        manifest["contract"]["outputs"]["additionalProperties"] = False
        manifest["contract"]["outputs"]["properties"]["structure"] = {
            "type": "array",
            "items": {"type": "string"},
        }
        requested = requirement()
        requested["outputs"]["properties"]["structure"] = {
            "type": "array",
            "items": {"type": "number"},
        }

        result = evaluate_contract_fit(requested, manifest)

        self.assertFalse(result["compatible"])
        self.assertIn("input schema does not accept all required inputs", result["reasons"])
        self.assertIn("output schema does not provide all required outputs", result["reasons"])

    def test_contract_fit_rejects_output_extras_when_consumer_forbids_them(self):
        manifest = package()
        requested = requirement()
        requested["outputs"]["additionalProperties"] = False
        manifest["contract"]["outputs"]["additionalProperties"] = True

        result = evaluate_contract_fit(requested, manifest)

        self.assertFalse(result["compatible"])
        self.assertIn("output schema does not provide all required outputs", result["reasons"])

    def test_authenticated_lifecycle_overrides_local_catalog_status_and_defaults_absent_to_experimental(self):
        promoted = package(status="production")
        absent = package(status="verified")
        absent["capability"]["id"] = "biomedical.protein.annotation"
        result = apply_authoritative_lifecycle(
            [promoted, absent],
            {"biomedical.protein.structure@2.1.0": "verified"},
        )
        self.assertEqual(result[0]["capability"]["status"], "verified")
        self.assertEqual(result[1]["capability"]["status"], "experimental")
        self.assertEqual(promoted["capability"]["status"], "production")

    def test_authoritative_lifecycle_rejects_invalid_status_or_manifest_identity(self):
        with self.assertRaisesRegex(ValueError, "invalid authoritative lifecycle"):
            apply_authoritative_lifecycle([package()], {"biomedical.protein.structure@2.1.0": "approved"})
        with self.assertRaisesRegex(ValueError, "stable capability ID"):
            apply_authoritative_lifecycle([{"capability": {"version": "1.0"}}], {})

    def test_registry_package_hash_ignores_untrusted_catalog_lifecycle_and_is_stable(self):
        original = package(status="production")
        canonical, digest = canonical_amcp_registry_package(original)
        self.assertNotIn("status", canonical["capability"])
        self.assertEqual(original["capability"]["status"], "production")
        repeated, second_digest = canonical_amcp_registry_package({**original, "capability": {**original["capability"], "status": "experimental"}})
        self.assertEqual(canonical, repeated)
        self.assertEqual(digest, second_digest)

    def test_local_embedding_identity_tracks_weight_bytes_not_directory_name(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "model"
            model.mkdir()
            weights = model / "model.safetensors"
            weights.write_bytes(b"weights-v1")
            first = hash_local_embedding_model(model)
            weights.write_bytes(b"weights-v2")
            second = hash_local_embedding_model(model)
            self.assertEqual(len(first), 64)
            self.assertNotEqual(first, second)

    def test_vector_search_persists_reuses_unchanged_embeddings_and_checks_equivalence(self):
        embedder = FakeEmbedder()
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", embedder)
            descriptor = package()
            self.assertEqual(index.index([descriptor]), 1)
            self.assertEqual(index.index([descriptor]), 0)
            self.assertEqual(len(embedder.calls), 1)
            result = index.search(
                "Find the folded protein structure",
                requirement(),
                reuse_similarity_threshold=0.82,
                permitted_effects={"read_only", "network_egress"},
                permitted_permissions={"biomedical.proteins.read"},
                runtime_statuses={"biomedical.protein.structure@2.1.0": {"active": True, "state": "active"}},
            )
            self.assertEqual(result[0]["decision"], Decision.REUSE.value)
            self.assertTrue(result[0]["contract_fit"]["compatible"])
            index.close()

    def test_vector_search_rejects_manifest_bytes_that_do_not_match_the_index_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([package()])
            tampered = json.dumps(package(permission="biomedical.proteins.admin"), sort_keys=True, separators=(",", ":"))
            index.db.execute(
                "UPDATE capability_vectors SET package_json=? WHERE capability_id=? AND version=?",
                (tampered, "biomedical.protein.structure", "2.1.0"),
            )
            index.db.commit()

            with self.assertRaisesRegex(ValueError, "manifest digest or identity mismatch"):
                index.search("protein fold structure", requirement())
            index.close()

    def test_index_rejects_mixed_dimensions_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", DimensionChangingEmbedder())
            another = package()
            another["capability"]["id"] = "biomedical.literature.lookup"
            another["capability"]["intent"] = "Search scientific literature"

            with self.assertRaisesRegex(ValueError, "embedding dimension changed"):
                index.index([package(), another])

            count = index.db.execute("SELECT count(*) FROM capability_vectors").fetchone()[0]
            self.assertEqual(count, 0)
            index.close()

    def test_search_rejects_corrupted_non_normalized_stored_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([package()])
            index.db.execute(
                "UPDATE capability_vectors SET vector_json=? WHERE capability_id=? AND version=?",
                ("[4.0,0.0]", "biomedical.protein.structure", "2.1.0"),
            )
            index.db.commit()

            with self.assertRaisesRegex(ValueError, "not normalized"):
                index.search("protein fold structure", requirement())
            index.close()

    def test_reuse_requires_an_exact_active_runtime_route_for_the_configured_service(self):
        descriptor = package()
        key = "biomedical.protein.structure@2.1.0"
        snapshot = {"revision": 12, "digest": "sha256:" + "a" * 64, "resources": [{
            "kind": "route",
            "name": "medai-biomedical.protein.structure-2.1.0",
            "state": "active",
            "spec": {"service": "medai-mcp", "capability": "biomedical.protein.structure", "strategy": "least_active"},
        }]}
        runtime = runtime_capability_statuses([descriptor], snapshot, "medai-mcp")
        self.assertTrue(runtime[key]["active"])
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([descriptor])
            result = index.search("protein fold structure", requirement(), reuse_similarity_threshold=0.82, runtime_statuses=runtime)[0]
            self.assertEqual(result["decision"], Decision.REUSE.value)
            self.assertEqual(result["runtime"]["snapshot_revision"], 12)
            absent = runtime_capability_statuses([descriptor], {"resources": []}, "medai-mcp")
            result = index.search("protein fold structure", requirement(), reuse_similarity_threshold=0.82, runtime_statuses=absent)[0]
            self.assertNotEqual(result["decision"], Decision.REUSE.value)
            self.assertIn("no exact active runtime route", result["reuse_blockers"][-1])
            wrong_scope = runtime_capability_statuses([descriptor], snapshot, "other-service")
            self.assertFalse(wrong_scope[key]["active"])
            index.close()

    def test_exact_active_route_does_not_override_draining_or_wrong_route_identity(self):
        descriptor = package()
        key = "biomedical.protein.structure@2.1.0"
        base = {"revision": 7, "digest": "sha256:" + "b" * 64}
        route = {"kind": "route", "name": "medai-biomedical.protein.structure-2.1.0", "state": "active", "spec": {"service": "medai-mcp", "capability": "biomedical.protein.structure"}}
        draining = runtime_capability_statuses([descriptor], {**base, "resources": [{**route, "state": "draining"}]}, "medai-mcp")
        wrong_identity = runtime_capability_statuses([descriptor], {**base, "resources": [{**route, "spec": {**route["spec"], "capability": "other.capability"}}]}, "medai-mcp")
        self.assertFalse(draining[key]["active"])
        self.assertFalse(wrong_identity[key]["active"])

    def test_similarity_never_overrides_contract_authority_or_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([package(status="experimental", permission="biomedical.proteins.admin")])
            result = index.search("protein fold structure", requirement(), permitted_effects={"read_only"}, permitted_permissions={"biomedical.proteins.read"})[0]
            self.assertEqual(result["semantic_similarity"], 1.0)
            self.assertEqual(result["decision"], Decision.EXTEND.value)
            self.assertFalse(result["contract_fit"]["compatible"])
            self.assertIn("capability permissions exceed the caller's available permissions", result["contract_fit"]["reasons"])
            self.assertIn("capability effects exceed the caller's allowed effects", result["contract_fit"]["reasons"])
            index.close()

    def test_uncalibrated_similarity_never_recommends_reuse_even_for_active_verified_route(self):
        descriptor = package(status="production")
        key = "biomedical.protein.structure@2.1.0"
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([descriptor])
            result = index.search(
                "protein fold structure", requirement(),
                permitted_effects={"read_only", "network_egress"},
                permitted_permissions={"biomedical.proteins.read"},
                runtime_statuses={key: {"active": True, "state": "active"}},
            )[0]
            self.assertNotEqual(result["decision"], Decision.REUSE.value)
            self.assertIn("no independently adjudicated, operator-pinned reuse threshold is configured", result["reuse_blockers"])
            index.close()

    def test_rejects_incompatible_schema_and_corrupt_embedding_dimensions(self):
        descriptor = package()
        descriptor["contract"]["inputs"]["required"].append("admin_token")
        checked = evaluate_contract_fit(requirement(), descriptor)
        self.assertFalse(checked["compatible"])
        with tempfile.TemporaryDirectory() as folder:
            index = SemanticCapabilityIndex(Path(folder) / "vectors.sqlite", FakeEmbedder())
            index.index([package()])
            with self.assertRaises(ValueError):
                index.search("protein structure", requirement(), limit=0)
            index.close()

    def test_catalog_removal_prunes_stale_vectors_and_model_change_invalidates_all_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.sqlite"
            index = SemanticCapabilityIndex(path, FakeEmbedder())
            index.index([package()])
            self.assertEqual(index.index([]), 0)
            self.assertEqual(index.search("protein structure", requirement()), [])
            index.index([package()])
            index.close()

            class ChangedEmbedder(FakeEmbedder):
                model_id = "fake-v2"

            rebuilt = SemanticCapabilityIndex(path, ChangedEmbedder())
            self.assertEqual(rebuilt.search("protein structure", requirement()), [])
            rebuilt.close()

    def test_catalog_lifecycle_defaults_to_experimental_and_only_accepts_explicit_status_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cap.yaml"
            path.write_text("schemaVersion: amcp/0.1\ncapability:\n  id: biomedical.test\n  version: 1.0.0\n  intent: Find protein structure\ncontract: {}\n", encoding="utf-8")
            loaded = load_amcp_catalog(folder)
            self.assertEqual(loaded[0]["capability"]["status"], "experimental")
            loaded = load_amcp_catalog(folder, {"biomedical.test@1.0.0": "verified"})
            self.assertEqual(loaded[0]["capability"]["status"], "verified")
            loaded = load_amcp_catalog(folder, {"biomedical.test@1.0.0": "rolled_back"})
            self.assertEqual(loaded[0]["capability"]["status"], "rolled_back")
            with self.assertRaises(ValueError):
                load_amcp_catalog(folder, {"biomedical.test@1.0.0": "production_without_review"})

    def test_local_model_refuses_remote_identifier_or_missing_directory(self):
        with self.assertRaises(ValueError):
            LocalSentenceTransformer("organization/model-name")


if __name__ == "__main__":
    unittest.main()
