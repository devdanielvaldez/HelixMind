"""Capability discovery and safe draft generation; generated code is never auto-executed."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Iterable

class Decision(str, Enum):
    REUSE = "reuse"
    EXTEND = "extend"
    BUILD = "build"

@dataclass(frozen=True)
class RegisteredCapability:
    id: str
    description: str
    status: str
    version: str

@dataclass(frozen=True)
class CapabilityMatch:
    capability: RegisteredCapability
    score: float


class EmbeddingProvider:
    """Structural protocol for a local or hosted, provider-neutral text embedder."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


def hash_local_embedding_model(model_path: str | Path) -> str:
    """Hash every file in a materialized local model directory for vector/calibration identity."""
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("embedding model must be an existing local directory")
    entries = []
    total_bytes = 0
    for directory, child_dirs, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for child in child_dirs:
            if (current / child).is_symlink():
                raise ValueError("embedding model directories cannot contain symlinks")
        for filename in filenames:
            path = current / filename
            if path.is_symlink():
                raise ValueError("embedding model files cannot contain symlinks")
            stat = path.stat()
            if not path.is_file():
                continue
            total_bytes += stat.st_size
            if len(entries) >= 50_000 or total_bytes > 20 * 1024 * 1024 * 1024:
                raise ValueError("embedding model exceeds 50000 files or 20 GiB")
            entries.append((path.relative_to(root).as_posix(), path, stat.st_size, stat.st_mtime_ns))
    if not entries:
        raise ValueError("embedding model directory contains no files")
    model_hash = hashlib.sha256()
    for relative, path, size, modified_ns in sorted(entries):
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_hash.update(chunk)
        after = path.stat()
        if after.st_size != size or after.st_mtime_ns != modified_ns:
            raise ValueError("embedding model changed while its identity was computed")
        model_hash.update(relative.encode("utf-8"))
        model_hash.update(b"\0")
        model_hash.update(size.to_bytes(8, "big"))
        model_hash.update(file_hash.digest())
    return model_hash.hexdigest()


class LocalSentenceTransformer(EmbeddingProvider):
    """Load an already-present local model; never fetch model code or weights."""

    def __init__(self, model_path: str | Path):
        from pathlib import Path as LocalPath

        path = LocalPath(model_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("embedding model must be an existing local directory; network model downloads are disabled")
        self.model_id = f"sentence-transformers:sha256:{hash_local_embedding_model(path)}"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("install medai-science[ml] to use local semantic capability search") from error
        self._model = SentenceTransformer(str(path), device="cpu")

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return [float(value) for value in vector.tolist()]


def _schema_compatible(expected: dict, actual: dict, direction: str) -> bool:
    """Return whether an implementation schema satisfies a requested JSON schema subset."""
    if expected.get("type") and actual.get("type") != expected["type"]:
        return False
    if expected.get("enum"):
        expected_values = {json.dumps(value, sort_keys=True) for value in expected["enum"]}
        if direction == "output":
            if not actual.get("enum"):
                return False
            actual_values = {json.dumps(value, sort_keys=True) for value in actual["enum"]}
            if not actual_values.issubset(expected_values):
                return False
        else:
            actual_values = {json.dumps(value, sort_keys=True) for value in actual.get("enum", expected["enum"])}
            if not expected_values.issubset(actual_values):
                return False
    if expected.get("const") is not None and actual.get("const") != expected["const"]:
        return False
    if direction == "input":
        if actual.get("enum"):
            if expected.get("enum"):
                expected_values = {json.dumps(value, sort_keys=True) for value in expected["enum"]}
                actual_values = {json.dumps(value, sort_keys=True) for value in actual["enum"]}
                if not expected_values.issubset(actual_values):
                    return False
            elif expected.get("const") not in actual["enum"]:
                return False
        if actual.get("const") is not None and expected.get("const") != actual["const"]:
            return False
        for low, high in (("minimum", "maximum"), ("minLength", "maxLength"), ("minItems", "maxItems")):
            if low in actual and (low not in expected or actual[low] > expected[low]):
                return False
            if high in actual and (high not in expected or actual[high] < expected[high]):
                return False
        if actual.get("pattern") and actual.get("pattern") != expected.get("pattern"):
            return False
    else:
        if expected.get("enum"):
            if not actual.get("enum"):
                return False
            actual_values = {json.dumps(value, sort_keys=True) for value in actual["enum"]}
            expected_values = {json.dumps(value, sort_keys=True) for value in expected["enum"]}
            if not actual_values.issubset(expected_values):
                return False
        if expected.get("pattern") and actual.get("pattern") != expected.get("pattern"):
            return False
        for low in ("minimum", "minLength", "minItems"):
            if low in expected and (low not in actual or actual[low] < expected[low]):
                return False
        for high in ("maximum", "maxLength", "maxItems"):
            if high in expected and (high not in actual or actual[high] > expected[high]):
                return False
    expected_properties = expected.get("properties", {})
    actual_properties = actual.get("properties", {})
    actual_required = set(actual.get("required", []))
    expected_required = set(expected.get("required", []))
    if direction == "input":
        # Every field an implementation requires must be guaranteed by the
        # caller's task contract, not merely mentioned as an optional field.
        if not actual_required.issubset(expected_required):
            return False
    else:
        # A consumer that requires an output field needs the capability to
        # guarantee it, rather than merely declaring it as optional.
        if not expected_required.issubset(actual_required):
            return False
        if not expected_required.issubset(actual_properties):
            return False
    expected_additional = expected.get("additionalProperties", True)
    actual_additional = actual.get("additionalProperties", True)
    for key, expected_child in expected_properties.items():
        actual_child = actual_properties.get(key)
        if actual_child is None:
            if direction == "output":
                continue  # Optional consumer fields need not be emitted.
            actual_child = actual_additional
        if actual_child is False or not isinstance(actual_child, dict):
            if actual_child is True:
                continue
            return False
        if not _schema_compatible(expected_child, actual_child, direction):
            return False
    if direction == "input":
        # The implementation must accept every additional value the task
        # contract allows, including fields not listed under properties.
        if expected_additional is True and actual_additional is not True:
            return False
        if isinstance(expected_additional, dict):
            if actual_additional is False:
                return False
            if isinstance(actual_additional, dict) and not _schema_compatible(expected_additional, actual_additional, "input"):
                return False
    else:
        # Every extra property the capability may emit must be accepted by
        # the consumer's output contract.
        if expected_additional is False:
            if actual_additional is not False or not set(actual_properties).issubset(expected_properties):
                return False
        elif isinstance(expected_additional, dict):
            if actual_additional is True:
                return False
            if isinstance(actual_additional, dict) and not _schema_compatible(expected_additional, actual_additional, "output"):
                return False
            for key, actual_child in actual_properties.items():
                if key not in expected_properties and not _schema_compatible(expected_additional, actual_child, "output"):
                    return False
    expected_items = expected.get("items", True)
    actual_items = actual.get("items", True)
    if direction == "input":
        if expected_items is True and actual_items is not True:
            return False
        if isinstance(expected_items, dict):
            if actual_items is False or (isinstance(actual_items, dict) and not _schema_compatible(expected_items, actual_items, "input")):
                return False
    elif isinstance(expected_items, dict):
        if actual_items is False or actual_items is True or not _schema_compatible(expected_items, actual_items, "output"):
            return False
    return True


def evaluate_contract_fit(requirement: dict, manifest: dict, permitted_effects: set[str] | None = None, permitted_permissions: set[str] | None = None) -> dict:
    """Compare declared schemas and authority; this does not prove behavioral equivalence."""
    contract = manifest.get("contract", {})
    expected_inputs = requirement.get("inputs", {})
    expected_outputs = requirement.get("outputs", {})
    actual_inputs = contract.get("inputs", {})
    actual_outputs = contract.get("outputs", {})
    reasons = []
    if not _schema_compatible(expected_inputs, actual_inputs, "input"):
        reasons.append("input schema does not accept all required inputs")
    if not _schema_compatible(expected_outputs, actual_outputs, "output"):
        reasons.append("output schema does not provide all required outputs")
    actual_effects = set(contract.get("effects", []))
    actual_permissions = set(manifest.get("authority", {}).get("permissions", []))
    if permitted_effects is not None and not actual_effects.issubset(permitted_effects):
        reasons.append("capability effects exceed the caller's allowed effects")
    if permitted_permissions is not None and not actual_permissions.issubset(permitted_permissions):
        reasons.append("capability permissions exceed the caller's available permissions")
    return {"compatible": not reasons, "reasons": reasons, "effects": sorted(actual_effects), "permissions": sorted(actual_permissions), "scope": "declared JSON schemas, effects, and permissions only; behavioral equivalence requires independent tests"}


def _capability_text(package: dict) -> str:
    capability = package.get("capability", {})
    contract = package.get("contract", {})
    domains = capability.get("domain", [])
    return " ".join([capability.get("id", ""), capability.get("intent", ""), " ".join(map(str, domains)), json.dumps(contract.get("inputs", {}), sort_keys=True), json.dumps(contract.get("outputs", {}), sort_keys=True)])


def load_amcp_catalog(directory: str | Path, status_overrides: dict[str, str] | None = None, limit: int = 1000) -> list[dict]:
    """Read bounded, local AMCP YAML packages; lifecycle status requires explicit registry metadata."""
    import yaml

    path = Path(directory)
    if not path.is_dir():
        raise ValueError("capability catalog directory does not exist")
    files = sorted([*path.glob("*.yaml"), *path.glob("*.yml")])
    if len(files) > limit:
        raise ValueError(f"capability catalog exceeds {limit} packages")
    overrides = status_overrides or {}
    packages = []
    for package_path in files:
        if package_path.stat().st_size > 1_000_000:
            raise ValueError(f"capability package exceeds 1 MiB: {package_path.name}")
        package = yaml.safe_load(package_path.read_text(encoding="utf-8"))
        if not isinstance(package, dict):
            raise ValueError(f"capability package must be a YAML object: {package_path.name}")
        capability = package.get("capability")
        if not isinstance(capability, dict) or not capability.get("id") or not capability.get("version"):
            raise ValueError(f"capability package is missing a stable id or version: {package_path.name}")
        key = f"{capability['id']}@{capability['version']}"
        status = overrides.get(key, capability.get("status", "experimental"))
        if status not in {"experimental", "testing", "validating", "verified", "production", "production_certified", "deprecated", "rolled_back"}:
            raise ValueError(f"invalid capability lifecycle status for {key}")
        capability["status"] = status
        package["capability"] = capability
        packages.append(package)
    return packages


def apply_authoritative_lifecycle(packages: Iterable[dict], registry_statuses: dict[str, str]) -> list[dict]:
    """Apply exact authenticated registry states; absent entries cannot inherit local promotion metadata."""
    if not isinstance(registry_statuses, dict):
        raise ValueError("authoritative lifecycle registry must be a JSON object")
    allowed = {"experimental", "testing", "validating", "verified", "production", "production_certified", "deprecated", "rolled_back"}
    result = []
    for source in packages:
        package = copy.deepcopy(source)
        capability = package.get("capability")
        if not isinstance(capability, dict) or not isinstance(capability.get("id"), str) or not isinstance(capability.get("version"), str):
            raise ValueError("AMCP package requires a stable capability ID and version")
        key = f"{capability['id']}@{capability['version']}"
        status = registry_statuses.get(key, "experimental")
        if status not in allowed:
            raise ValueError(f"invalid authoritative lifecycle state for {key}")
        capability["status"] = status
        result.append(package)
    return result


def canonical_amcp_registry_package(source: dict) -> tuple[dict, str]:
    """Strip catalog lifecycle metadata and hash the exact package bytes sent for registration."""
    if not isinstance(source, dict):
        raise ValueError("AMCP package must be a JSON object")
    package = copy.deepcopy(source)
    capability = package.get("capability")
    if not isinstance(capability, dict) or not isinstance(capability.get("id"), str) or not isinstance(capability.get("version"), str):
        raise ValueError("AMCP package requires a stable capability ID and version")
    capability.pop("status", None)
    try:
        encoded = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("AMCP package must contain only finite JSON values") from error
    if len(encoded) > 1_000_000:
        raise ValueError("AMCP package exceeds 1 MiB")
    return package, hashlib.sha256(encoded).hexdigest()


def runtime_capability_statuses(packages: Iterable[dict], snapshot: dict, service: str) -> dict[str, dict]:
    """Match exact capability versions to active, scope-checked AgentMesh route resources."""
    if not isinstance(service, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", service):
        raise ValueError("AgentMesh service must be a safe 1..128 character identifier")
    resources = snapshot.get("resources") if isinstance(snapshot, dict) else None
    if not isinstance(resources, list):
        raise ValueError("AgentMesh runtime snapshot has no resource list")
    by_name = {}
    for row in resources:
        if not isinstance(row, dict) or row.get("kind") != "route" or not isinstance(row.get("name"), str):
            continue
        if row["name"] in by_name:
            raise ValueError("AgentMesh runtime snapshot contains duplicate route names")
        by_name[row["name"]] = row
    revision = snapshot.get("revision")
    digest = snapshot.get("digest")
    statuses = {}
    for package in packages:
        capability = package.get("capability", {}) if isinstance(package, dict) else {}
        capability_id, version = capability.get("id"), capability.get("version")
        if not isinstance(capability_id, str) or not isinstance(version, str):
            raise ValueError("AMCP capability requires a stable id and version")
        key = f"{capability_id}@{version}"
        route_name = "medai-" + re.sub(r"[^A-Za-z0-9._-]", "-", key)
        row = by_name.get(route_name)
        spec = row.get("spec") if isinstance(row, dict) else None
        scope_matches = bool(
            isinstance(spec, dict)
            and spec.get("service") == service
            and spec.get("capability") == capability_id
        )
        active = bool(
            row
            and row.get("state") == "active"
            and scope_matches
        )
        statuses[key] = {
            "active": active,
            "route_present": row is not None,
            "state": row.get("state") if row else "absent",
            "route_name": route_name,
            "service": service,
            "scope_matches": scope_matches,
            "observed_service": spec.get("service") if isinstance(spec, dict) else None,
            "observed_capability": spec.get("capability") if isinstance(spec, dict) else None,
            "snapshot_revision": revision,
            "snapshot_digest": digest,
            "basis": "authenticated AgentMesh active route snapshot; this confirms routing state, not scientific validity",
        }
    return statuses


def _validated_vector(values: Iterable[float]) -> list[float]:
    raw_vector = list(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_vector):
        raise ValueError("embedding values must be numeric and cannot be booleans")
    vector = [float(value) for value in raw_vector]
    if not vector or len(vector) > 65_536 or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding must contain 1..65536 finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("zero-length embeddings cannot be indexed")
    return [value / norm for value in vector]


def _stored_vector(raw_vector: str) -> list[float]:
    try:
        values = json.loads(raw_vector, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("capability index contains an invalid embedding; rebuild the index") from error
    if not isinstance(values, list):
        raise ValueError("capability index contains a malformed embedding; rebuild the index")
    vector = _validated_vector(values)
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if abs(norm - 1.0) > 1e-6:
        raise ValueError("capability index embedding is not normalized; rebuild the index")
    return vector


class SemanticCapabilityIndex:
    """Durable SQLite vector index; embeddings are generated by an injected provider."""

    def __init__(self, path: str | Path, embedder: EmbeddingProvider):
        self.embedder = embedder
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS capability_vectors (capability_id TEXT NOT NULL, version TEXT NOT NULL, status TEXT NOT NULL, package_json TEXT NOT NULL, content_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL, PRIMARY KEY(capability_id,version))")
        self.db.execute("CREATE TABLE IF NOT EXISTS capability_vector_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        model_id = getattr(embedder, "model_id", f"{type(embedder).__module__}.{type(embedder).__qualname__}")
        previous_model = self.db.execute("SELECT value FROM capability_vector_meta WHERE key='embedding_model'").fetchone()
        if previous_model and previous_model[0] != model_id:
            self.db.execute("DELETE FROM capability_vectors")
        self.db.execute("INSERT INTO capability_vector_meta(key,value) VALUES('embedding_model',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (model_id,))
        self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def index(self, packages: Iterable[dict]) -> int:
        with self._lock:
            indexed = 0
            packages = list(packages)
            seen = set()
            with self.db:
                stored_vectors = self.db.execute("SELECT vector_json FROM capability_vectors").fetchall()
                dimensions = {len(_stored_vector(row[0])) for row in stored_vectors}
                if len(dimensions) > 1:
                    raise ValueError("capability index contains mixed embedding dimensions; rebuild it")
                dimension = next(iter(dimensions), None)
                for package in packages:
                    capability = package.get("capability", {})
                    capability_id, version = capability.get("id"), capability.get("version")
                    if not isinstance(capability_id, str) or not isinstance(version, str):
                        raise ValueError("capability package requires a stable id and version")
                    key = (capability_id, version)
                    if key in seen:
                        raise ValueError(f"duplicate capability package {capability_id}@{version}")
                    seen.add(key)
                    serialized = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    digest = hashlib.sha256(serialized.encode()).hexdigest()
                    previous = self.db.execute("SELECT content_sha256 FROM capability_vectors WHERE capability_id=? AND version=?", key).fetchone()
                    if previous and previous[0] == digest:
                        continue
                    vector = _validated_vector(self.embedder.embed(_capability_text(package)))
                    if dimension is not None and len(vector) != dimension:
                        raise ValueError("embedding dimension changed; rebuild the capability index with one model")
                    dimension = len(vector)
                    self.db.execute("INSERT INTO capability_vectors VALUES(?,?,?,?,?,?) ON CONFLICT(capability_id,version) DO UPDATE SET status=excluded.status,package_json=excluded.package_json,content_sha256=excluded.content_sha256,vector_json=excluded.vector_json", (capability_id, version, capability.get("status", "experimental"), serialized, digest, json.dumps(vector)))
                    indexed += 1
                existing = self.db.execute("SELECT capability_id,version FROM capability_vectors").fetchall()
                for key in existing:
                    if tuple(key) not in seen:
                        self.db.execute("DELETE FROM capability_vectors WHERE capability_id=? AND version=?", key)
            return indexed

    def search(self, need: str, requirement: dict, *, limit: int = 10, min_similarity: float = 0.0, reuse_similarity_threshold: float | None = None, permitted_effects: set[str] | None = None, permitted_permissions: set[str] | None = None, runtime_statuses: dict[str, dict] | None = None) -> list[dict]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        if not need.strip() or len(need) > 4096:
            raise ValueError("capability need must contain 1..4096 characters")
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be 0..1")
        if reuse_similarity_threshold is not None and (
            isinstance(reuse_similarity_threshold, bool)
            or not isinstance(reuse_similarity_threshold, (int, float))
            or not math.isfinite(reuse_similarity_threshold)
            or not 0 <= reuse_similarity_threshold <= 1
        ):
            raise ValueError("reuse_similarity_threshold must be a finite value in 0..1")
        with self._lock:
            wanted = _validated_vector(self.embedder.embed(need))
            rows = self.db.execute(
                "SELECT capability_id,version,package_json,content_sha256,vector_json FROM capability_vectors"
            ).fetchall()
        matches = []
        for row_id, row_version, raw_package, stored_digest, raw_vector in rows:
            package, vector = json.loads(raw_package), _stored_vector(raw_vector)
            canonical_package = json.dumps(
                package, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            actual_digest = hashlib.sha256(canonical_package.encode()).hexdigest()
            capability = package.get("capability", {}) if isinstance(package, dict) else {}
            if (
                actual_digest != stored_digest
                or capability.get("id") != row_id
                or capability.get("version") != row_version
            ):
                raise ValueError(
                    "capability index manifest digest or identity mismatch; rebuild the index from the trusted catalog"
                )
            if len(vector) != len(wanted):
                raise ValueError("embedding dimension changed; rebuild the capability index with one model")
            similarity = sum(a * b for a, b in zip(wanted, vector))
            if similarity < min_similarity:
                continue
            contract_fit = evaluate_contract_fit(requirement, package, permitted_effects, permitted_permissions)
            cap = package.get("capability", {})
            status = cap.get("status", "experimental")
            runtime = (runtime_statuses or {}).get(f"{cap.get('id')}@{cap.get('version')}", {"active": False, "state": "not_checked"})
            reusable = reuse_similarity_threshold is not None and status in {"verified", "production", "production_certified"} and similarity >= reuse_similarity_threshold and contract_fit["compatible"] and runtime.get("active") is True
            decision = Decision.REUSE if reusable else Decision.EXTEND if similarity >= 0.55 else Decision.BUILD
            intent = cap.get("intent", "")
            blockers = []
            if status not in {"verified", "production", "production_certified"}:
                blockers.append("capability lifecycle is not verified or production")
            if reuse_similarity_threshold is None:
                blockers.append("no independently adjudicated, operator-pinned reuse threshold is configured")
            elif similarity < reuse_similarity_threshold:
                blockers.append("semantic similarity is below the calibrated reuse threshold")
            blockers.extend(contract_fit["reasons"])
            if runtime.get("active") is not True:
                blockers.append("no exact active runtime route was found in the authenticated AgentMesh snapshot")
            matches.append({"capability_id": cap.get("id"), "version": cap.get("version"), "status": status, "runtime": runtime, "reuse_blockers": blockers, "semantic_similarity": round(similarity, 6), "contract_fit": contract_fit, "decision": decision.value, "intent": intent[:512]})
        return sorted(matches, key=lambda row: (-row["semantic_similarity"], row["capability_id"] or ""))[:limit]

STOP = {"the", "and", "for", "from", "with", "this", "that", "when", "need", "find", "get", "data"}
def terms(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9_.-]+", text.lower()) if len(x) > 2 and x not in STOP}

def discover(need: str, catalog: Iterable[RegisteredCapability], limit: int = 10) -> list[CapabilityMatch]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    wanted = terms(need)
    ranked = []
    for entry in catalog:
        available = terms(entry.id + " " + entry.description)
        union = wanted | available
        score = len(wanted & available) / len(union) if union else 0.0
        if score:
            ranked.append(CapabilityMatch(entry, score))
    return sorted(ranked, key=lambda x: (-x.score, x.capability.id))[:limit]

def choose(matches: list[CapabilityMatch], threshold: float = .55) -> tuple[Decision, CapabilityMatch | None]:
    if matches and matches[0].score >= threshold and matches[0].capability.status in {"verified", "production", "production_certified"}:
        return Decision.REUSE, matches[0]
    if matches and matches[0].score >= .2:
        return Decision.EXTEND, matches[0]
    return Decision.BUILD, None

def draft_manifest(capability_id: str, description: str, source_session: str) -> dict:
    """Create review metadata only; no code, dependency, or permission is granted."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", capability_id):
        raise ValueError("capability id must be namespaced lowercase segments")
    return {"id": capability_id, "version": "0.1.0", "type": "scientific_tool", "status": "experimental", "description": description, "created_by": "medai-capability-builder", "created_from": source_session, "permissions": [], "execution": {"mode": "disabled_pending_review", "network": "deny", "filesystem": "read_only"}, "validation": {"security": "pending", "scientific": "pending", "review_required": True}}
