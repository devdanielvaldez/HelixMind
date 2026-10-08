"""Journaled, revision-checked AgentMesh route synchronization with rollback."""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from medai_science.agentmesh_control import AgentMeshControlPlaneClient
from medai_science.provenance_signing import load_or_create_signing_key, read_signing_key, sign_capability_proposal, verify_capability_proposal_signature


_RESOURCE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_MAX_SYNC_STATE_BYTES = 64_000_000


def _history_with(state: dict[str, Any] | None, event: dict[str, Any]) -> list[dict[str, Any]]:
    existing = state.get("history", []) if state else []
    if not isinstance(existing, list) or any(not isinstance(row, dict) for row in existing):
        raise ValueError("AgentMesh sync audit history is malformed")
    return [*existing[-99:], {"occurred_at": int(time.time()), **event}]


class AgentMeshSyncError(RuntimeError):
    """A sync failure, including whether the prior desired state was restored."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _resource_key(resource: dict[str, Any]) -> str:
    return str(resource["kind"]) + "/" + str(resource["name"])


def route_plan_review_digest(plan: dict[str, Any]) -> str:
    """Hash every review-relevant plan field, including gateway and lifecycle provenance."""
    unsigned = {key: value for key, value in plan.items() if key != "plan_review_sha256"}
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def seal_route_plan_for_review(
    plan: dict[str, Any], tenant: str = "medai", namespace: str = "biomedical",
) -> dict[str, Any]:
    """Bind approval to the exact AgentMesh tenant/namespace before review."""
    if not isinstance(tenant, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", tenant):
        raise ValueError("AgentMesh tenant scope must be a safe 1..128 character identifier")
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", namespace):
        raise ValueError("AgentMesh namespace scope must be a safe 1..128 character identifier")
    sealed = dict(plan)
    sealed["control_plane_scope"] = {"tenant": tenant, "namespace": namespace}
    sealed["plan_review_sha256"] = route_plan_review_digest(sealed)
    return sealed


def _active_map(resources: list[dict[str, Any]], *, medai_only: bool = False) -> dict[str, dict[str, Any]]:
    result = {}
    for resource in resources:
        if not isinstance(resource, dict) or set(resource) != {"kind", "name", "state", "spec"}:
            raise ValueError("AgentMesh resource snapshot contains an invalid resource")
        if medai_only and not str(resource["name"]).startswith("medai-"):
            continue
        if resource["state"] in {"active", "draining"}:
            result[_resource_key(resource)] = resource
    return result


def _read_state(
    path: Path, tenant: str, namespace: str, signing_key: bytes | None, *, allow_recovery_required: bool = False,
) -> dict[str, Any] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("AgentMesh sync state must be a private regular file with mode 0600 or stricter")
    if info.st_size > _MAX_SYNC_STATE_BYTES:
        raise ValueError("AgentMesh sync state exceeds 64 MB")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("AgentMesh sync state is unreadable or malformed") from error
    if not isinstance(state, dict) or state.get("schema_version") != "1":
        raise ValueError("AgentMesh sync state has an unsupported schema")
    verify_capability_proposal_signature(state, signing_key)
    if state.get("tenant") != tenant or state.get("namespace") != namespace:
        raise ValueError("AgentMesh sync state belongs to a different control-plane scope")
    if state.get("recovery_required") is True and not allow_recovery_required:
        raise ValueError("AgentMesh sync state is marked recovery_required; inspect and reconcile the control plane before retrying")
    resources = state.get("resources")
    if not isinstance(resources, dict) or len(resources) > 50_000:
        raise ValueError("AgentMesh sync state resource journal is malformed or oversized")
    history = state.get("history", [])
    if not isinstance(history, list) or len(history) > 100 or any(not isinstance(row, dict) for row in history):
        raise ValueError("AgentMesh sync audit history is malformed or oversized")
    for key, row in resources.items():
        if not isinstance(key, str) or not isinstance(row, dict) or set(row) != {"resource", "revision"}:
            raise ValueError("AgentMesh sync state contains a malformed resource revision")
        resource, revision = row["resource"], row["revision"]
        if not isinstance(resource, dict) or _resource_key(resource) != key or isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("AgentMesh sync state contains an invalid resource or revision")
    return state


async def inspect_capability_sync(
    client: AgentMeshControlPlaneClient,
    state_path: str | Path,
    *,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Compare the signed MedAI route journal with live AgentMesh state without mutation."""
    path = Path(state_path).expanduser()
    if signing_key is None:
        signing_key = read_signing_key()
    try:
        path.lstat()
        state_exists = True
    except FileNotFoundError:
        state_exists = False
    if state_exists and signing_key is None:
        raise ValueError("cannot verify the signed AgentMesh sync journal without its existing provenance key")
    state = _read_state(path, client.tenant, client.namespace, signing_key, allow_recovery_required=True)
    snapshot = await client.snapshot()
    journal = state.get("resources", {}) if state else {}
    expected = {
        key: row["resource"] for key, row in journal.items()
        if row["resource"].get("state") in {"active", "draining"}
    }
    actual = _active_map(snapshot["resources"], medai_only=True)
    missing = sorted(expected.keys() - actual.keys())
    unmanaged = sorted(actual.keys() - expected.keys())
    changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])
    recovery_required = bool(state and state.get("recovery_required") is True)
    drift = bool(missing or unmanaged or changed)
    if recovery_required:
        status = "recovery_required"
    elif state is None and not actual:
        status = "uninitialized"
    elif state is None:
        status = "unjournaled_routes"
    else:
        status = "drift" if drift else "in_sync"
    return {
        "tenant": client.tenant,
        "namespace": client.namespace,
        "status": status,
        "state_file_present": state is not None,
        "recovery_required": recovery_required,
        "snapshot_revision": snapshot["revision"],
        "snapshot_digest": snapshot["digest"],
        "journaled_active_count": len(expected),
        "runtime_active_count": len(actual),
        "runtime_active_routes": [actual[key] for key in sorted(actual)],
        "managed_route_names": sorted({
            row["resource"]["name"] for row in journal.values()
            if row["resource"].get("kind") == "route" and str(row["resource"].get("name", "")).startswith("medai-")
        }),
        "reviewed_lifecycle_registry_sha256": state.get("lifecycle_registry_sha256") if state else None,
        "last_approval_receipt_id": (state.get("approval_receipt") or {}).get("receipt_id") if state else None,
        "last_approved_by": (state.get("approval_receipt") or {}).get("actor") if state else None,
        "approval_receipt": state.get("approval_receipt") if state else None,
        "missing_runtime_routes": missing,
        "untracked_runtime_routes": unmanaged,
        "changed_runtime_routes": changed,
        "mutation_performed": False,
        "automatic_recovery": False,
    }


def reconcile_runtime_lifecycle(
    sync_report: dict[str, Any], registry_rows: list[dict[str, Any]], registry_sha256: str,
) -> dict[str, Any]:
    """Compare live AgentMesh routes with the current authenticated MedAI lifecycle registry."""
    if not isinstance(registry_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", registry_sha256):
        raise ValueError("MedAI lifecycle registry digest is malformed")
    if not isinstance(registry_rows, list) or len(registry_rows) > 1000:
        raise ValueError("MedAI lifecycle registry rows are malformed or oversized")
    known_routes: dict[str, tuple[str, str, str]] = {}
    for row in registry_rows:
        if not isinstance(row, dict):
            raise ValueError("MedAI lifecycle registry contains a malformed record")
        capability_id, version, status = row.get("id"), row.get("version"), row.get("status")
        kind = row.get("kind")
        if not isinstance(capability_id, str) or not isinstance(version, str) or not isinstance(status, str) or not isinstance(kind, str):
            raise ValueError("MedAI lifecycle registry contains a malformed capability identity or status")
        if kind != "mcp":
            continue
        key = f"{capability_id}@{version}"
        route_name = "medai-" + re.sub(r"[^A-Za-z0-9._-]", "-", key)
        if route_name in known_routes:
            raise ValueError("distinct MedAI capabilities collide on an AgentMesh route name")
        known_routes[route_name] = (capability_id, version, status)

    active_rows = sync_report.get("runtime_active_routes")
    if not isinstance(active_rows, list):
        raise ValueError("AgentMesh sync report omitted active route records")
    actual_names = set()
    for resource in active_rows:
        if not isinstance(resource, dict) or resource.get("kind") != "route" or not isinstance(resource.get("name"), str):
            raise ValueError("AgentMesh sync report contains a malformed active route")
        if resource["name"].startswith("medai-"):
            actual_names.add(resource["name"])
    managed_names = sync_report.get("managed_route_names")
    if not isinstance(managed_names, list) or any(not isinstance(name, str) or not name.startswith("medai-") for name in managed_names):
        raise ValueError("AgentMesh sync report omitted managed route identities")
    managed_names = set(managed_names)

    approved_names = {
        name for name, (_capability_id, _version, status) in known_routes.items()
        if status in {"verified", "production", "production_certified"}
    }
    stale_active = [
        {"route_name": name, "capability_id": known_routes[name][0], "version": known_routes[name][1], "current_status": known_routes[name][2]}
        for name in sorted(actual_names & known_routes.keys())
        if known_routes[name][2] not in {"verified", "production", "production_certified"}
    ]
    unregistered_active = sorted(actual_names - known_routes.keys())
    missing_approved = sorted((approved_names & managed_names) - actual_names)
    approved_not_planned = sorted(approved_names - managed_names)
    drift = bool(stale_active or unregistered_active or missing_approved)
    previous_digest = sync_report.get("reviewed_lifecycle_registry_sha256")
    return {
        "status": "drift" if drift else "in_sync",
        "registry_sha256": registry_sha256,
        "reviewed_registry_sha256": previous_digest,
        "registry_changed_since_rollout": bool(previous_digest and previous_digest != registry_sha256),
        "stale_active_routes": stale_active,
        "unregistered_active_routes": unregistered_active,
        "missing_approved_routes": missing_approved,
        "approved_routes_not_in_plan": approved_not_planned,
        "mutation_performed": False,
    }


def _write_state(path: Path, state: dict[str, Any], signing_key: bytes | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_info = path.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_mode & 0o077:
        raise ValueError("AgentMesh sync state directory must be private with mode 0700 or stricter")
    signed = sign_capability_proposal(state, signing_key)
    payload = (_canonical(signed).decode("utf-8") + "\n").encode("utf-8")
    if len(payload) > _MAX_SYNC_STATE_BYTES:
        raise ValueError("AgentMesh sync state exceeds 64 MB")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".agentmesh-sync-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        path.chmod(0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_route_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(plan, dict) or plan.get("schema_version") != "1" or not isinstance(plan.get("resources"), list):
        raise ValueError("capability route plan has an unsupported or malformed schema")
    resources = plan["resources"]
    if len(resources) > 50_000 or plan.get("resource_count") != len(resources):
        raise ValueError("capability route plan resource count is invalid or exceeds 50000")
    expected_digest = hashlib.sha256(_canonical(resources)).hexdigest()
    if plan.get("desired_sha256") != expected_digest:
        raise ValueError("capability route plan digest does not match its resources")
    if plan.get("plan_review_sha256") != route_plan_review_digest(plan):
        raise ValueError("capability route plan review digest is missing or does not match")
    service, strategy = plan.get("service"), plan.get("strategy")
    if not isinstance(service, str) or not _RESOURCE_NAME.fullmatch(service) or not isinstance(strategy, str) or strategy not in {"least_active", "round_robin", "weighted"}:
        raise ValueError("capability route plan service or strategy is invalid")
    config_digest = plan.get("gateway_config_sha256")
    if not isinstance(config_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", config_digest):
        raise ValueError("route plan must bind the exact MedAI gateway config digest")
    override_digest = plan.get("lifecycle_overrides_sha256")
    if override_digest is not None and (not isinstance(override_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", override_digest)):
        raise ValueError("route plan lifecycle-override digest is invalid")
    upstream = plan.get("gateway_upstream")
    if not isinstance(upstream, dict) or set(upstream) != {"url", "request_timeout_ms"}:
        raise ValueError("route plan gateway upstream metadata is malformed")
    url = upstream.get("url")
    parsed_url = urlsplit(url) if isinstance(url, str) else None
    timeout_ms = upstream.get("request_timeout_ms")
    if (
        parsed_url is None or parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname
        or parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment
        or isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 1 <= timeout_ms <= 300_000
    ):
        raise ValueError("route plan gateway upstream URL or timeout is invalid")
    if parsed_url.scheme == "http":
        try:
            is_loopback = ipaddress.ip_address(parsed_url.hostname).is_loopback
        except ValueError:
            is_loopback = parsed_url.hostname.casefold() == "localhost"
        if not is_loopback:
            raise ValueError("cleartext remote gateway upstream is forbidden")
    seen = set()
    for resource in resources:
        if not isinstance(resource, dict) or set(resource) != {"kind", "name", "state", "spec"}:
            raise ValueError("capability route plan contains a malformed desired resource")
        if not isinstance(resource["name"], str) or not resource["name"].startswith("medai-"):
            raise ValueError("capability sync accepts only MedAI-owned route names")
        if resource["state"] not in {"active", "disabled"} or resource["kind"] != "route":
            raise ValueError("capability sync accepts only active or disabled MedAI routes")
        spec = resource["spec"]
        if (
            not isinstance(spec, dict)
            or set(spec) != {"service", "capability", "strategy"}
            or spec.get("service") != service
            or spec.get("strategy") != strategy
            or not isinstance(spec.get("capability"), str)
            or not _RESOURCE_NAME.fullmatch(spec["capability"])
        ):
            raise ValueError("route resource service, capability, and strategy must match the reviewed plan")
        key = _resource_key(resource)
        if key in seen:
            raise ValueError("capability route plan contains duplicate resources")
        seen.add(key)
    return resources


async def _rollback_applied(
    client: AgentMeshControlPlaneClient,
    applied: list[dict[str, Any]],
    journal: dict[str, Any],
    before_snapshot: dict[str, Any],
) -> list[str]:
    errors = []
    for item in reversed(applied):
        previous = item["previous"]
        rollback_resource = previous["resource"] if previous else {**item["resource"], "state": "disabled"}
        try:
            response = await client.apply(rollback_resource, item["revision"])
            revision = response.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("rollback apply response omitted a valid revision")
            journal[item["key"]] = {"resource": rollback_resource, "revision": revision}
        except Exception as error:  # preserve all partial recovery details for a human operator
            errors.append(f"{item['key']}: {type(error).__name__}: {error}"[:1000])
    try:
        await client.reconcile()
        restored = await client.snapshot()
        if _active_map(restored["resources"]) != _active_map(before_snapshot["resources"]):
            errors.append("restored snapshot does not match the pre-sync snapshot")
    except Exception as error:
        errors.append(f"reconcile verification: {type(error).__name__}: {error}"[:1000])
    return errors


async def sync_capability_routes(
    client: AgentMeshControlPlaneClient,
    plan: dict[str, Any],
    state_path: str | Path,
    *,
    approved_plan_sha256: str,
    approval_receipt: dict[str, Any] | None = None,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Apply one reviewed MedAI route plan, reconcile it, verify the snapshot, and roll back failures."""
    scope = plan.get("control_plane_scope") if isinstance(plan, dict) else None
    if scope != {"tenant": client.tenant, "namespace": client.namespace}:
        raise ValueError("reviewed route plan is bound to a different AgentMesh tenant/namespace")
    desired_rows = _validate_route_plan(plan)
    if approved_plan_sha256 != plan["plan_review_sha256"]:
        raise ValueError("approved plan digest does not match the reviewed route plan")
    if approval_receipt is None:
        raise ValueError("authenticated MedAI approval receipt is required before AgentMesh route mutation")
    if approval_receipt is not None:
        scope = plan["control_plane_scope"]
        now = int(time.time())
        if (
            not isinstance(approval_receipt.get("receipt_id"), str)
            or not approval_receipt.get("receipt_id")
            or not isinstance(approval_receipt.get("actor"), str)
            or not approval_receipt.get("actor")
            or not isinstance(approval_receipt.get("medai_tenant_id"), str)
            or not approval_receipt.get("medai_tenant_id")
            or approval_receipt.get("agentmesh_tenant") != scope["tenant"]
            or approval_receipt.get("agentmesh_namespace") != scope["namespace"]
            or approval_receipt.get("plan_review_sha256") != plan["plan_review_sha256"]
            or approval_receipt.get("desired_sha256") != plan["desired_sha256"]
            or isinstance(approval_receipt.get("expires_at"), bool)
            or not isinstance(approval_receipt.get("expires_at"), int)
            or approval_receipt["expires_at"] < now
        ):
            raise ValueError("authenticated MedAI approval receipt is expired or does not match this exact plan and scope")
    raw_state_path = Path(state_path).expanduser()
    path = raw_state_path.parent.resolve() / raw_state_path.name
    # Provision signing and writable private journal storage before any remote mutation.
    if signing_key is None:
        signing_key = load_or_create_signing_key()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent_info = path.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_mode & 0o077:
        raise ValueError("AgentMesh sync state directory must be private with mode 0700 or stricter")
    descriptor, probe_name = tempfile.mkstemp(prefix=".agentmesh-sync-probe-", dir=path.parent)
    os.close(descriptor)
    Path(probe_name).unlink()
    state = _read_state(path, client.tenant, client.namespace, signing_key)
    effective_approval = approval_receipt or (state.get("approval_receipt") if state else None)
    history = state.get("history", []) if state else []
    before = await client.snapshot()
    before_resources = before["resources"]
    before_all = _active_map(before_resources)
    before_medai = _active_map(before_resources, medai_only=True)
    prior = state.get("resources", {}) if state else {}
    if state is None and before_medai:
        raise ValueError("MedAI routes already exist but no signed sync journal is available; adopt them through reviewed control-plane state first")
    prior_active = {key: row["resource"] for key, row in prior.items() if row["resource"].get("state") in {"active", "draining"}}
    if prior_active != before_medai:
        raise ValueError("current AgentMesh MedAI routes differ from the signed sync journal; reconcile drift before applying changes")

    desired = {_resource_key(row): row for row in desired_rows}
    # Retire routes previously managed by this synchronizer but removed from the reviewed plan.
    for key, row in prior.items():
        if key not in desired:
            desired[key] = {**row["resource"], "state": "disabled"}

    next_journal = {key: dict(row) for key, row in prior.items()}
    applied: list[dict[str, Any]] = []
    changed = []
    for key, resource in sorted(desired.items()):
        previous = prior.get(key)
        if previous and previous["resource"] == resource:
            continue
        try:
            response = await client.apply(resource, previous["revision"] if previous else None)
            revision = response.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("AgentMesh apply response omitted a valid desired-state revision")
            applied.append({"key": key, "previous": previous, "resource": resource, "revision": revision})
            next_journal[key] = {"resource": resource, "revision": revision}
            changed.append(key)
        except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
            # Create conflicts are definitive: an unjournaled desired resource already exists.
            known_conflict = previous is None and isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 409
            rollback_errors = await _rollback_applied(client, applied, next_journal, before) if applied else []
            uncertain = not known_conflict
            state_body = {
                "schema_version": "1", "tenant": client.tenant, "namespace": client.namespace,
                "resources": next_journal, "recovery_required": uncertain or bool(rollback_errors),
                "pending_resource": None if known_conflict else resource,
                "pending_resource_key": None if known_conflict else key,
                "last_rollback_errors": rollback_errors,
                "last_error": f"{type(error).__name__}: {error}"[:2000],
                "history": _history_with(state, {
                    "outcome": "unmanaged_resource_conflict" if known_conflict else "recovery_required",
                    "plan_review_sha256": plan["plan_review_sha256"],
                    "approval_receipt_id": approval_receipt.get("receipt_id") if approval_receipt else None,
                    "approved_by": approval_receipt.get("actor") if approval_receipt else None,
                    "desired_sha256": plan["desired_sha256"],
                    "resource_key": key,
                    "confirmed_writes_rolled_back": len(applied) - len(rollback_errors),
                    "rollback_errors": rollback_errors,
                }),
            }
            _write_state(path, state_body, signing_key)
            if known_conflict:
                raise AgentMeshSyncError(f"unmanaged AgentMesh desired resource already exists at {key}; confirmed writes were rolled back; inspect or remove the existing resource before retrying") from error
            state_text = "earlier writes were rolled back" if not rollback_errors else "rollback was incomplete"
            raise AgentMeshSyncError(f"AgentMesh apply outcome is uncertain for {key}; {state_text}; sync journal marked recovery_required at {path}") from error

    target_active = {
        **{key: row for key, row in before_all.items() if not row["name"].startswith("medai-")},
        **{key: row for key, row in desired.items() if row["state"] in {"active", "draining"}},
    }
    try:
        if changed:
            await client.reconcile()
        after = await client.snapshot()
        after_map = _active_map(after["resources"])
        if after_map != target_active:
            raise ValueError("reconciled AgentMesh snapshot does not match the reviewed MedAI route plan")
    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
        rollback_errors = await _rollback_applied(client, applied, next_journal, before)
        _write_state(path, {
            "schema_version": "1", "tenant": client.tenant, "namespace": client.namespace,
            "resources": next_journal, "recovery_required": bool(rollback_errors),
            "last_rollback_errors": rollback_errors, "last_error": f"{type(error).__name__}: {error}"[:2000],
            "history": _history_with(state, {
                "outcome": "recovery_required" if rollback_errors else "rolled_back",
                "plan_review_sha256": plan["plan_review_sha256"],
                "approval_receipt_id": approval_receipt.get("receipt_id") if approval_receipt else None,
                "approved_by": approval_receipt.get("actor") if approval_receipt else None,
                "desired_sha256": plan["desired_sha256"],
                "before_snapshot_digest": before["digest"],
                "changed_resources": changed,
                "rollback_errors": rollback_errors,
            }),
        }, signing_key)
        state_text = "rollback failed; inspect sync journal" if rollback_errors else "previous active snapshot restored"
        raise AgentMeshSyncError(f"AgentMesh rollout failed; {state_text}: {error}") from error

    final_state = {
        "schema_version": "1", "tenant": client.tenant, "namespace": client.namespace,
        "resources": next_journal, "recovery_required": False,
        "last_plan_sha256": plan["desired_sha256"], "last_snapshot_revision": after["revision"],
        "last_snapshot_digest": after["digest"], "last_changed_resources": changed,
        "lifecycle_registry_sha256": plan.get("lifecycle_registry_sha256"),
        "approved_plan_sha256": approved_plan_sha256,
        "approval_receipt": effective_approval,
        "history": _history_with(state, {
            "outcome": "applied" if changed else "no_change",
            "plan_review_sha256": approved_plan_sha256,
            "approval_receipt_id": effective_approval.get("receipt_id") if effective_approval else None,
            "approved_by": effective_approval.get("actor") if effective_approval else None,
            "desired_sha256": plan["desired_sha256"],
            "before_snapshot_digest": before["digest"],
            "after_snapshot_digest": after["digest"],
            "changed_resources": changed,
        }),
    }
    _write_state(path, final_state, signing_key)
    return {
        "tenant": client.tenant, "namespace": client.namespace,
        "plan_sha256": plan["desired_sha256"], "before_revision": before["revision"],
        "after_revision": after["revision"], "snapshot_digest": after["digest"],
        "approved_plan_sha256": approved_plan_sha256,
        "approval_receipt_id": effective_approval.get("receipt_id") if effective_approval else None,
        "approved_by": effective_approval.get("actor") if effective_approval else None,
        "changed_resources": changed, "changed_count": len(changed),
        "rolled_back": False, "recovery_required": False, "audit_event_count": len(final_state["history"]),
        "state_path": str(path), "review_required": True,
    }
