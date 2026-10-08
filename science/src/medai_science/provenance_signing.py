"""Installation-local HMAC signatures for immutable generated capability proposals."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
from typing import Any

import fcntl


_MIN_KEY_BYTES = 32


def _canonical_payload(proposal: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in proposal.items() if key != "provenance_signature"}
    try:
        return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("capability proposal is not canonical finite JSON") from error


def _key_path(key_file: str | Path | None = None) -> Path:
    value = key_file or os.getenv("MEDAI_PROVENANCE_KEY_FILE", ".medai/provenance-hmac.key")
    return Path(value).expanduser().resolve()


def load_or_create_signing_key(key_file: str | Path | None = None) -> bytes:
    """Read or atomically provision one private installation key; never expose it."""
    configured = os.getenv("MEDAI_PROVENANCE_HMAC_KEY")
    if configured:
        key = configured.encode("utf-8")
        if len(key) < _MIN_KEY_BYTES:
            raise ValueError("MEDAI_PROVENANCE_HMAC_KEY must contain at least 32 UTF-8 bytes")
        return key
    path = _key_path(key_file)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                    raise ValueError("provenance signing key must be a private regular file with mode 0600 or stricter")
                key = path.read_bytes()
            except FileNotFoundError:
                key = secrets.token_bytes(32)
                descriptor, temporary_name = tempfile.mkstemp(prefix=".provenance-key-", dir=path.parent)
                temporary = Path(temporary_name)
                try:
                    os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(key)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                    path.chmod(0o600)
                except Exception:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    temporary.unlink(missing_ok=True)
                    raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError("provenance signing key must contain at least 32 bytes")
    return key


def read_signing_key(key_file: str | Path | None = None) -> bytes | None:
    """Read an existing installation key without creating or changing files."""
    configured = os.getenv("MEDAI_PROVENANCE_HMAC_KEY")
    if configured:
        key = configured.encode("utf-8")
    else:
        path = _key_path(key_file)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("provenance signing key must be a private regular file with mode 0600 or stricter")
        key = path.read_bytes()
    if len(key) < _MIN_KEY_BYTES:
        raise ValueError("provenance signing key must contain at least 32 bytes")
    return key


def sign_capability_proposal(proposal: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        raise ValueError("capability proposal must be a JSON object")
    signing_key = key if key is not None else load_or_create_signing_key()
    if len(signing_key) < _MIN_KEY_BYTES:
        raise ValueError("provenance signing key must contain at least 32 bytes")
    signed = dict(proposal)
    signed.pop("provenance_signature", None)
    canonical = _canonical_payload(signed)
    signed["provenance_signature"] = {
        "algorithm": "HMAC-SHA256",
        "key_id": hashlib.sha256(signing_key).hexdigest()[:24],
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
        "signature_hex": hmac.new(signing_key, canonical, hashlib.sha256).hexdigest(),
        "scope": "installation-local integrity and origin; not an external identity or non-repudiation claim",
    }
    return signed


def verify_capability_proposal_signature(proposal: dict[str, Any], key: bytes | None = None) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        raise ValueError("capability proposal must be a JSON object")
    signature = proposal.get("provenance_signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "HMAC-SHA256":
        raise ValueError("generated capability proposal lacks a supported provenance signature")
    signing_key = key if key is not None else load_or_create_signing_key()
    if len(signing_key) < _MIN_KEY_BYTES:
        raise ValueError("provenance signing key must contain at least 32 bytes")
    canonical = _canonical_payload(proposal)
    expected_digest = hashlib.sha256(canonical).hexdigest()
    expected_signature = hmac.new(signing_key, canonical, hashlib.sha256).hexdigest()
    if signature.get("key_id") != hashlib.sha256(signing_key).hexdigest()[:24]:
        raise ValueError("generated capability proposal was signed by a different installation key")
    if not hmac.compare_digest(str(signature.get("payload_sha256", "")), expected_digest):
        raise ValueError("generated capability proposal provenance payload hash does not match")
    if not hmac.compare_digest(str(signature.get("signature_hex", "")), expected_signature):
        raise ValueError("generated capability proposal provenance signature is invalid")
    return {"verified": True, "algorithm": "HMAC-SHA256", "key_id": signature["key_id"], "payload_sha256": expected_digest}
