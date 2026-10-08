from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medai_science.provenance_signing import (
    load_or_create_signing_key,
    sign_capability_proposal,
    verify_capability_proposal_signature,
)


class ProvenanceSigningTests(unittest.TestCase):
    def test_provisions_private_persistent_key_and_verifies_signed_proposal(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            key_path = Path(temp) / "nested" / "key.bin"
            first_key = load_or_create_signing_key(key_path)
            second_key = load_or_create_signing_key(key_path)
            self.assertEqual(first_key, second_key)
            self.assertGreaterEqual(len(first_key), 32)
            self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(key_path.parent.stat().st_mode & 0o777, 0o700)
            proposal = {"manifest": {"id": "genomics.variant_summary", "version": "0.1.0"}, "source": "def run(input):\n    return {}\n", "execution": {"enabled": False}}
            signed = sign_capability_proposal(proposal, first_key)
            receipt = verify_capability_proposal_signature(signed, second_key)
            self.assertTrue(receipt["verified"])
            self.assertEqual(signed["provenance_signature"]["algorithm"], "HMAC-SHA256")
            self.assertEqual(signed["source"], proposal["source"])

    def test_rejects_tampering_wrong_key_missing_signature_and_weak_key(self):
        key = b"a" * 32
        signed = sign_capability_proposal({"source": "def run(input): return {}"}, key)
        tampered = {**signed, "source": "def run(input): return {'x': 1}"}
        with self.assertRaisesRegex(ValueError, "payload hash"):
            verify_capability_proposal_signature(tampered, key)
        with self.assertRaisesRegex(ValueError, "different installation key"):
            verify_capability_proposal_signature(signed, b"b" * 32)
        with self.assertRaisesRegex(ValueError, "lacks a supported"):
            verify_capability_proposal_signature({"source": "code"}, key)
        with self.assertRaisesRegex(ValueError, "at least 32"):
            sign_capability_proposal({"source": "code"}, b"weak")

    def test_rejects_insecure_existing_key_permissions(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            path = Path(temp) / "key.bin"
            path.write_bytes(b"k" * 32)
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "private regular file"):
                load_or_create_signing_key(path)

    def test_environment_key_requires_minimum_length_without_touching_filesystem(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"MEDAI_PROVENANCE_HMAC_KEY": "too-short", "MEDAI_PROVENANCE_KEY_FILE": str(Path(temp) / "ignored")}, clear=True):
            with self.assertRaisesRegex(ValueError, "at least 32"):
                load_or_create_signing_key()
            self.assertFalse((Path(temp) / "ignored").exists())


if __name__ == "__main__":
    unittest.main()
