"""Exercise the host backup cycle with a deterministic fake MedAI binary."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


_BACKUP_SCRIPT = Path(__file__).parents[2] / "deploy/medai-backup.sh"


class BackupScriptTests(unittest.TestCase):
    def _run_backup(self, *, fail_on: str | None = None) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.TemporaryDirectory(prefix="medai-backup-test-") as temporary:
            root = Path(temporary)
            database = root / "database.sqlite"
            database.write_bytes(b"sqlite placeholder")
            backup_dir = root / "backups with spaces"
            call_log = root / "calls.log"
            binary = root / "medai-mock"
            binary.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$MEDAI_TEST_LOG\"\n"
                "if [ \"${MEDAI_FAIL_ON:-}\" = verify ]; then\n"
                "  case \"$*\" in *\"database verify\"*) exit 23 ;; esac\n"
                "fi\n",
                encoding="utf-8",
            )
            binary.chmod(0o700)
            flock = root / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            flock.chmod(0o700)
            env = os.environ.copy()
            env.update({
                "MEDAI_DB": str(database),
                "MEDAI_BACKUP_DIR": str(backup_dir),
                "MEDAI_BACKUP_KEEP_NEWEST": "3",
                "MEDAI_BACKUP_MAX_AGE_DAYS": "30",
                "MEDAI_BIN": str(binary),
                "MEDAI_TEST_LOG": str(call_log),
                "MEDAI_FAIL_ON": "verify" if fail_on else "",
                "PATH": f"{root}:/usr/bin:/bin",
            })
            result = subprocess.run(["/bin/sh", str(_BACKUP_SCRIPT)], env=env, text=True, capture_output=True, check=False)
            calls = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
            return result, calls

    def test_verified_backup_precedes_retention_and_preserves_quoted_paths(self):
        result, calls = self._run_backup()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = calls.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("--db "))
        self.assertIn("database backup ", lines[0])
        self.assertTrue(lines[1].startswith("database verify "))
        self.assertTrue(lines[2].startswith("database prune-backups "))
        self.assertIn("backups with spaces", lines[2])

    def test_verification_failure_prevents_retention(self):
        result, calls = self._run_backup(fail_on="database verify")
        self.assertEqual(result.returncode, 23, f"stdout={result.stdout!r} stderr={result.stderr!r} calls={calls!r}")
        self.assertEqual(len(calls.splitlines()), 2)
        self.assertIn("database verify", calls)
        self.assertNotIn("prune-backups", calls)

    def test_alert_sanitizes_unit_name_before_logging_and_json_encoding(self):
        script = Path(__file__).parents[2] / "deploy/medai-backup-alert.sh"
        with tempfile.TemporaryDirectory(prefix="medai-alert-test-") as temporary:
            root = Path(temporary)
            logger_log = root / "logger.log"
            curl_args = root / "curl.args"
            logger = root / "logger"
            logger.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$MEDAI_TEST_LOGGER\"\n", encoding="utf-8")
            logger.chmod(0o700)
            curl = root / "curl"
            curl.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$MEDAI_TEST_CURL_ARGS\"\nprintf 200\n", encoding="utf-8")
            curl.chmod(0o700)
            env = os.environ.copy()
            env.update({
                "PATH": f"{root}:/usr/bin:/bin",
                "MEDAI_TEST_LOGGER": str(logger_log),
                "MEDAI_TEST_CURL_ARGS": str(curl_args),
                "MEDAI_BACKUP_ALERT_WEBHOOK": "https://alerts.example.invalid/api/v2/alerts",
            })
            malicious_unit = 'bad"unit\nforged-entry'
            result = subprocess.run(["/bin/sh", str(script), malicious_unit], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            logged = logger_log.read_text(encoding="utf-8")
            self.assertIn("unknown-unit", logged)
            self.assertEqual(len(logged.splitlines()), 1)
            self.assertNotIn("forged-entry", logged)
            args = curl_args.read_text(encoding="utf-8").splitlines()
            payload = json.loads(args[args.index("--data") + 1])
            alert = payload["alerts"][0]
            self.assertEqual(alert["labels"]["alertname"], "MedAIBackupFailed")
            self.assertIn("unknown-unit", alert["annotations"]["description"])


if __name__ == "__main__":
    unittest.main()
