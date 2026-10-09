import tempfile
import unittest
import os
from pathlib import Path
import sys
from unittest.mock import patch

from medai_science.r_sandbox import DockerRSandbox, validate_r_source
from medai_science.sandbox import SandboxError


class RSandboxTests(unittest.TestCase):
    def test_source_policy_requires_one_supported_entrypoint_and_denies_obvious_escape_calls(self):
        receipt = validate_r_source("run <- function(input) { list(value = input$value + 1L) }\n")
        self.assertEqual(receipt["source_bytes"], len("run <- function(input) { list(value = input$value + 1L) }\n".encode()))
        self.assertEqual(len(receipt["source_sha256"]), 64)
        for source, expected in (
            ("other <- function(input) list()", "exactly one"),
            ("run <- function(input) list()\nrun <- function(input) list()", "exactly one"),
            ("run <- function(input, other) list()", "exactly one"),
            ("run <- function(input) system('id')", "denied function: system"),
            ("run <- function(input) download.file('https://example.invalid', tempfile())", "denied function"),
            ("run <- function(input) eval(parse(text='1'))", "denied function"),
        ):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                validate_r_source(source)

    def test_requires_digest_pinned_image_and_cases(self):
        with self.assertRaisesRegex(ValueError, "immutable sha256"):
            DockerRSandbox("r-base:4.4")
        sandbox = DockerRSandbox("r-base:4.4@sha256:" + "a" * 64)
        self.assertEqual(DockerRSandbox("sha256:" + "b" * 64).image, "sha256:" + "b" * 64)
        with self.assertRaisesRegex(ValueError, "1..100"):
            sandbox.execute("run <- function(input) list()", {}, [])

    def test_mock_docker_contract_enforces_container_limits_and_checks_r_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n"
                "    print('sha256:local')\n"
                "else:\n"
                "    required=['--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--memory=512m','--cpus=1.0','--entrypoint=Rscript','--ulimit=nofile=64:64','--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=64m']\n"
                "    assert all(x in args for x in required), args\n"
                "    assert args[-5] == '--vanilla', args\n"
                "    volume=args[args.index('--volume')+1]\n"
                "    assert volume.endswith(':/capability:ro,noexec,nosuid,nodev'), volume\n"
                "    host=volume.split(':/capability:',1)[0]\n"
                "    cidfile=next(x.split('=',1)[1] for x in args if x.startswith('--cidfile='))\n"
                "    assert not cidfile.startswith(host+os.sep), 'host container ID file must stay outside guest mount'\n"
                "    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+json.dumps({'ok':True,'result':{'value':3},'test_results':[{'result':{'value':5},'wall_time_microseconds':17}],'cpu_time_microseconds':2345}))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            sandbox = DockerRSandbox("r-base:4.4@sha256:" + "b" * 64, docker_binary=str(docker))
            result = sandbox.execute(
                "run <- function(input) list(value = input$value + 1L)\n",
                {"value": 2},
                [{"input": {"value": 4}, "expected": {"value": 5}}],
                contract={
                    "input": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
                    "output": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
                },
            )
        self.assertTrue(result["case_tests_passed"])
        self.assertEqual(result["runner"], "docker-no-network-r")
        self.assertEqual(result["cpu_time_microseconds"], 2345)
        self.assertEqual(result["validation"]["cases"][0]["wall_time_microseconds"], 17)
        self.assertEqual(result["result"], {"value": 3})
        self.assertEqual(result["lifecycle_status"], "experimental; no automatic registration or promotion")

    def test_mismatched_r_case_is_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n    print('sha256:local')\n"
                "else:\n    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+json.dumps({'ok':True,'result':{'value':3},'test_results':[{'result':{'value':6},'wall_time_microseconds':8}],'cpu_time_microseconds':1}))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            result = DockerRSandbox("r-base:4.4@sha256:" + "c" * 64, docker_binary=str(docker)).execute(
                "run <- function(input) list(value = 1L)", {}, [{"input": {}, "expected": {"value": 5}}]
            )
        self.assertFalse(result["case_tests_passed"])
        self.assertEqual(result["validation"]["failed"], 1)

    def test_rejects_ambiguous_or_nonstandard_container_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import os, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n    print('sha256:local')\n"
                "else:\n    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+os.environ['MOCK_RECEIPT'])\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            sandbox = DockerRSandbox("r-base:4.4@sha256:" + "d" * 64, docker_binary=str(docker))
            for receipt in (
                '{"ok":true,"ok":false,"result":{},"cpu_time_microseconds":1}',
                '{"ok":true,"result":{"value":Infinity},"cpu_time_microseconds":1}',
            ):
                with self.subTest(receipt=receipt), patch.dict(os.environ, {"MOCK_RECEIPT": receipt}):
                    with self.assertRaisesRegex(SandboxError, "no valid result receipt"):
                        sandbox.execute("run <- function(input) list()", {}, [], require_test_cases=False)

    def test_successful_null_result_cannot_bypass_output_schema_without_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n    print('sha256:local')\n"
                "else:\n    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+json.dumps({'ok':True,'result':None,'test_results':[],'cpu_time_microseconds':1}))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            sandbox = DockerRSandbox("r-base:4.4@sha256:" + "e" * 64, docker_binary=str(docker))
            with self.assertRaisesRegex(SandboxError, "violates its output contract"):
                sandbox.execute(
                    "run <- function(input) list(value = 1L)",
                    {},
                    [],
                    contract={"input": {"type": "object"}, "output": {"type": "object"}},
                    require_test_cases=False,
                )


if __name__ == "__main__":
    unittest.main()
