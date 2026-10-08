import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys
import threading
import time

from medai_science.sandbox import DockerPythonSandbox, SandboxCancelledError, SandboxError, _RESULT_PREFIX, _RUNNER_SOURCE, _run_limited, capability_validation_request_hash, validate_capability_run_request, validate_python_source


class PythonCapabilitySandboxTests(unittest.TestCase):
    def test_static_policy_requires_run_and_rejects_import_and_builtin_escapes(self):
        valid = validate_python_source("def run(input):\n    return {'value': input['value'] + 1}\n")
        self.assertEqual(len(valid["source_sha256"]), 64)
        for source, message in (
            ("def run(input):\n    import socket\n    return {}\n", "socket"),
            ("def run(input):\n    return open('/etc/passwd').read()\n", "open"),
            ("def run(input):\n    return getattr(__builtins__, '__import__')('os').getcwd()\n", "denied Python builtin: getattr"),
            ("def run(input):\n    return input.__class__\n", "denied Python dunder attribute"),
            ('def run(input):\n    return "{0.__class__}".format(1)\n', "denied reflective string formatting: format"),
            ('def run(input):\n    return "{value.__class__}".format_map(input)\n', "denied reflective string formatting: format_map"),
            ("loader = __import__\ndef run(input):\n    return {}\n", "denied Python builtin: __import__"),
            ("def run(input):\n    return __builtins__['__import__']('os').getcwd()\n", "denied Python builtin: __builtins__"),
            ("async def run(input):\n    return {}\n", "synchronous"),
            ("def other(input):\n    return {}\n", "top-level run"),
            ("def run():\n    return {}\n", "signature run"),
            ("def run(value):\n    return {}\n", "signature run"),
            ("def run(input, extra):\n    return {}\n", "signature run"),
            ("def run(input):\n    return {}\ndef run(value):\n    return {}\n", "exactly one top-level"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_python_source(source)

    def test_requires_pinned_image_and_bounded_finite_test_cases(self):
        with self.assertRaisesRegex(ValueError, "pinned by immutable sha256"):
            DockerPythonSandbox("python:3.12-slim")
        runner = DockerPythonSandbox("python:3.12-slim@sha256:" + "a" * 64)
        with self.assertRaisesRegex(ValueError, "1..100"):
            runner.execute("def run(input): return {}", {}, [])
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            runner.execute("def run(input): return {}", {"value": float("nan")}, [{"input": {}, "expected": {}}])

    def test_execution_only_request_is_explicit_and_still_checks_the_contract(self):
        contract = {
            "input": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
            "output": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
        }
        with self.assertRaisesRegex(ValueError, "scientific validation cases"):
            validate_capability_run_request({"value": 2}, [], contract)
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            validate_capability_run_request({"value": 2}, [], contract, require_test_cases=0)
        request = validate_capability_run_request({"value": 2}, [], contract, require_test_cases=False)
        self.assertEqual(request["input"], {"value": 2})
        self.assertEqual(request["test_cases"], [])
        with self.assertRaisesRegex(ValueError, "omits required contract fields"):
            validate_capability_run_request({}, [], contract, require_test_cases=False)

    def test_validation_request_digest_binds_inputs_and_expected_outputs_without_retaining_them(self):
        request = validate_capability_run_request(
            {"value": 2}, [{"input": {"value": 4}, "expected": {"value": 5}}]
        )
        digest = capability_validation_request_hash(request)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, capability_validation_request_hash(request))
        changed = validate_capability_run_request(
            {"value": 2}, [{"input": {"value": 4}, "expected": {"value": 6}}]
        )
        self.assertNotEqual(digest, capability_validation_request_hash(changed))
        with self.assertRaisesRegex(ValueError, "validation cases"):
            capability_validation_request_hash({"input": {}, "test_cases": []})

    def test_invokes_docker_with_network_and_filesystem_restrictions(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n"
                "    print('sha256:local')\n"
                "else:\n"
                "    required=['--pull=never','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--memory=512m','--cpus=1.0','--entrypoint=python','--ulimit=nofile=64:64']\n"
                "    assert all(x in args for x in required), args\n"
                "    volume=args[args.index('--volume')+1]\n"
                "    assert volume.endswith(':/capability:ro,noexec,nosuid,nodev'), volume\n"
                "    host=volume.split(':',1)[0]\n"
                "    assert os.path.isdir(host) and os.access(host, os.R_OK|os.X_OK)\n"
                "    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+json.dumps({'ok':True,'result':{'value':3},'cpu_time_microseconds':1234,'validation':{'passed':1,'failed':0,'cases':[{'index':0,'passed':True}]}}))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            image = "python:3.12-slim@sha256:" + "b" * 64
            result = DockerPythonSandbox(image, docker_binary=str(docker)).execute(
                "def run(input):\n    return {'value': input['value'] + 1}\n",
                {"value": 2},
                [{"input": {"value": 2}, "expected": {"value": 3}}],
            )
        self.assertTrue(result["case_tests_passed"])
        self.assertTrue(result["validation_performed"])
        self.assertEqual(result["lifecycle_status"], "experimental; no automatic registration or promotion")
        self.assertEqual(result["result"], {"value": 3})
        self.assertEqual(result["cpu_time_microseconds"], 1234)

    def test_execution_only_sandbox_receipt_does_not_claim_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n"
                "    print('sha256:local')\n"
                "else:\n"
                "    print('MEDAI_SANDBOX_RESULT:'+args[-1]+':'+json.dumps({'ok':True,'result':{'value':3},'cpu_time_microseconds':1234,'validation':{'performed':False,'passed':0,'failed':0,'cases':[]}}))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            image = "python:3.12-slim@sha256:" + "c" * 64
            result = DockerPythonSandbox(image, docker_binary=str(docker)).execute(
                "def run(input):\n    return {'value': input['value'] + 1}\n",
                {"value": 2},
                [],
                require_test_cases=False,
                contract={
                    "input": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
                    "output": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
                },
            )
        self.assertEqual(result["result"], {"value": 3})
        self.assertFalse(result["validation_performed"])
        self.assertIsNone(result["case_tests_passed"])

    def test_output_flood_is_terminated_with_a_bounded_capture(self):
        with self.assertRaisesRegex(SandboxError, "exceeded 1024 bytes"):
            _run_limited([sys.executable, "-c", "import sys; sys.stdout.write('x' * 1000000)"], timeout=5, output_limit=1024)

    def test_cancellation_signal_terminates_the_runner_process_group(self):
        cancel_event = threading.Event()
        outcome = []

        def execute():
            try:
                _run_limited([sys.executable, "-c", "import time; time.sleep(30)"], timeout=40, cancel_event=cancel_event)
            except Exception as error:
                outcome.append(error)

        worker = threading.Thread(target=execute)
        worker.start()
        time.sleep(0.1)
        cancel_event.set()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive(), "cancelled sandbox process should be reaped promptly")
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], SandboxCancelledError)

    def test_trusted_runner_fixture_reports_exact_case_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = root / "runner.py"
            capability = root / "capability.py"
            request = root / "request.json"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            capability.write_text("def run(input):\n    return {'value': input['value'] + 1}\n", encoding="utf-8")
            request.write_text(json.dumps({"input": {"value": 2}, "test_cases": [{"input": {"value": 4}, "expected": {"value": 5}}]}), encoding="utf-8")
            nonce = "testnonce"
            result = subprocess.run([sys.executable, str(runner), str(capability), str(request), nonce], capture_output=True, check=False)
            request.write_text(json.dumps({"input": {"value": 2}, "test_cases": [{"input": {"value": 4}, "expected": {"value": 99}}]}), encoding="utf-8")
            failed = subprocess.run([sys.executable, str(runner), str(capability), str(request), nonce], capture_output=True, check=False)
        self.assertEqual(result.returncode, 0)
        prefix = _RESULT_PREFIX + nonce.encode() + b":"
        line = next(line for line in result.stdout.splitlines() if line.startswith(prefix))
        receipt = json.loads(line[len(prefix):])
        self.assertEqual(receipt["result"], {"value": 3})
        self.assertIsInstance(receipt["cpu_time_microseconds"], int)
        self.assertGreaterEqual(receipt["cpu_time_microseconds"], 0)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["validation"]["passed"], 1)
        self.assertIsInstance(receipt["validation"]["cases"][0]["wall_time_microseconds"], int)
        self.assertGreaterEqual(receipt["validation"]["cases"][0]["wall_time_microseconds"], 0)
        self.assertEqual(failed.returncode, 3)
        failed_receipt = json.loads(next(line for line in failed.stdout.splitlines() if line.startswith(prefix))[len(prefix):])
        self.assertFalse(failed_receipt["ok"])
        self.assertIsInstance(failed_receipt["cpu_time_microseconds"], int)
        self.assertEqual(failed_receipt["validation"]["failed"], 1)

    def test_trusted_runner_executes_one_request_without_claiming_case_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = root / "runner.py"
            capability = root / "capability.py"
            request = root / "request.json"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            capability.write_text("def run(input):\n    return {'value': input['value'] + 1}\n", encoding="utf-8")
            request.write_text(json.dumps({
                "input": {"value": 2},
                "test_cases": [],
                "input_schema": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
                "output_schema": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
            }), encoding="utf-8")
            result = subprocess.run([sys.executable, str(runner), str(capability), str(request), "invokeonce"], capture_output=True, check=False)
        prefix = _RESULT_PREFIX + b"invokeonce:"
        receipt = json.loads(next(line for line in result.stdout.splitlines() if line.startswith(prefix))[len(prefix):])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(receipt["result"], {"value": 3})
        self.assertTrue(receipt["ok"])
        self.assertFalse(receipt["validation"]["performed"])
        self.assertEqual(receipt["validation"]["passed"], 0)

    def test_trusted_runner_reports_cpu_even_when_capability_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = root / "runner.py"
            capability = root / "capability.py"
            request = root / "request.json"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            capability.write_text("def run(input):\n    raise ValueError('private detail')\n", encoding="utf-8")
            request.write_text(json.dumps({"input": {}, "test_cases": [{"input": {}, "expected": {}}]}), encoding="utf-8")
            nonce = "errornonce"
            result = subprocess.run([sys.executable, str(runner), str(capability), str(request), nonce], capture_output=True, check=False)
        prefix = _RESULT_PREFIX + nonce.encode() + b":"
        line = next(line for line in result.stdout.splitlines() if line.startswith(prefix))
        receipt = json.loads(line[len(prefix):])
        self.assertEqual(result.returncode, 4)
        self.assertEqual(receipt["execution_error_type"], "ValueError")
        self.assertNotIn("private detail", line.decode("utf-8"))
        self.assertIsInstance(receipt["cpu_time_microseconds"], int)

    def test_trusted_runner_rejects_output_that_violates_capability_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runner = root / "runner.py"
            capability = root / "capability.py"
            request = root / "request.json"
            runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
            capability.write_text("def run(input): return {'count': 'many'}\n", encoding="utf-8")
            schema = {"type": "object", "required": ["count"], "properties": {"count": {"type": "integer", "minimum": 0}}}
            request.write_text(json.dumps({"input": {}, "test_cases": [{"input": {}, "expected": {"count": 1}}], "input_schema": {"type": "object"}, "output_schema": schema}), encoding="utf-8")
            result = subprocess.run([sys.executable, str(runner), str(capability), str(request), "contractnonce"], capture_output=True, check=False)
        self.assertEqual(result.returncode, 4)
        self.assertIn(b"TypeError", result.stdout)


if __name__ == "__main__":
    unittest.main()
