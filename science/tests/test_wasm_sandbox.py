import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from medai_science.sandbox import SandboxError, _run_limited
from medai_science.wasm_sandbox import DockerWasmSandbox


WASM_MODULE = b"\x00asm\x01\x00\x00\x00\x00"
CONTRACT = {
    "input": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
    "output": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
}


class WasmSandboxTests(unittest.TestCase):
    def test_validates_pinned_runtime_module_header_and_limits(self):
        with self.assertRaisesRegex(ValueError, "immutable sha256"):
            DockerWasmSandbox("wasmtime:latest")
        runner = DockerWasmSandbox("wasmtime@sha256:" + "a" * 64)
        self.assertEqual(DockerWasmSandbox("sha256:" + "b" * 64).image, "sha256:" + "b" * 64)
        with self.assertRaisesRegex(ValueError, "version 1 binary"):
            runner.execute(b"not wasm", {}, [{"input": {}, "expected": {}}])
        with self.assertRaisesRegex(ValueError, "fuel"):
            DockerWasmSandbox("wasmtime@sha256:" + "a" * 64, fuel=100_000_001)

    def test_rejects_out_of_contract_input_before_starting_container(self):
        runner = DockerWasmSandbox("wasmtime@sha256:" + "b" * 64)
        with self.assertRaisesRegex(ValueError, "does not match contract type integer"):
            runner.execute(WASM_MODULE, {"value": "bad"}, [{"input": {"value": 1}, "expected": {"value": 2}}], contract=CONTRACT)

    def test_mock_docker_runs_wasi_json_protocol_under_isolation_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n    print('sha256:local')\n"
                "else:\n"
                "    cidfile=next(x.split('=',1)[1] for x in args if x.startswith('--cidfile='))\n"
                "    assert not __import__('os').path.exists(cidfile), 'Docker cidfile must be fresh'\n"
                "    open(cidfile, 'w').write('a'*64)\n"
                "    required=['--pull=never','-i','--network=none','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--memory=384m','--cpus=1.0','--entrypoint=wasmtime','-W','fuel=100000000','-W','max-memory-size=268435456']\n"
                "    assert all(x in args for x in required), args\n"
                "    volume=args[args.index('--volume')+1]\n"
                "    assert volume.endswith(':/capability:ro,noexec,nosuid,nodev'), volume\n"
                "    mounted=volume.split(':/capability:',1)[0]\n"
                "    assert __import__('os').listdir(mounted) == ['capability.wasm']\n"
                "    assert __import__('os').path.exists(args[next(i for i,x in enumerate(args) if x.startswith('--cidfile='))].split('=',1)[1])\n"
                "    request=json.loads(sys.stdin.buffer.read())\n"
                "    print(os.environ.get('MOCK_WASM_OUTPUT', json.dumps({'value':request['value']+1}, separators=(',',':'))))\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            runner = DockerWasmSandbox("wasmtime@sha256:" + "c" * 64, docker_binary=str(docker))
            result = runner.execute(
                WASM_MODULE,
                {"value": 1},
                [{"input": {"value": 4}, "expected": {"value": 5}}],
                contract=CONTRACT,
            )
        self.assertEqual(result["result"], {"value": 2})
        self.assertTrue(result["case_tests_passed"])
        self.assertEqual(result["runner"], "docker-no-network-wasmtime-wasi-preview1")
        self.assertEqual(result["validation"]["cases"][0]["wall_time_microseconds"] >= 0, True)
        self.assertIn("not reported as CPU time", result["fuel_metering"])

    def test_rejects_ambiguous_or_nonstandard_guest_json(self):
        with tempfile.TemporaryDirectory() as folder:
            docker = Path(folder) / "docker"
            docker.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "args=sys.argv[1:]\n"
                "if args[:2] == ['image','inspect']:\n    print('sha256:local')\n"
                "else:\n    print(os.environ['MOCK_WASM_OUTPUT'])\n",
                encoding="utf-8",
            )
            docker.chmod(0o755)
            runner = DockerWasmSandbox("wasmtime@sha256:" + "d" * 64, docker_binary=str(docker))
            for output in ('{"value":1,"value":2}', '{"value":NaN}', '{"value":Infinity}'):
                with self.subTest(output=output), patch.dict(os.environ, {"MOCK_WASM_OUTPUT": output}):
                    with self.assertRaisesRegex(SandboxError, "exactly one JSON result object"):
                        runner.execute(WASM_MODULE, {"value": 1}, [], contract=CONTRACT, require_test_cases=False)

    def test_limited_runner_writes_bounded_stdin_and_closes_it(self):
        code = "import sys; value=sys.stdin.buffer.read(); sys.stdout.buffer.write(value)"
        status, stdout, _ = _run_limited([sys.executable, "-c", code], timeout=5, input_bytes=b'{"ok":true}')
        self.assertEqual(status, 0)
        self.assertEqual(stdout, b'{"ok":true}')


if __name__ == "__main__":
    unittest.main()
