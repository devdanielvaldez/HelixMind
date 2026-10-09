from __future__ import annotations

import ast
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).parents[2]
CAPABILITY_DIR = ROOT / "integrations/agentmesh/capabilities"
SERVER = ROOT / "science/src/medai_science/mcp_server.py"


def _registered_tool_names() -> set[str]:
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "mcp"
                and decorator.func.attr == "tool"
            ):
                continue
            name = next((
                keyword.value.value
                for keyword in decorator.keywords
                if keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ), None)
            if name is not None:
                if name in names:
                    raise AssertionError(f"duplicate MCP tool registration: {name}")
                names.add(name)
    return names


class AMCPServerReferenceTests(unittest.TestCase):
    def test_checked_in_agentmesh_mcp_references_resolve_to_unique_server_tools(self):
        registered = _registered_tool_names()
        packages = sorted(CAPABILITY_DIR.glob("*.yaml"))
        self.assertTrue(packages)
        package_ids: set[str] = set()
        missing: list[str] = []
        for path in packages:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            capability = document.get("capability", {}) if isinstance(document, dict) else {}
            package_id = capability.get("id") if isinstance(capability, dict) else None
            self.assertIsInstance(package_id, str, str(path))
            self.assertNotIn(package_id, package_ids, f"duplicate AMCP id in {path}")
            package_ids.add(package_id)
            implementations = capability.get("implementations", [])
            self.assertIsInstance(implementations, list, str(path))
            for implementation in implementations:
                if not isinstance(implementation, dict):
                    continue
                reference = implementation.get("reference")
                if isinstance(reference, str) and reference.startswith("mcp://medai/"):
                    tool_name = reference.removeprefix("mcp://medai/")
                    if tool_name not in registered:
                        missing.append(f"{path.name}: {reference}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
