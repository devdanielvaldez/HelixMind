import ast
import unittest
from pathlib import Path


class InternalApiAuthenticationTests(unittest.TestCase):
    def test_every_mcp_client_targeting_the_api_uses_request_or_service_authentication(self):
        source = Path(__file__).parents[1] / "src" / "medai_science" / "mcp_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        api_clients = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "AsyncClient":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            if isinstance(keywords.get("base_url"), ast.Name) and keywords["base_url"].id == "API":
                api_clients.append(keywords)

        self.assertGreaterEqual(len(api_clients), 50, "the MCP server should retain its API-backed operations")
        def authenticated_headers(client):
            header = client.get("headers")
            return (
                isinstance(header, ast.Call)
                and isinstance(header.func, ast.Name)
                and header.func.id == "_api_headers"
            ) or (isinstance(header, ast.Name) and header.id == "headers")

        self.assertTrue(all(authenticated_headers(client) for client in api_clients))
        auth_source = source.with_name("mcp_auth.py").read_text(encoding="utf-8")
        self.assertIn('"io.medai/specialist-api-bearer"', auth_source)
        self.assertIn('"/v1/auth/whoami"', auth_source)


if __name__ == "__main__":
    unittest.main()
