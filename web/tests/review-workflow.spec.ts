import { expect, test } from "@playwright/test";

test("a researcher logs in once and submits a reasoned task decision without exposing the bearer", async ({ page }) => {
  let reviewed = false;
  let reviewRequest: { authorization?: string; body?: Record<string, string> } = {};
  let loginBody: { token?: string } = {};
  await page.route("**/api/auth/session", async route => {
    if (route.request().method() === "GET") return route.fulfill({ status: 401, json: { error: "login required" } });
    if (route.request().method() === "POST") {
      loginBody = route.request().postDataJSON() as { token?: string };
      return route.fulfill({ json: { subject: "reviewer-7", tenant_id: "lab-a", role: "researcher", expires_at: 1_900_000_000 } });
    }
    return route.fulfill({ json: { logged_out: true } });
  });
  await page.route("**/api/medai/**", async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (path.endsWith("/v1/research") && request.method() === "GET") {
      return route.fulfill({ json: { items: [{ id: "RES-review", objective: "Review test", status: "active", created_at: 1, depth: "standard", budget: 20, credits_used: 0 }] } });
    }
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 1, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 0 } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/plan")) return route.fulfill({ json: { session_id: "RES-review", tasks: [{ id: "paper.extract", domain: "literature", description: "Review extracted paper evidence", depends_on: [], status: reviewed ? "succeeded" : "needs_review", attempts: 1, output: "Source-linked extraction receipt", error: null }], events: [] } });
    if (path.endsWith("/graph")) return route.fulfill({ json: { nodes: [], edges: [] } });
    if (path.endsWith("/budget")) return route.fulfill({ json: { limit: 20, used: 0, remaining: 20, storage: { artifact_payload_bytes: 0, credits_charged: 0, bytes_per_credit: 1048576 }, events: [] } });
    if (path.endsWith("/citation-links") || path.endsWith("/literature-watch-papers")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/artifacts")) return route.fulfill({ json: { items: [], next_cursor: null } });
    if (path.endsWith("/tasks/paper.extract/review") && request.method() === "POST") {
      reviewRequest = { authorization: request.headers().authorization, body: request.postDataJSON() as Record<string, string> };
      reviewed = true;
      return route.fulfill({ status: 200, json: { session_id: "RES-review", task_id: "paper.extract", status: "succeeded" } });
    }
    return route.fulfill({ status: 404, json: { error: `Unhandled test API request: ${request.method()} ${path}` } });
  });

  await page.goto("/");
  await page.getByLabel("Personal API credential").fill("researcher-personal-bearer");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Research workspace" })).toBeVisible();
  await expect(page.getByLabel("Personal API credential")).toHaveCount(0);
  await page.getByRole("button", { name: "RES-review" }).click();
  await page.getByRole("button", { name: "Review", exact: true }).click();
  await page.getByLabel("Review rationale for paper.extract").fill("Checked the retained source identifiers and extraction receipt.");
  await page.getByRole("button", { name: "Approve task" }).click();

  await expect(page.getByText("No research task is waiting for human review in this session.")).toBeVisible();
  expect(loginBody).toEqual({ token: "researcher-personal-bearer" });
  expect(reviewRequest.authorization).toBeUndefined();
  expect(reviewRequest.body).toEqual({ decision: "approve", rationale: "Checked the retained source identifiers and extraction receipt." });
});
