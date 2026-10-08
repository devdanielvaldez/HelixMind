import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("research workspace and tenant administration views have no automated accessibility violations", async ({ page }) => {
  await page.route("**/api/auth/session", async route => route.fulfill({ json: { subject: "a11y-reviewer", tenant_id: "lab-a", role: "tenant_admin", expires_at: 1_900_000_000 } }));
  await page.route("**/api/medai/**", async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/v1/admin/credentials")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/v1/research")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 0, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 0 } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    return route.fulfill({ status: 404, json: { error: "not found" } });
  });
  await page.goto("/");
  await expect(page.getByRole("main")).toBeVisible();
  const views = [
    ["Overview", "Research workspace"],
    ["Literature", "Literature explorer"],
    ["Biology", "Biomedical entities"],
    ["Knowledge graph", "Knowledge graph"],
    ["Hypotheses", "Hypothesis review"],
    ["Reproducibility", "Reproducibility record"],
    ["Capabilities", "Capability registry"],
    ["Review", "Human review"],
    ["Credentials", "Credential administration"],
  ] as const;
  for (const [tab, heading] of views) {
    if (tab !== "Overview") await page.getByRole("button", { name: tab, exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    const audit = await new AxeBuilder({ page }).analyze();
    expect(audit.violations, `${tab}: ${JSON.stringify(audit.violations, null, 2)}`).toEqual([]);
  }
});
