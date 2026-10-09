import * as axe from "axe-core";
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
  await page.addScriptTag({ content: axe.source });
  for (const [tab, heading] of views) {
    if (tab !== "Overview") await page.getByRole("button", { name: tab, exact: true }).click();
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    const violations = await page.evaluate(async () => {
      const engine = (window as Window & {
        axe?: {
          run: (
            context?: Document,
            options?: { runOnly: { type: "tag"; values: string[] } },
          ) => Promise<{
            violations: Array<{ id: string; impact: string | null; help: string; nodes: unknown[] }>;
          }>;
        };
      }).axe;
      if (!engine) throw new Error("axe-core was not installed in the audit page");
      return (await engine.run(document, {
        runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
      })).violations;
    });
    expect(violations, `${tab}: ${JSON.stringify(violations, null, 2)}`).toEqual([]);
  }
});

test("literature indexing metadata and PMID relationships remain accessible when populated", async ({ page }) => {
  const session = {
    id: "RES-A11Y-1",
    objective: "Audit PubMed indexing metadata",
    status: "completed",
    created_at: 1_800_000_000,
    depth: "standard",
    budget: 20,
    credits_used: 1,
  };
  const paper = {
    provenance: {
      title: "A representative indexed research article",
      source_id: "12345678",
      url: "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    },
    publication_types: ["Journal Article", "Comparative Study"],
    mesh_headings: [{ descriptor: "Alzheimer Disease", descriptor_ui: "D000544" }],
    related_publications: [{ ref_type: "RetractionIn", pmid: "87654321" }],
  };
  await page.route("**/api/auth/session", async route => route.fulfill({ json: { subject: "a11y-reviewer", tenant_id: "lab-a", role: "researcher", expires_at: 1_900_000_000 } }));
  await page.route("**/api/medai/**", async route => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.endsWith("/v1/research") && route.request().method() === "GET") return route.fulfill({ json: { items: [session] } });
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 1, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 1 } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/" + session.id + "/plan")) return route.fulfill({ json: { session_id: session.id, tasks: [] } });
    if (path.endsWith("/" + session.id + "/graph")) return route.fulfill({ json: { session_id: session.id, nodes: [], edges: [], next_node_cursor: null, next_edge_cursor: null } });
    if (path.endsWith("/" + session.id + "/budget")) return route.fulfill({ json: { limit: 20, used: 1, remaining: 19, storage: { artifact_payload_bytes: 0, credits_charged: 0, bytes_per_credit: 1 }, events: [], model_reservations: [] } });
    if (path.endsWith("/" + session.id + "/citation-links")) return route.fulfill({ json: { items: [], metrics: [] } });
    if (path.endsWith("/" + session.id + "/literature-watch-papers")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/" + session.id + "/artifacts")) {
      const kind = url.searchParams.get("kind");
      const items = kind === "paper" ? [{
        id: "ART-PMID-12345678",
        session_id: session.id,
        task_id: "literature.search",
        kind: "paper",
        source: "PubMed",
        source_id: "12345678",
        payload: JSON.stringify(paper),
        sha256: "a".repeat(64),
        recorded_at: 1_800_000_000,
      }] : [];
      return route.fulfill({ json: { items, next_cursor: null } });
    }
    return route.fulfill({ status: 404, json: { error: "not found" } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: session.objective }).click();
  await page.getByRole("button", { name: "Literature", exact: true }).click();
  await expect(page.getByRole("heading", { name: paper.provenance.title })).toBeVisible();
  await page.getByText("Publication and MeSH indexing metadata").click();
  await expect(page.getByText("Journal Article · Comparative Study")).toBeVisible();
  await expect(page.getByText("Alzheimer Disease [D000544]")).toBeVisible();
  await expect(page.getByText(/do not establish study quality/)).toBeVisible();

  await page.getByText("PubMed related-publication links (1)").click();
  const relatedPmid = page.getByRole("link", { name: "PMID 87654321" });
  await expect(relatedPmid).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/87654321/");
  await expect(page.getByText(/links alone do not grade evidence quality/)).toBeVisible();

  await page.addScriptTag({ content: axe.source });
  const violations = await page.evaluate(async () => {
    const engine = (window as Window & {
      axe?: {
        run: (
          context?: Document,
          options?: { runOnly: { type: "tag"; values: string[] } },
        ) => Promise<{
          violations: Array<{ id: string; impact: string | null; help: string; nodes: unknown[] }>;
        }>;
      };
    }).axe;
    if (!engine) throw new Error("axe-core was not installed in the audit page");
    return (await engine.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"] },
    })).violations;
  });
  expect(violations, JSON.stringify(violations, null, 2)).toEqual([]);
});
