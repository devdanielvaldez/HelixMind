import { expect, test } from "@playwright/test";

test("Spanish language choice localizes and persists the research workspace", async ({ page }) => {
  await page.route("**/api/auth/session", async route => route.fulfill({
    json: { subject: "locale-reviewer", tenant_id: "lab-a", role: "researcher", expires_at: 1_900_000_000 },
  }));
  await page.route("**/api/medai/**", async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith("/v1/research")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 0, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 0 } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    return route.fulfill({ status: 404, json: { error: "not found" } });
  });

  await page.goto("/");
  const language = page.locator("select[aria-label='Language']");
  await expect(language).toBeVisible();
  await language.selectOption("es");
  await expect(page.locator("html")).toHaveAttribute("lang", "es");
  await expect(page.getByRole("heading", { level: 1, name: "Espacio de investigación" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Resumen" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Objetivo de investigación" })).toHaveAttribute("placeholder", "¿Qué te gustaría investigar?");
  await expect(page.getByRole("heading", { level: 3, name: "Sesiones de investigación" })).toBeVisible();

  await page.reload();
  await expect(page.locator("select[aria-label='Idioma']")).toHaveValue("es");
  await expect(page.getByRole("heading", { level: 1, name: "Espacio de investigación" })).toBeVisible();
});
