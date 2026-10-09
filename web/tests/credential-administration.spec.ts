import { expect, test } from "@playwright/test";

test("tenant administrators create and revoke credentials without exposing tokens in the listing", async ({ page }) => {
  let credential: Record<string, unknown> | null = null;
  let createdBody: Record<string, string> | null = null;
  await page.route("**/api/auth/session", async route => {
    if (route.request().method() === "GET") return route.fulfill({ status: 401, json: { error: "login required" } });
    if (route.request().method() === "POST") return route.fulfill({ json: { subject: "tenant-owner", tenant_id: "lab-a", role: "tenant_admin", expires_at: 1_900_000_000 } });
    return route.fulfill({ json: { logged_out: true } });
  });
  await page.route("**/api/medai/**", async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path.endsWith("/v1/research")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 0, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 0 } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/v1/admin/credentials") && request.method() === "GET") return route.fulfill({ json: { items: credential ? [{ ...credential, token: undefined }] : [] } });
    if (path.endsWith("/v1/admin/credentials") && request.method() === "POST") {
      createdBody = request.postDataJSON() as Record<string, string>;
      credential = { id: "credential-1", subject: createdBody.subject, tenant_id: "lab-a", role: createdBody.role, created_at: 1_700_000_000, revoked_at: null, active: true };
      return route.fulfill({ status: 201, json: { ...credential, token: "medai_one_time_secret", token_disclosure: "shown once; store securely" } });
    }
    if (path.endsWith("/v1/admin/credentials/credential-1") && request.method() === "DELETE") {
      credential = { ...credential, active: false, revoked_at: 1_700_000_100 };
      return route.fulfill({ status: 204 });
    }
    return route.fulfill({ status: 404, json: { error: `Unhandled test API request: ${request.method()} ${path}` } });
  });

  await page.goto("/");
  await page.getByLabel("Personal API credential").fill("tenant-admin-personal-bearer");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("button", { name: "Credentials" })).toBeVisible();
  await page.getByRole("button", { name: "Credentials" }).click();
  await expect(page.getByRole("heading", { name: "Credential administration" })).toBeVisible();
  await expect(page.getByLabel("Tenant ID")).toHaveCount(0);
  const roleOptions = await page.getByLabel("Role").locator("option").allTextContents();
  expect(roleOptions).toEqual(["reader", "researcher"]);
  await page.getByLabel("Subject").fill("new-researcher");
  await page.getByRole("button", { name: "Create credential" }).click();
  await expect(page.getByText("medai_one_time_secret")).toBeVisible();
  expect(createdBody).toEqual({ subject: "new-researcher", role: "researcher" });
  await expect(page.getByText("medai_one_time_secret")).toHaveCount(1);

  page.once("dialog", dialog => dialog.accept());
  await page.getByRole("button", { name: "Revoke" }).click();
  await expect(page.getByText("revoked", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Revoke" })).toHaveCount(0);
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page.getByLabel("Personal API credential")).toBeVisible();
  await page.getByLabel("Personal API credential").fill("another-tenant-admin-bearer");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("button", { name: "Credentials" }).click();
  await expect(page.getByText("medai_one_time_secret")).toHaveCount(0);
});
