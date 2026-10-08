import { expect, test } from "@playwright/test";

test("researchers can adjudicate an exact critic paper lead and see the saved decision", async ({ page }) => {
  const savedReviews: Record<string, unknown>[] = [];
  const criticismId = "ART-CRITIC-1";
  const candidateId = "ART-PMID-1";
  const baseArtifact = (id: string, kind: string, payload: unknown, sourceId: string) => ({
    id, session_id: "RES-critic", task_id: "hypotheses.criticize", kind,
    source: "MedAI", source_id: sourceId, payload: JSON.stringify(payload),
    sha256: "a".repeat(64), recorded_at: 1,
  });
  const artifacts = [
    baseArtifact("ART-HYP-1", "hypothesis_candidates", { candidates: [{ id: "HYP-1", statement: "TREM2 and APOE may converge on lipid processing", path: [], dimensions: {} }] }, "objective-1"),
    baseArtifact(criticismId, "hypothesis_criticism", {
      proposal_artifact_id: "ART-HYP-1",
      reviews: [{
        hypothesis_id: "HYP-1", review_status: "requires_human_scientific_review",
        linked_contradictory_evidence: [], lexical_refutation_leads: [], alternative_graph_routes: [],
        retrieved_contradiction_candidates: [{
          candidate_artifact_id: candidateId, candidate_artifact_sha256: "b".repeat(64),
          source: "PubMed", source_id: "12345", title: "A context-dependent cohort result",
          target_evidence_id: "EV-1", review_status: "requires_human_scientific_review",
        }],
      }],
    }, "ART-HYP-1"),
  ];

  await page.route("**/api/auth/session", async route => {
    if (route.request().method() === "POST") return route.fulfill({ json: { subject: "researcher-7", tenant_id: "lab-a", role: "researcher", expires_at: 1_900_000_000 } });
    return route.fulfill({ status: 401, json: { error: "login required" } });
  });
  await page.route("**/api/medai/**", async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (path.endsWith("/v1/research") && request.method() === "GET") return route.fulfill({ json: { items: [{ id: "RES-critic", objective: "Review critic leads", status: "active", created_at: 1, depth: "standard", budget: 20, credits_used: 0 }] } });
    if (path.endsWith("/v1/summary")) return route.fulfill({ json: { sessions: 1, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: artifacts.length + savedReviews.length } });
    if (path.endsWith("/v1/capabilities")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/plan")) return route.fulfill({ json: { session_id: "RES-critic", tasks: [], events: [] } });
    if (path.endsWith("/graph")) return route.fulfill({ json: { nodes: [], edges: [] } });
    if (path.endsWith("/budget")) return route.fulfill({ json: { limit: 20, used: 0, remaining: 20, storage: { artifact_payload_bytes: 0, credits_charged: 0, bytes_per_credit: 1048576 }, events: [], model_reservations: [] } });
    if (path.endsWith("/citation-links") || path.endsWith("/literature-watch-papers")) return route.fulfill({ json: { items: [] } });
    if (path.endsWith("/artifacts")) {
      const kind = url.searchParams.get("kind");
      const items = [...artifacts, ...savedReviews.map((payload, index) => baseArtifact(`ART-REVIEW-${index + 1}`, "hypothesis_critic_adjudication", payload, criticismId))].filter(item => item.kind === kind);
      return route.fulfill({ json: { items, next_cursor: null } });
    }
    if (path.endsWith("/hypothesis-critic-reviews") && request.method() === "POST") {
      savedReviews.push({ ...(request.postDataJSON() as Record<string, unknown>), criticism_artifact_id: criticismId, reviewer: { subject: "researcher-7" } });
      return route.fulfill({ status: 201, json: { id: "ART-REVIEW-1", payload: JSON.stringify(savedReviews[0]) } });
    }
    return route.fulfill({ status: 404, json: { error: `Unhandled request ${request.method()} ${path}` } });
  });

  await page.goto("/");
  await page.getByLabel("Personal API credential").fill("researcher-personal-bearer");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("button", { name: "RES-critic" }).click();
  await page.getByRole("button", { name: "Hypotheses", exact: true }).click();
  await page.getByText("Human review of scientific criticism").waitFor();
  await page.getByText("Review: PMID 12345").click();
  await page.getByLabel("Assessment for PMID 12345").selectOption("context_only");
  await page.getByLabel("Rationale").first().fill("The result depends on a different cohort context.");
  await page.getByLabel("Study context, methods, population, and limitations").first().fill("Retrospective cohort; the population differs from the candidate mechanism context.");
  await page.getByRole("button", { name: "Save scientific review" }).first().click();

  await expect.poll(() => savedReviews.length).toBe(1);
  expect(savedReviews[0]).toMatchObject({
    criticism_artifact_id: criticismId, hypothesis_id: "HYP-1", item_type: "retrieved_candidate",
    item_id: candidateId, assessment: "context_only",
  });
  await expect(page.getByText(/researcher-7 · context_only/)).toBeVisible();
  await expect(page.getByText(/confidence unchanged/).first()).toBeVisible();
});
