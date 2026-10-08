"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  Activity, BookOpenText, ClipboardCheck, Dna, FlaskConical, GitBranch, KeyRound, Microscope,
  Network, Plus, Search, ShieldCheck,
} from "lucide-react";
import KnowledgeGraph from "./KnowledgeGraph";
import { officialTrialHistoryUrl } from "@/lib/trial-history.mjs";

type Session = { id: string; objective: string; status: string; created_at: number; depth: string; budget: number; credits_used: number };
type Graph = { nodes: { id: string; kind: string; label: string }[]; edges: { source: string; relation: string; target: string; confidence: number; evidence_ids: string; artifact_ids: string }[] };
type Summary = { sessions: number; evidence: number; graph_nodes: number; capabilities: number; artifacts: number };
type BrowserIdentity = { subject: string; tenant_id: string; role: string; expires_at: number };
type BudgetSummary = { limit: number; used: number; remaining: number; storage: { artifact_payload_bytes: number; credits_charged: number; bytes_per_credit: number }; events: { task_id: string; credits: number; reason: string; occurred_at: number }[]; model_reservations: { id: string; task_id: string; actor: string; reserved_credits: number; charged_credits: number | null; state: "reserved" | "settled" | "uncertain"; reason: string; created_at: number; settled_at: number | null }[] };
type Task = { id: string; domain: string; description: string; depends_on: string[]; status: string; attempts: number; output: string | null; error: string | null; lease_owner?: string | null; lease_until?: number | null; available_at?: number };
type Plan = { session_id: string; tasks: Task[]; events: { task_id: string; from_status: string | null; to_status: string; occurred_at: number; details: string | null }[] };
type CitationLink = { session_id: string; citing_doi: string; cited_doi: string; source_artifact_id: string; reference_index: number | null; indexed_at: number };
type CitationMetric = { doi: string; citations_received: number; references_tracked: number; pagerank: number };
type ResearchArtifact = { id: string; session_id: string; task_id: string; kind: string; source: string; source_id: string; payload: string; sha256: string; recorded_at: number };
type ArtifactCursor = { recorded_at: number; id: string } | null;
type LiteratureRecord = { key: string; title: string; abstract: string; authors: string[]; sourceId: string; url: string; journal: string; date: string; publicationTypes: string[]; meshHeadings: { descriptor: string; descriptorId: string }[]; relatedPublications: { refType: string; pmid: string }[] };
type WatchedPaper = { session_id: string; watch_id: string; event_id: string; pmid: string; title: string; publication_date: string | null; journal: string | null; doi: string | null; pmcid: string | null; captured_at: number; metadata_sha256: string };
type Capability = { id: string; version: string; kind: string; status: string; description: string; manifest: string; manifest_sha256: string };
type CapabilityEvent = { from: string | null; to: string; actor: string; reason: string; validation_sha256: string | null; occurred_at: number };
type ManagedCredential = { id: string; subject: string; tenant_id: string; role: string; created_at: number; revoked_at: number | null; active: boolean };
type WorkspaceView = "overview" | "literature" | "biology" | "graph" | "hypotheses" | "experiments" | "capabilities" | "review" | "credentials";
const api = "/api/medai";

function artifactData(artifact: ResearchArtifact): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(artifact.payload);
    return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
  } catch { return {}; }
}

async function readArtifactPages(
  sessionId: string,
  kind: string,
  budget: { pages: number; items: number; bytes: number },
): Promise<ResearchArtifact[]> {
  const items: ResearchArtifact[] = [];
  let cursor: ArtifactCursor = null;
  do {
    if (budget.pages >= 50) throw new Error("Artifact history exceeded the 50 page review limit.");
    const params = new URLSearchParams({ kind, limit: "200" });
    if (cursor) {
      params.set("after_recorded_at", String(cursor.recorded_at));
      params.set("after_id", cursor.id);
    }
    const response = await fetch(api + "/v1/research/" + sessionId + "/artifacts?" + params.toString());
    if (!response.ok) throw new Error(await response.text());
    const page = await response.json();
    const pageItems = Array.isArray(page.items) ? page.items as ResearchArtifact[] : [];
    budget.pages += 1;
    budget.items += pageItems.length;
    budget.bytes += new TextEncoder().encode(JSON.stringify(pageItems)).byteLength;
    if (budget.items > 10_000 || budget.bytes > 8 * 1024 * 1024) {
      throw new Error("Artifact history exceeded the 10,000 item or 8 MiB review limit.");
    }
    items.push(...pageItems);
    cursor = page.next_cursor ?? null;
  } while (cursor);
  return items;
}

function literatureRecord(artifact: ResearchArtifact): LiteratureRecord | null {
  const paper = artifactData(artifact);
  const provenance = paper.provenance && typeof paper.provenance === "object" ? paper.provenance as Record<string, unknown> : {};
  const title = typeof provenance.title === "string" ? provenance.title : "Untitled PubMed record";
  const authors = Array.isArray(paper.authors) ? paper.authors.filter((value): value is string => typeof value === "string").slice(0, 20) : [];
  const publicationTypes = Array.isArray(paper.publication_types) ? paper.publication_types.filter((value): value is string => typeof value === "string").slice(0, 30) : [];
  const relatedPublications = Array.isArray(paper.related_publications) ? paper.related_publications.flatMap(value => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [];
    const row = value as Record<string, unknown>;
    if (typeof row.ref_type !== "string" || !/^[A-Za-z][A-Za-z0-9]{0,63}$/.test(row.ref_type) || typeof row.pmid !== "string" || !/^\d{1,20}$/.test(row.pmid)) return [];
    return [{ refType: row.ref_type, pmid: row.pmid }];
  }).slice(0, 100) : [];
  const meshHeadings = Array.isArray(paper.mesh_headings) ? paper.mesh_headings.flatMap(value => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return [];
    const row = value as Record<string, unknown>;
    if (typeof row.descriptor !== "string" || !row.descriptor.trim()) return [];
    return [{ descriptor: row.descriptor.slice(0, 256), descriptorId: typeof row.descriptor_ui === "string" ? row.descriptor_ui.slice(0, 32) : "" }];
  }).slice(0, 50) : [];
  return {
    key: artifact.id,
    title,
    abstract: typeof paper.abstract === "string" ? paper.abstract : "",
    authors,
    sourceId: typeof provenance.source_id === "string" ? provenance.source_id : artifact.source_id,
    url: typeof provenance.url === "string" && provenance.url.startsWith("https://") ? provenance.url : "",
    journal: typeof paper.journal === "string" ? paper.journal : "",
    date: typeof paper.publication_date === "string" ? paper.publication_date : "",
    publicationTypes,
    meshHeadings,
    relatedPublications,
  };
}

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [graph, setGraph] = useState<Graph>({ nodes: [], edges: [] });
  const [summary, setSummary] = useState<Summary>({ sessions: 0, evidence: 0, graph_nodes: 0, capabilities: 0, artifacts: 0 });
  const [objective, setObjective] = useState("");
  const [depth, setDepth] = useState("standard");
  const [budget, setBudget] = useState(100);
  const [error, setError] = useState("");
  const [authReady, setAuthReady] = useState(false);
  const [browserIdentity, setBrowserIdentity] = useState<BrowserIdentity | null>(null);
  const [loginCredential, setLoginCredential] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [selected, setSelected] = useState<Session | null>(null);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [budgetSummary, setBudgetSummary] = useState<BudgetSummary | null>(null);
  const [loadingPlan, setLoadingPlan] = useState(false);
  const [citationLinks, setCitationLinks] = useState<CitationLink[]>([]);
  const [citationMetrics, setCitationMetrics] = useState<CitationMetric[]>([]);
  const [artifacts, setArtifacts] = useState<ResearchArtifact[]>([]);
  const [literatureArtifacts, setLiteratureArtifacts] = useState<ResearchArtifact[]>([]);
  const [watchedPapers, setWatchedPapers] = useState<WatchedPaper[]>([]);
  const [literatureCursor, setLiteratureCursor] = useState<ArtifactCursor>(null);
  const [literatureHasMore, setLiteratureHasMore] = useState(false);
  const [loadingLiterature, setLoadingLiterature] = useState(false);
  const [loadingMoreLiterature, setLoadingMoreLiterature] = useState(false);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [reviewReasons, setReviewReasons] = useState<Record<string, string>>({});
  const [validationReports, setValidationReports] = useState<Record<string, string>>({});
  const [rollbackTargets, setRollbackTargets] = useState<Record<string, string>>({});
  const [capabilityEvents, setCapabilityEvents] = useState<Record<string, CapabilityEvent[]>>({});
  const [reviewBusy, setReviewBusy] = useState("");
  const [studyReviews, setStudyReviews] = useState<Record<string, Record<string, string>>>({});
  const [interactionClaimReviewForms, setInteractionClaimReviewForms] = useState<Record<string, Record<string, string>>>({});
  const [hypothesisLiteratureReviewForms, setHypothesisLiteratureReviewForms] = useState<Record<string, Record<string, string>>>({});
  const [hypothesisCriticReviewForms, setHypothesisCriticReviewForms] = useState<Record<string, Record<string, string>>>({});
  const [claimAlignmentReviewForms, setClaimAlignmentReviewForms] = useState<Record<string, Record<string, string>>>({});
  const [ontologyMappingReviewForms, setOntologyMappingReviewForms] = useState<Record<string, Record<string, string>>>({});
  const [citationFilter, setCitationFilter] = useState("");
  const [literatureFilter, setLiteratureFilter] = useState("");
  const [literatureSearch, setLiteratureSearch] = useState("");
  const [citationFocus, setCitationFocus] = useState("");
  const [view, setView] = useState<WorkspaceView>("overview");
  const [managedCredentials, setManagedCredentials] = useState<ManagedCredential[]>([]);
  const [credentialSubject, setCredentialSubject] = useState("");
  const [credentialTenant, setCredentialTenant] = useState("");
  const [credentialRole, setCredentialRole] = useState("researcher");
  const [issuedCredential, setIssuedCredential] = useState<{ token: string; subject: string; tenant_id: string; role: string } | null>(null);
  const [credentialBusy, setCredentialBusy] = useState("");

  const refresh = async () => {
    const [sessionResponse, summaryResponse, capabilityResponse] = await Promise.all([
      fetch(api + "/v1/research"), fetch(api + "/v1/summary"), fetch(api + "/v1/capabilities"),
    ]);
    if (!sessionResponse.ok || !summaryResponse.ok || !capabilityResponse.ok) throw new Error("The MedAI API returned an error.");
    const [sessionData, summaryData, capabilityData] = await Promise.all([
      sessionResponse.json(), summaryResponse.json(), capabilityResponse.json(),
    ]);
    setSessions(sessionData.items ?? []);
    setSummary(summaryData);
    setCapabilities(capabilityData.items ?? []);
  };

  async function loadCredentials() {
    const response = await fetch(api + "/v1/admin/credentials");
    if (!response.ok) throw new Error(await response.text());
    const result = await response.json();
    setManagedCredentials(Array.isArray(result.items) ? result.items : []);
  }

  async function createManagedCredential(event: FormEvent) {
    event.preventDefault();
    setCredentialBusy("create"); setError(""); setIssuedCredential(null);
    try {
      const payload: Record<string, string> = { subject: credentialSubject.trim(), role: credentialRole };
      if (browserIdentity?.role === "platform_admin") payload.tenant_id = credentialTenant.trim() || browserIdentity.tenant_id;
      const response = await fetch(api + "/v1/admin/credentials", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "Credential creation failed.");
      setIssuedCredential({ token: result.token, subject: result.subject, tenant_id: result.tenant_id, role: result.role });
      setCredentialSubject("");
      await loadCredentials();
    } catch (cause) { setError(String(cause)); }
    finally { setCredentialBusy(""); }
  }

  async function revokeManagedCredential(credential: ManagedCredential) {
    if (!window.confirm(`Revoke ${credential.role} credential for ${credential.subject}? Existing sessions using it will no longer authenticate.`)) return;
    setCredentialBusy(credential.id); setError("");
    try {
      const response = await fetch(`${api}/v1/admin/credentials/${encodeURIComponent(credential.id)}`, { method: "DELETE" });
      if (!response.ok) throw new Error(await response.text());
      await loadCredentials();
    } catch (cause) { setError(String(cause)); }
    finally { setCredentialBusy(""); }
  }

  async function login(event: FormEvent) {
    event.preventDefault();
    setLoginBusy(true);
    setError("");
    try {
      const response = await fetch("/api/auth/session", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ token: loginCredential.trim() }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error ?? "Login failed.");
      setBrowserIdentity(result as BrowserIdentity);
      setLoginCredential("");
      await refresh();
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoginBusy(false);
      setAuthReady(true);
    }
  }

  async function logout() {
    await fetch("/api/auth/session", { method: "DELETE" });
    setBrowserIdentity(null);
    setIssuedCredential(null);
    setManagedCredentials([]);
    setSessions([]);
    setSelected(null);
    setPlan(null);
    setGraph({ nodes: [], edges: [] });
    setBudgetSummary(null);
    setArtifacts([]);
    setLiteratureArtifacts([]);
    setError("");
  }

  async function openPlan(session: Session) {
    setSelected(session);
    setLoadingPlan(true);
    setPlan(null);
    setBudgetSummary(null);
    setCitationLinks([]);
    setCitationMetrics([]);
    setArtifacts([]);
    setLiteratureArtifacts([]);
    setLiteratureCursor(null);
    setLiteratureHasMore(false);
    setLiteratureSearch("");
    setCitationFilter("");
    setLiteratureFilter("");
    setCitationFocus("");
    try {
      const [response, graphResponse, budgetResponse] = await Promise.all([
        fetch(api + "/v1/research/" + session.id + "/plan"),
        fetch(api + "/v1/research/" + session.id + "/graph"),
        fetch(api + "/v1/research/" + session.id + "/budget"),
      ]);
      const [citationResponse, watchedPaperResponse] = await Promise.all([
        fetch(api + "/v1/research/" + session.id + "/citation-links"),
        fetch(api + "/v1/research/" + session.id + "/literature-watch-papers?limit=100"),
      ]);
      if (!response.ok || !graphResponse.ok || !budgetResponse.ok || !citationResponse.ok || !watchedPaperResponse.ok) throw new Error(await (!response.ok ? response : !graphResponse.ok ? graphResponse : !budgetResponse.ok ? budgetResponse : !citationResponse.ok ? citationResponse : watchedPaperResponse).text());
      const [planData, graphData, budgetData, citationData, watchedPaperData] = await Promise.all([response.json(), graphResponse.json(), budgetResponse.json(), citationResponse.json(), watchedPaperResponse.json()]);
      const artifactBudget = { pages: 0, items: 0, bytes: 0 };
      const researchArtifacts: ResearchArtifact[][] = [];
      // These reads share a session-wide bounded budget. Keep them sequential and
      // load coverage first so concurrent mutations cannot race the counters or
      // starve the audit trail behind larger review artifact sets.
      for (const kind of ["literature_search_coverage", "drug_interaction_literature_coverage", "clinical_trial_snapshot", "clinical_trial_snapshot_comparison", "evidence_inventory", "dailymed_interaction_label_screen", "drug_interaction_distribution_comparison", "drug_interaction_pmc_fulltext_screen", "drug_interaction_study_adjudication", "drug_interaction_semantic_review", "drug_interaction_claim_adjudication", "hypothesis_candidates", "hypothesis_literature_comparison", "hypothesis_literature_candidate", "hypothesis_literature_relevance_review", "hypothesis_criticism", "hypothesis_critic_adjudication", "paper_claim_alignment_candidates", "paper_claim_semantic_alignment", "paper_claim_alignment_adjudication", "ontology_cross_mapping_review", "ontology_cross_mapping_adjudication", "capability_version_benchmark", "capability_version_benchmark_failure", "research_workflow_proposal", "research_reproducibility_bundle"]) {
        researchArtifacts.push(await readArtifactPages(session.id, kind, artifactBudget));
      }
      setPlan(planData);
      setGraph(graphData);
      setBudgetSummary(budgetData);
      setCitationLinks(citationData.items ?? []);
      setCitationMetrics(citationData.metrics ?? []);
      setArtifacts(researchArtifacts.flat());
      setWatchedPapers(watchedPaperData.items ?? []);
      setError("");
    } catch (cause) {
      setError(String(cause));
    } finally {
      setLoadingPlan(false);
    }
  }

  async function loadMoreLiterature() {
    if (!selected || !literatureCursor || loadingMoreLiterature) return;
    setLoadingMoreLiterature(true);
    try {
      const params = new URLSearchParams({ kind: "paper", limit: "100", after_recorded_at: String(literatureCursor.recorded_at), after_id: literatureCursor.id });
      if (literatureSearch) params.set("search", literatureSearch);
      const response = await fetch(api + "/v1/research/" + selected.id + "/artifacts?" + params.toString());
      if (!response.ok) throw new Error(await response.text());
      const page = await response.json();
      setLiteratureArtifacts(previous => [...previous, ...(page.items ?? [])]);
      setLiteratureCursor(page.next_cursor ?? null);
      setLiteratureHasMore(Boolean(page.next_cursor));
    } catch (cause) { setError(String(cause)); }
    finally { setLoadingMoreLiterature(false); }
  }

  useEffect(() => {
    let active = true;
    void fetch("/api/auth/session")
      .then(async response => {
        if (response.status === 401) return null;
        if (!response.ok) throw new Error(await response.text());
        return response.json() as Promise<BrowserIdentity>;
      })
      .then(async identity => {
        if (!active) return;
        setBrowserIdentity(identity);
        if (identity) await refresh();
      })
      .catch(cause => { if (active) setError(String(cause)); })
      .finally(() => { if (active) setAuthReady(true); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (view === "credentials" && ["tenant_admin", "platform_admin"].includes(browserIdentity?.role ?? "")) void loadCredentials().catch(cause => setError(String(cause)));
  }, [view, browserIdentity]);

  useEffect(() => {
    const timer = window.setTimeout(() => setLiteratureSearch(literatureFilter.trim()), 250);
    return () => window.clearTimeout(timer);
  }, [literatureFilter]);

  useEffect(() => {
    if (!selected || !plan || plan.session_id !== selected.id) return;
    const controller = new AbortController();
    const params = new URLSearchParams({ kind: "paper", limit: "100" });
    if (literatureSearch) params.set("search", literatureSearch);
    setLiteratureArtifacts([]);
    setLiteratureCursor(null);
    setLiteratureHasMore(false);
    setLoadingLiterature(true);
    void fetch(api + "/v1/research/" + selected.id + "/artifacts?" + params.toString(), { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(await response.text());
        return response.json();
      })
      .then(page => {
        setLiteratureArtifacts(page.items ?? []);
        setLiteratureCursor(page.next_cursor ?? null);
        setLiteratureHasMore(Boolean(page.next_cursor));
        setError("");
      })
      .catch(cause => { if (!controller.signal.aborted) setError(String(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoadingLiterature(false); });
    return () => controller.abort();
  }, [selected, plan, literatureSearch]);

  async function create(event: FormEvent) {
    event.preventDefault();
    if (!objective.trim()) return;
    setError("");
    try {
      const response = await fetch(api + "/v1/research", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ objective, depth, budget }),
      });
      if (!response.ok) throw new Error(await response.text());
      const session: Session = await response.json();
      setObjective("");
      await refresh();
      await openPlan(session);
    } catch (cause) {
      setError(String(cause));
    }
  }

  async function reviewResearchTask(task: Task, decision: "approve" | "reject") {
    if (!selected) return;
    const rationale = reviewReasons[task.id]?.trim();
    if (!rationale) { setError("A review rationale is required."); return; }
    setReviewBusy(task.id);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/tasks/${encodeURIComponent(task.id)}/review`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ decision, rationale }),
      });
      if (!response.ok) throw new Error(await response.text());
      setReviewReasons((previous: Record<string, string>) => ({ ...previous, [task.id]: "" }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function submitStudyReview(screen: ResearchArtifact, candidateId: string) {
    const key = `${screen.id}:${candidateId}`;
    if (!selected) return;
    const fields = studyReviews[key] ?? {};
    const required = ["study_design", "comparator", "dose_a", "dose_b", "route_a", "route_b", "endpoint", "outcome", "limitations", "label_evidence_overlap"];
    if (required.some(field => !fields[field]?.trim())) { setError("Complete every study review field before saving."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/drug-interaction-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ screen_artifact_id: screen.id, candidate_id: candidateId, exposure_status: fields.exposure_status ?? "unclear", ...fields }),
      });
      if (!response.ok) throw new Error(await response.text());
      setStudyReviews(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function submitInteractionClaimReview(screen: ResearchArtifact, reviewIndex: number) {
    const key = `${screen.id}:${reviewIndex}`;
    if (!selected) return;
    const fields = interactionClaimReviewForms[key] ?? {};
    if (!fields.rationale?.trim() || !fields.context_notes?.trim()) { setError("Add a rationale and study or label context before saving."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/drug-interaction-claim-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ screening_artifact_id: screen.id, review_index: reviewIndex, adjudication: fields.adjudication ?? "uncertain", claim_domain: fields.claim_domain ?? "uncertain", exposure_comparability: fields.exposure_comparability ?? "unclear", outcome_comparability: fields.outcome_comparability ?? "unclear", rationale: fields.rationale, context_notes: fields.context_notes }),
      });
      if (!response.ok) throw new Error(await response.text());
      setInteractionClaimReviewForms(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function submitHypothesisLiteratureReview(comparison: ResearchArtifact, paperArtifactId: string) {
    const key = paperArtifactId;
    if (!selected) return;
    const fields = hypothesisLiteratureReviewForms[key] ?? {};
    const required = ["rationale", "study_design", "population_or_model", "methods", "findings", "limitations"];
    if (required.some(field => !fields[field]?.trim())) { setError("Complete all structured literature review fields before saving."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/hypothesis-literature-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ comparison_artifact_id: comparison.id, paper_artifact_id: paperArtifactId, relevance: fields.relevance ?? "unclear", ...fields }),
      });
      if (!response.ok) throw new Error(await response.text());
      setHypothesisLiteratureReviewForms(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function submitHypothesisCriticReview(criticismId: string, hypothesisId: string, itemType: string, itemId: string) {
    if (!selected) return;
    const key = `${criticismId}:${hypothesisId}:${itemType}:${itemId}`;
    const fields = hypothesisCriticReviewForms[key] ?? {};
    if (!fields.rationale?.trim() || !fields.study_context?.trim()) { setError("Add a rationale and study context before saving the criticism review."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/hypothesis-critic-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ criticism_artifact_id: criticismId, hypothesis_id: hypothesisId, item_type: itemType, item_id: itemId, assessment: fields.assessment ?? "uncertain", rationale: fields.rationale, study_context: fields.study_context }),
      });
      if (!response.ok) throw new Error(await response.text());
      setHypothesisCriticReviewForms(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  function renderHypothesisCriticReview(criticismId: string, hypothesisId: string, itemType: string, itemId: string, label: string) {
    const key = `${criticismId}:${hypothesisId}:${itemType}:${itemId}`;
    const form = hypothesisCriticReviewForms[key] ?? {};
    const saved = artifacts.filter(item => item.kind === "hypothesis_critic_adjudication").map(item => ({ artifact: item, data: artifactData(item) })).filter(item => item.data.criticism_artifact_id === criticismId && item.data.hypothesis_id === hypothesisId && item.data.item_type === itemType && item.data.item_id === itemId);
    return <details key={key}><summary>{saved.length ? `Researcher reviews (${saved.length})` : `Review: ${label}`}</summary>{saved.map(item => <p key={item.artifact.id}>Reviewed by {String((item.data.reviewer as Record<string, unknown> | undefined)?.subject ?? "researcher")} · {String(item.data.assessment)} · {String(item.data.rationale)} · confidence unchanged.</p>)}<label>Assessment<select aria-label={`Assessment for ${label}`} value={form.assessment ?? "uncertain"} onChange={event => setHypothesisCriticReviewForms(previous => ({ ...previous, [key]: { ...previous[key], assessment: event.target.value } }))}><option value="supports_hypothesis">Supports hypothesis</option><option value="weakens_hypothesis">Weakens hypothesis</option><option value="context_only">Context only</option><option value="irrelevant">Not relevant</option><option value="uncertain">Uncertain</option></select></label><label className="review-field">Rationale<textarea required maxLength={2000} value={form.rationale ?? ""} onChange={event => setHypothesisCriticReviewForms(previous => ({ ...previous, [key]: { ...previous[key], rationale: event.target.value } }))} /></label><label className="review-field">Study context, methods, population, and limitations<textarea required maxLength={4000} value={form.study_context ?? ""} onChange={event => setHypothesisCriticReviewForms(previous => ({ ...previous, [key]: { ...previous[key], study_context: event.target.value } }))} /></label><button onClick={() => void submitHypothesisCriticReview(criticismId, hypothesisId, itemType, itemId)} disabled={reviewBusy === key}>{reviewBusy === key ? "Saving…" : "Save scientific review"}</button></details>;
  }

  async function submitClaimAlignmentReview(screening: ResearchArtifact) {
    const key = screening.id;
    if (!selected) return;
    const fields = claimAlignmentReviewForms[key] ?? {};
    if (!fields.rationale?.trim() || !fields.methods_population_notes?.trim()) { setError("Add a rationale and study methods/population notes before saving the review."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/claim-alignment-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ screening_artifact_id: screening.id, adjudication: fields.adjudication ?? "uncertain", population_comparability: fields.population_comparability ?? "unclear", method_comparability: fields.method_comparability ?? "unclear", rationale: fields.rationale, methods_population_notes: fields.methods_population_notes }),
      });
      if (!response.ok) throw new Error(await response.text());
      setClaimAlignmentReviewForms(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function submitOntologyMappingReview(source: ResearchArtifact, sourceIdentifier: string, candidate: Record<string, unknown>) {
    if (!selected || typeof candidate.target_curie !== "string") return;
    const target = candidate.target_curie;
    const key = `${source.id}:${target}`;
    const fields = ontologyMappingReviewForms[key] ?? {};
    if (!fields.rationale?.trim()) { setError("Add a rationale before saving the ontology mapping review."); return; }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/research/${encodeURIComponent(selected.id)}/ontology-cross-mapping-reviews`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ source_artifact_id: source.id, source_identifier: sourceIdentifier, target_curie: target, decision: fields.decision ?? "uncertain", rationale: fields.rationale }),
      });
      if (!response.ok) throw new Error(await response.text());
      setOntologyMappingReviewForms(previous => ({ ...previous, [key]: {} }));
      setError("");
      await openPlan(selected);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function transitionCapability(capability: Capability, next: "verified" | "production" | "deprecated") {
    const key = `${capability.id}@${capability.version}`;
    const reason = reviewReasons[key]?.trim();
    if (!reason) { setError("A capability lifecycle rationale is required."); return; }
    let validation: Record<string, unknown> | undefined;
    if (next === "verified" || next === "production") {
      try { validation = JSON.parse(validationReports[key] ?? "") as Record<string, unknown>; }
      catch { setError("Provide the JSON validation report for this promotion."); return; }
      validation = { ...validation, capability_id: capability.id, version: capability.version, manifest_sha256: capability.manifest_sha256 };
    }
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/capabilities/${encodeURIComponent(capability.id)}/transition`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ version: capability.version, next, reason, validation: validation ? JSON.stringify(validation) : null }),
      });
      if (!response.ok) throw new Error(await response.text());
      setError("");
      await refresh();
      const history = await fetch(`${api}/v1/capabilities/${encodeURIComponent(capability.id)}/${encodeURIComponent(capability.version)}/events`);
      if (history.ok) { const data = await history.json(); setCapabilityEvents((previous: Record<string, CapabilityEvent[]>) => ({ ...previous, [key]: data.items ?? [] })); }
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  async function loadCapabilityEvents(capability: Capability) {
    const key = `${capability.id}@${capability.version}`;
    try {
      const response = await fetch(`${api}/v1/capabilities/${encodeURIComponent(capability.id)}/${encodeURIComponent(capability.version)}/events`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setCapabilityEvents((previous: Record<string, CapabilityEvent[]>) => ({ ...previous, [key]: data.items ?? [] }));
      setError("");
    } catch (cause) { setError(String(cause)); }
  }

  async function rollbackCapability(capability: Capability) {
    const key = `${capability.id}@${capability.version}`;
    const reason = reviewReasons[key]?.trim();
    if (!reason || !rollbackTargets[key]) { setError("A rollback target and rationale are required."); return; }
    let validation: Record<string, unknown>;
    try { validation = JSON.parse(validationReports[key] ?? "") as Record<string, unknown>; }
    catch { setError("Provide the JSON validation report for this rollback."); return; }
    const target = capabilities.find(candidate => candidate.id === capability.id && candidate.version === rollbackTargets[key]);
    if (!target) { setError("The selected rollback target is no longer present in the capability registry. Refresh and review it again."); return; }
    validation = { ...validation, capability_id: target.id, version: target.version, manifest_sha256: target.manifest_sha256 };
    setReviewBusy(key);
    try {
      const response = await fetch(`${api}/v1/capabilities/${encodeURIComponent(capability.id)}/rollback`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ target_version: rollbackTargets[key], reason, validation: JSON.stringify(validation) }),
      });
      if (!response.ok) throw new Error(await response.text());
      setError("");
      await refresh();
      await loadCapabilityEvents(capability);
    } catch (cause) { setError(String(cause)); }
    finally { setReviewBusy(""); }
  }

  const stats = [
    ["Research sessions", summary.sessions, "persistent investigations"],
    ["Knowledge graph", summary.graph_nodes, graph.edges.length + " sourced relationships"],
    ["Evidence records", summary.evidence, "source linked claims"],
    ["Research artifacts", summary.artifacts, "content-hashed source records"],
    ["Capabilities", summary.capabilities, "governed registry entries"],
  ] as const;
  const filteredCitations = citationLinks.filter(link => !citationFilter.trim() || link.citing_doi.includes(citationFilter.trim().toLowerCase()) || link.cited_doi.includes(citationFilter.trim().toLowerCase()));
  const literatureRecords = literatureArtifacts.map(literatureRecord).filter((item): item is LiteratureRecord => item !== null);
  const literatureCoverage = artifacts.filter(item => item.kind === "literature_search_coverage" || item.kind === "drug_interaction_literature_coverage").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const trialSnapshots = artifacts.filter(item => item.kind === "clinical_trial_snapshot" || item.kind === "clinical_trial").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const trialComparisons = artifacts.filter(item => item.kind === "clinical_trial_snapshot_comparison").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const dailyMedLabelArtifacts = artifacts.filter(item => item.kind === "dailymed_interaction_label_screen").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const drugLabelComparisons = artifacts.filter(item => item.kind === "drug_interaction_distribution_comparison").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const interactionFulltextScreens = artifacts.filter(item => item.kind === "drug_interaction_pmc_fulltext_screen").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const interactionStudyReviews = artifacts.filter(item => item.kind === "drug_interaction_study_adjudication").map(item => ({ artifact: item, data: artifactData(item) }));
  const interactionSemanticScreens = artifacts.filter(item => item.kind === "drug_interaction_semantic_review").map(item => ({ artifact: item, data: artifactData(item) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const interactionClaimReviews = artifacts.filter(item => item.kind === "drug_interaction_claim_adjudication").map(item => ({ artifact: item, data: artifactData(item) }));
  const filteredLiterature = literatureRecords;
  const networkLinks = filteredCitations.slice(0, 60);
  const citationNodes: string[] = Array.from(new Set<string>(networkLinks.flatMap(link => [link.citing_doi, link.cited_doi]))).slice(0, 18);
  const point = (index: number) => {
    const angle = (Math.PI * 2 * index) / Math.max(1, citationNodes.length) - Math.PI / 2;
    return { x: 180 + Math.cos(angle) * 132, y: 165 + Math.sin(angle) * 122 };
  };
  const biologyNodes = graph.nodes.filter(node => /gene|variant|protein|pathway|disease|drug|compound|trial|phenotype|tissue|cell/i.test(node.kind));
  const hypothesisCandidates = artifacts.filter(item => item.kind === "hypothesis_candidates").flatMap(item => {
    const candidates = artifactData(item).candidates;
    return Array.isArray(candidates) ? candidates.map(candidate => ({ artifactId: item.id, candidate: candidate as Record<string, unknown> })) : [];
  });
  const hypothesisLiteratureComparisons = artifacts.filter(item => item.kind === "hypothesis_literature_comparison").map(item => ({ artifact: item, data: artifactData(item) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const hypothesisLiteraturePapers = artifacts.filter(item => item.kind === "hypothesis_literature_candidate").map(item => ({ artifact: item, data: artifactData(item) }));
  const hypothesisLiteratureReviews = artifacts.filter(item => item.kind === "hypothesis_literature_relevance_review").map(item => ({ artifact: item, data: artifactData(item) }));
  const hypothesisReviews = new Map<string, Record<string, unknown>>(artifacts.filter(item => item.kind === "hypothesis_criticism").flatMap(item => {
    const data = artifactData(item);
    const reviews = Array.isArray(data.reviews) ? data.reviews : [];
    return reviews.filter((review): review is Record<string, unknown> => Boolean(review) && typeof review === "object" && !Array.isArray(review)).map(review => [String(review.hypothesis_id ?? ""), { ...review, criticism_artifact_id: item.id }] as [string, Record<string, unknown>]);
  }));
  const claimAlignmentCandidates = artifacts.filter(item => item.kind === "paper_claim_alignment_candidates").flatMap(item => {
    const links = artifactData(item).links;
    return Array.isArray(links) ? links.map(link => ({ artifactId: item.id, link: link as Record<string, unknown> })) : [];
  });
  const semanticClaimReviews = artifacts.filter(item => item.kind === "paper_claim_semantic_alignment").map(item => ({ artifact: item, data: artifactData(item) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const workflowProposals = artifacts.filter(item => item.kind === "research_workflow_proposal").map(artifact => ({ artifact, data: artifactData(artifact) })).sort((a, b) => b.artifact.recorded_at - a.artifact.recorded_at);
  const reproducibilityArtifacts = artifacts.filter(item => item.kind === "research_reproducibility_bundle").sort((a, b) => b.recorded_at - a.recorded_at);
  const latestBundle = reproducibilityArtifacts[0];
  const evidenceInventoryArtifact = artifacts.filter(item => item.kind === "evidence_inventory").sort((a, b) => b.recorded_at - a.recorded_at)[0];
  const evidenceInventoryData = evidenceInventoryArtifact ? artifactData(evidenceInventoryArtifact) : {};
  const studyDesignInventory = evidenceInventoryData.study_design_inventory && typeof evidenceInventoryData.study_design_inventory === "object" ? evidenceInventoryData.study_design_inventory as Record<string, unknown> : {};
  const designCounts = studyDesignInventory.counts_by_provider_publication_type_signal && typeof studyDesignInventory.counts_by_provider_publication_type_signal === "object" ? studyDesignInventory.counts_by_provider_publication_type_signal as Record<string, unknown> : {};
  const designExamples = studyDesignInventory.example_pmids_by_signal && typeof studyDesignInventory.example_pmids_by_signal === "object" ? studyDesignInventory.example_pmids_by_signal as Record<string, unknown> : {};
  const truncatedDesignExamples = studyDesignInventory.example_pmids_truncated_by_signal && typeof studyDesignInventory.example_pmids_truncated_by_signal === "object" ? studyDesignInventory.example_pmids_truncated_by_signal as Record<string, unknown> : {};
  const qualityDomains = studyDesignInventory.quality_domains && typeof studyDesignInventory.quality_domains === "object" ? studyDesignInventory.quality_domains as Record<string, unknown> : {};
  const designCategories = Object.entries(designCounts).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1])).sort(([left], [right]) => left.localeCompare(right));
  const relatedPublicationCounts = studyDesignInventory.counts_by_pubmed_related_publication_type && typeof studyDesignInventory.counts_by_pubmed_related_publication_type === "object" ? studyDesignInventory.counts_by_pubmed_related_publication_type as Record<string, unknown> : {};
  const relatedPublicationExamples = studyDesignInventory.example_pmids_by_related_publication_type && typeof studyDesignInventory.example_pmids_by_related_publication_type === "object" ? studyDesignInventory.example_pmids_by_related_publication_type as Record<string, unknown> : {};
  const relatedPublicationCategories = Object.entries(relatedPublicationCounts).filter((entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1])).sort(([left], [right]) => left.localeCompare(right));
  const bundleData = latestBundle ? artifactData(latestBundle) : {};
  const bundleTasks = Array.isArray(bundleData.tasks) ? bundleData.tasks as Record<string, unknown>[] : [];
  const bundleArtifacts = Array.isArray(bundleData.artifacts) ? bundleData.artifacts as Record<string, unknown>[] : [];
  const bundleEvidence = Array.isArray(bundleData.evidence_annotations) ? bundleData.evidence_annotations as Record<string, unknown>[] : [];
  const bundleGraph = bundleData.graph && typeof bundleData.graph === "object" ? bundleData.graph as Record<string, unknown> : {};
  const bundleNodes = Array.isArray(bundleGraph.nodes) ? bundleGraph.nodes : [];
  const bundleEdges = Array.isArray(bundleGraph.edges) ? bundleGraph.edges : [];
  const reproductionTask = plan?.tasks.find(task => task.id === "report.reproduce");
  const meteredApiEvents = budgetSummary?.events.filter(event => event.task_id === "api_request") ?? [];
  const modelReservations = budgetSummary?.model_reservations ?? [];
  const pendingModelReservations = modelReservations.filter(reservation => reservation.state === "reserved").length;
  const uncertainModelReservations = modelReservations.filter(reservation => reservation.state === "uncertain").length;
  const meteredApiCredits = meteredApiEvents.reduce((total, event) => total + event.credits, 0);

  if (!authReady) return <main className="login-shell"><section className="login-card" role="status">Checking browser session…</section></main>;
  if (!browserIdentity) return (
    <main className="login-shell">
      <section className="login-card">
        <div className="brand"><div className="logo"><Microscope size={20} /></div><div><b>MedAI</b><small>RESEARCH INTELLIGENCE</small></div></div>
        <h1>Sign in to your research workspace</h1>
        <p>Use your personal MedAI API credential. The server validates its role and tenant, then keeps it in an encrypted, HttpOnly session cookie.</p>
        <form onSubmit={event => void login(event)}>
          <label htmlFor="api-login-credential">Personal API credential</label>
          <input id="api-login-credential" type="password" autoComplete="off" value={loginCredential} onChange={event => setLoginCredential(event.target.value)} required minLength={16} maxLength={4096} />
          {error && <div className="error" role="alert">{error}</div>}
          <button type="submit" disabled={loginBusy || !loginCredential.trim()}>{loginBusy ? "Signing in…" : "Sign in"}</button>
        </form>
      </section>
    </main>
  );

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><div className="logo"><Microscope size={20} /></div><div><b>MedAI</b><small>RESEARCH INTELLIGENCE</small></div></div>
        <div className="workspace-label">WORKSPACE</div>
        <nav>
          <button aria-current={view === "overview" ? "page" : undefined} className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}><Activity size={17} /> Overview</button>
          <button aria-current={view === "literature" ? "page" : undefined} className={view === "literature" ? "active" : ""} onClick={() => setView("literature")}><BookOpenText size={17} /> Literature</button>
          <button aria-current={view === "biology" ? "page" : undefined} className={view === "biology" ? "active" : ""} onClick={() => setView("biology")}><Dna size={17} /> Biology</button>
          <button aria-current={view === "graph" ? "page" : undefined} className={view === "graph" ? "active" : ""} onClick={() => setView("graph")}><Network size={17} /> Knowledge graph</button>
          <button aria-current={view === "hypotheses" ? "page" : undefined} className={view === "hypotheses" ? "active" : ""} onClick={() => setView("hypotheses")}><FlaskConical size={17} /> Hypotheses</button>
          <button aria-current={view === "experiments" ? "page" : undefined} className={view === "experiments" ? "active" : ""} onClick={() => setView("experiments")}><ShieldCheck size={17} /> Reproducibility</button>
          <button aria-current={view === "capabilities" ? "page" : undefined} className={view === "capabilities" ? "active" : ""} onClick={() => setView("capabilities")}><GitBranch size={17} /> Capabilities</button>
          <button aria-current={view === "review" ? "page" : undefined} className={view === "review" ? "active" : ""} onClick={() => setView("review")}><ClipboardCheck size={17} /> Review</button>
          {["tenant_admin", "platform_admin"].includes(browserIdentity.role) && <button aria-current={view === "credentials" ? "page" : undefined} className={view === "credentials" ? "active" : ""} onClick={() => setView("credentials")}><KeyRound size={17} /> Credentials</button>}
        </nav>
        <div className="sidebar-foot"><ShieldCheck size={16} /><span>Signed in as {browserIdentity.subject}<br /><small>{browserIdentity.tenant_id} · {browserIdentity.role}</small></span></div>
      </aside>
      <section className="main">
        <header><div><span className="eyebrow">BIOMEDICAL AUTONOMOUS RESEARCH</span><h1>{view === "literature" ? "Literature explorer" : view === "biology" ? "Biomedical entities" : view === "graph" ? "Knowledge graph" : view === "hypotheses" ? "Hypothesis review" : view === "experiments" ? "Reproducibility record" : view === "capabilities" ? "Capability registry" : view === "review" ? "Human review" : view === "credentials" ? "Credential administration" : "Research workspace"}</h1></div><div className="status"><i /> Local system</div><div className="account-controls"><span>{browserIdentity.subject} · {browserIdentity.role}</span><button type="button" onClick={() => void logout()}>Sign out</button></div></header>
        <section className="hero">
          <div className="hero-copy">
            <span className="pill">RESEARCH DIRECTOR</span>
            <h2>Turn a research question<br />into an evidence trail.</h2>
            <p>Create a persistent research session and inspect its dependency-aware investigation plan.</p>
            <form onSubmit={create}><Search size={18} /><input aria-label="Research objective" value={objective} onChange={event => setObjective(event.target.value)} placeholder="What would you like to investigate?" /><select aria-label="Research depth" value={depth} onChange={event => setDepth(event.target.value)}><option value="standard">Standard</option><option value="deep">Deep</option></select><input aria-label="Execution credit budget" type="number" min="1" max="100000" value={budget} onChange={event => setBudget(Number(event.target.value))} /><button aria-label="Start investigation"><Plus size={18} /> Investigate</button></form>
            {error && <div className="error">{error}</div>}
          </div>
          <div className="hero-art"><div className="orbit o1" /><div className="orbit o2" /><div className="core"><Dna size={42} /></div><span className="node n1">GENE</span><span className="node n2">PAPER</span><span className="node n3">PATHWAY</span><span className="node n4">EVIDENCE</span></div>
        </section>
        <div className="stats">{stats.map(([label, value, sub]) => <article className="stat" key={label}><small>{label}</small><strong>{value}</strong><span>{sub}</span></article>)}</div>
        <div className="section-heading"><div><h3>Research sessions</h3><p>Persistent computational investigations</p></div><span className="count">{sessions.length} total</span></div>
        <div className="sessions">
          {sessions.length === 0 ? <div className="empty"><div className="empty-icon"><BookOpenText /></div><b>No research sessions yet</b><span>Start with a disease, mechanism, gene, or scientific question.</span></div> : sessions.map(session => (
            <button className={"session " + (selected?.id === session.id ? "session-selected" : "")} key={session.id} onClick={() => void openPlan(session)}>
              <span className="session-icon"><Microscope size={18} /></span>
              <span className="session-body"><b>{session.objective}</b><span>{session.id} · {new Date(session.created_at * 1000).toLocaleString()}</span></span>
              <span className={"planned status-" + session.status}>{session.status.replace("_", " ")}</span>
            </button>
          ))}
        </div>
        {view === "credentials" && <section className="plan-panel">
          <div className="section-heading"><div><h3>API credentials</h3><p>Issue and revoke tenant-scoped credentials. Tokens are displayed only at creation.</p></div><button type="button" className="review-secondary" onClick={() => void loadCredentials().catch(cause => setError(String(cause)))}>Refresh</button></div>
          <form className="credential-create" onSubmit={event => void createManagedCredential(event)}>
            <label className="review-credential">Subject<input required minLength={1} maxLength={128} value={credentialSubject} onChange={event => setCredentialSubject(event.target.value)} placeholder="researcher-name" /></label>
            {browserIdentity.role === "platform_admin" && <label className="review-credential">Tenant ID<input required minLength={1} maxLength={128} value={credentialTenant} onChange={event => setCredentialTenant(event.target.value)} placeholder={browserIdentity.tenant_id} /></label>}
            <label className="review-credential">Role<select value={credentialRole} onChange={event => setCredentialRole(event.target.value)}>{(browserIdentity.role === "platform_admin" ? ["reader", "researcher", "tenant_admin", "platform_admin", "service"] : ["reader", "researcher"]).map(role => <option key={role} value={role}>{role}</option>)}</select></label>
            <button type="submit" disabled={credentialBusy === "create" || !credentialSubject.trim()}>{credentialBusy === "create" ? "Creating…" : "Create credential"}</button>
          </form>
          {issuedCredential && <section className="issued-credential" aria-live="polite"><b>Credential created · copy it now</b><p>{issuedCredential.subject} · {issuedCredential.tenant_id} · {issuedCredential.role}</p><code>{issuedCredential.token}</code><button type="button" onClick={() => void navigator.clipboard?.writeText(issuedCredential.token)}>Copy token</button><button type="button" className="review-secondary" onClick={() => setIssuedCredential(null)}>Dismiss</button></section>}
          {managedCredentials.length === 0 ? <div className="empty compact">No credentials are registered for this administration scope.</div> : <div className="capability-list">{managedCredentials.map(credential => <article key={credential.id}><div><b>{credential.subject}</b><span>{credential.tenant_id} · {credential.role}</span></div><code>{credential.id}</code><span className={`task-status ${credential.active ? "status-succeeded" : "status-deprecated"}`}>{credential.active ? "active" : "revoked"}</span><small>Created {new Date(credential.created_at * 1000).toLocaleString()}</small>{credential.active && <button type="button" className="review-reject" disabled={Boolean(credentialBusy)} onClick={() => void revokeManagedCredential(credential)}>{credentialBusy === credential.id ? "Revoking…" : "Revoke"}</button>}</article>)}</div>}
        </section>}
        {view === "literature" && <section className="plan-panel literature-panel">
          <div className="section-heading"><div><h3>PubMed records</h3><p>{selected ? `${selected.id} · search all retained paper metadata and abstracts in this session` : "Select a research session to inspect its papers."}</p></div><span className="count">{literatureRecords.length} results loaded</span></div>
          {selected && <section aria-label="Clinical trial record snapshots" className="coverage-list"><div className="section-heading"><div><h3>ClinicalTrials.gov records</h3><p>Captured registry records and observed field changes. Provider record history opens on ClinicalTrials.gov.</p></div><span className="count">{trialSnapshots.length} snapshots</span></div>{trialSnapshots.length === 0 ? <p className="empty compact">No ClinicalTrials.gov record snapshots are retained for this session.</p> : trialSnapshots.map(({ artifact, data }) => { const nctId = String(data.nct_id ?? artifact.source_id ?? ""); const lifecycle = data.lifecycle && typeof data.lifecycle === "object" ? data.lifecycle as Record<string, unknown> : {}; const historyUrl = officialTrialHistoryUrl(nctId); return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">CLINICALTRIALS.GOV</span><code>{nctId}</code><span>{String(lifecycle.overall_status ?? "status unavailable")}</span><span>Captured {new Date(artifact.recorded_at * 1000).toLocaleString()}</span></div><h4>{String((data.identification as Record<string, unknown> | undefined)?.brief_title ?? nctId)}</h4><p>Automated history includes snapshots retained by MedAI; it does not backfill earlier provider versions.</p>{historyUrl ? <a href={historyUrl} target="_blank" rel="noreferrer">Open official record history for {nctId}</a> : <p>Provider history link unavailable because the stored study identifier is invalid.</p>}</article>; })}{trialComparisons.length > 0 && <div><h4>Observed snapshot comparisons</h4>{trialComparisons.map(({ artifact, data }) => { const paths = Array.isArray(data.changed_field_paths) ? data.changed_field_paths : []; return <article key={artifact.id}><div className="literature-record-meta"><code>{String(data.nct_id ?? artifact.source_id)}</code><span>{data.changed === true ? `${paths.length} changed fields` : "No changed fields"}</span><span>{new Date(artifact.recorded_at * 1000).toLocaleString()}</span></div>{paths.length > 0 ? <p>{paths.slice(0, 20).map(String).join(" · ")}{paths.length > 20 ? ` · and ${paths.length - 20} more` : ""}</p> : null}<small>Observed field changes only; scientific interpretation not performed.</small></article>; })}</div>}</section>}
          {selected && dailyMedLabelArtifacts.length > 0 && <section aria-label="DailyMed label version history" className="coverage-list"><div className="section-heading"><div><h3>DailyMed label versions</h3><p>Provider label revisions with source hashes. Earlier versions belong to the same label record and are not independent clinical evidence.</p></div><span className="count">{dailyMedLabelArtifacts.length} retrievals</span></div>{dailyMedLabelArtifacts.map(({ artifact, data }) => { const labelsByDrug = data.labels && typeof data.labels === "object" && !Array.isArray(data.labels) ? data.labels as Record<string, unknown> : {}; return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">DAILYMED · VERSIONED SPL</span><code>{artifact.id}</code><span>{new Date(artifact.recorded_at * 1000).toLocaleString()}</span></div>{Object.entries(labelsByDrug).flatMap(([drug, rawRows]) => Array.isArray(rawRows) ? rawRows.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row)).map((label, index) => { const history = label.historical_versions && typeof label.historical_versions === "object" ? label.historical_versions as Record<string, unknown> : {}; const versions = Array.isArray(history.records) ? history.records.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : []; const setId = String(label.set_id ?? ""); const labelUrl = setId ? `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=${encodeURIComponent(setId)}` : ""; return <div className="hypothesis-review" key={`${drug}:${setId}:${index}`}><b>{drug} · {String(label.title ?? "Untitled label")} · version {String(label.spl_version ?? "unknown")}</b><p>Published {String(label.published_date ?? "date unavailable")} · {versions.length} earlier versions retained · {history.truncated === true ? "history is bounded or truncated" : "within configured history bound"}</p><p>Current SPL SHA-256: <code>{String(label.spl_sha256 ?? "unavailable")}</code></p>{labelUrl && <a href={labelUrl} target="_blank" rel="noreferrer">Open DailyMed label {setId}</a>}{versions.length > 0 && <ul>{versions.map((version, versionIndex) => <li key={`${setId}:${String(version.spl_version)}:${versionIndex}`}>Version {String(version.spl_version)} · {String(version.published_date ?? "date unavailable")} · section SHA-256 <code>{String(version.section_sha256 ?? "unavailable")}</code> · archive SHA-256 <code>{String(version.spl_sha256 ?? "unavailable")}</code>{version.archive_retrieved !== true && " · archive unavailable"}</li>)}</ul>}</div>; }) : [])}</article>; })}{drugLabelComparisons.length > 0 && <div><h4>Observed historical section changes</h4>{drugLabelComparisons.map(({ artifact, data }) => { const diffs = Array.isArray(data.historical_version_diffs) ? data.historical_version_diffs.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : []; return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">SOURCE VERSION COMPARISON</span><code>{artifact.id}</code><span>{diffs.length} comparisons · no clinical verdict</span></div>{diffs.length === 0 ? <p>No prior DailyMed versions were available for comparison.</p> : <ul>{diffs.map((diff, index) => { const setId = String(diff.set_id ?? ""); const recordUrl = setId ? `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=${encodeURIComponent(setId)}` : ""; return <li key={`${artifact.id}:${index}`}>{String(diff.drug_query)} · {String(diff.set_id)} · version {String(diff.current_version)} vs {String(diff.historical_version)} · interaction section {diff.interaction_section_hash_match === true ? "hash matches" : "hash differs or is unavailable"}{recordUrl && <> · <a href={recordUrl} target="_blank" rel="noreferrer">open source record</a></>}</li>; })}</ul>}<small>Hash changes are source review leads; they do not establish clinical significance or a drug interaction.</small></article>; })}</div>}</section>}
          {selected && <section aria-label="PubMed retrieval coverage" className="coverage-list"><div className="section-heading"><div><h3>Search coverage</h3><p>Bounded PubMed retrieval counts; a limited page does not establish a complete literature search.</p></div><span className="count">{literatureCoverage.length} search runs</span></div>{literatureCoverage.length === 0 ? <p className="empty compact">No PubMed coverage receipts are retained for this session.</p> : literatureCoverage.map(({ artifact, data }) => { const rawCoverage = artifact.kind === "drug_interaction_literature_coverage" ? data.provider_coverage : data.coverage; const coverage = rawCoverage && typeof rawCoverage === "object" ? rawCoverage as Record<string, unknown> : {}; const hasCoverage = Object.keys(coverage).length > 0; const missing = Array.isArray(coverage.missing_record_pmids) ? coverage.missing_record_pmids : []; return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">{artifact.kind === "drug_interaction_literature_coverage" ? "DRUG-PAIR PUBMED COVERAGE" : "PUBMED COVERAGE"}</span><code>{String(data.query_sha256 ?? artifact.source_id).slice(0, 16)}</code><span>{String(artifact.recorded_at ? new Date(artifact.recorded_at * 1000).toLocaleString() : "")}</span></div>{hasCoverage ? <><p>{String(coverage.returned_pmids ?? 0)} PMIDs returned of {String(coverage.total_hits ?? "unknown")} total hits; {String(coverage.fetched_records ?? 0)} records fetched.</p><p>{coverage.search_truncated === true ? "The search result page was truncated at its configured limit." : "The returned ESearch page covered the provider-reported result count."}{missing.length > 0 ? ` EFetch omitted ${missing.length} PMIDs.` : ""}</p></> : <p>Provider coverage unavailable for this retrieval. Status: {String(data.status ?? "unknown")}.</p>}</article>; })}</section>}
          {selected && interactionFulltextScreens.length > 0 && <section aria-label="Drug interaction full-text review leads" className="coverage-list"><div className="section-heading"><div><h3>Drug interaction full-text review leads</h3><p>PMC sentence matches are source locations for human review. They do not establish that the drug pair was administered or interacted.</p></div><span className="count">{interactionFulltextScreens.length} screens</span></div>{interactionFulltextScreens.map(({ artifact, data }) => { const candidates = Array.isArray(data.candidates) ? data.candidates.filter((candidate): candidate is Record<string, unknown> => Boolean(candidate) && typeof candidate === "object" && !Array.isArray(candidate)) : []; return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">PMC BIOC · REVIEW LEADS</span><code>{String(data.pmid ?? artifact.source_id)} / {String(data.pmcid ?? "")}</code><span>License: {String(data.source_license ?? "not reported")}</span></div><h4>{String(data.article_title ?? "Untitled PMC article")}</h4><p>{String(data.candidate_count ?? candidates.length)} source locations · {String(data.omitted_candidate_count ?? 0)} additional candidates omitted · verdict: none</p><p>{data.pmcid ? <a href={`https://pmc.ncbi.nlm.nih.gov/articles/${encodeURIComponent(String(data.pmcid))}/`} target="_blank" rel="noreferrer">Open article in PMC</a> : "PMC article link unavailable."}</p>{candidates.length > 0 ? <ul>{candidates.map(candidate => { const candidateId = String(candidate.candidate_id); const key = `${artifact.id}:${candidateId}`; const form = studyReviews[key] ?? {}; const related = interactionStudyReviews.filter(review => review.data.screen_artifact_id === artifact.id && review.data.candidate_id === candidateId); const fields = [["study_design", "Study design"], ["comparator", "Comparator"], ["dose_a", "Drug A dose"], ["dose_b", "Drug B dose"], ["route_a", "Drug A route"], ["route_b", "Drug B route"], ["endpoint", "Endpoint"], ["outcome", "Observed outcome"], ["limitations", "Limitations"], ["label_evidence_overlap", "Overlap with label evidence"]] as const; return <li key={candidateId}><b>{String(candidate.section ?? "unclassified")}</b> · source offset {String(candidate.source_offset ?? "unknown")}, length {String(candidate.source_length ?? "unknown")} · both ingredients matched: {candidate.both_ingredients_in_candidate === true ? "yes" : "no"} · <code>{String(candidate.text_sha256 ?? "")}</code><details><summary>Record structured researcher review</summary><div className="citation-toolbar"><label>Exposure status<select aria-label="Pair exposure status" value={form.exposure_status ?? "unclear"} onChange={event => setStudyReviews(previous => ({ ...previous, [key]: { ...previous[key], exposure_status: event.target.value } }))}><option value="both">Both administered</option><option value="drug_a_only">Drug A only</option><option value="drug_b_only">Drug B only</option><option value="neither">Neither administered</option><option value="unclear">Unclear</option></select></label></div>{fields.map(([field, label]) => <label className="review-field" key={field}>{label}<textarea maxLength={field === "outcome" || field === "limitations" ? 2000 : field === "comparator" || field === "endpoint" ? 1000 : field.startsWith("dose") || field.startsWith("route") ? 256 : 120} value={form[field] ?? ""} onChange={event => setStudyReviews(previous => ({ ...previous, [key]: { ...previous[key], [field]: event.target.value } }))} /></label>)}<button onClick={() => void submitStudyReview(artifact, candidateId)} disabled={reviewBusy === key || Boolean(related.length)}>{related.length ? "Review recorded" : reviewBusy === key ? "Saving…" : "Save researcher review"}</button>{related.map(review => <p key={review.artifact.id}>Reviewed by {String((review.data.reviewer as Record<string, unknown> | undefined)?.subject ?? "researcher")} · {new Date(review.artifact.recorded_at * 1000).toLocaleString()} · annotation only; no interaction verdict.</p>)}</details></li>; })}</ul> : <p>No anchored lexical review leads were retained for this article.</p>}<p>{String(data.interpretation_limit ?? "Full-text lexical matches are unvalidated review leads.")}</p></article>; })}</section>}
          {selected && interactionSemanticScreens.length > 0 && <section aria-label="Drug interaction label and paper claim adjudication" className="coverage-list"><div className="section-heading"><div><h3>Label and paper claim comparisons</h3><p>Versioned label and PubMed or licensed PMC spans are model-screening leads. Researcher adjudication records context without assigning an interaction verdict.</p></div><span className="count">{interactionSemanticScreens.length} screens</span></div>{interactionSemanticScreens.map(({ artifact, data }) => { const reviews = Array.isArray(data.reviews) ? data.reviews.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : []; return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">VERSIONED LABEL · {String(reviews[0]?.sources && Array.isArray(reviews[0].sources) ? (reviews[0].sources as Record<string, unknown>[])[1]?.source ?? "PAPER" : "PAPER").toUpperCase()}</span><code>{artifact.id}</code><span>{String(data.candidate_pair_count ?? reviews.length)} screened pairs · verdict: none</span></div>{reviews.map((screen, index) => { const key = `${artifact.id}:${index}`; const sources = Array.isArray(screen.sources) ? screen.sources.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : []; const saved = interactionClaimReviews.filter(item => item.data.screening_artifact_id === artifact.id && item.data.review_index === index); const form = interactionClaimReviewForms[key] ?? {}; const setField = (field: string, value: string) => setInteractionClaimReviewForms(previous => ({ ...previous, [key]: { ...previous[key], [field]: value } })); return <div className="hypothesis-review" key={index}><b>{String(screen.relation ?? "uncertain")} · {String(screen.lexical_similarity ?? "")} lexical similarity</b><div className="hypothesis-path">{sources.map((source, sourceIndex) => <span key={sourceIndex}>{String(source.source ?? "source")} · {String(source.source_id ?? "record")} · {String(source.set_id ?? "")} · version {String(source.version ?? "unavailable")} · offset {String(source.offset ?? "?")} · {String(source.text_sha256 ?? "").slice(0, 16)}</span>)}</div>{saved.map(item => <p key={item.artifact.id}>Reviewed by {String((item.data.reviewer as Record<string, unknown> | undefined)?.subject ?? "researcher")} · {String(item.data.adjudication)} · {String(item.data.claim_domain)} · interaction verdict: none. {String(item.data.rationale)}</p>)}<details><summary>{saved.length ? "Add another adjudication" : "Adjudicate source relation and context"}</summary><label>Claim relation<select aria-label="Claim relation" value={form.adjudication ?? "uncertain"} onChange={event => setField("adjudication", event.target.value)}><option value="same_question_consistent">Same question, consistent</option><option value="same_question_opposing">Same question, opposing</option><option value="related_but_distinct">Related, distinct questions</option><option value="unrelated">Unrelated</option><option value="uncertain">Uncertain</option></select></label><label>Claim domain<select aria-label="Claim domain" value={form.claim_domain ?? "uncertain"} onChange={event => setField("claim_domain", event.target.value)}><option value="pharmacokinetic">Pharmacokinetic</option><option value="pharmacodynamic">Pharmacodynamic</option><option value="label_scope">Label scope</option><option value="uncertain">Uncertain</option></select></label><label>Exposure comparability<select aria-label="Exposure comparability" value={form.exposure_comparability ?? "unclear"} onChange={event => setField("exposure_comparability", event.target.value)}><option value="comparable">Comparable</option><option value="different">Different</option><option value="unclear">Unclear</option><option value="not_reported">Not reported</option></select></label><label>Outcome comparability<select aria-label="Outcome comparability" value={form.outcome_comparability ?? "unclear"} onChange={event => setField("outcome_comparability", event.target.value)}><option value="comparable">Comparable</option><option value="different">Different</option><option value="unclear">Unclear</option><option value="not_reported">Not reported</option></select></label><label className="review-field">Researcher rationale<textarea maxLength={2000} value={form.rationale ?? ""} onChange={event => setField("rationale", event.target.value)} /></label><label className="review-field">Study and label context<textarea maxLength={4000} value={form.context_notes ?? ""} onChange={event => setField("context_notes", event.target.value)} /></label><button onClick={() => void submitInteractionClaimReview(artifact, index)} disabled={reviewBusy === key}>{reviewBusy === key ? "Saving…" : "Save authenticated adjudication"}</button><small>Opposing labels remain a research annotation; no clinical interaction conclusion is generated. Do not enter patient identifiers or protected health information.</small></details></div>; })}<p>{String(data.interpretation_limit ?? "Model screening only; human review required.")}</p></article>; })}</section>}
          {!selected ? <div className="empty compact">Choose a research session to load literature records.</div> : <>
            <div className="citation-toolbar"><label htmlFor="literature-filter"><Search size={15} /><input id="literature-filter" aria-label="Search all retained literature" value={literatureFilter} onChange={event => setLiteratureFilter(event.target.value)} placeholder="Search all titles, abstracts, authors, PMID" /></label><span>{filteredLiterature.length} matches on this page</span></div>
            {loadingLiterature && <div className="empty compact" role="status">Searching retained literature…</div>}
            {filteredLiterature.length === 0 && !loadingLiterature ? <div className="empty compact">{literatureSearch ? "No retained PubMed records match this search." : "No PubMed records are retained for this session yet. Run the research worker to retrieve literature."}</div> : <div className="literature-record-list">{filteredLiterature.slice(0, 100).map(record => <article key={record.key}>
              <div className="literature-record-meta"><span className="graph-kind">PUBMED</span><code>{record.sourceId}</code>{record.date && <span>{record.date}</span>}{record.journal && <span>{record.journal}</span>}</div>
              <h4>{record.url ? <a href={record.url} target="_blank" rel="noreferrer">{record.title}</a> : record.title}</h4>
              {record.authors.length > 0 && <p className="literature-authors">{record.authors.join(", ")}</p>}
              {(record.publicationTypes.length > 0 || record.meshHeadings.length > 0) && <details><summary>Publication and MeSH indexing metadata</summary><p className="literature-authors">Publication types: {record.publicationTypes.length > 0 ? record.publicationTypes.join(" · ") : "not supplied"}</p><p className="literature-authors">MeSH descriptors: {record.meshHeadings.length > 0 ? record.meshHeadings.map(item => `${item.descriptor}${item.descriptorId ? ` [${item.descriptorId}]` : ""}`).join(" · ") : "not supplied"}</p><small>Indexing metadata can be incomplete; MeSH terms describe indexed topics and do not establish study quality.</small></details>}
              {record.relatedPublications.length > 0 && <details><summary>PubMed related-publication links ({record.relatedPublications.length})</summary><ul>{record.relatedPublications.map((link, index) => <li key={`${link.refType}-${link.pmid}-${index}`}><code>{link.refType}</code> · <a href={`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(link.pmid)}/`} target="_blank" rel="noreferrer">PMID {link.pmid}</a></li>)}</ul><small>These are relationships supplied by PubMed, including possible correction, expression-of-concern, or retraction links. Check the linked records to determine scope; links alone do not grade evidence quality.</small></details>}
              {record.abstract ? <details><summary>Read abstract</summary><p className="literature-abstract">{record.abstract}</p></details> : <p className="literature-authors">Abstract text is not retained; open the PubMed record to read it.</p>}
            </article>)}</div>}
            {literatureHasMore && <button className="load-more" onClick={() => void loadMoreLiterature()} disabled={loadingMoreLiterature || loadingLiterature}>{loadingMoreLiterature ? "Loading…" : "Load 100 more results"}</button>}
          </>}
          <div className="section-heading citation-heading"><div><h3>PubMed watch discoveries</h3><p>New bibliographic records from scheduled PubMed watches. These are discovery leads, not validated evidence.</p></div><span className="count">{watchedPapers.length} records</span></div>
          {selected && watchedPapers.length > 0 ? <div className="literature-record-list">{watchedPapers.map(paper => <article key={`${paper.watch_id}-${paper.pmid}`}><div className="literature-record-meta"><span className="graph-kind">WATCH · PUBMED</span><code>{paper.pmid}</code>{paper.publication_date && <span>{paper.publication_date}</span>}{paper.journal && <span>{paper.journal}</span>}</div><h4><a href={`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(paper.pmid)}/`} target="_blank" rel="noreferrer">{paper.title}</a></h4><p className="literature-authors">Watch {paper.watch_id.slice(0, 12)} · discovery event {paper.event_id} · captured {new Date(paper.captured_at * 1000).toLocaleString()}{paper.doi ? ` · DOI ${paper.doi}` : ""}</p></article>)}</div> : selected ? <div className="empty compact">No new bibliographic records have been retained by PubMed watches for this session.</div> : null}
          <div className="section-heading citation-heading"><div><h3>Crossref citation network</h3><p>{selected ? `${selected.id} · publisher-deposited citation metadata` : "Select a research session to inspect citation links."}</p></div><span className="count">{citationLinks.length} indexed links</span></div>
          {!selected ? <div className="empty compact">Choose a research session to load its literature links.</div> : <>
            <div className="citation-toolbar"><label htmlFor="citation-filter"><Search size={15} /><input id="citation-filter" value={citationFilter} onChange={event => setCitationFilter(event.target.value)} placeholder="Filter by citing or cited DOI" /></label><span>Showing at most 60 links in the network view</span></div>
            {citationMetrics.length > 0 && <div className="citation-table-wrap"><table><caption>Most connected papers in the captured citation graph</caption><thead><tr><th>DOI</th><th>Citations received</th><th>References tracked</th><th>PageRank</th></tr></thead><tbody>{citationMetrics.slice(0, 8).map(metric => <tr key={metric.doi}><td><button onClick={() => setCitationFocus(metric.doi)}>{metric.doi}</button></td><td>{metric.citations_received}</td><td>{metric.references_tracked}</td><td>{metric.pagerank.toFixed(4)}</td></tr>)}</tbody></table><p>Connectivity within retrieved Crossref records only; coverage may be incomplete. These metrics do not indicate paper quality, claim support, evidence strength, or novelty.</p></div>}
            {filteredCitations.length === 0 ? <div className="empty compact">No Crossref citation links are indexed for this session. Resolve papers through the AgentMesh Crossref tool to populate this graph.</div> : <>
              <div className="citation-network-wrap"><svg className="citation-network" viewBox="0 0 360 330" role="img" aria-label="Directed graph of publisher-deposited paper citations">
                <defs><marker id="citation-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#83a99a" /></marker></defs>
                {networkLinks.map((link, index) => {
                  const from = citationNodes.indexOf(link.citing_doi); const to = citationNodes.indexOf(link.cited_doi);
                  if (from < 0 || to < 0) return null;
                  const a = point(from); const b = point(to);
                  return <line key={`${link.citing_doi}-${link.cited_doi}-${index}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#a9c7bb" strokeWidth="1.2" markerEnd="url(#citation-arrow)" />;
                })}
                {citationNodes.map((doi, index) => { const p = point(index); return <g key={doi} role="button" tabIndex={0} aria-label={`Focus ${doi}`} onClick={() => setCitationFocus(doi)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") setCitationFocus(doi); }} className="citation-node"><circle cx={p.x} cy={p.y} r="8" fill={citationFocus === doi ? "#176b5b" : "#72a891"} /><text x={p.x} y={p.y + 19} textAnchor="middle">{doi.length > 19 ? `${doi.slice(0, 17)}…` : doi}</text></g>; })}
              </svg><p>Arrow direction: citing paper → cited paper. Links report citation metadata, not evidence of support or contradiction.</p></div>
              <div className="citation-table-wrap"><table><thead><tr><th>Citing DOI</th><th>Cited DOI</th><th>Source artifact</th></tr></thead><tbody>{filteredCitations.map((link, index) => <tr key={`${link.citing_doi}-${link.cited_doi}-${index}`} className={citationFocus && (link.citing_doi === citationFocus || link.cited_doi === citationFocus) ? "citation-row-focus" : ""}><td><button onClick={() => setCitationFocus(link.citing_doi)}>{link.citing_doi}</button></td><td><button onClick={() => setCitationFocus(link.cited_doi)}>{link.cited_doi}</button></td><td><code>{link.source_artifact_id}</code></td></tr>)}</tbody></table></div>
            </>}
          </>}
        </section>}
        {view === "biology" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Genes, variants, proteins, pathways and related entities</h3><p>{selected ? `${selected.id} · source-linked session records` : "Select a research session to inspect its biological records."}</p></div><span className="count">{selected ? biologyNodes.length : 0} entities</span></div>
          {!selected ? <div className="empty compact">Choose a research session to load biological entities.</div> : biologyNodes.length === 0 ? <div className="empty compact">No biological entities have been resolved for this session yet.</div> : <div className="biomedical-graph-list">{biologyNodes.map(node => <article key={node.id}><span className="graph-kind">{node.kind}</span><b>{node.label}</b><code>{node.id}</code><span>{graph.edges.filter(edge => edge.source === node.id || edge.target === node.id).length} evidence-linked relationships</span></article>)}</div>}
        </section>}
        {view === "hypotheses" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Candidate research hypotheses</h3><p>Generated hypotheses remain candidates until reviewed by a scientist.</p></div><span className="count">{hypothesisCandidates.length} candidates</span></div>
          {!selected ? <div className="empty compact">Choose a research session to inspect its hypothesis artifacts.</div> : hypothesisCandidates.length === 0 ? <div className="empty compact">No hypothesis candidates have been persisted for this session yet.</div> : <div className="hypothesis-list">{hypothesisCandidates.map(({ artifactId, candidate }) => {
            const id = String(candidate.id ?? "unknown");
            const dimensions = candidate.dimensions && typeof candidate.dimensions === "object" ? candidate.dimensions as Record<string, unknown> : {};
            const path = Array.isArray(candidate.path) ? candidate.path as Record<string, unknown>[] : [];
            const pathEvidence = Array.isArray(candidate.path_evidence) ? candidate.path_evidence.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
            const review = hypothesisReviews.get(id);
            const limitations = Array.isArray(candidate.limitations) ? candidate.limitations : [];
            const modelLeads = review?.model_generated_leads && typeof review.model_generated_leads === "object" ? review.model_generated_leads as Record<string, unknown> : {};
            const criticSections = [
              ["Refutation questions", modelLeads.refutation_questions],
              ["Alternative explanations", modelLeads.alternative_explanations],
              ["Methodological gaps", modelLeads.methodological_gaps],
            ] as const;
            const contradictionCandidates = review && Array.isArray(review.retrieved_contradiction_candidates) ? review.retrieved_contradiction_candidates.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
            const alternativeRoutes = review && Array.isArray(review.alternative_graph_routes) ? review.alternative_graph_routes.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : [];
            return <article key={`${artifactId}-${id}`}><div className="hypothesis-top"><span className="graph-kind">CANDIDATE · HUMAN REVIEW REQUIRED</span><code>{id}</code></div><h4>{String(candidate.statement ?? "Unlabeled hypothesis")}</h4><div className="hypothesis-path">{path.map((step, index) => <span key={index}>{String(step.source)} <i>{String(step.relation)}</i> {String(step.target)}</span>)}</div>{pathEvidence.length > 0 && <details><summary>Evidence annotations by path edge</summary><ul>{pathEvidence.map((edge, index) => { const counts = edge.polarity_counts && typeof edge.polarity_counts === "object" ? edge.polarity_counts as Record<string, unknown> : {}; const evidenceIds = Array.isArray(edge.evidence_ids) ? edge.evidence_ids : []; return <li key={`${String(edge.source)}-${index}`}><b>{String(edge.source)} {String(edge.relation)} {String(edge.target)}</b><span>{evidenceIds.map(String).join(", ")} · source labels: {String(counts.supports ?? 0)} supporting, {String(counts.contradicts ?? 0)} contradicting, {String(counts.context ?? 0)} context</span><small>{String(edge.interpretation ?? "Uncalibrated source labels.")}</small></li>; })}</ul></details>}<div className="hypothesis-metrics"><span>{String(dimensions.supporting_records ?? 0)} supporting records</span><span>{String(dimensions.contradictory_records ?? 0)} contradictory records</span><span>{String(dimensions.context_records ?? 0)} context records</span><span>Novelty: unscored</span></div><div className="hypothesis-review"><b>{review ? String(review.review_status ?? "criticism artifact") : "Scientific criticism pending"}</b>{review && <span>{Array.isArray(review.lexical_refutation_leads) ? review.lexical_refutation_leads.length : 0} lexical refutation leads · {Array.isArray(review.linked_contradictory_evidence) ? review.linked_contradictory_evidence.length : 0} linked contradictions</span>}</div>{alternativeRoutes.length > 0 && <details><summary>{alternativeRoutes.length} evidence-linked alternative graph routes</summary><ul>{alternativeRoutes.slice(0, 10).map((route, index) => { const counts = route.evidence_polarity_counts && typeof route.evidence_polarity_counts === "object" ? route.evidence_polarity_counts as Record<string, unknown> : {}; const routePath = Array.isArray(route.path) ? route.path as Record<string, unknown>[] : []; const evidenceIds = Array.isArray(route.evidence_ids) ? route.evidence_ids : []; return <li key={`${String(route.intermediate)}-${index}`}><div className="hypothesis-path">{routePath.map((step, stepIndex) => <span key={stepIndex}>{String(step.source)} <i>{String(step.relation)}</i> {String(step.target)}</span>)}</div><span>Evidence IDs: {evidenceIds.map(String).join(", ")} · source labels: {String(counts.supports ?? 0)} supporting, {String(counts.contradicts ?? 0)} contradicting, {String(counts.context ?? 0)} context</span><small>{String(route.interpretation ?? "Alternative route; not an independent refutation.")}</small></li>; })}</ul></details>}{criticSections.map(([title, values]) => Array.isArray(values) && values.length > 0 && <details key={title}><summary>{title} · model review leads</summary><ul>{values.slice(0, 6).map((value, index) => <li key={index}>{String(value)}</li>)}</ul><small>Unverified suggestions for scientific review.</small></details>)}{contradictionCandidates.length > 0 && <div className="hypothesis-review"><b>{contradictionCandidates.length} retrieved PubMed candidates · unassessed</b><ul>{contradictionCandidates.slice(0, 5).map((item, index) => <li key={`${String(item.candidate_artifact_id)}-${index}`}><span>{String(item.title || `PMID ${item.source_id}`)} · PMID {String(item.source_id)} · targeted claim {String(item.target_evidence_id)}</span></li>)}</ul><small>Retrieval targets a linked supporting claim; no candidate has been classified as contradictory.</small></div>}{review && <div className="hypothesis-critic-adjudication"><h5>Human review of scientific criticism</h5><p>Review leads require source and study-context assessment. These annotations do not change hypothesis confidence.</p>{(() => { const criticismId = String(review.criticism_artifact_id ?? ""); const evidenceLeads = ["linked_contradictory_evidence", "lexical_refutation_leads"].flatMap(field => Array.isArray(review[field]) ? review[field] as Record<string, unknown>[] : []); return <><ul>{evidenceLeads.map((lead, index) => { const evidenceId = String(lead.evidence_id ?? ""); return <li key={`${evidenceId}-${index}`}><b>{String(lead.claim ?? `Evidence ${evidenceId}`)}</b> · {evidenceId}{renderHypothesisCriticReview(criticismId, id, "evidence", evidenceId, `evidence ${evidenceId}`)}</li>; })}{contradictionCandidates.map((lead, index) => { const candidateId = String(lead.candidate_artifact_id ?? ""); return <li key={`${candidateId}-${index}`}>{String(lead.title || `PMID ${lead.source_id ?? "unknown"}`)} · targeted evidence {String(lead.target_evidence_id ?? "unknown")}{renderHypothesisCriticReview(criticismId, id, "retrieved_candidate", candidateId, `PMID ${String(lead.source_id ?? "unknown")}`)}</li>; })}</ul>{renderHypothesisCriticReview(criticismId, id, "hypothesis_review", id, "overall criticism")}</>; })()}</div>}{limitations.length > 0 && <ul>{limitations.slice(0, 3).map((item, index) => <li key={index}>{String(item)}</li>)}</ul>}</article>;
          })}</div>}
          <div className="section-heading citation-heading"><div><h3>Candidate literature searches</h3><p>Bounded PubMed searches join the three graph-path entities. Results are relevance-review leads; novelty remains unscored.</p></div><span className="count">{hypothesisLiteratureComparisons.length} search receipts</span></div>
          {!selected ? <div className="empty compact">Choose a research session to inspect hypothesis literature searches.</div> : hypothesisLiteratureComparisons.length === 0 ? <div className="empty compact">No PubMed comparison receipts have been retained for this session.</div> : <div className="hypothesis-list">{hypothesisLiteratureComparisons.map(({ artifact, data }) => { const searches = Array.isArray(data.candidate_searches) ? data.candidate_searches.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === "object" && !Array.isArray(item)) : []; return <article key={artifact.id}><div className="hypothesis-top"><span className="graph-kind">LITERATURE SEARCH · NOT A NOVELTY SCORE</span><code>{String(data.proposal_artifact_id ?? artifact.source_id)}</code></div><p>{String(data.searched_candidate_count ?? 0)} candidate searches · {String(data.omitted_candidate_count ?? 0)} omitted by bounds · novelty unscored</p>{searches.map((search, index) => { const leads = Array.isArray(search.candidate_leads) ? search.candidate_leads.filter((lead): lead is Record<string, unknown> => Boolean(lead) && typeof lead === "object" && !Array.isArray(lead)) : []; const coverage = search.coverage && typeof search.coverage === "object" ? search.coverage as Record<string, unknown> : {}; return <div className="hypothesis-review" key={`${String(search.hypothesis_id)}-${index}`}><b>{String(search.hypothesis_id)} · {String(search.returned_record_count ?? 0)} PMID records · {String(coverage.total_hits ?? "unknown")} provider hits</b><span>Query {String(search.query_sha256 ?? "").slice(0, 16)} · coverage receipt {String(search.coverage_artifact_id ?? "unavailable")}</span>{leads.length > 0 ? <ul>{leads.map(lead => { const paperArtifactId = String(lead.artifact_id ?? ""); const paper = hypothesisLiteraturePapers.find(item => item.artifact.id === paperArtifactId); const saved = hypothesisLiteratureReviews.filter(review => review.data.paper_artifact_id === paperArtifactId && review.data.comparison_artifact_id === artifact.id); const form = hypothesisLiteratureReviewForms[paperArtifactId] ?? {}; const reviewFields = [["rationale", "Relevance rationale"], ["study_design", "Study design"], ["population_or_model", "Population or model"], ["methods", "Methods"], ["findings", "Findings"], ["limitations", "Limitations"]] as const; return <li key={paperArtifactId}>{paper?.data.url ? <a href={String(paper.data.url)} target="_blank" rel="noreferrer">{String(paper.data.title || `PMID ${lead.pmid}`)}</a> : String(paper?.data.title || `PMID ${lead.pmid}`)} · PMID {String(lead.pmid)} · source hash {String(lead.record_sha256 ?? "").slice(0, 16)}<details><summary>{saved.length ? `Human relevance reviews (${saved.length})` : "Review relevance and study context"}</summary>{saved.map(item => <p key={item.artifact.id}>Reviewed by {String((item.data.reviewer as Record<string, unknown> | undefined)?.subject ?? "researcher")} · {String(item.data.relevance)} · {String(item.data.rationale)} · novelty remains unscored.</p>)}<label>Relevance<select aria-label={`Literature relevance for PMID ${String(lead.pmid)}`} value={form.relevance ?? "unclear"} onChange={event => setHypothesisLiteratureReviewForms(previous => ({ ...previous, [paperArtifactId]: { ...previous[paperArtifactId], relevance: event.target.value } }))}><option value="directly_relevant">Directly relevant</option><option value="partially_relevant">Partially relevant</option><option value="unrelated">Unrelated</option><option value="unclear">Unclear</option></select></label>{reviewFields.map(([field, label]) => <label className="review-field" key={field}>{label}<textarea maxLength={field === "study_design" ? 120 : field === "population_or_model" ? 500 : 2000} value={form[field] ?? ""} onChange={event => setHypothesisLiteratureReviewForms(previous => ({ ...previous, [paperArtifactId]: { ...previous[paperArtifactId], [field]: event.target.value } }))} /></label>)}<button onClick={() => void submitHypothesisLiteratureReview(artifact, paperArtifactId)} disabled={reviewBusy === paperArtifactId}>{reviewBusy === paperArtifactId ? "Saving…" : "Save researcher review"}</button></details></li>; })}</ul> : <small>No PMID candidates returned for this exact-entity query.</small>}<small>Search retrieval does not establish semantic relevance or novelty.</small></div>; })}{Array.isArray(data.warnings) && data.warnings.length > 0 && <p>{data.warnings.length} searches require review because retrieval was incomplete.</p>}</article>; })}</div>}
          <div className="section-heading citation-heading"><div><h3>Source-linked claim review</h3><p>Exact-entity directional leads and local-model screening labels. These are review candidates, not confirmed contradictions.</p></div><span className="count">{claimAlignmentCandidates.length} leads · {semanticClaimReviews.length} screened</span></div>
          {!selected ? <div className="empty compact">Choose a research session to inspect claim review records.</div> : claimAlignmentCandidates.length === 0 && semanticClaimReviews.length === 0 ? <div className="empty compact">No cross-paper claim review candidates are retained for this session.</div> : <div className="hypothesis-list">
            {claimAlignmentCandidates.map(({ artifactId, link }, index) => {
              const sources = Array.isArray(link.sources) ? link.sources as Record<string, unknown>[] : [];
              const context = link.study_context_comparison && typeof link.study_context_comparison === "object" ? link.study_context_comparison as Record<string, unknown> : {};
              const dimensions = context.paper_context_dimensions && typeof context.paper_context_dimensions === "object" ? context.paper_context_dimensions as Record<string, unknown> : {};
              const contextLabel = (key: string, label: string) => {
                const item = dimensions[key] && typeof dimensions[key] === "object" ? dimensions[key] as Record<string, unknown> : {};
                const labels = Array.isArray(item.source_labels) ? item.source_labels.map(String).join(" ↔ ") : "unassessed";
                return `${label}: ${labels} (${String(item.relation ?? "unassessed")})`;
              };
              const studySystems = context.study_system && typeof context.study_system === "object" ? context.study_system as Record<string, unknown> : {};
              return <article key={`${artifactId}-${index}`}><div className="hypothesis-top"><span className="graph-kind">UNREVIEWED DIRECTIONAL LEAD</span><code>{String(link.shared_entity_id ?? "entity unavailable")}</code></div><h4>{String(link.result_kind ?? "Result")} · opposite source-labeled directions</h4><div className="hypothesis-path">{sources.map((source, sourceIndex) => <span key={sourceIndex}>{String(source.document_id ?? "document")} · {String(source.candidate_id ?? "candidate")} · {String(source.direction ?? "unknown")} · {String(source.source_sha256 ?? "").slice(0, 12)}</span>)}</div><div className="hypothesis-metrics"><span>Study system: {Array.isArray(studySystems.source_labels) ? studySystems.source_labels.map(String).join(" ↔ ") : "unassessed"} · {String(studySystems.relation ?? "unassessed")}</span><span>{contextLabel("study_design", "Study design")}</span><span>{contextLabel("population", "Population")}</span><span>{contextLabel("method_family", "Method family")}</span></div><p>{String(context.interpretation_limit ?? "Model context labels are unvalidated review leads; human assessment is required.")}</p><p>{String(link.interpretation ?? "Candidate retrieval only; human review required.")}</p></article>;
            })}
            {semanticClaimReviews.map(({ artifact, data }) => {
              const assessment = data.assessment && typeof data.assessment === "object" ? data.assessment as Record<string, unknown> : {};
              const sources = Array.isArray(data.sources) ? data.sources as Record<string, unknown>[] : [];
              const saved = artifacts.filter(item => item.kind === "paper_claim_alignment_adjudication" && artifactData(item).screening_artifact_id === artifact.id);
              const form = claimAlignmentReviewForms[artifact.id] ?? {};
              const setField = (field: string, value: string) => setClaimAlignmentReviewForms(previous => ({ ...previous, [artifact.id]: { ...previous[artifact.id], [field]: value } }));
              return <article key={artifact.id}><div className="hypothesis-top"><span className="graph-kind">MODEL SCREEN · HUMAN REVIEW REQUIRED</span><code>{String(assessment.relation ?? "uncertain")}</code></div><h4>{String(assessment.population_comparability ?? "unclear")} populations · {String(assessment.method_comparability ?? "unclear")} methods</h4><div className="hypothesis-path">{sources.map((source, index) => <span key={index}>{String(source.pmc_id ?? "PMC")} · {String(source.candidate_id ?? "candidate")} · {String(source.source_sha256 ?? "").slice(0, 12)} · {String(source.candidate_text_sha256 ?? "").slice(0, 12)}</span>)}</div><p>{String(assessment.interpretation_limit ?? "Categorical model screening only; scientific review required.")}</p><small>{artifact.source} · {new Date(artifact.recorded_at * 1000).toLocaleString()}</small><details><summary>{saved.length ? `Researcher adjudications (${saved.length})` : "Adjudicate source claims and compare study context"}</summary>{saved.map(item => { const review = artifactData(item); return <p key={item.id}>Reviewed by {String((review.reviewer as Record<string, unknown> | undefined)?.subject ?? "researcher")} · {String(review.adjudication)} · populations {String(review.population_comparability)} · methods {String(review.method_comparability)} · contradiction confirmed: no. {String(review.rationale)}</p>; })}<label>Claim relation<select value={form.adjudication ?? "uncertain"} onChange={event => setField("adjudication", event.target.value)}><option value="same_question_consistent">Same question, consistent</option><option value="same_question_opposing">Same question, opposing</option><option value="related_but_distinct">Related, distinct questions</option><option value="unrelated">Unrelated</option><option value="uncertain">Uncertain</option></select></label><label>Population comparability<select value={form.population_comparability ?? "unclear"} onChange={event => setField("population_comparability", event.target.value)}><option value="comparable">Comparable</option><option value="different">Different</option><option value="unclear">Unclear</option><option value="not_reported">Not reported</option></select></label><label>Method comparability<select value={form.method_comparability ?? "unclear"} onChange={event => setField("method_comparability", event.target.value)}><option value="comparable">Comparable</option><option value="different">Different</option><option value="unclear">Unclear</option><option value="not_reported">Not reported</option></select></label><label className="review-field">Adjudication rationale<textarea maxLength={2000} value={form.rationale ?? ""} onChange={event => setField("rationale", event.target.value)} /></label><label className="review-field">Methods/population notes<textarea maxLength={4000} value={form.methods_population_notes ?? ""} onChange={event => setField("methods_population_notes", event.target.value)} /></label><button onClick={() => void submitClaimAlignmentReview(artifact)} disabled={reviewBusy === artifact.id}>{reviewBusy === artifact.id ? "Saving…" : "Save researcher adjudication"}</button><small>Opposing directions are not automatically treated as a confirmed contradiction.</small></details></article>;
            })}
          </div>}
        </section>}
        {view === "experiments" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Reproducibility bundle and execution record</h3><p>{selected ? `${selected.id} · hash-linked snapshot of persisted research inputs and outcomes` : "Select a research session to inspect its reproducibility record."}</p></div><span className="count">{reproducibilityArtifacts.length} persisted bundles</span></div>
          <div className="section-heading repro-heading"><div><h3>Generated research workflow proposals</h3><p>Versioned topology templates derived from session DAGs; execution requires a separate authenticated review.</p></div><span className="count">{workflowProposals.length} proposals</span></div>
          {workflowProposals.length === 0 ? <p className="empty compact">No versioned workflow proposal has been generated for this session.</p> : <div className="repro-artifacts">{workflowProposals.map(({ artifact, data }) => {
            const execution = data.execution && typeof data.execution === "object" ? data.execution as Record<string, unknown> : {};
            const sourceRun = data.source_run && typeof data.source_run === "object" ? data.source_run as Record<string, unknown> : {};
            const steps = Array.isArray(data.steps) ? data.steps : [];
            return <article key={artifact.id}><div><b>{String(data.workflow_id ?? "biomedical.research.plan")} · {String(data.version ?? "unknown version")}</b><span>{steps.length} steps · {String(sourceRun.completed_task_count ?? 0)}/{String(sourceRun.task_count ?? 0)} source run tasks succeeded · {String(data.lifecycle_status ?? "unknown status")}</span><small>{String(data.interpretation_limit ?? "Workflow proposal requires scientific and operational review.")}</small></div><code>Topology SHA-256 · {String(data.topology_sha256 ?? artifact.sha256)}</code><code>Artifact SHA-256 · {artifact.sha256}</code><strong>{execution.enabled === true ? "Execution enabled" : "Execution disabled pending authenticated review"}</strong></article>;
          })}</div>}
          {!selected ? <div className="empty compact">Choose a research session to load its reproducibility bundle and task outcomes.</div> : !latestBundle ? <div className="empty compact">{reproductionTask?.status === "running" || reproductionTask?.status === "ready" ? "The reproducibility task has not persisted its bundle yet." : reproductionTask?.status === "failed" || reproductionTask?.status === "needs_review" ? `Reproducibility is ${reproductionTask.status.replace("_", " ")}. Inspect the execution plan for its recorded reason.` : "No reproducibility bundle has been persisted for this session."}</div> : <>
            <div className="repro-summary"><article><small>Bundle integrity</small><b>SHA-256 recorded</b><code>{String(bundleData.bundle_sha256 ?? latestBundle.sha256)}</code></article><article><small>Recorded</small><b>{new Date(latestBundle.recorded_at * 1000).toLocaleString()}</b><span>{latestBundle.source}</span></article><article><small>Research scope</small><b>{bundleTasks.length} task outcomes</b><span>{bundleArtifacts.length} source receipts · {bundleEvidence.length} evidence annotations</span></article><article><small>Graph snapshot</small><b>{bundleNodes.length} entities · {bundleEdges.length} relationships</b><span>Evidence references checked during bundle creation</span></article></div>
            <div className="repro-hash"><span>Artifact content hash</span><code>{latestBundle.sha256}</code><span>Bundle hash</span><code>{String(bundleData.bundle_sha256 ?? "Unavailable")}</code></div>
            <div className="section-heading repro-heading"><div><h3>Recorded task outcomes</h3><p>Output and error contents are represented by hashes in the bundle.</p></div></div>
            <div className="task-list">{bundleTasks.map((item, index) => <article className="task-row" key={String(item.id ?? index)}><span className="task-index">{String(index + 1).padStart(2, "0")}</span><div className="task-content"><b>{String(item.id ?? "Unknown task")}</b><span>{String(item.output_sha256 ? `Output SHA-256: ${item.output_sha256}` : "No output hash recorded")}</span>{item.error_sha256 && <small>Error SHA-256: {String(item.error_sha256)}</small>}</div><span className={`task-status status-${String(item.status ?? "unknown")}`}>{String(item.status ?? "unknown").replace("_", " ")}</span></article>)}</div>
            <div className="section-heading repro-heading"><div><h3>Source artifact receipts</h3><p>Identifiers, source labels, timestamps and hashes retained; source article text and full datasets are excluded.</p></div></div>
            <div className="repro-artifacts">{bundleArtifacts.length === 0 ? <p className="empty compact">No source artifact receipts were included.</p> : bundleArtifacts.map((item, index) => <article key={String(item.id ?? index)}><div><b>{String(item.kind ?? "artifact")}</b><span>{String(item.source ?? "Unknown source")} · {String(item.source_id ?? "No source ID")}</span></div><code>{String(item.sha256 ?? "No hash")}</code></article>)}</div>
            {typeof bundleData.report_markdown === "string" && <><div className="section-heading repro-heading"><div><h3>Generated research report</h3><p>Stored with the reproducibility bundle.</p></div></div><pre className="repro-report">{bundleData.report_markdown}</pre></>}
            {Array.isArray(bundleData.limitations) && <div className="repro-limitations"><b>Interpretation limits</b><ul>{bundleData.limitations.map((item, index) => <li key={index}>{String(item)}</li>)}</ul></div>}
            <div className="repro-limitations"><b>Scientific review</b><p>The bundle verifies captured record integrity. It does not establish scientific correctness or provide a clinical recommendation.</p></div>
          </>}
        </section>}
        {view === "review" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Review pending research tasks</h3><p>{selected ? `${selected.id} · only needs-review tasks can receive a decision` : "Choose a research session, then review its pending scientific tasks."}</p></div><span className="count">{plan?.tasks.filter(task => task.status === "needs_review").length ?? 0} awaiting review</span></div>
          {!selected ? <div className="empty compact">Choose a research session from the list above.</div> : !plan ? <div className="empty compact">The session task plan is not loaded yet.</div> : plan.tasks.filter(task => task.status === "needs_review").length === 0 ? <div className="empty compact">No research task is waiting for human review in this session.</div> : <div className="hypothesis-list">{plan.tasks.filter(task => task.status === "needs_review").map((task, index) => <article key={task.id}><div className="hypothesis-top"><span className="graph-kind">SCIENTIFIC REVIEW REQUIRED</span><code>{task.id}</code></div><h4>{task.description}</h4>{task.error && <p>{task.error}</p>}{task.output && <details><summary>Inspect submitted task output</summary><pre className="review-output">{task.output}</pre></details>}<label className="review-credential">Decision rationale<textarea aria-label={`Review rationale for ${task.id}`} maxLength={2000} value={reviewReasons[task.id] ?? ""} onChange={event => setReviewReasons(previous => ({ ...previous, [task.id]: event.target.value }))} placeholder="Describe the evidence or validation behind this decision."/></label><div className="review-actions"><button disabled={reviewBusy === task.id} onClick={() => void reviewResearchTask(task, "approve")}>{reviewBusy === task.id ? "Saving…" : "Approve task"}</button><button className="review-reject" disabled={reviewBusy === task.id} onClick={() => void reviewResearchTask(task, "reject")}>Reject task</button></div></article>)}</div>}
          <div className="section-heading citation-heading"><div><h3>Ontology cross-mapping candidates</h3><p>Direct OxO mappings are review leads. Each decision is attached to its exact source response and candidate; no ontology equivalence is applied automatically.</p></div><span className="count">{artifacts.filter(item => item.kind === "ontology_cross_mapping_review").length} source receipts</span></div>
          {artifacts.filter(item => item.kind === "ontology_cross_mapping_review").length === 0 ? <p className="empty compact">No OxO cross-mapping candidates are recorded for this session.</p> : <div className="hypothesis-list">{artifacts.filter(item => item.kind === "ontology_cross_mapping_review").map(source => {
            const data = artifactData(source);
            const sourceIdentifier = typeof data.source_identifier === "string" ? data.source_identifier : "";
            const candidates = Array.isArray(data.mappings) ? data.mappings.filter((row): row is Record<string, unknown> => Boolean(row) && typeof row === "object" && !Array.isArray(row)) : [];
            return <article key={source.id}><div className="hypothesis-top"><span className="graph-kind">DIRECT OXO MAPPING · DISTANCE 1</span><code>{sourceIdentifier || "Unknown source identifier"}</code></div><p>{String(data.provider_mapping_count ?? candidates.length)} provider rows · {candidates.length} unique candidates · response SHA-256 {String(data.source_response_sha256 ?? "unavailable")}</p>
              {candidates.length === 0 ? <p>No direct candidate was retained. {data.truncated === true ? "The provider response was truncated, so this does not establish that no mapping exists." : ""}</p> : candidates.map(candidate => {
                const target = typeof candidate.target_curie === "string" ? candidate.target_curie : "Unknown CURIE";
                const key = `${source.id}:${target}`;
                const saved = artifacts.filter(item => item.kind === "ontology_cross_mapping_adjudication" && artifactData(item).source_artifact_id === source.id && artifactData(item).target_curie === target);
                const form = ontologyMappingReviewForms[key] ?? {};
                const setField = (field: string, value: string) => setOntologyMappingReviewForms(previous => ({ ...previous, [key]: { ...(previous[key] ?? {}), [field]: value } }));
                return <div className="capability-review-card" key={key}><div><b>{target}</b><span>{String(candidate.target_label ?? "No target label supplied by OxO")} · distance {String(candidate.distance ?? "unknown")}</span><small>Target prefix {String(candidate.target_prefix ?? "unknown")} · {String(candidate.source_prefix ?? "unknown")} source ontology</small></div>
                  {saved.length > 0 ? saved.map(item => { const review = artifactData(item); const reviewer = review.reviewer && typeof review.reviewer === "object" ? review.reviewer as Record<string, unknown> : {}; return <p key={item.id}>Reviewed by {String(reviewer.subject ?? "researcher")} · {String(review.decision)} · {String(review.rationale)} · artifact SHA-256 {item.sha256}</p>; }) : <><label>Researcher decision<select value={form.decision ?? "uncertain"} onChange={event => setField("decision", event.target.value)}><option value="same_concept">Same concept</option><option value="related_but_distinct">Related but distinct</option><option value="incorrect_candidate">Incorrect candidate</option><option value="uncertain">Uncertain</option></select></label><label className="review-field">Rationale<textarea maxLength={2000} value={form.rationale ?? ""} onChange={event => setField("rationale", event.target.value)} /></label><button disabled={reviewBusy === key} onClick={() => void submitOntologyMappingReview(source, sourceIdentifier, candidate)}>{reviewBusy === key ? "Saving…" : "Save ontology adjudication"}</button></>}
                </div>;
              })}<small>OxO provides a direct cross-reference candidate, not proof of disease equivalence. Decisions are audit records and do not merge graph nodes.</small></article>;
          })}</div>}
          <div className="section-heading citation-heading"><div><h3>Capability review and lifecycle</h3><p>Verification reports must pass static analysis, contract, scientific, and reproducibility checks. Production also requires a benchmark check.</p></div><span className="count">{capabilities.filter(item => item.status === "pending_review").length} pending capabilities</span></div>
          {(() => {
            const benchmarks = artifacts.filter(item => item.kind === "capability_version_benchmark" || item.kind === "capability_version_benchmark_failure");
            return benchmarks.length === 0 ? null : <div className="coverage-list"><div className="section-heading"><div><h3>Paired capability benchmark receipts</h3><p>Session-scoped execution records for human lifecycle review; each comparison applies only to its recorded corpus.</p></div><span className="count">{benchmarks.length} receipts</span></div>{benchmarks.map(artifact => {
              const report = artifactData(artifact);
              const baseline = report.baseline && typeof report.baseline === "object" ? report.baseline as Record<string, unknown> : {};
              const candidate = report.candidate && typeof report.candidate === "object" ? report.candidate as Record<string, unknown> : {};
              const paired = report.paired && typeof report.paired === "object" ? report.paired as Record<string, unknown> : {};
              const outcomeTest = report.paired_outcome_exact_sign_test && typeof report.paired_outcome_exact_sign_test === "object" ? report.paired_outcome_exact_sign_test as Record<string, unknown> : {};
              const displayRate = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "unavailable";
              const pValue = outcomeTest.two_sided_p_value;
              return <article key={artifact.id}><div className="literature-record-meta"><span className="graph-kind">{String(report.status ?? "unknown").replaceAll("_", " ")}</span><code>{String(report.capability_id ?? "capability unavailable")}</code><span>{String(report.benchmark_id ?? "benchmark unavailable")}</span><span>n={String(report.n ?? "unavailable")}</span></div><p>Baseline {String(report.baseline_proposal_artifact_id ?? report.baseline_version ?? "unknown")}: {displayRate(baseline.pass_rate)} passed · median {typeof baseline.median_latency_ms === "number" ? `${baseline.median_latency_ms.toFixed(2)} ms` : "latency unavailable"}</p><p>Candidate {String(report.candidate_proposal_artifact_id ?? report.candidate_version ?? "unknown")}: {displayRate(candidate.pass_rate)} passed · median {typeof candidate.median_latency_ms === "number" ? `${candidate.median_latency_ms.toFixed(2)} ms` : "latency unavailable"}</p><p>Paired outcomes: {String(paired.candidate_only_passes ?? 0)} candidate-only passes · {String(paired.baseline_only_passes ?? 0)} baseline-only passes · exact two-sided sign-test p={typeof pValue === "number" && Number.isFinite(pValue) ? pValue.toPrecision(3) : "not estimable"}.</p><b>{String(report.recommendation ?? "No recommendation available")}</b><small>Attach as benchmark_receipt: session {artifact.session_id}, artifact {artifact.id}, SHA-256 {artifact.sha256}.</small><small>Benchmark corpus SHA-256 {String(report.benchmark_sha256 ?? "unavailable")} · {String(report.interpretation_limit ?? "Review evidence is limited to this benchmark and cannot independently justify scientific validity or a lifecycle transition.")}</small></article>;
            })}</div>;
          })()}
          {capabilities.length === 0 ? <div className="empty compact">No capability versions are registered in the MedAI registry.</div> : <div className="capability-list">{capabilities.map(item => {
            const key = `${item.id}@${item.version}`;
            const allowed: ("verified" | "production" | "deprecated")[] = item.status === "pending_review" ? ["verified", "deprecated"] : item.status === "verified" ? ["production", "deprecated"] : item.status === "production" ? ["deprecated"] : [];
            const rollbackVersions = capabilities.filter(candidate => candidate.id === item.id && candidate.status === "verified");
            const rollbackTarget = rollbackTargets[key] ?? rollbackVersions[0]?.version ?? "";
            return <article className="capability-review-card" key={key}>
              <div><b>{item.id}</b><span>{item.description}</span><code>{item.version} · {item.kind} · {item.status.replaceAll("_", " ")}</code></div>
              <label className="review-credential">Lifecycle rationale<textarea aria-label={`Lifecycle rationale for ${key}`} maxLength={2000} value={reviewReasons[key] ?? ""} onChange={event => setReviewReasons(previous => ({ ...previous, [key]: event.target.value }))} placeholder="Record why this lifecycle transition is justified."/></label>
              {(allowed.some(next => next !== "deprecated") || (item.status === "production" && rollbackVersions.length > 0)) && <><label className="review-credential">Validation report JSON<textarea aria-label={`Validation report for ${key}`} value={validationReports[key] ?? ""} onChange={event => setValidationReports(previous => ({ ...previous, [key]: event.target.value }))} placeholder={'{"capability_id":"...","version":"...","manifest_sha256":"...","checks":{"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true},"benchmark_receipt":{"session_id":"...","artifact_id":"...","artifact_sha256":"..."}}'}/></label>{item.status === "verified" && <small>Production approval for a generated AgentMesh capability also checks that the receipt points to its exact signed source proposal and a completed benchmark artifact from the same research session.</small>}</>}
              <div className="review-actions">{allowed.map(next => <button key={next} className={next === "deprecated" ? "review-reject" : ""} disabled={reviewBusy === key} onClick={() => void transitionCapability(item, next)}>{reviewBusy === key ? "Saving…" : `Set ${next.replaceAll("_", " ")}`}</button>)}
                {item.status === "production" && rollbackVersions.length > 0 && <><label className="review-credential">Verified rollback target<select aria-label={`Rollback target for ${key}`} value={rollbackTarget} onChange={event => setRollbackTargets(previous => ({ ...previous, [key]: event.target.value }))}>{rollbackVersions.map(candidate => <option key={candidate.version} value={candidate.version}>{candidate.version}</option>)}</select></label><button className="review-reject" disabled={reviewBusy === key} onClick={() => void rollbackCapability(item)}>{reviewBusy === key ? "Saving…" : "Roll back production"}</button></>}
                <button className="review-secondary" onClick={() => void loadCapabilityEvents(item)}>Audit history</button></div>
              {capabilityEvents[key] && <ol className="capability-events">{capabilityEvents[key].map((entry, index) => <li key={`${entry.occurred_at}-${index}`}><b>{entry.from ?? "created"} → {entry.to}</b><span>{entry.actor} · {new Date(entry.occurred_at * 1000).toLocaleString()}</span><small>{entry.reason}{entry.validation_sha256 ? ` · validation ${entry.validation_sha256}` : ""}</small></li>)}</ol>}
            </article>;
          })}</div>}
        </section>}
        {view === "capabilities" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Governed capability versions</h3><p>Registry status is administrative lifecycle metadata, not a scientific quality score.</p></div><span className="count">{capabilities.length} versions</span></div>
          {capabilities.length === 0 ? <div className="empty compact">No capability versions are registered in the local MedAI registry.</div> : <div className="capability-list">{capabilities.map(item => <article key={`${item.id}-${item.version}`}><div><b>{item.id}</b><span>{item.description}</span></div><code>{item.version}</code><span className={`task-status status-${item.status}`}>{item.status.replaceAll("_", " ")}</span><small>{item.kind}</small></article>)}</div>}
        </section>}
        {view === "graph" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Session biomedical graph</h3><p>{selected ? `${selected.id} · source-linked relationships` : "Select a research session to inspect its biomedical graph."}</p></div><span className="count">{graph.nodes.length} entities · {graph.edges.length} relationships</span></div>
          {!selected ? <div className="empty compact">Choose a research session to load its biomedical graph.</div> : graph.nodes.length === 0 ? <div className="empty compact">No biomedical entities are indexed for this session yet.</div> : <><KnowledgeGraph graph={graph} /><details className="graph-records"><summary>Browse all entities and source-linked relationships</summary><div className="biomedical-graph-list">{graph.nodes.map(node => <article key={node.id}><span className="graph-kind">{node.kind}</span><b>{node.label}</b><code>{node.id}</code><span>{graph.edges.filter(edge => edge.source === node.id || edge.target === node.id).length} provenance-linked relationships</span></article>)}</div><div className="graph-edge-list">{graph.edges.length === 0 ? <p>No source-linked relationships have been recorded.</p> : graph.edges.map((edge, index) => <article key={`${edge.source}-${edge.relation}-${edge.target}-${index}`}><b>{edge.source}</b><span>{edge.relation}</span><b>{edge.target}</b><small>Annotation {edge.confidence.toFixed(2)} · Evidence: {edge.evidence_ids || "none"} · Source artifacts: {edge.artifact_ids || "none"}</small></article>)}</div></details></>}
        </section>}
        {selected && view === "overview" && <section className="plan-panel">
          <div className="section-heading"><div><h3>Investigation plan</h3><p>{selected.id} · {selected.objective}</p></div>{plan && <span className="count">{plan.tasks.filter(task => task.status === "succeeded").length}/{plan.tasks.length} tasks succeeded · {selected.depth} · {selected.credits_used}/{selected.budget} credits</span>}</div>
          {budgetSummary && <div className="budget-resource-summary"><article><small>Execution credits</small><b>{budgetSummary.used} / {budgetSummary.limit}</b><span>{budgetSummary.remaining} remaining</span></article><article><small>Session API requests</small><b>{meteredApiEvents.length} requests · {meteredApiCredits} credits</b><span>Authenticated requests linked to this session</span></article><article><small>Model calls</small><b>{modelReservations.length} · {pendingModelReservations} pending</b><span>{uncertainModelReservations} uncertain; their full reservation remains charged</span></article><article><small>Stored artifact payload</small><b>{budgetSummary.storage.artifact_payload_bytes.toLocaleString()} bytes</b><span>{budgetSummary.storage.credits_charged} credits · {budgetSummary.storage.bytes_per_credit.toLocaleString()} bytes per credit</span></article></div>}
          {modelReservations.length > 0 && <details className="model-reservation-ledger"><summary>Model reservation history ({modelReservations.length})</summary><div>{modelReservations.slice(-10).reverse().map(reservation => <article key={reservation.id}><b>{reservation.task_id} · {reservation.state} · {reservation.actor}</b><span>{reservation.charged_credits ?? reservation.reserved_credits} charged / {reservation.reserved_credits} reserved credits</span><code>{reservation.reason}</code></article>)}</div></details>}
          <section className="study-design-inventory" aria-label="Evidence and study metadata inventory">
            <div className="section-heading"><div><h3>Evidence and study metadata</h3><p>Provider publication labels are descriptive signals; this inventory assigns no evidence grade.</p></div><span className="count">{evidenceInventoryArtifact ? String(studyDesignInventory.methodology ?? "inventory") : "Pending"}</span></div>
            {!evidenceInventoryArtifact ? <p className="empty compact">The evidence inventory has not been recorded for this session.</p> : <>
              <div className="study-design-counts">{designCategories.length === 0 ? <p>No recognized PubMed publication-type signals were retained.</p> : designCategories.map(([category, count]) => {
                const pmids = Array.isArray(designExamples[category]) ? designExamples[category].filter((value): value is string => typeof value === "string" && /^\d+$/.test(value)) : [];
                return <article key={category}><b>{category.replaceAll("_", " ")}</b><strong>{count} records</strong>{pmids.length > 0 && <span>Examples: {pmids.slice(0, 5).map((pmid, index) => <span key={pmid}>{index > 0 ? ", " : ""}<a href={`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/`} target="_blank" rel="noreferrer">{pmid}</a></span>)}</span>}{truncatedDesignExamples[category] === true && <small>Example PMID list truncated; category count is complete.</small>}</article>;
              })}</div>
              {relatedPublicationCategories.length > 0 && <div className="study-design-counts" role="group" aria-label="PubMed related-publication links"><h4>PubMed related-publication links</h4>{relatedPublicationCategories.map(([relationship, count]) => {
                const pmids = Array.isArray(relatedPublicationExamples[relationship]) ? relatedPublicationExamples[relationship].filter((value): value is string => typeof value === "string" && /^[0-9]{1,20}$/.test(value)) : [];
                return <article key={relationship}><b>{relationship}</b><strong>{count} links</strong>{pmids.length > 0 && <span>Linked PMIDs: {pmids.slice(0, 5).map((pmid, index) => <span key={`${pmid}-${index}`}>{index > 0 ? ", " : ""}<a href={`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/`} target="_blank" rel="noreferrer">{pmid}</a></span>)}</span>}</article>;
              })}</div>}
              <p className="study-design-limit">Quality dimensions: {Object.entries(qualityDomains).map(([domain, status]) => `${domain.replaceAll("_", " ")}: ${String(status)}`).join(" · ") || "not assessed"}.</p>
              <p className="study-design-limit">{String(studyDesignInventory.interpretation ?? "Study metadata is not an independent assessment of methods or quality.")} {String(studyDesignInventory.records_without_recognized_design_signal ?? 0)} records had no recognized design label; {String(studyDesignInventory.records_with_malformed_design_metadata ?? 0)} had malformed design metadata; {String(studyDesignInventory.records_with_malformed_related_publication_metadata ?? 0)} had malformed related-publication metadata.</p>
              {typeof evidenceInventoryData.review_reason === "string" && <p className="error">Review required: {evidenceInventoryData.review_reason}</p>}
            </>}
          </section>
          {loadingPlan ? <div className="empty compact">Loading persisted task plan…</div> : plan && <div className="task-list">{plan.tasks.map((task, index) => <article className="task-row" key={task.id}><span className="task-index">{String(index + 1).padStart(2, "0")}</span><div className="task-content"><b>{task.id}</b><span>{task.description}</span><small>Attempts: {task.attempts}</small>{task.depends_on.length > 0 && <small>Depends on: {task.depends_on.join(", ")}</small>}{task.status === "ready" && task.available_at && task.available_at > Date.now() / 1000 && <small>Retry available: {new Date(task.available_at * 1000).toLocaleTimeString()}</small>}{task.lease_owner && <small>Worker: {task.lease_owner}{task.lease_until ? " · lease expires " + new Date(task.lease_until * 1000).toLocaleTimeString() : ""}</small>}{task.error && <small className="error">{task.error}</small>}</div><span className={"task-status status-" + task.status}>{task.status.replace("_", " ")}</span></article>)}</div>}
          {plan?.events.length ? <div className="event-note">{plan.events.length} audited task transitions · latest {new Date(plan.events[plan.events.length - 1].occurred_at * 1000).toLocaleString()}</div> : plan && <div className="event-note">No task has started. Planned work is not reported as completed research.</div>}
        </section>}
        <footer>MedAI builds computational evidence summaries. It does not provide clinical recommendations. <span>AgentMesh routes governed MCP tools</span></footer>
      </section>
    </main>
  );
}
