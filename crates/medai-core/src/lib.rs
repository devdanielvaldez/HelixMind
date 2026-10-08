//! Biomedical research domain and durable local store.
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const ARTIFACT_STORAGE_BYTES_PER_CREDIT: i64 = 1024 * 1024;
pub const DEFAULT_TENANT_MAX_ACTIVE_SESSIONS: i64 = 25;
pub const DEFAULT_TENANT_MAX_SESSION_BUDGET: i64 = 100_000;
pub const DEFAULT_TENANT_MAX_ALLOCATED_CREDITS: i64 = 1_000_000;
const MAX_AUDIT_ARCHIVE_EVENTS: i64 = 20_000;
const SQLITE_SCHEMA_VERSION: i64 = 2;

fn api_audit_digest(event: &ApiAuditEvent) -> Result<String> {
    let mut value = serde_json::json!({
        "sequence": event.sequence,
        "request_id": event.request_id,
        "actor": event.actor,
        "method": event.method,
        "path": event.path,
        "status": event.status,
        "duration_ms": event.duration_ms,
        "occurred_at": event.occurred_at,
        "previous_sha256": event.previous_sha256,
    });
    if let Some(tenant_id) = event.tenant_id.as_ref() {
        value["tenant_id"] = serde_json::Value::String(tenant_id.clone());
    }
    let canonical = serde_json::to_vec(&value)?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseBackupReceipt {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub integrity_check: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseRestoreReceipt {
    pub restored_from: String,
    pub source_sha256: String,
    pub captured_snapshot_sha256: String,
    pub rollback_backup: String,
    pub integrity_check: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchSession {
    pub id: String,
    pub tenant_id: String,
    pub objective: String,
    pub status: String,
    pub created_at: i64,
    pub depth: String,
    pub budget: i64,
    pub credits_used: i64,
    pub cancellation_requested: bool,
}

struct ArtifactFields<'a> {
    kind: &'a str,
    source: &'a str,
    source_id: &'a str,
    payload: &'a str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantQuota {
    pub tenant_id: String,
    pub max_active_sessions: i64,
    pub max_session_budget: i64,
    pub max_allocated_credits: i64,
    pub updated_at: Option<i64>,
    pub updated_by: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantDirectoryEntry {
    pub tenant_id: String,
    pub status: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub updated_by: String,
    pub disabled_reason: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchTask {
    pub id: String,
    pub session_id: String,
    pub domain: String,
    pub description: String,
    pub depends_on: Vec<String>,
    pub status: String,
    pub attempts: i64,
    pub output: Option<String>,
    pub error: Option<String>,
    pub lease_owner: Option<String>,
    pub lease_until: Option<i64>,
    pub checkpoint: Option<String>,
    pub available_at: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskEvent {
    pub id: i64,
    pub session_id: String,
    pub task_id: String,
    pub from_status: Option<String>,
    pub to_status: String,
    pub occurred_at: i64,
    pub details: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAuditEvent {
    pub sequence: i64,
    pub request_id: String,
    pub actor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub duration_ms: i64,
    pub occurred_at: i64,
    pub previous_sha256: String,
    pub event_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAuditArchiveReceipt {
    pub first_sequence: i64,
    pub last_sequence: i64,
    pub archived_events: usize,
    pub archive_file: String,
    pub archive_sha256: String,
}
#[derive(Debug, Clone)]
pub struct ApiAuditActor {
    pub actor: String,
    pub tenant_id: Option<String>,
}
/// Public metadata for a bearer credential. The bearer itself is never persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiCredentialRecord {
    pub id: String,
    pub token_sha256: String,
    pub subject: String,
    pub tenant_id: String,
    pub role: String,
    pub created_at: i64,
    pub revoked_at: Option<i64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetEvent {
    pub id: i64,
    pub session_id: String,
    pub task_id: String,
    pub credits: i64,
    pub reason: String,
    pub occurred_at: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchArtifact {
    pub id: String,
    pub session_id: String,
    pub task_id: String,
    pub kind: String,
    pub source: String,
    pub source_id: String,
    pub payload: String,
    pub sha256: String,
    pub recorded_at: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRedactionCandidate {
    pub artifact_id: String,
    pub source_id: String,
    pub kind: String,
    pub original_sha256: String,
    pub redacted_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRedactionPlan {
    pub session_id: String,
    pub candidate_count: usize,
    pub candidates: Vec<ArtifactRedactionCandidate>,
    pub plan_sha256: String,
}

fn redact_pubmed_abstracts(value: &mut serde_json::Value, in_abstract_section: bool) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            let mut changed = false;
            let labeled_as_abstract = ["section", "label", "source_label", "nlm_category"]
                .iter()
                .filter_map(|key| object.get(*key).and_then(serde_json::Value::as_str))
                .any(|label| label.to_ascii_lowercase().contains("abstract"));
            let keys: Vec<String> = object.keys().cloned().collect();
            for key in keys {
                let normalized = key.to_ascii_lowercase();
                let abstract_content_field = normalized.contains("abstract")
                    && !["retained", "digest", "sha256", "length", "label"]
                        .iter()
                        .any(|metadata| normalized.contains(metadata));
                if abstract_content_field
                    || (in_abstract_section || labeled_as_abstract) && normalized == "text"
                {
                    object.remove(&key);
                    changed = true;
                } else if let Some(child) = object.get_mut(&key) {
                    changed |= redact_pubmed_abstracts(child, in_abstract_section);
                }
            }
            if changed {
                object.insert(
                    "abstract_text_retained".into(),
                    serde_json::Value::Bool(false),
                );
            }
            changed
        }
        serde_json::Value::Array(items) => items.iter_mut().fold(false, |changed, item| {
            redact_pubmed_abstracts(item, in_abstract_section) | changed
        }),
        _ => false,
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialWatch {
    pub session_id: String,
    pub nct_id: String,
    pub interval_seconds: i64,
    pub next_due: i64,
    pub enabled: bool,
    pub lease_owner: Option<String>,
    pub lease_until: Option<i64>,
    pub last_checked_at: Option<i64>,
    pub last_error: Option<String>,
    pub last_snapshot: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialWatchEvent {
    pub id: String,
    pub session_id: String,
    pub nct_id: String,
    pub captured_at: i64,
    pub snapshot_sha256: String,
    pub changed: Option<bool>,
    pub snapshot: String,
    pub comparison: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureWatch {
    pub session_id: String,
    pub watch_id: String,
    pub query: String,
    pub interval_seconds: i64,
    pub next_due: i64,
    pub enabled: bool,
    pub lease_owner: Option<String>,
    pub lease_until: Option<i64>,
    pub last_checked_at: Option<i64>,
    pub last_error: Option<String>,
    pub last_pmids: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureWatchEvent {
    pub id: String,
    pub session_id: String,
    pub watch_id: String,
    pub captured_at: i64,
    pub snapshot_sha256: String,
    pub result_count: i64,
    pub retrieved_count: i64,
    pub incomplete: bool,
    pub new_pmids: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureWatchPaper {
    #[serde(default)]
    pub event_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub watch_id: String,
    pub pmid: String,
    pub title: String,
    pub publication_date: Option<String>,
    pub journal: Option<String>,
    pub doi: Option<String>,
    pub pmcid: Option<String>,
    #[serde(default)]
    pub captured_at: i64,
    #[serde(default)]
    pub metadata_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationLink {
    pub session_id: String,
    pub citing_doi: String,
    pub cited_doi: String,
    pub source_artifact_id: String,
    pub reference_index: Option<i64>,
    pub indexed_at: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetric {
    pub doi: String,
    pub citations_received: usize,
    pub references_tracked: usize,
    pub pagerank: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: String,
    pub session_id: String,
    pub claim: String,
    pub source: String,
    pub source_id: Option<String>,
    pub polarity: String,
    pub quality: f64,
    pub provenance: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub session_id: String,
    pub id: String,
    pub kind: String,
    pub label: String,
    pub metadata: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub session_id: String,
    pub source: String,
    pub relation: String,
    pub target: String,
    pub confidence: f64,
    pub evidence_ids: String,
    #[serde(default = "empty_json_array")]
    pub artifact_ids: String,
}
fn empty_json_array() -> String {
    "[]".to_owned()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub version: String,
    pub kind: String,
    pub status: String,
    pub description: String,
    pub manifest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMeshRouteApproval {
    pub receipt_id: String,
    pub actor: String,
    pub medai_tenant_id: String,
    pub agentmesh_tenant: String,
    pub agentmesh_namespace: String,
    pub plan_review_sha256: String,
    pub desired_sha256: String,
    pub issued_at: i64,
    pub expires_at: i64,
}

const RESEARCH_TASKS: &[(&str, &str, &str, &str)] = &[
    (
        "objective.normalize",
        "planning",
        "Normalize the objective, scope, and requested depth",
        "[]",
    ),
    (
        "literature.discover",
        "literature",
        "Find relevant literature and retain source identifiers",
        "[\"objective.normalize\"]",
    ),
    (
        "trials.discover",
        "clinical_trials",
        "Search public trial registries and retain record identifiers",
        "[\"objective.normalize\"]",
    ),
    (
        "literature.extract",
        "literature",
        "Separate paper methods, results, limitations, and claims",
        "[\"literature.discover\"]",
    ),
    (
        "entities.resolve",
        "knowledge_graph",
        "Resolve biomedical entities to stable database identifiers",
        "[\"literature.extract\"]",
    ),
    (
        "genes.curate",
        "gene_disease",
        "Retrieve independent curated gene-disease validity assertions for exactly resolved human genes",
        "[\"entities.resolve\"]",
    ),
    (
        "evidence.grade",
        "evidence",
        "Summarize recorded evidence annotations and PubMed publication metadata without inferring evidence quality",
        "[\"literature.extract\"]",
    ),
    (
        "genomics.analyze",
        "genomics",
        "Analyze variants when a permitted genomic dataset is supplied",
        "[\"entities.resolve\"]",
    ),
    (
        "proteins.analyze",
        "proteomics",
        "Map protein functions, structures, and interactions",
        "[\"entities.resolve\"]",
    ),
    (
        "pathways.map",
        "pathways",
        "Map genes and proteins to sourced biological pathways",
        "[\"entities.resolve\"]",
    ),
    (
        "drugs.investigate",
        "pharmacology",
        "Connect compounds and targets to sourced evidence",
        "[\"entities.resolve\",\"trials.discover\"]",
    ),
    (
        "contradictions.search",
        "evidence",
        "Search PubMed for source-linked contradictory-evidence candidates for review",
        "[\"evidence.grade\",\"entities.resolve\"]",
    ),
    (
        "graph.build",
        "knowledge_graph",
        "Snapshot registered stable entities and evidence-linked relationships",
        "[\"evidence.grade\",\"entities.resolve\",\"genomics.analyze\",\"proteins.analyze\",\"pathways.map\",\"genes.curate\",\"drugs.investigate\"]",
    ),
    (
        "hypotheses.propose",
        "hypothesis",
        "Draft candidate hypotheses from supported graph paths",
        "[\"graph.build\",\"contradictions.search\"]",
    ),
    (
        "hypotheses.criticize",
        "scientific_criticism",
        "Search for refutations and alternative explanations",
        "[\"hypotheses.propose\"]",
    ),
    (
        "skills.learn",
        "self_extension",
        "Evaluate completed research histories and draft a disabled reusable Skill when a common procedure is supported",
        "[\"hypotheses.criticize\"]",
    ),
    (
        "report.reproduce",
        "reproducibility",
        "Create a hash-linked provenance report and reproducibility bundle",
        "[\"hypotheses.criticize\",\"skills.learn\",\"trials.discover\"]",
    ),
];

fn validate_worker_id(worker_id: &str) -> Result<()> {
    anyhow::ensure!(
        (1..=128).contains(&worker_id.len())
            && worker_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)),
        "worker_id must be 1..128 ASCII letters, digits, dots, underscores, colons, or hyphens"
    );
    Ok(())
}

fn validate_task_id(task_id: &str) -> Result<()> {
    anyhow::ensure!(
        (1..=128).contains(&task_id.len())
            && task_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)),
        "task IDs must be 1..128 ASCII letters, digits, dots, underscores, colons, or hyphens"
    );
    Ok(())
}

pub fn scientific_agent_for_task(task_id: &str) -> Option<&'static str> {
    #[cfg(test)]
    if task_id.starts_with("benchmark.load-") {
        return Some("research_director");
    }
    Some(match task_id {
        "objective.normalize" => "research_director",
        "literature.discover" => "literature_librarian",
        "literature.extract" => "paper_curator",
        "entities.resolve" => "entity_curator",
        "genes.curate" => "gene_disease_curator",
        "evidence.grade" => "evidence_synthesizer",
        "genomics.analyze" => "genomics_analyst",
        "proteins.analyze" => "protein_analyst",
        "pathways.map" => "pathway_analyst",
        "trials.discover" => "trial_librarian",
        "drugs.investigate" => "pharmacology_analyst",
        "contradictions.search" | "hypotheses.criticize" => "scientific_critic",
        "graph.build" => "knowledge_graph_curator",
        "hypotheses.propose" => "hypothesis_generator",
        "skills.learn" => "capability_learner",
        "report.reproduce" => "reproducibility_agent",
        _ => return None,
    })
}

pub fn is_scientific_agent(agent_id: &str) -> bool {
    [
        "research_director",
        "literature_librarian",
        "paper_curator",
        "entity_curator",
        "gene_disease_curator",
        "evidence_synthesizer",
        "genomics_analyst",
        "protein_analyst",
        "pathway_analyst",
        "trial_librarian",
        "pharmacology_analyst",
        "scientific_critic",
        "knowledge_graph_curator",
        "hypothesis_generator",
        "capability_learner",
        "reproducibility_agent",
    ]
    .contains(&agent_id)
}

fn specialist_budget_charge(
    tx: &Transaction<'_>,
    session_id: &str,
    agent_id: &str,
    credits: i64,
    now: i64,
) -> Result<bool> {
    let policy = tx
        .query_row(
            "SELECT credit_limit FROM specialist_credit_policies WHERE agent_id=?1",
            [agent_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    let Some(credit_limit) = policy else {
        return Ok(true);
    };
    tx.execute(
        "INSERT OR IGNORE INTO specialist_credit_usage(session_id,agent_id,credits_used,credit_limit,updated_at) VALUES(?1,?2,0,?3,?4)",
        params![session_id, agent_id, credit_limit, now],
    )?;
    let changed = tx.execute(
        "UPDATE specialist_credit_usage SET credits_used=credits_used+?3,updated_at=?4 WHERE session_id=?1 AND agent_id=?2 AND credits_used+?3<=credit_limit",
        params![session_id, agent_id, credits, now],
    )?;
    Ok(changed == 1)
}

fn now() -> Result<i64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64)
}

fn trial_watch_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TrialWatch> {
    Ok(TrialWatch {
        session_id: row.get(0)?,
        nct_id: row.get(1)?,
        interval_seconds: row.get(2)?,
        next_due: row.get(3)?,
        enabled: row.get(4)?,
        lease_owner: row.get(5)?,
        lease_until: row.get(6)?,
        last_checked_at: row.get(7)?,
        last_error: row.get(8)?,
        last_snapshot: row.get(9)?,
    })
}

fn literature_watch_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<LiteratureWatch> {
    Ok(LiteratureWatch {
        session_id: row.get(0)?,
        watch_id: row.get(1)?,
        query: row.get(2)?,
        interval_seconds: row.get(3)?,
        next_due: row.get(4)?,
        enabled: row.get::<_, i64>(5)? != 0,
        lease_owner: row.get(6)?,
        lease_until: row.get(7)?,
        last_checked_at: row.get(8)?,
        last_error: row.get(9)?,
        last_pmids: row.get(10)?,
    })
}

fn literature_watch_event_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<LiteratureWatchEvent> {
    Ok(LiteratureWatchEvent {
        id: row.get(0)?,
        session_id: row.get(1)?,
        watch_id: row.get(2)?,
        captured_at: row.get(3)?,
        snapshot_sha256: row.get(4)?,
        result_count: row.get(5)?,
        retrieved_count: row.get(6)?,
        incomplete: row.get::<_, i64>(7)? != 0,
        new_pmids: row.get(8)?,
    })
}

fn valid_doi(value: &str) -> bool {
    if value.len() > 255 || !value.is_ascii() {
        return false;
    }
    let Some((prefix, suffix)) = value.split_once('/') else {
        return false;
    };
    let Some(registrant) = prefix.strip_prefix("10.") else {
        return false;
    };
    (4..=9).contains(&registrant.len())
        && registrant.bytes().all(|byte| byte.is_ascii_digit())
        && !suffix.is_empty()
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-._;()/:".contains(&byte))
}

fn validation_report_hash(
    report: &str,
    capability_id: &str,
    version: &str,
    manifest: &str,
    target_status: &str,
) -> Result<String> {
    anyhow::ensure!(
        report.len() <= 1024 * 1024,
        "validation report exceeds 1 MiB"
    );
    let parsed: serde_json::Value =
        serde_json::from_str(report).context("validation report must be JSON")?;
    let expected_manifest_sha256 = format!("{:x}", Sha256::digest(manifest.as_bytes()));
    anyhow::ensure!(
        parsed
            .get("capability_id")
            .and_then(serde_json::Value::as_str)
            == Some(capability_id)
            && parsed.get("version").and_then(serde_json::Value::as_str) == Some(version)
            && parsed
                .get("manifest_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(expected_manifest_sha256.as_str()),
        "validation report identity or manifest digest does not match the capability version"
    );
    let checks = parsed
        .get("checks")
        .and_then(|v| v.as_object())
        .context("validation report requires a checks object")?;
    let mut required = vec![
        "static_analysis",
        "contract",
        "scientific",
        "reproducibility",
    ];
    if target_status == "production" {
        required.push("benchmark");
    }
    anyhow::ensure!(
        required
            .iter()
            .all(|name| checks.get(*name) == Some(&serde_json::Value::Bool(true))),
        "validation report must pass required checks: {}",
        required.join(", ")
    );
    anyhow::ensure!(
        checks
            .values()
            .all(|value| value == &serde_json::Value::Bool(true)),
        "all supplied validation checks must pass before promotion"
    );
    Ok(format!("{:x}", Sha256::digest(report.as_bytes())))
}

fn validate_paired_benchmark_artifact(
    db: &Connection,
    capability_id: &str,
    version: &str,
    manifest: &str,
    validation_report: &str,
) -> Result<()> {
    let manifest: serde_json::Value = serde_json::from_str(manifest)?;
    if manifest.get("source").and_then(serde_json::Value::as_str) != Some("agentmesh_amcp") {
        return Ok(());
    }
    let package = manifest
        .get("package")
        .context("registered capability manifest has no AMCP package")?;
    anyhow::ensure!(
        package
            .pointer("/capability/id")
            .and_then(serde_json::Value::as_str)
            == Some(capability_id)
            && package
                .pointer("/capability/version")
                .and_then(serde_json::Value::as_str)
                == Some(version),
        "registered AMCP identity differs from its capability record"
    );
    let Some(properties) = package
        .pointer("/contract/inputs/properties")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(());
    };
    let generated_binding_keys = [
        "source_session_id",
        "proposal_artifact_id",
        "proposal_artifact_sha256",
        "proposal_capability_id",
    ];
    if !generated_binding_keys
        .iter()
        .any(|key| properties.contains_key(*key))
    {
        return Ok(());
    }
    let constant = |key: &str| -> Result<&str> {
        properties
            .get(key)
            .and_then(|value| value.get("const"))
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("registered AMCP package is missing its {key} source binding"))
    };
    let source_session = constant("source_session_id")?;
    let proposal_artifact_id = constant("proposal_artifact_id")?;
    let proposal_artifact_sha256 = constant("proposal_artifact_sha256")?;
    let proposal_capability_id = constant("proposal_capability_id")?;
    let validation: serde_json::Value =
        serde_json::from_str(validation_report).context("validation report must be JSON")?;
    let receipt = validation
        .get("benchmark_receipt")
        .context("production validation requires a benchmark_receipt reference")?;
    let receipt_session = receipt
        .get("session_id")
        .and_then(serde_json::Value::as_str)
        .context("benchmark receipt must name its source session")?;
    let receipt_id = receipt
        .get("artifact_id")
        .and_then(serde_json::Value::as_str)
        .context("benchmark receipt must name its persisted artifact")?;
    let receipt_sha256 = receipt
        .get("artifact_sha256")
        .and_then(serde_json::Value::as_str)
        .context("benchmark receipt must include the persisted artifact digest")?;
    anyhow::ensure!(
        receipt_session == source_session,
        "benchmark receipt belongs to a different source session"
    );
    anyhow::ensure!(
        re_full_sha256(receipt_sha256),
        "benchmark receipt artifact digest must be a lowercase SHA-256"
    );
    let (kind, source, payload, stored_sha256): (String, String, String, String) = db.query_row(
        "SELECT kind,source,payload,sha256 FROM research_artifacts WHERE id=?1 AND session_id=?2",
        params![receipt_id, source_session],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    ).context("paired benchmark artifact was not found in the bound source session")?;
    anyhow::ensure!(
        kind == "capability_version_benchmark"
            && source == "MedAI paired pinned Docker capability benchmark",
        "benchmark receipt must reference a completed paired capability benchmark artifact"
    );
    anyhow::ensure!(
        stored_sha256 == receipt_sha256
            && format!("{:x}", Sha256::digest(payload.as_bytes())) == stored_sha256,
        "benchmark artifact digest does not match its persisted payload"
    );
    let benchmark: serde_json::Value =
        serde_json::from_str(&payload).context("persisted benchmark artifact is invalid JSON")?;
    anyhow::ensure!(
        benchmark.get("status").and_then(serde_json::Value::as_str) == Some("completed")
            && benchmark
                .get("lifecycle_mutation")
                .and_then(serde_json::Value::as_bool)
                == Some(false)
            && benchmark
                .get("registered")
                .and_then(serde_json::Value::as_bool)
                == Some(false)
            && benchmark
                .get("promoted")
                .and_then(serde_json::Value::as_bool)
                == Some(false),
        "benchmark artifact is incomplete or records a lifecycle mutation"
    );
    anyhow::ensure!(
        benchmark
            .get("capability_id")
            .and_then(serde_json::Value::as_str)
            == Some(proposal_capability_id)
            && benchmark
                .get("candidate_proposal_artifact_id")
                .and_then(serde_json::Value::as_str)
                == Some(proposal_artifact_id)
            && benchmark
                .get("candidate_version")
                .and_then(serde_json::Value::as_str)
                == Some(proposal_artifact_id)
            && benchmark
                .get("candidate_proposal_artifact_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(proposal_artifact_sha256)
            && benchmark
                .pointer("/candidate_proposal_signature/verified")
                .and_then(serde_json::Value::as_bool)
                == Some(true),
        "paired benchmark does not validate the exact signed proposal bound to this AMCP package"
    );
    let (proposal_kind, proposal_payload, stored_proposal_sha256): (String, String, String) = db
        .query_row(
            "SELECT kind,payload,sha256 FROM research_artifacts WHERE id=?1 AND session_id=?2",
            params![proposal_artifact_id, source_session],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .context("signed source proposal artifact was not found in its bound session")?;
    anyhow::ensure!(
        proposal_kind == "generated_capability_proposal"
            && stored_proposal_sha256 == proposal_artifact_sha256
            && format!("{:x}", Sha256::digest(proposal_payload.as_bytes()))
                == proposal_artifact_sha256,
        "AMCP package does not bind the persisted generated-proposal artifact"
    );
    let source_proposal: serde_json::Value = serde_json::from_str(&proposal_payload)
        .context("persisted generated capability proposal is invalid JSON")?;
    let source_code = source_proposal
        .get("source")
        .and_then(serde_json::Value::as_str)
        .context("persisted generated proposal has no source code")?;
    let source_sha256 = format!("{:x}", Sha256::digest(source_code.as_bytes()));
    anyhow::ensure!(
        source_proposal
            .pointer("/manifest/id")
            .and_then(serde_json::Value::as_str)
            == Some(proposal_capability_id)
            && source_proposal
                .pointer("/manifest/version")
                .and_then(serde_json::Value::as_str)
                == Some(version)
            && source_proposal
                .pointer("/static_review/source_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(source_sha256.as_str())
            && benchmark
                .get("candidate_source_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(source_sha256.as_str()),
        "paired benchmark source digest differs from the exact retained capability source"
    );
    anyhow::ensure!(
        benchmark
            .get("baseline_proposal_signature")
            .and_then(|value| value.get("verified"))
            .and_then(serde_json::Value::as_bool)
            == Some(true)
            && benchmark
                .get("benchmark_sha256")
                .and_then(serde_json::Value::as_str)
                .is_some_and(re_full_sha256)
            && benchmark
                .get("comparison_sha256")
                .and_then(serde_json::Value::as_str)
                .is_some_and(re_full_sha256)
            && benchmark
                .get("execution_receipt_sha256")
                .and_then(serde_json::Value::as_str)
                .is_some_and(re_full_sha256)
            && benchmark
                .get("n")
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|count| count >= 10),
        "paired benchmark is missing source signatures, content digests, execution provenance, or ten measured cases"
    );
    Ok(())
}

fn re_full_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub struct Store {
    db: Connection,
}
impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let db = Connection::open(path).context("open MedAI database")?;
        db.busy_timeout(Duration::from_secs(5))?;
        db.pragma_update(None, "foreign_keys", "ON")
            .context("enable SQLite foreign keys")?;
        let existing_schema_version: i64 =
            db.pragma_query_value(None, "user_version", |row| row.get(0))?;
        anyhow::ensure!(
            existing_schema_version <= SQLITE_SCHEMA_VERSION,
            "database schema version {existing_schema_version} is newer than this MedAI build supports ({SQLITE_SCHEMA_VERSION})"
        );
        enable_wal_mode(&db)?;
        db.execute_batch("
        CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, objective TEXT NOT NULL, status TEXT NOT NULL, created_at INTEGER NOT NULL, depth TEXT NOT NULL DEFAULT 'standard', budget INTEGER NOT NULL DEFAULT 100, credits_used INTEGER NOT NULL DEFAULT 0, cancellation_requested INTEGER NOT NULL DEFAULT 0, tenant_id TEXT NOT NULL DEFAULT 'local');
        CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), claim TEXT NOT NULL, source TEXT NOT NULL, source_id TEXT, polarity TEXT NOT NULL, quality REAL NOT NULL CHECK(quality>=0 AND quality<=1), provenance TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS graph_nodes(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, id TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL, metadata TEXT NOT NULL, PRIMARY KEY(session_id,id));
        CREATE TABLE IF NOT EXISTS graph_edges(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL, confidence REAL NOT NULL CHECK(confidence>=0 AND confidence<=1), evidence_ids TEXT NOT NULL, artifact_ids TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(session_id,source,relation,target), FOREIGN KEY(session_id,source) REFERENCES graph_nodes(session_id,id), FOREIGN KEY(session_id,target) REFERENCES graph_nodes(session_id,id));
        CREATE TABLE IF NOT EXISTS capabilities(id TEXT NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, description TEXT NOT NULL, manifest TEXT NOT NULL, PRIMARY KEY(id,version));
        CREATE TABLE IF NOT EXISTS capability_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, capability_id TEXT NOT NULL, version TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL, validation_sha256 TEXT, occurred_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS agentmesh_route_approvals(receipt_id TEXT PRIMARY KEY, actor TEXT NOT NULL, medai_tenant_id TEXT NOT NULL, agentmesh_tenant TEXT NOT NULL, agentmesh_namespace TEXT NOT NULL, plan_review_sha256 TEXT NOT NULL, desired_sha256 TEXT NOT NULL, issued_at INTEGER NOT NULL, expires_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS api_audit_events(sequence INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE, actor TEXT NOT NULL, tenant_id TEXT, method TEXT NOT NULL, path TEXT NOT NULL, status INTEGER NOT NULL, duration_ms INTEGER NOT NULL, occurred_at INTEGER NOT NULL, previous_sha256 TEXT NOT NULL, event_sha256 TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS api_audit_events_occurred_idx ON api_audit_events(occurred_at,sequence);
        CREATE TABLE IF NOT EXISTS api_audit_archives(first_sequence INTEGER PRIMARY KEY, last_sequence INTEGER NOT NULL UNIQUE, head_sha256 TEXT NOT NULL, archive_file TEXT NOT NULL UNIQUE, archive_sha256 TEXT NOT NULL, archived_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS api_credentials(id TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL UNIQUE, subject TEXT NOT NULL, tenant_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('reader','researcher','tenant_admin','platform_admin','service')), created_at INTEGER NOT NULL, revoked_at INTEGER);
        CREATE INDEX IF NOT EXISTS api_credentials_tenant_idx ON api_credentials(tenant_id,created_at);
        CREATE TABLE IF NOT EXISTS tenant_quotas(tenant_id TEXT PRIMARY KEY, max_active_sessions INTEGER NOT NULL CHECK(max_active_sessions BETWEEN 1 AND 10000), max_session_budget INTEGER NOT NULL CHECK(max_session_budget BETWEEN 1 AND 100000), max_allocated_credits INTEGER NOT NULL CHECK(max_allocated_credits>=1), updated_at INTEGER NOT NULL, updated_by TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tenant_directory(tenant_id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('active','disabled')), created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, updated_by TEXT NOT NULL, disabled_reason TEXT);
        CREATE TABLE IF NOT EXISTS research_tasks(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, id TEXT NOT NULL, domain TEXT NOT NULL, description TEXT NOT NULL, depends_on TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('ready','blocked','running','succeeded','failed','needs_review','skipped')), attempts INTEGER NOT NULL DEFAULT 0, output TEXT, error TEXT, lease_owner TEXT, lease_until INTEGER, checkpoint TEXT, available_at INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(session_id,id));
        CREATE TABLE IF NOT EXISTS task_events(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, task_id TEXT NOT NULL, from_status TEXT, to_status TEXT NOT NULL, occurred_at INTEGER NOT NULL, details TEXT, FOREIGN KEY(session_id,task_id) REFERENCES research_tasks(session_id,id));
        CREATE TABLE IF NOT EXISTS research_artifacts(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, task_id TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL, source_id TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, recorded_at INTEGER NOT NULL, UNIQUE(session_id,task_id,kind,source,source_id,sha256));
        CREATE TABLE IF NOT EXISTS artifact_redactions(artifact_id TEXT PRIMARY KEY REFERENCES research_artifacts(id) ON DELETE CASCADE, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, original_sha256 TEXT NOT NULL, redacted_sha256 TEXT NOT NULL, plan_sha256 TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL, recorded_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS artifact_storage_usage(session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE, bytes_used INTEGER NOT NULL DEFAULT 0 CHECK(bytes_used>=0), credits_charged INTEGER NOT NULL DEFAULT 0 CHECK(credits_charged>=0));
        CREATE TABLE IF NOT EXISTS citation_links(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, citing_doi TEXT NOT NULL, cited_doi TEXT NOT NULL, source_artifact_id TEXT NOT NULL REFERENCES research_artifacts(id) ON DELETE CASCADE, reference_index INTEGER, indexed_at INTEGER NOT NULL, PRIMARY KEY(session_id,citing_doi,cited_doi,source_artifact_id));
        CREATE TABLE IF NOT EXISTS budget_events(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, task_id TEXT NOT NULL, credits INTEGER NOT NULL CHECK(credits>0), reason TEXT NOT NULL, occurred_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS model_budget_reservations(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, task_id TEXT NOT NULL, actor TEXT NOT NULL, reserved_credits INTEGER NOT NULL CHECK(reserved_credits>0), charged_credits INTEGER, state TEXT NOT NULL CHECK(state IN ('reserved','settled','uncertain')), reason TEXT NOT NULL, created_at INTEGER NOT NULL, settled_at INTEGER);
        CREATE INDEX IF NOT EXISTS model_budget_reservations_session_idx ON model_budget_reservations(session_id,created_at);
        CREATE TABLE IF NOT EXISTS specialist_credit_policies(agent_id TEXT PRIMARY KEY, credit_limit INTEGER NOT NULL CHECK(credit_limit>0), updated_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS specialist_credit_usage(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, agent_id TEXT NOT NULL, credits_used INTEGER NOT NULL DEFAULT 0 CHECK(credits_used>=0), credit_limit INTEGER NOT NULL CHECK(credit_limit>0), updated_at INTEGER NOT NULL, PRIMARY KEY(session_id,agent_id));
        CREATE TABLE IF NOT EXISTS trial_watches(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, nct_id TEXT NOT NULL, interval_seconds INTEGER NOT NULL, next_due INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, lease_owner TEXT, lease_until INTEGER, last_checked_at INTEGER, last_error TEXT, last_snapshot TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(session_id,nct_id));
        CREATE TABLE IF NOT EXISTS trial_watch_events(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, nct_id TEXT NOT NULL, captured_at INTEGER NOT NULL, snapshot_sha256 TEXT NOT NULL, changed INTEGER, snapshot TEXT NOT NULL, comparison TEXT, FOREIGN KEY(session_id,nct_id) REFERENCES trial_watches(session_id,nct_id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS literature_watches(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, watch_id TEXT NOT NULL, query TEXT NOT NULL, interval_seconds INTEGER NOT NULL, next_due INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, lease_owner TEXT, lease_until INTEGER, last_checked_at INTEGER, last_error TEXT, last_pmids TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, PRIMARY KEY(session_id,watch_id));
        CREATE TABLE IF NOT EXISTS literature_watch_events(id TEXT PRIMARY KEY, session_id TEXT NOT NULL, watch_id TEXT NOT NULL, captured_at INTEGER NOT NULL, snapshot_sha256 TEXT NOT NULL, result_count INTEGER NOT NULL, retrieved_count INTEGER NOT NULL, incomplete INTEGER NOT NULL, new_pmids TEXT NOT NULL, FOREIGN KEY(session_id,watch_id) REFERENCES literature_watches(session_id,watch_id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS literature_watch_papers(session_id TEXT NOT NULL, watch_id TEXT NOT NULL, event_id TEXT NOT NULL REFERENCES literature_watch_events(id) ON DELETE CASCADE, pmid TEXT NOT NULL, title TEXT NOT NULL, publication_date TEXT, journal TEXT, doi TEXT, pmcid TEXT, captured_at INTEGER NOT NULL, metadata_sha256 TEXT NOT NULL, PRIMARY KEY(session_id,watch_id,pmid), FOREIGN KEY(session_id,watch_id) REFERENCES literature_watches(session_id,watch_id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS research_tasks_session_status_idx ON research_tasks(session_id,status);
        CREATE INDEX IF NOT EXISTS research_artifacts_session_idx ON research_artifacts(session_id,recorded_at);
        CREATE INDEX IF NOT EXISTS trial_watches_due_idx ON trial_watches(enabled,next_due,lease_until);
        CREATE INDEX IF NOT EXISTS trial_watch_events_session_idx ON trial_watch_events(session_id,nct_id,captured_at);
        CREATE INDEX IF NOT EXISTS literature_watches_due_idx ON literature_watches(enabled,next_due,lease_until);
        CREATE INDEX IF NOT EXISTS literature_watch_events_session_idx ON literature_watch_events(session_id,watch_id,captured_at);
        CREATE INDEX IF NOT EXISTS evidence_session_idx ON evidence(session_id);")
            .context("initialize MedAI SQLite schema")?;
        // Serialize the table-shape checks and legacy backfills across API and
        // worker processes. Without this lock, two first-time openers can both
        // observe a missing column and race to apply the same ALTER TABLE.
        db.execute_batch("BEGIN IMMEDIATE")?;
        let has_audit_tenant: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('api_audit_events') WHERE name='tenant_id')",
            [],
            |row| row.get(0),
        )?;
        if !has_audit_tenant {
            db.execute("ALTER TABLE api_audit_events ADD COLUMN tenant_id TEXT", [])?;
        }
        db.execute("CREATE INDEX IF NOT EXISTS api_audit_events_tenant_idx ON api_audit_events(tenant_id,sequence)", [])?;
        let has_session_tenant: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name='tenant_id')",
            [],
            |row| row.get(0),
        )?;
        if !has_session_tenant {
            db.execute(
                "ALTER TABLE sessions ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'local'",
                [],
            )?;
        }
        db.execute(
            "CREATE INDEX IF NOT EXISTS sessions_tenant_idx ON sessions(tenant_id,created_at DESC)",
            [],
        )?;
        db.execute(
            "INSERT OR IGNORE INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) SELECT tenant_id,'active',MIN(created_at),strftime('%s','now'),'migration' FROM sessions GROUP BY tenant_id",
            [],
        )?;
        db.execute(
            "INSERT OR IGNORE INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) SELECT tenant_id,'active',updated_at,updated_at,'migration' FROM tenant_quotas",
            [],
        )?;
        db.execute(
            "INSERT OR IGNORE INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) SELECT tenant_id,'active',MIN(created_at),strftime('%s','now'),'migration' FROM api_credentials GROUP BY tenant_id",
            [],
        )?;
        let has_watch_paper_event: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('literature_watch_papers') WHERE name='event_id')",
            [],
            |row| row.get(0),
        )?;
        if !has_watch_paper_event {
            db.execute(
                "ALTER TABLE literature_watch_papers ADD COLUMN event_id TEXT NOT NULL DEFAULT ''",
                [],
            )?;
            db.execute(
                "UPDATE literature_watch_papers AS paper SET event_id=COALESCE((SELECT event.id FROM literature_watch_events AS event, json_each(event.new_pmids) AS pmid WHERE event.session_id=paper.session_id AND event.watch_id=paper.watch_id AND pmid.value=paper.pmid ORDER BY event.captured_at DESC,event.id DESC LIMIT 1),'')",
                [],
            )?;
        }
        db.execute(
            "INSERT OR IGNORE INTO artifact_storage_usage(session_id,bytes_used,credits_charged) SELECT session_id,SUM(length(CAST(payload AS BLOB))),0 FROM research_artifacts GROUP BY session_id",
            [],
        )?;
        for (column, definition) in [("lease_owner", "TEXT"), ("lease_until", "INTEGER")] {
            let exists: bool = db.query_row(
                &format!("SELECT EXISTS(SELECT 1 FROM pragma_table_info('research_tasks') WHERE name='{}')", column),
                [],
                |r| r.get(0),
            )?;
            if !exists {
                db.execute(
                    &format!(
                        "ALTER TABLE research_tasks ADD COLUMN {} {}",
                        column, definition
                    ),
                    [],
                )?;
            }
        }
        for (table, column, definition) in [
            ("research_tasks", "checkpoint", "TEXT"),
            (
                "research_tasks",
                "available_at",
                "INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "sessions",
                "cancellation_requested",
                "INTEGER NOT NULL DEFAULT 0",
            ),
        ] {
            let exists: bool = db.query_row(
                &format!(
                    "SELECT EXISTS(SELECT 1 FROM pragma_table_info('{}') WHERE name='{}')",
                    table, column
                ),
                [],
                |r| r.get(0),
            )?;
            if !exists {
                db.execute(
                    &format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, definition),
                    [],
                )?;
            }
        }
        let has_depth: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name='depth')",
            [],
            |r| r.get(0),
        )?;
        if !has_depth {
            db.execute(
                "ALTER TABLE sessions ADD COLUMN depth TEXT NOT NULL DEFAULT 'standard'",
                [],
            )?;
        }
        let has_budget: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name='budget')",
            [],
            |r| r.get(0),
        )?;
        if !has_budget {
            db.execute(
                "ALTER TABLE sessions ADD COLUMN budget INTEGER NOT NULL DEFAULT 100",
                [],
            )?;
        }
        let has_credits_used: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name='credits_used')",
            [],
            |r| r.get(0),
        )?;
        if !has_credits_used {
            db.execute(
                "ALTER TABLE sessions ADD COLUMN credits_used INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        let graph_is_scoped: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('graph_nodes') WHERE name='session_id')",
            [],
            |row| row.get(0),
        )?;
        if !graph_is_scoped {
            // Old graph rows had no research owner. Keep them isolated in a
            // dedicated legacy workspace instead of exposing them to sessions.
            db.execute_batch("ALTER TABLE graph_edges RENAME TO graph_edges_unscoped;
                ALTER TABLE graph_nodes RENAME TO graph_nodes_unscoped;
                CREATE TABLE graph_nodes(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, id TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL, metadata TEXT NOT NULL, PRIMARY KEY(session_id,id));
                CREATE TABLE graph_edges(session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, source TEXT NOT NULL, relation TEXT NOT NULL, target TEXT NOT NULL, confidence REAL NOT NULL CHECK(confidence>=0 AND confidence<=1), evidence_ids TEXT NOT NULL, artifact_ids TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(session_id,source,relation,target), FOREIGN KEY(session_id,source) REFERENCES graph_nodes(session_id,id), FOREIGN KEY(session_id,target) REFERENCES graph_nodes(session_id,id));
                INSERT OR IGNORE INTO sessions(id,objective,status,created_at,depth,budget,credits_used) VALUES('__legacy__','Migrated unscoped graph data','legacy',0,'standard',1,0);
                INSERT INTO graph_nodes(session_id,id,kind,label,metadata) SELECT '__legacy__',id,kind,label,metadata FROM graph_nodes_unscoped;
                INSERT INTO graph_edges(session_id,source,relation,target,confidence,evidence_ids,artifact_ids) SELECT '__legacy__',source,relation,target,confidence,evidence_ids,'[]' FROM graph_edges_unscoped;
                DROP TABLE graph_edges_unscoped;
                DROP TABLE graph_nodes_unscoped;")?;
        }
        let has_graph_artifact_ids: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('graph_edges') WHERE name='artifact_ids')",
            [],
            |row| row.get(0),
        )?;
        if !has_graph_artifact_ids {
            db.execute(
                "ALTER TABLE graph_edges ADD COLUMN artifact_ids TEXT NOT NULL DEFAULT '[]'",
                [],
            )?;
        }
        let has_artifact_task: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('research_artifacts') WHERE name='task_id')",
            [],
            |r| r.get(0),
        )?;
        if !has_artifact_task {
            db.execute_batch("ALTER TABLE research_artifacts RENAME TO research_artifacts_legacy;
                CREATE TABLE research_artifacts(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, task_id TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL, source_id TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, recorded_at INTEGER NOT NULL, UNIQUE(session_id,task_id,kind,source,source_id,sha256));
                INSERT INTO research_artifacts(id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at) SELECT id,session_id,'legacy.unattributed',kind,source,source_id,payload,sha256,recorded_at FROM research_artifacts_legacy;
                DROP TABLE research_artifacts_legacy;
                DROP INDEX IF EXISTS research_artifacts_session_idx;
                CREATE INDEX research_artifacts_session_idx ON research_artifacts(session_id,recorded_at);")?;
        }
        let legacy_sessions: Vec<String> = {
            let mut statement = db.prepare("SELECT id FROM sessions WHERE id!='__legacy__'")?;
            statement
                .query_map([], |row| row.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        for session_id in legacy_sessions {
            let session_status: String = db.query_row(
                "SELECT status FROM sessions WHERE id=?1",
                [&session_id],
                |row| row.get(0),
            )?;
            for (task_id, domain, description, dependencies) in RESEARCH_TASKS {
                let dependency_ids: Vec<String> = serde_json::from_str(dependencies)?;
                let mut runnable = true;
                for dependency in &dependency_ids {
                    let state: Option<String> = db
                        .query_row(
                            "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
                            params![session_id, dependency],
                            |row| row.get(0),
                        )
                        .optional()?;
                    if !matches!(state.as_deref(), Some("succeeded" | "skipped")) {
                        runnable = false;
                        break;
                    }
                }
                let initial_status = if session_status == "completed" && *task_id == "skills.learn"
                {
                    "skipped"
                } else if runnable {
                    "ready"
                } else {
                    "blocked"
                };
                db.execute(
                    "INSERT OR IGNORE INTO research_tasks(session_id,id,domain,description,depends_on,status) VALUES(?1,?2,?3,?4,?5,?6)",
                    params![session_id,task_id,domain,description,dependencies,initial_status],
                )?;
                let existing: (String, String, String, String, i64) = db.query_row(
                    "SELECT domain,description,depends_on,status,attempts FROM research_tasks WHERE session_id=?1 AND id=?2",
                    params![session_id, task_id],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
                )?;
                if existing.4 == 0
                    && matches!(existing.3.as_str(), "ready" | "blocked")
                    && (existing.0 != *domain
                        || existing.1 != *description
                        || existing.2 != *dependencies
                        || existing.3 != initial_status)
                {
                    db.execute(
                        "UPDATE research_tasks SET domain=?3,description=?4,depends_on=?5,status=?6 WHERE session_id=?1 AND id=?2 AND attempts=0",
                        params![session_id, task_id, domain, description, dependencies, initial_status],
                    )?;
                    if existing.3 != initial_status {
                        db.execute(
                            "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,?3,?4,?5,'plan dependency migration before task start')",
                            params![session_id, task_id, existing.3, initial_status, now()?],
                        )?;
                    }
                }
            }
        }
        db.pragma_update(None, "user_version", SQLITE_SCHEMA_VERSION)?;
        db.execute_batch("COMMIT")?;
        Ok(Self { db })
    }

    /// Create a consistent, integrity-checked SQLite snapshot without copying live WAL files.
    /// The destination is created atomically and is never overwritten.
    pub fn backup(&self, destination: impl AsRef<Path>) -> Result<DatabaseBackupReceipt> {
        let destination = destination.as_ref();
        anyhow::ensure!(
            !destination.as_os_str().is_empty(),
            "backup destination is empty"
        );
        anyhow::ensure!(!destination.exists(), "backup destination already exists");
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).context("create backup parent directory")?;
        let parent = parent
            .canonicalize()
            .context("resolve backup parent directory")?;
        let file_name = destination
            .file_name()
            .context("backup destination must name a file")?;
        let final_path = parent.join(file_name);
        anyhow::ensure!(!final_path.exists(), "backup destination already exists");
        let temporary_dir = parent.join(format!(".medai-backup-{}", uuid::Uuid::new_v4()));
        create_private_directory(&temporary_dir)
            .context("create private backup staging directory")?;
        let temporary_path = temporary_dir.join("snapshot.sqlite");
        let result = (|| -> Result<DatabaseBackupReceipt> {
            set_private_directory(&temporary_dir)?;
            let temporary_name = temporary_path
                .to_str()
                .context("backup path must be valid UTF-8")?;
            self.db
                .execute("VACUUM INTO ?1", params![temporary_name])
                .context("create consistent SQLite snapshot")?;
            {
                let snapshot = Connection::open_with_flags(
                    &temporary_path,
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )
                .context("open staged database snapshot")?;
                check_database_integrity(&snapshot).context("validate staged SQLite snapshot")?;
            }
            set_private_file(&temporary_path)?;
            let mut file =
                File::open(&temporary_path).context("open staged snapshot for hashing")?;
            let mut digest = Sha256::new();
            let mut bytes = 0_u64;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer).context("read staged snapshot")?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
                bytes = bytes
                    .checked_add(count as u64)
                    .context("backup byte count overflow")?;
            }
            file.sync_all().context("sync staged snapshot")?;
            fs::hard_link(&temporary_path, &final_path)
                .context("publish backup without replacing an existing file")?;
            OpenOptions::new()
                .read(true)
                .open(&final_path)?
                .sync_all()
                .context("sync published backup")?;
            sync_directory(&parent)?;
            Ok(DatabaseBackupReceipt {
                path: final_path.to_string_lossy().into_owned(),
                sha256: format!("{:x}", digest.finalize()),
                bytes,
                integrity_check: "ok".into(),
            })
        })();
        let _ = fs::remove_dir_all(&temporary_dir);
        result
    }

    /// Verify a SQLite backup's structural integrity and return its content digest.
    pub fn verify_backup(path: impl AsRef<Path>) -> Result<DatabaseBackupReceipt> {
        let path = path.as_ref();
        ensure_standalone_sqlite_snapshot(path)?;
        let snapshot = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("open SQLite backup read-only")?;
        let check = check_database_integrity(&snapshot).context("validate SQLite backup")?;
        drop(snapshot);
        let mut file = File::open(path).context("open SQLite backup for hashing")?;
        let mut digest = Sha256::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = file.read(&mut buffer).context("read SQLite backup")?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
            bytes = bytes
                .checked_add(count as u64)
                .context("backup byte count overflow")?;
        }
        Ok(DatabaseBackupReceipt {
            path: path.to_string_lossy().into_owned(),
            sha256: format!("{:x}", digest.finalize()),
            bytes,
            integrity_check: check,
        })
    }

    /// Restore a verified backup into this connection, retaining a verified pre-restore snapshot.
    /// If the source copy fails, the pre-restore snapshot is copied back before returning.
    pub fn restore_backup(
        &mut self,
        source_path: impl AsRef<Path>,
    ) -> Result<DatabaseRestoreReceipt> {
        let source_path = source_path.as_ref();
        let source_verified = Self::verify_backup(source_path)
            .context("verify restore source before modifying the database")?;
        let live_path: String = self
            .db
            .query_row("PRAGMA database_list", [], |row| row.get(2))?;
        anyhow::ensure!(
            !live_path.is_empty(),
            "restore requires a file-backed MedAI database"
        );
        let live_path = PathBuf::from(live_path)
            .canonicalize()
            .context("resolve live MedAI database path")?;
        let source_canonical = source_path
            .canonicalize()
            .context("resolve restore source path")?;
        anyhow::ensure!(
            source_canonical != live_path,
            "restore source and live database must be different files"
        );
        let parent = live_path
            .parent()
            .context("live database has no parent directory")?;
        let file_name = live_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("live database filename is not valid UTF-8")?;
        let source = Connection::open_with_flags(
            &source_canonical,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .context("open restore source read-only")?;
        source.busy_timeout(Duration::from_secs(5))?;
        check_database_integrity(&source).context("validate restore source before staging")?;
        validate_restore_schema(&source)
            .context("restore source is not compatible with this MedAI build")?;

        // Capture the opened source connection once into a private immutable
        // staging directory. Hash and restore this exact snapshot so a path
        // replacement between verification and copy cannot change provenance.
        let staging =
            PrivateDirectory::create(parent, &format!(".medai-restore-{}", uuid::Uuid::new_v4()))?;
        let staged_path = staging.path().join("snapshot.sqlite");
        let captured = stage_restore_snapshot(&source, &staged_path)
            .context("capture and verify exact restore snapshot")?;
        drop(source);
        let source_after = Self::verify_backup(&source_canonical)
            .context("reverify restore source after snapshot capture")?;
        anyhow::ensure!(
            source_after.sha256 == source_verified.sha256,
            "restore source changed while its snapshot was being captured"
        );

        let rollback_path = parent.join(format!(
            "{file_name}.pre-restore-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let rollback = self
            .backup(&rollback_path)
            .context("create pre-restore rollback snapshot")?;

        if let Err(primary) = copy_verified_database(&mut self.db, &staged_path) {
            let rollback_result = copy_verified_database(&mut self.db, Path::new(&rollback.path));
            return match rollback_result {
                Ok(()) => Err(primary).with_context(|| {
                    format!(
                        "restore failed; original database was restored from {}",
                        rollback.path
                    )
                }),
                Err(recovery) => Err(anyhow::anyhow!(
                    "restore failed ({primary:#}); automatic recovery also failed ({recovery:#}); verified pre-restore snapshot retained at {}",
                    rollback.path
                )),
            };
        }
        Ok(DatabaseRestoreReceipt {
            restored_from: source_canonical.to_string_lossy().into_owned(),
            source_sha256: source_verified.sha256,
            captured_snapshot_sha256: captured.sha256,
            rollback_backup: rollback.path,
            integrity_check: "ok".into(),
        })
    }
    /// Replace operator-configured specialist credit ceilings. Existing sessions
    /// retain the ceiling captured at their first specialist charge.
    pub fn configure_specialist_credit_policies(
        &mut self,
        policies: &std::collections::HashMap<String, i64>,
    ) -> Result<()> {
        anyhow::ensure!(
            policies.len() <= 64,
            "at most 64 specialist credit policies may be configured"
        );
        let valid_agents = [
            "research_director",
            "literature_librarian",
            "paper_curator",
            "entity_curator",
            "gene_disease_curator",
            "evidence_synthesizer",
            "genomics_analyst",
            "protein_analyst",
            "pathway_analyst",
            "trial_librarian",
            "pharmacology_analyst",
            "scientific_critic",
            "knowledge_graph_curator",
            "hypothesis_generator",
            "capability_learner",
            "reproducibility_agent",
        ];
        for (agent_id, limit) in policies {
            anyhow::ensure!(
                valid_agents.contains(&agent_id.as_str()),
                "unknown scientific agent credit policy: {agent_id}"
            );
            anyhow::ensure!(
                (1..=1_000_000).contains(limit),
                "specialist credit limit must be 1..1000000"
            );
        }
        let tx = Transaction::new_unchecked(&self.db, TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM specialist_credit_policies", [])?;
        let timestamp = now()?;
        for (agent_id, credit_limit) in policies {
            tx.execute(
                "INSERT INTO specialist_credit_policies(agent_id,credit_limit,updated_at) VALUES(?1,?2,?3)",
                params![agent_id, credit_limit, timestamp],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn create_session(&self, objective: &str) -> Result<ResearchSession> {
        self.create_session_with_options(objective, "standard", 100)
    }
    fn validate_tenant_id(tenant_id: &str) -> Result<()> {
        anyhow::ensure!(
            !tenant_id.is_empty()
                && tenant_id.len() <= 128
                && tenant_id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')),
            "tenant ID must contain 1..128 safe identifier characters"
        );
        Ok(())
    }
    pub fn ensure_tenant(&mut self, tenant_id: &str, actor: &str) -> Result<TenantDirectoryEntry> {
        Self::validate_tenant_id(tenant_id)?;
        anyhow::ensure!(
            !actor.is_empty() && actor.len() <= 128,
            "invalid tenant directory actor"
        );
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        self.db.execute(
            "INSERT OR IGNORE INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) VALUES(?1,'active',?2,?2,?3)",
            params![tenant_id, timestamp, actor],
        )?;
        self.tenant_directory_entry(tenant_id)?
            .context("tenant directory entry was not created")
    }
    pub fn create_tenant(&mut self, tenant_id: &str, actor: &str) -> Result<TenantDirectoryEntry> {
        Self::validate_tenant_id(tenant_id)?;
        anyhow::ensure!(
            !actor.is_empty() && actor.len() <= 128,
            "invalid tenant directory actor"
        );
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        self.db.execute(
            "INSERT INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) VALUES(?1,'active',?2,?2,?3)",
            params![tenant_id, timestamp, actor],
        ).context("tenant ID already exists or could not be created")?;
        self.tenant_directory_entry(tenant_id)?
            .context("created tenant directory entry disappeared")
    }
    pub fn tenant_directory_entry(&self, tenant_id: &str) -> Result<Option<TenantDirectoryEntry>> {
        Ok(self.db.query_row(
            "SELECT tenant_id,status,created_at,updated_at,updated_by,disabled_reason FROM tenant_directory WHERE tenant_id=?1",
            [tenant_id],
            |row| Ok(TenantDirectoryEntry { tenant_id: row.get(0)?, status: row.get(1)?, created_at: row.get(2)?, updated_at: row.get(3)?, updated_by: row.get(4)?, disabled_reason: row.get(5)? }),
        ).optional()?)
    }
    pub fn list_tenant_directory(&self) -> Result<Vec<TenantDirectoryEntry>> {
        let mut query = self.db.prepare("SELECT tenant_id,status,created_at,updated_at,updated_by,disabled_reason FROM tenant_directory ORDER BY tenant_id")?;
        Ok(query
            .query_map([], |row| {
                Ok(TenantDirectoryEntry {
                    tenant_id: row.get(0)?,
                    status: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                    updated_by: row.get(4)?,
                    disabled_reason: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn tenant_is_active(&self, tenant_id: &str) -> Result<bool> {
        Ok(self.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM tenant_directory WHERE tenant_id=?1 AND status='active')",
            [tenant_id],
            |row| row.get(0),
        )?)
    }
    pub fn set_tenant_status(
        &mut self,
        tenant_id: &str,
        status: &str,
        reason: Option<&str>,
        actor: &str,
    ) -> Result<TenantDirectoryEntry> {
        Self::validate_tenant_id(tenant_id)?;
        anyhow::ensure!(
            matches!(status, "active" | "disabled"),
            "tenant status must be active or disabled"
        );
        anyhow::ensure!(
            !actor.is_empty() && actor.len() <= 128,
            "invalid tenant directory actor"
        );
        let reason = reason.map(str::trim).filter(|value| !value.is_empty());
        anyhow::ensure!(
            status != "disabled" || reason.is_some_and(|value| value.len() <= 1000),
            "disabling a tenant requires a reason of at most 1000 characters"
        );
        anyhow::ensure!(
            status != "active" || reason.is_none(),
            "active tenant cannot retain a disabled reason"
        );
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let changed = self.db.execute(
            "UPDATE tenant_directory SET status=?1,updated_at=?2,updated_by=?3,disabled_reason=?4 WHERE tenant_id=?5",
            params![status, timestamp, actor, reason, tenant_id],
        )?;
        anyhow::ensure!(changed == 1, "tenant does not exist");
        self.tenant_directory_entry(tenant_id)?
            .context("updated tenant directory entry disappeared")
    }
    pub fn tenant_quota(&self, tenant_id: &str) -> Result<TenantQuota> {
        let row = self.db.query_row(
            "SELECT tenant_id,max_active_sessions,max_session_budget,max_allocated_credits,updated_at,updated_by FROM tenant_quotas WHERE tenant_id=?1",
            [tenant_id],
            |row| Ok(TenantQuota { tenant_id: row.get(0)?, max_active_sessions: row.get(1)?, max_session_budget: row.get(2)?, max_allocated_credits: row.get(3)?, updated_at: Some(row.get(4)?), updated_by: Some(row.get(5)?) }),
        ).optional()?;
        Ok(row.unwrap_or_else(|| TenantQuota {
            tenant_id: tenant_id.to_owned(),
            max_active_sessions: DEFAULT_TENANT_MAX_ACTIVE_SESSIONS,
            max_session_budget: DEFAULT_TENANT_MAX_SESSION_BUDGET,
            max_allocated_credits: DEFAULT_TENANT_MAX_ALLOCATED_CREDITS,
            updated_at: None,
            updated_by: None,
        }))
    }
    pub fn set_tenant_quota(
        &mut self,
        tenant_id: &str,
        max_active_sessions: i64,
        max_session_budget: i64,
        max_allocated_credits: i64,
        updated_by: &str,
    ) -> Result<TenantQuota> {
        Self::validate_tenant_id(tenant_id)?;
        anyhow::ensure!(
            (1..=10_000).contains(&max_active_sessions),
            "max_active_sessions must be between 1 and 10000"
        );
        anyhow::ensure!(
            (1..=100_000).contains(&max_session_budget),
            "max_session_budget must be between 1 and 100000"
        );
        anyhow::ensure!(
            (1..=100_000_000).contains(&max_allocated_credits),
            "max_allocated_credits must be between 1 and 100000000"
        );
        anyhow::ensure!(
            !updated_by.is_empty() && updated_by.len() <= 128,
            "invalid quota administrator"
        );
        self.ensure_tenant(tenant_id, updated_by)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        self.db.execute(
            "INSERT INTO tenant_quotas(tenant_id,max_active_sessions,max_session_budget,max_allocated_credits,updated_at,updated_by) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(tenant_id) DO UPDATE SET max_active_sessions=excluded.max_active_sessions,max_session_budget=excluded.max_session_budget,max_allocated_credits=excluded.max_allocated_credits,updated_at=excluded.updated_at,updated_by=excluded.updated_by",
            params![tenant_id,max_active_sessions,max_session_budget,max_allocated_credits,now,updated_by],
        )?;
        self.tenant_quota(tenant_id)
    }
    pub fn tenant_quota_usage(&self, tenant_id: &str) -> Result<serde_json::Value> {
        let quota = self.tenant_quota(tenant_id)?;
        let active_sessions: i64 = self.db.query_row(
            "SELECT count(*) FROM sessions WHERE tenant_id=?1 AND id!='__legacy__' AND status NOT IN ('completed','cancelled','failed','budget_exhausted')",
            [tenant_id], |row| row.get(0),
        )?;
        let allocated_credits: i64 = self.db.query_row(
            "SELECT COALESCE(sum(budget),0) FROM sessions WHERE tenant_id=?1 AND id!='__legacy__' AND status NOT IN ('completed','cancelled','failed','budget_exhausted')",
            [tenant_id], |row| row.get(0),
        )?;
        Ok(serde_json::json!({
            "quota": quota,
            "usage": {"active_sessions": active_sessions, "allocated_credits": allocated_credits},
            "remaining": {
                "active_sessions": quota.max_active_sessions.saturating_sub(active_sessions),
                "allocated_credits": quota.max_allocated_credits.saturating_sub(allocated_credits)
            }
        }))
    }
    pub fn create_session_with_options(
        &self,
        objective: &str,
        depth: &str,
        budget: i64,
    ) -> Result<ResearchSession> {
        self.create_session_for_tenant(objective, depth, budget, "local")
    }
    pub fn create_session_for_tenant(
        &self,
        objective: &str,
        depth: &str,
        budget: i64,
        tenant_id: &str,
    ) -> Result<ResearchSession> {
        self.create_session_for_tenant_with_api_charge(objective, depth, budget, tenant_id, None)
    }

    /// Atomically creates a session and charges the authenticated API request
    /// that created it against that session's execution budget.
    pub fn create_session_for_tenant_with_api_charge(
        &self,
        objective: &str,
        depth: &str,
        budget: i64,
        tenant_id: &str,
        api_request_credits: Option<i64>,
    ) -> Result<ResearchSession> {
        anyhow::ensure!(!objective.trim().is_empty(), "objective cannot be empty");
        if let Some(credits) = api_request_credits {
            anyhow::ensure!(credits > 0, "API request credit charge must be positive");
        }
        Self::validate_tenant_id(tenant_id)?;
        anyhow::ensure!(
            ["standard", "deep"].contains(&depth),
            "depth must be standard or deep"
        );
        anyhow::ensure!(
            (1..=100_000).contains(&budget),
            "budget must be in the range 1..100000 execution credits"
        );
        let id = format!("RES-{}", uuid::Uuid::new_v4().simple());
        let created_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        tx.execute(
            "INSERT OR IGNORE INTO tenant_directory(tenant_id,status,created_at,updated_at,updated_by) VALUES(?1,'active',?2,?2,'session-bootstrap')",
            params![tenant_id, created_at],
        )?;
        let tenant_active: bool = tx.query_row(
            "SELECT status='active' FROM tenant_directory WHERE tenant_id=?1",
            [tenant_id],
            |row| row.get(0),
        )?;
        anyhow::ensure!(tenant_active, "tenant is disabled");
        tx.execute(
            "INSERT OR IGNORE INTO tenant_quotas(tenant_id,max_active_sessions,max_session_budget,max_allocated_credits,updated_at,updated_by) VALUES(?1,?2,?3,?4,?5,'system-default')",
            params![tenant_id,DEFAULT_TENANT_MAX_ACTIVE_SESSIONS,DEFAULT_TENANT_MAX_SESSION_BUDGET,DEFAULT_TENANT_MAX_ALLOCATED_CREDITS,created_at],
        )?;
        let (max_active, max_session, max_allocated): (i64,i64,i64) = tx.query_row(
            "SELECT max_active_sessions,max_session_budget,max_allocated_credits FROM tenant_quotas WHERE tenant_id=?1",
            [tenant_id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
        )?;
        let (active_sessions, allocated_credits): (i64,i64) = tx.query_row(
            "SELECT count(*),COALESCE(sum(budget),0) FROM sessions WHERE tenant_id=?1 AND id!='__legacy__' AND status NOT IN ('completed','cancelled','failed','budget_exhausted')",
            [tenant_id], |row| Ok((row.get(0)?,row.get(1)?)),
        )?;
        anyhow::ensure!(
            active_sessions < max_active,
            "tenant quota exceeded: maximum active research sessions reached"
        );
        anyhow::ensure!(
            budget <= max_session,
            "tenant quota exceeded: requested session budget exceeds per-session limit"
        );
        anyhow::ensure!(
            allocated_credits.saturating_add(budget) <= max_allocated,
            "tenant quota exceeded: active allocated credit limit reached"
        );
        tx.execute(
            "INSERT INTO sessions(id,objective,status,created_at,depth,budget,credits_used,cancellation_requested,tenant_id) VALUES(?1,?2,'planned',?3,?4,?5,0,0,?6)",
            params![id, objective.trim(), created_at, depth, budget, tenant_id],
        )?;
        for (task_id, domain, description, dependencies) in RESEARCH_TASKS {
            let status = if *dependencies == "[]" {
                "ready"
            } else {
                "blocked"
            };
            tx.execute(
                "INSERT INTO research_tasks(session_id,id,domain,description,depends_on,status) VALUES(?1,?2,?3,?4,?5,?6)",
                params![id, task_id, domain, description, dependencies, status],
            )?;
        }
        let credits_used = if let Some(credits) = api_request_credits {
            let changed = tx.execute(
                "UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget",
                params![id, credits],
            )?;
            anyhow::ensure!(
                changed == 1,
                "research session budget cannot cover its creation API request"
            );
            tx.execute(
                "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,'api_request',?2,'session API request POST /v1/research',?3)",
                params![id, credits, created_at],
            )?;
            credits
        } else {
            0
        };
        tx.commit()?;
        Ok(ResearchSession {
            id,
            tenant_id: tenant_id.to_owned(),
            objective: objective.trim().into(),
            status: "planned".into(),
            created_at,
            depth: depth.into(),
            budget,
            credits_used,
            cancellation_requested: false,
        })
    }
    pub fn sessions(&self) -> Result<Vec<ResearchSession>> {
        let mut q = self.db.prepare(
            "SELECT id,objective,status,created_at,depth,budget,credits_used,cancellation_requested,tenant_id FROM sessions WHERE id!='__legacy__' ORDER BY created_at DESC",
        )?;
        Ok(q.query_map([], |r| {
            Ok(ResearchSession {
                id: r.get(0)?,
                tenant_id: r.get(8)?,
                objective: r.get(1)?,
                status: r.get(2)?,
                created_at: r.get(3)?,
                depth: r.get(4)?,
                budget: r.get(5)?,
                credits_used: r.get(6)?,
                cancellation_requested: r.get::<_, i64>(7)? != 0,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn sessions_for_tenant(&self, tenant_id: &str) -> Result<Vec<ResearchSession>> {
        let mut q = self.db.prepare(
            "SELECT id,objective,status,created_at,depth,budget,credits_used,cancellation_requested,tenant_id FROM sessions WHERE id!='__legacy__' AND tenant_id=?1 ORDER BY created_at DESC",
        )?;
        Ok(q.query_map([tenant_id], |r| {
            Ok(ResearchSession {
                id: r.get(0)?,
                tenant_id: r.get(8)?,
                objective: r.get(1)?,
                status: r.get(2)?,
                created_at: r.get(3)?,
                depth: r.get(4)?,
                budget: r.get(5)?,
                credits_used: r.get(6)?,
                cancellation_requested: r.get::<_, i64>(7)? != 0,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn research_session_for_tenant(
        &self,
        session_id: &str,
        tenant_id: &str,
    ) -> Result<Option<ResearchSession>> {
        self.db
            .query_row(
                "SELECT id,objective,status,created_at,depth,budget,credits_used,cancellation_requested,tenant_id FROM sessions WHERE id=?1 AND tenant_id=?2",
                params![session_id, tenant_id],
                |row| Ok(ResearchSession {
                    id: row.get(0)?, objective: row.get(1)?, status: row.get(2)?,
                    created_at: row.get(3)?, depth: row.get(4)?, budget: row.get(5)?,
                    credits_used: row.get(6)?, cancellation_requested: row.get::<_, i64>(7)? != 0,
                    tenant_id: row.get(8)?,
                }),
            )
            .optional()
            .map_err(Into::into)
    }
    /// Return session and DAG-task status counts for one tenant, without row IDs or content.
    pub fn operational_status_counts(&self, tenant_id: &str) -> Result<Vec<(String, String, i64)>> {
        let mut q = self.db.prepare(
            "SELECT 'session',status,COUNT(*) FROM sessions WHERE id!='__legacy__' AND tenant_id=?1 GROUP BY status UNION ALL SELECT 'task',t.status,COUNT(*) FROM research_tasks t JOIN sessions s ON s.id=t.session_id WHERE s.id!='__legacy__' AND s.tenant_id=?1 GROUP BY t.status ORDER BY 1,2",
        )?;
        Ok(q.query_map([tenant_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn session_tenant(&self, session_id: &str) -> Result<Option<String>> {
        self.db
            .query_row(
                "SELECT tenant_id FROM sessions WHERE id=?1",
                [session_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }
    pub fn research_tasks(&self, session_id: &str) -> Result<Vec<ResearchTask>> {
        let mut q = self.db.prepare("SELECT id,session_id,domain,description,depends_on,status,attempts,output,error,lease_owner,lease_until,checkpoint,available_at FROM research_tasks WHERE session_id=?1 ORDER BY rowid")?;
        q.query_map([session_id], |r| {
            let dependencies: String = r.get(4)?;
            let depends_on = serde_json::from_str(&dependencies).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(ResearchTask {
                id: r.get(0)?,
                session_id: r.get(1)?,
                domain: r.get(2)?,
                description: r.get(3)?,
                depends_on,
                status: r.get(5)?,
                attempts: r.get(6)?,
                output: r.get(7)?,
                error: r.get(8)?,
                lease_owner: r.get(9)?,
                lease_until: r.get(10)?,
                checkpoint: r.get(11)?,
                available_at: r.get(12)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()
        .map_err(Into::into)
    }
    pub fn session_control(&self, session_id: &str) -> Result<serde_json::Value> {
        let (status, requested): (String, i64) = self.db.query_row(
            "SELECT status,cancellation_requested FROM sessions WHERE id=?1",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(
            serde_json::json!({"session_id":session_id,"status":status,"cancellation_requested":requested != 0}),
        )
    }
    /// Request cooperative cancellation. Queued work is skipped immediately; leased workers
    /// finish their current bounded operation and must then acknowledge cancellation.
    pub fn cancel_session(&self, session_id: &str) -> Result<serde_json::Value> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let prior_status: String = tx.query_row(
            "SELECT status FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        if ["completed", "failed", "budget_exhausted"].contains(&prior_status.as_str()) {
            anyhow::bail!("cannot cancel terminal research session in state {prior_status}");
        }
        let changed = tx.execute(
            "UPDATE sessions SET cancellation_requested=1,status='cancelling' WHERE id=?1 AND cancellation_requested=0",
            [session_id],
        )?;
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sessions WHERE id=?1)",
            [session_id],
            |row| row.get(0),
        )?;
        anyhow::ensure!(exists, "research session not found");
        if changed == 1 {
            let queued: Vec<(String, String)> = {
                let mut statement = tx.prepare("SELECT id,status FROM research_tasks WHERE session_id=?1 AND status IN ('ready','blocked')")?;
                statement
                    .query_map([session_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for (task_id, prior_task_status) in queued {
                tx.execute("UPDATE research_tasks SET status='skipped',error='session cancellation requested' WHERE session_id=?1 AND id=?2", params![session_id, task_id])?;
                tx.execute("INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,?3,'skipped',?4,'session cancellation requested')", params![session_id, task_id, prior_task_status, now])?;
            }
            let running_tasks: Vec<String> = {
                let mut statement = tx.prepare(
                    "SELECT id FROM research_tasks WHERE session_id=?1 AND status='running'",
                )?;
                statement
                    .query_map([session_id], |row| row.get(0))?
                    .collect::<rusqlite::Result<_>>()?
            };
            for task_id in running_tasks {
                tx.execute("INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'running','running',?3,'session cancellation requested; worker will stop at active-operation boundary')", params![session_id, task_id, now])?;
            }
        }
        let running: i64 = tx.query_row(
            "SELECT COUNT(*) FROM research_tasks WHERE session_id=?1 AND status='running'",
            [session_id],
            |row| row.get(0),
        )?;
        let status = if running == 0 {
            "cancelled"
        } else {
            "cancelling"
        };
        tx.execute(
            "UPDATE sessions SET status=?2 WHERE id=?1",
            params![session_id, status],
        )?;
        tx.commit()?;
        Ok(
            serde_json::json!({"session_id":session_id,"status":status,"cancellation_requested":true,"running_tasks":running}),
        )
    }
    /// Atomically leases currently runnable tasks to a worker and charges task starts.
    pub fn claim_ready_tasks(
        &self,
        session_id: &str,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> Result<Vec<ResearchTask>> {
        self.claim_ready_tasks_for(session_id, worker_id, limit, lease_seconds, None)
    }

    /// Atomically leases runnable tasks, optionally restricting a specialist actor to its task IDs.
    /// An empty filter is rejected so a misconfigured specialist cannot silently become a general worker.
    pub fn claim_ready_tasks_for(
        &self,
        session_id: &str,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
        task_ids: Option<&[String]>,
    ) -> Result<Vec<ResearchTask>> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!((1..=16).contains(&limit), "claim limit must be 1..16");
        if let Some(task_ids) = task_ids {
            anyhow::ensure!(
                !task_ids.is_empty() && task_ids.len() <= 100,
                "task filter must contain 1..100 IDs"
            );
            let unique: std::collections::HashSet<_> = task_ids.iter().collect();
            anyhow::ensure!(
                unique.len() == task_ids.len(),
                "task filter contains duplicate IDs"
            );
            for task_id in task_ids {
                validate_task_id(task_id)?;
            }
        }
        anyhow::ensure!(
            (5..=3600).contains(&lease_seconds),
            "lease duration must be 5..3600 seconds"
        );
        const MAX_ATTEMPTS: i64 = 5;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = Transaction::new_unchecked(&self.db, TransactionBehavior::Immediate)?;
        let cancellation_requested: i64 = tx.query_row(
            "SELECT cancellation_requested FROM sessions WHERE id=?1",
            [session_id],
            |r| r.get(0),
        )?;
        if cancellation_requested != 0 {
            tx.commit()?;
            return Ok(Vec::new());
        }
        let candidates: Vec<(String, String)> = {
            let (sql, allowed_params) = if let Some(task_ids) = task_ids {
                let placeholders = (5..5 + task_ids.len())
                    .map(|index| format!("?{index}"))
                    .collect::<Vec<_>>()
                    .join(",");
                (
                    format!(
                        "SELECT id,depends_on FROM research_tasks WHERE session_id=?1 AND status='ready' AND attempts<?2 AND available_at<=?3 AND id IN ({placeholders}) ORDER BY available_at,rowid LIMIT ?4"
                    ),
                    Some(task_ids),
                )
            } else {
                ("SELECT id,depends_on FROM research_tasks WHERE session_id=?1 AND status='ready' AND attempts<?2 AND available_at<=?3 ORDER BY available_at,rowid LIMIT ?4".to_owned(), None)
            };
            let mut q = tx.prepare(&sql)?;
            let base = params![session_id, MAX_ATTEMPTS, now, limit as i64];
            if let Some(allowed) = allowed_params {
                let mut values = vec![
                    rusqlite::types::Value::Text(session_id.to_owned()),
                    rusqlite::types::Value::Integer(MAX_ATTEMPTS),
                    rusqlite::types::Value::Integer(now),
                    rusqlite::types::Value::Integer(limit as i64),
                ];
                values.extend(allowed.iter().cloned().map(rusqlite::types::Value::Text));
                q.query_map(rusqlite::params_from_iter(values), |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?
                .collect::<rusqlite::Result<_>>()?
            } else {
                q.query_map(base, |r| Ok((r.get(0)?, r.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?
            }
        };
        let mut claimed = Vec::new();
        let mut specialist_budget_blocked = false;
        for (task_id, raw_dependencies) in candidates {
            let dependencies: Vec<String> = serde_json::from_str(&raw_dependencies)?;
            let mut runnable = true;
            for dependency in dependencies {
                let status: String = tx.query_row(
                    "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
                    params![session_id, dependency],
                    |r| r.get(0),
                )?;
                if status != "succeeded" && status != "skipped" {
                    runnable = false;
                    break;
                }
            }
            if !runnable {
                continue;
            }
            let session_has_credit: bool = tx.query_row(
                "SELECT credits_used<budget FROM sessions WHERE id=?1",
                [session_id],
                |row| row.get(0),
            )?;
            if !session_has_credit {
                let active: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM research_tasks WHERE session_id=?1 AND status='running')",
                    [session_id],
                    |r| r.get(0),
                )?;
                tx.execute(
                    "UPDATE sessions SET status=?2 WHERE id=?1",
                    params![
                        session_id,
                        if active {
                            "running"
                        } else {
                            "budget_exhausted"
                        }
                    ],
                )?;
                break;
            }
            let agent_id = scientific_agent_for_task(&task_id)
                .context("research task has no registered scientific agent")?;
            if !specialist_budget_charge(&tx, session_id, agent_id, 1, now)? {
                tx.execute(
                    "UPDATE research_tasks SET status='needs_review',error=?3,lease_owner=NULL,lease_until=NULL WHERE session_id=?1 AND id=?2 AND status='ready'",
                    params![session_id, task_id, format!("specialist credit budget exhausted for {agent_id}")],
                )?;
                tx.execute(
                    "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'ready','needs_review',?3,?4)",
                    params![session_id, task_id, now, format!("specialist agent {agent_id} reached its configured session credit ceiling")],
                )?;
                specialist_budget_blocked = true;
                continue;
            }
            let charged = tx.execute(
                "UPDATE sessions SET credits_used=credits_used+1 WHERE id=?1 AND credits_used<budget",
                [session_id],
            )?;
            if charged != 1 {
                let active: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM research_tasks WHERE session_id=?1 AND status='running')",
                    [session_id],
                    |r| r.get(0),
                )?;
                tx.execute(
                    "UPDATE sessions SET status=?2 WHERE id=?1",
                    params![
                        session_id,
                        if active {
                            "running"
                        } else {
                            "budget_exhausted"
                        }
                    ],
                )?;
                break;
            }
            let changed = tx.execute(
                "UPDATE research_tasks SET status='running',attempts=attempts+1,output=NULL,error=NULL,lease_owner=?3,lease_until=?4 WHERE session_id=?1 AND id=?2 AND status='ready'",
                params![session_id, task_id, worker_id, now + lease_seconds],
            )?;
            if changed != 1 {
                anyhow::bail!("task {} changed while being claimed", task_id);
            }
            tx.execute(
                "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,1,'task execution claimed',?3)",
                params![session_id, task_id, now],
            )?;
            tx.execute(
                "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'ready','running',?3,?4)",
                params![session_id, task_id, now, format!("worker={} lease_until={}", worker_id, now + lease_seconds)],
            )?;
            claimed.push(task_id);
        }
        if !claimed.is_empty() {
            tx.execute(
                "UPDATE sessions SET status='running' WHERE id=?1",
                [session_id],
            )?;
        } else if specialist_budget_blocked {
            let active: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM research_tasks WHERE session_id=?1 AND status='running')",
                [session_id],
                |row| row.get(0),
            )?;
            if !active {
                tx.execute(
                    "UPDATE sessions SET status='needs_review' WHERE id=?1",
                    [session_id],
                )?;
            }
        }
        tx.commit()?;
        if claimed.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .research_tasks(session_id)?
            .into_iter()
            .filter(|task| claimed.contains(&task.id))
            .collect())
    }
    /// Extend a live lease; expired or differently-owned leases cannot be revived.
    pub fn renew_task_lease(
        &self,
        session_id: &str,
        task_id: &str,
        worker_id: &str,
        lease_seconds: i64,
    ) -> Result<i64> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(
            (5..=3600).contains(&lease_seconds),
            "lease duration must be 5..3600 seconds"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let until = now + lease_seconds;
        let tx = self.db.unchecked_transaction()?;
        let changed = tx.execute(
            "UPDATE research_tasks SET lease_until=?4 WHERE session_id=?1 AND id=?2 AND status='running' AND lease_owner=?3 AND lease_until>?5",
            params![session_id, task_id, worker_id, until, now],
        )?;
        anyhow::ensure!(
            changed == 1,
            "task lease is missing, expired, or owned by another worker"
        );
        tx.execute(
            "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'running','running',?3,?4)",
            params![session_id, task_id, now, format!("worker={} lease_renewed_until={}", worker_id, until)],
        )?;
        tx.commit()?;
        Ok(until)
    }
    /// Persist a bounded JSON checkpoint while the caller still owns a live task lease.
    pub fn save_task_checkpoint(
        &self,
        session_id: &str,
        task_id: &str,
        worker_id: &str,
        checkpoint: &str,
    ) -> Result<()> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(
            checkpoint.len() <= 262_144,
            "task checkpoint exceeds 256 KiB"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(checkpoint).context("checkpoint must be valid JSON")?;
        anyhow::ensure!(parsed.is_object(), "checkpoint must be a JSON object");
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let changed = tx.execute("UPDATE research_tasks SET checkpoint=?4 WHERE session_id=?1 AND id=?2 AND status='running' AND lease_owner=?3 AND lease_until>?5", params![session_id, task_id, worker_id, checkpoint, now])?;
        anyhow::ensure!(
            changed == 1,
            "task lease is missing, expired, or owned by another worker"
        );
        let digest = format!("{:x}", Sha256::digest(checkpoint.as_bytes()));
        tx.execute("INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'running','running',?3,?4)", params![session_id, task_id, now, format!("checkpoint_sha256={digest}")])?;
        tx.commit()?;
        Ok(())
    }
    /// Requeue expired leased tasks, failing tasks that reached the retry ceiling.
    pub fn recover_expired_tasks(&self, session_id: &str, limit: usize) -> Result<Vec<String>> {
        anyhow::ensure!((1..=100).contains(&limit), "recovery limit must be 1..100");
        const MAX_ATTEMPTS: i64 = 5;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let cancellation_requested: i64 = tx.query_row(
            "SELECT cancellation_requested FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        let expired: Vec<(String, i64, Option<String>)> = {
            let mut q = tx.prepare("SELECT id,attempts,lease_owner FROM research_tasks WHERE session_id=?1 AND status='running' AND lease_until<=?2 ORDER BY lease_until LIMIT ?3")?;
            q.query_map(params![session_id, now, limit as i64], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?
        };
        let mut recovered = Vec::new();
        for (task_id, attempts, owner) in expired {
            tx.execute(
                "UPDATE model_budget_reservations SET state='uncertain',charged_credits=reserved_credits,settled_at=?3 WHERE session_id=?1 AND task_id=?2 AND state='reserved'",
                params![session_id, task_id, now],
            )?;
            let retry_delay = 5_i64
                .saturating_mul(1_i64 << attempts.saturating_sub(1).clamp(0, 6))
                .min(300);
            let available_at = if cancellation_requested == 0 && attempts < MAX_ATTEMPTS {
                now + retry_delay
            } else {
                0
            };
            let next = if cancellation_requested != 0 {
                "skipped"
            } else if attempts >= MAX_ATTEMPTS {
                "failed"
            } else {
                "ready"
            };
            let details = if cancellation_requested != 0 {
                format!(
                    "session cancellation requested; expired worker lease; previous_worker={}",
                    owner.as_deref().unwrap_or("unknown")
                )
            } else if next == "ready" {
                format!(
                    "expired worker lease; retry scheduled after {retry_delay}s at {available_at}; previous_worker={}",
                    owner.as_deref().unwrap_or("unknown")
                )
            } else {
                format!(
                    "expired worker lease; previous_worker={}",
                    owner.as_deref().unwrap_or("unknown")
                )
            };
            tx.execute(
                "UPDATE research_tasks SET status=?3,error=?4,lease_owner=NULL,lease_until=NULL,available_at=?5 WHERE session_id=?1 AND id=?2",
                params![session_id, task_id, next, details, available_at],
            )?;
            tx.execute(
                "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'running',?3,?4,?5)",
                params![session_id, task_id, next, now, details],
            )?;
            recovered.push(task_id);
        }
        if !recovered.is_empty() {
            let (running, ready, failed, review, remaining, used, budget): (i64,i64,i64,i64,i64,i64,i64) = tx.query_row(
                "SELECT SUM(status='running'),SUM(status='ready'),SUM(status='failed'),SUM(status='needs_review'),SUM(status NOT IN ('succeeded','skipped')),(SELECT credits_used FROM sessions WHERE id=?1),(SELECT budget FROM sessions WHERE id=?1) FROM research_tasks WHERE session_id=?1",
                [session_id],
                |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?)),
            )?;
            let status = if cancellation_requested != 0 && running == 0 {
                "cancelled"
            } else if cancellation_requested != 0 {
                "cancelling"
            } else if remaining == 0 {
                "completed"
            } else if running > 0 {
                "running"
            } else if used >= budget {
                "budget_exhausted"
            } else if review > 0 {
                "needs_review"
            } else if failed > 0 {
                "failed"
            } else if ready > 0 {
                "planned"
            } else {
                "needs_review"
            };
            tx.execute(
                "UPDATE sessions SET status=?2 WHERE id=?1",
                params![session_id, status],
            )?;
        }
        tx.commit()?;
        Ok(recovered)
    }
    pub fn task_events(&self, session_id: &str) -> Result<Vec<TaskEvent>> {
        let mut q=self.db.prepare("SELECT id,session_id,task_id,from_status,to_status,occurred_at,details FROM task_events WHERE session_id=?1 ORDER BY id")?;
        Ok(q.query_map([session_id], |r| {
            Ok(TaskEvent {
                id: r.get(0)?,
                session_id: r.get(1)?,
                task_id: r.get(2)?,
                from_status: r.get(3)?,
                to_status: r.get(4)?,
                occurred_at: r.get(5)?,
                details: r.get(6)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn register_api_credential(&mut self, credential: &ApiCredentialRecord) -> Result<()> {
        anyhow::ensure!(
            !credential.id.is_empty() && credential.id.len() <= 128,
            "invalid API credential ID"
        );
        anyhow::ensure!(
            credential.token_sha256.len() == 64
                && credential
                    .token_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "invalid API credential digest"
        );
        for (label, value) in [
            ("subject", &credential.subject),
            ("tenant", &credential.tenant_id),
        ] {
            anyhow::ensure!(
                !value.is_empty()
                    && value.len() <= 128
                    && value
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':')),
                "invalid API credential {label}"
            );
        }
        anyhow::ensure!(
            matches!(
                credential.role.as_str(),
                "reader" | "researcher" | "tenant_admin" | "platform_admin" | "service"
            ),
            "invalid API credential role"
        );
        anyhow::ensure!(
            credential.revoked_at.is_none(),
            "new API credential cannot start revoked"
        );
        self.ensure_tenant(&credential.tenant_id, "credential-bootstrap")?;
        self.db.execute(
            "INSERT INTO api_credentials(id,token_sha256,subject,tenant_id,role,created_at,revoked_at) VALUES(?1,?2,?3,?4,?5,?6,NULL) ON CONFLICT(token_sha256) DO NOTHING",
            params![credential.id, credential.token_sha256, credential.subject, credential.tenant_id, credential.role, credential.created_at],
        )?;
        Ok(())
    }
    pub fn active_api_credentials(&self) -> Result<Vec<ApiCredentialRecord>> {
        let mut query = self.db.prepare("SELECT id,token_sha256,subject,tenant_id,role,created_at,revoked_at FROM api_credentials WHERE revoked_at IS NULL ORDER BY created_at,id")?;
        Ok(query
            .query_map([], |row| {
                Ok(ApiCredentialRecord {
                    id: row.get(0)?,
                    token_sha256: row.get(1)?,
                    subject: row.get(2)?,
                    tenant_id: row.get(3)?,
                    role: row.get(4)?,
                    created_at: row.get(5)?,
                    revoked_at: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn api_credentials_for_tenant(
        &self,
        tenant_id: Option<&str>,
    ) -> Result<Vec<ApiCredentialRecord>> {
        let mut query = self.db.prepare("SELECT id,token_sha256,subject,tenant_id,role,created_at,revoked_at FROM api_credentials WHERE (?1 IS NULL OR tenant_id=?1) ORDER BY created_at DESC,id")?;
        Ok(query
            .query_map([tenant_id], |row| {
                Ok(ApiCredentialRecord {
                    id: row.get(0)?,
                    token_sha256: row.get(1)?,
                    subject: row.get(2)?,
                    tenant_id: row.get(3)?,
                    role: row.get(4)?,
                    created_at: row.get(5)?,
                    revoked_at: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn revoke_api_credential(&mut self, id: &str, tenant_id: Option<&str>) -> Result<bool> {
        let revoked_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        Ok(self.db.execute(
            "UPDATE api_credentials SET revoked_at=?1 WHERE id=?2 AND revoked_at IS NULL AND (?3 IS NULL OR tenant_id=?3)",
            params![revoked_at, id, tenant_id],
        )? == 1)
    }
    pub fn record_api_audit_event(
        &mut self,
        request_id: &str,
        actor: &str,
        method: &str,
        path: &str,
        status: u16,
        duration_ms: i64,
    ) -> Result<ApiAuditEvent> {
        self.record_api_audit_event_for_tenant(
            request_id,
            &ApiAuditActor {
                actor: actor.to_owned(),
                tenant_id: None,
            },
            method,
            path,
            status,
            duration_ms,
        )
    }
    pub fn record_api_audit_event_for_tenant(
        &mut self,
        request_id: &str,
        identity: &ApiAuditActor,
        method: &str,
        path: &str,
        status: u16,
        duration_ms: i64,
    ) -> Result<ApiAuditEvent> {
        anyhow::ensure!(
            !request_id.is_empty() && request_id.len() <= 128,
            "invalid audit request ID"
        );
        anyhow::ensure!(
            !identity.actor.is_empty() && identity.actor.len() <= 128,
            "invalid audit actor"
        );
        anyhow::ensure!(
            !method.is_empty() && method.len() <= 16,
            "invalid audit HTTP method"
        );
        anyhow::ensure!(
            path.starts_with('/') && path.len() <= 512 && !path.contains('?'),
            "invalid audit path"
        );
        anyhow::ensure!(duration_ms >= 0, "audit duration cannot be negative");
        if let Some(tenant_id) = identity.tenant_id.as_deref() {
            anyhow::ensure!(
                !tenant_id.is_empty() && tenant_id.len() <= 128,
                "invalid audit tenant"
            );
        }
        let occurred_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let latest_event_hash = tx
            .query_row(
                "SELECT event_sha256 FROM api_audit_events ORDER BY sequence DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let checkpoint_hash = tx
            .query_row(
                "SELECT head_sha256 FROM api_audit_archives ORDER BY last_sequence DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let previous_sha256 = latest_event_hash
            .or(checkpoint_hash)
            .unwrap_or_else(|| "0".repeat(64));
        let sequence: i64 = tx.query_row(
            "SELECT MAX(COALESCE((SELECT MAX(sequence) FROM api_audit_events),0),COALESCE((SELECT MAX(last_sequence) FROM api_audit_archives),0))+1",
            [],
            |row| row.get(0),
        )?;
        let mut event = ApiAuditEvent {
            sequence,
            request_id: request_id.to_owned(),
            actor: identity.actor.clone(),
            tenant_id: identity.tenant_id.clone(),
            method: method.to_owned(),
            path: path.to_owned(),
            status,
            duration_ms,
            occurred_at,
            previous_sha256: previous_sha256.clone(),
            event_sha256: String::new(),
        };
        let event_sha256 = api_audit_digest(&event)?;
        tx.execute(
            "INSERT INTO api_audit_events(sequence,request_id,actor,tenant_id,method,path,status,duration_ms,occurred_at,previous_sha256,event_sha256) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![sequence, request_id, identity.actor, identity.tenant_id, method, path, status, duration_ms, occurred_at, previous_sha256, event_sha256],
        )?;
        tx.commit()?;
        event.event_sha256 = event_sha256;
        Ok(event)
    }
    pub fn api_audit_events(
        &self,
        after_sequence: i64,
        limit: usize,
    ) -> Result<Vec<ApiAuditEvent>> {
        anyhow::ensure!(
            after_sequence >= 0 && (1..=501).contains(&limit),
            "audit cursor or page limit is invalid"
        );
        let mut query = self.db.prepare("SELECT sequence,request_id,actor,tenant_id,method,path,status,duration_ms,occurred_at,previous_sha256,event_sha256 FROM api_audit_events WHERE sequence>?1 ORDER BY sequence LIMIT ?2")?;
        Ok(query
            .query_map(params![after_sequence, limit as i64], |row| {
                Ok(ApiAuditEvent {
                    sequence: row.get(0)?,
                    request_id: row.get(1)?,
                    actor: row.get(2)?,
                    tenant_id: row.get(3)?,
                    method: row.get(4)?,
                    path: row.get(5)?,
                    status: row.get(6)?,
                    duration_ms: row.get(7)?,
                    occurred_at: row.get(8)?,
                    previous_sha256: row.get(9)?,
                    event_sha256: row.get(10)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn api_audit_events_for_tenant(
        &self,
        tenant_id: &str,
        after_sequence: i64,
        limit: usize,
    ) -> Result<Vec<ApiAuditEvent>> {
        anyhow::ensure!(
            !tenant_id.is_empty()
                && tenant_id.len() <= 128
                && after_sequence >= 0
                && (1..=5001).contains(&limit),
            "audit tenant, cursor, or page limit is invalid"
        );
        let mut query = self.db.prepare("SELECT sequence,request_id,actor,tenant_id,method,path,status,duration_ms,occurred_at,previous_sha256,event_sha256 FROM api_audit_events WHERE tenant_id=?1 AND sequence>?2 ORDER BY sequence LIMIT ?3")?;
        Ok(query
            .query_map(params![tenant_id, after_sequence, limit as i64], |row| {
                Ok(ApiAuditEvent {
                    sequence: row.get(0)?,
                    request_id: row.get(1)?,
                    actor: row.get(2)?,
                    tenant_id: row.get(3)?,
                    method: row.get(4)?,
                    path: row.get(5)?,
                    status: row.get(6)?,
                    duration_ms: row.get(7)?,
                    occurred_at: row.get(8)?,
                    previous_sha256: row.get(9)?,
                    event_sha256: row.get(10)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn archive_api_audit_before(
        &mut self,
        cutoff: i64,
        archive_dir: impl AsRef<Path>,
    ) -> Result<Option<ApiAuditArchiveReceipt>> {
        anyhow::ensure!(cutoff > 0, "audit archive cutoff must be positive");
        anyhow::ensure!(
            self.verify_api_audit_chain()?["valid"] == true,
            "cannot archive an invalid audit chain"
        );
        fs::create_dir_all(archive_dir.as_ref()).context("create audit archive directory")?;
        let archive_dir =
            fs::canonicalize(archive_dir.as_ref()).context("resolve audit archive directory")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&archive_dir, fs::Permissions::from_mode(0o700))?;
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint: Option<i64> = tx.query_row(
            "SELECT MAX(last_sequence) FROM api_audit_archives",
            [],
            |row| row.get(0),
        )?;
        let after = checkpoint.unwrap_or(0);
        let mut statement = tx.prepare("SELECT sequence,request_id,actor,tenant_id,method,path,status,duration_ms,occurred_at,previous_sha256,event_sha256 FROM api_audit_events WHERE sequence>?1 ORDER BY sequence LIMIT ?2")?;
        let events = statement
            .query_map(params![after, MAX_AUDIT_ARCHIVE_EVENTS + 1], |row| {
                Ok(ApiAuditEvent {
                    sequence: row.get(0)?,
                    request_id: row.get(1)?,
                    actor: row.get(2)?,
                    tenant_id: row.get(3)?,
                    method: row.get(4)?,
                    path: row.get(5)?,
                    status: row.get(6)?,
                    duration_ms: row.get(7)?,
                    occurred_at: row.get(8)?,
                    previous_sha256: row.get(9)?,
                    event_sha256: row.get(10)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        drop(statement);
        let prefix: Vec<_> = events
            .into_iter()
            .take_while(|event| event.occurred_at < cutoff)
            .take(MAX_AUDIT_ARCHIVE_EVENTS as usize)
            .collect();
        if prefix.is_empty() {
            tx.rollback()?;
            return Ok(None);
        }
        let mut previous = if after == 0 {
            "0".repeat(64)
        } else {
            tx.query_row(
                "SELECT head_sha256 FROM api_audit_archives WHERE last_sequence=?1",
                [after],
                |row| row.get(0),
            )?
        };
        for (expected_sequence, event) in (after + 1..).zip(&prefix) {
            let digest = api_audit_digest(event)?;
            anyhow::ensure!(
                event.sequence == expected_sequence
                    && event.previous_sha256 == previous
                    && event.event_sha256 == digest,
                "audit chain changed while preparing archive at sequence {}",
                event.sequence
            );
            previous.clone_from(&event.event_sha256);
        }
        let mut bytes = Vec::with_capacity(prefix.len() * 512);
        for event in &prefix {
            serde_json::to_writer(&mut bytes, event)?;
            bytes.push(b'\n');
        }
        let archive_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let first_sequence = prefix[0].sequence;
        let last_sequence = prefix[prefix.len() - 1].sequence;
        let archive_file =
            format!("audit-{first_sequence}-{last_sequence}-{archive_sha256}.ndjson");
        let destination = archive_dir.join(&archive_file);
        if destination.exists() {
            let existing = fs::read(&destination).context("read existing audit archive")?;
            anyhow::ensure!(
                existing == bytes,
                "existing audit archive does not match its digest"
            );
        } else {
            let temporary = archive_dir.join(format!(".{archive_file}.{}.tmp", std::process::id()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .context("create temporary audit archive")?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                file.set_permissions(fs::Permissions::from_mode(0o600))?;
            }
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &destination).context("publish audit archive")?;
            File::open(&archive_dir)?.sync_all()?;
        }
        let archived_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        tx.execute(
            "INSERT INTO api_audit_archives(first_sequence,last_sequence,head_sha256,archive_file,archive_sha256,archived_at) VALUES(?1,?2,?3,?4,?5,?6)",
            params![first_sequence,last_sequence,previous,destination.to_string_lossy(),archive_sha256,archived_at],
        )?;
        tx.execute(
            "DELETE FROM api_audit_events WHERE sequence<=?1",
            [last_sequence],
        )?;
        tx.commit()?;
        Ok(Some(ApiAuditArchiveReceipt {
            first_sequence,
            last_sequence,
            archived_events: prefix.len(),
            archive_file,
            archive_sha256,
        }))
    }
    pub fn verify_api_audit_chain(&self) -> Result<serde_json::Value> {
        let archives = {
            let mut statement = self.db.prepare("SELECT first_sequence,last_sequence,head_sha256,archive_file,archive_sha256 FROM api_audit_archives ORDER BY first_sequence")?;
            statement
                .query_map([], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        let mut previous = "0".repeat(64);
        let mut next_sequence = 1_i64;
        let mut checked = 0_i64;
        for (first, last, head, archive_file, archive_sha256) in &archives {
            let path = Path::new(archive_file);
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if !path.is_absolute()
                || !file_name.starts_with("audit-")
                || !file_name.ends_with(".ndjson")
            {
                return Ok(
                    serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":"unsafe audit archive path"}),
                );
            }
            let metadata = match fs::symlink_metadata(path) {
                Ok(metadata)
                    if metadata.file_type().is_file() && metadata.len() <= 128 * 1024 * 1024 =>
                {
                    metadata
                }
                Ok(_) => {
                    return Ok(
                        serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":"audit archive is not a bounded regular file"}),
                    );
                }
                Err(error) => {
                    return Ok(
                        serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":format!("audit archive unavailable: {error}")}),
                    );
                }
            };
            let _ = metadata;
            let bytes = match fs::read(path) {
                Ok(bytes) => bytes,
                Err(error) => {
                    return Ok(
                        serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":format!("audit archive unavailable: {error}")}),
                    );
                }
            };
            if format!("{:x}", Sha256::digest(&bytes)) != *archive_sha256 {
                return Ok(
                    serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":"audit archive digest mismatch"}),
                );
            }
            let mut archive_head = previous.clone();
            let mut archive_sequence = next_sequence;
            for line in bytes
                .split(|byte| *byte == b'\n')
                .filter(|line| !line.is_empty())
            {
                let event: ApiAuditEvent = match serde_json::from_slice(line) {
                    Ok(event) => event,
                    Err(error) => {
                        return Ok(
                            serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":archive_sequence,"error":format!("invalid archived event: {error}")}),
                        );
                    }
                };
                let expected = api_audit_digest(&event)?;
                if event.sequence != archive_sequence
                    || event.previous_sha256 != archive_head
                    || event.event_sha256 != expected
                {
                    return Ok(
                        serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":event.sequence,"expected_previous_sha256":archive_head,"recorded_previous_sha256":event.previous_sha256,"expected_event_sha256":expected,"recorded_event_sha256":event.event_sha256}),
                    );
                }
                archive_head.clone_from(&event.event_sha256);
                archive_sequence += 1;
                checked += 1;
            }
            if *first != next_sequence || archive_sequence - 1 != *last || archive_head != *head {
                return Ok(
                    serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":first,"error":"audit archive checkpoint does not match archived events"}),
                );
            }
            previous = archive_head;
            next_sequence = last + 1;
        }
        let mut statement = self.db.prepare("SELECT sequence,request_id,actor,tenant_id,method,path,status,duration_ms,occurred_at,previous_sha256,event_sha256 FROM api_audit_events ORDER BY sequence")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let event = ApiAuditEvent {
                sequence: row.get(0)?,
                request_id: row.get(1)?,
                actor: row.get(2)?,
                tenant_id: row.get(3)?,
                method: row.get(4)?,
                path: row.get(5)?,
                status: row.get(6)?,
                duration_ms: row.get(7)?,
                occurred_at: row.get(8)?,
                previous_sha256: row.get(9)?,
                event_sha256: row.get(10)?,
            };
            let expected = api_audit_digest(&event)?;
            if event.sequence != next_sequence
                || event.previous_sha256 != previous
                || event.event_sha256 != expected
            {
                return Ok(
                    serde_json::json!({"valid":false,"checked_events":checked,"invalid_sequence":event.sequence,"expected_previous_sha256":previous,"recorded_previous_sha256":event.previous_sha256,"expected_event_sha256":expected,"recorded_event_sha256":event.event_sha256}),
                );
            }
            previous.clone_from(&event.event_sha256);
            next_sequence += 1;
            checked += 1;
        }
        Ok(
            serde_json::json!({"valid":true,"checked_events":checked,"archived_events":archives.iter().map(|(first,last,_,_,_)| last-first+1).sum::<i64>(),"archive_segments":archives.len(),"head_sha256":previous}),
        )
    }
    pub fn add_artifact(
        &self,
        session_id: &str,
        task_id: &str,
        kind: &str,
        source: &str,
        source_id: &str,
        payload: &str,
    ) -> Result<ResearchArtifact> {
        self.add_artifact_with_task_policy(
            session_id,
            task_id,
            ArtifactFields {
                kind,
                source,
                source_id,
                payload,
            },
            true,
        )
    }
    /// Store an authenticated human adjudication against an existing research task.
    /// Human review commonly occurs after the source-producing task has completed.
    pub fn add_human_review_artifact(
        &self,
        session_id: &str,
        task_id: &str,
        kind: &str,
        source: &str,
        source_id: &str,
        payload: &str,
    ) -> Result<ResearchArtifact> {
        anyhow::ensure!(
            matches!(
                kind,
                "drug_interaction_study_adjudication"
                    | "drug_interaction_claim_adjudication"
                    | "hypothesis_critic_adjudication"
                    | "hypothesis_literature_relevance_review"
                    | "paper_claim_alignment_adjudication"
                    | "ontology_cross_mapping_adjudication"
            ),
            "human review artifact kind is not permitted"
        );
        self.add_artifact_with_task_policy(
            session_id,
            task_id,
            ArtifactFields {
                kind,
                source,
                source_id,
                payload,
            },
            false,
        )
    }
    /// Store an authenticated user-supplied scientific input before its DAG task is leased.
    pub fn add_research_input_artifact(
        &self,
        session_id: &str,
        task_id: &str,
        kind: &str,
        source: &str,
        source_id: &str,
        payload: &str,
    ) -> Result<ResearchArtifact> {
        anyhow::ensure!(
            kind == "genomic_dataset",
            "research input artifact kind is not permitted"
        );
        self.add_artifact_with_task_policy(
            session_id,
            task_id,
            ArtifactFields {
                kind,
                source,
                source_id,
                payload,
            },
            false,
        )
    }
    fn add_artifact_with_task_policy(
        &self,
        session_id: &str,
        task_id: &str,
        artifact: ArtifactFields<'_>,
        require_running_task: bool,
    ) -> Result<ResearchArtifact> {
        anyhow::ensure!(
            !artifact.kind.trim().is_empty()
                && !artifact.source.trim().is_empty()
                && !artifact.source_id.trim().is_empty(),
            "artifact kind, source, and source_id are required"
        );
        anyhow::ensure!(
            artifact.payload.len() <= 262_144,
            "artifact payload exceeds 256 KiB"
        );
        let parsed: serde_json::Value = serde_json::from_str(artifact.payload)
            .context("artifact payload must be valid JSON")?;
        anyhow::ensure!(
            parsed.is_object() || parsed.is_array(),
            "artifact payload must be a JSON object or array"
        );
        let sha256 = format!("{:x}", Sha256::digest(artifact.payload.as_bytes()));
        let id = format!("ART-{}", uuid::Uuid::new_v4().simple());
        let recorded_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let transaction = Transaction::new_unchecked(&self.db, TransactionBehavior::Immediate)?;
        let task_status: String = transaction.query_row(
            "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
            params![session_id, task_id],
            |row| row.get(0),
        )?;
        anyhow::ensure!(
            !require_running_task || task_status == "running",
            "artifacts must be recorded by a running research task"
        );
        let inserted = transaction.execute(
            "INSERT OR IGNORE INTO research_artifacts(id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![id, session_id, task_id, artifact.kind, artifact.source, artifact.source_id, artifact.payload, sha256, recorded_at],
        )?;
        if inserted == 1 {
            let previous_bytes: i64 = transaction.query_row(
                "SELECT COALESCE((SELECT bytes_used FROM artifact_storage_usage WHERE session_id=?1),0)",
                [session_id],
                |row| row.get(0),
            )?;
            let previous_credits: i64 = transaction.query_row(
                "SELECT COALESCE((SELECT credits_charged FROM artifact_storage_usage WHERE session_id=?1),0)",
                [session_id],
                |row| row.get(0),
            )?;
            let new_bytes = previous_bytes
                .checked_add(artifact.payload.len() as i64)
                .context("artifact storage byte count overflow")?;
            let required_credits = new_bytes / ARTIFACT_STORAGE_BYTES_PER_CREDIT;
            let previous_required = previous_bytes / ARTIFACT_STORAGE_BYTES_PER_CREDIT;
            let storage_credits = required_credits.saturating_sub(previous_required);
            let total_storage_credits = previous_credits
                .checked_add(storage_credits)
                .context("artifact storage credit count overflow")?;
            if storage_credits > 0 {
                let charged = transaction.execute(
                    "UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget",
                    params![session_id, storage_credits],
                )?;
                anyhow::ensure!(
                    charged == 1,
                    "research session has insufficient execution credits for artifact storage"
                );
                let reason = format!(
                    "artifact payload storage: added_bytes={}, total_bytes={}, rate={} bytes/credit",
                    artifact.payload.len(),
                    new_bytes,
                    ARTIFACT_STORAGE_BYTES_PER_CREDIT
                );
                transaction.execute(
                    "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,?3,?4,?5)",
                    params![session_id, task_id, storage_credits, reason, recorded_at],
                )?;
            }
            transaction.execute(
                "INSERT INTO artifact_storage_usage(session_id,bytes_used,credits_charged) VALUES(?1,?2,?3) ON CONFLICT(session_id) DO UPDATE SET bytes_used=excluded.bytes_used,credits_charged=excluded.credits_charged",
                params![session_id, new_bytes, total_storage_credits],
            )?;
        }
        let persisted = transaction.query_row(
            "SELECT id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at FROM research_artifacts WHERE session_id=?1 AND task_id=?2 AND kind=?3 AND source=?4 AND source_id=?5 AND sha256=?6",
            params![session_id, task_id, artifact.kind, artifact.source, artifact.source_id, sha256],
            |row| {
                Ok(ResearchArtifact {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    task_id: row.get(2)?,
                    kind: row.get(3)?,
                    source: row.get(4)?,
                    source_id: row.get(5)?,
                    payload: row.get(6)?,
                    sha256: row.get(7)?,
                    recorded_at: row.get(8)?,
                })
            },
        )?;
        transaction.commit()?;
        Ok(persisted)
    }
    pub fn artifacts(&self, session_id: &str) -> Result<Vec<ResearchArtifact>> {
        let mut q=self.db.prepare("SELECT id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at FROM research_artifacts WHERE session_id=?1 ORDER BY recorded_at,id")?;
        Ok(q.query_map([session_id], |r| {
            Ok(ResearchArtifact {
                id: r.get(0)?,
                session_id: r.get(1)?,
                task_id: r.get(2)?,
                kind: r.get(3)?,
                source: r.get(4)?,
                source_id: r.get(5)?,
                payload: r.get(6)?,
                sha256: r.get(7)?,
                recorded_at: r.get(8)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    fn pubmed_redaction_plan(&self, session_id: &str) -> Result<ArtifactRedactionPlan> {
        let mut statement = self.db.prepare("SELECT id,source_id,kind,payload,sha256 FROM research_artifacts WHERE session_id=?1 AND source='PubMed' AND NOT EXISTS (SELECT 1 FROM artifact_redactions r WHERE r.artifact_id=research_artifacts.id) ORDER BY id")?;
        let rows = statement
            .query_map([session_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut candidates = Vec::new();
        for (artifact_id, source_id, kind, payload, original_sha256) in rows {
            let mut value: serde_json::Value = match serde_json::from_str(&payload) {
                Ok(value) => value,
                Err(_) => continue,
            };
            if !redact_pubmed_abstracts(&mut value, false) {
                continue;
            }
            let redacted = serde_json::to_string(&value)?;
            let redacted_sha256 = format!("{:x}", Sha256::digest(redacted.as_bytes()));
            candidates.push(ArtifactRedactionCandidate {
                artifact_id,
                source_id,
                kind,
                original_sha256,
                redacted_sha256,
            });
        }
        let plan_value = serde_json::json!({"session_id":session_id,"candidates":candidates});
        let plan_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&plan_value)?));
        Ok(ArtifactRedactionPlan {
            session_id: session_id.to_owned(),
            candidate_count: candidates.len(),
            candidates,
            plan_sha256,
        })
    }
    pub fn preview_pubmed_abstract_redaction(
        &self,
        session_id: &str,
    ) -> Result<ArtifactRedactionPlan> {
        anyhow::ensure!(
            self.session_tenant(session_id)?.is_some(),
            "research session not found"
        );
        self.pubmed_redaction_plan(session_id)
    }
    /// Redact historical PubMed abstract payload fields in one explicitly scoped, plan-bound transaction.
    /// The immutable provenance record preserves the original payload digest and the digest of its replacement.
    pub fn redact_historical_pubmed_abstracts(
        &mut self,
        session_id: &str,
        expected_plan_sha256: &str,
        actor: &str,
        reason: &str,
    ) -> Result<ArtifactRedactionPlan> {
        anyhow::ensure!(
            re_full_sha256(expected_plan_sha256),
            "plan_sha256 must be a lowercase SHA-256 digest"
        );
        anyhow::ensure!(
            !actor.trim().is_empty() && actor.len() <= 256,
            "actor must contain 1..256 bytes"
        );
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 2000,
            "reason must contain 1..2000 bytes"
        );
        let transaction = self.db.unchecked_transaction()?;
        let plan = self.pubmed_redaction_plan(session_id)?;
        anyhow::ensure!(
            plan.candidate_count > 0,
            "no unredacted PubMed abstract payloads remain in this session"
        );
        anyhow::ensure!(
            plan.plan_sha256 == expected_plan_sha256,
            "redaction plan changed since preview; request a new preview"
        );
        let now = now()?;
        for candidate in &plan.candidates {
            let payload: String = transaction.query_row(
                "SELECT payload FROM research_artifacts WHERE session_id=?1 AND id=?2",
                params![session_id, candidate.artifact_id],
                |row| row.get(0),
            )?;
            let mut value: serde_json::Value = serde_json::from_str(&payload)?;
            anyhow::ensure!(
                redact_pubmed_abstracts(&mut value, false),
                "candidate no longer contains abstract text"
            );
            let redacted_payload = serde_json::to_string(&value)?;
            let digest = format!("{:x}", Sha256::digest(redacted_payload.as_bytes()));
            anyhow::ensure!(
                digest == candidate.redacted_sha256,
                "redacted payload digest changed during application"
            );
            let updated = transaction.execute("UPDATE research_artifacts SET payload=?3,sha256=?4 WHERE session_id=?1 AND id=?2 AND sha256=?5", params![session_id,candidate.artifact_id,redacted_payload,digest,candidate.original_sha256])?;
            anyhow::ensure!(updated == 1, "artifact changed since preview");
            let old_bytes: i64 = payload
                .len()
                .try_into()
                .context("artifact byte count overflow")?;
            let new_bytes: i64 = redacted_payload
                .len()
                .try_into()
                .context("redacted artifact byte count overflow")?;
            let usage_updated = transaction.execute(
                "UPDATE artifact_storage_usage SET bytes_used=bytes_used-?2+?3 WHERE session_id=?1 AND bytes_used>=?2",
                params![session_id, old_bytes, new_bytes],
            )?;
            anyhow::ensure!(
                usage_updated == 1,
                "artifact storage usage is missing or inconsistent"
            );
            transaction.execute("INSERT INTO artifact_redactions(artifact_id,session_id,original_sha256,redacted_sha256,plan_sha256,actor,reason,recorded_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![candidate.artifact_id,session_id,candidate.original_sha256,digest,plan.plan_sha256,actor.trim(),reason.trim(),now])?;
        }
        transaction.commit()?;
        Ok(plan)
    }
    pub fn artifact_redaction_receipts(&self, session_id: &str) -> Result<Vec<serde_json::Value>> {
        let mut statement = self.db.prepare("SELECT artifact_id,original_sha256,redacted_sha256,plan_sha256,actor,reason,recorded_at FROM artifact_redactions WHERE session_id=?1 ORDER BY recorded_at,artifact_id")?;
        Ok(statement.query_map([session_id], |row| Ok(serde_json::json!({"artifact_id":row.get::<_,String>(0)?,"original_sha256":row.get::<_,String>(1)?,"redacted_sha256":row.get::<_,String>(2)?,"plan_sha256":row.get::<_,String>(3)?,"actor":row.get::<_,String>(4)?,"reason":row.get::<_,String>(5)?,"recorded_at":row.get::<_,i64>(6)?})))?.collect::<rusqlite::Result<Vec<_>>>()?)
    }
    pub fn artifact(
        &self,
        session_id: &str,
        artifact_id: &str,
    ) -> Result<Option<ResearchArtifact>> {
        self.db.query_row(
            "SELECT id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at FROM research_artifacts WHERE session_id=?1 AND id=?2",
            params![session_id, artifact_id],
            |row| Ok(ResearchArtifact {
                id: row.get(0)?, session_id: row.get(1)?, task_id: row.get(2)?,
                kind: row.get(3)?, source: row.get(4)?, source_id: row.get(5)?,
                payload: row.get(6)?, sha256: row.get(7)?, recorded_at: row.get(8)?,
            }),
        ).optional().map_err(Into::into)
    }
    pub fn artifacts_page(
        &self,
        session_id: &str,
        kind: Option<&str>,
        after_recorded_at: Option<i64>,
        after_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ResearchArtifact>> {
        self.filtered_artifacts_page(
            session_id,
            kind,
            None,
            None,
            after_recorded_at,
            after_id,
            limit,
        )
    }

    pub fn search_artifacts_page(
        &self,
        session_id: &str,
        kind: Option<&str>,
        search: Option<&str>,
        after_recorded_at: Option<i64>,
        after_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ResearchArtifact>> {
        self.filtered_artifacts_page(
            session_id,
            kind,
            search,
            None,
            after_recorded_at,
            after_id,
            limit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn filtered_artifacts_page(
        &self,
        session_id: &str,
        kind: Option<&str>,
        search: Option<&str>,
        source_id: Option<&str>,
        after_recorded_at: Option<i64>,
        after_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<ResearchArtifact>> {
        anyhow::ensure!(
            (1..=201).contains(&limit),
            "artifact page limit must be 1..201"
        );
        anyhow::ensure!(
            after_recorded_at.is_some() == after_id.is_some(),
            "artifact cursor requires both recorded_at and id"
        );
        let search = search.map(str::trim).filter(|value| !value.is_empty());
        anyhow::ensure!(
            search.is_none_or(|value| value.len() <= 256),
            "artifact search query exceeds 256 bytes"
        );
        let mut sql = String::from(
            "SELECT id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at FROM research_artifacts WHERE session_id=?1",
        );
        let mut values = vec![rusqlite::types::Value::Text(session_id.to_owned())];
        let mut next_slot = 2;
        if let Some(kind) = kind {
            sql.push_str(&format!(" AND kind=?{next_slot}"));
            values.push(rusqlite::types::Value::Text(kind.to_owned()));
            next_slot += 1;
        }
        if let Some(source_id) = source_id {
            sql.push_str(&format!(" AND source_id=?{next_slot}"));
            values.push(rusqlite::types::Value::Text(source_id.to_owned()));
            next_slot += 1;
        }
        if let Some(search) = search {
            sql.push_str(&format!(
                " AND (instr(lower(payload),lower(?{next_slot}))>0 OR instr(lower(source_id),lower(?{next_slot}))>0)"
            ));
            values.push(rusqlite::types::Value::Text(search.to_owned()));
            next_slot += 1;
        }
        if after_recorded_at.is_some() {
            sql.push_str(&format!(
                " AND (recorded_at,id)>(?{next_slot},?{})",
                next_slot + 1
            ));
            next_slot += 2;
        }
        sql.push_str(" ORDER BY recorded_at,id LIMIT ?");
        sql.push_str(&next_slot.to_string());
        if let (Some(recorded_at), Some(id)) = (after_recorded_at, after_id) {
            values.push(rusqlite::types::Value::Integer(recorded_at));
            values.push(rusqlite::types::Value::Text(id.to_owned()));
        }
        values.push(rusqlite::types::Value::Integer(limit as i64));
        let mut statement = self.db.prepare(&sql)?;
        Ok(statement
            .query_map(rusqlite::params_from_iter(values), |row| {
                Ok(ResearchArtifact {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    task_id: row.get(2)?,
                    kind: row.get(3)?,
                    source: row.get(4)?,
                    source_id: row.get(5)?,
                    payload: row.get(6)?,
                    sha256: row.get(7)?,
                    recorded_at: row.get(8)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn register_trial_watch(
        &self,
        session_id: &str,
        nct_id: &str,
        interval_seconds: i64,
    ) -> Result<TrialWatch> {
        anyhow::ensure!(
            nct_id.len() == 11
                && nct_id.starts_with("NCT")
                && nct_id[3..].bytes().all(|byte| byte.is_ascii_digit()),
            "nct_id must match NCT followed by eight digits"
        );
        anyhow::ensure!(
            (3600..=31_536_000).contains(&interval_seconds),
            "trial watch interval must be 3600..31536000 seconds"
        );
        let now = now()?;
        self.db.execute(
            "INSERT INTO trial_watches(session_id,nct_id,interval_seconds,next_due,enabled,created_at,updated_at) VALUES(?1,?2,?3,?4,1,?5,?5) ON CONFLICT(session_id,nct_id) DO UPDATE SET interval_seconds=excluded.interval_seconds,next_due=MIN(trial_watches.next_due,excluded.next_due),enabled=1,last_error=NULL,updated_at=excluded.updated_at",
            params![session_id,nct_id,interval_seconds,now,now],
        )?;
        self.trial_watch(session_id, nct_id)
    }
    pub fn trial_watch(&self, session_id: &str, nct_id: &str) -> Result<TrialWatch> {
        self.db
            .query_row(
                "SELECT session_id,nct_id,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_snapshot FROM trial_watches WHERE session_id=?1 AND nct_id=?2",
                params![session_id,nct_id],
                trial_watch_from_row,
            )
            .context("trial watch not found")
    }
    pub fn trial_watches(&self, session_id: &str) -> Result<Vec<TrialWatch>> {
        let mut query = self.db.prepare("SELECT session_id,nct_id,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_snapshot FROM trial_watches WHERE session_id=?1 ORDER BY nct_id")?;
        Ok(query
            .query_map([session_id], trial_watch_from_row)?
            .collect::<rusqlite::Result<_>>()?)
    }
    pub fn disable_trial_watch(&self, session_id: &str, nct_id: &str) -> Result<()> {
        anyhow::ensure!(
            self.db.execute("UPDATE trial_watches SET enabled=0,lease_owner=NULL,lease_until=NULL,updated_at=?3 WHERE session_id=?1 AND nct_id=?2", params![session_id,nct_id,now()?])? == 1,
            "trial watch not found"
        );
        Ok(())
    }
    pub fn claim_due_trial_watches(
        &mut self,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> Result<Vec<TrialWatch>> {
        self.claim_due_trial_watches_for_tenant(worker_id, limit, lease_seconds, None)
    }
    pub fn claim_due_trial_watches_for_tenant(
        &mut self,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
        tenant_id: Option<&str>,
    ) -> Result<Vec<TrialWatch>> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!((1..=100).contains(&limit), "limit must be 1..100");
        anyhow::ensure!(
            (30..=3600).contains(&lease_seconds),
            "lease must be 30..3600 seconds"
        );
        let now = now()?;
        let tx = self.db.transaction()?;
        let keys = {
            let mut query = tx.prepare("SELECT w.session_id,w.nct_id FROM trial_watches w JOIN sessions s ON s.id=w.session_id WHERE w.enabled=1 AND s.cancellation_requested=0 AND (?2 IS NULL OR s.tenant_id=?2) AND w.next_due<=?1 AND (w.lease_until IS NULL OR w.lease_until<?1) ORDER BY w.next_due,w.session_id,w.nct_id LIMIT ?3")?;
            query
                .query_map(params![now, tenant_id, limit as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (session_id, nct_id) in &keys {
            anyhow::ensure!(tx.execute("UPDATE trial_watches SET lease_owner=?3,lease_until=?4,updated_at=?1 WHERE session_id=?2 AND nct_id=?5 AND enabled=1 AND (lease_until IS NULL OR lease_until<?1)", params![now,session_id,worker_id,now+lease_seconds,nct_id])? == 1, "trial watch was claimed concurrently");
        }
        let watches = keys.into_iter().map(|(session_id,nct_id)| {
            tx.query_row("SELECT session_id,nct_id,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_snapshot FROM trial_watches WHERE session_id=?1 AND nct_id=?2", params![session_id,nct_id], trial_watch_from_row)
        }).collect::<rusqlite::Result<Vec<_>>>()?;
        tx.commit()?;
        Ok(watches)
    }
    pub fn consume_trial_watch_credits(
        &self,
        session_id: &str,
        nct_id: &str,
        worker_id: &str,
        credits: i64,
        reason: &str,
    ) -> Result<i64> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(credits > 0, "credit charge must be positive");
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 256,
            "budget reason must contain 1..256 characters"
        );
        let now = now()?;
        let tx = self.db.unchecked_transaction()?;
        let valid_lease: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM trial_watches WHERE session_id=?1 AND nct_id=?2 AND enabled=1 AND lease_owner=?3 AND lease_until>=?4)", params![session_id,nct_id,worker_id,now], |row| row.get(0))?;
        anyhow::ensure!(
            valid_lease,
            "trial watch lease is not active for this worker"
        );
        let charged = tx.execute("UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget", params![session_id,credits])?;
        if charged != 1 {
            tx.execute("UPDATE trial_watches SET enabled=0,lease_owner=NULL,lease_until=NULL,last_error='paused: session execution budget exhausted',updated_at=?3 WHERE session_id=?1 AND nct_id=?2", params![session_id,nct_id,now])?;
            tx.commit()?;
            anyhow::bail!("session execution budget exhausted; trial watch paused")
        }
        let used: i64 = tx.query_row(
            "SELECT credits_used FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        tx.execute("INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,?3,?4,?5)", params![session_id,format!("trial-watch:{nct_id}"),credits,reason,now])?;
        tx.commit()?;
        Ok(used)
    }
    pub fn complete_trial_watch(
        &self,
        session_id: &str,
        nct_id: &str,
        worker_id: &str,
        snapshot: &str,
        comparison: Option<&str>,
    ) -> Result<TrialWatchEvent> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(
            snapshot.len() <= 262_144,
            "trial watch snapshot exceeds 256 KiB"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(snapshot).context("trial watch snapshot must be JSON")?;
        anyhow::ensure!(
            parsed.get("nct_id").and_then(|value| value.as_str()) == Some(nct_id),
            "snapshot NCT ID does not match watched record"
        );
        anyhow::ensure!(
            parsed.get("source").and_then(|value| value.as_str()) == Some("ClinicalTrials.gov"),
            "snapshot source must be ClinicalTrials.gov"
        );
        anyhow::ensure!(
            parsed
                .get("record")
                .and_then(|value| value.as_object())
                .is_some(),
            "snapshot must retain the raw provider record"
        );
        anyhow::ensure!(
            parsed
                .pointer("/record/protocolSection/identificationModule/nctId")
                .and_then(|value| value.as_str())
                == Some(nct_id),
            "raw provider record NCT ID does not match watched record"
        );
        let snapshot_hash = parsed
            .pointer("/provenance/record_snapshot_sha256")
            .and_then(|value| value.as_str())
            .context("snapshot must include the ClinicalTrials.gov record hash")?;
        anyhow::ensure!(
            snapshot_hash.len() == 64 && snapshot_hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "snapshot record hash must be SHA-256"
        );
        let comparison_value = comparison
            .map(serde_json::from_str::<serde_json::Value>)
            .transpose()
            .context("comparison must be JSON")?;
        let changed = comparison_value
            .as_ref()
            .and_then(|value| value.get("changed"))
            .and_then(|value| value.as_bool());
        let now = now()?;
        let id = format!("TW-{}", uuid::Uuid::new_v4().simple());
        let tx = self.db.unchecked_transaction()?;
        let watch: (i64,i64) = tx.query_row("SELECT interval_seconds,lease_until FROM trial_watches WHERE session_id=?1 AND nct_id=?2 AND enabled=1 AND lease_owner=?3", params![session_id,nct_id,worker_id], |row| Ok((row.get(0)?,row.get(1)?)))?;
        anyhow::ensure!(watch.1 >= now, "trial watch lease expired");
        tx.execute("INSERT INTO trial_watch_events(id,session_id,nct_id,captured_at,snapshot_sha256,changed,snapshot,comparison) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![id,session_id,nct_id,now,snapshot_hash,changed,snapshot,comparison])?;
        tx.execute("UPDATE trial_watches SET next_due=?4,lease_owner=NULL,lease_until=NULL,last_checked_at=?5,last_error=NULL,last_snapshot=?6,updated_at=?5 WHERE session_id=?1 AND nct_id=?2 AND lease_owner=?3", params![session_id,nct_id,worker_id,now+watch.0,now,snapshot])?;
        tx.commit()?;
        Ok(TrialWatchEvent {
            id,
            session_id: session_id.into(),
            nct_id: nct_id.into(),
            captured_at: now,
            snapshot_sha256: snapshot_hash.into(),
            changed,
            snapshot: snapshot.into(),
            comparison: comparison.map(str::to_owned),
        })
    }
    pub fn fail_trial_watch(
        &self,
        session_id: &str,
        nct_id: &str,
        worker_id: &str,
        error: &str,
    ) -> Result<()> {
        validate_worker_id(worker_id)?;
        let now = now()?;
        anyhow::ensure!(error.len() <= 4000, "watch error exceeds 4000 characters");
        anyhow::ensure!(self.db.execute("UPDATE trial_watches SET next_due=?4,lease_owner=NULL,lease_until=NULL,last_checked_at=?3,last_error=?5,updated_at=?3 WHERE session_id=?1 AND nct_id=?2 AND enabled=1 AND lease_owner=?6", params![session_id,nct_id,now,now+900,error,worker_id])? == 1, "active trial watch lease not found");
        Ok(())
    }
    pub fn trial_watch_events(
        &self,
        session_id: &str,
        nct_id: &str,
    ) -> Result<Vec<TrialWatchEvent>> {
        let mut query = self.db.prepare("SELECT id,session_id,nct_id,captured_at,snapshot_sha256,changed,snapshot,comparison FROM trial_watch_events WHERE session_id=?1 AND nct_id=?2 ORDER BY captured_at DESC,id DESC LIMIT 100")?;
        Ok(query
            .query_map(params![session_id, nct_id], |row| {
                Ok(TrialWatchEvent {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    nct_id: row.get(2)?,
                    captured_at: row.get(3)?,
                    snapshot_sha256: row.get(4)?,
                    changed: row.get(5)?,
                    snapshot: row.get(6)?,
                    comparison: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn register_literature_watch(
        &self,
        session_id: &str,
        query: &str,
        interval_seconds: i64,
    ) -> Result<LiteratureWatch> {
        let normalized = query.split_whitespace().collect::<Vec<_>>().join(" ");
        anyhow::ensure!(
            !normalized.is_empty() && normalized.len() <= 1000,
            "literature query must contain 1..1000 characters"
        );
        anyhow::ensure!(
            (86_400..=31_536_000).contains(&interval_seconds),
            "literature watch interval must be 86400..31536000 seconds"
        );
        let watch_id = format!("{:x}", Sha256::digest(normalized.as_bytes()));
        let timestamp = now()?;
        self.db.execute(
            "INSERT INTO literature_watches(session_id,watch_id,query,interval_seconds,next_due,enabled,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,1,?5,?5) ON CONFLICT(session_id,watch_id) DO UPDATE SET interval_seconds=excluded.interval_seconds,next_due=MIN(literature_watches.next_due,excluded.next_due),enabled=1,last_error=NULL,updated_at=excluded.updated_at",
            params![session_id,watch_id,normalized,interval_seconds,timestamp],
        )?;
        self.literature_watch(session_id, &watch_id)
    }

    pub fn literature_watch(&self, session_id: &str, watch_id: &str) -> Result<LiteratureWatch> {
        self.db.query_row(
            "SELECT session_id,watch_id,query,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_pmids FROM literature_watches WHERE session_id=?1 AND watch_id=?2",
            params![session_id,watch_id], literature_watch_from_row,
        ).context("literature watch not found")
    }

    pub fn literature_watches(&self, session_id: &str) -> Result<Vec<LiteratureWatch>> {
        let mut statement = self.db.prepare("SELECT session_id,watch_id,query,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_pmids FROM literature_watches WHERE session_id=?1 ORDER BY watch_id")?;
        Ok(statement
            .query_map([session_id], literature_watch_from_row)?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn disable_literature_watch(&self, session_id: &str, watch_id: &str) -> Result<()> {
        anyhow::ensure!(self.db.execute("UPDATE literature_watches SET enabled=0,lease_owner=NULL,lease_until=NULL,updated_at=?3 WHERE session_id=?1 AND watch_id=?2", params![session_id,watch_id,now()?])? == 1, "literature watch not found");
        Ok(())
    }

    pub fn claim_due_literature_watches(
        &mut self,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
    ) -> Result<Vec<LiteratureWatch>> {
        self.claim_due_literature_watches_for_tenant(worker_id, limit, lease_seconds, None)
    }
    pub fn claim_due_literature_watches_for_tenant(
        &mut self,
        worker_id: &str,
        limit: usize,
        lease_seconds: i64,
        tenant_id: Option<&str>,
    ) -> Result<Vec<LiteratureWatch>> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!((1..=100).contains(&limit), "limit must be 1..100");
        anyhow::ensure!(
            (30..=3600).contains(&lease_seconds),
            "lease must be 30..3600 seconds"
        );
        let timestamp = now()?;
        let tx = self.db.transaction()?;
        let keys = {
            let mut query = tx.prepare("SELECT w.session_id,w.watch_id FROM literature_watches w JOIN sessions s ON s.id=w.session_id WHERE w.enabled=1 AND s.cancellation_requested=0 AND (?2 IS NULL OR s.tenant_id=?2) AND w.next_due<=?1 AND (w.lease_until IS NULL OR w.lease_until<?1) ORDER BY w.next_due,w.session_id,w.watch_id LIMIT ?3")?;
            query
                .query_map(params![timestamp, tenant_id, limit as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        for (session_id, watch_id) in &keys {
            anyhow::ensure!(tx.execute("UPDATE literature_watches SET lease_owner=?3,lease_until=?4,updated_at=?1 WHERE session_id=?2 AND watch_id=?5 AND enabled=1 AND (lease_until IS NULL OR lease_until<?1)", params![timestamp,session_id,worker_id,timestamp+lease_seconds,watch_id])? == 1, "literature watch was claimed concurrently");
        }
        let watches = keys.into_iter().map(|(session_id,watch_id)| tx.query_row(
            "SELECT session_id,watch_id,query,interval_seconds,next_due,enabled,lease_owner,lease_until,last_checked_at,last_error,last_pmids FROM literature_watches WHERE session_id=?1 AND watch_id=?2",
            params![session_id,watch_id], literature_watch_from_row,
        )).collect::<rusqlite::Result<Vec<_>>>()?;
        tx.commit()?;
        Ok(watches)
    }

    pub fn consume_literature_watch_credits(
        &self,
        session_id: &str,
        watch_id: &str,
        worker_id: &str,
        credits: i64,
        reason: &str,
    ) -> Result<i64> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(credits > 0, "credit charge must be positive");
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 256,
            "budget reason must contain 1..256 characters"
        );
        let timestamp = now()?;
        let tx = self.db.unchecked_transaction()?;
        let active: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM literature_watches WHERE session_id=?1 AND watch_id=?2 AND enabled=1 AND lease_owner=?3 AND lease_until>=?4)", params![session_id,watch_id,worker_id,timestamp], |row| row.get(0))?;
        anyhow::ensure!(
            active,
            "literature watch lease is not active for this worker"
        );
        if tx.execute("UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget", params![session_id,credits])? != 1 {
            tx.execute("UPDATE literature_watches SET enabled=0,lease_owner=NULL,lease_until=NULL,last_error='paused: session execution budget exhausted',updated_at=?3 WHERE session_id=?1 AND watch_id=?2", params![session_id,watch_id,timestamp])?;
            tx.commit()?;
            anyhow::bail!("session execution budget exhausted; literature watch paused")
        }
        let used: i64 = tx.query_row(
            "SELECT credits_used FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        tx.execute("INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,?3,?4,?5)", params![session_id,format!("literature-watch:{watch_id}"),credits,reason,timestamp])?;
        tx.commit()?;
        Ok(used)
    }

    pub fn complete_literature_watch(
        &self,
        session_id: &str,
        watch_id: &str,
        worker_id: &str,
        pmids: &[String],
        result_count: i64,
        papers: &[LiteratureWatchPaper],
    ) -> Result<LiteratureWatchEvent> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(
            pmids.len() <= 200
                && pmids.iter().all(|id| !id.is_empty()
                    && id.len() <= 12
                    && id.bytes().all(|byte| byte.is_ascii_digit())),
            "PubMed watch accepts at most 200 numeric PMIDs"
        );
        anyhow::ensure!(
            result_count >= pmids.len() as i64,
            "PubMed search count is smaller than the returned PMID list"
        );
        let unique: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            pmids
                .iter()
                .filter(|id| seen.insert((*id).clone()))
                .cloned()
                .collect()
        };
        let pmids_json = serde_json::to_string(&unique)?;
        let timestamp = now()?;
        let tx = self.db.unchecked_transaction()?;
        let watch: (i64, i64, Option<String>) = tx.query_row(
            "SELECT interval_seconds,lease_until,last_pmids FROM literature_watches WHERE session_id=?1 AND watch_id=?2 AND enabled=1 AND lease_owner=?3",
            params![session_id,watch_id,worker_id], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)),
        )?;
        anyhow::ensure!(watch.1 >= timestamp, "literature watch lease expired");
        let previous: std::collections::HashSet<String> = watch
            .2
            .as_deref()
            .map(serde_json::from_str)
            .transpose()?
            .unwrap_or_default();
        let new_pmids: Vec<&String> = unique
            .iter()
            .filter(|pmid| !previous.contains(*pmid))
            .collect();
        let new_pmids_json = serde_json::to_string(&new_pmids)?;
        anyhow::ensure!(
            papers.len() <= 20,
            "PubMed watch retains at most 20 bibliographic records per refresh"
        );
        anyhow::ensure!(
            papers
                .iter()
                .map(|paper| paper.pmid.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len()
                == papers.len(),
            "PubMed bibliographic records must have unique PMIDs"
        );
        let new_set: std::collections::HashSet<&str> =
            new_pmids.iter().map(|value| value.as_str()).collect();
        for paper in papers {
            anyhow::ensure!(
                new_set.contains(paper.pmid.as_str()),
                "bibliographic metadata must belong to a newly discovered PMID"
            );
            anyhow::ensure!(
                !paper.title.trim().is_empty() && paper.title.len() <= 2000,
                "PubMed title is empty or too long"
            );
            for value in [
                &paper.publication_date,
                &paper.journal,
                &paper.doi,
                &paper.pmcid,
            ]
            .into_iter()
            .flatten()
            {
                anyhow::ensure!(
                    value.len() <= 500,
                    "PubMed bibliographic field exceeds 500 characters"
                );
            }
        }
        let paper_hashes = papers.iter().map(|paper| -> Result<String> {
            let value = serde_json::json!({"pmid":paper.pmid,"title":paper.title,"publication_date":paper.publication_date,"journal":paper.journal,"doi":paper.doi,"pmcid":paper.pmcid});
            Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
        }).collect::<Result<Vec<_>>>()?;
        let mut canonical_paper_hashes = paper_hashes.clone();
        canonical_paper_hashes.sort();
        let incomplete = result_count > unique.len() as i64;
        let canonical = serde_json::json!({"watch_id":watch_id,"result_count":result_count,"pmids":unique,"bibliographic_metadata_sha256":canonical_paper_hashes});
        let canonical_bytes = serde_json::to_vec(&canonical)?;
        let snapshot_sha256 = format!("{:x}", Sha256::digest(&canonical_bytes));
        let event_id = format!("LW-{}", uuid::Uuid::new_v4().simple());
        tx.execute("INSERT INTO literature_watch_events(id,session_id,watch_id,captured_at,snapshot_sha256,result_count,retrieved_count,incomplete,new_pmids) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)", params![event_id,session_id,watch_id,timestamp,snapshot_sha256,result_count,unique.len() as i64,incomplete,new_pmids_json])?;
        for (paper, hash) in papers.iter().zip(paper_hashes.iter()) {
            tx.execute("INSERT OR IGNORE INTO literature_watch_papers(session_id,watch_id,event_id,pmid,title,publication_date,journal,doi,pmcid,captured_at,metadata_sha256) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)", params![session_id,watch_id,event_id,paper.pmid,paper.title,paper.publication_date,paper.journal,paper.doi,paper.pmcid,timestamp,hash])?;
        }
        tx.execute("UPDATE literature_watches SET next_due=?4,lease_owner=NULL,lease_until=NULL,last_checked_at=?5,last_error=CASE WHEN ?6 THEN 'review required: PubMed result window exceeded the 200-record collection limit' ELSE NULL END,last_pmids=?7,updated_at=?5 WHERE session_id=?1 AND watch_id=?2 AND lease_owner=?3", params![session_id,watch_id,worker_id,timestamp+watch.0,timestamp,incomplete,pmids_json])?;
        tx.commit()?;
        Ok(LiteratureWatchEvent {
            id: event_id,
            session_id: session_id.into(),
            watch_id: watch_id.into(),
            captured_at: timestamp,
            snapshot_sha256,
            result_count,
            retrieved_count: unique.len() as i64,
            incomplete,
            new_pmids: new_pmids_json,
        })
    }

    pub fn literature_watch_papers(
        &self,
        session_id: &str,
        watch_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<LiteratureWatchPaper>> {
        anyhow::ensure!(
            (1..=500).contains(&limit),
            "literature watch paper limit must be 1..500"
        );
        let mut statement = self.db.prepare("SELECT session_id,watch_id,event_id,pmid,title,publication_date,journal,doi,pmcid,captured_at,metadata_sha256 FROM literature_watch_papers WHERE session_id=?1 AND (?2 IS NULL OR watch_id=?2) ORDER BY captured_at DESC,pmid LIMIT ?3")?;
        Ok(statement
            .query_map(params![session_id, watch_id, limit as i64], |row| {
                Ok(LiteratureWatchPaper {
                    session_id: row.get(0)?,
                    watch_id: row.get(1)?,
                    event_id: row.get(2)?,
                    pmid: row.get(3)?,
                    title: row.get(4)?,
                    publication_date: row.get(5)?,
                    journal: row.get(6)?,
                    doi: row.get(7)?,
                    pmcid: row.get(8)?,
                    captured_at: row.get(9)?,
                    metadata_sha256: row.get(10)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub fn fail_literature_watch(
        &self,
        session_id: &str,
        watch_id: &str,
        worker_id: &str,
        error: &str,
    ) -> Result<()> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(error.len() <= 4000, "watch error exceeds 4000 characters");
        let timestamp = now()?;
        anyhow::ensure!(self.db.execute("UPDATE literature_watches SET next_due=?4,lease_owner=NULL,lease_until=NULL,last_checked_at=?3,last_error=?5,updated_at=?3 WHERE session_id=?1 AND watch_id=?2 AND enabled=1 AND lease_owner=?6", params![session_id,watch_id,timestamp,timestamp+900,error,worker_id])? == 1, "active literature watch lease not found");
        Ok(())
    }

    pub fn literature_watch_events(
        &self,
        session_id: &str,
        watch_id: &str,
    ) -> Result<Vec<LiteratureWatchEvent>> {
        let mut statement = self.db.prepare("SELECT id,session_id,watch_id,captured_at,snapshot_sha256,result_count,retrieved_count,incomplete,new_pmids FROM literature_watch_events WHERE session_id=?1 AND watch_id=?2 ORDER BY captured_at DESC,id DESC LIMIT 100")?;
        Ok(statement
            .query_map(
                params![session_id, watch_id],
                literature_watch_event_from_row,
            )?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Derive DOI citation links only from a retained Crossref metadata artifact.
    pub fn index_crossref_citations(&self, session_id: &str, artifact_id: &str) -> Result<usize> {
        let (kind, source, payload): (String, String, String) = self
            .db
            .query_row(
                "SELECT kind,source,payload FROM research_artifacts WHERE id=?1 AND session_id=?2",
                params![artifact_id, session_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .context("Crossref citation source artifact was not found in this research session")?;
        anyhow::ensure!(
            kind == "crossref_citation_metadata" && source == "Crossref REST API",
            "citation links can only be indexed from Crossref metadata artifacts"
        );
        let payload: serde_json::Value = serde_json::from_str(&payload)?;
        anyhow::ensure!(
            payload.get("source").and_then(|v| v.as_str()) == Some("Crossref REST API"),
            "artifact payload source is not Crossref"
        );
        let source_hash = payload
            .get("complete_result_sha256")
            .and_then(|v| v.as_str())
            .context("Crossref artifact is missing its complete-result hash")?;
        anyhow::ensure!(
            source_hash.len() == 64 && source_hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Crossref artifact has an invalid complete-result hash"
        );
        let records = payload
            .get("records")
            .and_then(|v| v.as_array())
            .context("Crossref artifact has no records array")?;
        let tx = self.db.unchecked_transaction()?;
        let indexed_at = now()?;
        let mut inserted = 0;
        for record in records {
            let citing = record
                .get("doi")
                .and_then(|v| v.as_str())
                .context("Crossref citation record is missing DOI")?;
            anyhow::ensure!(
                valid_doi(citing),
                "Crossref artifact contains invalid citing DOI"
            );
            let references = record
                .get("references")
                .and_then(|v| v.as_array())
                .context("Crossref citation record is missing references array")?;
            for reference in references {
                let Some(cited) = reference.get("DOI").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !valid_doi(cited) {
                    continue;
                }
                let index = reference.get("reference_index").and_then(|v| v.as_i64());
                inserted += tx.execute("INSERT OR IGNORE INTO citation_links(session_id,citing_doi,cited_doi,source_artifact_id,reference_index,indexed_at) VALUES(?1,?2,?3,?4,?5,?6)", params![session_id,citing.to_ascii_lowercase(),cited.to_ascii_lowercase(),artifact_id,index,indexed_at])?;
            }
        }
        tx.commit()?;
        Ok(inserted)
    }
    pub fn citation_links(&self, session_id: &str, doi: Option<&str>) -> Result<Vec<CitationLink>> {
        if let Some(value) = doi {
            anyhow::ensure!(valid_doi(value), "filter must be a DOI");
        }
        let mut q = self.db.prepare("SELECT session_id,citing_doi,cited_doi,source_artifact_id,reference_index,indexed_at FROM citation_links WHERE session_id=?1 AND (?2 IS NULL OR citing_doi=?2 OR cited_doi=?2) ORDER BY citing_doi,cited_doi")?;
        Ok(
            q.query_map(params![session_id, doi.map(str::to_ascii_lowercase)], |r| {
                Ok(CitationLink {
                    session_id: r.get(0)?,
                    citing_doi: r.get(1)?,
                    cited_doi: r.get(2)?,
                    source_artifact_id: r.get(3)?,
                    reference_index: r.get(4)?,
                    indexed_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?,
        )
    }
    /// Rank papers by connectivity within the session's retained Crossref citation graph.
    /// These scores describe only the captured graph and are not evidence or paper quality.
    pub fn citation_metrics(&self, session_id: &str) -> Result<Vec<CitationMetric>> {
        let links = self.citation_links(session_id, None)?;
        let edges: BTreeSet<(String, String)> = links
            .iter()
            .map(|link| (link.citing_doi.clone(), link.cited_doi.clone()))
            .collect();
        let nodes: BTreeSet<String> = edges
            .iter()
            .flat_map(|(citing, cited)| [citing.clone(), cited.clone()])
            .collect();
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let count = nodes.len();
        let index: BTreeMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .map(|(index, doi)| (doi.as_str(), index))
            .collect();
        let mut outgoing = vec![Vec::new(); count];
        let mut citations_received = vec![0usize; count];
        let mut references_tracked = vec![0usize; count];
        for (citing, cited) in &edges {
            let source = index[citing.as_str()];
            let target = index[cited.as_str()];
            outgoing[source].push(target);
            references_tracked[source] += 1;
            citations_received[target] += 1;
        }
        let mut rank = vec![1.0 / count as f64; count];
        for _ in 0..20 {
            let dangling = rank
                .iter()
                .enumerate()
                .filter(|(node, _)| outgoing[*node].is_empty())
                .map(|(_, value)| value)
                .sum::<f64>();
            let base = (1.0 - 0.85) / count as f64 + 0.85 * dangling / count as f64;
            let mut next = vec![base; count];
            for (source, targets) in outgoing.iter().enumerate() {
                if targets.is_empty() {
                    continue;
                }
                let contribution = 0.85 * rank[source] / targets.len() as f64;
                for target in targets {
                    next[*target] += contribution;
                }
            }
            rank = next;
        }
        let mut metrics: Vec<CitationMetric> = nodes
            .into_iter()
            .enumerate()
            .map(|(node, doi)| CitationMetric {
                doi,
                citations_received: citations_received[node],
                references_tracked: references_tracked[node],
                pagerank: rank[node],
            })
            .collect();
        metrics.sort_by(|left, right| {
            right
                .pagerank
                .total_cmp(&left.pagerank)
                .then_with(|| left.doi.cmp(&right.doi))
        });
        Ok(metrics)
    }
    /// Atomically records a charge only when the session still has enough credits.
    pub fn reserve_model_credits(
        &self,
        session_id: &str,
        task_id: &str,
        reservation_id: &str,
        actor: &str,
        credits: i64,
        reason: &str,
    ) -> Result<i64> {
        anyhow::ensure!(
            !reservation_id.is_empty() && reservation_id.len() <= 128,
            "reservation ID must contain 1..128 bytes"
        );
        anyhow::ensure!(
            !actor.trim().is_empty() && actor.len() <= 256,
            "reservation actor must contain 1..256 bytes"
        );
        anyhow::ensure!(
            credits > 0 && credits <= 1_000_000,
            "model reservation must be 1..1000000 credits"
        );
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 256,
            "reservation reason must contain 1..256 bytes"
        );
        let timestamp = now()?;
        let tx = self.db.unchecked_transaction()?;
        let status: String = tx.query_row(
            "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
            params![session_id, task_id],
            |row| row.get(0),
        )?;
        anyhow::ensure!(
            status == "running",
            "model credits can only be reserved for a running task"
        );
        let agent = scientific_agent_for_task(task_id)
            .context("research task has no registered scientific agent")?;
        anyhow::ensure!(
            specialist_budget_charge(&tx, session_id, agent, credits, timestamp)?,
            "specialist credit budget exhausted for {agent}"
        );
        let changed = tx.execute(
            "UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget",
            params![session_id, credits],
        )?;
        anyhow::ensure!(
            changed == 1,
            "research session is missing or has insufficient execution credits"
        );
        tx.execute(
            "INSERT INTO model_budget_reservations(id,session_id,task_id,actor,reserved_credits,charged_credits,state,reason,created_at,settled_at) VALUES(?1,?2,?3,?4,?5,NULL,'reserved',?6,?7,NULL)",
            params![reservation_id, session_id, task_id, actor.trim(), credits, reason.trim(), timestamp],
        )?;
        let used: i64 = tx.query_row(
            "SELECT credits_used FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(used)
    }

    /// Settle an in-flight model reservation to measured usage, or retain all reserved credits when outcome is uncertain.
    pub fn settle_model_credits(
        &self,
        session_id: &str,
        reservation_id: &str,
        actor: &str,
        actual_credits: Option<i64>,
    ) -> Result<serde_json::Value> {
        if let Some(actual) = actual_credits {
            anyhow::ensure!(actual >= 0, "actual model credits cannot be negative");
        }
        let timestamp = now()?;
        let tx = self.db.unchecked_transaction()?;
        let record: (String, String, i64, Option<i64>, String) = tx.query_row(
            "SELECT task_id,actor,reserved_credits,charged_credits,state FROM model_budget_reservations WHERE session_id=?1 AND id=?2",
            params![session_id, reservation_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )?;
        anyhow::ensure!(
            record.1 == actor,
            "model reservation belongs to another service actor"
        );
        if record.4 != "reserved" {
            let same_outcome = (record.4 == "settled" && record.3 == actual_credits)
                || (record.4 == "uncertain" && actual_credits.is_none());
            anyhow::ensure!(
                same_outcome,
                "reservation was already settled with a different usage outcome"
            );
            tx.commit()?;
            return Ok(
                serde_json::json!({"reservation_id":reservation_id,"state":record.4,"reserved_credits":record.2,"charged_credits":record.3}),
            );
        }
        let charged = actual_credits.unwrap_or(record.2);
        anyhow::ensure!(
            charged <= record.2,
            "reported model usage exceeded its pre-call credit reservation"
        );
        let refund = record.2 - charged;
        if refund > 0 {
            let updated = tx.execute(
                "UPDATE sessions SET credits_used=credits_used-?2 WHERE id=?1 AND credits_used>=?2",
                params![session_id, refund],
            )?;
            anyhow::ensure!(
                updated == 1,
                "session credit balance is inconsistent with the reservation"
            );
            let agent = scientific_agent_for_task(&record.0)
                .context("research task has no registered scientific agent")?;
            tx.execute(
                "UPDATE specialist_credit_usage SET credits_used=credits_used-?3,updated_at=?4 WHERE session_id=?1 AND agent_id=?2 AND credits_used>=?3",
                params![session_id, agent, refund, timestamp],
            )?;
        }
        let state = if actual_credits.is_some() {
            "settled"
        } else {
            "uncertain"
        };
        tx.execute(
            "UPDATE model_budget_reservations SET charged_credits=?3,state=?4,settled_at=?5 WHERE session_id=?1 AND id=?2 AND state='reserved'",
            params![session_id, reservation_id, charged, state, timestamp],
        )?;
        tx.commit()?;
        Ok(
            serde_json::json!({"reservation_id":reservation_id,"state":state,"reserved_credits":record.2,"charged_credits":charged,"released_credits":refund}),
        )
    }

    pub fn model_reservation_task_id(
        &self,
        session_id: &str,
        reservation_id: &str,
    ) -> Result<Option<String>> {
        self.db
            .query_row(
                "SELECT task_id FROM model_budget_reservations WHERE session_id=?1 AND id=?2",
                params![session_id, reservation_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn consume_credits(
        &self,
        session_id: &str,
        task_id: &str,
        credits: i64,
        reason: &str,
    ) -> Result<i64> {
        anyhow::ensure!(credits > 0, "credit charge must be positive");
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 256,
            "budget reason must contain 1..256 characters"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let task_status: String = tx.query_row(
            "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
            params![session_id, task_id],
            |r| r.get(0),
        )?;
        anyhow::ensure!(
            task_status == "running",
            "credits can only be charged to a running research task"
        );
        let agent_id = scientific_agent_for_task(task_id)
            .context("research task has no registered scientific agent")?;
        anyhow::ensure!(
            specialist_budget_charge(&tx, session_id, agent_id, credits, now)?,
            "specialist credit budget exhausted for {agent_id}"
        );
        let changed = tx.execute(
            "UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget",
            params![session_id, credits],
        )?;
        anyhow::ensure!(
            changed == 1,
            "research session is missing or has insufficient execution credits"
        );
        let used: i64 = tx.query_row(
            "SELECT credits_used FROM sessions WHERE id=?1",
            [session_id],
            |r| r.get(0),
        )?;
        tx.execute(
            "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,?3,?4,?5)",
            params![session_id, task_id, credits, reason, now],
        )?;
        tx.commit()?;
        Ok(used)
    }

    /// Atomically meters one session-scoped API request against its execution budget.
    /// This ledger is separate from scientific task charges because ordinary API
    /// reads and writes do not necessarily belong to a running DAG task.
    pub fn consume_api_request_credits(
        &self,
        session_id: &str,
        credits: i64,
        method: &str,
        route: &str,
    ) -> Result<i64> {
        anyhow::ensure!(credits > 0, "API request credit charge must be positive");
        anyhow::ensure!(
            matches!(method, "GET" | "POST" | "PUT" | "PATCH" | "DELETE"),
            "unsupported API request method"
        );
        anyhow::ensure!(
            route.starts_with("/v1/research/") && route.len() <= 256,
            "API request route is outside the session API or too long"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let changed = tx.execute(
            "UPDATE sessions SET credits_used=credits_used+?2 WHERE id=?1 AND credits_used+?2<=budget",
            params![session_id, credits],
        )?;
        anyhow::ensure!(
            changed == 1,
            "research session is missing or has insufficient execution credits"
        );
        let used: i64 = tx.query_row(
            "SELECT credits_used FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,'api_request',?2,?3,?4)",
            params![session_id, credits, format!("session API request {method} {route}"), now],
        )?;
        tx.commit()?;
        Ok(used)
    }
    pub fn budget_events(&self, session_id: &str) -> Result<Vec<BudgetEvent>> {
        let mut q = self.db.prepare(
            "SELECT id,session_id,task_id,credits,reason,occurred_at FROM budget_events WHERE session_id=?1 ORDER BY id",
        )?;
        Ok(q.query_map([session_id], |r| {
            Ok(BudgetEvent {
                id: r.get(0)?,
                session_id: r.get(1)?,
                task_id: r.get(2)?,
                credits: r.get(3)?,
                reason: r.get(4)?,
                occurred_at: r.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn budget_summary(&self, session_id: &str) -> Result<serde_json::Value> {
        let (limit, used): (i64, i64) = self.db.query_row(
            "SELECT budget,credits_used FROM sessions WHERE id=?1",
            [session_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let events = self.budget_events(session_id)?;
        let model_reservations: Vec<serde_json::Value> = {
            let mut query = self.db.prepare("SELECT id,task_id,actor,reserved_credits,charged_credits,state,reason,created_at,settled_at FROM model_budget_reservations WHERE session_id=?1 ORDER BY created_at,id")?;
            query.query_map([session_id], |row| Ok(serde_json::json!({"id":row.get::<_,String>(0)?,"task_id":row.get::<_,String>(1)?,"actor":row.get::<_,String>(2)?,"reserved_credits":row.get::<_,i64>(3)?,"charged_credits":row.get::<_,Option<i64>>(4)?,"state":row.get::<_,String>(5)?,"reason":row.get::<_,String>(6)?,"created_at":row.get::<_,i64>(7)?,"settled_at":row.get::<_,Option<i64>>(8)?})))?.collect::<rusqlite::Result<_>>()?
        };
        let specialist_credits: Vec<serde_json::Value> = {
            let mut query = self.db.prepare(
                "SELECT agent_id,credits_used,credit_limit FROM specialist_credit_usage WHERE session_id=?1
                 UNION ALL
                 SELECT policy.agent_id,0,policy.credit_limit FROM specialist_credit_policies AS policy
                 LEFT JOIN specialist_credit_usage AS usage ON usage.session_id=?1 AND usage.agent_id=policy.agent_id
                 WHERE usage.agent_id IS NULL
                 ORDER BY agent_id",
            )?;
            query
                .query_map([session_id], |row| {
                    Ok(serde_json::json!({
                        "agent_id": row.get::<_, String>(0)?,
                        "used": row.get::<_, i64>(1)?,
                        "limit": row.get::<_, i64>(2)?,
                    }))
                })?
                .collect::<rusqlite::Result<_>>()?
        };
        let (storage_bytes, storage_credits): (i64, i64) = self.db.query_row(
            "SELECT COALESCE((SELECT bytes_used FROM artifact_storage_usage WHERE session_id=?1),0),COALESCE((SELECT credits_charged FROM artifact_storage_usage WHERE session_id=?1),0)",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(serde_json::json!({
            "session_id": session_id,
            "limit": limit,
            "used": used,
            "remaining": limit - used,
            "specialists": specialist_credits,
            "model_reservations": model_reservations,
            "storage": {
                "artifact_payload_bytes": storage_bytes,
                "credits_charged": storage_credits,
                "bytes_per_credit": ARTIFACT_STORAGE_BYTES_PER_CREDIT
            },
            "events": events
        }))
    }
    /// Apply a hash-verified model objective plan while the Research Director owns
    /// objective.normalize. Unselected ready/blocked DAG nodes are skipped atomically
    /// before any specialist can claim and spend credits on them.
    pub fn apply_research_task_scope(
        &self,
        session_id: &str,
        worker_id: &str,
        plan_artifact_id: &str,
    ) -> Result<Vec<String>> {
        validate_worker_id(worker_id)?;
        anyhow::ensure!(
            !plan_artifact_id.is_empty() && plan_artifact_id.len() <= 128,
            "invalid objective plan artifact ID"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let (objective, depth, cancellation_requested): (String, String, i64) = tx.query_row(
            "SELECT objective,depth,cancellation_requested FROM sessions WHERE id=?1",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        anyhow::ensure!(
            cancellation_requested == 0,
            "cannot apply objective task scope to a cancelled session"
        );
        let (task_status, lease_owner, lease_until): (String, Option<String>, Option<i64>) = tx.query_row(
            "SELECT status,lease_owner,lease_until FROM research_tasks WHERE session_id=?1 AND id='objective.normalize'",
            [session_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        anyhow::ensure!(
            task_status == "running"
                && lease_owner.as_deref() == Some(worker_id)
                && lease_until.is_some_and(|until| until > now),
            "live objective.normalize lease ownership is required to apply task scope"
        );
        let (artifact_task, artifact_kind, artifact_source, artifact_source_id, payload, claimed_sha256): (String, String, String, String, String, String) = tx.query_row(
            "SELECT task_id,kind,source,source_id,payload,sha256 FROM research_artifacts WHERE session_id=?1 AND id=?2",
            params![session_id, plan_artifact_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?)),
        )?;
        anyhow::ensure!(
            artifact_task == "objective.normalize"
                && artifact_kind == "research_objective_plan"
                && artifact_source == "MedAI configured model planner",
            "objective plan artifact has an unexpected source or task"
        );
        let actual_sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
        anyhow::ensure!(
            actual_sha256 == claimed_sha256,
            "objective plan artifact payload hash does not match"
        );
        let envelope: serde_json::Value = serde_json::from_str(&payload)?;
        let objective_sha256 = format!("{:x}", Sha256::digest(objective.trim().as_bytes()));
        anyhow::ensure!(
            envelope
                .get("schema_version")
                .and_then(serde_json::Value::as_str)
                == Some("2"),
            "unsupported objective plan schema version"
        );
        anyhow::ensure!(
            envelope
                .get("objective_sha256")
                .and_then(serde_json::Value::as_str)
                == Some(objective_sha256.as_str())
                && artifact_source_id == objective_sha256,
            "objective plan is bound to another research objective"
        );
        anyhow::ensure!(
            envelope.get("depth").and_then(serde_json::Value::as_str) == Some(depth.as_str()),
            "objective plan is bound to another research depth"
        );
        let selected: Vec<String> = serde_json::from_value(
            envelope
                .pointer("/planning/selected_task_ids")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("objective plan is missing selected task IDs"))?,
        )?;
        let selected_set: std::collections::HashSet<&str> =
            selected.iter().map(String::as_str).collect();
        anyhow::ensure!(
            selected_set.len() == selected.len(),
            "objective plan contains duplicate task IDs"
        );
        let registered: std::collections::HashSet<&str> =
            RESEARCH_TASKS.iter().map(|row| row.0).collect();
        anyhow::ensure!(
            selected_set
                .iter()
                .all(|task_id| registered.contains(task_id)),
            "objective plan contains an unregistered task ID"
        );
        for mandatory in [
            "objective.normalize",
            "literature.discover",
            "literature.extract",
            "entities.resolve",
            "evidence.grade",
            "contradictions.search",
            "graph.build",
            "hypotheses.propose",
            "hypotheses.criticize",
            "report.reproduce",
        ] {
            anyhow::ensure!(
                selected_set.contains(mandatory),
                "objective plan omits mandatory task {mandatory}"
            );
        }
        for (task_id, _, _, raw_dependencies) in RESEARCH_TASKS {
            if !selected_set.contains(task_id)
                || matches!(*task_id, "graph.build" | "report.reproduce")
            {
                continue;
            }
            let dependencies: Vec<String> = serde_json::from_str(raw_dependencies)?;
            anyhow::ensure!(
                dependencies
                    .iter()
                    .all(|dependency| selected_set.contains(dependency.as_str())),
                "objective plan omits a prerequisite for {task_id}"
            );
        }
        let selected_value = serde_json::to_value(&selected)?;
        let prior_scope: Option<String> = tx.query_row(
            "SELECT details FROM task_events WHERE session_id=?1 AND task_id='objective.normalize' AND details LIKE 'objective task scope applied:%' ORDER BY id DESC LIMIT 1",
            [session_id],
            |row| row.get(0),
        ).optional()?;
        if let Some(details) = prior_scope {
            let receipt: serde_json::Value = serde_json::from_str(
                details
                    .strip_prefix("objective task scope applied:")
                    .unwrap_or(""),
            )?;
            anyhow::ensure!(
                receipt
                    .get("selected_task_ids")
                    .is_some_and(|prior| prior == &selected_value),
                "a different research task scope was already applied"
            );
            let skipped = serde_json::from_value(
                receipt
                    .get("skipped_task_ids")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])),
            )?;
            tx.commit()?;
            return Ok(skipped);
        }
        let mut skipped = Vec::new();
        for (task_id, _, _, _) in RESEARCH_TASKS {
            if selected_set.contains(task_id) {
                continue;
            }
            let status: String = tx.query_row(
                "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
                params![session_id, task_id],
                |row| row.get(0),
            )?;
            anyhow::ensure!(
                matches!(status.as_str(), "ready" | "blocked"),
                "unselected task {task_id} already started or completed"
            );
            let details = serde_json::json!({
                "reason": "Excluded by the hash-verified objective-specific research plan.",
                "plan_artifact_id": plan_artifact_id,
                "plan_sha256": actual_sha256,
                "selected_task_ids": selected,
            })
            .to_string();
            tx.execute(
                "UPDATE research_tasks SET status='skipped',output=?3,error=NULL,lease_owner=NULL,lease_until=NULL WHERE session_id=?1 AND id=?2",
                params![session_id, task_id, details],
            )?;
            tx.execute(
                "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,?3,'skipped',?4,?5)",
                params![session_id, task_id, status, now, format!("objective task scope applied from artifact {plan_artifact_id}")],
            )?;
            skipped.push((*task_id).to_owned());
        }
        let scope_receipt = serde_json::json!({
            "plan_artifact_id": plan_artifact_id,
            "plan_sha256": actual_sha256,
            "selected_task_ids": selected,
            "skipped_task_ids": skipped,
        });
        tx.execute(
            "INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,'objective.normalize','running','running',?2,?3)",
            params![session_id, now, format!("objective task scope applied:{scope_receipt}")],
        )?;
        tx.commit()?;
        Ok(skipped)
    }
    /// Transitions a task through the audited state machine and unlocks ready dependants.
    pub fn transition_task(
        &self,
        session_id: &str,
        task_id: &str,
        worker_id: Option<&str>,
        next: &str,
        output: Option<&str>,
        error: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(
            [
                "ready",
                "running",
                "succeeded",
                "failed",
                "needs_review",
                "skipped"
            ]
            .contains(&next),
            "invalid task state"
        );
        anyhow::ensure!(
            output.is_none_or(|value| value.len() <= 16_384),
            "task output exceeds 16 KiB"
        );
        anyhow::ensure!(
            error.is_none_or(|value| value.len() <= 4_096),
            "task error exceeds 4 KiB"
        );
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let tx = self.db.unchecked_transaction()?;
        let (current, lease_owner, lease_until, attempts): (String, Option<String>, Option<i64>, i64) = tx.query_row(
            "SELECT status,lease_owner,lease_until,attempts FROM research_tasks WHERE session_id=?1 AND id=?2",
            params![session_id, task_id],
            |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)),
        )?;
        let cancellation_requested: i64 = tx.query_row(
            "SELECT cancellation_requested FROM sessions WHERE id=?1",
            [session_id],
            |row| row.get(0),
        )?;
        if cancellation_requested != 0 {
            anyhow::ensure!(
                matches!(
                    (current.as_str(), next),
                    ("running", "skipped") | ("ready", "skipped") | ("blocked", "skipped")
                ),
                "research session cancellation prevents further task execution or completion"
            );
        }
        if let Some(owner) = lease_owner.as_deref() {
            anyhow::ensure!(
                worker_id == Some(owner),
                "task lease is owned by another worker"
            );
            if let Some(worker_id) = worker_id {
                validate_worker_id(worker_id)?;
            }
            anyhow::ensure!(
                lease_until.is_some_and(|until| until > now),
                "task lease has expired"
            );
        }
        let allowed = match (current.as_str(), next) {
            ("ready", "running" | "skipped") => true,
            ("running", "succeeded" | "failed" | "needs_review") => cancellation_requested == 0,
            ("running", "skipped") => cancellation_requested != 0,
            ("failed", "ready" | "skipped") => true,
            ("needs_review", "succeeded" | "failed") => true,
            _ => false,
        };
        if current == "failed" && next == "ready" {
            anyhow::ensure!(
                attempts < 5,
                "task has reached the maximum of 5 execution attempts"
            );
        }
        anyhow::ensure!(
            allowed,
            "task transition {current} -> {next} is not allowed"
        );
        if next == "running" {
            let dependencies: String = tx.query_row(
                "SELECT depends_on FROM research_tasks WHERE session_id=?1 AND id=?2",
                params![session_id, task_id],
                |r| r.get(0),
            )?;
            let required: Vec<String> = serde_json::from_str(&dependencies)?;
            for dependency in required {
                let status: String = tx.query_row(
                    "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
                    params![session_id, dependency],
                    |r| r.get(0),
                )?;
                anyhow::ensure!(
                    status == "succeeded" || status == "skipped",
                    "dependency {dependency} has not succeeded or been skipped"
                );
            }
            let charged = tx.execute(
                "UPDATE sessions SET credits_used=credits_used+1 WHERE id=?1 AND credits_used<budget",
                [session_id],
            )?;
            if charged != 1 {
                tx.execute(
                    "UPDATE sessions SET status='budget_exhausted' WHERE id=?1",
                    [session_id],
                )?;
                tx.commit()?;
                anyhow::bail!("research session has insufficient execution credits");
            }
            tx.execute(
                "INSERT INTO budget_events(session_id,task_id,credits,reason,occurred_at) VALUES(?1,?2,1,'task execution started',?3)",
                params![session_id, task_id, now],
            )?;
        }
        tx.execute("UPDATE research_tasks SET status=?3,attempts=attempts+CASE WHEN ?3='running' THEN 1 ELSE 0 END,output=CASE WHEN ?3='running' THEN NULL ELSE COALESCE(?4,output) END,error=CASE WHEN ?3 IN ('ready','running') THEN NULL ELSE ?5 END,lease_owner=CASE WHEN ?3='running' THEN lease_owner ELSE NULL END,lease_until=CASE WHEN ?3='running' THEN lease_until ELSE NULL END,available_at=CASE WHEN ?3='ready' THEN 0 ELSE available_at END WHERE session_id=?1 AND id=?2", params![session_id,task_id,next,output,error])?;
        let mut detail = output
            .map(|value| format!("output_recorded_bytes={}", value.len()))
            .or_else(|| error.map(str::to_owned));
        if let Some(worker_id) = worker_id {
            let worker_detail = format!("worker={worker_id}");
            detail = Some(match detail {
                Some(current) => format!("{current}; {worker_detail}"),
                None => worker_detail,
            });
        }
        tx.execute("INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,?3,?4,?5,?6)",params![session_id,task_id,current,next,now,detail])?;
        let blocked: Vec<(String, String)> = {
            let mut q = tx.prepare(
                "SELECT id,depends_on FROM research_tasks WHERE session_id=?1 AND status='blocked'",
            )?;
            q.query_map([session_id], |r| Ok((r.get(0)?, r.get(1)?)))?
                .collect::<rusqlite::Result<_>>()?
        };
        for (candidate, raw_dependencies) in blocked {
            let dependencies: Vec<String> = serde_json::from_str(&raw_dependencies)?;
            let mut ready = true;
            for dependency in dependencies {
                let state: String = tx.query_row(
                    "SELECT status FROM research_tasks WHERE session_id=?1 AND id=?2",
                    params![session_id, dependency],
                    |r| r.get(0),
                )?;
                if state != "succeeded" && state != "skipped" {
                    ready = false;
                    break;
                }
            }
            if ready {
                tx.execute(
                    "UPDATE research_tasks SET status='ready' WHERE session_id=?1 AND id=?2",
                    params![session_id, candidate],
                )?;
                tx.execute("INSERT INTO task_events(session_id,task_id,from_status,to_status,occurred_at,details) VALUES(?1,?2,'blocked','ready',?3,'dependencies satisfied')",params![session_id,candidate,now])?;
            }
        }
        let (remaining,failed,running,review,used,limit): (i64,i64,i64,i64,i64,i64) = tx.query_row("SELECT SUM(CASE WHEN status NOT IN ('succeeded','skipped') THEN 1 ELSE 0 END),SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),SUM(CASE WHEN status='running' THEN 1 ELSE 0 END),SUM(CASE WHEN status='needs_review' THEN 1 ELSE 0 END),(SELECT credits_used FROM sessions WHERE id=?1),(SELECT budget FROM sessions WHERE id=?1) FROM research_tasks WHERE session_id=?1",[session_id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?)))?;
        let session_status = if cancellation_requested != 0 {
            if running == 0 {
                "cancelled"
            } else {
                "cancelling"
            }
        } else if remaining == 0 {
            "completed"
        } else if running > 0 {
            "running"
        } else if used >= limit {
            "budget_exhausted"
        } else if review > 0 {
            "needs_review"
        } else if failed > 0 {
            "failed"
        } else {
            "planned"
        };
        tx.execute(
            "UPDATE sessions SET status=?2 WHERE id=?1",
            params![session_id, session_status],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn add_evidence(&self, e: &Evidence) -> Result<()> {
        anyhow::ensure!(!e.claim.trim().is_empty(), "claim cannot be empty");
        anyhow::ensure!(!e.source.trim().is_empty(), "source cannot be empty");
        anyhow::ensure!((0.0..=1.0).contains(&e.quality), "quality must be 0..1");
        anyhow::ensure!(
            ["supports", "contradicts", "context"].contains(&e.polarity.as_str()),
            "polarity must be supports, contradicts, or context"
        );
        let provenance: serde_json::Value =
            serde_json::from_str(&e.provenance).context("provenance must be valid JSON")?;
        anyhow::ensure!(provenance.is_object(), "provenance must be a JSON object");
        self.db.execute(
            "INSERT INTO evidence VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                e.id,
                e.session_id,
                e.claim,
                e.source,
                e.source_id,
                e.polarity,
                e.quality,
                e.provenance
            ],
        )?;
        Ok(())
    }
    pub fn evidence(&self, session: &str) -> Result<Vec<Evidence>> {
        let mut q=self.db.prepare("SELECT id,session_id,claim,source,source_id,polarity,quality,provenance FROM evidence WHERE session_id=?1 ORDER BY rowid")?;
        Ok(q.query_map([session], |r| {
            Ok(Evidence {
                id: r.get(0)?,
                session_id: r.get(1)?,
                claim: r.get(2)?,
                source: r.get(3)?,
                source_id: r.get(4)?,
                polarity: r.get(5)?,
                quality: r.get(6)?,
                provenance: r.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn put_node(&self, n: &GraphNode) -> Result<()> {
        anyhow::ensure!(
            !n.session_id.trim().is_empty(),
            "graph node requires a research session"
        );
        self.db.execute("INSERT INTO graph_nodes VALUES(?1,?2,?3,?4,?5) ON CONFLICT(session_id,id) DO UPDATE SET kind=excluded.kind,label=excluded.label,metadata=excluded.metadata",params![n.session_id,n.id,n.kind,n.label,n.metadata])?;
        Ok(())
    }
    pub fn put_edge(&self, e: &GraphEdge) -> Result<()> {
        anyhow::ensure!(
            !e.source.trim().is_empty()
                && !e.target.trim().is_empty()
                && !e.relation.trim().is_empty(),
            "edge source, relation, and target are required"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&e.confidence),
            "confidence must be 0..1"
        );
        let evidence_ids: Vec<String> = serde_json::from_str(&e.evidence_ids)
            .context("evidence_ids must be a JSON array of evidence identifiers")?;
        let artifact_ids: Vec<String> = serde_json::from_str(&e.artifact_ids)
            .context("artifact_ids must be a JSON array of research artifact identifiers")?;
        anyhow::ensure!(
            !evidence_ids.is_empty() || !artifact_ids.is_empty(),
            "graph relationships require at least one evidence record or source artifact"
        );
        let transaction = Transaction::new_unchecked(&self.db, TransactionBehavior::Immediate)?;
        let mut unique = std::collections::HashSet::new();
        for evidence_id in &evidence_ids {
            anyhow::ensure!(
                unique.insert(evidence_id),
                "duplicate evidence identifier {evidence_id}"
            );
            let exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM evidence WHERE id=?1 AND session_id=?2)",
                params![evidence_id, e.session_id],
                |r| r.get(0),
            )?;
            anyhow::ensure!(
                exists,
                "evidence record {evidence_id} does not exist in research session {}",
                e.session_id
            );
        }
        for artifact_id in &artifact_ids {
            anyhow::ensure!(
                unique.insert(artifact_id),
                "duplicate provenance identifier {artifact_id}"
            );
            let exists: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM research_artifacts WHERE id=?1 AND session_id=?2)",
                params![artifact_id, e.session_id],
                |row| row.get(0),
            )?;
            anyhow::ensure!(
                exists,
                "source artifact {artifact_id} does not exist in research session {}",
                e.session_id
            );
        }
        let (confidence, mut evidence_set, mut artifact_set) = match transaction
            .query_row(
                "SELECT confidence,evidence_ids,artifact_ids FROM graph_edges WHERE session_id=?1 AND source=?2 AND relation=?3 AND target=?4",
                params![e.session_id, e.source, e.relation, e.target],
                |row| Ok((row.get::<_, f64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?)),
            )
            .optional()?
        {
            Some((existing_confidence, existing_evidence, existing_artifacts)) => {
                anyhow::ensure!(
                    existing_confidence == e.confidence,
                    "graph edge confidence is immutable; represent source-specific confidence on separate assertion nodes"
                );
                let existing_evidence: Vec<String> = serde_json::from_str(&existing_evidence)
                    .context("stored graph edge evidence identifiers are invalid JSON")?;
                let existing_artifacts: Vec<String> = serde_json::from_str(&existing_artifacts)
                    .context("stored graph edge artifact identifiers are invalid JSON")?;
                let mut existing_unique = std::collections::HashSet::new();
                anyhow::ensure!(
                    existing_evidence
                        .iter()
                        .chain(existing_artifacts.iter())
                        .all(|identifier| existing_unique.insert(identifier)),
                    "stored graph edge has duplicate provenance identifiers"
                );
                (
                    existing_confidence,
                    existing_evidence.into_iter().collect::<std::collections::BTreeSet<_>>(),
                    existing_artifacts.into_iter().collect::<std::collections::BTreeSet<_>>(),
                )
            }
            None => (
                e.confidence,
                std::collections::BTreeSet::new(),
                std::collections::BTreeSet::new(),
            ),
        };
        evidence_set.extend(evidence_ids);
        artifact_set.extend(artifact_ids);
        let evidence_json = serde_json::to_string(&evidence_set)?;
        let artifacts_json = serde_json::to_string(&artifact_set)?;
        transaction.execute("INSERT INTO graph_edges(session_id,source,relation,target,confidence,evidence_ids,artifact_ids) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(session_id,source,relation,target) DO UPDATE SET evidence_ids=excluded.evidence_ids,artifact_ids=excluded.artifact_ids",params![e.session_id,e.source,e.relation,e.target,confidence,evidence_json,artifacts_json])?;
        transaction.commit()?;
        Ok(())
    }
    pub fn graph(&self, session_id: &str) -> Result<(Vec<GraphNode>, Vec<GraphEdge>)> {
        let mut n = self
            .db
            .prepare("SELECT session_id,id,kind,label,metadata FROM graph_nodes WHERE session_id=?1 ORDER BY label")?;
        let nodes = n
            .query_map([session_id], |r| {
                Ok(GraphNode {
                    session_id: r.get(0)?,
                    id: r.get(1)?,
                    kind: r.get(2)?,
                    label: r.get(3)?,
                    metadata: r.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut e=self.db.prepare("SELECT session_id,source,relation,target,confidence,evidence_ids,artifact_ids FROM graph_edges WHERE session_id=?1 ORDER BY source")?;
        let edges = e
            .query_map([session_id], |r| {
                Ok(GraphEdge {
                    session_id: r.get(0)?,
                    source: r.get(1)?,
                    relation: r.get(2)?,
                    target: r.get(3)?,
                    confidence: r.get(4)?,
                    evidence_ids: r.get(5)?,
                    artifact_ids: r.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok((nodes, edges))
    }
    pub fn register_capability(&self, c: &Capability) -> Result<()> {
        self.db.execute("INSERT INTO capabilities VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(id,version) DO UPDATE SET status=excluded.status,description=excluded.description,manifest=excluded.manifest",params![c.id,c.version,c.kind,c.status,c.description,c.manifest])?;
        Ok(())
    }
    /// Store an immutable, non-executable capability proposal for operator review.
    pub fn propose_capability(&self, c: &Capability) -> Result<()> {
        self.propose_capability_as(c, "capability-builder", "proposal created disabled")
    }
    /// Persist a disabled immutable proposal with the authenticated submitting actor.
    pub fn propose_capability_as(&self, c: &Capability, actor: &str, reason: &str) -> Result<()> {
        anyhow::ensure!(
            c.status == "pending_review",
            "new capability versions must start pending_review"
        );
        anyhow::ensure!(
            !actor.trim().is_empty() && actor.len() <= 200,
            "a named proposal submitter is required"
        );
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 2000,
            "a proposal rationale is required"
        );
        anyhow::ensure!(
            c.id.len() <= 200 && !c.id.trim().is_empty(),
            "capability id must be 1..200 characters"
        );
        anyhow::ensure!(
            c.version.len() <= 64 && !c.version.trim().is_empty(),
            "capability version must be 1..64 characters"
        );
        anyhow::ensure!(
            matches!(
                c.kind.as_str(),
                "mcp"
                    | "skill"
                    | "workflow"
                    | "tool"
                    | "model"
                    | "data_source"
                    | "parser"
                    | "algorithm"
                    | "pipeline"
            ),
            "unsupported capability kind"
        );
        let manifest: serde_json::Value =
            serde_json::from_str(&c.manifest).context("capability manifest must be valid JSON")?;
        anyhow::ensure!(
            manifest.is_object(),
            "capability manifest must be a JSON object"
        );
        let safety = manifest.get("execution").and_then(|v| v.as_object());
        anyhow::ensure!(
            safety
                .and_then(|v| v.get("enabled"))
                .and_then(|v| v.as_bool())
                == Some(false),
            "proposals must explicitly disable execution"
        );
        anyhow::ensure!(
            safety
                .and_then(|v| v.get("network"))
                .and_then(|v| v.as_bool())
                == Some(false),
            "proposals must explicitly deny network access"
        );
        let tx = self.db.unchecked_transaction()?;
        let occurred_at = now()?;
        tx.execute("INSERT INTO capabilities(id,version,kind,status,description,manifest) VALUES(?1,?2,?3,'pending_review',?4,?5)", params![c.id,c.version,c.kind,c.description,c.manifest])?;
        tx.execute("INSERT INTO capability_events(capability_id,version,from_status,to_status,actor,reason,occurred_at) VALUES(?1,?2,NULL,'pending_review',?3,?4,?5)", params![c.id,c.version,actor,reason.trim(),occurred_at])?;
        tx.commit()?;
        Ok(())
    }
    /// Apply an audited lifecycle transition. Promotion requires a JSON report with all checks true.
    pub fn transition_capability(
        &self,
        id: &str,
        version: &str,
        next: &str,
        actor: &str,
        reason: &str,
        validation: Option<&str>,
    ) -> Result<()> {
        anyhow::ensure!(
            !actor.trim().is_empty() && actor.len() <= 200,
            "a named reviewer/operator is required"
        );
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 2000,
            "a transition reason is required"
        );
        anyhow::ensure!(
            matches!(next, "verified" | "production" | "deprecated"),
            "unsupported capability transition"
        );
        let (current, manifest): (String, String) = self
            .db
            .query_row(
                "SELECT status,manifest FROM capabilities WHERE id=?1 AND version=?2",
                params![id, version],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .context("capability version not found")?;
        let valid = matches!(
            (current.as_str(), next),
            ("pending_review", "verified")
                | ("verified", "production")
                | ("production", "deprecated")
                | ("verified", "deprecated")
                | ("pending_review", "deprecated")
        );
        anyhow::ensure!(
            valid,
            "invalid capability lifecycle transition: {current} -> {next}"
        );
        let validation_hash = if next == "verified" || next == "production" {
            let report = validation.context("promotion requires a validation report")?;
            if next == "production" {
                validate_paired_benchmark_artifact(&self.db, id, version, &manifest, report)?;
            }
            Some(validation_report_hash(
                report, id, version, &manifest, next,
            )?)
        } else {
            None
        };
        let tx = self.db.unchecked_transaction()?;
        let occurred_at = now()?;
        if next == "production" {
            let other_production: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM capabilities WHERE id=?1 AND status='production' AND version!=?2)", params![id,version], |r| r.get(0))?;
            anyhow::ensure!(
                !other_production,
                "another version is already production; rollback or deprecate it first"
            );
        }
        let changed = tx.execute(
            "UPDATE capabilities SET status=?3 WHERE id=?1 AND version=?2 AND status=?4",
            params![id, version, next, current],
        )?;
        anyhow::ensure!(changed == 1, "capability status changed concurrently");
        tx.execute("INSERT INTO capability_events(capability_id,version,from_status,to_status,actor,reason,validation_sha256,occurred_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![id,version,current,next,actor,reason,validation_hash,occurred_at])?;
        tx.commit()?;
        Ok(())
    }
    /// Atomically retire the active production version and restore a previously verified version.
    pub fn rollback_capability(
        &self,
        id: &str,
        target_version: &str,
        actor: &str,
        reason: &str,
        validation: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            !actor.trim().is_empty() && actor.len() <= 200,
            "a named reviewer/operator is required"
        );
        anyhow::ensure!(
            !reason.trim().is_empty() && reason.len() <= 2000,
            "a rollback reason is required"
        );
        let tx = self.db.unchecked_transaction()?;
        let active: String = tx
            .query_row(
                "SELECT version FROM capabilities WHERE id=?1 AND status='production'",
                [id],
                |r| r.get(0),
            )
            .context("no production version exists for this capability")?;
        anyhow::ensure!(
            active != target_version,
            "rollback target is already the active production version"
        );
        let (target_status, target_manifest): (String, String) = tx
            .query_row(
                "SELECT status,manifest FROM capabilities WHERE id=?1 AND version=?2",
                params![id, target_version],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .context("rollback target version not found")?;
        anyhow::ensure!(
            target_status == "verified",
            "rollback target must be verified (found {target_status})"
        );
        validate_paired_benchmark_artifact(&tx, id, target_version, &target_manifest, validation)?;
        let report_hash = validation_report_hash(
            validation,
            id,
            target_version,
            &target_manifest,
            "production",
        )?;
        let occurred_at = now()?;
        anyhow::ensure!(tx.execute("UPDATE capabilities SET status='rolled_back' WHERE id=?1 AND version=?2 AND status='production'", params![id,active])? == 1, "active production version changed concurrently");
        anyhow::ensure!(tx.execute("UPDATE capabilities SET status='production' WHERE id=?1 AND version=?2 AND status='verified'", params![id,target_version])? == 1, "rollback target changed concurrently");
        tx.execute("INSERT INTO capability_events(capability_id,version,from_status,to_status,actor,reason,occurred_at) VALUES(?1,?2,'production','rolled_back',?3,?4,?5)", params![id,active,actor,reason,occurred_at])?;
        tx.execute("INSERT INTO capability_events(capability_id,version,from_status,to_status,actor,reason,validation_sha256,occurred_at) VALUES(?1,?2,'verified','production',?3,?4,?5,?6)", params![id,target_version,actor,reason,report_hash,occurred_at])?;
        tx.commit()?;
        Ok(())
    }
    pub fn capability_events(&self, id: &str, version: &str) -> Result<Vec<serde_json::Value>> {
        let mut q = self.db.prepare("SELECT from_status,to_status,actor,reason,validation_sha256,occurred_at FROM capability_events WHERE capability_id=?1 AND version=?2 ORDER BY event_id")?;
        Ok(q.query_map(params![id,version], |r| Ok(serde_json::json!({"from":r.get::<_,Option<String>>(0)?,"to":r.get::<_,String>(1)?,"actor":r.get::<_,String>(2)?,"reason":r.get::<_,String>(3)?,"validation_sha256":r.get::<_,Option<String>>(4)?,"occurred_at":r.get::<_,i64>(5)?})))?.collect::<rusqlite::Result<_>>()?)
    }
    pub fn capabilities(&self) -> Result<Vec<Capability>> {
        let mut q=self.db.prepare("SELECT id,version,kind,status,description,manifest FROM capabilities ORDER BY id,version")?;
        Ok(q.query_map([], |r| {
            Ok(Capability {
                id: r.get(0)?,
                version: r.get(1)?,
                kind: r.get(2)?,
                status: r.get(3)?,
                description: r.get(4)?,
                manifest: r.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub fn summary(&self) -> Result<serde_json::Value> {
        let sessions: i64 = self.db.query_row(
            "SELECT count(*) FROM sessions WHERE id!='__legacy__'",
            [],
            |r| r.get(0),
        )?;
        let evidence: i64 = self
            .db
            .query_row("SELECT count(*) FROM evidence", [], |r| r.get(0))?;
        let nodes: i64 = self.db.query_row(
            "SELECT count(*) FROM graph_nodes WHERE session_id!='__legacy__'",
            [],
            |r| r.get(0),
        )?;
        let capabilities: i64 =
            self.db
                .query_row("SELECT count(*) FROM capabilities", [], |r| r.get(0))?;
        let artifacts: i64 =
            self.db
                .query_row("SELECT count(*) FROM research_artifacts", [], |r| r.get(0))?;
        Ok(
            serde_json::json!({"sessions":sessions,"evidence":evidence,"graph_nodes":nodes,"capabilities":capabilities,"artifacts":artifacts}),
        )
    }
    pub fn summary_for_tenant(&self, tenant_id: &str) -> Result<serde_json::Value> {
        let sessions: i64 = self.db.query_row(
            "SELECT count(*) FROM sessions WHERE id!='__legacy__' AND tenant_id=?1",
            [tenant_id],
            |row| row.get(0),
        )?;
        let evidence: i64 = self.db.query_row("SELECT count(*) FROM evidence JOIN sessions ON sessions.id=evidence.session_id WHERE sessions.tenant_id=?1", [tenant_id], |row| row.get(0))?;
        let nodes: i64 = self.db.query_row("SELECT count(*) FROM graph_nodes JOIN sessions ON sessions.id=graph_nodes.session_id WHERE sessions.tenant_id=?1 AND sessions.id!='__legacy__'", [tenant_id], |row| row.get(0))?;
        let artifacts: i64 = self.db.query_row("SELECT count(*) FROM research_artifacts JOIN sessions ON sessions.id=research_artifacts.session_id WHERE sessions.tenant_id=?1", [tenant_id], |row| row.get(0))?;
        let capabilities: i64 =
            self.db
                .query_row("SELECT count(*) FROM capabilities", [], |row| row.get(0))?;
        Ok(
            serde_json::json!({"sessions":sessions,"evidence":evidence,"graph_nodes":nodes,"capabilities":capabilities,"artifacts":artifacts}),
        )
    }
    pub fn create_agentmesh_route_approval(
        &mut self,
        actor: &str,
        medai_tenant_id: &str,
        agentmesh_tenant: &str,
        agentmesh_namespace: &str,
        plan_review_sha256: &str,
        desired_sha256: &str,
    ) -> Result<AgentMeshRouteApproval> {
        let valid_hash = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        };
        anyhow::ensure!(
            !actor.is_empty() && actor.len() <= 128,
            "approval actor is invalid"
        );
        anyhow::ensure!(
            !medai_tenant_id.is_empty() && medai_tenant_id.len() <= 128,
            "MedAI tenant is invalid"
        );
        anyhow::ensure!(
            !agentmesh_tenant.is_empty() && agentmesh_tenant.len() <= 128,
            "AgentMesh tenant is invalid"
        );
        anyhow::ensure!(
            !agentmesh_namespace.is_empty() && agentmesh_namespace.len() <= 128,
            "AgentMesh namespace is invalid"
        );
        anyhow::ensure!(
            valid_hash(plan_review_sha256) && valid_hash(desired_sha256),
            "route plan digests must be lowercase SHA-256 values"
        );
        let issued_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        let approval = AgentMeshRouteApproval {
            receipt_id: uuid::Uuid::new_v4().to_string(),
            actor: actor.to_owned(),
            medai_tenant_id: medai_tenant_id.to_owned(),
            agentmesh_tenant: agentmesh_tenant.to_owned(),
            agentmesh_namespace: agentmesh_namespace.to_owned(),
            plan_review_sha256: plan_review_sha256.to_owned(),
            desired_sha256: desired_sha256.to_owned(),
            issued_at,
            expires_at: issued_at + 600,
        };
        self.db.execute(
            "INSERT INTO agentmesh_route_approvals(receipt_id,actor,medai_tenant_id,agentmesh_tenant,agentmesh_namespace,plan_review_sha256,desired_sha256,issued_at,expires_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![approval.receipt_id, approval.actor, approval.medai_tenant_id, approval.agentmesh_tenant, approval.agentmesh_namespace, approval.plan_review_sha256, approval.desired_sha256, approval.issued_at, approval.expires_at],
        )?;
        Ok(approval)
    }
    pub fn validate_agentmesh_route_approval(
        &self,
        receipt_id: &str,
        agentmesh_tenant: &str,
        agentmesh_namespace: &str,
        plan_review_sha256: &str,
        desired_sha256: &str,
    ) -> Result<Option<AgentMeshRouteApproval>> {
        let approval = self.db.query_row(
            "SELECT receipt_id,actor,medai_tenant_id,agentmesh_tenant,agentmesh_namespace,plan_review_sha256,desired_sha256,issued_at,expires_at FROM agentmesh_route_approvals WHERE receipt_id=?1",
            [receipt_id],
            |row| Ok(AgentMeshRouteApproval { receipt_id: row.get(0)?, actor: row.get(1)?, medai_tenant_id: row.get(2)?, agentmesh_tenant: row.get(3)?, agentmesh_namespace: row.get(4)?, plan_review_sha256: row.get(5)?, desired_sha256: row.get(6)?, issued_at: row.get(7)?, expires_at: row.get(8)? }),
        ).optional()?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
        Ok(approval.filter(|item| {
            item.expires_at >= now
                && item.agentmesh_tenant == agentmesh_tenant
                && item.agentmesh_namespace == agentmesh_namespace
                && item.plan_review_sha256 == plan_review_sha256
                && item.desired_sha256 == desired_sha256
        }))
    }
    pub fn agentmesh_route_approval(
        &self,
        receipt_id: &str,
    ) -> Result<Option<AgentMeshRouteApproval>> {
        Ok(self.db.query_row(
            "SELECT receipt_id,actor,medai_tenant_id,agentmesh_tenant,agentmesh_namespace,plan_review_sha256,desired_sha256,issued_at,expires_at FROM agentmesh_route_approvals WHERE receipt_id=?1",
            [receipt_id],
            |row| Ok(AgentMeshRouteApproval {
                receipt_id: row.get(0)?, actor: row.get(1)?, medai_tenant_id: row.get(2)?,
                agentmesh_tenant: row.get(3)?, agentmesh_namespace: row.get(4)?,
                plan_review_sha256: row.get(5)?, desired_sha256: row.get(6)?,
                issued_at: row.get(7)?, expires_at: row.get(8)?,
            }),
        ).optional()?)
    }
}

struct PrivateDirectory {
    path: PathBuf,
}

impl PrivateDirectory {
    fn create(parent: &Path, name: &str) -> Result<Self> {
        let path = parent.join(name);
        create_private_directory(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PrivateDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn stage_restore_snapshot(
    source: &Connection,
    destination: &Path,
) -> Result<DatabaseBackupReceipt> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .context("reserve private restore snapshot path")?;
    set_private_file(destination)?;
    let mut staged = Connection::open(destination).context("open private restore snapshot")?;
    staged.busy_timeout(Duration::from_secs(5))?;
    {
        let backup = rusqlite::backup::Backup::new(source, &mut staged)
            .context("initialize restore source capture")?;
        backup
            .run_to_completion(128, Duration::from_millis(20), None)
            .context("capture restore source into private snapshot")?;
    }
    check_database_integrity(&staged).context("validate captured restore snapshot")?;
    validate_restore_schema(&staged).context("captured restore schema is incompatible")?;
    drop(staged);
    File::open(destination)?.sync_all()?;
    Store::verify_backup(destination).context("hash and verify captured restore snapshot")
}

fn enable_wal_mode(connection: &Connection) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match connection.query_row("PRAGMA journal_mode=WAL", [], |row| row.get::<_, String>(0)) {
            Ok(mode) => {
                anyhow::ensure!(mode.eq_ignore_ascii_case("wal"), "SQLite refused WAL mode");
                return Ok(());
            }
            Err(rusqlite::Error::SqliteFailure(code, _))
                if matches!(
                    code.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                ) && Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return Err(error).context("enable SQLite WAL mode"),
        }
    }
}

fn copy_verified_database(destination: &mut Connection, source_path: &Path) -> Result<()> {
    let source = Connection::open_with_flags(
        source_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .context("open validated restore snapshot read-only")?;
    source.busy_timeout(Duration::from_secs(5))?;
    {
        let backup = rusqlite::backup::Backup::new(&source, destination)
            .context("initialize SQLite restore operation")?;
        backup
            .run_to_completion(128, Duration::from_millis(20), None)
            .context("copy SQLite restore snapshot")?;
    }
    drop(source);
    check_database_integrity(destination).context("validate restored database")?;
    Ok(())
}

fn check_database_integrity(connection: &Connection) -> Result<String> {
    let check: String = connection.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    anyhow::ensure!(check == "ok", "SQLite integrity check failed: {check}");
    let invalid_table: Option<String> = connection
        .query_row("PRAGMA foreign_key_check", [], |row| row.get(0))
        .optional()?;
    anyhow::ensure!(
        invalid_table.is_none(),
        "SQLite foreign-key check failed for table {}",
        invalid_table.as_deref().unwrap_or("unknown")
    );
    Ok(check)
}

fn ensure_standalone_sqlite_snapshot(path: &Path) -> Result<()> {
    let file_name = path
        .file_name()
        .context("SQLite snapshot path must name a file")?;
    for suffix in ["-wal", "-shm", "-journal"] {
        let mut sidecar_name = file_name.to_os_string();
        sidecar_name.push(suffix);
        let sidecar = path.with_file_name(sidecar_name);
        anyhow::ensure!(
            !sidecar.exists(),
            "SQLite snapshot is not standalone: sidecar {} exists",
            sidecar.display()
        );
    }
    Ok(())
}

fn validate_restore_schema(connection: &Connection) -> Result<()> {
    let version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    anyhow::ensure!(
        version <= SQLITE_SCHEMA_VERSION,
        "database schema version {version} is newer than supported version {SQLITE_SCHEMA_VERSION}"
    );
    let has_sessions: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions')",
        [],
        |row| row.get(0),
    )?;
    anyhow::ensure!(has_sessions, "MedAI sessions table is missing");
    if version >= 1 {
        for table in [
            "research_tasks",
            "research_artifacts",
            "graph_nodes",
            "graph_edges",
            "api_credentials",
            "tenant_directory",
        ] {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
                [table],
                |row| row.get(0),
            )?;
            anyhow::ensure!(
                exists,
                "versioned MedAI restore source is missing table {table}"
            );
        }
    }
    if version >= 2 {
        for table in ["model_budget_reservations", "artifact_redactions"] {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
                [table],
                |row| row.get(0),
            )?;
            anyhow::ensure!(
                exists,
                "versioned MedAI restore source is missing table {table}"
            );
        }
        let actor_column: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('model_budget_reservations') WHERE name='actor')",
            [],
            |row| row.get(0),
        )?;
        anyhow::ensure!(
            actor_column,
            "versioned MedAI restore source is missing model reservation actor provenance"
        );
    }
    Ok(())
}

#[cfg(unix)]
fn set_private_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(unix)]
fn create_private_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    let mut builder = fs::DirBuilder::new();
    builder.mode(0o700);
    builder.create(path)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn create_private_directory(path: &Path) -> Result<()> {
    fs::create_dir(path)?;
    Ok(())
}

#[cfg(unix)]
fn set_private_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?
        .sync_all()
        .context("sync backup directory")
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod scheduler_tests {
    use super::*;

    #[test]
    fn graph_edges_accept_only_same_session_source_artifacts_as_provenance() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("pathway evidence graph")?;
        let foreign_session = store.create_session("foreign source")?;
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        let foreign_task = store
            .claim_ready_tasks(&foreign_session.id, "worker", 1, 60)?
            .remove(0);
        let artifact = store.add_artifact(
            &session.id,
            &task.id,
            "pathway_analysis",
            "Reactome Analysis Service",
            "analysis-1",
            "{}",
        )?;
        let second_artifact = store.add_artifact(
            &session.id,
            &task.id,
            "pathway_analysis",
            "Reactome Analysis Service",
            "analysis-1-repeat",
            "{\"repeat\":true}",
        )?;
        let foreign_artifact = store.add_artifact(
            &foreign_session.id,
            &foreign_task.id,
            "pathway_analysis",
            "Reactome Analysis Service",
            "analysis-2",
            "{}",
        )?;
        for id in ["NCBIGene:7157", "ReactomeAnalysis:analysis-1"] {
            store.put_node(&GraphNode {
                session_id: session.id.clone(),
                id: id.into(),
                kind: "Analysis".into(),
                label: id.into(),
                metadata: "{}".into(),
            })?;
        }
        let edge = GraphEdge {
            session_id: session.id.clone(),
            source: "NCBIGene:7157".into(),
            relation: "SUBMITTED_TO_REACTOME_ANALYSIS".into(),
            target: "ReactomeAnalysis:analysis-1".into(),
            confidence: 1.0,
            evidence_ids: "[]".into(),
            artifact_ids: serde_json::to_string(&vec![artifact.id.clone()])?,
        };
        store.put_edge(&edge)?;
        let (_, edges) = store.graph(&session.id)?;
        assert_eq!(edges.len(), 1);
        assert_eq!(
            serde_json::from_str::<Vec<String>>(&edges[0].artifact_ids)?,
            vec![artifact.id.clone()]
        );
        let repeated_edge = GraphEdge {
            artifact_ids: serde_json::to_string(&vec![second_artifact.id.clone()])?,
            ..edge.clone()
        };
        store.put_edge(&repeated_edge)?;
        let (_, edges) = store.graph(&session.id)?;
        let retained: std::collections::BTreeSet<String> =
            serde_json::from_str(&edges[0].artifact_ids)?;
        assert_eq!(
            retained.len(),
            2,
            "edge updates must retain earlier provenance"
        );
        assert!(retained.contains(&artifact.id));
        assert!(retained.contains(&second_artifact.id));

        let conflicting_confidence = GraphEdge {
            confidence: 0.5,
            ..repeated_edge
        };
        assert!(store.put_edge(&conflicting_confidence).is_err());
        let (_, edges) = store.graph(&session.id)?;
        assert_eq!(edges[0].confidence, 1.0);
        let still_retained: std::collections::BTreeSet<String> =
            serde_json::from_str(&edges[0].artifact_ids)?;
        assert_eq!(still_retained, retained);
        let foreign_edge = GraphEdge {
            artifact_ids: serde_json::to_string(&vec![foreign_artifact.id])?,
            ..edge
        };
        assert!(store.put_edge(&foreign_edge).is_err());
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn agentmesh_approval_receipt_is_exact_scope_bound_expiring_and_auditable_data() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let approval = store.create_agentmesh_route_approval(
            "release-operator",
            "platform",
            "tenant-a",
            "biomedical",
            &"a".repeat(64),
            &"b".repeat(64),
        )?;
        assert_eq!(approval.expires_at - approval.issued_at, 600);
        assert!(
            store
                .validate_agentmesh_route_approval(
                    &approval.receipt_id,
                    "tenant-a",
                    "biomedical",
                    &"a".repeat(64),
                    &"b".repeat(64),
                )?
                .is_some()
        );
        assert!(
            store
                .validate_agentmesh_route_approval(
                    &approval.receipt_id,
                    "tenant-b",
                    "biomedical",
                    &"a".repeat(64),
                    &"b".repeat(64),
                )?
                .is_none()
        );
        store.db.execute(
            "UPDATE agentmesh_route_approvals SET expires_at=0 WHERE receipt_id=?1",
            [&approval.receipt_id],
        )?;
        assert!(
            store
                .validate_agentmesh_route_approval(
                    &approval.receipt_id,
                    "tenant-a",
                    "biomedical",
                    &"a".repeat(64),
                    &"b".repeat(64),
                )?
                .is_none()
        );
        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn tenant_directory_lifecycle_is_persistent_and_requires_a_disable_reason() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let created = store.create_tenant("research-lab", "platform-admin")?;
        assert_eq!(created.status, "active");
        assert!(
            store
                .create_tenant("research-lab", "platform-admin")
                .is_err()
        );
        assert!(
            store
                .set_tenant_status("research-lab", "disabled", None, "platform-admin")
                .is_err()
        );
        let disabled = store.set_tenant_status(
            "research-lab",
            "disabled",
            Some("security review"),
            "platform-admin",
        )?;
        assert_eq!(disabled.disabled_reason.as_deref(), Some("security review"));
        assert!(!store.tenant_is_active("research-lab")?);
        assert!(
            store
                .create_session_for_tenant("should be rejected", "standard", 10, "research-lab")
                .is_err()
        );
        let reactivated =
            store.set_tenant_status("research-lab", "active", None, "platform-admin")?;
        assert_eq!(reactivated.disabled_reason, None);
        assert!(store.tenant_is_active("research-lab")?);
        drop(store);
        let reopened = Store::open(&path)?;
        assert_eq!(
            reopened
                .tenant_directory_entry("research-lab")?
                .unwrap()
                .status,
            "active"
        );
        drop(reopened);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn tenant_quotas_limit_active_sessions_and_reserved_budget() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        store.set_tenant_quota("quota-lab", 2, 100, 100, "platform-admin")?;
        let first = store.create_session_for_tenant("first", "standard", 70, "quota-lab")?;
        assert!(
            store
                .create_session_for_tenant("second", "standard", 40, "quota-lab")
                .is_err()
        );
        store.db.execute(
            "UPDATE sessions SET status='completed' WHERE id=?1",
            [&first.id],
        )?;
        assert!(
            store
                .create_session_for_tenant("too much per session", "standard", 101, "quota-lab")
                .is_err()
        );
        let second = store.create_session_for_tenant(
            "second after completion",
            "standard",
            40,
            "quota-lab",
        )?;
        store.create_session_for_tenant("third", "standard", 40, "quota-lab")?;
        assert_eq!(
            store.tenant_quota_usage("quota-lab")?["usage"]["active_sessions"],
            2
        );
        assert_eq!(
            store.tenant_quota_usage("quota-lab")?["usage"]["allocated_credits"],
            80
        );
        assert_eq!(
            store.tenant_quota("quota-lab")?.updated_by.as_deref(),
            Some("platform-admin")
        );
        assert!(
            store
                .create_session_for_tenant("fourth", "standard", 10, "quota-lab")
                .is_err()
        );
        assert!(store.session_tenant(&second.id)?.is_some());
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn session_creation_api_charge_is_atomic_and_budget_checked() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let created = store.create_session_for_tenant_with_api_charge(
            "metered API session",
            "standard",
            5,
            "metered-lab",
            Some(2),
        )?;
        assert_eq!(created.credits_used, 2);
        let events = store.budget_events(&created.id)?;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].task_id, "api_request");
        assert_eq!(events[0].credits, 2);
        assert!(
            store
                .create_session_for_tenant_with_api_charge(
                    "unaffordable API session",
                    "standard",
                    1,
                    "metered-lab",
                    Some(2),
                )
                .is_err()
        );
        let remaining_sessions: i64 = store.db.query_row(
            "SELECT count(*) FROM sessions WHERE tenant_id='metered-lab'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(remaining_sessions, 1);
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn concurrent_session_creation_cannot_overbook_tenant_quota() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        store.set_tenant_quota("quota-race", 1, 100, 100, "platform-admin")?;
        drop(store);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0..2)
            .map(|index| {
                let path = path.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let store = Store::open(&path).unwrap();
                    barrier.wait();
                    store
                        .create_session_for_tenant(
                            &format!("concurrent {index}"),
                            "standard",
                            10,
                            "quota-race",
                        )
                        .is_ok()
                })
            })
            .collect();
        let succeeded = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .filter(|result| *result)
            .count();
        assert_eq!(succeeded, 1);
        let store = Store::open(&path)?;
        assert_eq!(store.sessions_for_tenant("quota-race")?.len(), 1);
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn api_credentials_are_persisted_hashed_scoped_and_revocable() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let record = ApiCredentialRecord {
            id: "credential-1".into(),
            token_sha256: format!("{:x}", Sha256::digest(b"secret-token")),
            subject: "researcher-1".into(),
            tenant_id: "lab-a".into(),
            role: "researcher".into(),
            created_at: 10,
            revoked_at: None,
        };
        store.register_api_credential(&record)?;
        store.register_api_credential(&record)?;
        assert_eq!(store.active_api_credentials()?.len(), 1);
        assert_eq!(store.api_credentials_for_tenant(Some("lab-b"))?.len(), 0);
        assert_eq!(
            store.api_credentials_for_tenant(Some("lab-a"))?[0].token_sha256,
            record.token_sha256
        );
        assert!(!store.revoke_api_credential("credential-1", Some("lab-b"))?);
        assert!(store.revoke_api_credential("credential-1", Some("lab-a"))?);
        assert!(store.active_api_credentials()?.is_empty());
        drop(store);
        let mut store = Store::open(&path)?;
        assert!(store.active_api_credentials()?.is_empty());
        // Reimporting the same environment token at startup must not undo revocation.
        store.register_api_credential(&record)?;
        assert!(store.active_api_credentials()?.is_empty());
        let history = store.api_credentials_for_tenant(Some("lab-a"))?;
        assert_eq!(history.len(), 1);
        assert!(history[0].revoked_at.is_some());
        assert!(
            !std::fs::read(&path)?
                .windows(b"secret-token".len())
                .any(|window| window == b"secret-token")
        );
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn api_audit_events_are_bounded_hash_chained_and_detect_changes() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        store.record_api_audit_event("req-1", "test-actor", "GET", "/health", 200, 3)?;
        store.record_api_audit_event("req-2", "test-actor", "POST", "/v1/research", 201, 8)?;
        let first_page = store.api_audit_events(0, 1)?;
        assert_eq!(first_page.len(), 1);
        assert_eq!(first_page[0].sequence, 1);
        let second_page = store.api_audit_events(first_page[0].sequence, 2)?;
        assert_eq!(second_page.len(), 1);
        assert_eq!(second_page[0].previous_sha256, first_page[0].event_sha256);
        let verified = store.verify_api_audit_chain()?;
        assert_eq!(verified["valid"], true);
        assert_eq!(verified["checked_events"], 2);
        store.db.execute(
            "UPDATE api_audit_events SET path='/tampered' WHERE sequence=1",
            [],
        )?;
        let tampered = store.verify_api_audit_chain()?;
        assert_eq!(tampered["valid"], false);
        assert_eq!(tampered["invalid_sequence"], 1);
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn audit_archival_retains_verifiable_chain_and_continues_sequence() -> Result<()> {
        let path = store_path();
        let archive_dir = PathBuf::from(format!("{}.audit-archive", path.display()));
        let _ = fs::remove_dir_all(&archive_dir);
        let mut store = Store::open(&path)?;
        store.record_api_audit_event("archive-1", "operator", "GET", "/v1/summary", 200, 2)?;
        store.record_api_audit_event("archive-2", "operator", "GET", "/v1/research", 200, 3)?;
        let cutoff = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64 + 60;
        let receipt = store
            .archive_api_audit_before(cutoff, &archive_dir)?
            .expect("old audit prefix archived");
        assert_eq!(receipt.first_sequence, 1);
        assert_eq!(receipt.last_sequence, 2);
        assert_eq!(store.api_audit_events(0, 10)?.len(), 0);
        let verified = store.verify_api_audit_chain()?;
        assert_eq!(verified["valid"], true);
        assert_eq!(verified["checked_events"], 2);
        assert_eq!(verified["archive_segments"], 1);
        let next =
            store.record_api_audit_event("archive-3", "operator", "GET", "/v1/summary", 200, 1)?;
        assert_eq!(next.sequence, 3);
        assert_eq!(
            next.previous_sha256,
            store.db.query_row(
                "SELECT head_sha256 FROM api_audit_archives WHERE last_sequence=2",
                [],
                |row| row.get::<_, String>(0)
            )?
        );
        assert_eq!(store.verify_api_audit_chain()?["checked_events"], 3);
        fs::write(archive_dir.join(&receipt.archive_file), b"tampered\n")?;
        let tampered = store.verify_api_audit_chain()?;
        assert_eq!(tampered["valid"], false);
        assert_eq!(tampered["error"], "audit archive digest mismatch");
        drop(store);
        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(archive_dir);
        Ok(())
    }

    #[test]
    fn tenant_audit_filter_is_exact_and_preserves_global_chain() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        for (id, actor, tenant) in [
            ("audit-a1", "lab-a:researcher:one", Some("lab-a")),
            ("audit-b1", "lab-b:researcher:two", Some("lab-b")),
            ("audit-a2", "lab-a:tenant_admin:admin", Some("lab-a")),
        ] {
            store.record_api_audit_event_for_tenant(
                id,
                &ApiAuditActor {
                    actor: actor.to_owned(),
                    tenant_id: tenant.map(str::to_owned),
                },
                "GET",
                "/v1/summary",
                200,
                2,
            )?;
        }
        let rows = store.api_audit_events_for_tenant("lab-a", 0, 20)?;
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter()
                .all(|event| event.tenant_id.as_deref() == Some("lab-a"))
        );
        assert_eq!(
            store
                .api_audit_events_for_tenant("lab-a", rows[0].sequence, 20)?
                .len(),
            1
        );
        assert_eq!(store.verify_api_audit_chain()?["valid"], true);
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn audit_tenant_column_migration_keeps_legacy_chain_verifiable() -> Result<()> {
        let path = store_path();
        let old = Connection::open(&path)?;
        old.execute_batch("CREATE TABLE api_audit_events(sequence INTEGER PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,actor TEXT NOT NULL,method TEXT NOT NULL,path TEXT NOT NULL,status INTEGER NOT NULL,duration_ms INTEGER NOT NULL,occurred_at INTEGER NOT NULL,previous_sha256 TEXT NOT NULL,event_sha256 TEXT NOT NULL);")?;
        let event = ApiAuditEvent {
            sequence: 1,
            request_id: "legacy-audit".into(),
            actor: "legacy-actor".into(),
            tenant_id: None,
            method: "GET".into(),
            path: "/v1/summary".into(),
            status: 200,
            duration_ms: 1,
            occurred_at: 10,
            previous_sha256: "0".repeat(64),
            event_sha256: String::new(),
        };
        let digest = api_audit_digest(&event)?;
        old.execute(
            "INSERT INTO api_audit_events VALUES(1,?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                event.request_id,
                event.actor,
                event.method,
                event.path,
                event.status,
                event.duration_ms,
                event.occurred_at,
                event.previous_sha256,
                digest
            ],
        )?;
        drop(old);
        let store = Store::open(&path)?;
        assert_eq!(store.verify_api_audit_chain()?["valid"], true);
        assert!(store.api_audit_events(0, 2)?[0].tenant_id.is_none());
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn failed_schema_migration_rolls_back_prior_ddl_and_backfills() -> Result<()> {
        let path = store_path();
        let legacy = Connection::open(&path)?;
        legacy.execute_batch(
            "CREATE TABLE sessions(id TEXT PRIMARY KEY,objective TEXT NOT NULL,status TEXT NOT NULL,created_at INTEGER NOT NULL);
             INSERT INTO sessions VALUES('legacy-session','legacy objective','active',1);
             CREATE TABLE api_audit_events(sequence INTEGER PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,actor TEXT NOT NULL,method TEXT NOT NULL,path TEXT NOT NULL,status INTEGER NOT NULL,duration_ms INTEGER NOT NULL,occurred_at INTEGER NOT NULL,previous_sha256 TEXT NOT NULL,event_sha256 TEXT NOT NULL);
             CREATE TABLE tenant_directory(tenant_id TEXT PRIMARY KEY,status TEXT NOT NULL CHECK(status IN ('active','disabled')),created_at INTEGER NOT NULL,updated_at INTEGER NOT NULL,updated_by TEXT NOT NULL,disabled_reason TEXT);
             CREATE TRIGGER reject_tenant_backfill BEFORE INSERT ON tenant_directory BEGIN SELECT RAISE(ABORT,'injected migration failure'); END;",
        )?;
        drop(legacy);

        let error = match Store::open(&path) {
            Ok(_) => anyhow::bail!("the injected backfill failure must abort startup"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("injected migration failure"));
        let restored = Connection::open(&path)?;
        let audit_has_tenant: bool = restored.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('api_audit_events') WHERE name='tenant_id')",
            [],
            |row| row.get(0),
        )?;
        let session_has_tenant: bool = restored.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('sessions') WHERE name='tenant_id')",
            [],
            |row| row.get(0),
        )?;
        let session_count: i64 =
            restored.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
        assert!(!audit_has_tenant, "earlier audit DDL must roll back");
        assert!(!session_has_tenant, "earlier session DDL must roll back");
        assert_eq!(session_count, 1, "legacy data must remain intact");
        drop(restored);
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        Ok(())
    }

    #[test]
    fn session_tenants_filter_lists_summaries_and_migrate_legacy_rows() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let alpha =
            store.create_session_for_tenant("alpha research", "standard", 10, "tenant-alpha")?;
        let beta =
            store.create_session_for_tenant("beta research", "standard", 10, "tenant-beta")?;
        assert_eq!(
            store
                .sessions_for_tenant("tenant-alpha")?
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec![alpha.id.as_str()]
        );
        assert_eq!(
            store
                .sessions_for_tenant("tenant-beta")?
                .iter()
                .map(|row| row.id.as_str())
                .collect::<Vec<_>>(),
            vec![beta.id.as_str()]
        );
        assert_eq!(
            store.session_tenant(&alpha.id)?.as_deref(),
            Some("tenant-alpha")
        );
        assert_eq!(store.summary_for_tenant("tenant-alpha")?["sessions"], 1);
        store.claim_ready_tasks(&alpha.id, "worker.alpha", 1, 120)?;
        let alpha_counts = store.operational_status_counts("tenant-alpha")?;
        let beta_counts = store.operational_status_counts("tenant-beta")?;
        assert_eq!(
            alpha_counts
                .iter()
                .filter(|row| row.0 == "session")
                .map(|row| row.2)
                .sum::<i64>(),
            1
        );
        assert!(
            alpha_counts
                .iter()
                .any(|row| row == &("task".into(), "running".into(), 1))
        );
        assert_eq!(
            beta_counts
                .iter()
                .filter(|row| row.0 == "session")
                .map(|row| row.2)
                .sum::<i64>(),
            1
        );
        assert!(
            !beta_counts
                .iter()
                .any(|row| row.0 == "task" && row.1 == "running")
        );
        store.register_trial_watch(&alpha.id, "NCT01234567", 3600)?;
        store.register_trial_watch(&beta.id, "NCT07654321", 3600)?;
        let alpha_trial_watches = store.claim_due_trial_watches_for_tenant(
            "worker.alpha",
            10,
            120,
            Some("tenant-alpha"),
        )?;
        assert_eq!(alpha_trial_watches.len(), 1);
        assert_eq!(alpha_trial_watches[0].session_id, alpha.id);
        store.register_literature_watch(&alpha.id, "alpha discovery", 86_400)?;
        store.register_literature_watch(&beta.id, "beta discovery", 86_400)?;
        let alpha_literature_watches = store.claim_due_literature_watches_for_tenant(
            "worker.alpha",
            10,
            120,
            Some("tenant-alpha"),
        )?;
        assert_eq!(alpha_literature_watches.len(), 1);
        assert_eq!(alpha_literature_watches[0].session_id, alpha.id);
        drop(store);
        let migration_path = store_path();
        {
            let old = Connection::open(&migration_path)?;
            old.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY,objective TEXT NOT NULL,status TEXT NOT NULL,created_at INTEGER NOT NULL,depth TEXT NOT NULL DEFAULT 'standard',budget INTEGER NOT NULL DEFAULT 100,credits_used INTEGER NOT NULL DEFAULT 0,cancellation_requested INTEGER NOT NULL DEFAULT 0); INSERT INTO sessions VALUES('RES-legacy','legacy session','planned',1,'standard',10,0,0);")?;
        }
        let migrated = Store::open(&migration_path)?;
        assert_eq!(
            migrated.session_tenant("RES-legacy")?.as_deref(),
            Some("local")
        );
        assert!(
            migrated
                .sessions_for_tenant("local")?
                .iter()
                .any(|row| row.id == "RES-legacy")
        );
        drop(migrated);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(migration_path);
        Ok(())
    }

    fn store_path() -> std::path::PathBuf {
        std::path::PathBuf::from("/private/tmp").join(format!(
            "medai-scheduler-{}-{}.sqlite",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn historical_pubmed_redaction_is_scoped_plan_bound_and_auditable() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let session = store.create_session("retention migration")?;
        store.transition_task(
            &session.id,
            "objective.normalize",
            None,
            "running",
            None,
            None,
        )?;
        store.transition_task(
            &session.id,
            "objective.normalize",
            None,
            "succeeded",
            Some("{}"),
            None,
        )?;
        store.transition_task(
            &session.id,
            "literature.discover",
            None,
            "running",
            None,
            None,
        )?;
        let source = store.add_artifact(&session.id, "literature.discover", "paper", "PubMed", "123", r#"{"pmid":"123","title":"Example","abstract":"copyrighted abstract","abstract_sections":[{"source_label":"RESULTS","text":"copyrighted abstract"}]}"#)?;
        let other = store.add_artifact(
            &session.id,
            "literature.discover",
            "paper",
            "Crossref",
            "10.1/x",
            r#"{"title":"untouched"}"#,
        )?;
        let storage_before = store.budget_summary(&session.id)?["storage"].clone();
        let preview = store.preview_pubmed_abstract_redaction(&session.id)?;
        assert_eq!(preview.candidate_count, 1);
        assert_eq!(preview.candidates[0].artifact_id, source.id);
        assert!(
            store
                .redact_historical_pubmed_abstracts(
                    &session.id,
                    &"0".repeat(64),
                    "researcher",
                    "retention policy"
                )
                .is_err()
        );
        let applied = store.redact_historical_pubmed_abstracts(
            &session.id,
            &preview.plan_sha256,
            "researcher",
            "retention policy",
        )?;
        assert_eq!(applied.plan_sha256, preview.plan_sha256);
        let retained = store.artifact(&session.id, &source.id)?.unwrap();
        assert!(!retained.payload.contains("copyrighted abstract"));
        assert_eq!(
            format!("{:x}", Sha256::digest(retained.payload.as_bytes())),
            retained.sha256
        );
        let storage_after = store.budget_summary(&session.id)?["storage"].clone();
        assert_eq!(
            storage_after["artifact_payload_bytes"],
            storage_before["artifact_payload_bytes"].as_i64().unwrap()
                - source.payload.len() as i64
                + retained.payload.len() as i64
        );
        assert_eq!(
            storage_after["credits_charged"],
            storage_before["credits_charged"]
        );
        assert_eq!(
            store.artifact(&session.id, &other.id)?.unwrap().sha256,
            other.sha256
        );
        let receipts = store.artifact_redaction_receipts(&session.id)?;
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0]["original_sha256"], source.sha256);
        assert_eq!(receipts[0]["redacted_sha256"], retained.sha256);
        assert_eq!(receipts[0]["actor"], "researcher");
        assert_eq!(
            store
                .preview_pubmed_abstract_redaction(&session.id)?
                .candidate_count,
            0
        );
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn model_budget_reservations_hold_credits_until_measured_settlement() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session_with_options("model reservation", "standard", 7)?;
        store.transition_task(
            &session.id,
            "objective.normalize",
            None,
            "running",
            None,
            None,
        )?;
        store.reserve_model_credits(
            &session.id,
            "objective.normalize",
            "model-call-1",
            "agent-1",
            4,
            "pre-call model token ceiling",
        )?;
        assert_eq!(store.budget_summary(&session.id)?["remaining"], 2);
        assert!(
            store
                .consume_credits(
                    &session.id,
                    "objective.normalize",
                    3,
                    "must not spend reserved credits"
                )
                .is_err()
        );
        assert!(
            store
                .settle_model_credits(&session.id, "model-call-1", "attacker", Some(0))
                .is_err()
        );
        let settled =
            store.settle_model_credits(&session.id, "model-call-1", "agent-1", Some(2))?;
        assert_eq!(settled["released_credits"], 2);
        assert_eq!(store.budget_summary(&session.id)?["remaining"], 4);
        assert_eq!(
            store.settle_model_credits(&session.id, "model-call-1", "agent-1", Some(2))?["state"],
            "settled"
        );
        assert!(
            store
                .settle_model_credits(&session.id, "model-call-1", "agent-1", Some(3))
                .is_err()
        );
        store.reserve_model_credits(
            &session.id,
            "objective.normalize",
            "model-call-2",
            "agent-1",
            2,
            "uncertain provider outcome",
        )?;
        let uncertain = store.settle_model_credits(&session.id, "model-call-2", "agent-1", None)?;
        assert_eq!(uncertain["state"], "uncertain");
        assert_eq!(uncertain["charged_credits"], 2);
        assert_eq!(
            store.settle_model_credits(&session.id, "model-call-2", "agent-1", None)?["state"],
            "uncertain"
        );
        store.reserve_model_credits(
            &session.id,
            "objective.normalize",
            "crashed-call",
            "agent-1",
            1,
            "worker crash reconciliation",
        )?;
        store.db.execute("UPDATE research_tasks SET lease_until=0 WHERE session_id=?1 AND id='objective.normalize'", [session.id.as_str()])?;
        assert_eq!(
            store.recover_expired_tasks(&session.id, 10)?,
            vec!["objective.normalize"]
        );
        let recovered_reservation = store.budget_summary(&session.id)?["model_reservations"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["id"] == "crashed-call")
            .unwrap()
            .clone();
        assert_eq!(recovered_reservation["state"], "uncertain");
        assert_eq!(recovered_reservation["charged_credits"], 1);
        drop(store);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    fn capability_validation_report(capability: &Capability, checks: serde_json::Value) -> String {
        let manifest_sha256 = format!("{:x}", Sha256::digest(capability.manifest.as_bytes()));
        serde_json::json!({
            "capability_id": capability.id,
            "version": capability.version,
            "manifest_sha256": manifest_sha256,
            "checks": checks,
        })
        .to_string()
    }

    #[test]
    fn generated_capability_production_requires_exact_persisted_paired_benchmark() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("Validate generated capability", "standard", 100)?;
        let proposal_id = "ART-SOURCE";
        let proposal_code = "def run(input):\n    return {'value': input['value']}\n";
        let source_sha256 = format!("{:x}", Sha256::digest(proposal_code.as_bytes()));
        let proposal_payload = serde_json::json!({
            "manifest": {"id":"genomics.example","version":"1.0.0"},
            "source": proposal_code,
            "static_review": {"source_sha256":source_sha256}
        })
        .to_string();
        let proposal_sha256 = format!("{:x}", Sha256::digest(proposal_payload.as_bytes()));
        store.db.execute(
            "INSERT INTO research_artifacts(id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at) VALUES(?1,?2,'test','generated_capability_proposal','MedAI','proposal',?3,?4,1)",
            params![proposal_id,session.id,proposal_payload,proposal_sha256],
        )?;
        let benchmark_payload = serde_json::json!({
            "status":"completed", "lifecycle_mutation":false, "registered":false, "promoted":false,
            "capability_id":"genomics.example", "candidate_proposal_artifact_id":proposal_id,
            "candidate_version":proposal_id,
            "candidate_proposal_artifact_sha256":proposal_sha256, "candidate_source_sha256":source_sha256,
            "candidate_proposal_signature":{"verified":true}, "baseline_proposal_signature":{"verified":true},
            "benchmark_sha256":"a".repeat(64), "comparison_sha256":"b".repeat(64),
            "execution_receipt_sha256":"c".repeat(64), "n":10
        }).to_string();
        let benchmark_id = "ART-BENCHMARK";
        let benchmark_sha256 = format!("{:x}", Sha256::digest(benchmark_payload.as_bytes()));
        store.db.execute(
            "INSERT INTO research_artifacts(id,session_id,task_id,kind,source,source_id,payload,sha256,recorded_at) VALUES(?1,?2,'test','capability_version_benchmark','MedAI paired pinned Docker capability benchmark','benchmark',?3,?4,1)",
            params![benchmark_id,session.id,benchmark_payload,benchmark_sha256],
        )?;
        let manifest = serde_json::json!({
            "source":"agentmesh_amcp",
            "execution":{"enabled":false,"network":false},
            "package":{
                "capability":{"id":"biomedical.generated.example-1234567890abcdef","version":"1.0.0"},
                "contract":{"inputs":{"properties":{
                    "source_session_id":{"const":session.id},
                    "proposal_artifact_id":{"const":proposal_id},
                    "proposal_artifact_sha256":{"const":proposal_sha256},
                    "proposal_capability_id":{"const":"genomics.example"}
                }}}
            }
        }).to_string();
        let validation = serde_json::json!({"benchmark_receipt":{
            "session_id":session.id, "artifact_id":benchmark_id, "artifact_sha256":benchmark_sha256
        }})
        .to_string();
        validate_paired_benchmark_artifact(
            &store.db,
            "biomedical.generated.example-1234567890abcdef",
            "1.0.0",
            &manifest,
            &validation,
        )?;

        let wrong_manifest = manifest.replace("ART-SOURCE", "ART-OTHER");
        assert!(
            validate_paired_benchmark_artifact(
                &store.db,
                "biomedical.generated.example-1234567890abcdef",
                "1.0.0",
                &wrong_manifest,
                &validation
            )
            .is_err()
        );
        let wrong_receipt = validation.replace(&benchmark_sha256, &"d".repeat(64));
        assert!(
            validate_paired_benchmark_artifact(
                &store.db,
                "biomedical.generated.example-1234567890abcdef",
                "1.0.0",
                &manifest,
                &wrong_receipt
            )
            .is_err()
        );
        let capability = Capability {
            id: "biomedical.generated.example-1234567890abcdef".into(),
            version: "1.0.0".into(),
            kind: "mcp".into(),
            status: "pending_review".into(),
            description: "Generated proposal lifecycle gate test".into(),
            manifest: manifest.clone(),
        };
        store.propose_capability(&capability)?;
        let verified = capability_validation_report(
            &capability,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true}),
        );
        store.transition_capability(
            &capability.id,
            &capability.version,
            "verified",
            "reviewer",
            "verification passed",
            Some(&verified),
        )?;
        let forged = capability_validation_report(
            &capability,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true}),
        );
        assert!(
            store
                .transition_capability(
                    &capability.id,
                    &capability.version,
                    "production",
                    "reviewer",
                    "forged benchmark check",
                    Some(&forged),
                )
                .is_err()
        );
        let production = serde_json::json!({
            "capability_id":capability.id,
            "version":capability.version,
            "manifest_sha256":format!("{:x}", Sha256::digest(capability.manifest.as_bytes())),
            "checks":{"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true},
            "benchmark_receipt":{"session_id":session.id,"artifact_id":benchmark_id,"artifact_sha256":benchmark_sha256}
        }).to_string();
        store.transition_capability(
            &capability.id,
            &capability.version,
            "production",
            "reviewer",
            "verified benchmark receipt reviewed",
            Some(&production),
        )?;
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn sqlite_backup_is_consistent_private_integrity_checked_and_never_overwrites() -> Result<()> {
        let database = store_path();
        let backup = database.with_extension("backup.sqlite");
        let store = Store::open(&database)?;
        let session = store.create_session("SQLite backup integrity test")?;
        let receipt = store.backup(&backup)?;
        assert_eq!(receipt.integrity_check, "ok");
        assert!(receipt.bytes > 0);
        assert_eq!(receipt.sha256.len(), 64);
        let verified = Store::verify_backup(&backup)?;
        assert_eq!(verified.sha256, receipt.sha256);
        let restored_snapshot = Store::open(&backup)?;
        assert!(
            restored_snapshot
                .sessions()?
                .iter()
                .any(|row| row.id == session.id)
        );
        assert!(store.backup(&backup).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&backup)?.permissions().mode() & 0o777,
                0o600
            );
        }
        drop(restored_snapshot);
        drop(store);
        let _ = fs::remove_file(database);
        let _ = fs::remove_file(backup);
        Ok(())
    }

    #[test]
    fn backup_verification_rejects_live_wal_sidecars_outside_the_file_digest() -> Result<()> {
        let path = store_path();
        let live = Connection::open(&path)?;
        live.execute_batch(
            "PRAGMA journal_mode=WAL; CREATE TABLE snapshot_data(value TEXT); INSERT INTO snapshot_data VALUES('uncheckpointed');",
        )?;
        let wal_path = PathBuf::from(format!("{}-wal", path.display()));
        assert!(
            wal_path.exists(),
            "fixture must retain an active WAL sidecar"
        );
        let error = match Store::verify_backup(&path) {
            Ok(_) => anyhow::bail!("a live WAL database is not a standalone backup"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("is not standalone"));
        drop(live);
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(wal_path);
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        Ok(())
    }

    #[test]
    fn sqlite_restore_verifies_source_and_retains_pre_restore_rollback_copy() -> Result<()> {
        let target_path = store_path();
        let source_path = store_path();
        let source_backup = source_path.with_extension("backup.sqlite");
        let source_store = Store::open(&source_path)?;
        let restored_session = source_store.create_session("Restored snapshot record")?;
        source_store.backup(&source_backup)?;
        drop(source_store);

        let mut target_store = Store::open(&target_path)?;
        let replaced_session = target_store.create_session("Pre-restore record")?;
        let invalid_source = store_path();
        fs::write(&invalid_source, b"not a SQLite database")?;
        assert!(target_store.restore_backup(&invalid_source).is_err());
        assert!(
            target_store
                .sessions()?
                .iter()
                .any(|row| row.id == replaced_session.id)
        );
        fs::remove_file(invalid_source)?;

        let unrelated_source = store_path();
        let unrelated = Connection::open(&unrelated_source)?;
        unrelated.execute_batch("CREATE TABLE unrelated(value TEXT);")?;
        drop(unrelated);
        assert!(target_store.restore_backup(&unrelated_source).is_err());
        assert!(
            target_store
                .sessions()?
                .iter()
                .any(|row| row.id == replaced_session.id),
            "a structurally valid but unrelated SQLite file must not replace MedAI data"
        );
        fs::remove_file(unrelated_source)?;

        let receipt = target_store.restore_backup(&source_backup)?;
        assert_eq!(receipt.integrity_check, "ok");
        assert_eq!(
            receipt.source_sha256,
            Store::verify_backup(&source_backup)?.sha256
        );
        assert_eq!(receipt.captured_snapshot_sha256.len(), 64);
        assert!(Path::new(&receipt.rollback_backup).exists());
        drop(target_store);

        let restored = Store::open(&target_path)?;
        let session_ids = restored
            .sessions()?
            .into_iter()
            .map(|row| row.id)
            .collect::<Vec<_>>();
        assert!(session_ids.contains(&restored_session.id));
        assert!(!session_ids.contains(&replaced_session.id));
        drop(restored);
        let rollback = Store::open(&receipt.rollback_backup)?;
        assert!(
            rollback
                .sessions()?
                .iter()
                .any(|row| row.id == replaced_session.id)
        );
        drop(rollback);

        for path in [
            target_path,
            source_path,
            source_backup,
            PathBuf::from(receipt.rollback_backup),
        ] {
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    #[test]
    fn sqlite_backup_verification_rejects_foreign_key_corruption() -> Result<()> {
        let path = store_path();
        {
            let connection = Connection::open(&path)?;
            connection.execute_batch(
                "PRAGMA foreign_keys=OFF;
                 CREATE TABLE parent(id INTEGER PRIMARY KEY);
                 CREATE TABLE child(parent_id INTEGER REFERENCES parent(id));
                 INSERT INTO child(parent_id) VALUES (42);",
            )?;
            let structural: String =
                connection.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
            assert_eq!(structural, "ok");
        }
        assert!(Store::verify_backup(&path).is_err());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn every_persisted_research_stage_has_a_registered_budget_owner() {
        for (task_id, _, _, _) in RESEARCH_TASKS {
            assert!(
                scientific_agent_for_task(task_id).is_some(),
                "missing scientific agent mapping for {task_id}"
            );
        }
    }

    #[test]
    fn specialist_claim_filter_leases_only_allowed_ready_tasks() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("specialist actor routing", "standard", 20)?;
        let literature_filter = vec!["literature.discover".to_owned()];
        assert!(
            store
                .claim_ready_tasks_for(
                    &session.id,
                    "actor.literature",
                    1,
                    60,
                    Some(&literature_filter)
                )?
                .is_empty()
        );
        let objective_filter = vec!["objective.normalize".to_owned()];
        let claimed = store.claim_ready_tasks_for(
            &session.id,
            "actor.director",
            1,
            60,
            Some(&objective_filter),
        )?;
        assert_eq!(
            claimed
                .iter()
                .map(|task| task.id.as_str())
                .collect::<Vec<_>>(),
            ["objective.normalize"]
        );
        assert!(
            store
                .claim_ready_tasks_for(&session.id, "actor.director", 1, 60, Some(&[]))
                .is_err()
        );
        assert!(
            store
                .claim_ready_tasks_for(
                    &session.id,
                    "actor.director",
                    1,
                    60,
                    Some(&[
                        "objective.normalize".to_owned(),
                        "objective.normalize".to_owned()
                    ])
                )
                .is_err()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn specialist_credit_ceiling_is_atomic_persistent_and_blocks_retry_for_review() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        store.configure_specialist_credit_policies(&std::collections::HashMap::from([(
            "literature_librarian".to_owned(),
            2,
        )]))?;
        let session = store.create_session_with_options("specialist budget", "standard", 50)?;
        let initial_budget = store.budget_summary(&session.id)?;
        assert_eq!(
            initial_budget["specialists"][0]["agent_id"],
            "literature_librarian"
        );
        assert_eq!(initial_budget["specialists"][0]["used"], 0);
        assert_eq!(initial_budget["specialists"][0]["limit"], 2);
        let objective = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        store.transition_task(
            &session.id,
            &objective.id,
            Some("worker"),
            "succeeded",
            None,
            None,
        )?;
        let literature = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        assert_eq!(literature.id, "literature.discover");
        store.consume_credits(&session.id, &literature.id, 1, "bounded source request")?;
        let before_rejection = store.budget_summary(&session.id)?;
        assert_eq!(
            before_rejection["specialists"][0]["agent_id"],
            "literature_librarian"
        );
        assert_eq!(before_rejection["specialists"][0]["used"], 2);
        assert_eq!(before_rejection["specialists"][0]["limit"], 2);
        assert!(
            store
                .consume_credits(
                    &session.id,
                    &literature.id,
                    1,
                    "over-ceiling source request"
                )
                .is_err()
        );
        let after_rejection = store.budget_summary(&session.id)?;
        assert_eq!(after_rejection["used"], before_rejection["used"]);
        drop(store);

        let reopened = Store::open(&path)?;
        let before_retry = reopened.budget_summary(&session.id)?;
        assert_eq!(before_retry["specialists"][0]["used"], 2);
        reopened.transition_task(
            &session.id,
            &literature.id,
            Some("worker"),
            "failed",
            None,
            Some("test recovery"),
        )?;
        reopened.transition_task(&session.id, &literature.id, None, "ready", None, None)?;
        assert!(
            reopened
                .claim_ready_tasks(&session.id, "worker.retry", 1, 60)?
                .is_empty()
        );
        assert_eq!(
            reopened
                .research_tasks(&session.id)?
                .iter()
                .find(|task| task.id == literature.id)
                .unwrap()
                .status,
            "needs_review"
        );
        assert_eq!(
            reopened
                .sessions()?
                .iter()
                .find(|row| row.id == session.id)
                .unwrap()
                .status,
            "needs_review"
        );
        drop(reopened);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn concurrent_workers_never_claim_the_same_ready_task() -> Result<()> {
        use std::sync::{Arc, Barrier};

        let path = store_path();
        let setup = Store::open(&path)?;
        let session = setup.create_session("Concurrent worker lease stress")?;
        drop(setup);

        let worker_count = 24;
        let barrier = Arc::new(Barrier::new(worker_count));
        let connections = (0..worker_count)
            .map(|_| Store::open(&path))
            .collect::<Result<Vec<_>>>()?;
        let workers = connections
            .into_iter()
            .enumerate()
            .map(|(index, store)| {
                let barrier = Arc::clone(&barrier);
                let session_id = session.id.clone();
                std::thread::spawn(move || -> Result<usize> {
                    barrier.wait();
                    Ok(store
                        .claim_ready_tasks(&session_id, &format!("worker.stress.{index}"), 1, 60)?
                        .len())
                })
            })
            .collect::<Vec<_>>();
        let claims = workers
            .into_iter()
            .map(|worker| worker.join().expect("worker thread must not panic"))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(claims.iter().sum::<usize>(), 1);

        let store = Store::open(&path)?;
        let root = store
            .research_tasks(&session.id)?
            .into_iter()
            .find(|task| task.id == "objective.normalize")
            .unwrap();
        assert_eq!(root.status, "running");
        assert_eq!(root.attempts, 1);
        assert!(root.lease_owner.is_some());
        drop(store);
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        Ok(())
    }

    #[test]
    fn concurrent_first_openers_serialize_schema_migrations() -> Result<()> {
        use std::sync::{Arc, Barrier};

        let path = store_path();
        let opener_count = 8;
        let barrier = Arc::new(Barrier::new(opener_count));
        let openers = (0..opener_count)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let path = path.clone();
                std::thread::spawn(move || -> Result<Store> {
                    barrier.wait();
                    Store::open(path)
                })
            })
            .collect::<Vec<_>>();
        let stores = openers
            .into_iter()
            .map(|opener| opener.join().expect("database opener must not panic"))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(stores.len(), opener_count);
        let check = Store::open(&path)?;
        assert_eq!(check_database_integrity(&check.db)?, "ok");
        let schema_version: i64 = check
            .db
            .pragma_query_value(None, "user_version", |row| row.get(0))?;
        assert_eq!(schema_version, SQLITE_SCHEMA_VERSION);
        drop(check);
        drop(stores);
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        Ok(())
    }

    #[test]
    fn unsupported_future_schema_version_fails_without_mutating_database() -> Result<()> {
        let path = store_path();
        let future = Connection::open(&path)?;
        future.execute_batch("CREATE TABLE future_marker(value TEXT); PRAGMA user_version=999;")?;
        drop(future);

        let error = match Store::open(&path) {
            Ok(_) => anyhow::bail!("future schema version must be rejected"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("newer than this MedAI build supports"));
        let unchanged = Connection::open(&path)?;
        let has_medai_sessions: bool = unchanged.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions')",
            [],
            |row| row.get(0),
        )?;
        let marker_count: i64 =
            unchanged.query_row("SELECT COUNT(*) FROM future_marker", [], |row| row.get(0))?;
        let version: i64 = unchanged.pragma_query_value(None, "user_version", |row| row.get(0))?;
        assert!(!has_medai_sessions);
        assert_eq!(marker_count, 0);
        assert_eq!(version, 999);
        drop(unchanged);
        std::fs::remove_file(&path)?;
        Ok(())
    }

    #[test]
    fn schema_v1_migrates_model_budget_reservations_atomically_to_v2() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        store.db.execute_batch("DROP TABLE model_budget_reservations; DROP TABLE artifact_redactions; PRAGMA user_version=1;")?;
        drop(store);
        let migrated = Store::open(&path)?;
        let version: i64 = migrated
            .db
            .pragma_query_value(None, "user_version", |row| row.get(0))?;
        let table_exists: bool = migrated.db.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_budget_reservations') AND EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_redactions') AND EXISTS(SELECT 1 FROM pragma_table_info('model_budget_reservations') WHERE name='actor')",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(version, 2);
        assert!(table_exists);
        migrated
            .db
            .execute_batch("DROP TABLE artifact_redactions;")?;
        assert!(validate_restore_schema(&migrated.db).is_err());
        drop(migrated);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    /// Sustained scheduler benchmark. Run explicitly with:
    /// `cargo test -p medai-core sustained_queue_claim_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "10,000-task sustained queue benchmark; run explicitly for performance measurements"]
    fn sustained_queue_claim_benchmark() -> Result<()> {
        use std::time::Instant;

        const TASKS: usize = 10_000;
        const WORKERS: usize = 8;
        const CLAIM_BATCH: usize = 16;

        let path = store_path();
        let setup = Store::open(&path)?;
        let session = setup.create_session_with_options(
            "Sustained scheduler queue benchmark",
            "deep",
            100_000,
        )?;
        let transaction = setup.db.unchecked_transaction()?;
        transaction.execute(
            "UPDATE research_tasks SET status='skipped' WHERE session_id=?1",
            [&session.id],
        )?;
        {
            let mut insert = transaction.prepare(
                "INSERT INTO research_tasks(session_id,id,domain,description,depends_on,status,available_at) VALUES(?1,?2,'benchmark','synthetic ready task','[]','ready',0)",
            )?;
            for index in 0..TASKS {
                insert.execute(params![session.id, format!("benchmark.load-{index:05}")])?;
            }
        }
        transaction.commit()?;
        drop(setup);

        let stores = (0..WORKERS)
            .map(|_| Store::open(&path))
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let workers = stores
            .into_iter()
            .enumerate()
            .map(|(index, store)| {
                let session_id = session.id.clone();
                std::thread::spawn(move || -> Result<(Vec<String>, Vec<u128>)> {
                    let mut claimed_ids = Vec::new();
                    let mut claim_latencies_ms = Vec::new();
                    loop {
                        let before = Instant::now();
                        let batch = store.claim_ready_tasks(
                            &session_id,
                            &format!("benchmark.worker.{index}"),
                            CLAIM_BATCH,
                            60,
                        )?;
                        claim_latencies_ms.push(before.elapsed().as_millis());
                        if batch.is_empty() {
                            break;
                        }
                        claimed_ids.extend(batch.into_iter().map(|task| task.id));
                    }
                    Ok((claimed_ids, claim_latencies_ms))
                })
            })
            .collect::<Vec<_>>();

        let mut claimed_ids = Vec::with_capacity(TASKS);
        let mut claim_latencies_ms = Vec::new();
        for worker in workers {
            let (mut ids, mut latencies) =
                worker.join().expect("benchmark worker must not panic")?;
            claimed_ids.append(&mut ids);
            claim_latencies_ms.append(&mut latencies);
        }
        let elapsed = started.elapsed();
        let unique_ids = claimed_ids.iter().collect::<std::collections::HashSet<_>>();
        assert_eq!(
            claimed_ids.len(),
            TASKS,
            "every queued task must be claimed"
        );
        assert_eq!(
            unique_ids.len(),
            TASKS,
            "no task may be claimed more than once"
        );
        claim_latencies_ms.sort_unstable();
        let percentile =
            |p: usize| -> u128 { claim_latencies_ms[(claim_latencies_ms.len() - 1) * p / 100] };
        let throughput = TASKS as f64 / elapsed.as_secs_f64();
        eprintln!(
            "scheduler benchmark: tasks={TASKS} workers={WORKERS} batch={CLAIM_BATCH} elapsed={:.3}s throughput={throughput:.0} tasks/s claim_tx_p50={}ms claim_tx_p95={}ms",
            elapsed.as_secs_f64(),
            percentile(50),
            percentile(95),
        );

        drop(unique_ids);
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
        Ok(())
    }

    #[test]
    fn capability_proposals_are_disabled_immutable_and_audited_through_promotion() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let capability = Capability {
            id: "biomedical.example".into(),
            version: "1.0.0".into(),
            kind: "tool".into(),
            status: "pending_review".into(),
            description: "Example generated tool".into(),
            manifest: r#"{"execution":{"enabled":false,"network":false},"source":"print(1)"}"#
                .into(),
        };
        store.propose_capability(&capability)?;
        assert!(store.propose_capability(&capability).is_err());
        assert!(
            store
                .transition_capability(
                    "biomedical.example",
                    "1.0.0",
                    "verified",
                    "reviewer",
                    "checked",
                    None
                )
                .is_err()
        );
        let report = capability_validation_report(
            &capability,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true}),
        );
        store.transition_capability(
            "biomedical.example",
            "1.0.0",
            "verified",
            "reviewer",
            "all checks passed",
            Some(&report),
        )?;
        store.transition_capability(
            "biomedical.example",
            "1.0.0",
            "production",
            "operator",
            "approved for use",
            Some(&report),
        )?;
        let events = store.capability_events("biomedical.example", "1.0.0")?;
        assert_eq!(events.len(), 3);
        assert_eq!(events[2]["to"], "production");
        assert!(events[2]["validation_sha256"].as_str().is_some());
        assert!(
            store
                .transition_capability(
                    "biomedical.example",
                    "1.0.0",
                    "verified",
                    "operator",
                    "rollback",
                    Some(&report)
                )
                .is_err()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn capability_promotion_requires_scientific_review_dimensions_and_production_benchmark()
    -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let proposal = Capability {
            id: "biomedical.validation-gate".into(),
            version: "1.0.0".into(),
            kind: "tool".into(),
            status: "pending_review".into(),
            description: "Validation gate regression".into(),
            manifest: r#"{"execution":{"enabled":false,"network":false}}"#.into(),
        };
        store.propose_capability(&proposal)?;
        let incomplete_report = capability_validation_report(
            &proposal,
            serde_json::json!({"tests":true,"contract":true}),
        );
        assert!(
            store
                .transition_capability(
                    &proposal.id,
                    &proposal.version,
                    "verified",
                    "reviewer",
                    "incomplete report",
                    Some(&incomplete_report),
                )
                .is_err()
        );
        let wrong_identity = Capability {
            id: "biomedical.other-capability".into(),
            ..proposal.clone()
        };
        let mismatched_report = capability_validation_report(
            &wrong_identity,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true}),
        );
        assert!(
            store
                .transition_capability(
                    &proposal.id,
                    &proposal.version,
                    "verified",
                    "reviewer",
                    "report from a different capability",
                    Some(&mismatched_report),
                )
                .is_err()
        );
        let wrong_manifest = Capability {
            manifest: r#"{"execution":{"enabled":false,"network":false},"changed":true}"#.into(),
            ..proposal.clone()
        };
        let drifted_report = capability_validation_report(
            &wrong_manifest,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true}),
        );
        assert!(
            store
                .transition_capability(
                    &proposal.id,
                    &proposal.version,
                    "verified",
                    "reviewer",
                    "report bound to drifted manifest",
                    Some(&drifted_report),
                )
                .is_err()
        );
        let verified_report = capability_validation_report(
            &proposal,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true}),
        );
        store.transition_capability(
            &proposal.id,
            &proposal.version,
            "verified",
            "reviewer",
            "required review passed",
            Some(&verified_report),
        )?;
        assert!(
            store
                .transition_capability(
                    &proposal.id,
                    &proposal.version,
                    "production",
                    "operator",
                    "benchmark missing",
                    Some(&verified_report),
                )
                .is_err()
        );
        let production_report = capability_validation_report(
            &proposal,
            serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true}),
        );
        store.transition_capability(
            &proposal.id,
            &proposal.version,
            "production",
            "operator",
            "all production checks passed",
            Some(&production_report),
        )?;
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn capability_proposal_rejects_enabled_or_networked_manifests() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let proposal = Capability {
            id: "unsafe".into(),
            version: "1".into(),
            kind: "tool".into(),
            status: "pending_review".into(),
            description: "unsafe".into(),
            manifest: r#"{"execution":{"enabled":true,"network":false}}"#.into(),
        };
        assert!(store.propose_capability(&proposal).is_err());
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn capability_rollback_is_atomic_audited_and_removes_rolled_back_version_from_production()
    -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let checks = serde_json::json!({"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true,"benchmark":true});
        let mut reports = std::collections::HashMap::new();
        for version in ["1.0.0", "2.0.0"] {
            let proposal = Capability {
                id: "biomedical.rollback".into(),
                version: version.into(),
                kind: "tool".into(),
                status: "pending_review".into(),
                description: "rollback test".into(),
                manifest: r#"{"execution":{"enabled":false,"network":false}}"#.into(),
            };
            store.propose_capability(&proposal)?;
            let report = capability_validation_report(&proposal, checks.clone());
            store.transition_capability(
                "biomedical.rollback",
                version,
                "verified",
                "reviewer",
                "checks passed",
                Some(&report),
            )?;
            reports.insert(version, report);
        }
        store.transition_capability(
            "biomedical.rollback",
            "1.0.0",
            "production",
            "operator",
            "release",
            Some(reports["1.0.0"].as_str()),
        )?;
        assert!(
            store
                .transition_capability(
                    "biomedical.rollback",
                    "2.0.0",
                    "production",
                    "operator",
                    "duplicate release",
                    Some(reports["2.0.0"].as_str())
                )
                .is_err()
        );
        assert!(
            store
                .rollback_capability(
                    "biomedical.rollback",
                    "2.0.0",
                    "operator",
                    "restore release",
                    &capability_validation_report(
                        &Capability {
                            id: "biomedical.rollback".into(),
                            version: "2.0.0".into(),
                            kind: "tool".into(),
                            status: "pending_review".into(),
                            description: "rollback test".into(),
                            manifest: r#"{"execution":{"enabled":false,"network":false}}"#.into(),
                        },
                        serde_json::json!({"rollback":false}),
                    )
                )
                .is_err()
        );
        assert_eq!(
            store
                .capabilities()?
                .iter()
                .find(|c| c.version == "1.0.0")
                .unwrap()
                .status,
            "production"
        );
        store.rollback_capability(
            "biomedical.rollback",
            "2.0.0",
            "operator",
            "restore release",
            reports["2.0.0"].as_str(),
        )?;
        let versions = store.capabilities()?;
        assert_eq!(
            versions
                .iter()
                .find(|c| c.version == "1.0.0")
                .unwrap()
                .status,
            "rolled_back"
        );
        assert_eq!(
            versions
                .iter()
                .find(|c| c.version == "2.0.0")
                .unwrap()
                .status,
            "production"
        );
        assert_eq!(
            store
                .capability_events("biomedical.rollback", "1.0.0")?
                .last()
                .unwrap()["to"],
            "rolled_back"
        );
        assert_eq!(
            store
                .capability_events("biomedical.rollback", "2.0.0")?
                .last()
                .unwrap()["to"],
            "production"
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn citation_links_are_derived_from_session_owned_crossref_artifacts_and_idempotent()
    -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("citation graph")?;
        let objective = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        store.transition_task(
            &session.id,
            &objective.id,
            Some("worker"),
            "succeeded",
            None,
            None,
        )?;
        let literature = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        let payload = r#"{"source":"Crossref REST API","complete_result_sha256":"0000000000000000000000000000000000000000000000000000000000000000","records":[{"doi":"10.1234/citing","references":[{"DOI":"10.5678/cited","reference_index":0},{"DOI":"not-a-doi","reference_index":1},{"article-title":"No DOI","reference_index":2}]},{"doi":"10.2222/citing-other","references":[{"DOI":"10.5678/cited","reference_index":0}]}]}"#;
        let artifact = store.add_artifact(
            &session.id,
            &literature.id,
            "crossref_citation_metadata",
            "Crossref REST API",
            "10.1234/citing",
            payload,
        )?;
        assert_eq!(
            store.index_crossref_citations(&session.id, &artifact.id)?,
            2
        );
        assert_eq!(
            store.index_crossref_citations(&session.id, &artifact.id)?,
            0
        );
        let links = store.citation_links(&session.id, Some("10.5678/cited"))?;
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].citing_doi, "10.1234/citing");
        assert_eq!(links[0].source_artifact_id, artifact.id);
        let metrics = store.citation_metrics(&session.id)?;
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].doi, "10.5678/cited");
        assert_eq!(metrics[0].citations_received, 2);
        assert!(metrics[0].pagerank > metrics[1].pagerank);
        assert!(
            store
                .index_crossref_citations("unknown-session", &artifact.id)
                .is_err()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn authenticated_research_input_can_be_attached_before_its_worker_task_runs() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("genomic input")?;
        let artifact = store.add_research_input_artifact(
            &session.id,
            "genomics.analyze",
            "genomic_dataset",
            "test local VCF report",
            &"a".repeat(64),
            r#"{"analysis":"local VCF parsing and validation"}"#,
        )?;
        assert_eq!(artifact.kind, "genomic_dataset");
        assert_eq!(artifact.task_id, "genomics.analyze");
        assert!(
            store
                .add_research_input_artifact(
                    &session.id,
                    "genomics.analyze",
                    "paper",
                    "test",
                    "source",
                    "{}",
                )
                .is_err()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn artifact_pages_filter_and_advance_with_a_stable_composite_cursor() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("artifact pagination")?;
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        for index in 0..5 {
            store.add_artifact(
                &session.id,
                &task.id,
                if index == 2 { "other" } else { "paper" },
                "test",
                &format!("source-{index}"),
                &format!(r#"{{"value":"payload-{index}"}}"#),
            )?;
        }
        let first = store.artifacts_page(&session.id, Some("paper"), None, None, 2)?;
        assert_eq!(first.len(), 2);
        let second = store.artifacts_page(
            &session.id,
            Some("paper"),
            Some(first[1].recorded_at),
            Some(&first[1].id),
            2,
        )?;
        assert_eq!(second.len(), 2);
        assert!(
            first
                .iter()
                .all(|left| second.iter().all(|right| left.id != right.id))
        );
        assert!(
            store
                .artifacts_page(&session.id, None, Some(first[0].recorded_at), None, 2)
                .is_err()
        );
        assert!(
            store
                .artifacts_page(&session.id, None, None, None, 202)
                .is_err()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn artifact_storage_is_metered_by_unique_persisted_payload_bytes() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("artifact storage budget", "standard", 10)?;
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        let data = "x".repeat(ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize / 4 - 11);
        let payload = format!(r#"{{"blob":"{data}"}}"#);
        assert_eq!(
            payload.len(),
            ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize / 4
        );

        let mut artifacts = Vec::new();
        for index in 0..4 {
            artifacts.push(store.add_artifact(
                &session.id,
                &task.id,
                "storage_fixture",
                "unit-test",
                &format!("fixture-{index}"),
                &payload,
            )?);
        }
        let duplicate = store.add_artifact(
            &session.id,
            &task.id,
            "storage_fixture",
            "unit-test",
            "fixture-3",
            &payload,
        )?;
        assert_eq!(duplicate.id, artifacts[3].id);

        let summary = store.budget_summary(&session.id)?;
        assert_eq!(
            summary["storage"]["artifact_payload_bytes"],
            ARTIFACT_STORAGE_BYTES_PER_CREDIT
        );
        assert_eq!(summary["storage"]["credits_charged"], 1);
        assert_eq!(summary["used"], 2);
        assert_eq!(
            store
                .budget_events(&session.id)?
                .iter()
                .filter(|event| event.reason.starts_with("artifact payload storage:"))
                .map(|event| event.credits)
                .sum::<i64>(),
            1
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn artifact_storage_budget_failure_rolls_back_the_crossing_artifact() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("artifact storage exhaustion", "standard", 1)?;
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        let data = "x".repeat(ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize / 4 - 11);
        let payload = format!(r#"{{"blob":"{data}"}}"#);

        for index in 0..3 {
            store.add_artifact(
                &session.id,
                &task.id,
                "storage_fixture",
                "unit-test",
                &format!("fixture-{index}"),
                &payload,
            )?;
        }
        assert!(
            store
                .add_artifact(
                    &session.id,
                    &task.id,
                    "storage_fixture",
                    "unit-test",
                    "fixture-3",
                    &payload,
                )
                .is_err()
        );

        assert_eq!(store.artifacts(&session.id)?.len(), 3);
        let summary = store.budget_summary(&session.id)?;
        assert_eq!(
            summary["storage"]["artifact_payload_bytes"],
            ARTIFACT_STORAGE_BYTES_PER_CREDIT * 3 / 4
        );
        assert_eq!(summary["storage"]["credits_charged"], 0);
        assert_eq!(summary["used"], 1);
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn artifact_storage_migration_preserves_legacy_bytes_without_retroactive_charge() -> Result<()>
    {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session_with_options(
            "legacy artifact storage migration",
            "standard",
            10,
        )?;
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)?
            .remove(0);
        let legacy_payload = r#"{"legacy":true}"#;
        let artifact = store.add_artifact(
            &session.id,
            &task.id,
            "legacy_fixture",
            "migration-test",
            "legacy-1",
            legacy_payload,
        )?;
        drop(store);

        // Simulate a database created before artifact_storage_usage existed.
        let legacy_db = Connection::open(&path)?;
        let historical_payload = format!(
            r#"{{"blob":"{}"}}"#,
            "x".repeat(ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize - r#"{"blob":""}"#.len())
        );
        assert_eq!(
            historical_payload.len() as i64,
            ARTIFACT_STORAGE_BYTES_PER_CREDIT
        );
        legacy_db.execute(
            "UPDATE research_artifacts SET payload=?2 WHERE id=?1",
            params![artifact.id, historical_payload],
        )?;
        legacy_db.execute("DROP TABLE artifact_storage_usage", [])?;
        drop(legacy_db);

        let migrated = Store::open(&path)?;
        let first = migrated.budget_summary(&session.id)?;
        assert_eq!(
            first["storage"]["artifact_payload_bytes"],
            ARTIFACT_STORAGE_BYTES_PER_CREDIT
        );
        assert_eq!(first["storage"]["credits_charged"], 0);
        assert_eq!(
            first["used"], 1,
            "migration must not charge historical payloads"
        );

        // Crossing the next full MiB charges only the bytes added after migration.
        for index in 0..4 {
            let target_bytes = ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize / 4;
            let empty_payload = format!(r#"{{"index":{index},"blob":""}}"#);
            let padding = "y".repeat(target_bytes - empty_payload.len());
            let payload = format!(r#"{{"index":{index},"blob":"{padding}"}}"#);
            assert_eq!(payload.len(), target_bytes);
            migrated.add_artifact(
                &session.id,
                &task.id,
                "post_migration_fixture",
                "migration-test",
                &format!("new-{index}"),
                &payload,
            )?;
        }
        let after_new_data = migrated.budget_summary(&session.id)?;
        assert_eq!(
            after_new_data["storage"]["artifact_payload_bytes"],
            2 * ARTIFACT_STORAGE_BYTES_PER_CREDIT
        );
        assert_eq!(after_new_data["storage"]["credits_charged"], 1);
        assert_eq!(after_new_data["used"], 2);

        drop(migrated);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn objective_scope_skips_unselected_tasks_without_charging_task_start_credits() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("APOE and Alzheimer disease", "standard", 50)?;
        let task = store
            .claim_ready_tasks(&session.id, "worker.scope", 1, 60)?
            .remove(0);
        assert_eq!(task.id, "objective.normalize");
        let objective_sha256 = format!("{:x}", Sha256::digest(session.objective.trim().as_bytes()));
        let selected = vec![
            "objective.normalize",
            "literature.discover",
            "literature.extract",
            "entities.resolve",
            "evidence.grade",
            "contradictions.search",
            "graph.build",
            "hypotheses.propose",
            "hypotheses.criticize",
            "report.reproduce",
        ];
        let payload = serde_json::json!({
            "schema_version": "2",
            "objective_sha256": objective_sha256,
            "depth": "standard",
            "planning": {"selected_task_ids": selected},
        })
        .to_string();
        let artifact = store.add_artifact(
            &session.id,
            "objective.normalize",
            "research_objective_plan",
            "MedAI configured model planner",
            &objective_sha256,
            &payload,
        )?;
        let before = store.budget_summary(&session.id)?["used"].as_i64().unwrap();
        let skipped = store.apply_research_task_scope(&session.id, "worker.scope", &artifact.id)?;
        assert_eq!(before, 1);
        assert_eq!(store.budget_summary(&session.id)?["used"], before);
        assert_eq!(skipped.len(), RESEARCH_TASKS.len() - 10);
        assert_eq!(
            store.apply_research_task_scope(&session.id, "worker.scope", &artifact.id)?,
            skipped,
            "a recovered director task can safely replay its existing scope"
        );
        let tasks = store.research_tasks(&session.id)?;
        assert!(
            tasks
                .iter()
                .filter(|task| skipped.contains(&task.id))
                .all(|task| task.status == "skipped")
        );
        store.transition_task(
            &session.id,
            "objective.normalize",
            Some("worker.scope"),
            "succeeded",
            Some("scope applied"),
            None,
        )?;
        let next = store.claim_ready_tasks(&session.id, "worker.next", 16, 60)?;
        assert_eq!(
            next.iter().map(|task| task.id.as_str()).collect::<Vec<_>>(),
            vec!["literature.discover"]
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn objective_scope_rejects_incomplete_prerequisites_without_mutating_tasks() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session =
            store.create_session_with_options("APOE and Alzheimer disease", "standard", 50)?;
        store.claim_ready_tasks(&session.id, "worker.scope", 1, 60)?;
        let objective_sha256 = format!("{:x}", Sha256::digest(session.objective.trim().as_bytes()));
        let selected = vec![
            "objective.normalize",
            "literature.discover",
            "literature.extract",
            "entities.resolve",
            "evidence.grade",
            "contradictions.search",
            "graph.build",
            "hypotheses.propose",
            "hypotheses.criticize",
            "report.reproduce",
            "drugs.investigate",
        ];
        let payload = serde_json::json!({
            "schema_version": "2",
            "objective_sha256": objective_sha256,
            "depth": "standard",
            "planning": {"selected_task_ids": selected},
        })
        .to_string();
        let artifact = store.add_artifact(
            &session.id,
            "objective.normalize",
            "research_objective_plan",
            "MedAI configured model planner",
            &objective_sha256,
            &payload,
        )?;
        assert!(
            store
                .apply_research_task_scope(&session.id, "worker.scope", &artifact.id)
                .is_err()
        );
        assert!(
            store
                .research_tasks(&session.id)?
                .iter()
                .all(|task| task.status != "skipped")
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn checkpoint_is_owned_and_cancel_skips_queued_then_running_tasks() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session_with_options("Test scheduler", "standard", 50)?;
        let claimed = store.claim_ready_tasks(&session.id, "worker.test", 1, 60)?;
        assert_eq!(claimed.len(), 1);
        let task = &claimed[0];
        assert_eq!(task.id, "objective.normalize");
        store.save_task_checkpoint(
            &session.id,
            &task.id,
            "worker.test",
            r#"{"stage":"normalized","items":1}"#,
        )?;
        assert!(
            store
                .save_task_checkpoint(&session.id, &task.id, "worker.other", r#"{"stage":"bad"}"#)
                .is_err()
        );
        assert!(
            store
                .research_tasks(&session.id)?
                .iter()
                .find(|row| row.id == task.id)
                .unwrap()
                .checkpoint
                .is_some()
        );

        let cancellation = store.cancel_session(&session.id)?;
        assert_eq!(cancellation["status"], "cancelling");
        assert_eq!(cancellation["running_tasks"], 1);
        assert!(
            store
                .claim_ready_tasks(&session.id, "worker.test", 1, 60)?
                .is_empty()
        );
        store.transition_task(
            &session.id,
            &task.id,
            Some("worker.test"),
            "skipped",
            Some("cancelled"),
            None,
        )?;
        assert_eq!(
            store
                .sessions()?
                .into_iter()
                .find(|row| row.id == session.id)
                .unwrap()
                .status,
            "cancelled"
        );
        assert!(
            store
                .research_tasks(&session.id)?
                .iter()
                .all(|row| row.status == "skipped")
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn expired_lease_is_skipped_after_cancellation_instead_of_requeued() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("Test expired cancellation")?;
        let task = store
            .claim_ready_tasks(&session.id, "worker.test", 1, 60)?
            .remove(0);
        store.cancel_session(&session.id)?;
        store.db.execute(
            "UPDATE research_tasks SET lease_until=0 WHERE session_id=?1 AND id=?2",
            params![session.id, task.id],
        )?;
        assert_eq!(store.recover_expired_tasks(&session.id, 10)?, vec![task.id]);
        assert_eq!(
            store
                .research_tasks(&session.id)?
                .iter()
                .find(|row| row.id == "objective.normalize")
                .unwrap()
                .status,
            "skipped"
        );
        assert_eq!(
            store
                .sessions()?
                .into_iter()
                .find(|row| row.id == session.id)
                .unwrap()
                .status,
            "cancelled"
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn expired_worker_lease_uses_persistent_exponential_retry_delay() -> Result<()> {
        let path = store_path();
        let store = Store::open(&path)?;
        let session = store.create_session("Test retry backoff")?;
        let task = store
            .claim_ready_tasks(&session.id, "worker.first", 1, 60)?
            .remove(0);
        store.db.execute(
            "UPDATE research_tasks SET lease_until=0 WHERE session_id=?1 AND id=?2",
            params![session.id, task.id],
        )?;
        drop(store);
        let store = Store::open(&path)?;
        store.recover_expired_tasks(&session.id, 10)?;
        let retried = store
            .research_tasks(&session.id)?
            .into_iter()
            .find(|row| row.id == task.id)
            .unwrap();
        assert_eq!(retried.status, "ready");
        assert!(retried.available_at > now()?);
        assert!(
            store
                .claim_ready_tasks(&session.id, "worker.second", 1, 60)?
                .is_empty()
        );
        store.db.execute(
            "UPDATE research_tasks SET available_at=0 WHERE session_id=?1 AND id=?2",
            params![session.id, task.id],
        )?;
        let retry = store
            .claim_ready_tasks(&session.id, "worker.second", 1, 60)?
            .remove(0);
        assert_eq!(retry.attempts, 2);
        store.db.execute(
            "UPDATE research_tasks SET lease_until=0 WHERE session_id=?1 AND id=?2",
            params![session.id, task.id],
        )?;
        store.recover_expired_tasks(&session.id, 10)?;
        let second_recovery = store
            .research_tasks(&session.id)?
            .into_iter()
            .find(|row| row.id == task.id)
            .unwrap();
        assert!(
            second_recovery
                .error
                .as_deref()
                .is_some_and(|detail| detail.contains("retry scheduled after 10s"))
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn literature_watch_leases_budget_new_pmids_and_marks_truncated_windows() -> Result<()> {
        let mut store = Store::open(store_path())?;
        let session = store.create_session_with_options(
            "Monitor APOE microglia publications",
            "standard",
            5,
        )?;
        let watch = store.register_literature_watch(&session.id, "APOE   microglia", 86_400)?;
        assert_eq!(watch.query, "APOE microglia");
        assert_eq!(watch.watch_id.len(), 64);
        assert!(
            store
                .register_literature_watch(&session.id, "", 86_400)
                .is_err()
        );
        assert!(
            store
                .register_literature_watch(&session.id, "APOE", 3600)
                .is_err()
        );
        let claimed = store.claim_due_literature_watches("worker.literature", 5, 120)?;
        assert_eq!(claimed.len(), 1);
        store.consume_literature_watch_credits(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            1,
            "PubMed scheduled query",
        )?;
        let first_ids = vec!["101".to_owned(), "102".to_owned()];
        let first = store.complete_literature_watch(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            &first_ids,
            500,
            &[],
        )?;
        assert!(first.incomplete);
        assert_eq!(
            serde_json::from_str::<Vec<String>>(&first.new_pmids)?,
            first_ids
        );
        let incomplete_watch = store.literature_watch(&session.id, &watch.watch_id)?;
        assert!(
            incomplete_watch
                .last_error
                .as_deref()
                .unwrap_or_default()
                .contains("review required")
        );

        let updated = store.register_literature_watch(&session.id, "APOE microglia", 86_400)?;
        assert_eq!(updated.watch_id, watch.watch_id);
        let claimed_again = store.claim_due_literature_watches("worker.literature", 5, 120)?;
        assert_eq!(claimed_again.len(), 1);
        store.consume_literature_watch_credits(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            1,
            "PubMed scheduled query",
        )?;
        let second_ids = vec!["101".to_owned(), "102".to_owned(), "103".to_owned()];
        let second = store.complete_literature_watch(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            &second_ids,
            3,
            &[LiteratureWatchPaper {
                session_id: String::new(),
                watch_id: String::new(),
                event_id: String::new(),
                pmid: "103".into(),
                title: "A bounded PubMed metadata record".into(),
                publication_date: Some("2026".into()),
                journal: Some("Research Journal".into()),
                doi: Some("10.1000/example".into()),
                pmcid: None,
                captured_at: 0,
                metadata_sha256: String::new(),
            }],
        )?;
        assert!(!second.incomplete);
        assert_eq!(
            serde_json::from_str::<Vec<String>>(&second.new_pmids)?,
            vec!["103"]
        );
        assert_eq!(
            store
                .literature_watch_events(&session.id, &watch.watch_id)?
                .len(),
            2
        );
        let retained = store.literature_watch_papers(&session.id, Some(&watch.watch_id), 10)?;
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].pmid, "103");
        assert_eq!(retained[0].event_id, second.id);
        assert!(!retained[0].metadata_sha256.is_empty());
        assert!(
            store
                .disable_literature_watch(&session.id, &watch.watch_id)
                .is_ok()
        );
        assert!(
            !store
                .literature_watch(&session.id, &watch.watch_id)?
                .enabled
        );
        Ok(())
    }

    #[test]
    fn literature_watch_keeps_metadata_hash_bound_to_its_pmid() -> Result<()> {
        let mut store = Store::open(store_path())?;
        let session = store.create_session_with_options(
            "Monitor a public neuroscience query",
            "standard",
            5,
        )?;
        let watch =
            store.register_literature_watch(&session.id, "public neuroscience query", 86_400)?;
        store.claim_due_literature_watches("worker.literature", 1, 120)?;

        let mut papers = vec![
            LiteratureWatchPaper {
                session_id: String::new(),
                watch_id: String::new(),
                event_id: String::new(),
                pmid: "201".into(),
                title: "Alpha paper".into(),
                publication_date: Some("2025".into()),
                journal: Some("Journal A".into()),
                doi: Some("10.1000/alpha".into()),
                pmcid: None,
                captured_at: 0,
                metadata_sha256: String::new(),
            },
            LiteratureWatchPaper {
                session_id: String::new(),
                watch_id: String::new(),
                event_id: String::new(),
                pmid: "202".into(),
                title: "Beta paper".into(),
                publication_date: Some("2026".into()),
                journal: Some("Journal B".into()),
                doi: Some("10.1000/beta".into()),
                pmcid: None,
                captured_at: 0,
                metadata_sha256: String::new(),
            },
        ];
        let digest = |paper: &LiteratureWatchPaper| -> Result<String> {
            let value = serde_json::json!({"pmid":paper.pmid,"title":paper.title,"publication_date":paper.publication_date,"journal":paper.journal,"doi":paper.doi,"pmcid":paper.pmcid});
            Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
        };
        let mut expected = papers.iter().map(digest).collect::<Result<Vec<_>>>()?;
        if expected[0] < expected[1] {
            papers.reverse();
            expected.reverse();
        }
        let pmids = papers
            .iter()
            .map(|paper| paper.pmid.clone())
            .collect::<Vec<_>>();
        store.complete_literature_watch(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            &pmids,
            2,
            &papers,
        )?;

        let retained = store.literature_watch_papers(&session.id, Some(&watch.watch_id), 10)?;
        let by_pmid = retained
            .into_iter()
            .map(|paper| (paper.pmid.clone(), paper.metadata_sha256))
            .collect::<std::collections::HashMap<_, _>>();
        for (paper, expected_hash) in papers.iter().zip(expected.iter()) {
            assert_eq!(by_pmid.get(&paper.pmid), Some(expected_hash));
        }
        Ok(())
    }

    #[test]
    fn literature_watch_paper_migration_links_legacy_metadata_to_discovery_event() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let session =
            store.create_session_with_options("watch metadata migration", "standard", 10)?;
        let watch =
            store.register_literature_watch(&session.id, "public neuroscience query", 86_400)?;
        store.claim_due_literature_watches("worker.literature", 1, 120)?;
        let pmids = vec!["456789".to_owned()];
        let event = store.complete_literature_watch(
            &session.id,
            &watch.watch_id,
            "worker.literature",
            &pmids,
            1,
            &[],
        )?;
        drop(store);

        let legacy_db = Connection::open(&path)?;
        legacy_db.execute("DROP TABLE literature_watch_papers", [])?;
        legacy_db.execute(
            "CREATE TABLE literature_watch_papers(session_id TEXT NOT NULL, watch_id TEXT NOT NULL, pmid TEXT NOT NULL, title TEXT NOT NULL, publication_date TEXT, journal TEXT, doi TEXT, pmcid TEXT, captured_at INTEGER NOT NULL, metadata_sha256 TEXT NOT NULL, PRIMARY KEY(session_id,watch_id,pmid), FOREIGN KEY(session_id,watch_id) REFERENCES literature_watches(session_id,watch_id) ON DELETE CASCADE)",
            [],
        )?;
        legacy_db.execute(
            "INSERT INTO literature_watch_papers(session_id,watch_id,pmid,title,captured_at,metadata_sha256) VALUES(?1,?2,?3,'Legacy retained record',1,'legacy-hash')",
            params![session.id, watch.watch_id, pmids[0]],
        )?;
        drop(legacy_db);

        let migrated = Store::open(&path)?;
        let records = migrated.literature_watch_papers(&session.id, Some(&watch.watch_id), 10)?;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].event_id, event.id);
        assert_eq!(records[0].pmid, pmids[0]);
        drop(migrated);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn trial_watch_persists_schedule_lease_budget_and_snapshot_diff() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let session = store.create_session_with_options("Monitor trial record", "standard", 10)?;
        store.register_trial_watch(&session.id, "NCT01234567", 3600)?;
        let claimed = store.claim_due_trial_watches("worker.watch", 5, 120)?;
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].nct_id, "NCT01234567");
        assert!(
            store
                .claim_due_trial_watches("worker.other", 5, 120)?
                .is_empty()
        );
        store.consume_trial_watch_credits(
            &session.id,
            "NCT01234567",
            "worker.watch",
            2,
            "trial registry update",
        )?;
        let snapshot = serde_json::json!({
            "nct_id":"NCT01234567",
            "source":"ClinicalTrials.gov",
            "record":{"protocolSection":{"identificationModule":{"nctId":"NCT01234567"}}},
            "provenance":{"record_snapshot_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
        }).to_string();
        let comparison = r#"{"changed":true,"changed_field_paths":["protocolSection.statusModule.overallStatus"],"scientific_or_safety_interpretation":"not_performed"}"#;
        let event = store.complete_trial_watch(
            &session.id,
            "NCT01234567",
            "worker.watch",
            &snapshot,
            Some(comparison),
        )?;
        assert_eq!(event.changed, Some(true));
        assert_eq!(
            store.trial_watch_events(&session.id, "NCT01234567")?.len(),
            1
        );
        let watch = store.trial_watch(&session.id, "NCT01234567")?;
        assert!(watch.last_snapshot.is_some());
        assert!(watch.lease_owner.is_none());
        assert!(watch.next_due > watch.last_checked_at.unwrap());
        assert_eq!(store.budget_summary(&session.id)?["used"], 2);
        store.disable_trial_watch(&session.id, "NCT01234567")?;
        assert!(!store.trial_watch(&session.id, "NCT01234567")?.enabled);
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn trial_watch_pauses_when_session_budget_is_exhausted() -> Result<()> {
        let path = store_path();
        let mut store = Store::open(&path)?;
        let session =
            store.create_session_with_options("Pause exhausted trial watch", "standard", 1)?;
        store.register_trial_watch(&session.id, "NCT01234567", 3600)?;
        store.claim_due_trial_watches("worker.watch", 1, 120)?;
        assert!(
            store
                .consume_trial_watch_credits(
                    &session.id,
                    "NCT01234567",
                    "worker.watch",
                    2,
                    "scheduled study refresh"
                )
                .is_err()
        );
        let watch = store.trial_watch(&session.id, "NCT01234567")?;
        assert!(!watch.enabled);
        assert!(watch.lease_owner.is_none());
        assert!(
            watch
                .last_error
                .as_deref()
                .unwrap_or_default()
                .contains("budget exhausted")
        );
        assert!(
            store
                .claim_due_trial_watches("worker.next", 1, 120)?
                .is_empty()
        );
        drop(store);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
