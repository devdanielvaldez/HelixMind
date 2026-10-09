use anyhow::Context;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderValue, Method, Request, StatusCode, header},
    middleware::{Next, from_fn_with_state},
    response::{IntoResponse, Response},
    routing::get,
};
use medai_core::{
    AgentMeshRouteApproval, ApiAuditActor, ApiCredentialRecord, Capability, Evidence, GraphEdge,
    GraphNode, LiteratureWatchPaper, MAX_ARTIFACT_PAYLOAD_BYTES, Store,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Clone)]
struct App {
    store: Arc<Mutex<Store>>,
    api_credentials: Arc<Mutex<Vec<ApiCredential>>>,
}

#[derive(Clone)]
struct AuditArchivePolicy {
    retention_days: i64,
    archive_dir: std::path::PathBuf,
    interval: Duration,
}

fn load_audit_archive_policy() -> anyhow::Result<Option<AuditArchivePolicy>> {
    let raw_days = match std::env::var("MEDAI_AUDIT_RETENTION_DAYS") {
        Ok(value) if !value.trim().is_empty() => value,
        Ok(_) | Err(std::env::VarError::NotPresent) => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let retention_days: i64 = raw_days
        .parse()
        .context("MEDAI_AUDIT_RETENTION_DAYS must be an integer")?;
    anyhow::ensure!(
        (1..=36_500).contains(&retention_days),
        "MEDAI_AUDIT_RETENTION_DAYS must be from 1 to 36500"
    );
    let archive_dir = std::env::var("MEDAI_AUDIT_ARCHIVE_DIR")
        .context("MEDAI_AUDIT_ARCHIVE_DIR is required when audit retention is enabled")?;
    let archive_dir = std::path::PathBuf::from(archive_dir);
    anyhow::ensure!(
        archive_dir.is_absolute(),
        "MEDAI_AUDIT_ARCHIVE_DIR must be an absolute path"
    );
    let interval_seconds: u64 = std::env::var("MEDAI_AUDIT_ARCHIVE_INTERVAL_SECONDS")
        .unwrap_or_else(|_| "86400".into())
        .parse()
        .context("MEDAI_AUDIT_ARCHIVE_INTERVAL_SECONDS must be an integer")?;
    anyhow::ensure!(
        (60..=31_536_000).contains(&interval_seconds),
        "MEDAI_AUDIT_ARCHIVE_INTERVAL_SECONDS must be from 60 to 31536000"
    );
    Ok(Some(AuditArchivePolicy {
        retention_days,
        archive_dir,
        interval: Duration::from_secs(interval_seconds),
    }))
}

fn spawn_audit_archive_scheduler(store: Arc<Mutex<Store>>, policy: Option<AuditArchivePolicy>) {
    let Some(policy) = policy else { return };
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(policy.interval);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        ticks.tick().await;
        loop {
            ticks.tick().await;
            let now = match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(duration) => duration.as_secs() as i64,
                Err(error) => {
                    eprintln!("MedAI audit retention clock error: {error}");
                    continue;
                }
            };
            let cutoff = now.saturating_sub(policy.retention_days.saturating_mul(86_400));
            let store = Arc::clone(&store);
            let archive_dir = policy.archive_dir.clone();
            match tokio::task::spawn_blocking(move || {
                let mut guard = store
                    .lock()
                    .map_err(|_| anyhow::anyhow!("store lock poisoned"))?;
                guard.archive_api_audit_before(cutoff, archive_dir)
            })
            .await
            {
                Ok(Ok(Some(receipt))) => eprintln!(
                    "MedAI audit retention archived sequences {}..{} ({})",
                    receipt.first_sequence, receipt.last_sequence, receipt.archive_sha256
                ),
                Ok(Ok(None)) => {}
                Ok(Err(error)) => eprintln!("MedAI audit retention failed: {error:#}"),
                Err(error) => eprintln!("MedAI audit retention worker failed: {error}"),
            }
        }
    });
}

#[derive(Clone, Debug, serde::Deserialize)]
struct ApiCredentialConfig {
    token: String,
    subject: String,
    tenant_id: String,
    role: String,
    #[serde(default)]
    scientific_agent_id: Option<String>,
}
#[derive(Clone)]
struct ApiCredential {
    id: String,
    token: Arc<[u8]>,
    subject: String,
    tenant_id: String,
    role: String,
    scientific_agent_id: Option<String>,
}
#[derive(Clone, Debug)]
struct ApiPrincipal {
    subject: String,
    tenant_id: String,
    role: String,
    scientific_agent_id: Option<String>,
}

fn is_general_service(principal: &ApiPrincipal) -> bool {
    principal.role == "service" && principal.scientific_agent_id.is_none()
}

fn token_matches(provided: &[u8], expected_token: &[u8]) -> bool {
    let mut difference = expected_token.len() ^ provided.len();
    for index in 0..expected_token.len().max(provided.len()) {
        difference |= usize::from(
            expected_token.get(index).copied().unwrap_or(0)
                ^ provided.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}
#[cfg(test)]
fn bearer_authorized(header: Option<&HeaderValue>, configured_token: Option<&[u8]>) -> bool {
    let Some(expected) = configured_token else {
        return true;
    };
    let Some(provided) = header
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    token_matches(provided.as_bytes(), expected)
}

fn principal_for_header(header: Option<&HeaderValue>, app: &App) -> Option<ApiPrincipal> {
    let Some(value) = header.and_then(|value| value.to_str().ok()) else {
        if app
            .api_credentials
            .lock()
            .map(|credentials| credentials.is_empty())
            .unwrap_or(false)
        {
            return Some(ApiPrincipal {
                subject: "local-anonymous".into(),
                tenant_id: "local".into(),
                role: "platform_admin".into(),
                scientific_agent_id: None,
            });
        }
        return None;
    };
    let provided_token = value.strip_prefix("Bearer ")?;
    let provided = provided_token.as_bytes();
    let provided_hash = Sha256::digest(provided);
    if let Ok(credentials) = app.api_credentials.lock() {
        for credential in credentials.iter() {
            if token_matches(&provided_hash, &credential.token) {
                return Some(ApiPrincipal {
                    subject: credential.subject.clone(),
                    tenant_id: credential.tenant_id.clone(),
                    role: credential.role.clone(),
                    scientific_agent_id: credential.scientific_agent_id.clone(),
                });
            }
        }
    }
    None
}

fn authorize_specialist_task(
    principal: &ApiPrincipal,
    task_id: &str,
) -> Result<(), (StatusCode, String)> {
    if let Some(agent_id) = principal.scientific_agent_id.as_deref()
        && medai_core::scientific_agent_for_task(task_id) != Some(agent_id)
    {
        return Err((
            StatusCode::FORBIDDEN,
            "specialist credential cannot modify a task assigned to another role".into(),
        ));
    }
    Ok(())
}

fn validate_scientific_artifact_origin(
    principal: &ApiPrincipal,
    task_id: &str,
    kind: &str,
    source: &str,
) -> Result<(), (StatusCode, String)> {
    let expected = match kind {
        "compound_chemical_identity" => Some(("compounds.analyze", "PubChem PUG REST")),
        "paper_extraction_metadata" | "paper_extraction_receipt" => {
            Some(("literature.extract", "PMC BioC"))
        }
        _ => None,
    };
    let Some((expected_task, expected_source)) = expected else {
        return Ok(());
    };
    if principal.role != "service" {
        return Err((
            StatusCode::FORBIDDEN,
            "source scientific artifacts can only be written by an authenticated service".into(),
        ));
    }
    if task_id != expected_task || source != expected_source {
        return Err((
            StatusCode::BAD_REQUEST,
            "scientific artifact kind, source, and producing task do not match".into(),
        ));
    }
    Ok(())
}

async fn authenticate_request(
    State(app): State<App>,
    mut request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let started = Instant::now();
    let path = request.uri().path().to_owned();
    let method = request.method().as_str().to_owned();
    let public_health = path == "/health";
    let principal = if public_health {
        None
    } else {
        principal_for_header(request.headers().get(header::AUTHORIZATION), &app)
    };
    let mut rejection: Option<(StatusCode, &'static str)> = None;
    if !public_health && principal.is_none() {
        rejection = Some((StatusCode::UNAUTHORIZED, "valid bearer token required"));
    }
    if let Some(identity) = principal.as_ref() {
        if identity.role != "platform_admin" {
            match app.store.lock() {
                Ok(store) => match store.tenant_is_active(&identity.tenant_id) {
                    Ok(true) => {}
                    Ok(false) => rejection = Some((StatusCode::FORBIDDEN, "tenant is disabled")),
                    Err(_) => {
                        rejection = Some((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "tenant status lookup failed",
                        ))
                    }
                },
                Err(_) => {
                    rejection = Some((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "tenant status lookup failed",
                    ))
                }
            }
        }
        if path.starts_with("/v1/admin/")
            && !(identity.role == "platform_admin"
                || is_general_service(identity)
                || (path.starts_with("/v1/admin/credentials") && identity.role == "tenant_admin")
                || (path.starts_with("/v1/admin/tenants/") && identity.role == "tenant_admin")
                || (path == "/v1/admin/tenants" && identity.role == "tenant_admin")
                || (matches!(path.as_str(), "/v1/admin/audit" | "/v1/admin/audit/export")
                    && identity.role == "tenant_admin")
                || (path == "/v1/admin/metrics" && identity.role == "tenant_admin"))
        {
            rejection = Some((
                StatusCode::FORBIDDEN,
                "platform administrator role required",
            ));
        } else if matches!(
            path.as_str(),
            "/v1/trial-watches/due/claim" | "/v1/literature-watches/due/claim"
        ) && identity.role != "platform_admin"
            && !is_general_service(identity)
        {
            rejection = Some((
                StatusCode::FORBIDDEN,
                "service role required for worker queue claims",
            ));
        } else if matches!(
            *request.method(),
            Method::POST | Method::PATCH | Method::PUT | Method::DELETE
        ) && !matches!(
            identity.role.as_str(),
            "researcher" | "tenant_admin" | "platform_admin" | "service"
        ) {
            rejection = Some((StatusCode::FORBIDDEN, "write role required"));
        } else if *request.method() == Method::POST
            && path.starts_with("/v1/research/")
            && path
                == format!(
                    "/v1/research/{}/control",
                    path.split('/').nth(3).unwrap_or_default()
                )
            && identity
                .scientific_agent_id
                .as_deref()
                .is_some_and(|agent_id| agent_id != "research_director")
        {
            rejection = Some((
                StatusCode::FORBIDDEN,
                "only a researcher or research director may cancel a session",
            ));
        }
        if path.starts_with("/v1/research/") {
            let session_id = path.split('/').nth(3).unwrap_or_default();
            match app.store.lock() {
                Ok(store) => match store.session_tenant(session_id) {
                    Ok(Some(tenant)) if tenant == identity.tenant_id => {}
                    Ok(_) => {
                        rejection = Some((StatusCode::NOT_FOUND, "research session not found"))
                    }
                    Err(_) => {
                        rejection = Some((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "research session ownership lookup failed",
                        ))
                    }
                },
                Err(_) => {
                    rejection = Some((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "research session ownership lookup failed",
                    ))
                }
            }
        }
        request.extensions_mut().insert(identity.clone());
    }
    let authorized = public_health || (principal.is_some() && rejection.is_none());
    let metered_session = if authorized && path.starts_with("/v1/research/") {
        let session_id = path.split('/').nth(3).unwrap_or_default();
        let is_budget_read =
            *request.method() == Method::GET && path == format!("/v1/research/{session_id}/budget");
        let is_status_read =
            *request.method() == Method::GET && path == format!("/v1/research/{session_id}/status");
        let route_parts: Vec<&str> = path.split('/').collect();
        let is_reservation_settlement = *request.method() == Method::POST
            && principal
                .as_ref()
                .is_some_and(|identity| identity.role == "service")
            && route_parts.len() == 8
            && route_parts[1] == "v1"
            && route_parts[2] == "research"
            && route_parts[4] == "budget"
            && route_parts[5] == "reservations"
            && route_parts[7] == "settle"
            && app.store.lock().is_ok_and(|store| {
                store
                    .model_reservation_task_id(session_id, route_parts[6])
                    .ok()
                    .flatten()
                    .is_some()
            });
        // Cancellation is an owner-controlled stop signal and must remain
        // available after research execution credits have been exhausted.
        let is_session_cancellation = *request.method() == Method::POST
            && path == format!("/v1/research/{session_id}/control");
        // Task transitions meter themselves after decoding the bounded body so
        // an already exhausted worker can still move its leased task to review.
        let is_task_transition = *request.method() == Method::PATCH
            && route_parts.len() == 6
            && route_parts[1] == "v1"
            && route_parts[2] == "research"
            && route_parts[4] == "tasks"
            && !route_parts[5].is_empty();
        (!is_budget_read
            && !is_status_read
            && !is_reservation_settlement
            && !is_session_cancellation
            && !is_task_transition)
            .then_some(session_id.to_owned())
    } else {
        None
    };
    let mut metering_rejection = None;
    if let Some(session_id) = metered_session.as_deref() {
        let request_credits = std::env::var("MEDAI_API_REQUEST_CREDITS")
            .ok()
            .map(|value| value.parse::<i64>())
            .transpose();
        match request_credits {
            Ok(request_credits) if request_credits.unwrap_or(1) > 0 => match app.store.lock() {
                Ok(store) => match store.consume_api_request_credits(
                    session_id,
                    request_credits.unwrap_or(1),
                    &method,
                    &path,
                ) {
                    Ok(_) => {}
                    Err(error) if error.to_string().contains("insufficient execution credits") => {
                        metering_rejection = Some((
                            StatusCode::PAYMENT_REQUIRED,
                            "research session execution budget exhausted",
                        ));
                    }
                    Err(_) => {
                        metering_rejection = Some((
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "research API request metering failed",
                        ));
                    }
                },
                Err(_) => {
                    metering_rejection = Some((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "research API request metering failed",
                    ));
                }
            },
            _ => {
                metering_rejection = Some((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "MEDAI_API_REQUEST_CREDITS must be a positive integer",
                ));
            }
        }
    }
    let actor = if public_health {
        "public-health".to_owned()
    } else if let Some(identity) = principal.as_ref() {
        format!(
            "{}:{}:{}",
            identity.tenant_id, identity.role, identity.subject
        )
    } else if !authorized {
        "unauthenticated".to_owned()
    } else {
        "loopback-anonymous".to_owned()
    };
    let mut response = if let Some((status, message)) = metering_rejection {
        (status, message).into_response()
    } else if authorized {
        next.run(request).await
    } else {
        let (status, message) =
            rejection.unwrap_or((StatusCode::UNAUTHORIZED, "valid bearer token required"));
        (status, message).into_response()
    };
    let request_id = uuid::Uuid::new_v4().to_string();
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-medai-request-id", value);
    }
    if let Ok(mut store) = app.store.lock() {
        let elapsed = started.elapsed().as_millis().min(i64::MAX as u128) as i64;
        if let Err(error) = store.record_api_audit_event_for_tenant(
            &request_id,
            &ApiAuditActor {
                actor: actor.clone(),
                tenant_id: principal
                    .as_ref()
                    .map(|identity| identity.tenant_id.clone()),
            },
            &method,
            &path,
            response.status().as_u16(),
            elapsed,
        ) {
            eprintln!("MedAI API audit write failed request_id={request_id}: {error:#}");
        }
    } else {
        eprintln!("MedAI API audit write skipped request_id={request_id}: store lock poisoned");
    }
    response
}

fn validate_api_token(
    configured: Option<String>,
    bind: SocketAddr,
) -> anyhow::Result<Option<Arc<[u8]>>> {
    let Some(value) = configured.filter(|value| !value.is_empty()) else {
        anyhow::ensure!(
            bind.ip().is_loopback(),
            "MEDAI_API_BEARER_TOKEN is required when MEDAI_BIND is not loopback"
        );
        return Ok(None);
    };
    anyhow::ensure!(
        value.len() >= 32 && value.chars().all(|character| !character.is_whitespace()),
        "MEDAI_API_BEARER_TOKEN must contain at least 32 non-whitespace bytes"
    );
    Ok(Some(Arc::<[u8]>::from(value.into_bytes())))
}
fn valid_identity_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
}
fn credential_set(credentials: Vec<ApiCredential>) -> Arc<Mutex<Vec<ApiCredential>>> {
    Arc::new(Mutex::new(credentials))
}
#[cfg(test)]
fn test_credential(raw_token: &[u8], subject: &str, tenant_id: &str, role: &str) -> ApiCredential {
    ApiCredential {
        id: uuid::Uuid::new_v4().to_string(),
        token: Arc::from(Sha256::digest(raw_token).to_vec()),
        subject: subject.into(),
        tenant_id: tenant_id.into(),
        role: role.into(),
        scientific_agent_id: None,
    }
}
fn load_api_credentials(
    legacy_token: Option<String>,
    registry_json: Option<String>,
    service_tenant: &str,
    bind: SocketAddr,
) -> anyhow::Result<Vec<ApiCredential>> {
    let mut credentials = Vec::new();
    if let Some(raw) = registry_json.filter(|value| !value.trim().is_empty()) {
        let configured: Vec<ApiCredentialConfig> = serde_json::from_str(&raw)
            .context("MEDAI_API_TOKENS must be a JSON array of credential records")?;
        anyhow::ensure!(
            configured.len() <= 1000,
            "MEDAI_API_TOKENS may contain at most 1000 credentials"
        );
        for entry in configured {
            anyhow::ensure!(
                entry.token.len() >= 32 && entry.token.chars().all(|c| !c.is_whitespace()),
                "each MEDAI_API_TOKENS token must contain at least 32 non-whitespace bytes"
            );
            anyhow::ensure!(
                valid_identity_identifier(&entry.subject),
                "credential subject must be a safe 1..128 character identifier"
            );
            anyhow::ensure!(
                valid_identity_identifier(&entry.tenant_id),
                "credential tenant_id must be a safe 1..128 character identifier"
            );
            anyhow::ensure!(
                matches!(
                    entry.role.as_str(),
                    "reader" | "researcher" | "tenant_admin" | "platform_admin" | "service"
                ),
                "credential role must be reader, researcher, tenant_admin, platform_admin, or service"
            );
            if let Some(agent_id) = entry.scientific_agent_id.as_deref() {
                anyhow::ensure!(
                    entry.role == "service"
                        && (1..=64).contains(&agent_id.len())
                        && agent_id.bytes().all(|byte| byte.is_ascii_lowercase()
                            || byte.is_ascii_digit()
                            || byte == b'_')
                        && medai_core::is_scientific_agent(agent_id),
                    "scientific_agent_id must be a valid registered specialist on a service credential"
                );
            }
            let token = Arc::<[u8]>::from(Sha256::digest(entry.token.as_bytes()).to_vec());
            anyhow::ensure!(
                credentials
                    .iter()
                    .all(|credential: &ApiCredential| !token_matches(&token, &credential.token)),
                "MEDAI_API_TOKENS contains duplicate credentials"
            );
            credentials.push(ApiCredential {
                id: uuid::Uuid::new_v4().to_string(),
                token,
                subject: entry.subject,
                tenant_id: entry.tenant_id,
                role: entry.role,
                scientific_agent_id: entry.scientific_agent_id,
            });
        }
    }
    if let Some(token) = legacy_token.filter(|value| !value.is_empty()) {
        anyhow::ensure!(
            valid_identity_identifier(service_tenant),
            "MEDAI_SERVICE_TENANT_ID must be a safe 1..128 character identifier"
        );
        let raw = validate_api_token(Some(token), "127.0.0.1:0".parse()?)?
            .expect("configured legacy token");
        let bytes = Arc::<[u8]>::from(Sha256::digest(&raw).to_vec());
        anyhow::ensure!(
            credentials
                .iter()
                .all(|credential| !token_matches(&bytes, &credential.token)),
            "MEDAI_API_BEARER_TOKEN duplicates a registry credential"
        );
        let fingerprint = format!("{:x}", Sha256::digest(&bytes));
        credentials.push(ApiCredential {
            id: uuid::Uuid::new_v4().to_string(),
            token: bytes,
            subject: format!("service-{}", &fingerprint[..12]),
            tenant_id: service_tenant.to_owned(),
            role: "service".into(),
            scientific_agent_id: None,
        });
    }
    anyhow::ensure!(
        bind.ip().is_loopback() || !credentials.is_empty(),
        "MEDAI_API_TOKENS or MEDAI_API_BEARER_TOKEN is required when MEDAI_BIND is not loopback"
    );
    Ok(credentials)
}
#[derive(Deserialize)]
struct NewResearch {
    objective: String,
    depth: Option<String>,
    budget: Option<i64>,
}
#[derive(Deserialize)]
struct RegisterTrialWatch {
    nct_id: String,
    interval_seconds: i64,
}
#[derive(Deserialize)]
struct ClaimTrialWatches {
    worker_id: String,
    limit: usize,
    lease_seconds: i64,
}
#[derive(Deserialize)]
struct CompleteTrialWatch {
    worker_id: String,
    snapshot: serde_json::Value,
    comparison: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct FailTrialWatch {
    worker_id: String,
    error: String,
}
#[derive(Deserialize)]
struct TrialWatchCharge {
    worker_id: String,
    credits: i64,
    reason: String,
}
#[derive(Deserialize)]
struct RegisterLiteratureWatch {
    query: String,
    interval_seconds: i64,
}
#[derive(Deserialize)]
struct ClaimLiteratureWatches {
    worker_id: String,
    limit: usize,
    lease_seconds: i64,
}
#[derive(Deserialize)]
struct LiteratureWatchCharge {
    worker_id: String,
    credits: i64,
    reason: String,
}
#[derive(Deserialize)]
struct CompleteLiteratureWatch {
    worker_id: String,
    pmids: Vec<String>,
    result_count: i64,
    #[serde(default)]
    papers: Vec<LiteratureWatchPaper>,
}
#[derive(Deserialize)]
struct FailLiteratureWatch {
    worker_id: String,
    error: String,
}
async fn health() -> &'static str {
    "ok"
}

fn render_operational_metrics(rows: &[(String, String, i64)]) -> String {
    const SESSION_STATUSES: &[&str] = &[
        "planned",
        "running",
        "cancelling",
        "cancelled",
        "completed",
        "budget_exhausted",
        "needs_review",
        "failed",
    ];
    const TASK_STATUSES: &[&str] = &[
        "ready",
        "blocked",
        "running",
        "succeeded",
        "failed",
        "needs_review",
        "skipped",
    ];
    let mut output = String::from(
        "# HELP medai_research_sessions Research sessions visible to the authenticated tenant.\n# TYPE medai_research_sessions gauge\n# HELP medai_research_tasks Research DAG tasks visible to the authenticated tenant.\n# TYPE medai_research_tasks gauge\n",
    );
    for (resource, allowed_statuses, metric) in [
        ("session", SESSION_STATUSES, "medai_research_sessions"),
        ("task", TASK_STATUSES, "medai_research_tasks"),
    ] {
        let mut counts = std::collections::BTreeMap::<&str, i64>::new();
        for status in allowed_statuses {
            counts.insert(status, 0);
        }
        for (row_resource, status, count) in rows {
            if row_resource != resource || *count < 0 {
                continue;
            }
            let label = if allowed_statuses.contains(&status.as_str()) {
                status.as_str()
            } else {
                "other"
            };
            *counts.entry(label).or_default() += count;
        }
        for (status, count) in counts {
            output.push_str(&format!("{metric}{{status=\"{status}\"}} {count}\n"));
        }
    }
    output
}

async fn operational_metrics(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Response, (StatusCode, String)> {
    let rows = s
        .store
        .lock()
        .map_err(internal)?
        .operational_status_counts(&principal.tenant_id)
        .map_err(internal)?;
    let body = render_operational_metrics(&rows);
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response())
}

async fn authenticated_identity(
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "subject": principal.subject,
        "tenant_id": principal.tenant_id,
        "role": principal.role,
        "scientific_agent_id": principal.scientific_agent_id,
    }))
}
async fn sessions(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let data = s
        .store
        .lock()
        .map_err(internal)?
        .sessions_for_tenant(&principal.tenant_id)
        .map_err(internal)?;
    Ok(Json(serde_json::json!({"items":data})))
}
async fn create(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewResearch>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let request_credits = std::env::var("MEDAI_API_REQUEST_CREDITS")
        .ok()
        .map(|value| value.parse::<i64>())
        .transpose()
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MEDAI_API_REQUEST_CREDITS must be a positive integer".into(),
            )
        })?
        .unwrap_or(1);
    if request_credits <= 0 {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "MEDAI_API_REQUEST_CREDITS must be a positive integer".into(),
        ));
    }
    create_with_request_credits(
        State(s),
        axum::Extension(principal),
        Json(input),
        request_credits,
    )
    .await
}

async fn create_with_request_credits(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewResearch>,
    request_credits: i64,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let row = s
        .store
        .lock()
        .map_err(internal)?
        .create_session_for_tenant_with_api_charge(
            &input.objective,
            input.depth.as_deref().unwrap_or("standard"),
            input.budget.unwrap_or(100),
            &principal.tenant_id,
            Some(request_credits),
        )
        .map_err(|error| {
            let message = error.to_string();
            if message.starts_with("tenant quota exceeded:") {
                (StatusCode::TOO_MANY_REQUESTS, message)
            } else if message.contains("budget cannot cover") {
                (StatusCode::PAYMENT_REQUIRED, message)
            } else {
                internal(error)
            }
        })?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(row).map_err(internal)?),
    ))
}
async fn research_plan(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    let tasks = store.research_tasks(&id).map_err(internal)?;
    let events = store.task_events(&id).map_err(internal)?;
    Ok(Json(
        serde_json::json!({"session_id":id,"tasks":tasks,"events":events}),
    ))
}
async fn research_status(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    let session = store
        .research_session_for_tenant(&id, &principal.tenant_id)
        .map_err(internal)?
        .ok_or((StatusCode::NOT_FOUND, "research session not found".into()))?;
    let tasks = store.research_tasks(&id).map_err(internal)?;
    let mut task_counts = std::collections::BTreeMap::<String, usize>::new();
    for task in &tasks {
        *task_counts.entry(task.status.clone()).or_default() += 1;
    }
    let terminal = !tasks.is_empty()
        && tasks
            .iter()
            .all(|task| !matches!(task.status.as_str(), "ready" | "running"));
    let task_rows: Vec<_> = tasks
        .into_iter()
        .map(|task| {
            serde_json::json!({
                "id": task.id,
                "domain": task.domain,
                "status": task.status,
                "attempts": task.attempts,
                "available_at": task.available_at,
            })
        })
        .collect();
    Ok(Json(serde_json::json!({
        "session": {
            "id": session.id,
            "objective": session.objective,
            "status": session.status,
            "created_at": session.created_at,
            "depth": session.depth,
            "budget": session.budget,
            "credits_used": session.credits_used,
            "cancellation_requested": session.cancellation_requested,
        },
        "task_counts": task_counts,
        "tasks": task_rows,
        "terminal": terminal,
        "interpretation_limit": "Terminal means the DAG has no ready or running task; blocked tasks may remain behind a failed, skipped, or needs-review dependency. This is operational status only and does not establish scientific validity or clinical suitability.",
    })))
}
async fn research_control(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = s
        .store
        .lock()
        .map_err(internal)?
        .session_control(&id)
        .map_err(internal)?;
    Ok(Json(result))
}
async fn cancel_research(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = s
        .store
        .lock()
        .map_err(internal)?
        .cancel_session(&id)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(Json(result))
}
async fn budget_summary(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = s
        .store
        .lock()
        .map_err(internal)?
        .budget_summary(&id)
        .map_err(internal)?;
    Ok(Json(result))
}
#[derive(Deserialize)]
struct CreditCharge {
    task_id: String,
    credits: i64,
    reason: String,
}
#[derive(Deserialize)]
struct ModelCreditReservation {
    task_id: String,
    reservation_id: String,
    credits: i64,
    reason: String,
}
#[derive(Deserialize)]
struct ModelCreditSettlement {
    actual_credits: Option<i64>,
}
async fn reserve_model_credits(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ModelCreditReservation>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "service" {
        return Err((
            StatusCode::FORBIDDEN,
            "model budget reservations require a service identity".into(),
        ));
    }
    authorize_specialist_task(&principal, &input.task_id)?;
    let store = s.store.lock().map_err(internal)?;
    store
        .reserve_model_credits(
            &id,
            &input.task_id,
            &input.reservation_id,
            &principal.subject,
            input.credits,
            &input.reason,
        )
        .map_err(|error| (StatusCode::PAYMENT_REQUIRED, error.to_string()))?;
    Ok(Json(store.budget_summary(&id).map_err(internal)?))
}
async fn settle_model_credits(
    State(s): State<App>,
    Path((id, reservation_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ModelCreditSettlement>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "service" {
        return Err((
            StatusCode::FORBIDDEN,
            "model budget settlements require a service identity".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let task_id = store
        .model_reservation_task_id(&id, &reservation_id)
        .map_err(internal)?
        .ok_or((
            StatusCode::NOT_FOUND,
            "model budget reservation not found".into(),
        ))?;
    authorize_specialist_task(&principal, &task_id)?;
    store
        .settle_model_credits(
            &id,
            &reservation_id,
            &principal.subject,
            input.actual_credits,
        )
        .map(Json)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))
}
async fn consume_budget(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(charge): Json<CreditCharge>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_specialist_task(&principal, &charge.task_id)?;
    let store = s.store.lock().map_err(internal)?;
    store
        .consume_credits(&id, &charge.task_id, charge.credits, &charge.reason)
        .map_err(|error| (StatusCode::TOO_MANY_REQUESTS, error.to_string()))?;
    Ok(Json(store.budget_summary(&id).map_err(internal)?))
}
#[derive(Deserialize)]
struct TaskTransition {
    status: String,
    output: Option<String>,
    error: Option<String>,
    worker_id: Option<String>,
}
async fn update_task(
    State(s): State<App>,
    Path((id, task_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<TaskTransition>,
) -> Result<StatusCode, (StatusCode, String)> {
    authorize_specialist_task(&principal, &task_id)?;
    let request_credits = std::env::var("MEDAI_API_REQUEST_CREDITS")
        .ok()
        .map(|value| value.parse::<i64>())
        .transpose()
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "MEDAI_API_REQUEST_CREDITS must be a positive integer".into(),
            )
        })?
        .unwrap_or(1);
    if request_credits <= 0 {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "MEDAI_API_REQUEST_CREDITS must be a positive integer".into(),
        ));
    }
    let charge_result = s
        .store
        .lock()
        .map_err(internal)?
        .consume_api_request_credits(
            &id,
            request_credits,
            "PATCH",
            &format!("/v1/research/{id}/tasks/{task_id}"),
        );
    match charge_result {
        Ok(_) => {}
        Err(error)
            if input.status == "needs_review"
                && error.to_string().contains("insufficient execution credits") => {}
        Err(error) if error.to_string().contains("insufficient execution credits") => {
            return Err((
                StatusCode::PAYMENT_REQUIRED,
                "research session execution budget exhausted".into(),
            ));
        }
        Err(error) => return Err(internal(error)),
    }
    {
        let current = s
            .store
            .lock()
            .map_err(internal)?
            .research_tasks(&id)
            .map_err(internal)?
            .into_iter()
            .find(|task| task.id == task_id)
            .map(|task| task.status);
        if current.as_deref() == Some("needs_review") {
            return Err((
                StatusCode::FORBIDDEN,
                "needs_review tasks must use the authenticated human review endpoint".into(),
            ));
        }
    }
    s.store
        .lock()
        .map_err(internal)?
        .transition_task(
            &id,
            &task_id,
            input.worker_id.as_deref(),
            &input.status,
            input.output.as_deref(),
            input.error.as_deref(),
        )
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
#[derive(Deserialize)]
struct HumanTaskReview {
    decision: String,
    rationale: String,
}
async fn review_task(
    State(s): State<App>,
    Path((id, task_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<HumanTaskReview>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "human scientific review requires researcher or platform_admin role".into(),
        ));
    }
    if input.rationale.trim().is_empty() || input.rationale.len() > 2000 {
        return Err((
            StatusCode::BAD_REQUEST,
            "review rationale must contain 1..2000 characters".into(),
        ));
    }
    let next = match input.decision.as_str() {
        "approve" => "succeeded",
        "reject" => "failed",
        "resume" => "ready",
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "review decision must be approve, reject, or resume".into(),
            ));
        }
    };
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let review = serde_json::json!({
        "review_type": "human_scientific_review",
        "decision": input.decision,
        "reviewer_subject": principal.subject,
        "reviewer_role": principal.role,
        "rationale": input.rationale.trim(),
    });
    let output = serde_json::to_string(&review).map_err(internal)?;
    if input.decision == "resume" {
        if task_id != "contradictions.search" {
            return Err((
                StatusCode::CONFLICT,
                "only contradictions.search can be resumed through human review".into(),
            ));
        }
        store
            .resume_semantic_claim_task(&id, &principal.subject, &principal.role, &input.rationale)
            .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    } else {
        store
            .transition_task(
                &id,
                &task_id,
                Some(&principal.subject),
                next,
                Some(&output),
                None,
            )
            .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    }
    Ok(Json(serde_json::json!({
        "session_id": id,
        "task_id": task_id,
        "status": next,
        "review": review,
    })))
}
#[derive(Deserialize)]
struct ClaimTasks {
    worker_id: String,
    limit: usize,
    lease_seconds: i64,
    task_ids: Option<Vec<String>>,
}
async fn claim_tasks(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ClaimTasks>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if let Some(agent_id) = principal.scientific_agent_id.as_deref() {
        let Some(task_ids) = input.task_ids.as_deref() else {
            return Err((
                StatusCode::FORBIDDEN,
                "specialist credentials must claim explicit task IDs".into(),
            ));
        };
        if task_ids
            .iter()
            .any(|task_id| medai_core::scientific_agent_for_task(task_id) != Some(agent_id))
        {
            return Err((
                StatusCode::FORBIDDEN,
                "specialist credential cannot claim tasks assigned to another role".into(),
            ));
        }
    }
    let tasks = s
        .store
        .lock()
        .map_err(internal)?
        .claim_ready_tasks_for(
            &id,
            &input.worker_id,
            input.limit,
            input.lease_seconds,
            input.task_ids.as_deref(),
        )
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(serde_json::json!({"session_id":id,"tasks":tasks})))
}
#[derive(Deserialize)]
struct ApplyResearchTaskScope {
    worker_id: String,
    plan_artifact_id: String,
}
async fn apply_research_task_scope(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ApplyResearchTaskScope>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "service"
        || principal.scientific_agent_id.as_deref() != Some("research_director")
    {
        return Err((
            StatusCode::FORBIDDEN,
            "only the authenticated Research Director can apply objective task scope".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    if store.session_tenant(&id).map_err(internal)?.as_deref() != Some(principal.tenant_id.as_str())
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let skipped = store
        .apply_research_task_scope(&id, &input.worker_id, &input.plan_artifact_id)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id": id, "plan_artifact_id": input.plan_artifact_id, "skipped_task_ids": skipped}),
    ))
}
#[derive(Deserialize)]
struct RenewLease {
    worker_id: String,
    lease_seconds: i64,
}
async fn renew_task_lease(
    State(s): State<App>,
    Path((id, task_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<RenewLease>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_specialist_task(&principal, &task_id)?;
    let lease_until = s
        .store
        .lock()
        .map_err(internal)?
        .renew_task_lease(&id, &task_id, &input.worker_id, input.lease_seconds)
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id":id,"task_id":task_id,"lease_until":lease_until}),
    ))
}
#[derive(Deserialize)]
struct TaskCheckpoint {
    worker_id: String,
    checkpoint: serde_json::Value,
}
async fn save_task_checkpoint(
    State(s): State<App>,
    Path((id, task_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<TaskCheckpoint>,
) -> Result<StatusCode, (StatusCode, String)> {
    authorize_specialist_task(&principal, &task_id)?;
    let payload = serde_json::to_string(&input.checkpoint).map_err(internal)?;
    s.store
        .lock()
        .map_err(internal)?
        .save_task_checkpoint(&id, &task_id, &input.worker_id, &payload)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
#[derive(Deserialize)]
struct RecoverTasks {
    limit: usize,
}
async fn recover_tasks(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<RecoverTasks>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.scientific_agent_id.is_some() {
        return Err((
            StatusCode::FORBIDDEN,
            "specialist credentials cannot recover leases across roles".into(),
        ));
    }
    let tasks = s
        .store
        .lock()
        .map_err(internal)?
        .recover_expired_tasks(&id, input.limit)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id":id,"recovered_task_ids":tasks}),
    ))
}
async fn evidence(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(filter): Query<EvidenceFilter>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let limit = filter.limit.unwrap_or(200);
    if !(1..=500).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "evidence page limit must be 1..500".into(),
        ));
    }
    if filter
        .after_id
        .as_ref()
        .is_some_and(|cursor| cursor.is_empty() || cursor.len() > 512)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "evidence cursor must be a bounded non-empty identifier".into(),
        ));
    }
    if filter.after_id.is_some() != filter.snapshot_rowid.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "evidence cursor requires both after_id and snapshot_rowid".into(),
        ));
    }
    if filter.snapshot_rowid.is_some_and(|rowid| rowid < 0) {
        return Err((
            StatusCode::BAD_REQUEST,
            "evidence snapshot_rowid cannot be negative".into(),
        ));
    }
    let (items, has_more, snapshot_rowid) = s
        .store
        .lock()
        .map_err(internal)?
        .evidence_page(
            &id,
            filter.after_id.as_deref(),
            filter.snapshot_rowid,
            limit,
        )
        .map_err(internal)?;
    let next_cursor = has_more
        .then(|| items.last().map(|row| row.id.as_str()))
        .flatten();
    Ok(Json(serde_json::json!({
        "session_id": id,
        "items": items,
        "has_more": has_more,
        "next_cursor": next_cursor,
        "snapshot_rowid": snapshot_rowid,
    })))
}
#[derive(Deserialize)]
struct NewEvidence {
    task_id: Option<String>,
    claim: String,
    source: String,
    source_id: Option<String>,
    polarity: String,
    quality: f64,
    provenance: String,
}
#[derive(Deserialize)]
struct NewArtifact {
    task_id: String,
    kind: String,
    source: String,
    source_id: String,
    payload: serde_json::Value,
}
#[derive(Deserialize)]
struct NewGenomicDataset {
    report: serde_json::Value,
}
#[derive(Clone, Deserialize)]
struct DrugInteractionStudyReview {
    screen_artifact_id: String,
    candidate_id: String,
    study_design: String,
    exposure_status: String,
    comparator: String,
    dose_a: String,
    dose_b: String,
    route_a: String,
    route_b: String,
    endpoint: String,
    outcome: String,
    limitations: String,
    label_evidence_overlap: String,
}
#[derive(Clone, Deserialize)]
struct DrugInteractionClaimReview {
    screening_artifact_id: String,
    review_index: usize,
    adjudication: String,
    claim_domain: String,
    exposure_comparability: String,
    outcome_comparability: String,
    rationale: String,
    context_notes: String,
}
#[derive(Clone, Deserialize)]
struct HypothesisLiteratureReview {
    comparison_artifact_id: String,
    paper_artifact_id: String,
    relevance: String,
    rationale: String,
    study_design: String,
    population_or_model: String,
    methods: String,
    findings: String,
    limitations: String,
}
#[derive(Clone, Deserialize)]
struct HypothesisCriticReview {
    criticism_artifact_id: String,
    hypothesis_id: String,
    item_type: String,
    item_id: String,
    assessment: String,
    rationale: String,
    study_context: String,
}
#[derive(Clone, Deserialize)]
struct SemanticClaimReview {
    screening_artifact_id: String,
    adjudication: String,
    population_comparability: String,
    method_comparability: String,
    rationale: String,
    methods_population_notes: String,
}
#[derive(Clone, Deserialize)]
struct OntologyCrossMappingReview {
    source_artifact_id: String,
    source_identifier: String,
    target_curie: String,
    decision: String,
    rationale: String,
}
#[derive(Clone, Deserialize)]
struct CrossrefReferenceReview {
    candidate_artifact_id: String,
    candidate_artifact_sha256: String,
    reference_sha256: String,
    citing_doi: String,
    reference_index: i64,
    decision: String,
    selected_cited_doi: Option<String>,
    rationale: String,
}
#[derive(Deserialize)]
struct GraphNodeWrite {
    #[serde(flatten)]
    node: GraphNode,
    task_id: Option<String>,
}
#[derive(Deserialize)]
struct GraphEdgeWrite {
    #[serde(flatten)]
    edge: GraphEdge,
    task_id: Option<String>,
}
#[derive(Deserialize)]
struct ArtifactFilter {
    kind: Option<String>,
    search: Option<String>,
    source_id: Option<String>,
    limit: Option<usize>,
    after_recorded_at: Option<i64>,
    after_id: Option<String>,
}
#[derive(Deserialize)]
struct GraphFilter {
    limit: Option<usize>,
    include_nodes: Option<bool>,
    include_edges: Option<bool>,
    after_node_label: Option<String>,
    after_node_id: Option<String>,
    after_edge_source: Option<String>,
    after_edge_relation: Option<String>,
    after_edge_target: Option<String>,
}
#[derive(Deserialize)]
struct EvidenceFilter {
    limit: Option<usize>,
    after_id: Option<String>,
    snapshot_rowid: Option<i64>,
}
#[derive(Deserialize)]
struct GraphSearchFilter {
    q: String,
    limit: Option<usize>,
}
#[derive(Deserialize)]
struct CitationFilter {
    doi: Option<String>,
}
#[derive(Deserialize)]
struct IndexCitations {
    artifact_id: String,
}
#[derive(Deserialize)]
struct AuditFilter {
    after_sequence: Option<i64>,
    limit: Option<usize>,
}
async fn citation_links(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(filter): Query<CitationFilter>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    let rows = store
        .citation_links(&id, filter.doi.as_deref())
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let metrics = store
        .citation_metrics(&id)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id":id,"items":rows,"metrics":metrics,"semantics":"Citation counts and PageRank describe connectivity only within this session's captured Crossref citation links. The graph may be incomplete; these metrics are not evidence polarity, claim support, paper quality, or source-complete citation counts."}),
    ))
}
async fn index_citations(
    State(s): State<App>,
    Path(id): Path<String>,
    Json(input): Json<IndexCitations>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let inserted = s
        .store
        .lock()
        .map_err(internal)?
        .index_crossref_citations(&id, &input.artifact_id)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let rows = s
        .store
        .lock()
        .map_err(internal)?
        .citation_links(&id, None)
        .map_err(internal)?;
    Ok(Json(
        serde_json::json!({"session_id":id,"artifact_id":input.artifact_id,"inserted":inserted,"total":rows.len()}),
    ))
}
async fn artifacts(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(filter): Query<ArtifactFilter>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    let limit = filter.limit.unwrap_or(200);
    if !(1..=200).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "artifact page limit must be 1..200".into(),
        ));
    }
    if filter
        .search
        .as_ref()
        .is_some_and(|query| query.len() > 256)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "artifact search query exceeds 256 bytes".into(),
        ));
    }
    if filter
        .source_id
        .as_ref()
        .is_some_and(|source_id| source_id.len() > 256)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "artifact source_id filter exceeds 256 bytes".into(),
        ));
    }
    if filter.after_recorded_at.is_some() != filter.after_id.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "artifact cursor requires both recorded_at and id".into(),
        ));
    }
    if (filter.search.is_some() || filter.after_id.is_some()) && filter.limit.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "artifact search and cursor requests require an explicit page limit".into(),
        ));
    }
    let mut items = store
        .filtered_artifacts_page(
            &id,
            filter.kind.as_deref(),
            filter.search.as_deref(),
            filter.source_id.as_deref(),
            filter.after_recorded_at,
            filter.after_id.as_deref(),
            limit + 1,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let has_more = items.len() > limit;
    items.truncate(limit);
    let next_cursor = if has_more {
        items
            .last()
            .map(|item| serde_json::json!({"recorded_at":item.recorded_at,"id":item.id}))
    } else {
        None
    };
    Ok(Json(
        serde_json::json!({"items":items,"next_cursor":next_cursor,"page_size":limit}),
    ))
}
async fn artifact_by_id(
    State(s): State<App>,
    Path((id, artifact_id)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    if store.session_tenant(&id).map_err(internal)?.as_deref() != Some(principal.tenant_id.as_str())
        && principal.role != "platform_admin"
    {
        return Err((
            StatusCode::NOT_FOUND,
            "research artifact not found in this session".into(),
        ));
    }
    let artifact = store
        .artifact(&id, &artifact_id)
        .map_err(internal)?
        .ok_or((
            StatusCode::NOT_FOUND,
            "research artifact not found in this session".into(),
        ))?;
    Ok(Json(serde_json::to_value(artifact).map_err(internal)?))
}
#[derive(Deserialize)]
struct PubMedRedactionApply {
    plan_sha256: String,
    reason: String,
}
async fn preview_pubmed_redaction(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !matches!(
        principal.role.as_str(),
        "researcher" | "tenant_admin" | "platform_admin"
    ) {
        return Err((
            StatusCode::FORBIDDEN,
            "PubMed retention preview requires a researcher or administrator role".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    if store.session_tenant(&id).map_err(internal)?.as_deref() != Some(principal.tenant_id.as_str())
        && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let plan = store
        .preview_pubmed_abstract_redaction(&id)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(
        serde_json::json!({"plan":plan,"requires_explicit_apply":true,"effect":"Replaces eligible historical PubMed abstract fields in this session; preserves original and replacement hashes in a redaction receipt."}),
    ))
}
async fn apply_pubmed_redaction(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<PubMedRedactionApply>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !matches!(
        principal.role.as_str(),
        "researcher" | "tenant_admin" | "platform_admin"
    ) {
        return Err((
            StatusCode::FORBIDDEN,
            "PubMed redaction requires a researcher or administrator role".into(),
        ));
    }
    let mut store = s.store.lock().map_err(internal)?;
    if store.session_tenant(&id).map_err(internal)?.as_deref() != Some(principal.tenant_id.as_str())
        && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let plan = store
        .redact_historical_pubmed_abstracts(
            &id,
            &input.plan_sha256,
            &principal.subject,
            &input.reason,
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    let receipts = store.artifact_redaction_receipts(&id).map_err(internal)?;
    Ok(Json(
        serde_json::json!({"session_id":id,"applied":plan.candidate_count,"plan_sha256":plan.plan_sha256,"receipts":receipts}),
    ))
}
async fn pubmed_redaction_receipts(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = s.store.lock().map_err(internal)?;
    if store.session_tenant(&id).map_err(internal)?.as_deref() != Some(principal.tenant_id.as_str())
        && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let receipts = store.artifact_redaction_receipts(&id).map_err(internal)?;
    Ok(Json(serde_json::json!({"session_id":id,"items":receipts})))
}
async fn register_trial_watch(
    State(s): State<App>,
    Path(id): Path<String>,
    Json(input): Json<RegisterTrialWatch>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let watch = s
        .store
        .lock()
        .map_err(internal)?
        .register_trial_watch(&id, &input.nct_id, input.interval_seconds)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(watch).map_err(internal)?),
    ))
}
async fn list_trial_watches(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let watches = s
        .store
        .lock()
        .map_err(internal)?
        .trial_watches(&id)
        .map_err(internal)?;
    let mut items = Vec::with_capacity(watches.len());
    for watch in watches {
        let mut value = serde_json::to_value(watch).map_err(internal)?;
        if let Some(object) = value.as_object_mut() {
            object.remove("last_snapshot");
        }
        items.push(value);
    }
    Ok(Json(serde_json::json!({"session_id":id,"items":items})))
}
async fn latest_trial_watch_snapshot(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let watch = s
        .store
        .lock()
        .map_err(internal)?
        .trial_watch(&id, &nct_id)
        .map_err(|error| (StatusCode::NOT_FOUND, error.to_string()))?;
    let snapshot = watch.last_snapshot.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "trial watch has no retained snapshot".into(),
        )
    })?;
    let study: serde_json::Value = serde_json::from_str(&snapshot).map_err(internal)?;
    Ok(Json(
        serde_json::json!({"session_id":id,"nct_id":nct_id,"study":study,"history_scope":"snapshots captured by this MedAI watch"}),
    ))
}
async fn disable_trial_watch(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    s.store
        .lock()
        .map_err(internal)?
        .disable_trial_watch(&id, &nct_id)
        .map_err(|error| (StatusCode::NOT_FOUND, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn claim_trial_watches(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ClaimTrialWatches>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let items = s
        .store
        .lock()
        .map_err(internal)?
        .claim_due_trial_watches_for_tenant(
            &input.worker_id,
            input.limit,
            input.lease_seconds,
            Some(&principal.tenant_id),
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(serde_json::json!({"items":items})))
}
async fn charge_trial_watch(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
    Json(input): Json<TrialWatchCharge>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let used = s
        .store
        .lock()
        .map_err(internal)?
        .consume_trial_watch_credits(&id, &nct_id, &input.worker_id, input.credits, &input.reason)
        .map_err(|error| (StatusCode::TOO_MANY_REQUESTS, error.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id":id,"nct_id":nct_id,"used":used}),
    ))
}
async fn complete_trial_watch(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
    Json(input): Json<CompleteTrialWatch>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let snapshot = serde_json::to_string(&input.snapshot).map_err(internal)?;
    let comparison = input
        .comparison
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(internal)?;
    let event = s
        .store
        .lock()
        .map_err(internal)?
        .complete_trial_watch(
            &id,
            &nct_id,
            &input.worker_id,
            &snapshot,
            comparison.as_deref(),
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    let comparison = event
        .comparison
        .as_deref()
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok());
    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::json!({"id":event.id,"session_id":event.session_id,"nct_id":event.nct_id,"captured_at":event.captured_at,"snapshot_sha256":event.snapshot_sha256,"changed":event.changed,"comparison":comparison}),
        ),
    ))
}
async fn fail_trial_watch(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
    Json(input): Json<FailTrialWatch>,
) -> Result<StatusCode, (StatusCode, String)> {
    s.store
        .lock()
        .map_err(internal)?
        .fail_trial_watch(&id, &nct_id, &input.worker_id, &input.error)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn trial_watch_events(
    State(s): State<App>,
    Path((id, nct_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let events = s
        .store
        .lock()
        .map_err(internal)?
        .trial_watch_events(&id, &nct_id)
        .map_err(internal)?;
    let items: Vec<_> = events.into_iter().map(|event| serde_json::json!({"id":event.id,"session_id":event.session_id,"nct_id":event.nct_id,"captured_at":event.captured_at,"snapshot_sha256":event.snapshot_sha256,"changed":event.changed,"comparison":event.comparison.and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())})).collect();
    Ok(Json(
        serde_json::json!({"session_id":id,"nct_id":nct_id,"items":items}),
    ))
}
async fn register_literature_watch(
    State(s): State<App>,
    Path(id): Path<String>,
    Json(input): Json<RegisterLiteratureWatch>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let watch = s
        .store
        .lock()
        .map_err(internal)?
        .register_literature_watch(&id, &input.query, input.interval_seconds)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(watch).map_err(internal)?),
    ))
}
async fn list_literature_watches(
    State(s): State<App>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let watches = s
        .store
        .lock()
        .map_err(internal)?
        .literature_watches(&id)
        .map_err(internal)?;
    let items = watches
        .into_iter()
        .map(|watch| {
            let mut value = serde_json::to_value(watch).map_err(internal)?;
            if let Some(object) = value.as_object_mut() {
                object.remove("last_pmids");
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>, (StatusCode, String)>>()?;
    Ok(Json(serde_json::json!({"session_id":id,"items":items})))
}
async fn disable_literature_watch(
    State(s): State<App>,
    Path((id, watch_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, String)> {
    s.store
        .lock()
        .map_err(internal)?
        .disable_literature_watch(&id, &watch_id)
        .map_err(|error| (StatusCode::NOT_FOUND, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn claim_literature_watches(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<ClaimLiteratureWatches>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let items = s
        .store
        .lock()
        .map_err(internal)?
        .claim_due_literature_watches_for_tenant(
            &input.worker_id,
            input.limit,
            input.lease_seconds,
            Some(&principal.tenant_id),
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(serde_json::json!({"items":items})))
}
async fn charge_literature_watch(
    State(s): State<App>,
    Path((id, watch_id)): Path<(String, String)>,
    Json(input): Json<LiteratureWatchCharge>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let used = s
        .store
        .lock()
        .map_err(internal)?
        .consume_literature_watch_credits(
            &id,
            &watch_id,
            &input.worker_id,
            input.credits,
            &input.reason,
        )
        .map_err(|error| (StatusCode::TOO_MANY_REQUESTS, error.to_string()))?;
    Ok(Json(
        serde_json::json!({"session_id":id,"watch_id":watch_id,"used":used}),
    ))
}
async fn complete_literature_watch(
    State(s): State<App>,
    Path((id, watch_id)): Path<(String, String)>,
    Json(input): Json<CompleteLiteratureWatch>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let event = s
        .store
        .lock()
        .map_err(internal)?
        .complete_literature_watch(
            &id,
            &watch_id,
            &input.worker_id,
            &input.pmids,
            input.result_count,
            &input.papers,
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(event).map_err(internal)?),
    ))
}
async fn fail_literature_watch(
    State(s): State<App>,
    Path((id, watch_id)): Path<(String, String)>,
    Json(input): Json<FailLiteratureWatch>,
) -> Result<StatusCode, (StatusCode, String)> {
    s.store
        .lock()
        .map_err(internal)?
        .fail_literature_watch(&id, &watch_id, &input.worker_id, &input.error)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn literature_watch_events(
    State(s): State<App>,
    Path((id, watch_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let events = s
        .store
        .lock()
        .map_err(internal)?
        .literature_watch_events(&id, &watch_id)
        .map_err(internal)?;
    Ok(Json(
        serde_json::json!({"session_id":id,"watch_id":watch_id,"items":events}),
    ))
}
async fn literature_watch_papers(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(query): Query<std::collections::HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let limit = query
        .get("limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100);
    let items = s
        .store
        .lock()
        .map_err(internal)?
        .literature_watch_papers(&id, query.get("watch_id").map(String::as_str), limit)
        .map_err(internal)?;
    Ok(Json(serde_json::json!({"session_id":id,"items":items})))
}
async fn add_artifact(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewArtifact>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    authorize_specialist_task(&principal, &input.task_id)?;
    validate_scientific_artifact_origin(&principal, &input.task_id, &input.kind, &input.source)?;
    let payload = serde_json::to_string(&input.payload).map_err(internal)?;
    let item = s
        .store
        .lock()
        .map_err(internal)?
        .add_artifact(
            &id,
            &input.task_id,
            &input.kind,
            &input.source,
            &input.source_id,
            &payload,
        )
        .map_err(|error| {
            let message = error.to_string();
            let status = if message.contains("insufficient execution credits") {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, message)
        })?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}
async fn submit_genomic_dataset(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewGenomicDataset>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(
        principal.role.as_str(),
        "researcher" | "tenant_admin" | "platform_admin"
    ) {
        return Err((
            StatusCode::FORBIDDEN,
            "genomic dataset submission requires researcher, tenant_admin, or platform_admin role"
                .into(),
        ));
    }
    let report = input.report.as_object().ok_or((
        StatusCode::BAD_REQUEST,
        "report must be a JSON object".into(),
    ))?;
    let input_sha256 = report
        .get("source_vcf_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let valid_sha256 = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    };
    let build = report
        .get("genome_build")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let implementation_hashes = report
        .get("analysis_implementation_sha256")
        .and_then(serde_json::Value::as_object);
    let reference_validated = report
        .get("reference_validated")
        .and_then(serde_json::Value::as_bool);
    let reference_sha256 = report
        .get("reference_fasta_sha256")
        .and_then(serde_json::Value::as_str);
    let allowed_report_fields = [
        "schema_version",
        "source",
        "source_vcf_sha256",
        "genome_build",
        "analysis_implementation_sha256",
        "reference_fasta_sha256",
        "reference_validated",
        "records_parsed",
        "alternate_alleles_emitted",
        "variant_class_counts",
        "validation_counts",
        "local_transcript_consequence_counts",
        "local_annotated_gene_identifier_count",
        "sample_identifiers_retained",
        "raw_variant_records_retained_in_summary",
        "remote_sources_queried",
        "interpretation_limit",
    ];
    if report
        .keys()
        .any(|key| !allowed_report_fields.contains(&key.as_str()))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "report contains fields outside the local MedAI VCF report schema".into(),
        ));
    }
    if report
        .get("schema_version")
        .and_then(serde_json::Value::as_str)
        != Some("1")
        || report.get("source").and_then(serde_json::Value::as_str)
            != Some("MedAI local VCF analysis report")
        || !valid_sha256(input_sha256)
        || !matches!(build, "GRCh37" | "GRCh38")
        || reference_validated.is_none()
        || (reference_validated == Some(true)
            && reference_sha256.is_none_or(|digest| !valid_sha256(digest)))
        || report.get("sample_identifiers_retained") != Some(&serde_json::Value::Bool(false))
        || report.get("raw_variant_records_retained_in_summary")
            != Some(&serde_json::Value::Bool(false))
        || report.get("remote_sources_queried") != Some(&serde_json::Value::Bool(false))
        || report
            .get("interpretation_limit")
            .and_then(serde_json::Value::as_str)
            .is_none()
        || [
            "records_parsed",
            "alternate_alleles_emitted",
            "local_annotated_gene_identifier_count",
        ]
        .iter()
        .any(|key| {
            report
                .get(*key)
                .and_then(serde_json::Value::as_u64)
                .is_none_or(|count| count > 2_000_000)
        })
        || implementation_hashes.is_none_or(|hashes| {
            hashes.is_empty()
                || hashes.keys().any(|name| {
                    !["__init__.py", "genomics.py", "transcripts.py"].contains(&name.as_str())
                })
                || hashes
                    .values()
                    .any(|value| value.as_str().is_none_or(|digest| !valid_sha256(digest)))
        })
    {
        return Err((StatusCode::BAD_REQUEST, "report is not a bounded privacy-minimized MedAI VCF aggregate receipt with valid build, hashes, and privacy flags".into()));
    }
    let valid_count_map = |key: &str| {
        report
            .get(key)
            .and_then(serde_json::Value::as_object)
            .is_some_and(|counts| {
                counts.len() <= 128
                    && counts.iter().all(|(name, count)| {
                        !name.is_empty()
                            && name.len() <= 128
                            && count.as_u64().is_some_and(|value| value <= 2_000_000)
                    })
            })
    };
    let count_total = |key: &str| {
        report
            .get(key)
            .and_then(serde_json::Value::as_object)
            .map(|counts| {
                counts
                    .values()
                    .filter_map(serde_json::Value::as_u64)
                    .sum::<u64>()
            })
    };
    let emitted = report
        .get("alternate_alleles_emitted")
        .and_then(serde_json::Value::as_u64);
    let reference_validation_total = report
        .get("validation_counts")
        .and_then(serde_json::Value::as_object)
        .map(|counts| {
            counts
                .get("reference_validated")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                + counts
                    .get("reference_unvalidated")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
        });
    if !valid_count_map("variant_class_counts")
        || !valid_count_map("validation_counts")
        || !valid_count_map("local_transcript_consequence_counts")
        || count_total("variant_class_counts") != emitted
        || reference_validation_total != emitted
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "VCF aggregate contains invalid bounded count maps".into(),
        ));
    }
    let payload = serde_json::to_string(&input.report).map_err(internal)?;
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let item = store
        .add_research_input_artifact(
            &id,
            "genomics.analyze",
            "genomic_dataset",
            "MedAI local VCF aggregate receipt; variant and sample data remain local",
            input_sha256,
            &payload,
        )
        .map_err(|error| {
            let message = error.to_string();
            let status = if message.contains("insufficient execution credits") {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, message)
        })?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}
async fn review_drug_interaction_study(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<DrugInteractionStudyReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "study adjudication requires researcher or platform_admin role".into(),
        ));
    }
    let fields = [
        (&input.study_design, 120),
        (&input.exposure_status, 32),
        (&input.comparator, 1000),
        (&input.dose_a, 256),
        (&input.dose_b, 256),
        (&input.route_a, 256),
        (&input.route_b, 256),
        (&input.endpoint, 1000),
        (&input.outcome, 2000),
        (&input.limitations, 2000),
        (&input.label_evidence_overlap, 1000),
    ];
    if fields
        .iter()
        .any(|(value, max)| value.trim().is_empty() || value.len() > *max)
        || !matches!(
            input.exposure_status.as_str(),
            "both" | "drug_a_only" | "drug_b_only" | "neither" | "unclear"
        )
    {
        return Err((StatusCode::BAD_REQUEST, "review fields must be non-empty and within limits; exposure_status must be both, drug_a_only, drug_b_only, neither, or unclear".into()));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let screen = artifacts
        .iter()
        .find(|item| {
            item.id == input.screen_artifact_id
                && item.kind == "drug_interaction_pmc_fulltext_screen"
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screen_artifact_id must identify a full-text review screen in this session".into(),
        ))?;
    if format!("{:x}", Sha256::digest(screen.payload.as_bytes())) != screen.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "full-text screen digest does not match its retained bytes".into(),
        ));
    }
    let screen_data: serde_json::Value = serde_json::from_str(&screen.payload).map_err(internal)?;
    if screen_data.get("interaction_verdict") != Some(&serde_json::Value::Null) {
        return Err((
            StatusCode::BAD_REQUEST,
            "full-text screen is not an unadjudicated source receipt".into(),
        ));
    }
    let candidates = screen_data
        .get("candidates")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screen has no retained candidate locations".into(),
        ))?;
    let lead = candidates
        .iter()
        .find(|candidate| {
            candidate
                .get("candidate_id")
                .and_then(serde_json::Value::as_str)
                == Some(input.candidate_id.as_str())
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "candidate_id is not present in this screen".into(),
        ))?;
    let source_artifact_id = screen_data
        .get("candidate_artifact_id")
        .and_then(serde_json::Value::as_str)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screen has no linked PubMed candidate receipt".into(),
        ))?;
    let source = artifacts
        .iter()
        .find(|item| {
            item.id == source_artifact_id
                && item.kind == "drug_interaction_literature_candidate"
                && item.task_id == screen.task_id
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "linked PubMed candidate receipt is missing or belongs to another task".into(),
        ))?;
    if format!("{:x}", Sha256::digest(source.payload.as_bytes())) != source.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "linked PubMed candidate digest does not match its retained bytes".into(),
        ));
    }
    if screen_data
        .get("candidate_artifact_sha256")
        .and_then(serde_json::Value::as_str)
        != Some(source.sha256.as_str())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "full-text screen is not bound to the exact PubMed candidate digest".into(),
        ));
    }
    let source_data: serde_json::Value = serde_json::from_str(&source.payload).map_err(internal)?;
    let screen_pmid = screen_data.get("pmid").and_then(serde_json::Value::as_str);
    let source_pmid = source_data.get("pmid").and_then(serde_json::Value::as_str);
    let source_pmcid = source_data
        .pointer("/identifiers/pmc")
        .and_then(serde_json::Value::as_str);
    if screen_pmid != source_pmid
        || source_pmcid.map(str::to_ascii_uppercase).as_deref()
            != screen_data
                .get("pmcid")
                .and_then(serde_json::Value::as_str)
                .map(str::to_ascii_uppercase)
                .as_deref()
        || lead
            .get("text_sha256")
            .and_then(serde_json::Value::as_str)
            .is_none()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "screen location does not bind to the same PMID, PMCID, and source passage receipt"
                .into(),
        ));
    }
    let payload = serde_json::json!({
        "schema_version": 1, "review_type": "researcher_fulltext_study_adjudication", "research_only": true,
        "clinical_advice": false, "interaction_verdict": null,
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "screen_artifact_id": screen.id, "screen_sha256": screen.sha256,
        "candidate_artifact_id": source.id, "candidate_artifact_sha256": source.sha256,
        "candidate_id": input.candidate_id, "pmid": screen_pmid, "pmcid": screen_data.get("pmcid"),
        "passage_sha256": lead.get("text_sha256"), "source_sha256": screen_data.get("source_sha256"),
        "study_design": input.study_design.trim(), "exposure_status": input.exposure_status,
        "comparator": input.comparator.trim(), "dose_a": input.dose_a.trim(), "dose_b": input.dose_b.trim(),
        "route_a": input.route_a.trim(), "route_b": input.route_b.trim(), "endpoint": input.endpoint.trim(),
        "outcome": input.outcome.trim(), "limitations": input.limitations.trim(),
        "label_evidence_overlap": input.label_evidence_overlap.trim(),
        "interpretation_limit": "Researcher-authored study annotation only. It does not determine a clinical interaction or provide treatment advice."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &screen.task_id,
            "drug_interaction_study_adjudication",
            "authenticated researcher full-text review",
            &input.candidate_id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}

fn source_span_matches(text: &str, offset: usize, length: usize, expected_sha256: &str) -> bool {
    let mut chars = text.chars();
    let Some(start) = chars.nth(offset) else {
        return false;
    };
    let mut span = String::new();
    span.push(start);
    span.extend(chars.take(length.saturating_sub(1)));
    span.chars().count() == length
        && format!("{:x}", Sha256::digest(span.as_bytes())) == expected_sha256
}

fn label_span_matches(
    payload: &serde_json::Value,
    set_id: &str,
    version: &str,
    offset: usize,
    length: usize,
    expected_sha256: &str,
) -> bool {
    let Some(labels) = payload.get("labels").and_then(serde_json::Value::as_object) else {
        return false;
    };
    labels.values().any(|rows| {
        rows.as_array().is_some_and(|rows| {
            rows.iter().any(|row| {
                let row_set = row.get("set_id").and_then(serde_json::Value::as_str);
                let row_version = row
                    .get("version")
                    .or_else(|| row.get("spl_version"))
                    .and_then(serde_json::Value::as_str);
                row_set == Some(set_id)
                    && row_version == Some(version)
                    && row
                        .get("drug_interactions")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|text| {
                            source_span_matches(text, offset, length, expected_sha256)
                        })
            })
        })
    })
}

async fn review_drug_interaction_claim(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<DrugInteractionClaimReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "interaction claim review requires researcher or platform_admin role".into(),
        ));
    }
    let allowed = |value: &str, choices: &[&str]| choices.contains(&value);
    if !allowed(
        &input.adjudication,
        &[
            "same_question_consistent",
            "same_question_opposing",
            "related_but_distinct",
            "unrelated",
            "uncertain",
        ],
    ) || !allowed(
        &input.claim_domain,
        &[
            "pharmacokinetic",
            "pharmacodynamic",
            "label_scope",
            "uncertain",
        ],
    ) || !allowed(
        &input.exposure_comparability,
        &["comparable", "different", "unclear", "not_reported"],
    ) || !allowed(
        &input.outcome_comparability,
        &["comparable", "different", "unclear", "not_reported"],
    ) || input.rationale.trim().is_empty()
        || input.rationale.len() > 2000
        || input.context_notes.trim().is_empty()
        || input.context_notes.len() > 4000
        || input.review_index > 9
    {
        return Err((StatusCode::BAD_REQUEST, "claim review labels must be supported; rationale and context notes are required and bounded; review index must be 0..9".into()));
    }

    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let screening = artifacts
        .iter()
        .find(|item| {
            item.id == input.screening_artifact_id
                && item.kind == "drug_interaction_semantic_review"
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screening_artifact_id must identify a retained drug-interaction semantic screen"
                .into(),
        ))?;
    if format!("{:x}", Sha256::digest(screening.payload.as_bytes())) != screening.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "semantic screening artifact digest does not match its retained bytes".into(),
        ));
    }
    let screen_payload: serde_json::Value =
        serde_json::from_str(&screening.payload).map_err(internal)?;
    if screen_payload.get("interaction_verdict") != Some(&serde_json::Value::Null) {
        return Err((
            StatusCode::BAD_REQUEST,
            "interaction screening must retain a null verdict".into(),
        ));
    }
    let review = screen_payload
        .get("reviews")
        .and_then(serde_json::Value::as_array)
        .and_then(|reviews| reviews.get(input.review_index))
        .ok_or((
            StatusCode::BAD_REQUEST,
            "review_index is not present in the selected screening artifact".into(),
        ))?;
    let sources = review
        .get("sources")
        .and_then(serde_json::Value::as_array)
        .filter(|sources| sources.len() == 2)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screened claim must contain exactly two source spans".into(),
        ))?;

    let mut verified_sources = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        let artifact_id = source
            .get("artifact_id")
            .and_then(serde_json::Value::as_str)
            .ok_or((
                StatusCode::BAD_REQUEST,
                "screened source has no artifact ID".into(),
            ))?;
        let artifact = artifacts
            .iter()
            .find(|item| item.id == artifact_id && item.task_id == screening.task_id)
            .ok_or((
                StatusCode::BAD_REQUEST,
                "screened source artifact is missing or belongs to another task".into(),
            ))?;
        let digest = source
            .get("artifact_sha256")
            .and_then(serde_json::Value::as_str);
        if digest != Some(artifact.sha256.as_str())
            || format!("{:x}", Sha256::digest(artifact.payload.as_bytes())) != artifact.sha256
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "screened source artifact digest does not match its retained bytes".into(),
            ));
        }
        let payload: serde_json::Value =
            serde_json::from_str(&artifact.payload).map_err(internal)?;
        let expected_kind = if index == 0 {
            matches!(
                artifact.kind.as_str(),
                "drug_interaction_label_screen" | "dailymed_interaction_label_screen"
            )
        } else {
            (artifact.kind == "drug_interaction_literature_candidate"
                && artifact.source == "PubMed")
                || (artifact.kind == "drug_interaction_pmc_fulltext_screen"
                    && artifact.source == "PMC BioC hash-and-offset lexical review leads")
        };
        let expected_field = if index == 0 {
            "labels.drug_interactions"
        } else if artifact.kind == "drug_interaction_pmc_fulltext_screen" {
            "full_text"
        } else {
            "abstract"
        };
        let offset = source
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok());
        let length = source
            .get("length")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok());
        let span_hash = source
            .get("text_sha256")
            .and_then(serde_json::Value::as_str);
        let candidate_id = source
            .get("candidate_id")
            .and_then(serde_json::Value::as_str);
        if !expected_kind
            || source.get("source").and_then(serde_json::Value::as_str)
                != Some(artifact.source.as_str())
            || source.get("source_id").and_then(serde_json::Value::as_str)
                != Some(artifact.source_id.as_str())
            || source.get("field").and_then(serde_json::Value::as_str) != Some(expected_field)
            || offset.is_none()
            || length.is_none_or(|length| length == 0 || length > 1200)
            || span_hash.is_none_or(|hash| {
                hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
            || candidate_id.is_none()
        {
            return Err((
                StatusCode::BAD_REQUEST,
                "screened source location has invalid identity, type, field, or bounds".into(),
            ));
        }
        let span_matches = if index == 0 {
            let set_id = source.get("set_id").and_then(serde_json::Value::as_str);
            let version = source.get("version").and_then(serde_json::Value::as_str);
            matches!((set_id, version, offset, length, span_hash), (Some(set_id), Some(version), Some(offset), Some(length), Some(span_hash))
                if label_span_matches(&payload, set_id, version, offset, length, span_hash))
        } else if artifact.kind == "drug_interaction_pmc_fulltext_screen" {
            let pmcid = payload.get("pmcid").and_then(serde_json::Value::as_str);
            let source_sha256 = payload
                .get("source_sha256")
                .and_then(serde_json::Value::as_str);
            let screen_digest = source
                .get("source_sha256")
                .and_then(serde_json::Value::as_str);
            let expected_candidate_id = match (source_sha256, offset, span_hash) {
                (Some(source_hash), Some(offset), Some(span_hash)) => Some(format!(
                    "{:x}",
                    Sha256::digest(format!("{source_hash}\0{offset}\0{span_hash}").as_bytes())
                )),
                _ => None,
            };
            let candidate = payload
                .get("candidates")
                .and_then(serde_json::Value::as_array)
                .and_then(|rows| {
                    rows.iter().find(|row| {
                        row.get("candidate_id").and_then(serde_json::Value::as_str) == candidate_id
                            && candidate_id == expected_candidate_id.as_deref()
                            && row.get("source_offset").and_then(serde_json::Value::as_u64)
                                == offset.and_then(|value| u64::try_from(value).ok())
                            && row.get("source_length").and_then(serde_json::Value::as_u64)
                                == length.and_then(|value| u64::try_from(value).ok())
                            && row.get("text_sha256").and_then(serde_json::Value::as_str)
                                == span_hash
                    })
                });
            let linked_id = payload
                .get("candidate_artifact_id")
                .and_then(serde_json::Value::as_str);
            let linked_hash = payload
                .get("candidate_artifact_sha256")
                .and_then(serde_json::Value::as_str);
            let linked = linked_id.and_then(|linked_id| {
                artifacts.iter().find(|item| {
                    item.id == linked_id
                        && item.task_id == screening.task_id
                        && item.kind == "drug_interaction_literature_candidate"
                        && item.source == "PubMed"
                        && Some(item.sha256.as_str()) == linked_hash
                        && format!("{:x}", Sha256::digest(item.payload.as_bytes())) == item.sha256
                })
            });
            let linked_payload = linked
                .and_then(|item| serde_json::from_str::<serde_json::Value>(&item.payload).ok());
            pmcid == Some(artifact.source_id.as_str())
                && source_sha256.is_some()
                && source_sha256 == screen_digest
                && payload.get("status").and_then(serde_json::Value::as_str)
                    == Some("review_leads_available")
                && candidate.is_some()
                && linked_payload.as_ref().is_some_and(|paper| {
                    paper.get("pmid").and_then(serde_json::Value::as_str)
                        == payload.get("pmid").and_then(serde_json::Value::as_str)
                        && paper
                            .get("identifiers")
                            .and_then(|ids| ids.get("pmc"))
                            .and_then(serde_json::Value::as_str)
                            == Some(artifact.source_id.as_str())
                })
        } else {
            let pmid = payload.get("pmid").and_then(serde_json::Value::as_str);
            let abstract_text = payload.get("abstract").and_then(serde_json::Value::as_str);
            pmid == Some(artifact.source_id.as_str())
                && matches!((abstract_text, offset, length, span_hash), (Some(text), Some(offset), Some(length), Some(hash))
                    if source_span_matches(text, offset, length, hash))
        };
        if !span_matches {
            return Err((
                StatusCode::BAD_REQUEST,
                "source claim span does not match the retained versioned source text".into(),
            ));
        }
        verified_sources.push(serde_json::json!({
            "artifact_id": artifact.id, "artifact_sha256": artifact.sha256,
            "source": artifact.source, "source_id": artifact.source_id,
            "kind": artifact.kind, "candidate_id": candidate_id,
            "field": expected_field, "offset": offset, "length": length,
            "text_sha256": span_hash,
            "set_id": source.get("set_id"), "version": source.get("version"),
            "source_sha256": source.get("source_sha256"),
        }));
    }
    if review
        .get("candidate_a")
        .and_then(serde_json::Value::as_str)
        != sources[0]
            .get("candidate_id")
            .and_then(serde_json::Value::as_str)
        || review
            .get("candidate_b")
            .and_then(serde_json::Value::as_str)
            != sources[1]
                .get("candidate_id")
                .and_then(serde_json::Value::as_str)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "screening candidate IDs do not match their source spans".into(),
        ));
    }
    let payload = serde_json::json!({
        "schema_version": 1,
        "review_type": "human_drug_interaction_claim_adjudication",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "screening_artifact_id": screening.id,
        "screening_sha256": screening.sha256,
        "review_index": input.review_index,
        "source_claims": verified_sources,
        "model_screening": review,
        "adjudication": input.adjudication,
        "claim_domain": input.claim_domain,
        "exposure_comparability": input.exposure_comparability,
        "outcome_comparability": input.outcome_comparability,
        "rationale": input.rationale.trim(),
        "context_notes": input.context_notes.trim(),
        "interaction_verdict": null,
        "confirmed_contradiction": false,
            "interpretation_limit": "Human adjudication binds one categorical screening to exact retained PubMed or PMC and versioned label spans. It does not establish a drug interaction, clinical significance, or treatment recommendation.",
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &screening.task_id,
            "drug_interaction_claim_adjudication",
            "authenticated researcher drug-interaction claim review",
            &screening.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}

async fn review_hypothesis_literature(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<HypothesisLiteratureReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "literature relevance review requires researcher or platform_admin role".into(),
        ));
    }
    if !matches!(
        input.relevance.as_str(),
        "directly_relevant" | "partially_relevant" | "unrelated" | "unclear"
    ) {
        return Err((
            StatusCode::BAD_REQUEST,
            "relevance must be directly_relevant, partially_relevant, unrelated, or unclear".into(),
        ));
    }
    let fields = [
        (&input.rationale, 2000),
        (&input.study_design, 120),
        (&input.population_or_model, 500),
        (&input.methods, 2000),
        (&input.findings, 2000),
        (&input.limitations, 2000),
    ];
    if fields
        .iter()
        .any(|(value, max)| value.trim().is_empty() || value.len() > *max)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "all literature review fields must be non-empty and within their size limits".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let comparison = artifacts
        .iter()
        .find(|item| {
            item.id == input.comparison_artifact_id
                && item.kind == "hypothesis_literature_comparison"
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "comparison_artifact_id must identify a literature comparison in this session".into(),
        ))?;
    let paper = artifacts
        .iter()
        .find(|item| {
            item.id == input.paper_artifact_id
                && item.kind == "hypothesis_literature_candidate"
                && item.task_id == comparison.task_id
                && item.source.starts_with("PubMed")
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "paper_artifact_id must identify a PubMed candidate from the comparison task".into(),
        ))?;
    if format!("{:x}", Sha256::digest(comparison.payload.as_bytes())) != comparison.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "hypothesis literature comparison digest does not match its retained bytes".into(),
        ));
    }
    if format!("{:x}", Sha256::digest(paper.payload.as_bytes())) != paper.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "hypothesis literature candidate digest does not match its retained bytes".into(),
        ));
    }
    let comparison_data: serde_json::Value =
        serde_json::from_str(&comparison.payload).map_err(internal)?;
    let paper_data: serde_json::Value = serde_json::from_str(&paper.payload).map_err(internal)?;
    let pmid = paper_data.get("pmid").and_then(serde_json::Value::as_str);
    if pmid.map(|value| value != paper.source_id).unwrap_or(true)
        || paper_data.get("abstract_retained") != Some(&serde_json::Value::Bool(false))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "PubMed source identity or no-abstract retention invariant failed".into(),
        ));
    }
    let query_sha256 = paper_data
        .get("query_sha256")
        .and_then(serde_json::Value::as_str);
    let hypothesis_id = paper_data
        .get("hypothesis_id")
        .and_then(serde_json::Value::as_str);
    let record_sha256 = paper_data
        .get("record_sha256")
        .and_then(serde_json::Value::as_str);
    if !record_sha256.is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        return Err((
            StatusCode::BAD_REQUEST,
            "PubMed candidate source-record digest is malformed".into(),
        ));
    }
    let searches = comparison_data
        .get("candidate_searches")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "comparison has no candidate search list".into(),
        ))?;
    let linked = searches.iter().any(|search| {
        if search
            .get("hypothesis_id")
            .and_then(serde_json::Value::as_str)
            != hypothesis_id
            || search
                .get("query_sha256")
                .and_then(serde_json::Value::as_str)
                != query_sha256
        {
            return false;
        }
        search
            .get("candidate_leads")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|leads| {
                leads.iter().any(|lead| {
                    lead.get("artifact_id").and_then(serde_json::Value::as_str)
                        == Some(paper.id.as_str())
                        && lead.get("pmid").and_then(serde_json::Value::as_str) == pmid
                        && lead
                            .get("record_sha256")
                            .and_then(serde_json::Value::as_str)
                            == record_sha256
                })
            })
    });
    if !linked || hypothesis_id.is_none() || query_sha256.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "paper candidate is not linked to this hypothesis and exact query receipt".into(),
        ));
    }
    let payload = serde_json::json!({
        "schema_version": 1,
        "review_type": "human_hypothesis_literature_relevance_review",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "hypothesis_id": hypothesis_id,
        "comparison_artifact_id": comparison.id,
        "comparison_sha256": comparison.sha256,
        "paper_artifact_id": paper.id,
        "paper_sha256": paper.sha256,
        "query_sha256": query_sha256,
        "pmid": pmid,
        "relevance": input.relevance,
        "rationale": input.rationale.trim(),
        "study_design": input.study_design.trim(),
        "population_or_model": input.population_or_model.trim(),
        "methods": input.methods.trim(),
        "findings": input.findings.trim(),
        "limitations": input.limitations.trim(),
        "novelty_assessment": null,
        "interpretation_limit": "Human relevance annotation for one retrieved literature lead. It does not establish source-complete literature coverage, absence of prior work, or hypothesis novelty."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &comparison.task_id,
            "hypothesis_literature_relevance_review",
            "authenticated researcher literature review",
            &paper.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}
async fn review_hypothesis_critic(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<HypothesisCriticReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "hypothesis criticism review requires researcher or platform_admin role".into(),
        ));
    }
    if !matches!(
        input.assessment.as_str(),
        "supports_hypothesis" | "weakens_hypothesis" | "context_only" | "irrelevant" | "uncertain"
    ) || !matches!(
        input.item_type.as_str(),
        "evidence" | "retrieved_candidate" | "hypothesis_review"
    ) || input.rationale.trim().is_empty()
        || input.rationale.len() > 2000
        || input.study_context.trim().is_empty()
        || input.study_context.len() > 4000
        || input.item_id.trim().is_empty()
        || input.item_id.len() > 160
    {
        return Err((StatusCode::BAD_REQUEST, "assessment/item type must use supported labels; item ID, rationale, and study context are required and bounded".into()));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let criticism = artifacts
        .iter()
        .find(|item| item.id == input.criticism_artifact_id && item.kind == "hypothesis_criticism")
        .ok_or((
            StatusCode::BAD_REQUEST,
            "criticism_artifact_id must identify a criticism artifact in this session".into(),
        ))?;
    if format!("{:x}", Sha256::digest(criticism.payload.as_bytes())) != criticism.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "criticism artifact payload hash does not match its retained digest".into(),
        ));
    }
    let criticism_data: serde_json::Value =
        serde_json::from_str(&criticism.payload).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                "criticism artifact payload is malformed".into(),
            )
        })?;
    let proposal_id = criticism_data
        .get("proposal_artifact_id")
        .and_then(serde_json::Value::as_str);
    let proposal = artifacts
        .iter()
        .find(|item| Some(item.id.as_str()) == proposal_id && item.kind == "hypothesis_candidates");
    if proposal_id != Some(criticism.source_id.as_str())
        || !proposal.is_some_and(|item| {
            format!("{:x}", Sha256::digest(item.payload.as_bytes())) == item.sha256
                && criticism_data
                    .get("proposal_artifact_sha256")
                    .and_then(serde_json::Value::as_str)
                    == Some(item.sha256.as_str())
        })
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "criticism artifact is not bound to a retained hypothesis proposal".into(),
        ));
    }
    let reviews = criticism_data
        .get("reviews")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "criticism artifact has no review list".into(),
        ))?;
    let review = reviews
        .iter()
        .find(|review| {
            review
                .get("hypothesis_id")
                .and_then(serde_json::Value::as_str)
                == Some(input.hypothesis_id.as_str())
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "hypothesis is not present in the selected criticism artifact".into(),
        ))?;
    let item_exists = match input.item_type.as_str() {
        "hypothesis_review" => input.item_id == input.hypothesis_id,
        "evidence" => ["linked_contradictory_evidence", "lexical_refutation_leads"]
            .iter()
            .any(|field| {
                review
                    .get(*field)
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            item.get("evidence_id").and_then(serde_json::Value::as_str)
                                == Some(input.item_id.as_str())
                                && store
                                    .evidence(&id)
                                    .map(|records| {
                                        records.iter().any(|record| record.id == input.item_id)
                                    })
                                    .unwrap_or(false)
                        })
                    })
            }),
        "retrieved_candidate" => review
            .get("retrieved_contradiction_candidates")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    let candidate_id = item
                        .get("candidate_artifact_id")
                        .and_then(serde_json::Value::as_str)
                        .filter(|candidate_id| *candidate_id == input.item_id);
                    let candidate_digest = item
                        .get("candidate_artifact_sha256")
                        .and_then(serde_json::Value::as_str)
                        .filter(|digest| {
                            digest.len() == 64
                                && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                        });
                    candidate_id
                        .zip(candidate_digest)
                        .is_some_and(|(candidate_id, digest)| {
                            artifacts.iter().any(|candidate| {
                                if candidate.id != candidate_id
                                    || candidate.kind != "contradiction_search_candidate"
                                    || candidate.source != "PubMed"
                                    || candidate.source_id.is_empty()
                                    || !candidate.source_id.bytes().all(|byte| byte.is_ascii_digit())
                                    || candidate.sha256 != digest
                                    || format!("{:x}", Sha256::digest(candidate.payload.as_bytes())) != candidate.sha256
                                {
                                    return false;
                                }
                                let Ok(payload) = serde_json::from_str::<serde_json::Value>(&candidate.payload) else {
                                    return false;
                                };
                                payload.get("target_evidence_id")
                                    == item.get("target_evidence_id")
                                    && payload.get("interpretation").and_then(serde_json::Value::as_str)
                                        == Some("Retrieved as a contradiction-search candidate only; no contradictory relation or evidence polarity was inferred.")
                                    && payload.get("paper").and_then(|paper| paper.get("provenance"))
                                        .is_some_and(|provenance| {
                                            provenance.get("source").and_then(serde_json::Value::as_str) == Some("PubMed")
                                                && provenance.get("source_id").and_then(serde_json::Value::as_str) == Some(candidate.source_id.as_str())
                                        })
                            })
                        })
                })
            }),
        _ => false,
    };
    if !item_exists {
        return Err((
            StatusCode::BAD_REQUEST,
            "review item is not linked to this hypothesis in the selected criticism artifact"
                .into(),
        ));
    }
    let payload = serde_json::json!({
        "schema_version": 1,
        "review_type": "human_hypothesis_critic_adjudication",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "hypothesis_id": input.hypothesis_id,
        "criticism_artifact_id": criticism.id,
        "criticism_sha256": criticism.sha256,
        "item_type": input.item_type,
        "item_id": input.item_id,
        "assessment": input.assessment,
        "rationale": input.rationale.trim(),
        "study_context": input.study_context.trim(),
        "hypothesis_confidence_changed": false,
        "interpretation_limit": "Human review of a source-linked scientific critic lead. This annotation does not independently establish contradiction or change the candidate hypothesis or its confidence."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &criticism.task_id,
            "hypothesis_critic_adjudication",
            "authenticated researcher criticism review",
            &criticism.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}

async fn review_semantic_claim_alignment(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<SemanticClaimReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "researcher" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "claim alignment review requires researcher or platform_admin role".into(),
        ));
    }
    let allowed = |value: &str, choices: &[&str]| choices.contains(&value);
    if !allowed(
        &input.adjudication,
        &[
            "same_question_consistent",
            "same_question_opposing",
            "related_but_distinct",
            "unrelated",
            "uncertain",
        ],
    ) || !allowed(
        &input.population_comparability,
        &["comparable", "different", "unclear", "not_reported"],
    ) || !allowed(
        &input.method_comparability,
        &["comparable", "different", "unclear", "not_reported"],
    ) || input.rationale.trim().is_empty()
        || input.rationale.len() > 2000
        || input.methods_population_notes.trim().is_empty()
        || input.methods_population_notes.len() > 4000
    {
        return Err((StatusCode::BAD_REQUEST, "adjudication and comparability must use supported labels; rationale and methods/population notes are required and bounded".into()));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let screen = artifacts
        .iter()
        .find(|item| {
            item.id == input.screening_artifact_id && item.kind == "paper_claim_semantic_alignment"
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "screening_artifact_id must identify a semantic claim screening in this session".into(),
        ))?;
    if format!("{:x}", Sha256::digest(screen.payload.as_bytes())) != screen.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "semantic screening artifact digest does not match its retained bytes".into(),
        ));
    }
    let data: serde_json::Value = serde_json::from_str(&screen.payload).map_err(internal)?;
    let assessment = data.get("assessment").ok_or((
        StatusCode::BAD_REQUEST,
        "semantic screening has no assessment".into(),
    ))?;
    let sources = data
        .get("sources")
        .and_then(serde_json::Value::as_array)
        .filter(|rows| rows.len() == 2)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "semantic screening must link exactly two source candidates".into(),
        ))?;
    let candidate_a = assessment
        .get("candidate_a")
        .and_then(serde_json::Value::as_str);
    let candidate_b = assessment
        .get("candidate_b")
        .and_then(serde_json::Value::as_str);
    let source_candidate_ids: Vec<_> = sources
        .iter()
        .filter_map(|source| {
            source
                .get("candidate_id")
                .and_then(serde_json::Value::as_str)
        })
        .collect();
    if candidate_a.is_none()
        || candidate_b.is_none()
        || source_candidate_ids.len() != 2
        || !source_candidate_ids.contains(&candidate_a.unwrap())
        || !source_candidate_ids.contains(&candidate_b.unwrap())
        || sources.iter().any(|source| {
            ["source_sha256", "candidate_text_sha256"]
                .iter()
                .any(|key| {
                    source
                        .get(*key)
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(|digest| {
                            digest.len() != 64
                                || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                })
        })
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "screening source anchors do not match the two assessed candidates".into(),
        ));
    }
    let annotation_receipts: Vec<_> = sources
        .iter()
        .map(|source| {
            let artifact_id = source
                .get("annotation_artifact_id")
                .and_then(serde_json::Value::as_str)
                .ok_or((
                    StatusCode::BAD_REQUEST,
                    "claim source is missing its annotation artifact ID".into(),
                ))?;
            let annotation = artifacts
                .iter()
                .find(|item| {
                    item.id == artifact_id
                        && item.kind == "paper_result_annotations"
                        && item.task_id == screen.task_id
                })
                .ok_or((
                    StatusCode::BAD_REQUEST,
                    "claim source annotation artifact is missing or belongs to another task".into(),
                ))?;
            if format!("{:x}", Sha256::digest(annotation.payload.as_bytes())) != annotation.sha256 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "claim source annotation artifact digest does not match its retained bytes".into(),
                ));
            }
            let annotation_data: serde_json::Value =
                serde_json::from_str(&annotation.payload).map_err(internal)?;
            let candidate_id = source
                .get("candidate_id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let candidate_hash = source
                .get("candidate_text_sha256")
                .and_then(serde_json::Value::as_str);
            let recorded_candidate_hash = annotation_data
                .get("candidate_text_hashes")
                .and_then(|hashes| hashes.get(candidate_id))
                .and_then(serde_json::Value::as_str);
            if annotation_data
                .get("document_id")
                .and_then(serde_json::Value::as_str)
                != source.get("document_id").and_then(serde_json::Value::as_str)
                || annotation_data
                    .get("source_sha256")
                    .and_then(serde_json::Value::as_str)
                    != source.get("source_sha256").and_then(serde_json::Value::as_str)
                || recorded_candidate_hash != candidate_hash
            {
                return Err((
                    StatusCode::BAD_REQUEST,
                    "claim source document or candidate hash does not match its retained annotation receipt".into(),
                ));
            }
            Ok(annotation.id.as_str())
        })
        .collect::<Result<_, _>>()?;
    if annotation_receipts.len() != 2 || annotation_receipts[0] == annotation_receipts[1] {
        return Err((
            StatusCode::BAD_REQUEST,
            "claim adjudication requires two distinct retained PMC annotation artifacts".into(),
        ));
    }
    let source_refs: Vec<serde_json::Value> = sources.iter().map(|source| serde_json::json!({
        "annotation_artifact_id": source.get("annotation_artifact_id"), "pmc_id": source.get("pmc_id"),
        "document_id": source.get("document_id"), "source_sha256": source.get("source_sha256"),
        "candidate_id": source.get("candidate_id"), "candidate_text_sha256": source.get("candidate_text_sha256"),
        "passage_index": source.get("passage_index"), "sentence_index": source.get("sentence_index"),
        "source_offset": source.get("source_offset"), "source_local_offset": source.get("source_local_offset"),
        "source_length": source.get("source_length"), "license": source.get("license")
    })).collect();
    let payload = serde_json::json!({
        "schema_version": 1, "review_type": "human_semantic_claim_alignment_adjudication",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "screening_artifact_id": screen.id, "screening_sha256": screen.sha256,
        "source_candidates": source_refs, "model_screening": assessment,
        "adjudication": input.adjudication, "population_comparability": input.population_comparability,
        "method_comparability": input.method_comparability, "rationale": input.rationale.trim(),
        "methods_population_notes": input.methods_population_notes.trim(),
        "confirmed_contradiction": false,
        "interpretation_limit": "Researcher adjudication of source-anchored claim relation and study comparability. Opposing claims are not by themselves a confirmed contradiction; evaluate outcomes, uncertainty, design, and context before scientific synthesis."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &screen.task_id,
            "paper_claim_alignment_adjudication",
            "authenticated researcher claim alignment review",
            &screen.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}

fn valid_disease_curie(value: &str) -> bool {
    let valid_local = |prefix: &str, local: &str| {
        local.len() >= 7
            && local.len() <= 16
            && local.bytes().all(|byte| byte.is_ascii_digit())
            && !prefix.is_empty()
    };
    if let Some(local) = value.strip_prefix("EFO_") {
        return valid_local("EFO", local);
    }
    if let Some(local) = value.strip_prefix("MONDO:") {
        return local.len() == 7 && local.bytes().all(|byte| byte.is_ascii_digit());
    }
    false
}

async fn review_ontology_cross_mapping(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<OntologyCrossMappingReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if principal.scientific_agent_id.is_some()
        || !matches!(principal.role.as_str(), "researcher" | "platform_admin")
    {
        return Err((
            StatusCode::FORBIDDEN,
            "ontology mapping review requires researcher or platform_admin role".into(),
        ));
    }
    if !valid_disease_curie(&input.source_identifier)
        || input.target_curie.len() > 128
        || input.target_curie.trim() != input.target_curie
        || input.rationale.trim().is_empty()
        || input.rationale.len() > 2000
        || !matches!(
            input.decision.as_str(),
            "same_concept" | "related_but_distinct" | "incorrect_candidate" | "uncertain"
        )
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "source and target identifiers, decision, and rationale must be valid and bounded"
                .into(),
        ));
    }
    let target_parts = input
        .target_curie
        .split_once(':')
        .or_else(|| input.target_curie.split_once('_'));
    let (target_prefix, target_local) = target_parts.ok_or((
        StatusCode::BAD_REQUEST,
        "target_curie must be a prefixed stable identifier".into(),
    ))?;
    if target_prefix.is_empty()
        || target_prefix.len() > 32
        || target_local.is_empty()
        || !target_prefix.bytes().all(|byte| {
            byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
        || !target_local
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "target_curie must be a prefixed stable identifier".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let artifacts = store.artifacts(&id).map_err(internal)?;
    let source = artifacts
        .iter()
        .find(|artifact| {
            artifact.id == input.source_artifact_id
                && artifact.kind == "ontology_cross_mapping_review"
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "source_artifact_id must identify an OxO mapping artifact in this session".into(),
        ))?;
    if format!("{:x}", Sha256::digest(source.payload.as_bytes())) != source.sha256 {
        return Err((
            StatusCode::BAD_REQUEST,
            "OxO mapping artifact payload digest failed verification".into(),
        ));
    }
    let data: serde_json::Value = serde_json::from_str(&source.payload).map_err(internal)?;
    let source_response_sha256 = data
        .get("source_response_sha256")
        .and_then(serde_json::Value::as_str);
    if data.get("source")
        != Some(&serde_json::Value::String(
            "EMBL-EBI Ontology Xref Service (OxO)".into(),
        ))
        || data
            .get("source_identifier")
            .and_then(serde_json::Value::as_str)
            != Some(input.source_identifier.as_str())
        || !source_response_sha256.is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "OxO mapping source identity or response digest does not match this review".into(),
        ));
    }
    let candidates = data
        .get("mappings")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "OxO artifact has no mapping candidate list".into(),
        ))?;
    let candidate = candidates
        .iter()
        .find(|candidate| {
            candidate
                .get("target_curie")
                .and_then(serde_json::Value::as_str)
                == Some(input.target_curie.as_str())
                && candidate
                    .get("target_prefix")
                    .and_then(serde_json::Value::as_str)
                    == Some(target_prefix)
                && candidate
                    .get("distance")
                    .and_then(serde_json::Value::as_u64)
                    == Some(1)
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "target_curie is not a direct candidate in the selected OxO artifact".into(),
        ))?;
    let payload = serde_json::json!({
        "schema_version": 1,
        "review_type": "human_ontology_cross_mapping_adjudication",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "source_artifact_id": source.id,
        "source_artifact_sha256": source.sha256,
        "source_response_sha256": source_response_sha256,
        "source_identifier": input.source_identifier,
        "target_curie": input.target_curie,
        "candidate": candidate,
        "decision": input.decision,
        "rationale": input.rationale.trim(),
        "ontology_mapping_applied": false,
        "interpretation_limit": "Authenticated human adjudication of one exact OxO candidate. This record does not automatically merge ontology identifiers, rewrite source data, or propagate disease equivalence to other assertions."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let item = store
        .add_human_review_artifact(
            &id,
            &source.task_id,
            "ontology_cross_mapping_adjudication",
            "authenticated researcher ontology mapping review",
            &source.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(item).map_err(internal)?),
    ))
}

fn valid_citation_doi(value: &str) -> bool {
    if value.len() > 255 || !value.is_ascii() {
        return false;
    }
    let Some((registrant, suffix)) = value.split_once('/') else {
        return false;
    };
    let Some(registrant) = registrant.strip_prefix("10.") else {
        return false;
    };
    (4..=9).contains(&registrant.len())
        && registrant.bytes().all(|byte| byte.is_ascii_digit())
        && !suffix.is_empty()
        && suffix
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-._;()/:".contains(&byte))
}

async fn review_crossref_reference(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<CrossrefReferenceReview>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if principal.scientific_agent_id.is_some()
        || !matches!(principal.role.as_str(), "researcher" | "platform_admin")
    {
        return Err((
            StatusCode::FORBIDDEN,
            "citation reference review requires researcher or platform_admin role".into(),
        ));
    }
    let rationale = input.rationale.trim();
    if !matches!(
        input.decision.as_str(),
        "match" | "not_a_match" | "uncertain"
    ) || rationale.is_empty()
        || rationale.len() > 2000
        || !valid_citation_doi(&input.citing_doi)
        || !(0..200).contains(&input.reference_index)
        || input.candidate_artifact_id.is_empty()
        || input.candidate_artifact_id.len() > 128
        || input.candidate_artifact_sha256.len() != 64
        || !input
            .candidate_artifact_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || input.reference_sha256.len() != 64
        || !input
            .reference_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err((StatusCode::BAD_REQUEST, "citation review decision, rationale, DOI, artifact identity, and reference index must be valid and bounded".into()));
    }
    let selected_cited_doi = input.selected_cited_doi.as_deref();
    if (input.decision == "match") != selected_cited_doi.is_some()
        || selected_cited_doi.is_some_and(|doi| !valid_citation_doi(doi))
    {
        return Err((StatusCode::BAD_REQUEST, "a match decision requires exactly one valid selected_cited_doi; other decisions must omit it".into()));
    }
    let store = s.store.lock().map_err(internal)?;
    let tenant = store.session_tenant(&id).map_err(internal)?;
    if tenant.as_deref() != Some(principal.tenant_id.as_str()) && principal.role != "platform_admin"
    {
        return Err((StatusCode::NOT_FOUND, "research session not found".into()));
    }
    let candidate = store
        .artifact(&id, &input.candidate_artifact_id)
        .map_err(internal)?
        .ok_or((
            StatusCode::BAD_REQUEST,
            "candidate_artifact_id was not found in this session".into(),
        ))?;
    if candidate.kind != "crossref_reference_resolution_candidates"
        || candidate.source != "Crossref REST API"
        || candidate.sha256 != input.candidate_artifact_sha256
        || format!("{:x}", Sha256::digest(candidate.payload.as_bytes())) != candidate.sha256
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "candidate artifact identity or digest does not match this review".into(),
        ));
    }
    let candidate_data: serde_json::Value =
        serde_json::from_str(&candidate.payload).map_err(internal)?;
    let candidate_reference_sha = candidate_data
        .get("reference_sha256")
        .and_then(serde_json::Value::as_str);
    let source_artifact_id = candidate_data
        .get("source_artifact_id")
        .and_then(serde_json::Value::as_str)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "candidate artifact is missing its source Crossref artifact".into(),
        ))?;
    let source_artifact_sha = candidate_data
        .get("source_artifact_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "candidate artifact is missing its source artifact digest".into(),
        ))?;
    if candidate_data
        .get("status")
        .and_then(serde_json::Value::as_str)
        != Some("unverified_candidates")
        || candidate_data
            .get("citing_doi")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|doi| !doi.eq_ignore_ascii_case(&input.citing_doi))
        || candidate_data
            .get("reference_index")
            .and_then(serde_json::Value::as_i64)
            != Some(input.reference_index)
        || candidate_reference_sha != Some(input.reference_sha256.as_str())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "candidate artifact is not bound to the exact cited reference".into(),
        ));
    }
    let source = store
        .artifact(&id, source_artifact_id)
        .map_err(internal)?
        .ok_or((
            StatusCode::BAD_REQUEST,
            "source Crossref artifact was not found in this session".into(),
        ))?;
    if source.kind != "crossref_citation_metadata"
        || source.source != "Crossref REST API"
        || source.sha256 != source_artifact_sha
        || format!("{:x}", Sha256::digest(source.payload.as_bytes())) != source.sha256
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "source Crossref artifact identity or digest failed verification".into(),
        ));
    }
    let source_data: serde_json::Value = serde_json::from_str(&source.payload).map_err(internal)?;
    let records = source_data
        .get("records")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "source Crossref artifact has no record array".into(),
        ))?;
    if format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(records).map_err(internal)?)
    ) != source_data
        .get("retained_records_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "source Crossref record digest failed verification".into(),
        ));
    }
    let reference = records
        .iter()
        .find(|record| {
            record
                .get("doi")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|doi| doi.eq_ignore_ascii_case(&input.citing_doi))
        })
        .and_then(|record| {
            record
                .get("references")
                .and_then(serde_json::Value::as_array)
        })
        .and_then(|references| {
            references.iter().find(|reference| {
                reference
                    .get("reference_index")
                    .and_then(serde_json::Value::as_i64)
                    == Some(input.reference_index)
            })
        })
        .ok_or((
            StatusCode::BAD_REQUEST,
            "selected reference index is not present in the exact Crossref source record".into(),
        ))?;
    if reference
        .get("doi_validated_by_pattern")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
        || format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(reference).map_err(internal)?)
        ) != input.reference_sha256
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "selected reference has a source DOI or its digest does not match the candidate".into(),
        ));
    }
    let candidate_rows = candidate_data
        .get("candidate_records")
        .and_then(serde_json::Value::as_array)
        .ok_or((
            StatusCode::BAD_REQUEST,
            "candidate artifact has no candidate record array".into(),
        ))?;
    if format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(candidate_rows).map_err(internal)?)
    ) != candidate_data
        .get("candidate_records_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "candidate record digest failed verification".into(),
        ));
    }
    if let Some(doi) = selected_cited_doi
        && !candidate_rows.iter().any(|row| {
            row.get("doi")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|candidate_doi| candidate_doi.eq_ignore_ascii_case(doi))
        })
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "selected DOI is not present among retained Crossref candidates".into(),
        ));
    }
    let payload = serde_json::json!({
        "schema_version": 1,
        "review_type": "human_crossref_reference_resolution_adjudication",
        "reviewer": {"subject": principal.subject, "role": principal.role},
        "candidate_artifact_id": candidate.id,
        "candidate_artifact_sha256": candidate.sha256,
        "candidate_records_sha256": candidate_data.get("candidate_records_sha256"),
        "source_artifact_id": source.id,
        "source_artifact_sha256": source.sha256,
        "reference_sha256": input.reference_sha256,
        "citing_doi": input.citing_doi.to_ascii_lowercase(),
        "reference_index": input.reference_index,
        "decision": input.decision,
        "selected_cited_doi": selected_cited_doi.map(str::to_ascii_lowercase),
        "rationale": rationale,
        "citation_edge_created": input.decision == "match",
        "interpretation_limit": "Authenticated researcher adjudication of one exact DOI-less Crossref reference and a retained search candidate. This does not establish scientific support, contradiction, article quality, or evidence strength."
    });
    let serialized = serde_json::to_string(&payload).map_err(internal)?;
    let review = store
        .add_human_review_artifact(
            &id,
            &candidate.task_id,
            "crossref_reference_resolution_adjudication",
            "authenticated researcher citation resolution review",
            &candidate.id,
            &serialized,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let citation_link_created = if input.decision == "match" {
        store
            .index_reviewed_crossref_reference(&id, &review.id)
            .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?
    } else {
        false
    };
    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::json!({"review_artifact_id":review.id,"review_artifact_sha256":review.sha256,"decision":input.decision,"citation_link_created":citation_link_created,"citation_link_source_artifact_id":if citation_link_created { Some(review.id) } else { None }}),
        ),
    ))
}

async fn add_evidence(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewEvidence>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if principal.scientific_agent_id.is_some() {
        let task_id = input.task_id.as_deref().ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "specialist evidence writes require an assigned task ID".into(),
            )
        })?;
        authorize_specialist_task(&principal, task_id)?;
    }
    let record = Evidence {
        id: format!("EVD-{}", uuid::Uuid::new_v4().simple()),
        session_id: id,
        claim: input.claim,
        source: input.source,
        source_id: input.source_id,
        polarity: input.polarity,
        quality: input.quality,
        provenance: input.provenance,
    };
    s.store
        .lock()
        .map_err(internal)?
        .add_evidence(&record)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({"id":record.id})),
    ))
}
async fn graph(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(filter): Query<GraphFilter>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let limit = filter.limit.unwrap_or(200);
    if !(1..=500).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "graph page limit must be 1..500".into(),
        ));
    }
    if filter.after_node_label.is_some() != filter.after_node_id.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "node cursor requires both after_node_label and after_node_id".into(),
        ));
    }
    let edge_cursor_parts = [
        filter.after_edge_source.is_some(),
        filter.after_edge_relation.is_some(),
        filter.after_edge_target.is_some(),
    ];
    if edge_cursor_parts.iter().any(|value| *value) && !edge_cursor_parts.iter().all(|value| *value)
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "edge cursor requires after_edge_source, after_edge_relation, and after_edge_target"
                .into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let (nodes, edges, has_more_nodes, has_more_edges) = store
        .graph_page(
            &id,
            filter
                .after_node_label
                .as_deref()
                .zip(filter.after_node_id.as_deref()),
            filter
                .after_edge_source
                .as_deref()
                .zip(filter.after_edge_relation.as_deref())
                .zip(filter.after_edge_target.as_deref())
                .map(|((source, relation), target)| (source, relation, target)),
            filter.include_nodes.unwrap_or(true),
            filter.include_edges.unwrap_or(true),
            limit,
        )
        .map_err(internal)?;
    let next_node_cursor = has_more_nodes
        .then(|| {
            nodes
                .last()
                .map(|node| serde_json::json!({"label":node.label,"id":node.id}))
        })
        .flatten();
    let next_edge_cursor = has_more_edges.then(|| edges.last().map(|edge| serde_json::json!({"source":edge.source,"relation":edge.relation,"target":edge.target}))).flatten();
    Ok(Json(
        serde_json::json!({"session_id":id,"nodes":nodes,"edges":edges,"next_node_cursor":next_node_cursor,"next_edge_cursor":next_edge_cursor,"page_size":limit}),
    ))
}
async fn graph_search(
    State(s): State<App>,
    Path(id): Path<String>,
    Query(filter): Query<GraphSearchFilter>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let limit = filter.limit.unwrap_or(50);
    let (nodes, edges, has_more_nodes, has_more_edges) = s
        .store
        .lock()
        .map_err(internal)?
        .graph_search(&id, &filter.q, limit)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(serde_json::json!({
        "session_id": id,
        "nodes": nodes,
        "edges": edges,
        "truncated": {"nodes": has_more_nodes, "edges": has_more_edges},
        "limits": {"nodes": limit, "edges": (limit * 5).min(500)},
        "matching": "SQLite case-insensitive substring search over ID, label, and kind",
    })))
}
async fn add_node(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<GraphNodeWrite>,
) -> Result<StatusCode, (StatusCode, String)> {
    if principal.scientific_agent_id.is_some() {
        let task_id = input.task_id.as_deref().ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "specialist graph writes require an assigned task ID".into(),
            )
        })?;
        authorize_specialist_task(&principal, task_id)?;
    }
    let n = input.node;
    if n.session_id != id {
        return Err((
            StatusCode::BAD_REQUEST,
            "graph session_id must match URL".into(),
        ));
    }
    s.store
        .lock()
        .map_err(internal)?
        .put_node(&n)
        .map_err(internal)?;
    Ok(StatusCode::CREATED)
}
async fn add_edge(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<GraphEdgeWrite>,
) -> Result<StatusCode, (StatusCode, String)> {
    if principal.scientific_agent_id.is_some() {
        let task_id = input.task_id.as_deref().ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "specialist graph writes require an assigned task ID".into(),
            )
        })?;
        authorize_specialist_task(&principal, task_id)?;
    }
    let e = input.edge;
    if e.session_id != id {
        return Err((
            StatusCode::BAD_REQUEST,
            "graph session_id must match URL".into(),
        ));
    }
    if !(0.0..=1.0).contains(&e.confidence) {
        return Err((
            StatusCode::BAD_REQUEST,
            "confidence must be between 0 and 1".into(),
        ));
    }
    s.store
        .lock()
        .map_err(internal)?
        .put_edge(&e)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(StatusCode::CREATED)
}
async fn capabilities(
    State(s): State<App>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let capabilities: Vec<Capability> = s
        .store
        .lock()
        .map_err(internal)?
        .capabilities()
        .map_err(internal)?;
    let items = capabilities
        .into_iter()
        .map(|capability| {
            let manifest_sha256 = format!("{:x}", Sha256::digest(capability.manifest.as_bytes()));
            serde_json::json!({
                "id": capability.id,
                "version": capability.version,
                "kind": capability.kind,
                "status": capability.status,
                "description": capability.description,
                "manifest": capability.manifest,
                "manifest_sha256": manifest_sha256,
            })
        })
        .collect::<Vec<_>>();
    Ok(Json(serde_json::json!({"items":items})))
}
#[derive(Deserialize)]
struct RegisterCapabilityProposal {
    id: String,
    version: String,
    kind: String,
    description: String,
    package: serde_json::Value,
}
async fn register_capability_proposal(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<RegisterCapabilityProposal>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "capability registration requires platform_admin role".into(),
        ));
    }
    if input.description.trim().is_empty() || input.description.len() > 4000 {
        return Err((
            StatusCode::BAD_REQUEST,
            "capability description must contain 1..4000 characters".into(),
        ));
    }
    if input.id.is_empty()
        || input.id.len() > 128
        || !input
            .id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        || input.version.is_empty()
        || input.version.len() > 64
        || !input
            .version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._+-".contains(&byte))
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "capability ID or version contains invalid characters".into(),
        ));
    }
    let package_bytes = serde_json::to_vec(&input.package).map_err(internal)?;
    if package_bytes.len() > 1_000_000 {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "AMCP package exceeds 1 MiB".into(),
        ));
    }
    let declared = input.package.get("capability");
    if declared
        .and_then(|value| value.get("id"))
        .and_then(serde_json::Value::as_str)
        != Some(input.id.as_str())
        || declared
            .and_then(|value| value.get("version"))
            .and_then(serde_json::Value::as_str)
            != Some(input.version.as_str())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            "registration ID/version must exactly match the AMCP package".into(),
        ));
    }
    let package_sha256 = format!("{:x}", Sha256::digest(&package_bytes));
    let manifest = serde_json::json!({
        "source": "agentmesh_amcp",
        "package": input.package,
        "package_sha256": package_sha256,
        "execution": {"enabled": false, "network": false}
    });
    let proposal = Capability {
        id: input.id.clone(),
        version: input.version.clone(),
        kind: input.kind,
        status: "pending_review".into(),
        description: input.description.trim().into(),
        manifest: serde_json::to_string(&manifest).map_err(internal)?,
    };
    s.store
        .lock()
        .map_err(internal)?
        .propose_capability_as(
            &proposal,
            &principal.subject,
            "AMCP package registered for lifecycle review",
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::json!({"id": input.id, "version": input.version, "status": "pending_review", "package_sha256": package_sha256}),
        ),
    ))
}
#[derive(Deserialize)]
struct CapabilityLifecycleTransition {
    version: String,
    next: String,
    reason: String,
    validation: Option<String>,
}
async fn transition_capability(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<CapabilityLifecycleTransition>,
) -> Result<StatusCode, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "capability lifecycle approval requires platform_admin role".into(),
        ));
    }
    s.store
        .lock()
        .map_err(internal)?
        .transition_capability(
            &id,
            &input.version,
            &input.next,
            &principal.subject,
            &input.reason,
            input.validation.as_deref(),
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
#[derive(Deserialize)]
struct CapabilityRollback {
    target_version: String,
    reason: String,
    validation: String,
}
async fn rollback_capability(
    State(s): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<CapabilityRollback>,
) -> Result<StatusCode, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "capability rollback approval requires platform_admin role".into(),
        ));
    }
    s.store
        .lock()
        .map_err(internal)?
        .rollback_capability(
            &id,
            &input.target_version,
            &principal.subject,
            &input.reason,
            &input.validation,
        )
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}
async fn capability_lifecycle_events(
    State(s): State<App>,
    Path((id, version)): Path<(String, String)>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "capability lifecycle audit requires platform_admin role".into(),
        ));
    }
    let items = s
        .store
        .lock()
        .map_err(internal)?
        .capability_events(&id, &version)
        .map_err(internal)?;
    Ok(Json(
        serde_json::json!({"capability_id": id, "version": version, "items": items}),
    ))
}

#[derive(Deserialize)]
struct AgentMeshRouteApprovalInput {
    plan: serde_json::Value,
}
fn validated_agentmesh_plan_fields(
    plan: &serde_json::Value,
) -> Result<(String, String, String, String), String> {
    let object = plan.as_object().ok_or("route plan must be a JSON object")?;
    let tenant = plan
        .pointer("/control_plane_scope/tenant")
        .and_then(serde_json::Value::as_str)
        .ok_or("route plan tenant scope is missing")?;
    let namespace = plan
        .pointer("/control_plane_scope/namespace")
        .and_then(serde_json::Value::as_str)
        .ok_or("route plan namespace scope is missing")?;
    let plan_digest = plan
        .get("plan_review_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or("route plan review digest is missing")?;
    let desired_digest = plan
        .get("desired_sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or("route plan desired digest is missing")?;
    let resources = plan
        .get("resources")
        .ok_or("route plan resources are missing")?;
    let calculated_desired = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(resources).map_err(|error| error.to_string())?)
    );
    if calculated_desired != desired_digest {
        return Err("route plan desired digest does not match resources".into());
    }
    let mut unsigned = object.clone();
    unsigned.remove("plan_review_sha256");
    let calculated_plan = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&unsigned).map_err(|error| error.to_string())?)
    );
    if calculated_plan != plan_digest {
        return Err("route plan review digest is invalid".into());
    }
    Ok((
        tenant.to_owned(),
        namespace.to_owned(),
        plan_digest.to_owned(),
        desired_digest.to_owned(),
    ))
}
async fn create_agentmesh_route_approval(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<AgentMeshRouteApprovalInput>,
) -> Result<(StatusCode, Json<AgentMeshRouteApproval>), (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "AgentMesh route approval requires platform_admin role".into(),
        ));
    }
    let (tenant, namespace, plan_digest, desired_digest) =
        validated_agentmesh_plan_fields(&input.plan)
            .map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let approval = s
        .store
        .lock()
        .map_err(internal)?
        .create_agentmesh_route_approval(
            &principal.subject,
            &principal.tenant_id,
            &tenant,
            &namespace,
            &plan_digest,
            &desired_digest,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok((StatusCode::CREATED, Json(approval)))
}
async fn validate_agentmesh_route_approval(
    State(s): State<App>,
    Path(receipt_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<AgentMeshRouteApprovalInput>,
) -> Result<Json<AgentMeshRouteApproval>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "AgentMesh route approval validation requires platform_admin role".into(),
        ));
    }
    let (tenant, namespace, plan_digest, desired_digest) =
        validated_agentmesh_plan_fields(&input.plan)
            .map_err(|error| (StatusCode::BAD_REQUEST, error))?;
    let approval = s
        .store
        .lock()
        .map_err(internal)?
        .validate_agentmesh_route_approval(&receipt_id, &tenant, &namespace, &plan_digest, &desired_digest)
        .map_err(internal)?
        .ok_or((StatusCode::NOT_FOUND, "approval receipt is missing, expired, or does not match this exact route plan and scope".into()))?;
    Ok(Json(approval))
}
async fn get_agentmesh_route_approval(
    State(s): State<App>,
    Path(receipt_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<AgentMeshRouteApproval>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "AgentMesh route approval lookup requires platform_admin role".into(),
        ));
    }
    let approval = s
        .store
        .lock()
        .map_err(internal)?
        .agentmesh_route_approval(&receipt_id)
        .map_err(internal)?
        .ok_or((
            StatusCode::NOT_FOUND,
            "AgentMesh approval receipt was not found".into(),
        ))?;
    Ok(Json(approval))
}
async fn summary(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    Ok(Json(
        s.store
            .lock()
            .map_err(internal)?
            .summary_for_tenant(&principal.tenant_id)
            .map_err(internal)?,
    ))
}
async fn audit_events(
    State(s): State<App>,
    Query(filter): Query<AuditFilter>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !(matches!(principal.role.as_str(), "tenant_admin" | "platform_admin")
        || is_general_service(&principal))
    {
        return Err((
            StatusCode::FORBIDDEN,
            "audit administrator role required".into(),
        ));
    }
    let limit = filter.limit.unwrap_or(200);
    if !(1..=500).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "audit page limit must be 1..500".into(),
        ));
    }
    let after_sequence = filter.after_sequence.unwrap_or(0);
    if after_sequence < 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "audit cursor cannot be negative".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let mut items = if principal.role == "tenant_admin" {
        store.api_audit_events_for_tenant(&principal.tenant_id, after_sequence, limit + 1)
    } else {
        store.api_audit_events(after_sequence, limit + 1)
    }
    .map_err(internal)?;
    let has_more = items.len() > limit;
    items.truncate(limit);
    let next_cursor = if has_more {
        items.last().map(|event| event.sequence)
    } else {
        None
    };
    Ok(Json(
        serde_json::json!({"items":items,"next_cursor":next_cursor,"page_size":limit,"integrity":"events belong to the platform global chain; platform administrators and service credentials can verify the complete chain"}),
    ))
}
async fn audit_export(
    State(s): State<App>,
    Query(filter): Query<AuditFilter>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Response, (StatusCode, String)> {
    if !(matches!(principal.role.as_str(), "tenant_admin" | "platform_admin")
        || is_general_service(&principal))
    {
        return Err((
            StatusCode::FORBIDDEN,
            "audit administrator role required".into(),
        ));
    }
    let limit = filter.limit.unwrap_or(1000);
    if !(1..=5000).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "audit export page limit must be 1..5000".into(),
        ));
    }
    let after_sequence = filter.after_sequence.unwrap_or(0);
    if after_sequence < 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "audit cursor cannot be negative".into(),
        ));
    }
    let store = s.store.lock().map_err(internal)?;
    let mut events = if principal.role == "tenant_admin" {
        store.api_audit_events_for_tenant(&principal.tenant_id, after_sequence, limit + 1)
    } else {
        store.api_audit_events(after_sequence, limit + 1)
    }
    .map_err(internal)?;
    let has_more = events.len() > limit;
    events.truncate(limit);
    let next_cursor = has_more
        .then(|| events.last().map(|event| event.sequence))
        .flatten();
    let mut output = Vec::with_capacity(events.len() * 512);
    for event in events {
        serde_json::to_writer(&mut output, &event).map_err(internal)?;
        output.push(b'\n');
    }
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
        .header(header::CACHE_CONTROL, "no-store")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=medai-audit-export.ndjson",
        );
    if let Some(cursor) = next_cursor {
        response = response.header("x-medai-next-cursor", cursor.to_string());
    }
    response
        .body(axum::body::Body::from(output))
        .map_err(internal)
}
async fn audit_verify(
    State(s): State<App>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let result = s
        .store
        .lock()
        .map_err(internal)?
        .verify_api_audit_chain()
        .map_err(internal)?;
    Ok(Json(result))
}
async fn audit_archive(
    State(s): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "audit archival requires platform_admin role".into(),
        ));
    }
    let policy = load_audit_archive_policy()
        .map_err(|error| (StatusCode::SERVICE_UNAVAILABLE, error.to_string()))?
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "MEDAI_AUDIT_RETENTION_DAYS must be configured".into(),
        ))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(internal)?
        .as_secs() as i64;
    let cutoff = now.saturating_sub(policy.retention_days.saturating_mul(86_400));
    let receipt = s
        .store
        .lock()
        .map_err(internal)?
        .archive_api_audit_before(cutoff, &policy.archive_dir)
        .map_err(internal)?;
    Ok(Json(serde_json::json!({
        "retention_days": policy.retention_days,
        "cutoff": cutoff,
        "archive": receipt,
        "message": if receipt.is_some() { "expired audit prefix archived and pruned" } else { "no expired contiguous audit prefix" }
    })))
}
#[derive(Deserialize)]
struct TenantQuotaInput {
    max_active_sessions: i64,
    max_session_budget: i64,
    max_allocated_credits: i64,
}
#[derive(Deserialize)]
struct TenantCreateInput {
    tenant_id: String,
}
#[derive(Deserialize)]
struct TenantStatusInput {
    status: String,
    reason: Option<String>,
}
async fn list_tenants(
    State(app): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let store = app.store.lock().map_err(internal)?;
    let items = if principal.role == "platform_admin" {
        store.list_tenant_directory().map_err(internal)?
    } else if principal.role == "tenant_admin" {
        store
            .tenant_directory_entry(&principal.tenant_id)
            .map_err(internal)?
            .into_iter()
            .collect()
    } else {
        return Err((
            StatusCode::FORBIDDEN,
            "tenant administrator role required".into(),
        ));
    };
    Ok(Json(serde_json::json!({"items": items})))
}
async fn create_tenant(
    State(app): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<TenantCreateInput>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "platform administrator role required".into(),
        ));
    }
    if !valid_identity_identifier(&input.tenant_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            "tenant_id must be a safe identifier".into(),
        ));
    }
    let entry = app
        .store
        .lock()
        .map_err(internal)?
        .create_tenant(&input.tenant_id, &principal.subject)
        .map_err(|error| (StatusCode::CONFLICT, error.to_string()))?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::to_value(entry).map_err(internal)?),
    ))
}
async fn get_tenant(
    State(app): State<App>,
    Path(tenant_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "platform_admin"
        && !(principal.role == "tenant_admin" && principal.tenant_id == tenant_id)
    {
        return Err((StatusCode::NOT_FOUND, "tenant not found".into()));
    }
    let entry = app
        .store
        .lock()
        .map_err(internal)?
        .tenant_directory_entry(&tenant_id)
        .map_err(internal)?
        .ok_or((StatusCode::NOT_FOUND, "tenant not found".into()))?;
    Ok(Json(serde_json::to_value(entry).map_err(internal)?))
}
async fn update_tenant_status(
    State(app): State<App>,
    Path(tenant_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<TenantStatusInput>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "platform administrator role required".into(),
        ));
    }
    let entry = app
        .store
        .lock()
        .map_err(internal)?
        .set_tenant_status(
            &tenant_id,
            &input.status,
            input.reason.as_deref(),
            &principal.subject,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    Ok(Json(serde_json::to_value(entry).map_err(internal)?))
}
async fn get_tenant_quota(
    State(app): State<App>,
    Path(tenant_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "tenant_admin" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "tenant administrator role required".into(),
        ));
    }
    if principal.role == "tenant_admin" && principal.tenant_id != tenant_id {
        return Err((StatusCode::NOT_FOUND, "tenant not found".into()));
    }
    let result = app
        .store
        .lock()
        .map_err(internal)?
        .tenant_quota_usage(&tenant_id)
        .map_err(internal)?;
    Ok(Json(result))
}
async fn set_tenant_quota(
    State(app): State<App>,
    Path(tenant_id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<TenantQuotaInput>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if principal.role != "platform_admin" {
        return Err((
            StatusCode::FORBIDDEN,
            "platform administrator role required".into(),
        ));
    }
    let mut store = app.store.lock().map_err(internal)?;
    store
        .set_tenant_quota(
            &tenant_id,
            input.max_active_sessions,
            input.max_session_budget,
            input.max_allocated_credits,
            &principal.subject,
        )
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    let result = store.tenant_quota_usage(&tenant_id).map_err(internal)?;
    Ok(Json(result))
}
#[derive(Deserialize)]
struct NewCredential {
    subject: String,
    tenant_id: Option<String>,
    role: String,
}
async fn list_credentials(
    State(app): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "tenant_admin" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "credential administrator role required".into(),
        ));
    }
    let tenant = (principal.role == "tenant_admin").then_some(principal.tenant_id.as_str());
    let records = app
        .store
        .lock()
        .map_err(internal)?
        .api_credentials_for_tenant(tenant)
        .map_err(internal)?;
    let items: Vec<_> = records
        .into_iter()
        .map(|record| {
            serde_json::json!({
                "id": record.id, "subject": record.subject, "tenant_id": record.tenant_id,
                "role": record.role, "created_at": record.created_at,
                "revoked_at": record.revoked_at, "active": record.revoked_at.is_none()
            })
        })
        .collect();
    Ok(Json(serde_json::json!({"items": items})))
}
async fn create_credential(
    State(app): State<App>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
    Json(input): Json<NewCredential>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "tenant_admin" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "credential administrator role required".into(),
        ));
    }
    let tenant_id = if principal.role == "tenant_admin" {
        if input
            .tenant_id
            .as_deref()
            .is_some_and(|tenant| tenant != principal.tenant_id)
        {
            return Err((
                StatusCode::FORBIDDEN,
                "tenant administrators can only create credentials for their tenant".into(),
            ));
        }
        principal.tenant_id.clone()
    } else {
        input
            .tenant_id
            .unwrap_or_else(|| principal.tenant_id.clone())
    };
    if principal.role == "tenant_admin" && !matches!(input.role.as_str(), "reader" | "researcher") {
        return Err((
            StatusCode::FORBIDDEN,
            "tenant administrators may grant reader or researcher roles".into(),
        ));
    }
    if !matches!(
        input.role.as_str(),
        "reader" | "researcher" | "tenant_admin" | "platform_admin" | "service"
    ) {
        return Err((StatusCode::BAD_REQUEST, "invalid credential role".into()));
    }
    if !valid_identity_identifier(&input.subject) || !valid_identity_identifier(&tenant_id) {
        return Err((
            StatusCode::BAD_REQUEST,
            "subject and tenant_id must be safe identifiers".into(),
        ));
    }
    if !app
        .store
        .lock()
        .map_err(internal)?
        .tenant_is_active(&tenant_id)
        .map_err(internal)?
    {
        return Err((
            StatusCode::CONFLICT,
            "credentials require an existing active tenant".into(),
        ));
    }
    let token = format!(
        "medai_{}_{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    let token_sha256 = format!("{:x}", Sha256::digest(token.as_bytes()));
    let id = uuid::Uuid::new_v4().to_string();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(internal)?
        .as_secs() as i64;
    let record = ApiCredentialRecord {
        id: id.clone(),
        token_sha256: token_sha256.clone(),
        subject: input.subject.clone(),
        tenant_id: tenant_id.clone(),
        role: input.role.clone(),
        created_at,
        revoked_at: None,
    };
    app.store
        .lock()
        .map_err(internal)?
        .register_api_credential(&record)
        .map_err(internal)?;
    app.api_credentials
        .lock()
        .map_err(internal)?
        .push(ApiCredential {
            id: id.clone(),
            token: Arc::from(token_sha256.into_bytes()),
            subject: input.subject,
            tenant_id: tenant_id.clone(),
            role: input.role,
            scientific_agent_id: None,
        });
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": id, "subject": record.subject, "tenant_id": tenant_id,
            "role": record.role, "created_at": created_at, "token": token,
            "token_disclosure": "shown once; store securely"
        })),
    ))
}
async fn revoke_credential(
    State(app): State<App>,
    Path(id): Path<String>,
    axum::Extension(principal): axum::Extension<ApiPrincipal>,
) -> Result<StatusCode, (StatusCode, String)> {
    if !matches!(principal.role.as_str(), "tenant_admin" | "platform_admin") {
        return Err((
            StatusCode::FORBIDDEN,
            "credential administrator role required".into(),
        ));
    }
    let tenant = (principal.role == "tenant_admin").then_some(principal.tenant_id.as_str());
    let revoked = app
        .store
        .lock()
        .map_err(internal)?
        .revoke_api_credential(&id, tenant)
        .map_err(internal)?;
    if !revoked {
        return Err((StatusCode::NOT_FOUND, "active credential not found".into()));
    }
    app.api_credentials
        .lock()
        .map_err(internal)?
        .retain(|credential| credential.id != id);
    Ok(StatusCode::NO_CONTENT)
}
fn internal(e: impl std::fmt::Display) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db = std::env::var("MEDAI_DB").unwrap_or_else(|_| ".medai/medai.sqlite".into());
    let audit_archive_policy = load_audit_archive_policy()?;
    let addr: SocketAddr = std::env::var("MEDAI_BIND")
        .unwrap_or_else(|_| "127.0.0.1:8090".into())
        .parse()?;
    let configured_api_token = match std::env::var("MEDAI_API_BEARER_TOKEN") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let token_registry = match std::env::var("MEDAI_API_TOKENS") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let service_tenant =
        std::env::var("MEDAI_SERVICE_TENANT_ID").unwrap_or_else(|_| "local".into());
    let configured_credentials =
        load_api_credentials(configured_api_token, token_registry, &service_tenant, addr)?;
    let mut store = Store::open(db)?;
    let agent_credit_policies = std::env::var("MEDAI_AGENT_CREDIT_LIMITS").unwrap_or_default();
    let agent_credit_policies: std::collections::HashMap<String, i64> =
        if agent_credit_policies.trim().is_empty() {
            std::collections::HashMap::new()
        } else {
            serde_json::from_str(&agent_credit_policies).context(
            "MEDAI_AGENT_CREDIT_LIMITS must be a JSON object of specialist IDs to credit ceilings",
        )?
        };
    store.configure_specialist_credit_policies(&agent_credit_policies)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;
    for credential in configured_credentials {
        let digest = credential
            .token
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        store.register_api_credential(&ApiCredentialRecord {
            id: credential.id,
            token_sha256: digest,
            subject: credential.subject,
            tenant_id: credential.tenant_id,
            role: credential.role,
            created_at: now,
            revoked_at: None,
        })?;
    }
    let api_credentials = store
        .active_api_credentials()?
        .into_iter()
        .map(|record| ApiCredential {
            id: record.id,
            token: Arc::from(record.token_sha256.into_bytes()),
            subject: record.subject,
            tenant_id: record.tenant_id,
            role: record.role,
            scientific_agent_id: None,
        })
        .collect();
    let state = App {
        store: Arc::new(Mutex::new(store)),
        api_credentials: credential_set(api_credentials),
    };
    spawn_audit_archive_scheduler(Arc::clone(&state.store), audit_archive_policy);
    let origins = std::env::var("MEDAI_CORS_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:3000,http://127.0.0.1:3000".into());
    let origins = origins
        .split(',')
        .map(str::trim)
        .filter(|origin| !origin.is_empty())
        .map(str::parse::<HeaderValue>)
        .collect::<Result<Vec<_>, _>>()?;
    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::PUT,
            Method::DELETE,
        ])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION]);
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/auth/me", get(authenticated_identity))
        .route("/v1/admin/metrics", get(operational_metrics))
        .route("/v1/auth/whoami", get(authenticated_identity))
        .route("/v1/research", get(sessions).post(create))
        .route("/v1/research/{id}/plan", get(research_plan))
        .route("/v1/research/{id}/status", get(research_status))
        .route(
            "/v1/research/{id}/control",
            get(research_control).post(cancel_research),
        )
        .route(
            "/v1/research/{id}/tasks/claim",
            axum::routing::post(claim_tasks),
        )
        .route(
            "/v1/research/{id}/tasks/scope",
            axum::routing::post(apply_research_task_scope),
        )
        .route(
            "/v1/research/{id}/tasks/recover",
            axum::routing::post(recover_tasks),
        )
        .route(
            "/v1/research/{id}/tasks/{task_id}/lease",
            axum::routing::post(renew_task_lease),
        )
        .route(
            "/v1/research/{id}/tasks/{task_id}/checkpoint",
            axum::routing::post(save_task_checkpoint),
        )
        .route("/v1/research/{id}/budget", get(budget_summary))
        .route(
            "/v1/research/{id}/budget/consume",
            axum::routing::post(consume_budget),
        )
        .route(
            "/v1/research/{id}/budget/reservations",
            axum::routing::post(reserve_model_credits),
        )
        .route(
            "/v1/research/{id}/budget/reservations/{reservation_id}/settle",
            axum::routing::post(settle_model_credits),
        )
        .route(
            "/v1/research/{id}/tasks/{task_id}",
            axum::routing::patch(update_task),
        )
        .route(
            "/v1/research/{id}/tasks/{task_id}/review",
            axum::routing::post(review_task),
        )
        .route(
            "/v1/research/{id}/evidence",
            get(evidence).post(add_evidence),
        )
        .route(
            "/v1/research/{id}/artifacts",
            get(artifacts)
                .post(add_artifact)
                .layer(DefaultBodyLimit::max(
                    MAX_ARTIFACT_PAYLOAD_BYTES + 1024 * 1024,
                )),
        )
        .route(
            "/v1/research/{id}/genomic-datasets",
            axum::routing::post(submit_genomic_dataset),
        )
        .route(
            "/v1/research/{id}/drug-interaction-reviews",
            axum::routing::post(review_drug_interaction_study),
        )
        .route(
            "/v1/research/{id}/drug-interaction-claim-reviews",
            axum::routing::post(review_drug_interaction_claim),
        )
        .route(
            "/v1/research/{id}/hypothesis-literature-reviews",
            axum::routing::post(review_hypothesis_literature),
        )
        .route(
            "/v1/research/{id}/hypothesis-critic-reviews",
            axum::routing::post(review_hypothesis_critic),
        )
        .route(
            "/v1/research/{id}/claim-alignment-reviews",
            axum::routing::post(review_semantic_claim_alignment),
        )
        .route(
            "/v1/research/{id}/ontology-cross-mapping-reviews",
            axum::routing::post(review_ontology_cross_mapping),
        )
        .route(
            "/v1/research/{id}/artifacts/{artifact_id}",
            get(artifact_by_id),
        )
        .route(
            "/v1/research/{id}/pubmed-abstract-retention/preview",
            get(preview_pubmed_redaction),
        )
        .route(
            "/v1/research/{id}/pubmed-abstract-retention/receipts",
            get(pubmed_redaction_receipts),
        )
        .route(
            "/v1/research/{id}/pubmed-abstract-retention",
            axum::routing::post(apply_pubmed_redaction),
        )
        .route(
            "/v1/research/{id}/trial-watches",
            get(list_trial_watches).post(register_trial_watch),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/disable",
            axum::routing::post(disable_trial_watch),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/events",
            get(trial_watch_events),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/latest",
            get(latest_trial_watch_snapshot),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/budget/consume",
            axum::routing::post(charge_trial_watch),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/complete",
            axum::routing::post(complete_trial_watch),
        )
        .route(
            "/v1/research/{id}/trial-watches/{nct_id}/fail",
            axum::routing::post(fail_trial_watch),
        )
        .route(
            "/v1/trial-watches/due/claim",
            axum::routing::post(claim_trial_watches),
        )
        .route(
            "/v1/research/{id}/literature-watches",
            get(list_literature_watches).post(register_literature_watch),
        )
        .route(
            "/v1/research/{id}/literature-watches/{watch_id}/disable",
            axum::routing::post(disable_literature_watch),
        )
        .route(
            "/v1/research/{id}/literature-watches/{watch_id}/events",
            get(literature_watch_events),
        )
        .route(
            "/v1/research/{id}/literature-watch-papers",
            get(literature_watch_papers),
        )
        .route(
            "/v1/research/{id}/literature-watches/{watch_id}/budget/consume",
            axum::routing::post(charge_literature_watch),
        )
        .route(
            "/v1/research/{id}/literature-watches/{watch_id}/complete",
            axum::routing::post(complete_literature_watch),
        )
        .route(
            "/v1/research/{id}/literature-watches/{watch_id}/fail",
            axum::routing::post(fail_literature_watch),
        )
        .route(
            "/v1/literature-watches/due/claim",
            axum::routing::post(claim_literature_watches),
        )
        .route(
            "/v1/research/{id}/citation-links",
            get(citation_links).post(index_citations),
        )
        .route(
            "/v1/research/{id}/citation-reference-review",
            axum::routing::post(review_crossref_reference),
        )
        .route("/v1/research/{id}/graph", get(graph).post(add_node))
        .route("/v1/research/{id}/graph/search", get(graph_search))
        .route(
            "/v1/research/{id}/graph/edges",
            axum::routing::post(add_edge),
        )
        .route(
            "/v1/capabilities",
            get(capabilities).post(register_capability_proposal),
        )
        .route(
            "/v1/capabilities/{id}/transition",
            axum::routing::post(transition_capability),
        )
        .route(
            "/v1/capabilities/{id}/rollback",
            axum::routing::post(rollback_capability),
        )
        .route(
            "/v1/capabilities/{id}/{version}/events",
            get(capability_lifecycle_events),
        )
        .route(
            "/v1/admin/agentmesh/route-approvals",
            axum::routing::post(create_agentmesh_route_approval),
        )
        .route(
            "/v1/admin/agentmesh/route-approvals/{receipt_id}/validate",
            axum::routing::post(validate_agentmesh_route_approval),
        )
        .route(
            "/v1/admin/agentmesh/route-approvals/{receipt_id}",
            get(get_agentmesh_route_approval),
        )
        .route("/v1/admin/audit", get(audit_events))
        .route("/v1/admin/audit/export", get(audit_export))
        .route("/v1/admin/audit/verify", get(audit_verify))
        .route(
            "/v1/admin/audit/archive",
            axum::routing::post(audit_archive),
        )
        .route(
            "/v1/admin/credentials",
            get(list_credentials).post(create_credential),
        )
        .route(
            "/v1/admin/credentials/{id}",
            axum::routing::delete(revoke_credential),
        )
        .route("/v1/admin/tenants", get(list_tenants).post(create_tenant))
        .route(
            "/v1/admin/tenants/{tenant_id}",
            get(get_tenant).patch(update_tenant_status),
        )
        .route(
            "/v1/admin/tenants/{tenant_id}/quota",
            get(get_tenant_quota).put(set_tenant_quota),
        )
        .route("/v1/summary", get(summary))
        .layer(from_fn_with_state(state.clone(), authenticate_request))
        .layer(cors)
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod authentication_tests {
    use super::*;
    use tower::ServiceExt;

    #[tokio::test]
    async fn artifact_endpoint_accepts_payloads_above_the_default_json_limit() {
        let path = std::env::temp_dir().join(format!(
            "medai-large-artifact-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("large artifact API", "standard", 20)
            .unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "api-test", 1, 60)
            .unwrap()
            .remove(0);
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"large-artifact-api-token-123456789",
                "api-test",
                "local",
                "service",
            )]),
        };
        let router = Router::new()
            .route(
                "/v1/research/{id}/artifacts",
                axum::routing::post(add_artifact).layer(DefaultBodyLimit::max(
                    MAX_ARTIFACT_PAYLOAD_BYTES + 1024 * 1024,
                )),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let payload = serde_json::json!({"records": "x".repeat(2_250_000)});
        let body = serde_json::json!({
            "task_id": task.id,
            "kind": "biomedical_domain_benchmark_input",
            "source": "researcher-supplied adjudicated biomedical benchmark",
            "source_id": "benchmark-large",
            "payload": payload,
        })
        .to_string();
        let response = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{}/artifacts", session.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer large-artifact-api-token-123456789",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let stored = state.store.lock().unwrap().artifacts(&session.id).unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].payload.len(), 2_250_014);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn exhausted_session_can_only_use_unmetered_task_transition_to_review() {
        let path = std::env::temp_dir().join(format!(
            "medai-budget-task-review-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("budget exhaustion recovery", "standard", 2)
            .unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "budget-worker", 1, 60)
            .unwrap()
            .remove(0);
        store
            .consume_credits(&session.id, &task.id, 1, "exhaust test budget")
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"budget-task-review-token-123456789",
                "budget-worker",
                "local",
                "service",
            )]),
        };
        let router = Router::new()
            .route(
                "/v1/research/{id}/tasks/{task_id}",
                axum::routing::patch(update_task),
            )
            .route(
                "/v1/research/{id}/control",
                axum::routing::post(cancel_research),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let transition = |status: &str| {
            Request::builder()
                .method(Method::PATCH)
                .uri(format!("/v1/research/{}/tasks/{}", session.id, task.id))
                .header(
                    header::AUTHORIZATION,
                    "Bearer budget-task-review-token-123456789",
                )
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({
                        "status": status,
                        "output": "budget exhausted; awaiting researcher review",
                        "worker_id": "budget-worker"
                    })
                    .to_string(),
                ))
                .unwrap()
        };
        let failed = router.clone().oneshot(transition("failed")).await.unwrap();
        assert_eq!(failed.status(), StatusCode::PAYMENT_REQUIRED);
        let review = router
            .clone()
            .oneshot(transition("needs_review"))
            .await
            .unwrap();
        assert_eq!(review.status(), StatusCode::NO_CONTENT);
        let tasks = state
            .store
            .lock()
            .unwrap()
            .research_tasks(&session.id)
            .unwrap();
        assert_eq!(
            tasks.iter().find(|item| item.id == task.id).unwrap().status,
            "needs_review"
        );
        let cancellable = state
            .store
            .lock()
            .unwrap()
            .create_session_with_options("cancel exhausted session", "standard", 2)
            .unwrap();
        let cancel_task = state
            .store
            .lock()
            .unwrap()
            .claim_ready_tasks(&cancellable.id, "budget-worker", 1, 60)
            .unwrap()
            .remove(0);
        state
            .store
            .lock()
            .unwrap()
            .consume_credits(
                &cancellable.id,
                &cancel_task.id,
                1,
                "exhaust cancellation test",
            )
            .unwrap();
        let cancelled = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{}/control", cancellable.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer budget-task-review-token-123456789",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::OK);
        let cancelled_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(cancelled.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(cancelled_json["cancellation_requested"], true);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn specialist_cannot_cancel_session_but_researcher_can() {
        let path = std::env::temp_dir().join(format!(
            "medai-specialist-cancel-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_for_tenant("cancel authorization", "standard", 5, "tenant-a")
            .unwrap();
        let mut specialist = test_credential(
            b"specialist-cancel-token-1234567890",
            "agent-literature",
            "tenant-a",
            "service",
        );
        specialist.scientific_agent_id = Some("literature_librarian".into());
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                specialist,
                test_credential(
                    b"researcher-cancel-token-1234567890",
                    "researcher",
                    "tenant-a",
                    "researcher",
                ),
            ]),
        };
        let router = Router::new()
            .route(
                "/v1/research/{id}/control",
                axum::routing::post(cancel_research),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state);
        let specialist_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{}/control", session.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer specialist-cancel-token-1234567890",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(specialist_response.status(), StatusCode::FORBIDDEN);
        let researcher_response = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{}/control", session.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-cancel-token-1234567890",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(researcher_response.status(), StatusCode::OK);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn agentmesh_route_approval_is_authenticated_scoped_and_bound_to_plan_hashes() {
        let path = std::env::temp_dir().join(format!(
            "medai-agentmesh-approval-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let mut store = Store::open(&path).unwrap();
        store.ensure_tenant("lab-a", "test-bootstrap").unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                test_credential(
                    b"route-researcher-token-123456",
                    "researcher",
                    "lab-a",
                    "researcher",
                ),
                test_credential(
                    b"route-admin-token-123456789",
                    "release-operator",
                    "platform",
                    "platform_admin",
                ),
            ]),
        };
        let router = Router::new()
            .route(
                "/v1/admin/agentmesh/route-approvals",
                axum::routing::post(create_agentmesh_route_approval),
            )
            .route(
                "/v1/admin/agentmesh/route-approvals/{receipt_id}/validate",
                axum::routing::post(validate_agentmesh_route_approval),
            )
            .route(
                "/v1/admin/agentmesh/route-approvals/{receipt_id}",
                get(get_agentmesh_route_approval),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let resources = serde_json::json!([{"kind":"route","name":"medai-gene-1","state":"active","spec":{"service":"medai","capability":"gene","strategy":"least_active"}}]);
        let desired_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&resources).unwrap())
        );
        let mut plan = serde_json::json!({"control_plane_scope":{"tenant":"tenant-a","namespace":"biomedical"},"resources":resources,"desired_sha256":desired_sha256});
        let plan_review_sha256 =
            format!("{:x}", Sha256::digest(serde_json::to_vec(&plan).unwrap()));
        plan["plan_review_sha256"] = serde_json::Value::String(plan_review_sha256);
        let body = serde_json::json!({"plan":plan.clone()}).to_string();
        let denied = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/admin/agentmesh/route-approvals")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer route-researcher-token-123456",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/admin/agentmesh/route-approvals")
                    .header(header::AUTHORIZATION, "Bearer route-admin-token-123456789")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let receipt: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(created.into_body(), 20_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["actor"], "release-operator");
        assert_eq!(receipt["agentmesh_tenant"], "tenant-a");
        assert_eq!(
            receipt["expires_at"].as_i64().unwrap() - receipt["issued_at"].as_i64().unwrap(),
            600
        );
        let validate_uri = format!(
            "/v1/admin/agentmesh/route-approvals/{}/validate",
            receipt["receipt_id"].as_str().unwrap()
        );
        let valid = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&validate_uri)
                    .header(header::AUTHORIZATION, "Bearer route-admin-token-123456789")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(valid.status(), StatusCode::OK);
        let retrieved = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/v1/admin/agentmesh/route-approvals/{}",
                        receipt["receipt_id"].as_str().unwrap()
                    ))
                    .header(header::AUTHORIZATION, "Bearer route-admin-token-123456789")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retrieved.status(), StatusCode::OK);
        let retrieved: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(retrieved.into_body(), 20_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(retrieved, receipt);
        let audit = state.store.lock().unwrap().api_audit_events(0, 10).unwrap();
        assert!(audit.iter().any(|event| event.actor
            == "platform:platform_admin:release-operator"
            && event.path == "/v1/admin/agentmesh/route-approvals"));
        assert!(audit.iter().any(|event| event.actor
            == "platform:platform_admin:release-operator"
            && event.path.contains("/validate")));
        assert_eq!(
            state
                .store
                .lock()
                .unwrap()
                .verify_api_audit_chain()
                .unwrap()["valid"],
            true
        );
        let mut changed_scope = plan.clone();
        changed_scope["control_plane_scope"]["tenant"] =
            serde_json::Value::String("tenant-b".into());
        changed_scope
            .as_object_mut()
            .unwrap()
            .remove("plan_review_sha256");
        let changed_hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&changed_scope).unwrap())
        );
        changed_scope["plan_review_sha256"] = serde_json::Value::String(changed_hash);
        let mismatch = serde_json::json!({"plan":changed_scope}).to_string();
        let rejected = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&validate_uri)
                    .header(header::AUTHORIZATION, "Bearer route-admin-token-123456789")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(mismatch))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::NOT_FOUND);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn operational_metrics_bound_labels_and_exclude_source_content() {
        let rows = vec![
            ("session".into(), "planned".into(), 2),
            ("session".into(), "objective-with-sensitive-text".into(), 1),
            ("task".into(), "ready".into(), 4),
            ("task".into(), "failed\"} injected=".into(), 3),
            ("task".into(), "running".into(), -9),
        ];
        let rendered = render_operational_metrics(&rows);
        assert!(rendered.contains("medai_research_sessions{status=\"planned\"} 2"));
        assert!(rendered.contains("medai_research_sessions{status=\"other\"} 1"));
        assert!(rendered.contains("medai_research_tasks{status=\"ready\"} 4"));
        assert!(rendered.contains("medai_research_tasks{status=\"other\"} 3"));
        assert!(!rendered.contains("objective-with-sensitive-text"));
        assert!(!rendered.contains("injected"));
        assert!(!rendered.contains("-9"));
    }

    #[tokio::test]
    async fn drug_interaction_review_is_authenticated_and_bound_to_the_screen_location() {
        let path = std::env::temp_dir().join(format!(
            "medai-study-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("interaction study review").unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let _ = claim_tasks(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                subject: "worker".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(ClaimTasks {
                worker_id: "actor.director".into(),
                limit: 1,
                lease_seconds: 120,
                task_ids: Some(vec!["objective.normalize".into()]),
            }),
        )
        .await
        .unwrap();
        let task_id = "objective.normalize";
        let source = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_literature_candidate",
                "PubMed",
                "12345",
                &serde_json::json!({"pmid":"12345","identifiers":{"pmc":"PMC6789"}}).to_string(),
            )
            .unwrap();
        let screen_payload = serde_json::json!({"interaction_verdict":null,"pmid":"12345","pmcid":"PMC6789","candidate_artifact_id":source.id,"candidate_artifact_sha256":source.sha256,"source_sha256":"source-hash","candidates":[{"candidate_id":"lead-1","text_sha256":"passage-hash"}]});
        let screen = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_pmc_fulltext_screen",
                "PMC BioC",
                "12345",
                &screen_payload.to_string(),
            )
            .unwrap();
        let principal = ApiPrincipal {
            subject: "scientist-7".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = DrugInteractionStudyReview {
            screen_artifact_id: screen.id.clone(),
            candidate_id: "lead-1".into(),
            study_design: "cohort".into(),
            exposure_status: "both".into(),
            comparator: "usual care".into(),
            dose_a: "dose A reported".into(),
            dose_b: "dose B reported".into(),
            route_a: "oral".into(),
            route_b: "oral".into(),
            endpoint: "exposure".into(),
            outcome: "as reported".into(),
            limitations: "small cohort".into(),
            label_evidence_overlap: "overlap unclear".into(),
        };
        state
            .store
            .lock()
            .unwrap()
            .transition_task(
                &session.id,
                task_id,
                Some("actor.director"),
                "succeeded",
                None,
                None,
            )
            .unwrap();
        let result = review_drug_interaction_study(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        let saved: serde_json::Value =
            serde_json::from_str(result.1["payload"].as_str().unwrap()).unwrap();
        assert_eq!(saved["reviewer"]["subject"], "scientist-7");
        assert!(saved["research_only"].as_bool().unwrap());
        assert!(saved["clinical_advice"].as_bool() == Some(false));
        assert!(saved["interaction_verdict"].is_null());

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=payload||' ' WHERE id=?1",
            [screen.id.as_str()],
        )
        .unwrap();
        drop(db);
        let corrupted_screen = review_drug_interaction_study(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_screen.0, StatusCode::BAD_REQUEST);
        assert!(corrupted_screen.1.contains("screen digest"));

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=?1 WHERE id=?2",
            rusqlite::params![screen_payload.to_string(), screen.id],
        )
        .unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=payload||' ' WHERE id=?1",
            [source.id.as_str()],
        )
        .unwrap();
        drop(db);
        let corrupted_source = review_drug_interaction_study(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_source.0, StatusCode::BAD_REQUEST);
        assert!(corrupted_source.1.contains("candidate digest"));

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=substr(payload,1,length(payload)-1) WHERE id=?1",
            [source.id.as_str()],
        )
        .unwrap();
        drop(db);

        let mut mismatched = DrugInteractionStudyReview {
            screen_artifact_id: screen.id,
            candidate_id: "foreign-lead".into(),
            study_design: "cohort".into(),
            exposure_status: "both".into(),
            comparator: "control".into(),
            dose_a: "reported".into(),
            dose_b: "reported".into(),
            route_a: "oral".into(),
            route_b: "oral".into(),
            endpoint: "exposure".into(),
            outcome: "reported".into(),
            limitations: "small".into(),
            label_evidence_overlap: "unclear".into(),
        };
        assert_eq!(
            review_drug_interaction_study(
                State(state.clone()),
                Path(session.id.clone()),
                axum::Extension(principal.clone()),
                Json(mismatched.clone())
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::BAD_REQUEST
        );
        mismatched.candidate_id = "lead-1".into();
        let forbidden = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        assert_eq!(
            review_drug_interaction_study(
                State(state),
                Path(session.id.clone()),
                axum::Extension(forbidden),
                Json(mismatched)
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::FORBIDDEN
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn drug_interaction_claim_review_rechecks_versioned_source_spans() {
        let path = std::env::temp_dir().join(format!(
            "medai-interaction-claim-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("interaction claim review").unwrap();
        let task_id = "objective.normalize";
        store
            .claim_ready_tasks(&session.id, "actor.director", 1, 120)
            .unwrap();
        let label_text = "Concurrent use may alter exposure.";
        let paper_text = "A study found no change in exposure.";
        let digest = |value: &str| format!("{:x}", Sha256::digest(value.as_bytes()));
        let label = store
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_label_screen",
                "openFDA label source",
                "pair-label-search",
                &serde_json::json!({"labels":{"Drug A":[{"set_id":"label-set-1","version":"2025-01","drug_interactions":label_text}]}}).to_string(),
            )
            .unwrap();
        let paper = store
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_literature_candidate",
                "PubMed",
                "12345",
                &serde_json::json!({"pmid":"12345","abstract":paper_text}).to_string(),
            )
            .unwrap();
        let label_candidate = "LABEL-claim-1";
        let paper_candidate = "PMID-claim-1";
        let sources = serde_json::json!([
            {"artifact_id":label.id,"artifact_sha256":label.sha256,"source":"openFDA label source","source_id":"pair-label-search","candidate_id":label_candidate,"field":"labels.drug_interactions","set_id":"label-set-1","version":"2025-01","offset":0,"length":label_text.chars().count(),"text_sha256":digest(label_text)},
            {"artifact_id":paper.id,"artifact_sha256":paper.sha256,"source":"PubMed","source_id":"12345","candidate_id":paper_candidate,"field":"abstract","offset":0,"length":paper_text.chars().count(),"text_sha256":digest(paper_text)}
        ]);
        let screening_payload = serde_json::json!({
            "interaction_verdict": null,
            "reviews":[{"candidate_a":label_candidate,"candidate_b":paper_candidate,"relation":"same_question_opposing","sources":sources}]
        });
        let screening = store
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_semantic_review",
                "local model screening",
                "review-pair-1",
                &screening_payload.to_string(),
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let principal = ApiPrincipal {
            subject: "researcher-1".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = DrugInteractionClaimReview {
            screening_artifact_id: screening.id.clone(),
            review_index: 0,
            adjudication: "same_question_opposing".into(),
            claim_domain: "pharmacokinetic".into(),
            exposure_comparability: "comparable".into(),
            outcome_comparability: "comparable".into(),
            rationale: "The sources address the same exposure outcome.".into(),
            context_notes:
                "The publication and label should still be assessed for population differences."
                    .into(),
        };
        let (status, Json(saved)) = review_drug_interaction_claim(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        let saved_payload: serde_json::Value =
            serde_json::from_str(saved["payload"].as_str().unwrap()).unwrap();
        assert_eq!(saved_payload["reviewer"]["subject"], "researcher-1");
        assert_eq!(
            saved_payload["interaction_verdict"],
            serde_json::Value::Null
        );
        assert_eq!(saved_payload["confirmed_contradiction"], false);
        assert_eq!(saved_payload["source_claims"][0]["version"], "2025-01");

        let linked_pmc_paper = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_literature_candidate",
                "PubMed",
                "12345",
                &serde_json::json!({"pmid":"12345","identifiers":{"pmc":"PMC123"}}).to_string(),
            )
            .unwrap();
        let pmc_text_hash = "f".repeat(64);
        let pmc_sha = "a".repeat(64);
        let pmc_candidate_id = format!(
            "{:x}",
            Sha256::digest(format!("{pmc_sha}\0{}\0{pmc_text_hash}", 300).as_bytes())
        );
        let pmc = state.store.lock().unwrap().add_artifact(
            &session.id,
            task_id,
            "drug_interaction_pmc_fulltext_screen",
            "PMC BioC hash-and-offset lexical review leads",
            "PMC123",
            &serde_json::json!({
                "source":"PMC BioC","status":"review_leads_available","pmcid":"PMC123","pmid":"12345","source_sha256":pmc_sha,
                "candidate_artifact_id":linked_pmc_paper.id,"candidate_artifact_sha256":linked_pmc_paper.sha256,
                "candidates":[{"candidate_id":pmc_candidate_id,"source_offset":300,"source_length":42,"text_sha256":pmc_text_hash}]
            }).to_string(),
        ).unwrap();
        let pmc_sources = serde_json::json!([
            sources[0].clone(),
            {"artifact_id":pmc.id,"artifact_sha256":pmc.sha256,"source":"PMC BioC hash-and-offset lexical review leads","source_id":"PMC123","candidate_id":pmc_candidate_id,"field":"full_text","offset":300,"length":42,"text_sha256":pmc_text_hash,"source_sha256":pmc_sha}
        ]);
        let pmc_screen = state.store.lock().unwrap().add_artifact(
            &session.id,
            task_id,
            "drug_interaction_semantic_review",
            "local model PMC screening",
            "pmc-review-pair",
            &serde_json::json!({"interaction_verdict":null,"reviews":[{"candidate_a":label_candidate,"candidate_b":pmc_candidate_id,"relation":"uncertain","sources":pmc_sources}]}).to_string(),
        ).unwrap();
        let (_, Json(pmc_saved)) = review_drug_interaction_claim(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(DrugInteractionClaimReview {
                screening_artifact_id: pmc_screen.id,
                ..input.clone()
            }),
        )
        .await
        .unwrap();
        let pmc_saved_payload: serde_json::Value =
            serde_json::from_str(pmc_saved["payload"].as_str().unwrap()).unwrap();
        assert_eq!(
            pmc_saved_payload["source_claims"][1]["source_sha256"],
            pmc_sha
        );
        assert_eq!(
            pmc_saved_payload["interaction_verdict"],
            serde_json::Value::Null
        );

        let mut tampered = screening_payload;
        tampered["reviews"][0]["sources"][0]["text_sha256"] = serde_json::json!("f".repeat(64));
        let tampered_screen = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "drug_interaction_semantic_review",
                "tampered screening fixture",
                "tampered-pair",
                &tampered.to_string(),
            )
            .unwrap();
        let rejected = review_drug_interaction_claim(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(DrugInteractionClaimReview {
                screening_artifact_id: tampered_screen.id,
                ..input.clone()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(rejected.0, StatusCode::BAD_REQUEST);

        let forbidden = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        assert_eq!(
            review_drug_interaction_claim(
                State(state),
                Path(session.id.clone()),
                axum::Extension(forbidden),
                Json(input),
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::FORBIDDEN
        );
        drop(session);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn ontology_mapping_review_binds_exact_candidate_and_requires_researcher() {
        let path = std::path::Path::new("/private/tmp").join(format!(
            "medai-ontology-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("ontology mapping review").unwrap();
        let task_id = "objective.normalize";
        store
            .claim_ready_tasks(&session.id, "actor.director", 1, 120)
            .unwrap();
        let mapping = store.add_artifact(
            &session.id, task_id, "ontology_cross_mapping_review", "EMBL-EBI Ontology Xref Service (OxO)", "EFO_0000249",
            &serde_json::json!({"source":"EMBL-EBI Ontology Xref Service (OxO)","source_identifier":"EFO_0000249","source_response_sha256":"a".repeat(64),"mappings":[{"target_curie":"MONDO:0004979","target_prefix":"MONDO","distance":1,"target_label":"Alzheimer disease"}]}).to_string(),
        ).unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let principal = ApiPrincipal {
            subject: "reviewer-1".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = OntologyCrossMappingReview {
            source_artifact_id: mapping.id.clone(),
            source_identifier: "EFO_0000249".into(),
            target_curie: "MONDO:0004979".into(),
            decision: "related_but_distinct".into(),
            rationale:
                "The source records are related, but their scope is not demonstrably identical."
                    .into(),
        };
        let saved = review_ontology_cross_mapping(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(saved.0, StatusCode::CREATED);
        let review: serde_json::Value =
            serde_json::from_str(saved.1["payload"].as_str().unwrap()).unwrap();
        assert_eq!(review["reviewer"]["subject"], "reviewer-1");
        assert_eq!(review["source_artifact_sha256"], mapping.sha256);
        assert_eq!(review["ontology_mapping_applied"], false);
        let mut absent = input.clone();
        absent.target_curie = "MONDO:0000001".into();
        let rejected = review_ontology_cross_mapping(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(absent),
        )
        .await
        .unwrap_err();
        assert_eq!(rejected.0, StatusCode::BAD_REQUEST);
        let denied = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        let forbidden = review_ontology_cross_mapping(
            State(state),
            Path(session.id),
            axum::Extension(denied),
            Json(input),
        )
        .await
        .unwrap_err();
        assert_eq!(forbidden.0, StatusCode::FORBIDDEN);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn semantic_claim_review_rechecks_candidate_hashes_and_retains_human_adjudication() {
        let path = std::env::temp_dir().join(format!(
            "medai-claim-alignment-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("claim alignment review").unwrap();
        let task_id = "objective.normalize";
        store
            .claim_ready_tasks(&session.id, "actor.director", 1, 120)
            .unwrap();
        let mut source_refs = Vec::new();
        for (index, candidate_id) in ["cand-a", "cand-b"].iter().enumerate() {
            let candidate_id = *candidate_id;
            let doc_id = format!("doc-{index}");
            let source_hash = (if index == 0 { "a" } else { "b" }).repeat(64);
            let text_hash = (if index == 0 { "c" } else { "d" }).repeat(64);
            let annotation = store.add_artifact(
                &session.id,
                task_id,
                "paper_result_annotations",
                "PMC BioC local-model result annotation",
                &format!("PMC{}", index + 1),
                &serde_json::json!({"document_id":doc_id,"source_sha256":source_hash,"candidate_text_hashes":{candidate_id:text_hash}}).to_string(),
            ).unwrap();
            source_refs.push(serde_json::json!({
                "annotation_artifact_id":annotation.id,"pmc_id":format!("PMC{}", index+1),
                "document_id":doc_id,"source_sha256":source_hash,"candidate_id":candidate_id,
                "candidate_text_sha256":text_hash,"passage_index":0,"sentence_index":0,
                "source_offset":10,"source_local_offset":10,"source_length":40,"license":"CC BY"
            }));
        }
        let screen = store.add_artifact(
            &session.id,
            task_id,
            "paper_claim_semantic_alignment",
            "local model screening",
            "pair-1",
            &serde_json::json!({"assessment":{"candidate_a":"cand-a","candidate_b":"cand-b","relation":"same_question_opposing"},"sources":source_refs.clone()}).to_string(),
        ).unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let principal = ApiPrincipal {
            subject: "reviewer-1".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let result = review_semantic_claim_alignment(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(SemanticClaimReview {
                screening_artifact_id: screen.id.clone(),
                adjudication: "same_question_opposing".into(),
                population_comparability: "comparable".into(),
                method_comparability: "different".into(),
                rationale: "Outcomes point in opposite directions.".into(),
                methods_population_notes: "One study used a cell model; the other a cohort.".into(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        let review: serde_json::Value =
            serde_json::from_str(result.1["payload"].as_str().unwrap()).unwrap();
        assert_eq!(review["reviewer"]["subject"], "reviewer-1");
        assert_eq!(review["confirmed_contradiction"], false);
        assert_eq!(review["source_candidates"].as_array().unwrap().len(), 2);
        let screening_payload: String = state
            .store
            .lock()
            .unwrap()
            .artifacts(&session.id)
            .unwrap()
            .into_iter()
            .find(|artifact| artifact.id == screen.id)
            .unwrap()
            .payload;
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=?1 WHERE id=?2",
            rusqlite::params![format!("{screening_payload} "), screen.id],
        )
        .unwrap();
        drop(db);
        let corrupted_screen = review_semantic_claim_alignment(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(SemanticClaimReview {
                screening_artifact_id: screen.id.clone(),
                adjudication: "uncertain".into(),
                population_comparability: "unclear".into(),
                method_comparability: "unclear".into(),
                rationale: "review".into(),
                methods_population_notes: "context".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_screen.0, StatusCode::BAD_REQUEST);
        assert!(corrupted_screen.1.contains("screening artifact digest"));

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=?1 WHERE id=?2",
            rusqlite::params![screening_payload, screen.id],
        )
        .unwrap();
        let annotation_id = source_refs[0]["annotation_artifact_id"].as_str().unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=payload||' ' WHERE id=?1",
            [annotation_id],
        )
        .unwrap();
        drop(db);
        let corrupted_annotation = review_semantic_claim_alignment(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(SemanticClaimReview {
                screening_artifact_id: screen.id.clone(),
                adjudication: "uncertain".into(),
                population_comparability: "unclear".into(),
                method_comparability: "unclear".into(),
                rationale: "review".into(),
                methods_population_notes: "context".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_annotation.0, StatusCode::BAD_REQUEST);
        assert!(
            corrupted_annotation
                .1
                .contains("annotation artifact digest")
        );

        let mut tampered_sources = source_refs;
        tampered_sources[0]["candidate_text_sha256"] = serde_json::json!("e".repeat(64));
        let tampered = state.store.lock().unwrap().add_artifact(
            &session.id, task_id, "paper_claim_semantic_alignment", "local model screening", "pair-tampered",
            &serde_json::json!({"assessment":{"candidate_a":"cand-a","candidate_b":"cand-b"},"sources":tampered_sources}).to_string(),
        ).unwrap();
        let rejected = review_semantic_claim_alignment(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(SemanticClaimReview {
                screening_artifact_id: tampered.id,
                adjudication: "uncertain".into(),
                population_comparability: "unclear".into(),
                method_comparability: "unclear".into(),
                rationale: "review".into(),
                methods_population_notes: "context".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(rejected.0, StatusCode::BAD_REQUEST);
        let denied = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        let forbidden = review_semantic_claim_alignment(
            State(state),
            Path(session.id),
            axum::Extension(denied),
            Json(SemanticClaimReview {
                screening_artifact_id: screen.id,
                adjudication: "uncertain".into(),
                population_comparability: "unclear".into(),
                method_comparability: "unclear".into(),
                rationale: "review".into(),
                methods_population_notes: "context".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(forbidden.0, StatusCode::FORBIDDEN);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn hypothesis_literature_review_binds_relevance_to_exact_pubmed_and_query_receipts() {
        let path = std::env::temp_dir().join(format!(
            "medai-hypothesis-literature-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session("hypothesis literature review")
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let worker = ApiPrincipal {
            subject: "worker".into(),
            tenant_id: "local".into(),
            role: "service".into(),
            scientific_agent_id: None,
        };
        let _ = claim_tasks(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(worker),
            Json(ClaimTasks {
                worker_id: "actor.director".into(),
                limit: 1,
                lease_seconds: 120,
                task_ids: Some(vec!["objective.normalize".into()]),
            }),
        )
        .await
        .unwrap();
        let task_id = "objective.normalize";
        let paper_payload = serde_json::json!({"hypothesis_id":"HYP-123","query_sha256":"query-digest","record_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pmid":"12345","abstract_retained":false,"title":"A graph-path study"});
        let paper = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "hypothesis_literature_candidate",
                "PubMed exact-entity hypothesis discovery",
                "12345",
                &paper_payload.to_string(),
            )
            .unwrap();
        let comparison_payload = serde_json::json!({"candidate_searches":[{"hypothesis_id":"HYP-123","query_sha256":"query-digest","candidate_leads":[{"pmid":"12345","artifact_id":paper.id,"record_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]}],"novelty_assessment":null});
        let comparison = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "hypothesis_literature_comparison",
                "MedAI bounded PubMed candidate search",
                "proposal-1",
                &comparison_payload.to_string(),
            )
            .unwrap();
        let bad_comparison = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                task_id,
                "hypothesis_literature_comparison",
                "MedAI bounded PubMed candidate search",
                "proposal-2",
                &serde_json::json!({"candidate_searches":[]}).to_string(),
            )
            .unwrap();
        state
            .store
            .lock()
            .unwrap()
            .transition_task(
                &session.id,
                task_id,
                Some("actor.director"),
                "succeeded",
                None,
                None,
            )
            .unwrap();
        let principal = ApiPrincipal {
            subject: "scientist-9".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = HypothesisLiteratureReview {
            comparison_artifact_id: comparison.id.clone(),
            paper_artifact_id: paper.id.clone(),
            relevance: "partially_relevant".into(),
            rationale: "Shares the mechanism, but the outcome differs.".into(),
            study_design: "observational cohort".into(),
            population_or_model: "adult cohort".into(),
            methods: "longitudinal association".into(),
            findings: "association reported".into(),
            limitations: "single cohort".into(),
        };
        let result = review_hypothesis_literature(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        let saved: serde_json::Value =
            serde_json::from_str(result.1["payload"].as_str().unwrap()).unwrap();
        assert_eq!(saved["reviewer"]["subject"], "scientist-9");
        assert_eq!(saved["hypothesis_id"], "HYP-123");
        assert_eq!(saved["pmid"], "12345");
        assert!(saved["novelty_assessment"].is_null());

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=payload||' ' WHERE id=?1",
            [comparison.id.as_str()],
        )
        .unwrap();
        drop(db);
        let corrupted_comparison = review_hypothesis_literature(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_comparison.0, StatusCode::BAD_REQUEST);
        assert!(corrupted_comparison.1.contains("comparison digest"));

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=?1 WHERE id=?2",
            rusqlite::params![comparison_payload.to_string(), comparison.id],
        )
        .unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=payload||' ' WHERE id=?1",
            [paper.id.as_str()],
        )
        .unwrap();
        drop(db);
        let corrupted_paper = review_hypothesis_literature(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap_err();
        assert_eq!(corrupted_paper.0, StatusCode::BAD_REQUEST);
        assert!(corrupted_paper.1.contains("candidate digest"));

        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE research_artifacts SET payload=?1 WHERE id=?2",
            rusqlite::params![paper_payload.to_string(), paper.id],
        )
        .unwrap();
        drop(db);

        let bad = HypothesisLiteratureReview {
            comparison_artifact_id: bad_comparison.id,
            ..input.clone()
        };
        assert_eq!(
            review_hypothesis_literature(
                State(state.clone()),
                Path(session.id.clone()),
                axum::Extension(principal.clone()),
                Json(bad)
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::BAD_REQUEST
        );
        let forbidden = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        assert_eq!(
            review_hypothesis_literature(
                State(state),
                Path(session.id),
                axum::Extension(forbidden),
                Json(input)
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::FORBIDDEN
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn hypothesis_critic_review_binds_the_exact_human_reviewed_lead() {
        let path = std::env::temp_dir().join(format!(
            "medai-hypothesis-critic-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("hypothesis critic review").unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let worker = ApiPrincipal {
            subject: "worker".into(),
            tenant_id: "local".into(),
            role: "service".into(),
            scientific_agent_id: None,
        };
        let _ = claim_tasks(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(worker),
            Json(ClaimTasks {
                worker_id: "actor.director".into(),
                limit: 1,
                lease_seconds: 120,
                task_ids: Some(vec!["objective.normalize".into()]),
            }),
        )
        .await
        .unwrap();
        let proposal = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                "objective.normalize",
                "hypothesis_candidates",
                "MedAI hypothesis generator",
                "objective-1",
                &serde_json::json!({"candidates": [{"id": "HYP-1"}]}).to_string(),
            )
            .unwrap();
        let candidate = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                "objective.normalize",
                "contradiction_search_candidate",
                "PubMed",
                "12345",
                &serde_json::json!({
                    "target_evidence_id": "EV-1",
                    "paper": {"provenance": {"source": "PubMed", "source_id": "12345"}},
                    "interpretation": "Retrieved as a contradiction-search candidate only; no contradictory relation or evidence polarity was inferred."
                }).to_string(),
            )
            .unwrap();
        let criticism_payload = serde_json::json!({
            "proposal_artifact_id": proposal.id.clone(),
            "proposal_artifact_sha256": proposal.sha256.clone(),
            "reviews": [{
                "hypothesis_id": "HYP-1",
                "linked_contradictory_evidence": [{"evidence_id": "EV-7"}],
                "lexical_refutation_leads": [],
                "retrieved_contradiction_candidates": [{"candidate_artifact_id": candidate.id.clone(), "candidate_artifact_sha256": candidate.sha256.clone(), "target_evidence_id": "EV-1"}]
            }]
        });
        let criticism = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                "objective.normalize",
                "hypothesis_criticism",
                "MedAI scientific critic",
                &proposal.id,
                &criticism_payload.to_string(),
            )
            .unwrap();
        let principal = ApiPrincipal {
            subject: "scientist-9".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = HypothesisCriticReview {
            criticism_artifact_id: criticism.id.clone(),
            hypothesis_id: "HYP-1".into(),
            item_type: "retrieved_candidate".into(),
            item_id: candidate.id.clone(),
            assessment: "context_only".into(),
            rationale: "The cohort differs from the proposed population.".into(),
            study_context: "Retrospective cohort; population mismatch requires follow-up.".into(),
        };
        let result = review_hypothesis_critic(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(result.0, StatusCode::CREATED);
        let saved: serde_json::Value =
            serde_json::from_str(result.1["payload"].as_str().unwrap()).unwrap();
        assert_eq!(saved["reviewer"]["subject"], "scientist-9");
        assert_eq!(saved["criticism_sha256"], criticism.sha256);
        assert_eq!(saved["assessment"], "context_only");
        assert_eq!(saved["hypothesis_confidence_changed"], false);

        let unbound_criticism = state
            .store
            .lock()
            .unwrap()
            .add_artifact(
                &session.id,
                "objective.normalize",
                "hypothesis_criticism",
                "MedAI scientific critic",
                &proposal.id,
                &serde_json::json!({
                    "proposal_artifact_id": proposal.id,
                    "reviews": [{
                        "hypothesis_id": "HYP-1",
                        "linked_contradictory_evidence": [{"evidence_id": "EV-7"}],
                        "lexical_refutation_leads": [],
                        "retrieved_contradiction_candidates": [{"candidate_artifact_id": candidate.id, "candidate_artifact_sha256": candidate.sha256, "target_evidence_id": "EV-1"}]
                    }]
                }).to_string(),
            )
            .unwrap();
        let mut unbound_input = input.clone();
        unbound_input.criticism_artifact_id = unbound_criticism.id;
        assert_eq!(
            review_hypothesis_critic(
                State(state.clone()),
                Path(session.id.clone()),
                axum::Extension(principal.clone()),
                Json(unbound_input),
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::BAD_REQUEST
        );

        let mut unrelated = input.clone();
        unrelated.item_id = "ART-UNLINKED".into();
        assert_eq!(
            review_hypothesis_critic(
                State(state.clone()),
                Path(session.id.clone()),
                axum::Extension(principal.clone()),
                Json(unrelated)
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::BAD_REQUEST
        );
        let forbidden = ApiPrincipal {
            role: "service".into(),
            ..principal
        };
        assert_eq!(
            review_hypothesis_critic(
                State(state),
                Path(session.id.clone()),
                axum::Extension(forbidden),
                Json(input)
            )
            .await
            .unwrap_err()
            .0,
            StatusCode::FORBIDDEN
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn specialist_claim_endpoint_applies_the_optional_task_filter() {
        let path = std::env::temp_dir().join(format!(
            "medai-specialist-claim-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("specialist claim endpoint").unwrap();
        let app = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let unrelated = claim_tasks(
            State(app.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                subject: "test".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(ClaimTasks {
                worker_id: "actor.literature".to_owned(),
                limit: 1,
                lease_seconds: 60,
                task_ids: Some(vec!["literature.discover".to_owned()]),
            }),
        )
        .await
        .unwrap()
        .0;
        assert!(unrelated["tasks"].as_array().unwrap().is_empty());
        let assigned = claim_tasks(
            State(app.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                subject: "test".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(ClaimTasks {
                worker_id: "actor.director".to_owned(),
                limit: 1,
                lease_seconds: 60,
                task_ids: Some(vec!["objective.normalize".to_owned()]),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(assigned["tasks"][0]["id"], "objective.normalize");
        let specialist = ApiPrincipal {
            subject: "worker.literature".into(),
            tenant_id: "local".into(),
            role: "service".into(),
            scientific_agent_id: Some("literature_librarian".into()),
        };
        assert!(authorize_specialist_task(&specialist, "literature.discover").is_ok());
        assert_eq!(
            authorize_specialist_task(&specialist, "objective.normalize")
                .unwrap_err()
                .0,
            StatusCode::FORBIDDEN
        );
        let forbidden = claim_tasks(
            State(app.clone()),
            Path(session.id.clone()),
            axum::Extension(specialist.clone()),
            Json(ClaimTasks {
                worker_id: "untrusted-worker-id".into(),
                limit: 1,
                lease_seconds: 60,
                task_ids: Some(vec!["objective.normalize".into()]),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(forbidden.0, StatusCode::FORBIDDEN);
        let no_scope = claim_tasks(
            State(app),
            Path(session.id.clone()),
            axum::Extension(specialist),
            Json(ClaimTasks {
                worker_id: "untrusted-worker-id".into(),
                limit: 1,
                lease_seconds: 60,
                task_ids: None,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(no_scope.0, StatusCode::FORBIDDEN);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn apply_task_scope_requires_research_director_and_skips_before_specialist_claims() {
        let path = std::env::temp_dir().join(format!(
            "medai-task-scope-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("APOE and Alzheimer disease", "standard", 50)
            .unwrap();
        let objective_task = store
            .claim_ready_tasks(&session.id, "actor.director", 1, 60)
            .unwrap()
            .remove(0);
        assert_eq!(objective_task.id, "objective.normalize");
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
            "schema_version": "2", "objective_sha256": objective_sha256,
            "depth": "standard", "planning": {"selected_task_ids": selected},
        })
        .to_string();
        let artifact = store
            .add_artifact(
                &session.id,
                "objective.normalize",
                "research_objective_plan",
                "MedAI configured model planner",
                &objective_sha256,
                &payload,
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let director = ApiPrincipal {
            subject: "worker.director".into(),
            tenant_id: "local".into(),
            role: "service".into(),
            scientific_agent_id: Some("research_director".into()),
        };
        let forbidden = apply_research_task_scope(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                scientific_agent_id: Some("literature_librarian".into()),
                ..director.clone()
            }),
            Json(ApplyResearchTaskScope {
                worker_id: "actor.director".into(),
                plan_artifact_id: artifact.id.clone(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(forbidden.0, StatusCode::FORBIDDEN);
        let result = apply_research_task_scope(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(director),
            Json(ApplyResearchTaskScope {
                worker_id: "actor.director".into(),
                plan_artifact_id: artifact.id,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(result["skipped_task_ids"].as_array().unwrap().len(), 9);
        let remaining = state
            .store
            .lock()
            .unwrap()
            .budget_summary(&session.id)
            .unwrap()["used"]
            .as_i64()
            .unwrap();
        assert_eq!(
            remaining, 1,
            "skipping excluded tasks must not charge task-start credits"
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn genomic_dataset_upload_is_authenticated_bounded_and_attached_to_the_pending_dag_task()
    {
        let path = std::path::Path::new("/private/tmp")
            .join(format!("medai-vcf-upload-{}.sqlite", uuid::Uuid::new_v4()));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_for_tenant("VCF report upload", "standard", 20, "tenant-a")
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![]),
        };
        let principal = ApiPrincipal {
            subject: "scientist-1".into(),
            tenant_id: "tenant-a".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let report = serde_json::json!({
            "schema_version":"1",
            "source":"MedAI local VCF analysis report",
            "source_vcf_sha256":"a".repeat(64),
            "genome_build":"GRCh38",
            "sample_identifiers_retained":false,
            "raw_variant_records_retained_in_summary":false,
            "remote_sources_queried":false,
            "interpretation_limit":"aggregate receipt only",
            "reference_validated":false,
            "reference_fasta_sha256":null,
            "analysis_implementation_sha256":{"genomics.py":"b".repeat(64)},
            "records_parsed":1,
            "alternate_alleles_emitted":1,
            "variant_class_counts":{"SNV":1},
            "validation_counts":{"reference_unvalidated":1},
            "local_transcript_consequence_counts":{},
            "local_annotated_gene_identifier_count":0
        });
        let (status, body) = submit_genomic_dataset(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(NewGenomicDataset {
                report: report.clone(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(body["kind"], "genomic_dataset");
        assert_eq!(body["task_id"], "genomics.analyze");
        let retained_payload: serde_json::Value =
            serde_json::from_str(body["payload"].as_str().unwrap()).unwrap();
        assert!(retained_payload.get("variants").is_none());
        assert!(retained_payload.get("genotype_call_counts").is_none());

        let reader = ApiPrincipal {
            role: "reader".into(),
            ..principal.clone()
        };
        let forbidden = submit_genomic_dataset(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(reader),
            Json(NewGenomicDataset {
                report: report.clone(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(forbidden.0, StatusCode::FORBIDDEN);
        let wrong_tenant = ApiPrincipal {
            tenant_id: "tenant-b".into(),
            ..principal.clone()
        };
        let hidden = submit_genomic_dataset(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(wrong_tenant),
            Json(NewGenomicDataset {
                report: report.clone(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(hidden.0, StatusCode::NOT_FOUND);
        let mut raw_report = report.clone();
        raw_report["variants"] = serde_json::json!([{"contig":"1","position_1_based":10,"reference_allele":"A","alternate_allele":"G"}]);
        let raw_rejected = submit_genomic_dataset(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(NewGenomicDataset { report: raw_report }),
        )
        .await
        .unwrap_err();
        assert_eq!(raw_rejected.0, StatusCode::BAD_REQUEST);
        let mut unsafe_report = report;
        unsafe_report["sample_identifiers_retained"] = serde_json::Value::Bool(true);
        let rejected = submit_genomic_dataset(
            State(state),
            Path(session.id.clone()),
            axum::Extension(principal),
            Json(NewGenomicDataset {
                report: unsafe_report,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(rejected.0, StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn human_review_uses_authenticated_reviewer_role_and_tenant() {
        let path =
            std::env::temp_dir().join(format!("medai-task-review-{}.sqlite", uuid::Uuid::new_v4()));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_for_tenant("review endpoint", "standard", 20, "tenant-a")
            .unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)
            .unwrap()
            .remove(0);
        store
            .transition_task(
                &session.id,
                &task.id,
                Some("worker"),
                "needs_review",
                Some("source needs scientist review"),
                None,
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                test_credential(
                    b"researcher-token-0123456789abcdef",
                    "reviewer-1",
                    "tenant-a",
                    "researcher",
                ),
                test_credential(
                    b"reader-token-0123456789abcdef00",
                    "reader-1",
                    "tenant-a",
                    "reader",
                ),
                test_credential(
                    b"other-tenant-token-0123456789abcd",
                    "reviewer-2",
                    "tenant-b",
                    "researcher",
                ),
                test_credential(
                    b"service-token-0123456789abcdef",
                    "worker-service",
                    "tenant-a",
                    "service",
                ),
            ]),
        };
        let url = format!("/v1/research/{}/tasks/{}", session.id, task.id);
        let review_url = format!("{url}/review");
        let router = Router::new()
            .route(
                "/v1/research/{id}/tasks/{task_id}/review",
                axum::routing::post(review_task),
            )
            .route(
                "/v1/research/{id}/tasks/{task_id}",
                axum::routing::patch(update_task),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let review_body = serde_json::json!({"decision":"approve","rationale":"Reviewed source-linked evidence and limitations."}).to_string();
        let rejected_by_service = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::PATCH)
                    .uri(&url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer service-token-0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"status":"succeeded"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected_by_service.status(), StatusCode::FORBIDDEN);
        let rejected_reader = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&review_url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer reader-token-0123456789abcdef00",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(review_body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected_reader.status(), StatusCode::FORBIDDEN);
        let wrong_tenant = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&review_url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer other-tenant-token-0123456789abcd",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(review_body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_tenant.status(), StatusCode::NOT_FOUND);
        let approved = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&review_url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-token-0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(review_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(approved.status(), StatusCode::OK);
        let response: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(approved.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(response["status"], "succeeded");
        assert_eq!(response["review"]["reviewer_subject"], "reviewer-1");
        let persisted = state
            .store
            .lock()
            .unwrap()
            .research_tasks(&session.id)
            .unwrap();
        let reviewed_task = persisted.iter().find(|item| item.id == task.id).unwrap();
        assert_eq!(reviewed_task.status, "succeeded");
        assert!(
            reviewed_task
                .output
                .as_deref()
                .unwrap()
                .contains("human_scientific_review")
        );

        let resume_session = state
            .store
            .lock()
            .unwrap()
            .create_session_for_tenant("resume semantic comparisons", "standard", 100, "tenant-a")
            .unwrap();
        let mut resume_target_found = false;
        for _ in 0..40 {
            let store = state.store.lock().unwrap();
            let claimed = store
                .claim_ready_tasks(&resume_session.id, "resume-test-worker", 16, 60)
                .unwrap();
            if claimed.is_empty() {
                break;
            }
            for claimed_task in claimed {
                let (status, output) = if claimed_task.id == "contradictions.search" {
                    resume_target_found = true;
                    ("needs_review", r#"{"semantic_alignment_omitted_count":2}"#)
                } else {
                    ("succeeded", r#"{"test":"dependency setup"}"#)
                };
                store
                    .transition_task(
                        &resume_session.id,
                        &claimed_task.id,
                        Some("resume-test-worker"),
                        status,
                        Some(output),
                        None,
                    )
                    .unwrap();
            }
            if resume_target_found {
                break;
            }
        }
        assert!(
            resume_target_found,
            "contradictions.search should become runnable in the research DAG"
        );
        let resume_url = format!(
            "/v1/research/{}/tasks/contradictions.search/review",
            resume_session.id
        );
        let direct_resume = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::PATCH)
                    .uri(format!(
                        "/v1/research/{}/tasks/contradictions.search",
                        resume_session.id
                    ))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-token-0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"status":"ready"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(direct_resume.status(), StatusCode::FORBIDDEN);
        let resumed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&resume_url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-token-0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "decision":"resume",
                            "rationale":"Continue the remaining bounded semantic comparison batch."
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resumed.status(), StatusCode::OK);
        let resumed_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(resumed.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(resumed_body["status"], "ready");
        assert_eq!(resumed_body["review"]["decision"], "resume");
        let repeated_resume = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(&resume_url)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-token-0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({
                            "decision":"resume",
                            "rationale":"Do not resume after pending work was scheduled."
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(repeated_resume.status(), StatusCode::CONFLICT);
        let resumed_store = state.store.lock().unwrap();
        let resumed_task = resumed_store
            .research_tasks(&resume_session.id)
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == "contradictions.search")
            .unwrap();
        assert_eq!(resumed_task.status, "ready");
        let resumed_session = resumed_store
            .sessions()
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.id == resume_session.id)
            .unwrap();
        assert_eq!(resumed_session.status, "planned");
        let review_event = resumed_store
            .task_events(&resume_session.id)
            .unwrap()
            .into_iter()
            .rev()
            .find(|event| event.task_id == "contradictions.search" && event.to_status == "ready")
            .unwrap();
        assert!(
            review_event
                .details
                .as_deref()
                .unwrap_or_default()
                .contains("Continue the remaining bounded semantic comparison batch.")
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn capability_lifecycle_approval_requires_platform_identity_and_is_audited() {
        let path = std::env::temp_dir().join(format!(
            "medai-capability-approval-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                test_credential(
                    b"researcher-approval-token-123456",
                    "researcher-1",
                    "local",
                    "researcher",
                ),
                test_credential(
                    b"platform-approval-token-12345",
                    "release-operator",
                    "local",
                    "platform_admin",
                ),
            ]),
        };
        let router = Router::new()
            .route(
                "/v1/capabilities",
                get(capabilities).post(register_capability_proposal),
            )
            .route(
                "/v1/capabilities/{id}/transition",
                axum::routing::post(transition_capability),
            )
            .route(
                "/v1/capabilities/{id}/{version}/events",
                get(capability_lifecycle_events),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let registration = r#"{"id":"biomedical.reviewable","version":"1.0.0","kind":"mcp","description":"Reviewed capability","package":{"capability":{"id":"biomedical.reviewable","version":"1.0.0","intent":"Reviewable scientific operation"},"execution":{"enabled":false,"network":false}}}"#;
        let denied_registration = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-approval-token-123456",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(registration))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_registration.status(), StatusCode::FORBIDDEN);
        let registered = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(registration))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(registered.status(), StatusCode::CREATED);
        let registration_result: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(registered.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        let imported_package = serde_json::json!({
            "capability": {"id":"biomedical.reviewable","version":"1.0.0","intent":"Reviewable scientific operation"},
            "execution": {"enabled":false,"network":false}
        });
        let expected_hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&imported_package).unwrap())
        );
        assert_eq!(registration_result["status"], "pending_review");
        assert_eq!(registration_result["package_sha256"], expected_hash);
        let duplicate = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(registration))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
        let registry_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/capabilities")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(registry_response.status(), StatusCode::OK);
        let registry_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(registry_response.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        let manifest_sha256 = registry_json["items"][0]["manifest_sha256"]
            .as_str()
            .unwrap();
        assert_eq!(manifest_sha256.len(), 64);
        let stored_manifest = registry_json["items"][0]["manifest"].as_str().unwrap();
        assert_eq!(
            manifest_sha256,
            format!("{:x}", Sha256::digest(stored_manifest.as_bytes()))
        );
        let incomplete_validation = serde_json::json!({
            "capability_id":"biomedical.reviewable",
            "version":"1.0.0",
            "manifest_sha256":manifest_sha256,
            "checks":{"tests":true,"contract":true}
        })
        .to_string();
        let incomplete_body = serde_json::json!({
            "version":"1.0.0",
            "next":"verified",
            "reason":"Incomplete review.",
            "validation":incomplete_validation,
        })
        .to_string();
        let incomplete = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities/biomedical.reviewable/transition")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(incomplete_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(incomplete.status(), StatusCode::CONFLICT);
        let validation = serde_json::json!({
            "capability_id":"biomedical.reviewable",
            "version":"1.0.0",
            "manifest_sha256":manifest_sha256,
            "checks":{"static_analysis":true,"contract":true,"scientific":true,"reproducibility":true}
        }).to_string();
        let body = serde_json::json!({
            "version":"1.0.0",
            "next":"verified",
            "reason":"Static, contract, scientific, and reproducibility review passed.",
            "validation":validation,
        })
        .to_string();
        let denied = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities/biomedical.reviewable/transition")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-approval-token-123456",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        let approved = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/capabilities/biomedical.reviewable/transition")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(approved.status(), StatusCode::NO_CONTENT);
        let events = router
            .oneshot(
                Request::builder()
                    .uri("/v1/capabilities/biomedical.reviewable/1.0.0/events")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-approval-token-12345",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(events.status(), StatusCode::OK);
        let event_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(events.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(event_json["items"][0]["actor"], "release-operator");
        assert_eq!(event_json["items"][0]["to"], "pending_review");
        assert_eq!(event_json["items"][1]["actor"], "release-operator");
        assert_eq!(event_json["items"][1]["to"], "verified");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bearer_auth_requires_the_exact_configured_token() {
        let secret = b"0123456789abcdef0123456789abcdef";
        let valid = HeaderValue::from_static("Bearer 0123456789abcdef0123456789abcdef");
        let wrong = HeaderValue::from_static("Bearer 0123456789abcdef0123456789abcdeg");
        let malformed = HeaderValue::from_static("Basic 0123456789abcdef0123456789abcdef");
        assert!(bearer_authorized(Some(&valid), Some(secret)));
        assert!(!bearer_authorized(Some(&wrong), Some(secret)));
        assert!(!bearer_authorized(Some(&malformed), Some(secret)));
        assert!(!bearer_authorized(None, Some(secret)));
        assert!(
            bearer_authorized(None, None),
            "loopback development mode has no configured token"
        );
    }

    #[test]
    fn source_scientific_artifacts_require_the_registered_kind_source_and_task() {
        let service = ApiPrincipal {
            subject: "chemistry-worker".into(),
            tenant_id: "lab-a".into(),
            role: "service".into(),
            scientific_agent_id: Some("chemistry_analyst".into()),
        };
        assert!(
            validate_scientific_artifact_origin(
                &service,
                "compounds.analyze",
                "compound_chemical_identity",
                "PubChem PUG REST",
            )
            .is_ok()
        );
        assert!(
            validate_scientific_artifact_origin(
                &service,
                "drugs.investigate",
                "compound_chemical_identity",
                "PubChem PUG REST",
            )
            .is_err()
        );
        assert!(
            validate_scientific_artifact_origin(
                &service,
                "compounds.analyze",
                "compound_chemical_identity",
                "unverified source",
            )
            .is_err()
        );
        assert!(
            validate_scientific_artifact_origin(
                &service,
                "literature.extract",
                "paper_extraction_receipt",
                "PMC BioC",
            )
            .is_ok()
        );
        let researcher = ApiPrincipal {
            role: "researcher".into(),
            ..service
        };
        assert!(
            validate_scientific_artifact_origin(
                &researcher,
                "compounds.analyze",
                "compound_chemical_identity",
                "PubChem PUG REST",
            )
            .is_err()
        );
    }

    #[test]
    fn token_is_required_for_network_binds_and_validated_before_startup() {
        let loopback: SocketAddr = "127.0.0.1:8090".parse().unwrap();
        let network: SocketAddr = "0.0.0.0:8090".parse().unwrap();
        assert!(validate_api_token(None, loopback).unwrap().is_none());
        assert!(
            validate_api_token(Some(String::new()), loopback)
                .unwrap()
                .is_none()
        );
        assert!(validate_api_token(None, network).is_err());
        assert!(validate_api_token(Some("short".into()), network).is_err());
        assert!(
            validate_api_token(Some("0123456789abcdef0123456789abcdef".into()), network)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn token_registry_binds_subject_role_and_tenant() {
        let bind: SocketAddr = "0.0.0.0:8090".parse().unwrap();
        let registry = serde_json::json!([
            {"token":"0123456789abcdef0123456789abcdef","subject":"researcher-7","tenant_id":"tenant-a","role":"researcher"},
            {"token":"abcdef0123456789abcdef0123456789","subject":"reader-2","tenant_id":"tenant-b","role":"reader"}
        ]).to_string();
        let credentials = load_api_credentials(None, Some(registry), "local", bind).unwrap();
        assert_eq!(credentials.len(), 2);
        assert_eq!(credentials[0].tenant_id, "tenant-a");
        assert_eq!(credentials[0].role, "researcher");
        assert!(load_api_credentials(None, None, "local", bind).is_err());
        let invalid_role = serde_json::json!([{"token":"0123456789abcdef0123456789abcdef","subject":"person","tenant_id":"tenant-a","role":"owner"}]).to_string();
        assert!(load_api_credentials(None, Some(invalid_role), "local", bind).is_err());
        let specialist = serde_json::json!([{"token":"0123456789abcdef0123456789abcdef","subject":"agent-literature","tenant_id":"tenant-a","role":"service","scientific_agent_id":"literature_librarian"}]).to_string();
        let credentials = load_api_credentials(None, Some(specialist), "local", bind).unwrap();
        assert_eq!(
            credentials[0].scientific_agent_id.as_deref(),
            Some("literature_librarian")
        );
        let invalid_specialist = serde_json::json!([{"token":"0123456789abcdef0123456789abcdef","subject":"agent-fake","tenant_id":"tenant-a","role":"service","scientific_agent_id":"system_admin"}]).to_string();
        assert!(load_api_credentials(None, Some(invalid_specialist), "local", bind).is_err());
    }

    #[tokio::test]
    async fn tenant_credentials_cannot_read_or_list_other_tenant_sessions() {
        let path =
            std::env::temp_dir().join(format!("medai-api-tenants-{}.sqlite", uuid::Uuid::new_v4()));
        let store = Store::open(&path).unwrap();
        let owned = store
            .create_session_for_tenant("owned objective", "standard", 20, "tenant-a")
            .unwrap();
        let other = store
            .create_session_for_tenant("private objective", "standard", 20, "tenant-b")
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"0123456789abcdef0123456789abcdef",
                "researcher-7",
                "tenant-a",
                "researcher",
            )]),
        };
        let router = Router::new()
            .route("/v1/research", get(sessions).post(create))
            .route("/v1/research/{id}/plan", get(research_plan))
            .route("/v1/research/{id}/status", get(research_status))
            .route(
                "/v1/research/{id}/control",
                get(research_control).post(cancel_research),
            )
            .route("/v1/summary", get(summary))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let listed = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        let listed_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(listed.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        let rows = listed_json["items"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], owned.id);
        let status_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/research/{}/status", owned.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status_response.status(), StatusCode::OK);
        let status_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(status_response.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(status_json["session"]["id"], owned.id);
        assert_eq!(status_json["task_counts"]["ready"], 1);
        assert_eq!(status_json["terminal"], false);
        assert_eq!(status_json["session"]["credits_used"], 0);
        let first_task = state
            .store
            .lock()
            .unwrap()
            .research_tasks(&owned.id)
            .unwrap()
            .into_iter()
            .find(|task| task.status == "ready")
            .unwrap();
        state
            .store
            .lock()
            .unwrap()
            .transition_task(&owned.id, &first_task.id, None, "running", None, None)
            .unwrap();
        state
            .store
            .lock()
            .unwrap()
            .transition_task(
                &owned.id,
                &first_task.id,
                None,
                "failed",
                None,
                Some("fixture failure"),
            )
            .unwrap();
        let stopped = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/research/{}/status", owned.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stopped.status(), StatusCode::OK);
        let stopped_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(stopped.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stopped_json["session"]["status"], "failed");
        assert_eq!(stopped_json["task_counts"]["failed"], 1);
        assert!(stopped_json["task_counts"]["blocked"].as_u64().unwrap() > 0);
        assert_eq!(stopped_json["terminal"], true);
        assert_eq!(stopped_json["session"]["credits_used"], 1);
        let private = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/research/{}/plan", other.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(private.status(), StatusCode::NOT_FOUND);
        let private_status = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/research/{}/status", other.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(private_status.status(), StatusCode::NOT_FOUND);
        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"objective":"new tenant research"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let created_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(created.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(created_json["tenant_id"], "tenant-a");
        let created_id = created_json["id"].as_str().unwrap();
        let cancelled = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{created_id}/control"))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cancelled.status(), StatusCode::OK);
        let cancelled_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(cancelled.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(cancelled_json["cancellation_requested"], true);
        let denied_cancel = router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/v1/research/{}/control", other.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied_cancel.status(), StatusCode::NOT_FOUND);
        let reader_state = App {
            store: state.store.clone(),
            api_credentials: credential_set(vec![test_credential(
                b"abcdef0123456789abcdef0123456789",
                "reader-2",
                "tenant-a",
                "reader",
            )]),
        };
        let reader_router = Router::new()
            .route("/v1/research", axum::routing::post(create))
            .layer(from_fn_with_state(
                reader_state.clone(),
                authenticate_request,
            ))
            .with_state(reader_state);
        let reader_write = reader_router
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"objective":"reader cannot write"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reader_write.status(), StatusCode::FORBIDDEN);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn tenant_admin_can_create_list_and_revoke_credentials_for_own_tenant() {
        let path = std::env::temp_dir().join(format!(
            "medai-api-credentials-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let mut store = Store::open(&path).unwrap();
        store.ensure_tenant("lab-a", "test-bootstrap").unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"0123456789abcdef0123456789abcdef",
                "admin-a",
                "lab-a",
                "tenant_admin",
            )]),
        };
        let router = Router::new()
            .route(
                "/v1/admin/credentials",
                get(list_credentials).post(create_credential),
            )
            .route(
                "/v1/admin/credentials/{id}",
                axum::routing::delete(revoke_credential),
            )
            .route("/v1/summary", get(summary))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/admin/credentials")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"subject":"researcher-9","role":"researcher"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(created.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        let id = body["id"].as_str().unwrap().to_owned();
        let token = body["token"].as_str().unwrap().to_owned();
        assert!(token.len() >= 64);
        let listed = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/credentials")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        let listed_json: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(listed.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(listed_json["items"].as_array().unwrap().len(), 1);
        assert!(listed_json.to_string().find(&token).is_none());
        let revoked = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::DELETE)
                    .uri(format!("/v1/admin/credentials/{id}"))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(revoked.status(), StatusCode::NO_CONTENT);
        let denied = router
            .oneshot(
                Request::builder()
                    .uri("/v1/summary")
                    .header(header::AUTHORIZATION, format!("Bearer {token}"))
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn tenant_directory_can_create_disable_and_reactivate_tenants() {
        let path = std::path::Path::new("/private/tmp").join(format!(
            "medai-tenant-directory-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let mut store = Store::open(&path).unwrap();
        store.ensure_tenant("local", "test-bootstrap").unwrap();
        store.ensure_tenant("lab-a", "test-bootstrap").unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                test_credential(
                    b"platform-tenant-directory-token-001",
                    "platform",
                    "local",
                    "platform_admin",
                ),
                test_credential(
                    b"tenant-admin-directory-token-0001",
                    "admin-a",
                    "lab-a",
                    "tenant_admin",
                ),
                test_credential(
                    b"researcher-directory-token-00001",
                    "researcher-a",
                    "lab-a",
                    "researcher",
                ),
            ]),
        };
        let router = Router::new()
            .route("/v1/auth/whoami", get(authenticated_identity))
            .route("/v1/admin/tenants", get(list_tenants).post(create_tenant))
            .route(
                "/v1/admin/tenants/{tenant_id}",
                get(get_tenant).patch(update_tenant_status),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/admin/tenants")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-tenant-directory-token-001",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"tenant_id":"lab-b"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let created_duplicate = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/admin/tenants")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-tenant-directory-token-001",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"tenant_id":"lab-b"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created_duplicate.status(), StatusCode::CONFLICT);
        let own_entry = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/tenants/lab-a")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer tenant-admin-directory-token-0001",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(own_entry.status(), StatusCode::OK);
        let disabled = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::PATCH)
                    .uri("/v1/admin/tenants/lab-a")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-tenant-directory-token-001",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"status":"disabled","reason":"security review"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(disabled.status(), StatusCode::OK);
        let denied = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/auth/whoami")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-directory-token-00001",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::FORBIDDEN);
        let reactivated = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::PATCH)
                    .uri("/v1/admin/tenants/lab-a")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer platform-tenant-directory-token-001",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(r#"{"status":"active"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(reactivated.status(), StatusCode::OK);
        let accepted = router
            .oneshot(
                Request::builder()
                    .uri("/v1/auth/whoami")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer researcher-directory-token-00001",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn platform_admin_sets_tenant_quotas_and_api_enforces_them() {
        let path =
            std::env::temp_dir().join(format!("medai-api-quota-{}.sqlite", uuid::Uuid::new_v4()));
        let state = App {
            store: Arc::new(Mutex::new(Store::open(&path).unwrap())),
            api_credentials: credential_set(vec![
                test_credential(
                    b"0123456789abcdef0123456789abcdef",
                    "platform",
                    "local",
                    "platform_admin",
                ),
                test_credential(
                    b"abcdef0123456789abcdef0123456789",
                    "admin-a",
                    "lab-a",
                    "tenant_admin",
                ),
            ]),
        };
        let router = Router::new()
            .route(
                "/v1/admin/tenants/{tenant_id}/quota",
                get(get_tenant_quota).put(set_tenant_quota),
            )
            .route("/v1/research", axum::routing::post(create))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let configured = router.clone().oneshot(Request::builder().method(Method::PUT)
            .uri("/v1/admin/tenants/lab-a/quota")
            .header(header::AUTHORIZATION, "Bearer 0123456789abcdef0123456789abcdef")
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(r#"{"max_active_sessions":1,"max_session_budget":50,"max_allocated_credits":50}"#)).unwrap())
            .await.unwrap();
        assert_eq!(configured.status(), StatusCode::OK);
        let oversized = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"objective":"quota test","budget":100}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversized.status(), StatusCode::TOO_MANY_REQUESTS);
        let underfunded = create_with_request_credits(
            State(state.clone()),
            axum::Extension(ApiPrincipal {
                subject: "researcher-a".into(),
                tenant_id: "lab-a".into(),
                role: "researcher".into(),
                scientific_agent_id: None,
            }),
            Json(NewResearch {
                objective: "underfunded request".into(),
                depth: None,
                budget: Some(1),
            }),
            2,
        )
        .await;
        assert_eq!(underfunded.err().unwrap().0, StatusCode::PAYMENT_REQUIRED);
        assert!(
            state
                .store
                .lock()
                .unwrap()
                .sessions_for_tenant("lab-a")
                .unwrap()
                .is_empty()
        );
        let created = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"objective":"quota test","budget":50}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let created_session: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(created.into_body(), 10_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(created_session["credits_used"], 1);
        let at_limit = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/v1/research")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(
                        r#"{"objective":"quota test 2","budget":1}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(at_limit.status(), StatusCode::TOO_MANY_REQUESTS);
        let cross_tenant = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/tenants/lab-b/quota")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cross_tenant.status(), StatusCode::NOT_FOUND);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn tenant_admin_audit_page_and_export_never_include_other_tenant_rows() {
        let path = std::env::temp_dir().join(format!(
            "medai-api-tenant-audit-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let mut store = Store::open(&path).unwrap();
        store.ensure_tenant("lab-a", "test-bootstrap").unwrap();
        store.ensure_tenant("lab-b", "test-bootstrap").unwrap();
        store
            .record_api_audit_event_for_tenant(
                "a-event-1",
                &ApiAuditActor {
                    actor: "lab-a:researcher:r1".into(),
                    tenant_id: Some("lab-a".into()),
                },
                "GET",
                "/v1/summary",
                200,
                1,
            )
            .unwrap();
        store
            .record_api_audit_event_for_tenant(
                "b-event-1",
                &ApiAuditActor {
                    actor: "lab-b:researcher:r2".into(),
                    tenant_id: Some("lab-b".into()),
                },
                "GET",
                "/v1/summary",
                200,
                1,
            )
            .unwrap();
        store
            .record_api_audit_event_for_tenant(
                "a-event-2",
                &ApiAuditActor {
                    actor: "lab-a:researcher:r1".into(),
                    tenant_id: Some("lab-a".into()),
                },
                "POST",
                "/v1/research",
                201,
                2,
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![
                test_credential(
                    b"0123456789abcdef0123456789abcdef",
                    "admin-a",
                    "lab-a",
                    "tenant_admin",
                ),
                ApiCredential {
                    scientific_agent_id: Some("literature_librarian".into()),
                    ..test_credential(
                        b"abcdef0123456789abcdef0123456789",
                        "worker-literature",
                        "lab-a",
                        "service",
                    )
                },
                test_credential(
                    b"fedcba9876543210fedcba9876543210",
                    "general-worker",
                    "lab-a",
                    "service",
                ),
            ]),
        };
        let router = Router::new()
            .route("/v1/admin/audit", get(audit_events))
            .route("/v1/admin/audit/export", get(audit_export))
            .route("/v1/admin/audit/verify", get(audit_verify))
            .route(
                "/v1/literature-watches/due/claim",
                axum::routing::post(|| async { StatusCode::OK }),
            )
            .route(
                "/v1/admin/audit/archive",
                axum::routing::post(audit_archive),
            )
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let page = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit?limit=10")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(page.status(), StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(page.into_body(), 100_000)
                .await
                .unwrap(),
        )
        .unwrap();
        let items = body["items"].as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|event| event["tenant_id"] == "lab-a"));
        let export = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit/export?limit=20")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(export.status(), StatusCode::OK);
        assert_eq!(
            export.headers()[header::CONTENT_TYPE],
            "application/x-ndjson; charset=utf-8"
        );
        let text = String::from_utf8(
            axum::body::to_bytes(export.into_body(), 100_000)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(!text.contains("b-event-1"));
        assert!(text.lines().all(
            |line| serde_json::from_str::<serde_json::Value>(line).unwrap()["tenant_id"] == "lab-a"
        ));
        let verify = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit/verify")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(verify.status(), StatusCode::FORBIDDEN);
        let specialist_global_audit = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(specialist_global_audit.status(), StatusCode::FORBIDDEN);
        let specialist_global_queue = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/literature-watches/due/claim")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer abcdef0123456789abcdef0123456789",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(specialist_global_queue.status(), StatusCode::FORBIDDEN);
        let general_service_audit = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer fedcba9876543210fedcba9876543210",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(general_service_audit.status(), StatusCode::OK);
        let general_service_global_queue = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/literature-watches/due/claim")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer fedcba9876543210fedcba9876543210",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(general_service_global_queue.status(), StatusCode::OK);
        let archive = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/admin/audit/archive")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(archive.status(), StatusCode::FORBIDDEN);
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn bearer_middleware_protects_api_routes_and_leaves_health_public() {
        let path =
            std::env::temp_dir().join(format!("medai-api-auth-{}.sqlite", uuid::Uuid::new_v4()));
        let app_state = App {
            store: Arc::new(Mutex::new(Store::open(&path).unwrap())),
            api_credentials: credential_set(vec![test_credential(
                b"0123456789abcdef0123456789abcdef",
                "test-operator",
                "local",
                "platform_admin",
            )]),
        };
        let router = Router::new()
            .route("/health", get(health))
            .route("/v1/auth/me", get(authenticated_identity))
            .route("/v1/summary", get(summary))
            .route("/v1/admin/metrics", get(operational_metrics))
            .route("/v1/admin/audit", get(audit_events))
            .route("/v1/admin/audit/verify", get(audit_verify))
            .layer(from_fn_with_state(app_state.clone(), authenticate_request))
            .with_state(app_state.clone());

        let unauthorized = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/summary")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let audit_unauthorized = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(audit_unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/summary")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        assert!(authorized.headers().contains_key("x-medai-request-id"));
        let identity = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/auth/me")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(identity.status(), StatusCode::OK);
        let identity: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(identity.into_body(), 10_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(identity["subject"], "test-operator");
        assert_eq!(identity["tenant_id"], "local");
        assert_eq!(identity["role"], "platform_admin");
        let audit_verified = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit/verify")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(audit_verified.status(), StatusCode::OK);
        let audit_page = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/audit?limit=10")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(audit_page.status(), StatusCode::OK);
        let metrics = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/admin/metrics")
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(metrics.status(), StatusCode::OK);
        assert_eq!(
            metrics.headers()[header::CONTENT_TYPE],
            "text/plain; version=0.0.4; charset=utf-8"
        );

        let health_response = router
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health_response.status(), StatusCode::OK);
        let audit = app_state
            .store
            .lock()
            .unwrap()
            .verify_api_audit_chain()
            .unwrap();
        assert_eq!(audit["valid"], true);
        assert_eq!(audit["checked_events"], 8);
        drop(health_response);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn session_api_requests_are_charged_atomically_and_budget_read_remains_available() {
        let path = std::path::Path::new("/private/tmp").join(format!(
            "medai-api-request-meter-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("API metering test", "standard", 1)
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"0123456789abcdef0123456789abcdef",
                "metering-test",
                "local",
                "platform_admin",
            )]),
        };
        let router = Router::new()
            .route("/v1/research/{id}/graph", get(|| async { "graph" }))
            .route("/v1/research/{id}/budget", get(budget_summary))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state.clone());
        let graph_request = || {
            Request::builder()
                .uri(format!("/v1/research/{}/graph", session.id))
                .header(
                    header::AUTHORIZATION,
                    "Bearer 0123456789abcdef0123456789abcdef",
                )
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let first = router.clone().oneshot(graph_request()).await.unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let second = router.clone().oneshot(graph_request()).await.unwrap();
        assert_eq!(second.status(), StatusCode::PAYMENT_REQUIRED);
        let budget = router
            .oneshot(
                Request::builder()
                    .uri(format!("/v1/research/{}/budget", session.id))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer 0123456789abcdef0123456789abcdef",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(budget.status(), StatusCode::OK);
        let summary: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(budget.into_body(), 10_000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(summary["used"], 1);
        assert_eq!(summary["events"][0]["task_id"], "api_request");
        drop(state);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn model_reservation_endpoints_hold_then_refund_measured_credits() {
        let path = std::path::Path::new("/private/tmp").join(format!(
            "medai-model-reservation-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("model reservation API", "standard", 8)
            .unwrap();
        store
            .transition_task(
                &session.id,
                "objective.normalize",
                None,
                "running",
                None,
                None,
            )
            .unwrap();
        let principal = ApiPrincipal {
            subject: "researcher-1".into(),
            tenant_id: "local".into(),
            role: "service".into(),
            scientific_agent_id: None,
        };
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let reserved = reserve_model_credits(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(ModelCreditReservation {
                task_id: "objective.normalize".into(),
                reservation_id: "call-abc".into(),
                credits: 6,
                reason: "pre-call output ceiling".into(),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(reserved["remaining"], 1);
        let researcher = ApiPrincipal {
            role: "researcher".into(),
            ..principal.clone()
        };
        assert!(
            settle_model_credits(
                State(state.clone()),
                Path((session.id.clone(), "call-abc".into())),
                axum::Extension(researcher),
                Json(ModelCreditSettlement {
                    actual_credits: Some(0)
                }),
            )
            .await
            .is_err()
        );
        let settled = settle_model_credits(
            State(state.clone()),
            Path((session.id.clone(), "call-abc".into())),
            axum::Extension(principal),
            Json(ModelCreditSettlement {
                actual_credits: Some(3),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(settled["released_credits"], 3);
        assert_eq!(
            state
                .store
                .lock()
                .unwrap()
                .budget_summary(&session.id)
                .unwrap()["remaining"],
            4
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn artifacts_endpoint_uses_bounded_cursor_pages() {
        let path = std::env::temp_dir().join(format!(
            "medai-artifact-page-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("artifact paging").unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "worker", 1, 60)
            .unwrap()
            .remove(0);
        for index in 0..205 {
            store
                .add_artifact(
                    &session.id,
                    &task.id,
                    "paper",
                    "test",
                    &format!("pmid-{index}"),
                    &format!(
                        r#"{{"id":"{index}","title":"{}"}}"#,
                        if index == 1 {
                            "Parkinson study"
                        } else {
                            "Alzheimer study"
                        }
                    ),
                )
                .unwrap();
        }
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let first = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: Some("paper".into()),
                search: None,
                source_id: None,
                limit: Some(2),
                after_recorded_at: None,
                after_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(first["items"].as_array().unwrap().len(), 2);
        let cursor = first["next_cursor"].clone();
        assert!(cursor.is_object());
        let second = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: Some("paper".into()),
                search: None,
                source_id: None,
                limit: Some(2),
                after_recorded_at: cursor["recorded_at"].as_i64(),
                after_id: cursor["id"].as_str().map(str::to_owned),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(second["items"].as_array().unwrap().len(), 2);
        assert!(second["next_cursor"].is_object());
        let default_page = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: None,
                search: None,
                source_id: None,
                limit: None,
                after_recorded_at: None,
                after_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(default_page["items"].as_array().unwrap().len(), 200);
        assert_eq!(default_page["page_size"], 200);
        assert!(default_page["next_cursor"].is_object());
        let unbounded_search = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: None,
                search: Some("paper".into()),
                source_id: None,
                limit: None,
                after_recorded_at: None,
                after_id: None,
            }),
        )
        .await;
        assert!(unbounded_search.is_err());
        let source_page = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: Some("paper".into()),
                search: None,
                source_id: Some("pmid-204".into()),
                limit: Some(10),
                after_recorded_at: None,
                after_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(source_page["items"].as_array().unwrap().len(), 1);
        assert_eq!(source_page["items"][0]["source_id"], "pmid-204");
        let search_page = artifacts(
            State(state.clone()),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: Some("paper".into()),
                search: Some("ALZHEIMER".into()),
                source_id: None,
                limit: Some(1),
                after_recorded_at: None,
                after_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(search_page["items"].as_array().unwrap().len(), 1);
        let search_cursor = search_page["next_cursor"].clone();
        assert!(search_cursor.is_object());
        let search_second = artifacts(
            State(state),
            Path(session.id.clone()),
            Query(ArtifactFilter {
                kind: Some("paper".into()),
                search: Some("ALZHEIMER".into()),
                source_id: None,
                limit: Some(1),
                after_recorded_at: search_cursor["recorded_at"].as_i64(),
                after_id: search_cursor["id"].as_str().map(str::to_owned),
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(search_second["items"].as_array().unwrap().len(), 1);
        assert!(search_second["next_cursor"].is_object());
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn artifact_lookup_is_scoped_to_the_requested_research_session() {
        let path = std::env::temp_dir().join(format!(
            "medai-artifact-read-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let first = store.create_session("artifact owner").unwrap();
        let second = store.create_session("other session").unwrap();
        let task = store
            .claim_ready_tasks(&first.id, "worker", 1, 60)
            .unwrap()
            .remove(0);
        let artifact = store
            .add_artifact(
                &first.id,
                &task.id,
                "paper",
                "PubMed",
                "12345",
                r#"{"pmid":"12345"}"#,
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let found = artifact_by_id(
            State(state.clone()),
            Path((first.id.clone(), artifact.id.clone())),
            axum::Extension(ApiPrincipal {
                subject: "test.admin".into(),
                tenant_id: "local".into(),
                role: "platform_admin".into(),
                scientific_agent_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(found["id"], artifact.id);
        let cross_tenant = artifact_by_id(
            State(state.clone()),
            Path((first.id.clone(), artifact.id.clone())),
            axum::Extension(ApiPrincipal {
                subject: "test.researcher".into(),
                tenant_id: "other-tenant".into(),
                role: "researcher".into(),
                scientific_agent_id: None,
            }),
        )
        .await;
        assert!(cross_tenant.is_err());
        let hidden = artifact_by_id(
            State(state),
            Path((second.id.clone(), artifact.id)),
            axum::Extension(ApiPrincipal {
                subject: "test.admin".into(),
                tenant_id: "local".into(),
                role: "platform_admin".into(),
                scientific_agent_id: None,
            }),
        )
        .await;
        assert!(hidden.is_err());
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn historical_pubmed_redaction_requires_reviewed_plan_and_records_actor() {
        let path = std::path::Path::new("/private/tmp").join(format!(
            "medai-retention-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("retention API test").unwrap();
        store
            .transition_task(
                &session.id,
                "objective.normalize",
                None,
                "running",
                None,
                None,
            )
            .unwrap();
        store
            .transition_task(
                &session.id,
                "objective.normalize",
                None,
                "succeeded",
                Some("{}"),
                None,
            )
            .unwrap();
        store
            .transition_task(
                &session.id,
                "literature.discover",
                None,
                "running",
                None,
                None,
            )
            .unwrap();
        let artifact = store
            .add_artifact(
                &session.id,
                "literature.discover",
                "paper",
                "PubMed",
                "12345",
                r#"{"pmid":"12345","abstract":"private abstract"}"#,
            )
            .unwrap();
        let principal = ApiPrincipal {
            subject: "reviewer-1".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let preview = preview_pubmed_redaction(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
        )
        .await
        .unwrap()
        .0;
        let plan_sha256 = preview["plan"]["plan_sha256"].as_str().unwrap().to_owned();
        assert_eq!(preview["plan"]["candidate_count"], 1);
        let stale = apply_pubmed_redaction(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal.clone()),
            Json(PubMedRedactionApply {
                plan_sha256: "0".repeat(64),
                reason: "license retention review".into(),
            }),
        )
        .await;
        assert!(stale.is_err());
        let _applied = apply_pubmed_redaction(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(principal),
            Json(PubMedRedactionApply {
                plan_sha256,
                reason: "license retention review".into(),
            }),
        )
        .await
        .unwrap();
        let retained = state
            .store
            .lock()
            .unwrap()
            .artifact(&session.id, &artifact.id)
            .unwrap()
            .unwrap();
        assert!(!retained.payload.contains("private abstract"));
        let receipts = state
            .store
            .lock()
            .unwrap()
            .artifact_redaction_receipts(&session.id)
            .unwrap();
        assert_eq!(receipts[0]["actor"], "reviewer-1");
        let listed = pubmed_redaction_receipts(
            State(state),
            Path(session.id),
            axum::Extension(ApiPrincipal {
                subject: "reviewer-1".into(),
                tenant_id: "local".into(),
                role: "researcher".into(),
                scientific_agent_id: None,
            }),
        )
        .await
        .unwrap()
        .0;
        assert_eq!(listed["items"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn graph_search_route_is_authenticated_bounded_and_reports_truncation() {
        use tower::ServiceExt;
        let path = std::env::temp_dir().join(format!(
            "medai-graph-search-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("graph search API").unwrap();
        for (id, label) in [("gene-a", "Alpha"), ("gene-b", "Alpha beta")] {
            store
                .put_node(&GraphNode {
                    session_id: session.id.clone(),
                    id: id.into(),
                    kind: "Gene".into(),
                    label: label.into(),
                    metadata: "{}".into(),
                })
                .unwrap();
        }
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(vec![test_credential(
                b"graph-search-api-token-123456",
                "graph-researcher",
                "local",
                "researcher",
            )]),
        };
        let router = Router::new()
            .route("/v1/research/{id}/graph/search", get(graph_search))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state);
        let uri = format!("/v1/research/{}/graph/search?q=alpha&limit=1", session.id);
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .header(
                        header::AUTHORIZATION,
                        "Bearer graph-search-api-token-123456",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(result["nodes"].as_array().unwrap().len(), 1);
        assert_eq!(result["truncated"]["nodes"], true);
        assert_eq!(result["truncated"]["edges"], false);
        let invalid = router
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/v1/research/{}/graph/search?q=alpha&limit=101",
                        session.id
                    ))
                    .header(
                        header::AUTHORIZATION,
                        "Bearer graph-search-api-token-123456",
                    )
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn evidence_route_returns_authenticated_keyset_pages() {
        use tower::ServiceExt;
        let path = std::env::temp_dir().join(format!(
            "medai-evidence-page-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("evidence page API").unwrap();
        for id in ["EVD-002", "EVD-001"] {
            store
                .add_evidence(&Evidence {
                    id: id.into(),
                    session_id: session.id.clone(),
                    claim: "pageable claim".into(),
                    source: "test".into(),
                    source_id: None,
                    polarity: "supports".into(),
                    quality: 0.5,
                    provenance: "{}".into(),
                })
                .unwrap();
        }
        let shared_store = Arc::new(Mutex::new(store));
        let state = App {
            store: shared_store.clone(),
            api_credentials: credential_set(vec![test_credential(
                b"evidence-page-api-token-123456",
                "evidence-reader",
                "local",
                "researcher",
            )]),
        };
        let router = Router::new()
            .route("/v1/research/{id}/evidence", get(evidence))
            .layer(from_fn_with_state(state.clone(), authenticate_request))
            .with_state(state);
        let request = |uri: String| {
            Request::builder()
                .uri(uri)
                .header(
                    header::AUTHORIZATION,
                    "Bearer evidence-page-api-token-123456",
                )
                .body(axum::body::Body::empty())
                .unwrap()
        };
        let first = router
            .clone()
            .oneshot(request(format!(
                "/v1/research/{}/evidence?limit=1",
                session.id
            )))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let body = axum::body::to_bytes(first.into_body(), 64 * 1024)
            .await
            .unwrap();
        let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(page["items"][0]["id"], "EVD-001");
        assert_eq!(page["has_more"], true);
        assert_eq!(page["next_cursor"], "EVD-001");
        let snapshot_rowid = page["snapshot_rowid"].as_i64().unwrap();
        shared_store
            .lock()
            .unwrap()
            .add_evidence(&Evidence {
                id: "EVD-0015".into(),
                session_id: session.id.clone(),
                claim: "inserted after snapshot capture".into(),
                source: "test".into(),
                source_id: None,
                polarity: "supports".into(),
                quality: 0.5,
                provenance: "{}".into(),
            })
            .unwrap();

        let last = router
            .clone()
            .oneshot(request(format!(
                "/v1/research/{}/evidence?limit=1&after_id=EVD-001&snapshot_rowid={snapshot_rowid}",
                session.id,
            )))
            .await
            .unwrap();
        let body = axum::body::to_bytes(last.into_body(), 64 * 1024)
            .await
            .unwrap();
        let page: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(page["items"][0]["id"], "EVD-002");
        assert_eq!(page["has_more"], false);
        assert!(page["next_cursor"].is_null());
        assert_eq!(page["snapshot_rowid"], snapshot_rowid);

        let invalid = router
            .clone()
            .oneshot(request(format!(
                "/v1/research/{}/evidence?limit=501",
                session.id
            )))
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        let invalid = router
            .oneshot(request(format!(
                "/v1/research/{}/evidence?after_id=EVD-001",
                session.id
            )))
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod citation_api_tests {
    use super::*;

    #[tokio::test]
    async fn crossref_artifact_routes_index_and_read_session_citation_links() {
        let path = std::env::temp_dir().join(format!(
            "medai-citation-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("citation API test").unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "api-test", 1, 60)
            .unwrap()
            .remove(0);
        store
            .transition_task(
                &session.id,
                &task.id,
                Some("api-test"),
                "succeeded",
                None,
                None,
            )
            .unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "api-test", 1, 60)
            .unwrap()
            .remove(0);
        let app = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let records = serde_json::json!([{"doi":"10.1234/citing","references":[{"DOI":"10.5678/cited","reference_index":0}]}]);
        let retained_records_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&records).unwrap())
        );
        let payload = serde_json::json!({
            "source": "Crossref REST API",
            "complete_result_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "retained_records_sha256": retained_records_sha256,
            "records": records
        });
        let (_, Json(artifact)) = add_artifact(
            State(app.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                subject: "api-test".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(NewArtifact {
                task_id: task.id,
                kind: "crossref_citation_metadata".into(),
                source: "Crossref REST API".into(),
                source_id: "query-test".into(),
                payload,
            }),
        )
        .await
        .unwrap();
        let indexed = index_citations(
            State(app.clone()),
            Path(session.id.clone()),
            Json(IndexCitations {
                artifact_id: artifact["id"].as_str().unwrap().into(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(indexed.0["inserted"], 1);
        let listed = citation_links(
            State(app.clone()),
            Path(session.id.clone()),
            Query(CitationFilter { doi: None }),
        )
        .await
        .unwrap();
        assert_eq!(listed.0["items"].as_array().unwrap().len(), 1);
        assert_eq!(listed.0["items"][0]["cited_doi"], "10.5678/cited");
        assert_eq!(listed.0["metrics"].as_array().unwrap().len(), 2);
        assert_eq!(listed.0["metrics"][0]["doi"], "10.5678/cited");
        assert_eq!(listed.0["metrics"][0]["citations_received"], 1);
        assert!(
            listed.0["semantics"]
                .as_str()
                .unwrap()
                .contains("not evidence polarity")
        );
        drop(app);
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn citation_reference_edge_requires_exact_researcher_adjudication() {
        let path = std::env::temp_dir().join(format!(
            "medai-citation-review-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store.create_session("citation reference review").unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "review-test", 1, 60)
            .unwrap()
            .remove(0);
        let records = serde_json::json!([{
            "doi": "10.1234/citing",
            "references": [{"article-title":"An unresolved but identifiable citation title","author":"Researcher","year":"2020","reference_index":3}]
        }]);
        let records_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&records).unwrap())
        );
        let source_payload = serde_json::json!({
            "source":"Crossref REST API", "complete_result_sha256":"a".repeat(64),
            "retained_records_sha256":records_sha256, "records":records
        });
        let source = store
            .add_artifact(
                &session.id,
                &task.id,
                "crossref_citation_metadata",
                "Crossref REST API",
                "query-source",
                &source_payload.to_string(),
            )
            .unwrap();
        let reference_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&records[0]["references"][0]).unwrap())
        );
        let candidate_records = serde_json::json!([{"doi":"10.5678/candidate","title":["Matching candidate"],"published_date":"2020"}]);
        let candidate_records_sha256 = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&candidate_records).unwrap())
        );
        let candidate_payload = serde_json::json!({
            "schema_version":"1", "source":"Crossref REST API", "status":"unverified_candidates",
            "citing_doi":"10.1234/citing", "reference_index":3,
            "source_artifact_id":source.id, "source_artifact_sha256":source.sha256,
            "reference_sha256":reference_sha256, "query_sha256":"b".repeat(64),
            "candidate_records":candidate_records, "candidate_records_sha256":candidate_records_sha256,
            "complete_result_sha256":"c".repeat(64), "candidate_count":1,
        });
        let candidate = store
            .add_artifact(
                &session.id,
                &task.id,
                "crossref_reference_resolution_candidates",
                "Crossref REST API",
                "reference-review",
                &candidate_payload.to_string(),
            )
            .unwrap();
        let state = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let reviewer = ApiPrincipal {
            subject: "scientist-1".into(),
            tenant_id: "local".into(),
            role: "researcher".into(),
            scientific_agent_id: None,
        };
        let input = CrossrefReferenceReview {
            candidate_artifact_id: candidate.id.clone(),
            candidate_artifact_sha256: candidate.sha256.clone(),
            reference_sha256,
            citing_doi: "10.1234/citing".into(),
            reference_index: 3,
            decision: "match".into(),
            selected_cited_doi: Some("10.5678/candidate".into()),
            rationale:
                "The retained title, author, and publication year match the citation record.".into(),
        };
        let mut uncertain = input.clone();
        uncertain.decision = "uncertain".into();
        uncertain.selected_cited_doi = None;
        uncertain.rationale = "The available bibliographic fields are insufficient.".into();
        let (uncertain_status, Json(uncertain_result)) = review_crossref_reference(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(reviewer.clone()),
            Json(uncertain),
        )
        .await
        .unwrap();
        assert_eq!(uncertain_status, StatusCode::CREATED);
        assert_eq!(uncertain_result["citation_link_created"], false);
        assert!(
            state
                .store
                .lock()
                .unwrap()
                .citation_links(&session.id, None)
                .unwrap()
                .is_empty()
        );
        let (status, Json(first)) = review_crossref_reference(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(reviewer.clone()),
            Json(input.clone()),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(first["citation_link_created"], true);
        assert_eq!(first["decision"], "match");
        let rows = state
            .store
            .lock()
            .unwrap()
            .citation_links(&session.id, None)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].citing_doi, "10.1234/citing");
        assert_eq!(rows[0].cited_doi, "10.5678/candidate");
        assert_eq!(
            rows[0].source_artifact_id,
            first["review_artifact_id"].as_str().unwrap()
        );
        let (_, Json(second)) = review_crossref_reference(
            State(state.clone()),
            Path(session.id.clone()),
            axum::Extension(reviewer),
            Json(input),
        )
        .await
        .unwrap();
        assert_eq!(second["citation_link_created"], false);
        assert_eq!(
            state
                .store
                .lock()
                .unwrap()
                .citation_links(&session.id, None)
                .unwrap()
                .len(),
            1
        );
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn artifact_storage_exhaustion_returns_budget_status_and_keeps_summary_visible() {
        let path = std::env::temp_dir().join(format!(
            "medai-storage-budget-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("storage budget API test", "standard", 1)
            .unwrap();
        let task = store
            .claim_ready_tasks(&session.id, "api-test", 1, 60)
            .unwrap()
            .remove(0);
        let app = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let blob = "x".repeat(medai_core::ARTIFACT_STORAGE_BYTES_PER_CREDIT as usize / 4 - 11);
        let payload = serde_json::json!({"blob": blob});
        for index in 0..3 {
            let _ = add_artifact(
                State(app.clone()),
                Path(session.id.clone()),
                axum::Extension(ApiPrincipal {
                    subject: "api-test".into(),
                    tenant_id: "local".into(),
                    role: "service".into(),
                    scientific_agent_id: None,
                }),
                Json(NewArtifact {
                    task_id: task.id.clone(),
                    kind: "storage_fixture".into(),
                    source: "API test".into(),
                    source_id: format!("fixture-{index}"),
                    payload: payload.clone(),
                }),
            )
            .await
            .unwrap();
        }
        let rejected = add_artifact(
            State(app.clone()),
            Path(session.id.clone()),
            axum::Extension(ApiPrincipal {
                subject: "api-test".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(NewArtifact {
                task_id: task.id,
                kind: "storage_fixture".into(),
                source: "API test".into(),
                source_id: "fixture-3".into(),
                payload,
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(rejected.0, StatusCode::TOO_MANY_REQUESTS);
        let Json(summary) = budget_summary(State(app.clone()), Path(session.id.clone()))
            .await
            .unwrap();
        assert_eq!(summary["used"], 1);
        assert_eq!(
            summary["storage"]["artifact_payload_bytes"],
            medai_core::ARTIFACT_STORAGE_BYTES_PER_CREDIT * 3 / 4
        );
        drop(app);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod trial_watch_api_tests {
    use super::*;

    #[tokio::test]
    async fn session_watch_registration_lease_refresh_history_and_disable() {
        let path = std::env::temp_dir().join(format!(
            "medai-trial-watch-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("API trial watch", "standard", 10)
            .unwrap();
        let app = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };

        let (status, Json(registered)) = register_trial_watch(
            State(app.clone()),
            Path(session.id.clone()),
            Json(RegisterTrialWatch {
                nct_id: "NCT01234567".into(),
                interval_seconds: 3600,
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(registered["enabled"], true);

        let Json(claimed) = claim_trial_watches(
            State(app.clone()),
            axum::Extension(ApiPrincipal {
                subject: "api-test-worker".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(ClaimTrialWatches {
                worker_id: "api-test-worker".into(),
                limit: 1,
                lease_seconds: 120,
            }),
        )
        .await
        .unwrap();
        assert_eq!(claimed["items"].as_array().unwrap().len(), 1);

        let Json(charge) = charge_trial_watch(
            State(app.clone()),
            Path((session.id.clone(), "NCT01234567".into())),
            Json(TrialWatchCharge {
                worker_id: "api-test-worker".into(),
                credits: 1,
                reason: "test source refresh".into(),
            }),
        )
        .await
        .unwrap();
        assert_eq!(charge["used"], 1);
        let snapshot = serde_json::json!({
            "source":"ClinicalTrials.gov",
            "nct_id":"NCT01234567",
            "record":{"protocolSection":{"identificationModule":{"nctId":"NCT01234567"}}},
            "provenance":{"record_snapshot_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
        });
        let (status, Json(event)) = complete_trial_watch(
            State(app.clone()),
            Path((session.id.clone(), "NCT01234567".into())),
            Json(CompleteTrialWatch { worker_id: "api-test-worker".into(), snapshot, comparison: Some(serde_json::json!({"nct_id":"NCT01234567","changed":true,"changed_field_paths":["protocolSection.statusModule.overallStatus"]})) }),
        ).await.unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(event["changed"], true);
        assert!(
            event.get("snapshot").is_none(),
            "refresh response must not duplicate the full source record"
        );

        let Json(watches) = list_trial_watches(State(app.clone()), Path(session.id.clone()))
            .await
            .unwrap();
        assert_eq!(watches["items"][0]["nct_id"], "NCT01234567");
        assert!(watches["items"][0].get("last_snapshot").is_none());
        let Json(latest) = latest_trial_watch_snapshot(
            State(app.clone()),
            Path((session.id.clone(), "NCT01234567".into())),
        )
        .await
        .unwrap();
        assert_eq!(latest["study"]["nct_id"], "NCT01234567");
        let Json(events) = trial_watch_events(
            State(app.clone()),
            Path((session.id.clone(), "NCT01234567".into())),
        )
        .await
        .unwrap();
        assert_eq!(events["items"].as_array().unwrap().len(), 1);
        assert_eq!(events["items"][0]["comparison"]["changed"], true);
        assert_eq!(
            disable_trial_watch(
                State(app.clone()),
                Path((session.id.clone(), "NCT01234567".into()))
            )
            .await
            .unwrap(),
            StatusCode::NO_CONTENT
        );

        drop(app);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn literature_watch_routes_expose_new_pmid_snapshots_and_budgeted_leases() {
        let path = std::env::temp_dir().join(format!(
            "medai-literature-watch-api-{}.sqlite",
            uuid::Uuid::new_v4()
        ));
        let store = Store::open(&path).unwrap();
        let session = store
            .create_session_with_options("API literature watch", "standard", 5)
            .unwrap();
        let app = App {
            store: Arc::new(Mutex::new(store)),
            api_credentials: credential_set(Vec::new()),
        };
        let (status, Json(watch)) = register_literature_watch(
            State(app.clone()),
            Path(session.id.clone()),
            Json(RegisterLiteratureWatch {
                query: "APOE microglia".into(),
                interval_seconds: 86_400,
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        let watch_id = watch["watch_id"].as_str().unwrap().to_owned();
        let Json(claimed) = claim_literature_watches(
            State(app.clone()),
            axum::Extension(ApiPrincipal {
                subject: "worker.literature".into(),
                tenant_id: "local".into(),
                role: "service".into(),
                scientific_agent_id: None,
            }),
            Json(ClaimLiteratureWatches {
                worker_id: "worker.literature".into(),
                limit: 5,
                lease_seconds: 120,
            }),
        )
        .await
        .unwrap();
        assert_eq!(claimed["items"].as_array().unwrap().len(), 1);
        assert_eq!(
            claimed["items"][0]["pending_metadata_pmids"],
            serde_json::json!([])
        );
        let _ = charge_literature_watch(
            State(app.clone()),
            Path((session.id.clone(), watch_id.clone())),
            Json(LiteratureWatchCharge {
                worker_id: "worker.literature".into(),
                credits: 1,
                reason: "PubMed query".into(),
            }),
        )
        .await
        .unwrap();
        let (status, Json(event)) = complete_literature_watch(
            State(app.clone()),
            Path((session.id.clone(), watch_id.clone())),
            Json(CompleteLiteratureWatch {
                worker_id: "worker.literature".into(),
                pmids: vec!["12345".into(), "67890".into()],
                result_count: 500,
                papers: vec![LiteratureWatchPaper {
                    session_id: String::new(),
                    watch_id: String::new(),
                    event_id: String::new(),
                    pmid: "12345".into(),
                    title: "Watch-discovered paper".into(),
                    publication_date: Some("2026".into()),
                    journal: None,
                    doi: None,
                    pmcid: None,
                    captured_at: 0,
                    metadata_sha256: String::new(),
                }],
            }),
        )
        .await
        .unwrap();
        assert_eq!(status, StatusCode::CREATED);
        assert_eq!(event["incomplete"], true);
        assert_eq!(event["retrieved_count"], 2);
        assert_eq!(event["metadata_pending_count"], 1);
        assert_eq!(event["metadata_overflow_count"], 0);
        let Json(watches) = list_literature_watches(State(app.clone()), Path(session.id.clone()))
            .await
            .unwrap();
        assert_eq!(watches["items"][0]["query"], "APOE microglia");
        assert!(watches["items"][0].get("last_pmids").is_none());
        let Json(events) = literature_watch_events(
            State(app.clone()),
            Path((session.id.clone(), watch_id.clone())),
        )
        .await
        .unwrap();
        assert_eq!(events["items"].as_array().unwrap().len(), 1);
        assert_eq!(events["items"][0]["new_pmids"], "[\"12345\",\"67890\"]");
        assert_eq!(events["items"][0]["metadata_pending_count"], 1);
        assert_eq!(events["items"][0]["metadata_overflow_count"], 0);
        let Json(papers) = literature_watch_papers(
            State(app.clone()),
            Path(session.id.clone()),
            Query(std::collections::HashMap::from([(
                "watch_id".to_owned(),
                watch_id.clone(),
            )])),
        )
        .await
        .unwrap();
        assert_eq!(papers["items"][0]["title"], "Watch-discovered paper");
        assert_eq!(papers["items"][0]["pmid"], "12345");
        assert_eq!(
            papers["items"][0]["metadata_sha256"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        assert_eq!(
            disable_literature_watch(State(app.clone()), Path((session.id.clone(), watch_id)))
                .await
                .unwrap(),
            StatusCode::NO_CONTENT
        );
        drop(app);
        std::fs::remove_file(path).unwrap();
    }
}
