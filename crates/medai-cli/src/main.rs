use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use medai_core::{Capability, Evidence, GraphEdge, GraphNode, Store};
use std::{
    env,
    ffi::OsString,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::Command as ProcessCommand,
    time::Duration,
};

#[derive(Parser)]
#[command(
    name = "medai",
    version,
    about = "Biomedical Autonomous Research Intelligence"
)]
struct Cli {
    #[arg(long, env = "MEDAI_HOME")]
    home: Option<PathBuf>,
    #[arg(
        long = "db",
        env = "MEDAI_DB",
        help = "Use the same SQLite database path as the API"
    )]
    database: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Create a persistent computational research session.
    Investigate {
        objective: String,
        #[arg(long, default_value = "standard", value_parser = ["standard", "deep"])]
        depth: String,
        #[arg(long, default_value_t = 100)]
        budget: i64,
        /// Create the session through the authenticated API, enforcing tenant quotas.
        #[arg(long)]
        api: bool,
    },
    /// Create a tenant-scoped research session and execute its DAG through AgentMesh.
    Autopilot {
        #[arg(long)]
        objective: String,
        #[arg(long, default_value = "standard", value_parser = ["standard", "deep"])]
        depth: String,
        #[arg(long, default_value_t = 100, value_parser = clap::value_parser!(i64).range(1..=100000))]
        budget: i64,
        #[arg(long)]
        paper_limit: Option<usize>,
        #[arg(long, default_value_t = 32)]
        max_tasks: usize,
    },
    /// List local sessions, inspect local tasks, or access tenant-scoped API sessions.
    Research {
        session_id: Option<String>,
        /// Query authenticated sessions owned by the current API tenant.
        #[arg(long)]
        api: bool,
        /// Poll an API-owned session until its DAG has no ready or running work.
        #[arg(long)]
        watch: bool,
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(5..=3600))]
        poll_seconds: u64,
    },
    /// Request cooperative cancellation of a research session.
    Cancel {
        session_id: String,
        /// Cancel a tenant-scoped session through the authenticated API.
        #[arg(long)]
        api: bool,
    },
    /// Claim, heartbeat, complete, and recover leased DAG work.
    Worker {
        #[command(subcommand)]
        command: WorkerCommand,
    },
    /// Inspect or add biomedical graph data scoped to one research session.
    Graph {
        #[command(subcommand)]
        command: GraphCommand,
    },
    /// Search the capability registry.
    Capabilities {
        #[command(subcommand)]
        command: Option<CapabilityCommand>,
    },
    /// Create or verify consistent snapshots of the local SQLite research database.
    Database {
        #[command(subcommand)]
        command: DatabaseCommand,
    },
    /// List Crossref citation links indexed from retained source metadata.
    Citations {
        session_id: String,
        #[arg(long)]
        doi: Option<String>,
    },
    /// Add and list source-linked evidence records.
    Evidence {
        #[command(subcommand)]
        command: EvidenceCommand,
    },
    /// Show AgentMesh integration guidance.
    Agentmesh,
    /// Check that the local MedAI API is accepting health requests.
    Healthcheck {
        #[arg(long, default_value = "127.0.0.1:8090")]
        address: SocketAddr,
    },
    /// Forward a biomedical science command to the Python scientific runtime.
    #[command(external_subcommand)]
    Science(Vec<OsString>),
}

const SCIENCE_COMMANDS: &[&str] = &[
    "papers",
    "research-create",
    "research-list",
    "research-status",
    "research-watch",
    "research-cancel",
    "trials",
    "trial",
    "preprint",
    "preprints-search",
    "crossref",
    "ontology-search",
    "ontology-term",
    "gene",
    "gene-disease",
    "variant",
    "variant-reconcile",
    "variant-annotate-vep",
    "gene-panel",
    "gencc-gene",
    "gencc-disease",
    "clingen-gene",
    "clingen-disease",
    "compound",
    "drug",
    "drug-interactions",
    "protein",
    "protein-record",
    "protein-mapping",
    "structure",
    "interactions",
    "paper-fulltext",
    "pathways",
    "genome-vcf",
    "bundle-verify",
    "bundle-diff",
    "research-diff",
    "sequence-fasta",
    "sequence-fastq",
    "capability-run-python",
    "capability-draft",
    "capabilities-search",
    "benchmark-evaluate",
    "capability-calibrate",
    "capability-compare-versions",
    "capability-benchmark-versions",
    "skills",
    "mcp",
    "agentmesh",
    "workflows",
    "execute",
    "work",
];

fn normalized_science_args(args: &[OsString]) -> Result<Vec<OsString>> {
    let Some(first) = args.first().and_then(|value| value.to_str()) else {
        anyhow::bail!("provide a biomedical science command (for example: medai gene APOE)");
    };
    if first == "agentmesh" {
        anyhow::bail!(
            "use `medai science agentmesh ...`; `medai agentmesh` is reserved for integration guidance"
        );
    }
    let (command, remaining) = if first == "science" {
        let Some(command) = args.get(1).and_then(|value| value.to_str()) else {
            anyhow::bail!("provide a science command after `science`");
        };
        (command, &args[2..])
    } else {
        (first, &args[1..])
    };
    anyhow::ensure!(
        SCIENCE_COMMANDS.contains(&command),
        "unsupported MedAI command `{command}`"
    );
    let mut normalized = Vec::with_capacity(remaining.len() + 1);
    normalized.push(OsString::from(command));
    normalized.extend_from_slice(remaining);
    Ok(normalized)
}

fn run_science_command(args: &[OsString]) -> Result<()> {
    let normalized = normalized_science_args(args)?;
    let executable =
        env::var_os("MEDAI_SCIENCE_BIN").unwrap_or_else(|| OsString::from("medai-science"));
    let status = ProcessCommand::new(executable)
        .args(normalized)
        .status()
        .context(
            "could not start medai-science; install the science package or set MEDAI_SCIENCE_BIN",
        )?;
    anyhow::ensure!(status.success(), "medai-science exited with {status}");
    Ok(())
}

fn run_science_output(args: &[OsString]) -> Result<std::process::Output> {
    let normalized = normalized_science_args(args)?;
    let executable =
        env::var_os("MEDAI_SCIENCE_BIN").unwrap_or_else(|| OsString::from("medai-science"));
    ProcessCommand::new(executable)
        .args(normalized)
        .output()
        .context(
            "could not start medai-science; install the science package or set MEDAI_SCIENCE_BIN",
        )
}

fn research_session_id_from_output(output: &[u8]) -> Result<String> {
    let response: serde_json::Value = serde_json::from_slice(output)
        .context("research-create returned invalid JSON; inspect the API response")?;
    let id = response
        .get("id")
        .and_then(serde_json::Value::as_str)
        .filter(|value| {
            !value.is_empty()
                && value.len() <= 128
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
        .context("research-create response did not contain a valid session id")?;
    Ok(id.to_owned())
}

fn execute_research_args(
    session_id: &str,
    max_tasks: usize,
    paper_limit: Option<usize>,
) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("execute"),
        OsString::from(session_id),
        OsString::from("--max-tasks"),
        OsString::from(max_tasks.to_string()),
    ];
    if let Some(limit) = paper_limit {
        args.push(OsString::from("--paper-limit"));
        args.push(OsString::from(limit.to_string()));
    }
    args
}

fn check_api_health(address: SocketAddr) -> Result<()> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2))
        .with_context(|| format!("could not connect to MedAI API health endpoint at {address}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut status_line = String::new();
    BufReader::new(stream)
        .take(256)
        .read_line(&mut status_line)?;
    validate_health_status_line(&status_line)
}

fn validate_health_status_line(status_line: &str) -> Result<()> {
    anyhow::ensure!(
        status_line.ends_with('\n'),
        "MedAI API health endpoint returned an incomplete status line"
    );
    anyhow::ensure!(
        status_line.starts_with("HTTP/1.1 200 ") || status_line.starts_with("HTTP/1.0 200 "),
        "MedAI API health endpoint returned an unhealthy status"
    );
    Ok(())
}

fn resolve_database_path(home: &std::path::Path, configured_database: Option<PathBuf>) -> PathBuf {
    configured_database.unwrap_or_else(|| home.join("medai.sqlite"))
}

fn api_research_args(objective: &str, depth: &str, budget: i64) -> Vec<OsString> {
    vec![
        OsString::from("research-create"),
        OsString::from(objective),
        OsString::from("--depth"),
        OsString::from(depth),
        OsString::from("--budget"),
        OsString::from(budget.to_string()),
    ]
}
fn api_research_status_args(
    session_id: Option<&str>,
    watch: bool,
    poll_seconds: u64,
) -> Result<Vec<OsString>> {
    anyhow::ensure!(
        !watch || session_id.is_some(),
        "`medai research --api --watch` requires a session ID"
    );
    let command = match (session_id, watch) {
        (None, false) => "research-list",
        (Some(_), false) => "research-status",
        (Some(_), true) => "research-watch",
        (None, true) => unreachable!("watch without a session was rejected"),
    };
    let mut args = vec![OsString::from(command)];
    if let Some(session_id) = session_id {
        args.push(OsString::from(session_id));
    }
    if watch {
        args.push(OsString::from("--poll-seconds"));
        args.push(OsString::from(poll_seconds.to_string()));
    }
    Ok(args)
}
fn api_research_cancel_args(session_id: &str) -> Vec<OsString> {
    ["research-cancel", session_id].map(OsString::from).to_vec()
}
#[derive(Subcommand)]
enum CapabilityCommand {
    /// Register an immutable disabled proposal from a JSON manifest.
    Propose {
        id: String,
        version: String,
        kind: String,
        description: String,
        manifest: PathBuf,
    },
    /// Promote a proposal after local review and a passing JSON validation report.
    Promote {
        id: String,
        version: String,
        #[arg(long)]
        to: String,
        #[arg(long)]
        reviewer: String,
        #[arg(long)]
        reason: String,
        #[arg(long)]
        validation: Option<PathBuf>,
    },
    /// Retire a capability version with an audit reason.
    Deprecate {
        id: String,
        version: String,
        #[arg(long)]
        reviewer: String,
        #[arg(long)]
        reason: String,
    },
    /// Atomically restore a previously verified version and retire the active production version.
    Rollback {
        id: String,
        target_version: String,
        #[arg(long)]
        reviewer: String,
        #[arg(long)]
        reason: String,
        #[arg(long)]
        validation: PathBuf,
    },
    /// Show immutable lifecycle transition history.
    History { id: String, version: String },
}
#[derive(Subcommand)]
enum DatabaseCommand {
    /// Create a new integrity-checked backup without replacing an existing file.
    Backup { destination: PathBuf },
    /// Verify a SQLite backup read-only and print its digest and integrity result.
    Verify { path: PathBuf },
    /// Restore a verified snapshot; stop the API and every worker before confirming.
    Restore {
        source: PathBuf,
        #[arg(
            long,
            required = true,
            help = "Confirm the API and all MedAI workers are stopped"
        )]
        confirm_no_writers: bool,
    },
}
#[derive(Subcommand)]
enum GraphCommand {
    List {
        session_id: String,
    },
    AddNode {
        session_id: String,
        id: String,
        kind: String,
        label: String,
    },
    AddEdge {
        session_id: String,
        source: String,
        relation: String,
        target: String,
        confidence: f64,
        #[arg(long = "evidence-id", required = true)]
        evidence_ids: Vec<String>,
    },
}
#[derive(Subcommand)]
enum EvidenceCommand {
    Add {
        session_id: String,
        claim: String,
        source: String,
        #[arg(long)]
        source_id: Option<String>,
        #[arg(long, value_parser = ["supports", "contradicts", "context"])]
        polarity: String,
        #[arg(long)]
        quality: f64,
    },
    List {
        session_id: String,
    },
}
#[derive(Subcommand)]
enum WorkerCommand {
    Claim {
        session_id: String,
        #[arg(long)]
        worker_id: String,
        #[arg(long, default_value_t = 1)]
        limit: usize,
        #[arg(long, default_value_t = 300)]
        lease_seconds: i64,
    },
    Renew {
        session_id: String,
        task_id: String,
        #[arg(long)]
        worker_id: String,
        #[arg(long, default_value_t = 300)]
        lease_seconds: i64,
    },
    Checkpoint {
        session_id: String,
        task_id: String,
        #[arg(long)]
        worker_id: String,
        checkpoint_json: String,
    },
    Finish {
        session_id: String,
        task_id: String,
        #[arg(long)]
        worker_id: String,
        #[arg(long, value_parser = ["succeeded", "failed", "needs_review", "skipped"])]
        status: String,
        #[arg(long)]
        output: Option<String>,
        #[arg(long)]
        error: Option<String>,
    },
    Recover {
        session_id: String,
        #[arg(long, default_value_t = 25)]
        limit: usize,
    },
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    let home = cli
        .home
        .or_else(|| env::var_os("MEDAI_HOME").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(".medai"));
    let database_path = resolve_database_path(&home, cli.database);
    if let Command::Database {
        command:
            DatabaseCommand::Restore {
                source,
                confirm_no_writers: true,
            },
    } = &cli.command
    {
        let mut store = Store::open(&database_path)?;
        let receipt = store.restore_backup(source)?;
        drop(store);
        // Reopen through the normal startup path so schema migrations run after restoration.
        let _migrated = Store::open(&database_path)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
        return Ok(());
    }
    if let Command::Science(arguments) = &cli.command {
        return run_science_command(arguments);
    }
    if let Command::Healthcheck { address } = &cli.command {
        check_api_health(*address)?;
        return Ok(());
    }
    if let Command::Investigate {
        objective,
        depth,
        budget,
        api: true,
    } = &cli.command
    {
        anyhow::ensure!(
            env::var_os("MEDAI_API_BEARER_TOKEN").is_some(),
            "--api requires MEDAI_API_BEARER_TOKEN"
        );
        return run_science_command(&api_research_args(objective, depth, *budget));
    }
    if let Command::Research {
        session_id,
        api: true,
        watch,
        poll_seconds,
    } = &cli.command
    {
        anyhow::ensure!(
            env::var_os("MEDAI_API_BEARER_TOKEN").is_some(),
            "`medai research --api` requires MEDAI_API_BEARER_TOKEN"
        );
        return run_science_command(&api_research_status_args(
            session_id.as_deref(),
            *watch,
            *poll_seconds,
        )?);
    }
    if let Command::Cancel {
        session_id,
        api: true,
    } = &cli.command
    {
        anyhow::ensure!(
            env::var_os("MEDAI_API_BEARER_TOKEN").is_some(),
            "`medai cancel --api` requires MEDAI_API_BEARER_TOKEN"
        );
        return run_science_command(&api_research_cancel_args(session_id));
    }
    if let Command::Autopilot {
        objective,
        depth,
        budget,
        paper_limit,
        max_tasks,
    } = &cli.command
    {
        anyhow::ensure!(
            env::var_os("MEDAI_API_BEARER_TOKEN").is_some(),
            "autopilot requires MEDAI_API_BEARER_TOKEN for tenant-scoped session creation"
        );
        anyhow::ensure!(
            !objective.trim().is_empty(),
            "autopilot objective must not be empty"
        );
        if let Some(limit) = paper_limit {
            anyhow::ensure!((1..=20).contains(limit), "paper limit must be 1..20");
        }
        anyhow::ensure!((1..=256).contains(max_tasks), "max tasks must be 1..256");
        let creation = run_science_output(&api_research_args(objective, depth, *budget))?;
        anyhow::ensure!(
            creation.status.success(),
            "research session creation failed: {}",
            String::from_utf8_lossy(&creation.stderr).trim()
        );
        let session_id = research_session_id_from_output(&creation.stdout)?;
        println!("Research session {session_id} created; dispatching its DAG through AgentMesh.");
        return run_science_command(&execute_research_args(
            &session_id,
            *max_tasks,
            *paper_limit,
        ))
        .with_context(|| {
            format!(
                "autopilot created session {session_id}, but AgentMesh execution failed; resume with `medai science execute {session_id}`"
            )
        });
    }
    let store = Store::open(&database_path)?;
    match cli.command {
        Command::Investigate {
            objective,
            depth,
            budget,
            api: false,
        } => {
            let s = store.create_session_with_options(&objective, &depth, budget)?;
            println!(
                "{}\nStatus: {}\nDepth: {}\nBudget: {} execution credits\nObjective: {}",
                s.id, s.status, s.depth, s.budget, s.objective
            );
            println!("Session persisted at {}", database_path.display());
        }
        Command::Investigate { api: true, .. } => {
            unreachable!("API research creation returns before opening the local store")
        }
        Command::Autopilot { .. } => {
            unreachable!("autopilot creates and dispatches before opening the local store")
        }
        Command::Research {
            session_id: None, ..
        } => {
            for s in store.sessions()? {
                println!("{}  [{}]  {}", s.id, s.status, s.objective)
            }
        }
        Command::Research {
            session_id: Some(session_id),
            ..
        } => {
            for task in store.research_tasks(&session_id)? {
                println!(
                    "{}  [{}]  {} — {}{}",
                    task.id,
                    task.status,
                    task.domain,
                    task.description,
                    task.lease_owner
                        .as_deref()
                        .map(|owner| format!(" (lease: {}, until {:?})", owner, task.lease_until))
                        .unwrap_or_default()
                );
            }
            for event in store.task_events(&session_id)? {
                println!(
                    "  event {}: {:?} → {} at {}",
                    event.task_id, event.from_status, event.to_status, event.occurred_at
                );
            }
        }
        Command::Cancel {
            session_id,
            api: false,
        } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&store.cancel_session(&session_id)?)?
            );
        }
        Command::Cancel { api: true, .. } => {
            unreachable!("API cancellation returns before opening the local store")
        }
        Command::Worker { command } => match command {
            WorkerCommand::Claim {
                session_id,
                worker_id,
                limit,
                lease_seconds,
            } => {
                let tasks =
                    store.claim_ready_tasks(&session_id, &worker_id, limit, lease_seconds)?;
                println!("{}", serde_json::to_string_pretty(&tasks)?);
            }
            WorkerCommand::Renew {
                session_id,
                task_id,
                worker_id,
                lease_seconds,
            } => {
                let until =
                    store.renew_task_lease(&session_id, &task_id, &worker_id, lease_seconds)?;
                println!("Lease renewed until {until} (Unix time).");
            }
            WorkerCommand::Checkpoint {
                session_id,
                task_id,
                worker_id,
                checkpoint_json,
            } => {
                store.save_task_checkpoint(&session_id, &task_id, &worker_id, &checkpoint_json)?;
                println!("Checkpoint saved for {task_id}.");
            }
            WorkerCommand::Finish {
                session_id,
                task_id,
                worker_id,
                status,
                output,
                error,
            } => {
                store.transition_task(
                    &session_id,
                    &task_id,
                    Some(&worker_id),
                    &status,
                    output.as_deref(),
                    error.as_deref(),
                )?;
                println!("{task_id}: {status}");
            }
            WorkerCommand::Recover { session_id, limit } => {
                let recovered = store.recover_expired_tasks(&session_id, limit)?;
                println!("{}", serde_json::to_string_pretty(&recovered)?);
            }
        },
        Command::Capabilities { command: None } => {
            for c in store.capabilities()? {
                println!("{}@{} [{}] {}", c.id, c.version, c.status, c.description)
            }
        }
        Command::Capabilities {
            command: Some(command),
        } => match command {
            CapabilityCommand::Propose {
                id,
                version,
                kind,
                description,
                manifest,
            } => {
                let manifest = std::fs::read_to_string(manifest)?;
                store.propose_capability(&Capability {
                    id: id.clone(),
                    version,
                    kind,
                    status: "pending_review".into(),
                    description,
                    manifest,
                })?;
                println!("{id}: registered pending_review (execution disabled)");
            }
            CapabilityCommand::Promote {
                id,
                version,
                to,
                reviewer,
                reason,
                validation,
            } => {
                let report = validation.map(std::fs::read_to_string).transpose()?;
                store.transition_capability(
                    &id,
                    &version,
                    &to,
                    &reviewer,
                    &reason,
                    report.as_deref(),
                )?;
                println!("{id}@{version}: {to}");
            }
            CapabilityCommand::Deprecate {
                id,
                version,
                reviewer,
                reason,
            } => {
                store.transition_capability(
                    &id,
                    &version,
                    "deprecated",
                    &reviewer,
                    &reason,
                    None,
                )?;
                println!("{id}@{version}: deprecated");
            }
            CapabilityCommand::Rollback {
                id,
                target_version,
                reviewer,
                reason,
                validation,
            } => {
                let report = std::fs::read_to_string(validation)?;
                store.rollback_capability(&id, &target_version, &reviewer, &reason, &report)?;
                println!(
                    "{id}: restored {target_version} as production; previous production version is rolled_back"
                );
            }
            CapabilityCommand::History { id, version } => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&store.capability_events(&id, &version)?)?
                );
            }
        },
        Command::Database { command } => match command {
            DatabaseCommand::Backup { destination } => {
                let receipt = store.backup(destination)?;
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            DatabaseCommand::Verify { path } => {
                let receipt = Store::verify_backup(path)?;
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            DatabaseCommand::Restore { .. } => {
                anyhow::bail!("database restore requires --confirm-no-writers")
            }
        },
        Command::Citations { session_id, doi } => {
            let items = store.citation_links(&session_id, doi.as_deref())?;
            let metrics = store.citation_metrics(&session_id)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "session_id": session_id,
                    "items": items,
                    "metrics": metrics,
                    "semantics": "Connectivity metrics describe only the potentially incomplete captured Crossref graph; they do not establish paper quality, evidence strength, claim support, or novelty."
                }))?
            );
        }
        Command::Evidence { command } => match command {
            EvidenceCommand::Add {
                session_id,
                claim,
                source,
                source_id,
                polarity,
                quality,
            } => {
                let evidence = Evidence {
                    id: format!("EVD-{}", uuid::Uuid::new_v4().simple()),
                    session_id,
                    claim,
                    source,
                    source_id,
                    polarity,
                    quality,
                    provenance: serde_json::json!({"entered_by":"cli","timestamp_unix":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs()}).to_string(),
                };
                store.add_evidence(&evidence)?;
                println!("{}", evidence.id);
            }
            EvidenceCommand::List { session_id } => {
                for evidence in store.evidence(&session_id)? {
                    println!(
                        "{} [{} {:.2}] {} — {} {}",
                        evidence.id,
                        evidence.polarity,
                        evidence.quality,
                        evidence.claim,
                        evidence.source,
                        evidence.source_id.unwrap_or_default()
                    );
                }
            }
        },
        Command::Agentmesh => {
            println!(
                "AgentMesh integration uses the Streamable HTTP MCP service and config at integrations/agentmesh/config/agentmesh-medai.yaml."
            );
            println!(
                "Biomedical workflows target AgentMesh Workflow IR 1.0 and AMCP capability contracts."
            );
            println!(
                "Generated code remains outside AgentMesh and requires separate sandbox and scientific validation."
            );
        }
        Command::Science(_) => {
            unreachable!("science command was dispatched before opening the database")
        }
        Command::Healthcheck { .. } => {
            unreachable!("healthcheck was dispatched before opening the database")
        }
        Command::Graph { command } => match command {
            GraphCommand::List { session_id } => {
                let (nodes, edges) = store.graph(&session_id)?;
                println!("Nodes: {} | Edges: {}", nodes.len(), edges.len());
                for n in nodes {
                    println!("{} [{}] {}", n.id, n.kind, n.label)
                }
                for e in edges {
                    println!(
                        "{} -{}-> {} ({:.2})",
                        e.source, e.relation, e.target, e.confidence
                    )
                }
            }
            GraphCommand::AddNode {
                session_id,
                id,
                kind,
                label,
            } => {
                store.put_node(&GraphNode {
                    session_id,
                    id,
                    kind,
                    label,
                    metadata: "{}".into(),
                })?;
                println!("Node saved.")
            }
            GraphCommand::AddEdge {
                session_id,
                source,
                relation,
                target,
                confidence,
                evidence_ids,
            } => {
                anyhow::ensure!((0.0..=1.0).contains(&confidence), "confidence must be 0..1");
                store
                    .put_edge(&GraphEdge {
                        session_id,
                        source,
                        relation,
                        target,
                        confidence,
                        evidence_ids: serde_json::to_string(&evidence_ids)?,
                        artifact_ids: "[]".to_owned(),
                    })
                    .context("save edge; both nodes must exist")?;
                println!("Edge saved.")
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthcheck_accepts_only_a_successful_health_response() {
        assert!(validate_health_status_line("HTTP/1.1 200 OK\r\n").is_ok());
        assert!(validate_health_status_line("HTTP/1.0 200 OK\n").is_ok());
        assert!(validate_health_status_line("HTTP/1.1 503 Service Unavailable\r\n").is_err());
        assert!(validate_health_status_line("HTTP/1.1 2000 Not HTTP 200\r\n").is_err());
        assert!(validate_health_status_line("HTTP/1.1 200 OK").is_err());
    }

    #[test]
    fn healthcheck_defaults_to_the_compose_api_address_and_allows_an_override() {
        let default = Cli::try_parse_from(["medai", "healthcheck"]).unwrap();
        assert!(
            matches!(default.command, Command::Healthcheck { address } if address == "127.0.0.1:8090".parse().unwrap())
        );
        let custom =
            Cli::try_parse_from(["medai", "healthcheck", "--address", "127.0.0.1:9090"]).unwrap();
        assert!(
            matches!(custom.command, Command::Healthcheck { address } if address == "127.0.0.1:9090".parse().unwrap())
        );
    }

    #[test]
    fn direct_science_command_is_forwarded_without_shell_parsing() {
        let args = vec![
            OsString::from("gene"),
            OsString::from("APOE"),
            OsString::from("--species"),
            OsString::from("human"),
        ];
        assert_eq!(normalized_science_args(&args).unwrap(), args);
    }

    #[test]
    fn science_namespace_and_options_are_forwarded() {
        let args = vec![
            OsString::from("science"),
            OsString::from("drug-interactions"),
            OsString::from("drug A"),
            OsString::from("drug B"),
        ];
        assert_eq!(normalized_science_args(&args).unwrap(), args[1..]);
    }

    #[test]
    fn agentmesh_admin_commands_are_forwarded_only_inside_the_science_namespace() {
        let args = vec![
            OsString::from("science"),
            OsString::from("agentmesh"),
            OsString::from("sync-status"),
            OsString::from("--state-file"),
            OsString::from("state.json"),
        ];
        assert_eq!(normalized_science_args(&args).unwrap(), args[1..]);
        let direct = vec![OsString::from("agentmesh"), OsString::from("sync-status")];
        assert!(
            normalized_science_args(&direct)
                .unwrap_err()
                .to_string()
                .contains("medai science agentmesh")
        );
    }

    #[test]
    fn authenticated_research_dispatch_preserves_options_without_forwarding_credentials() {
        let args = api_research_args("Alzheimer disease", "deep", 55);
        assert_eq!(
            normalized_science_args(&args).unwrap(),
            [
                "research-create",
                "Alzheimer disease",
                "--depth",
                "deep",
                "--budget",
                "55"
            ]
            .map(OsString::from)
        );
        assert!(
            !args
                .iter()
                .any(|value| value.to_string_lossy().contains("TOKEN"))
        );
    }

    #[test]
    fn tenant_research_status_dispatch_uses_authenticated_science_commands() {
        assert_eq!(
            api_research_status_args(None, false, 30).unwrap(),
            ["research-list"].map(OsString::from),
        );
        assert_eq!(
            api_research_status_args(Some("RES-2026-000128"), false, 30).unwrap(),
            ["research-status", "RES-2026-000128"].map(OsString::from),
        );
        assert_eq!(
            api_research_status_args(Some("RES-2026-000128"), true, 10).unwrap(),
            ["research-watch", "RES-2026-000128", "--poll-seconds", "10"].map(OsString::from),
        );
        assert!(api_research_status_args(None, true, 30).is_err());
        assert_eq!(
            api_research_cancel_args("RES-2026-000128"),
            ["research-cancel", "RES-2026-000128"].map(OsString::from),
        );
    }

    #[test]
    fn autopilot_binds_the_created_session_to_one_agentmesh_execution() {
        assert_eq!(
            research_session_id_from_output(br#"{"id":"RES-2026-000128"}"#).unwrap(),
            "RES-2026-000128"
        );
        assert_eq!(
            execute_research_args("RES-2026-000128", 40, Some(12)),
            [
                "execute",
                "RES-2026-000128",
                "--max-tasks",
                "40",
                "--paper-limit",
                "12"
            ]
            .map(OsString::from)
        );
        assert!(research_session_id_from_output(br#"{"id":"../../outside"}"#).is_err());
        assert!(research_session_id_from_output(br#"{"session_id":"RES-1"}"#).is_err());
    }

    #[test]
    fn arbitrary_subcommands_are_rejected() {
        let args = vec![
            OsString::from("rm"),
            OsString::from("-rf"),
            OsString::from("/"),
        ];
        assert!(
            normalized_science_args(&args)
                .unwrap_err()
                .to_string()
                .contains("unsupported MedAI command")
        );
    }

    #[test]
    fn shared_database_environment_overrides_the_home_default() {
        assert_eq!(
            resolve_database_path(std::path::Path::new("/workspace/.medai"), None),
            PathBuf::from("/workspace/.medai/medai.sqlite")
        );
        assert_eq!(
            resolve_database_path(
                std::path::Path::new("/workspace/.medai"),
                Some(PathBuf::from("/shared/medai.sqlite")),
            ),
            PathBuf::from("/shared/medai.sqlite")
        );
    }
}
