# Operations runbook

This runbook covers the implemented local and single-host deployment. It does not certify a multi-user internet-facing deployment: external immutable audit anchoring, PostgreSQL/object storage, and a production security review are not implemented.

Compose isolates service paths with an internal `api-backend` network: the API and web join it, while scientific MCP and workers also join `provider-egress` for public data sources and optional AgentMesh access. Prometheus can reach the API only through the internal `metrics` network; Alertmanager joins metrics and provider egress to send operator-configured notifications. The API, MCP, and web host ports bind to loopback. MCP HTTP and WebSocket requests require the dedicated `X-MedAI-MCP-Token`; the middleware compares it in constant time and removes credential headers before dispatch. Direct workers use this transport token over the internal Compose network. Keep host firewall rules restrictive and review TLS and trusted proxy behavior before exposing any service. CI verifies this rendered network topology for all Compose profiles.

See [the deployment security review](SECURITY_REVIEW.md) for the reviewed trust boundaries, implemented controls, and remaining production gates.

On Unix, the SQLite database is created with owner-only `0600` permissions, and MedAI tightens an existing database file to the same mode before opening it. Database paths must be regular files; symlinks and special files are rejected. Keep the parent directory writable only by the MedAI service account so another local user cannot replace the database path or inspect SQLite sidecars and backups.

### Capability version comparisons

`medai-science capability-benchmark-versions baseline.py candidate.py benchmark.json --image "$MEDAI_PYTHON_SANDBOX_IMAGE"` runs each version repeatedly in alternating order in the same digest-pinned no-network Docker sandbox. The benchmark JSON contains `schema_version`, `capability_id`, both version IDs, `benchmark_id`, and 10–100 cases with unique `case_id`, object `input`, and object `expected`. The runner derives the benchmark digest from the exact input, records per-case monotonic wall time and per-run process CPU, binds both source digests, execution order, and sandbox image, and reports pass-rate and median-latency differences. `medai-science capability-compare-versions comparison.json` remains available for already collected measurements. Neither command changes lifecycle state; reports are review evidence only, and the corpus still needs independent adjudication for scientific promotion decisions.

AgentMesh clients can run the same comparison on two installation-signed proposals already stored in one session with `medai_capabilities_benchmark_proposal_versions`. This path verifies both proposal signatures and exact source hashes, meters the bounded repeated sandbox executions plus measured CPU, and persists the outcome as a session artifact. It refuses proposals from different capability IDs and never registers or promotes either one.

### Scientific source retention

Newly ingested PubMed paper artifacts and MCP search results omit abstract text and retain only bibliographic metadata, source section labels, character counts, and SHA-256 receipts; transient retrieval text is discarded after ranking and section-label extraction. Drug-interaction receipts additionally use an explicit bibliographic allowlist and mark article licensing as undetermined. PubMed record availability does not establish reuse rights: [NLM notes abstracts may remain copyrighted by publishers or authors](https://www.ncbi.nlm.nih.gov/books/NBK25501/). PMC full-text model processing requires an exact article license allowlisted by the operator. For existing data, first call `GET /v1/research/{id}/pubmed-abstract-retention/preview`, review the exact artifact IDs and plan digest, then explicitly apply with `POST /v1/research/{id}/pubmed-abstract-retention` and JSON `{ "plan_sha256": "<reviewed digest>", "reason": "<retention rationale>" }`. Retrieve durable migration receipts with `GET /v1/research/{id}/pubmed-abstract-retention/receipts`. This is tenant scoped and requires researcher/admin authentication. The transaction replaces eligible PubMed abstract fields, updates the current artifact digest, and records original/replacement digests, plan, actor, reason, and time in an audit receipt; any intervening artifact change invalidates the reviewed plan. It does not rewrite external backups or previously exported bundles, so operators must apply their separate retention policy to those copies. Source-specific cache expiry/update scheduling remains open.

Crossref metadata is cached locally for five minutes in Compose at `/data/crossref-metadata-cache.sqlite`; set `MEDAI_CROSSREF_METADATA_CACHE` to a SQLite path for local runs, or leave it unset to disable the cache. The cache is bounded to 500 entries and 256 KiB per entry, hashes DOI/query keys rather than persisting search text, checks a payload digest on read, and stores normalized bibliographic metadata only. It preserves source-provided license metadata but stores no abstracts, full text, or unstructured reference strings; this metadata cache does not grant rights to referenced articles. Entries expire automatically, and oversized results are returned without caching. Cache hits do not bypass the separate session-budget charge for a requested research operation.

## Configuration and startup

Copy `.env.example` to a private deployment environment. Generate the service bearer outside source control:

```sh
export MEDAI_API_BEARER_TOKEN="$(openssl rand -hex 32)"
export MEDAI_MCP_BEARER_TOKEN="$(openssl rand -hex 32)"
export MEDAI_WEB_SESSION_SECRET="$(openssl rand -hex 32)"
docker compose config --quiet
docker compose up -d --build
curl --fail http://127.0.0.1:8090/health
```

Compose publishes API, MCP, and web ports on host loopback. The API itself listens on the container network. Its healthcheck uses `medai healthcheck` to issue a bounded loopback request to `/health`; MCP, web, and worker containers wait until the API has opened its database and is serving requests. Root filesystems are read-only; durable state is in the `medai-data` volume. API authentication is bearer based; `/health` is public. Direct browser API access uses exact `MEDAI_CORS_ORIGINS`; the web app uses an origin-checked same-origin proxy and requires an interactive login. `MEDAI_WEB_SESSION_SECRET` must be unique, stable across web replicas/restarts, and generated independently from API credentials; the web tier uses it to encrypt eight-hour `HttpOnly`, `SameSite=Strict` session cookies.

For user and tenant scoped API credentials, set `MEDAI_API_TOKENS` to a JSON array such as:

```json
[{"token":"<at-least-32-random-bytes>","subject":"researcher-7","tenant_id":"lab-a","role":"researcher"}]
```

Supported roles are `reader`, `researcher`, `tenant_admin`, `platform_admin`, and `service`. Credentials are imported idempotently; the database stores only SHA-256 digests. A revoked environment credential stays revoked after restart. Create credentials through `/v1/admin/credentials`; the bearer is returned once. Tenant administrators can create only reader/researcher credentials in their own tenant. Revoke with `DELETE /v1/admin/credentials/{id}`. In the web workspace, users log in with their own API credential; the server validates its identity with `/v1/auth/me`, stores it only inside the encrypted browser session cookie, and forwards it from the server proxy. It is not exposed to client JavaScript or local storage. Logout clears the cookie; revocation is enforced by the API on the next request. The cookie expires after eight hours.

Prometheus-compatible operational counts are available at `GET /v1/admin/metrics`. Authenticate with a `tenant_admin`, `platform_admin`, or `service` bearer; counts are scoped to the credential's tenant. The endpoint exports only session and research-task totals with bounded status labels, and does not include session IDs, objectives, or biomedical artifact content. The metrics endpoint is not public. Configure the scraper to attach its bearer from the deployment's secret manager rather than storing credentials in a metrics configuration file.

### Optional Prometheus scraper and alerts

Compose includes opt-in Prometheus and Alertmanager services with tenant-scoped alert rules and webhook delivery. Create a dedicated `service` API credential for the tenant to monitor through the platform-admin credential endpoint; do not reuse the shared API bootstrap token or a human's login credential. Store the one-time token in a host file readable only by the operator (mode `0600`) and set `MEDAI_METRICS_TOKEN_FILE` to that file. Configure a trusted HTTPS webhook compatible with Alertmanager's generic webhook payload, write its URL without a trailing newline to an operator-only file (mode `0600`), and set `MEDAI_ALERT_WEBHOOK_URL_FILE` to that path. The tracked `deploy/metrics-token.example` and `deploy/alert-webhook-url.example` are invalid placeholders for Compose validation and must never be used for live credentials or delivery. Start monitoring with `docker compose --profile monitoring up -d prometheus alertmanager`. Prometheus reaches the API and Alertmanager over an internal Docker network; Alertmanager also joins the default network to deliver to the approved external webhook. The Prometheus and Alertmanager UIs bind to `127.0.0.1:9090` and `127.0.0.1:9093` on the host. Treat those UIs as sensitive tenant data and use a protected host or SSH tunnel for remote access.

Rules are defined in `deploy/prometheus-rules.yml`: they report a missing API scrape, failed research tasks, and tasks awaiting human review. They evaluate tenant-scoped status gauges and therefore remain active while the corresponding state remains present; inspect the MedAI session plan before retrying or clearing work. Prometheus evaluates and displays alerts at `/alerts`, then Alertmanager groups, deduplicates, and sends them to the configured webhook. Webhook delivery contains alert labels and operational descriptions only; do not add objectives, identifiers, source passages, or other research content to alert rules. Pin `MEDAI_PROMETHEUS_IMAGE` and `MEDAI_ALERTMANAGER_IMAGE` to approved immutable digests in production. Prometheus retains 15 days by default (`MEDAI_PROMETHEUS_RETENTION`) in `medai-prometheus`; Alertmanager retains notification state for 120 hours by default (`MEDAI_ALERTMANAGER_RETENTION`) in `medai-alertmanager`. Configure and test backups for these volumes independently from the MedAI database.

Import the checked-in AgentMesh contract catalog into the persistent lifecycle registry with a platform-admin credential: `MEDAI_API_BEARER_TOKEN=... medai-science agentmesh register-capabilities`. The command registers immutable AMCP package hashes as `pending_review`, skips exact previously imported packages, and refuses ID/version collisions; it does not promote capabilities or change AgentMesh routes. Review each imported package in the workspace before it can be routed or recommended for semantic reuse.

Human task reviews use `POST /v1/research/{research_id}/tasks/{task_id}/review` with `{"decision":"approve|reject","rationale":"..."}`. A `researcher` may review tasks only in their own tenant; a `platform_admin` may review across tenants. The API records the authenticated subject, role, decision, and rationale. The generic task update endpoint cannot approve or reject a `needs_review` task. Capability lifecycle transitions, rollback, and event history require a `platform_admin` credential (`POST /v1/capabilities/{id}/transition`, `POST /v1/capabilities/{id}/rollback`, and `GET /v1/capabilities/{id}/{version}/events`). The web workspace's **Review** view exposes these actions after a researcher enters a personal bearer credential; the credential is held in page memory and sent through the origin-checked same-origin proxy. Verification reports must pass `static_analysis`, `contract`, `scientific`, and `reproducibility`; production and rollback also require `benchmark`. The review UI binds reports to the exact capability ID, version, and manifest SHA-256 exposed by `GET /v1/capabilities`; direct API callers must include those fields themselves. The API hashes the full report into the lifecycle audit event. These check values are reviewer attestations and should link to independently retained validation evidence. The backend still enforces role and tenant checks for every request.

For label-versus-literature interaction screening, researchers can adjudicate one exact screened pair through `POST /v1/research/{research_id}/drug-interaction-claim-reviews`. The API rechecks the screening artifact, same-session label and paper source hashes, exact text offsets and hashes, and the label set/version before storing the authenticated reviewer, relation, PK/PD or label-scope domain, exposure/outcome comparability, rationale, and context notes. Paper sources may be PubMed abstracts or PMC full-text sentence receipts linked to the exact PubMed candidate. PMC model screening re-fetches BioC, requires the current digest to match, accepts only a loopback model provider, and requires the exact article license to appear in `MEDAI_PMC_ALLOWED_MODEL_LICENSES` (pipe-separated license values). The full-text sentence itself is used transiently and is not retained in the screen or adjudication. This record remains an evidence annotation: `interaction_verdict` stays null and it cannot be used as clinical advice. The web Literature view uses the authenticated browser session through the same-origin proxy.

IntAct pair reconciliation also reports `source_identity_conflicts` when one IntAct record identifier maps to multiple stable participant-pair groups in a retrieval. This is a source integrity review flag, not evidence that the biology is contradictory; resolve the source records before interpreting it. It does not replace an independently adjudicated precision/recall benchmark.

`medai-science agentmesh sync-status` is read-only and compares the signed local journal with the live AgentMesh snapshot, the current authenticated MedAI capability lifecycle registry, and the persisted MedAI approval receipt. Set `MEDAI_API_URL` and `MEDAI_API_BEARER_TOKEN` as well as the AgentMesh control-plane credentials. The report identifies active routes whose capability is no longer routable, routes absent from the MedAI registry, approved MCP routes in the signed plan that are missing from runtime, approved capabilities not included in that plan, and approval receipts missing or changed in MedAI; legacy journals without receipts are marked unverified. It does not repair drift automatically.

To plan AgentMesh routes from these platform-admin decisions, set `MEDAI_API_URL` and `MEDAI_API_BEARER_TOKEN`, then run `medai-science agentmesh plan-capabilities --service <registered-medai-service> --use-approved-api-lifecycle --output /private/path/medai-routes.json`. The generated plan includes the exact API registry SHA-256 and the `AGENTMESH_TENANT`/`AGENTMESH_NAMESPACE` scope (default `medai`/`biomedical`); catalog entries absent from that registry remain experimental. The review digest covers that scope. After inspecting it, create an authenticated, ten-minute approval receipt with `medai-science agentmesh approve-plan /private/path/medai-routes.json`. Apply only with both the displayed `plan_review_sha256` (`--approve-plan-sha256`) and receipt ID (`--approval-receipt-id`). The API recomputes the plan and resource digests during approval and again before rollout, and the signed local sync journal retains the approving actor and receipt ID.

Local ClinVar-to-VCF reconciliation (`medai-science variant-reconcile`) accepts only E-utilities result records with a VCV accession, matching ClinVar URL/source label, and a recomputed canonical digest of their assembly-location list. It normalizes each location against the exact same local FASTA used for the VCF. Those checks detect inconsistent records; hashes inside caller-supplied JSON do not authenticate ClinVar as the origin, so retain source downloads and retrieval provenance separately when audit-grade external verification is required. Composite haplotype/genotype and complex records remain excluded from exact single-allele matches.

`medai-science dbsnp rs328` (also available as `medai dbsnp rs328`) retrieves one bounded NCBI dbSNP RefSNP record and reports a digest of the exact response. Its placements may span multiple sequence accessions and assemblies; the command does not select an assembly, normalize alleles, or reconcile to local variants. Use the exact local reference-validated variant workflow before joining a placement. The session-scoped `medai_variants_lookup_dbsnp` MCP tool charges the session before retrieval and retains the source receipt as an artifact; it does not run automatically on user VCF data.

To reconcile a retained RefSNP receipt locally, provide the generated VCF report, source receipt, the exact reference FASTA, and a JSON map from RefSeq placement accessions to FASTA contig names. The reconciler accepts only placements explicitly labeled for the requested GRCh37/GRCh38 assembly, checks deleted sequence against the exact local FASTA, converts SPDI to VCF representation, normalizes simple indels, and joins exact alleles:

```sh
medai-science variant-reconcile-dbsnp report.json rs-receipt.json reference.fa sequence-accession-map.json --genome-build GRCh38
```

Example map: `{"NC_000001.11":"1","NC_000016.10":"16"}`. Validate accession versions against the FASTA provider/build. Unsupported or unmapped placements are returned for review; the reconciler does not infer chromosome mappings or clinical significance.

Capture this comparison in a private offline-replayable VCF bundle by passing `--dbsnp-receipt` and `--dbsnp-sequence-map` alongside `--bundle-dir`. Verification recomputes the local normalization and placement join from the captured receipt, FASTA, map, and hash-pinned code; it makes no NCBI request. The response hash is a provenance receipt and does not authenticate the provider or prove scientific interpretation.

To run a specialist worker, give it a separate random token configured in `MEDAI_API_TOKENS` as a `service` credential with its `scientific_agent_id`, then set that exact token as `MEDAI_SPECIALIST_API_TOKEN`. Example role record: `{"token":"<secret>","subject":"worker-literature","tenant_id":"local","role":"service","scientific_agent_id":"literature_librarian"}`. The specialist worker profile does not receive the general `MEDAI_API_BEARER_TOKEN`. It sends its role-bound token as MCP `_meta`; the MCP service validates the identity with the API and scopes every API request for that invocation to the specialist credential. All claims and task-attributed writes must match the role's DAG assignment. Lease recovery remains the general worker's responsibility. Specialist credentials are also denied platform-wide admin/audit routes and global recurring queue claims; only platform admins or unscoped general service credentials can perform those operations.

Platform administrators set per-tenant research limits:

```sh
curl --fail-with-body -X PUT http://127.0.0.1:8090/v1/admin/tenants/lab-a/quota \
  -H "Authorization: Bearer $MEDAI_PLATFORM_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"max_active_sessions":25,"max_session_budget":100000,"max_allocated_credits":1000000}'
```

Read the same URL with `GET` for limits and current usage. Tenant administrators can read only their own quota. Defaults are 25 active sessions, 100,000 credits per session, and 1,000,000 credits allocated across active sessions. Creation is rejected with HTTP 429 when any limit would be exceeded. Completed, cancelled, failed, and budget-exhausted sessions release their allocation.

## AgentMesh and workers

Workers can connect directly to MedAI's authenticated MCP service without AgentMesh credential injection. Set `MEDAI_DIRECT_MCP_URL=http://science-mcp:8091/mcp` and provide the distinct `MEDAI_MCP_BEARER_TOKEN`; then start the opt-in direct profile:

```sh
docker compose --profile direct-worker up -d --build
```

The direct worker service is attached only to the internal `api-backend` network; it has no provider egress and receives the MCP transport token without the general API service credential. It sends the secret only in `X-MedAI-MCP-Token`. This credential must differ from API bearer credentials. AgentMesh remains an optional governed gateway mode: its workers stay on the egress network but do not receive the direct MCP token. Leave `MEDAI_DIRECT_MCP_URL` empty, configure `MEDAI_AGENTMESH_URL`, install reviewed contracts using `integrations/agentmesh/register-capability.sh`, and enable its worker profile:

```sh
docker compose --profile agentmesh-worker up -d --build
```

For independent queue consumers, deploy the direct specialist profile with one registered role selected by `MEDAI_SCIENTIFIC_AGENT_ID` (use `agentmesh-specialist` only when routing through the AgentMesh gateway):

```sh
MEDAI_SCIENTIFIC_AGENT_ID=literature_librarian docker compose --profile direct-specialist up -d --build
```

Other accepted IDs are defined in `science/src/medai_science/scientific_agents.py`. Specialist workers lease only their role's task IDs; each process has a unique lease owner. Run the general and specialist profiles together only when a general worker is intentionally allowed to consume tasks from every specialist. Task filtering is operational routing; the API also enforces specialist identity and DAG role. Keep workers on a trusted network and give them only the service credentials they need. For semantic capability discovery, configure `MEDAI_API_URL`, `MEDAI_API_BEARER_TOKEN`, `AGENTMESH_CONTROL_URL`, `AGENTMESH_ADMIN_TOKEN`, `AGENTMESH_TENANT`, `AGENTMESH_NAMESPACE`, and `MEDAI_AGENTMESH_SERVICE`; that path requires an authenticated MedAI lifecycle registry and a valid AgentMesh runtime snapshot with an exact active route. Any local catalog entry without an exact approved MedAI registry record remains experimental. The search recommends reuse only and does not execute the capability.

Set `MEDAI_AGENT_CREDIT_LIMITS` on the API service to enforce persistent session ceilings per role, for example `{"scientific_critic":40,"research_director":10}`. Task starts and source/model charges count against the specialist assigned to the persisted DAG task and the global session budget. The service also verifies that the authenticated AgentMesh specialist identity is authorized for that task before it can be charged or changed. Existing sessions retain each specialist's ceiling as captured on its first charge; tasks that hit their ceiling move to `needs_review`.

If the worker exits, restart it; expired leases are recovered and retry delays are persisted. Cancel a session with `medai cancel <session-id>` or the API/MCP cancellation operation. Check task status and lease details with `medai research <session-id>`. Do not run the restore procedure while any API, MCP process, or worker can write to the database.

Semantic search in Compose loads a pre-downloaded SentenceTransformer directory at `/data/embedding-model` by default (`MEDAI_EMBEDDING_MODEL`). The model must be materialized without symlinks; its complete file contents are hashed to bind persisted vectors and calibration to the exact weights. The MCP process caches that model for its lifetime, so restart it after replacing model files. By default, semantic scores rank candidates but reuse remains disabled. To enable reuse, generate a report from an independently adjudicated dataset using `medai-science capability-calibrate`, review its holdout metrics and benchmark provenance, place the report inside the shared `medai-data` volume, then set both `MEDAI_CAPABILITY_CALIBRATION_FILE` (for example `/data/capability-calibration.json`) and its exact `MEDAI_CAPABILITY_CALIBRATION_SHA256`. The server rejects a partial configuration, digest mismatch, failed holdout, or model fingerprint mismatch. Synthetic/example labels are not a substitute for independent scientific adjudication.

## Backups, upgrades, and restore

Back up before upgrades, maintenance, or restore. The CLI uses SQLite's online backup mechanism, includes committed WAL state, checks structural and foreign-key integrity, publishes privately without overwriting, and prints a SHA-256 receipt. Verification and restore accept only standalone snapshots without live `-wal`, `-shm`, or rollback-journal sidecars, so the digest covers the complete file:

```sh
medai database backup ./backups/medai-before-upgrade.sqlite
medai database verify ./backups/medai-before-upgrade.sqlite
```

For a dedicated directory containing only MedAI snapshots, preview retention by age and count before deleting anything:

```sh
medai database prune-backups ./backups --keep-newest 7 --max-age-days 90
```

The planner verifies every `.sqlite` candidate and prints its size, digest, timestamp, action, and a plan digest. Invalid or symlinked candidates stop the operation without deleting files. The policy retains the newest requested count and every backup still within the age window. To apply a reviewed plan, rerun it with the same policy and the exact printed digest:

```sh
medai database prune-backups ./backups --keep-newest 7 --max-age-days 90 --apply --confirm-plan-sha256 "$REVIEWED_PLAN_SHA256"
```

The command recalculates the plan and re-verifies every deletion candidate before removing any file. A changed directory or mismatched digest aborts the prune. This operates on local SQLite snapshots only; off-host backup, retention, and disaster recovery remain separately managed.

An operator can schedule this fixed policy after reviewing the dedicated directory and retention limits. The scheduled invocation is explicit and still verifies each backup and binds deletion to the freshly computed plan:

```sh
medai database prune-backups /data/backups --keep-newest 7 --max-age-days 90 --apply-policy
```

Run the scheduled job under a service account that can write only to the backup directory and keep its output as an operational receipt. The command does not upload or remove off-host backups.

For a managed Linux host, install the included systemd timer. The backup job runs as the `medai` account, is network-isolated, locks against overlapping runs, verifies each new snapshot, and applies retention only after verification succeeds. `Persistent=true` catches up a missed daily run after the host returns. Its failure handler always logs to journald and can also send a content-free alert to the Alertmanager v2 API (or another compatible endpoint); remote endpoints must use HTTPS and loopback HTTP is allowed for a local Alertmanager:

```sh
sudo install -d -o root -g medai -m 0750 /etc/medai
sudo install -o root -g medai -m 0640 deploy/backup.env.example /etc/medai/backup.env
sudo install -d -o medai -g medai -m 0700 /var/lib/medai/backups
sudo install -o root -g root -m 0755 deploy/medai-backup.sh /usr/local/libexec/medai-backup.sh
sudo install -o root -g root -m 0755 deploy/medai-backup-alert.sh /usr/local/libexec/medai-backup-alert.sh
sudo install -o root -g root -m 0644 deploy/systemd/medai-backup.service /etc/systemd/system/
sudo install -o root -g root -m 0644 deploy/systemd/medai-backup.timer /etc/systemd/system/
sudo install -o root -g root -m 0644 deploy/systemd/medai-backup-alert@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start medai-backup.service
sudo systemctl enable --now medai-backup.timer
```

Edit `/etc/medai/backup.env` to match the live database and backup paths before starting the service. Set `MEDAI_BACKUP_ALERT_WEBHOOK=http://127.0.0.1:9093/api/v2/alerts` when the local Compose Alertmanager is enabled, or configure a compatible HTTPS endpoint. Keep the backup directory dedicated to MedAI snapshots; the service applies its age/count deletion policy there. Success and failure output is available with `journalctl -u medai-backup.service`; failures also reach the configured alert receiver. Check `systemctl list-timers medai-backup.timer` and run `systemctl start medai-backup.service` after changing the policy. This host scheduler does not provide off-host copies or guarantee restore readiness; replicate snapshots to separate protected storage and periodically test restore.

Schema migrations run when MedAI opens the database. The SQLite `user_version` records the schema version only after a successful transactional upgrade; startup rejects a database stamped by a newer MedAI build before it makes schema changes. Migration DDL and legacy backfills are serialized across concurrent API/worker startup. Test upgrades against a copy of the production database first. For restore:

1. Stop Compose services and every external MedAI API, MCP server, worker, or process using the database.
2. Verify the candidate snapshot and compare its digest with the separately recorded backup receipt.
3. Restore with the explicit maintenance confirmation:

   ```sh
   medai database restore ./backups/medai-before-upgrade.sqlite --confirm-no-writers
   ```

4. Retain the reported pre-restore rollback snapshot. Start services and verify `/health`, session listings, and worker task recovery.

The restore operation re-verifies integrity and checks that the candidate contains the MedAI session schema and a supported schema version before creating a rollback copy or changing live data. A structurally valid unrelated SQLite file is rejected. It captures the opened source connection into a private staged snapshot, confirms the source-file digest did not change during capture, and restores from that exact snapshot. The receipt reports the submitted source digest and captured-snapshot digest separately. It then reopens through current schema migrations. These local snapshots are not an off-host disaster recovery plan; copy them to separately managed encrypted storage and test recovery there.

## Audit and incident review

Authenticated API requests record method, path without query parameters, status, duration, actor, and tenant in a SHA-256-linked chain. Bodies, bearer values, and query strings are not recorded. Platform administrators can page `/v1/admin/audit` and verify the full chain with `/v1/admin/audit/verify`. Tenant administrators can page only their own rows and download NDJSON using `/v1/admin/audit/export?limit=5000`; continue with `after_sequence` and the `X-MedAI-Next-Cursor` response header. Legacy entries without tenant attribution remain verifiable but are omitted from tenant exports.

The chain detects changes across retained rows and archived NDJSON segments. Configure `MEDAI_AUDIT_RETENTION_DAYS` (1–36500) and a durable `MEDAI_AUDIT_ARCHIVE_DIR` (Compose defaults to `/data/audit-archive`) to enable automatic archival. The API runs it periodically using `MEDAI_AUDIT_ARCHIVE_INTERVAL_SECONDS` (default 86400; minimum 60); a platform administrator can also trigger it with `POST /v1/admin/audit/archive`. It archives only the contiguous expired prefix, verifies it before pruning database rows, and keeps sequence/hash checkpoints so `/v1/admin/audit/verify` checks archived and live events as one chain. Each run handles at most 20,000 events. Copy archive files with the database backups and restrict access to the archive directory. The operation does nothing when no contiguous expired prefix exists. Do not manually delete archive files or audit rows. The chain is not externally anchored or immutable against an administrator who can rewrite both the database and its local files. On suspected credential exposure, create a replacement credential, update dependent services, revoke the exposed credential, and review the global or tenant audit history.

## Data handling and limits

Use public research queries only; do not send patient identifiers or protected health information to public literature APIs or model providers. Configure `NCBI_EMAIL` and follow each provider's usage and license terms. Retained artifacts include source IDs, hashes, retrieval metadata, and bounded structured payloads; PMC full text is parsed transiently and excluded from persistent extraction receipts. A source record or model-generated review lead is not automatically a validated scientific claim. Computational hypotheses are not clinical decisions.

Session budgets cover task starts, implemented provider/model/storage operations, and authenticated research API calls. Model calls reserve credits before the provider request using a conservative ceiling of UTF-8 system/prompt/schema bytes, the configured output-token ceiling, and protocol overhead, converted by `MEDAI_MODEL_TOKENS_PER_CREDIT`; successful responses settle to provider-reported usage, while uncertain outcomes retain the full reservation. Only service identities can reserve or settle, and the reservation is bound to the same authenticated subject for both operations. A crashed worker's outstanding reservation becomes uncertain when its task lease is recovered. Each API request under `/v1/research/{session_id}/...` costs one credit by default; set `MEDAI_API_REQUEST_CREDITS` to a positive integer to change the per-request rate. The `POST /v1/research` creation request is charged at the same rate inside the session-creation transaction; creation fails with HTTP 402 and leaves no session behind when its requested budget cannot cover that charge. A valid budget-reservation settlement request is exempt from the per-session request surcharge so a fully reserved call can always settle. Task transitions meter themselves; if the balance cannot cover a transition charge, only a leased task's transition to `needs_review` is allowed through without charge so the worker can pause cleanly. Other task transitions still receive HTTP 402. Researcher and research-director cancellation is also available after exhaustion so an operator can stop a depleted run; other scoped specialist credentials cannot cancel the session. The `GET /v1/research/{session_id}/budget` endpoint remains available after exhaustion so operators can inspect usage. Health checks and non-research API requests without a research-session context are not charged to a session. Physical storage overhead, GPU time, and every system/provider compute cost remain unmetered. Tenant quotas limit active sessions and their declared credit budgets, not complete infrastructure consumption. Generated Python execution requires the digest-pinned, no-network Docker sandbox and still needs a host with a working hardened container runner.

## Common recovery checks

- **API fails on startup:** confirm `MEDAI_API_TOKENS` is valid JSON and non-loopback binds have a valid bearer credential; inspect database migration errors before retrying.
- **Worker cannot claim work:** check that the configured MCP endpoint is reachable, the MCP transport token and API credentials are current, the session is not cancelled, and the task lease/retry time has elapsed.
- **Quota rejects session creation:** inspect `GET /v1/admin/tenants/{tenant_id}/quota`; a platform administrator can adjust the limit after reviewing current usage.
- **Capability reuse returns no match:** check the local embedding model/catalog, authenticated AgentMesh control-plane health and scope, active service route, capability lifecycle metadata, and declared schema/permission fit.
- **Restore or backup fails integrity checks:** preserve the source and receipts, do not replace the live database manually, and recover from another verified snapshot.
# Drug-pair benchmark evaluation

`medai-science drug-pair-benchmark-evaluate benchmark.json` evaluates a frozen two-stage corpus. The top-level document contains `schema_version: "1"`, a benchmark ID and adjudication protocol, plus `retrieval` and `conflict_screen` documents. Each stage uses the double-adjudicated schema (`schema_version: "2"`): source-backed cases have a stable case ID, lowercase SHA-256 digest of the exact source artifact, evidence type (`versioned_label`, `pubmed_record`, or `pmc_full_text`), two independent primary labels, a distinct third-party resolution for disagreements, a categorical prediction, and correctness confidence. Retrieval labels are `retrieved` and `not_retrieved`; conflict-screen labels are `conflict`, `no_conflict`, and `unassessable`.

The retrieval stage must include eligible records that were not returned if recall is to be measured. Precision, recall, and F1 are reported per label in the overall result and separately for each evidence type. Conflict-screen cases must join to a retrieval case by case ID, exact source digest, and evidence type, and the system must have predicted that record as retrieved. Consequently, conflict-screen metrics describe only the retrieved population; they do not include retrieval misses. Calibration bins and correctness Brier/ECE are also reported for each evidence type. The benchmark digest binds the complete input. Fixture corpora validate evaluator behavior only and are not scientific performance results; report corpus selection, reviewer protocol, and independent adjudication alongside any real result.
