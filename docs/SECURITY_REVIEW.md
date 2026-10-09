# Deployment security review

## Scope and method

This is a repository-level review of the single-host Compose deployment, API authentication and tenancy, AgentMesh worker identity, scientific source calls, generated-code execution, audit records, and local database/backup handling. It is based on the checked-in implementation and configuration; it is not a penetration test, external audit, or certification for internet-facing or regulated workloads.

## Assets and trust boundaries

| Asset or boundary | Implemented controls | Remaining exposure |
| --- | --- | --- |
| Research records, source receipts, and tenant data | Session ownership and tenant-filtered API access; bounded artifact reads; SQLite file mode `0600`; transactional schema upgrades; artifact hashes and provenance checks | SQLite, WAL, audit archives, backups, and the HMAC key share a single-host trust boundary; filesystem administrators can replace them together |
| API credentials and browser sessions | Stored API credentials are SHA-256 digests; revocation is persisted; browser sessions are encrypted, `HttpOnly`, same-origin, and expire after eight hours; mutation proxy calls require exact `Origin` | TLS termination and proxy header behavior are operator responsibilities; shared bearer credentials remain available to Compose services that require them |
| Specialist workers and AgentMesh | Specialist credentials bind subject, role, tenant, and task allowlist; API rechecks task assignment; route changes require exact plan digests and an authenticated approval receipt | Compromise of a specialist container exposes that specialist's token and its assigned session data; AgentMesh control-plane compromise remains outside MedAI's boundary |
| Public-source and model traffic | Bounded requests, explicit timeouts, redirects disabled for source clients, patient-query screening, source metering, and documented source/license limits | MCP and worker containers require outbound network access; provider outages, malicious provider content, DNS/TLS infrastructure, and model-provider handling are external dependencies |
| Generated capabilities | Proposals are disabled and pending review; Python/R/WASM execution uses digest-pinned no-network containers with resource limits and schema checks | Real hardened Docker/Wasmtime execution is host-dependent and not exercised in this environment; host kernel/container-runtime vulnerabilities remain in scope |
| Audit and backup records | Tenant-attributed hash chain, authenticated verification/export, bounded archival, verified local snapshots, guarded restore and rollback copy | Hash chains and snapshots are not externally immutable; a privileged host administrator can rewrite local state and its local hashes |
| Network paths | API, web, and MCP are published only on host loopback; internal service paths use `api-backend`, provider traffic uses a separate egress bridge, and Prometheus uses internal `metrics` | MCP requires a dedicated `MEDAI_MCP_BEARER_TOKEN` in `X-MedAI-MCP-Token`, distinct from the API service credential. Direct workers can authenticate to MedAI MCP over the internal Compose network. The optional AgentMesh route still needs trusted upstream credential injection. Compose network isolation does not replace host firewalling, egress allowlists, TLS policy, or review of operator-added networks and proxy rules |

## Required operating conditions

- Keep API, MCP, worker, web, metrics, and backup endpoints behind the documented loopback bindings. MCP HTTP and WebSocket callers must present the dedicated transport bearer; caller `Authorization` is ignored. Do not publish it through a reverse proxy or to an untrusted network. AgentMesh currently filters caller credentials and its checked-in static upstream config cannot inject the separate transport credential, so authenticated AgentMesh routing remains pending the coordinated upstream-credential configuration support.
- Issue distinct, least-privilege credentials per tenant and specialist; do not reuse the platform-admin token for workers.
- Restrict the Docker socket and generated-code runner to the operator-controlled host. Do not mount the Docker socket into application containers.
- Protect `/data`, audit archives, backup directories, model credentials, AgentMesh admin credentials, and the local provenance-signing key with host-level access controls and encrypted off-host backups.
- Review provider licenses and retention terms before changing the explicit allowlists or persisting additional source text.
- Test database restore and audit-chain verification after upgrades and backup-policy changes.

## Findings and release gates

The repository implements meaningful single-host safeguards, but the following items prevent a production security sign-off: no external immutable audit anchor; no PostgreSQL/object-store deployment; no host firewall or provider-domain egress policy in Compose; no independent penetration test; and no verified live execution of generated Python/R/WASM inside the target hardened runner. The Python dependency suite also requires Python 3.11 or newer, and browser checks require the web dependencies and browser runtime declared by CI.

The network segmentation is checked by `scripts/verify-compose-network-isolation.mjs` against the rendered Compose model with every profile enabled. A green result validates only the checked-in Compose topology; it does not attest to the deployed host's effective routes or firewall.
