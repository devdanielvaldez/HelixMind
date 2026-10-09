# Database migrations and recovery

This guide describes the database path currently implemented by MedAI. The runtime currently uses a local SQLite database. PostgreSQL, remote object storage, and their migration and recovery procedures are still open work in `TASKS.md`; this guide does not imply those adapters are available.

On Unix, the runtime creates the database with owner-only `0600` permissions and corrects existing database-file permissions before connecting. Database paths must be regular files rather than symlinks or special files. Keep the parent directory restricted to the MedAI service account to protect WAL/SHM sidecars and prevent path replacement.

## Upgrade

MedAI applies SQLite schema migrations when it opens the database. The current schema version is **2**, stored in `PRAGMA user_version`. Schema changes and legacy data backfills run in a serialized transaction. The version is updated only after the migration succeeds, so a failed migration rolls back its changes. A binary refuses a database whose schema version is newer than it supports before making startup writes.

For every upgrade:

1. Record the current MedAI build and database path. Stop API, MCP, and worker processes before maintenance.
2. Create and verify a standalone backup with the release's CLI:

   ```sh
   medai database backup ./backups/medai-before-upgrade.sqlite
   medai database verify ./backups/medai-before-upgrade.sqlite
   ```

   Retain the printed SHA-256 receipt separately. The backup command does not overwrite an existing destination.
3. Copy the production database and relevant configuration to a restricted staging environment. Start the candidate build against that copy first and check `/health`, session reads, and worker task recovery.
4. If staging succeeds, stop all writers, take and verify a fresh backup, then deploy the candidate build. Opening the database applies the migration.
5. Check service health and representative research sessions before resuming normal traffic.

Do not run two MedAI releases against the same database during an upgrade. If startup reports a newer unsupported schema version, stop and use the matching newer binary or restore a verified pre-upgrade snapshot. Do not try to lower `user_version` manually.

## Restore and downgrade

There is no in-place schema downgrade. To return to an earlier release, restore a snapshot from before the upgrade while all database writers are stopped, then run the compatible earlier release.

```sh
medai database verify ./backups/medai-before-upgrade.sqlite
medai database restore ./backups/medai-before-upgrade.sqlite --confirm-no-writers
```

The restore command requires the explicit `--confirm-no-writers` maintenance flag. It verifies SQLite integrity, MedAI's session schema, and supported schema version before changing live data; it also retains a verified rollback snapshot and reopens the restored database through current migrations. Keep that rollback snapshot until the restored service has been checked.

Backup and restore reject source files with live SQLite `-wal`, `-shm`, or journal sidecars. Use the CLI snapshot instead of copying an active database file. Keep an encrypted off-host copy of backups and test recovery from that copy; local snapshots alone do not provide disaster recovery.

## Migration development

When changing the SQLite schema, add a regression fixture for the prior schema, exercise both successful migration and rollback on injected failure, and test concurrent first openers. Run those checks against a disposable database copy, never a production database. Update this guide and `docs/OPERATIONS.md` when migration or recovery behavior changes.

For operational recovery details, including Compose shutdown and post-restore checks, see [OPERATIONS.md](OPERATIONS.md#backups-upgrades-and-restore).
