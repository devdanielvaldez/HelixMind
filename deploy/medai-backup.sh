#!/bin/sh
set -eu

: "${MEDAI_DB:?MEDAI_DB must point to the live SQLite database}"
: "${MEDAI_BACKUP_DIR:?MEDAI_BACKUP_DIR must point to a dedicated backup directory}"

keep_newest=${MEDAI_BACKUP_KEEP_NEWEST:-7}
max_age_days=${MEDAI_BACKUP_MAX_AGE_DAYS:-90}
medai_bin=${MEDAI_BIN:-/usr/local/bin/medai}

case "$keep_newest" in
  ''|*[!0-9]*) echo "MEDAI_BACKUP_KEEP_NEWEST must be an integer" >&2; exit 2 ;;
esac
case "$max_age_days" in
  ''|*[!0-9]*) echo "MEDAI_BACKUP_MAX_AGE_DAYS must be an integer" >&2; exit 2 ;;
esac
[ "$keep_newest" -ge 1 ] && [ "$keep_newest" -le 1000 ] || { echo "MEDAI_BACKUP_KEEP_NEWEST must be 1..1000" >&2; exit 2; }
[ "$max_age_days" -ge 1 ] && [ "$max_age_days" -le 36500 ] || { echo "MEDAI_BACKUP_MAX_AGE_DAYS must be 1..36500" >&2; exit 2; }
[ -x "$medai_bin" ] || { echo "MedAI CLI is not executable: $medai_bin" >&2; exit 127; }
[ -x "$(command -v flock 2>/dev/null || true)" ] || { echo "flock from util-linux is required" >&2; exit 127; }
[ -f "$MEDAI_DB" ] || { echo "MedAI database does not exist: $MEDAI_DB" >&2; exit 1; }

umask 077
mkdir -p "$MEDAI_BACKUP_DIR"
chmod 700 "$MEDAI_BACKUP_DIR"
exec 9>"$MEDAI_BACKUP_DIR/.medai-backup.lock"
if ! flock -n 9; then
  echo "Another MedAI backup job is running" >&2
  exit 1
fi

timestamp=$(date -u '+%Y%m%dT%H%M%SZ')
snapshot="$MEDAI_BACKUP_DIR/medai-$timestamp.sqlite"
echo "Starting MedAI backup: $snapshot"
"$medai_bin" --db "$MEDAI_DB" database backup "$snapshot"
"$medai_bin" database verify "$snapshot"
"$medai_bin" database prune-backups "$MEDAI_BACKUP_DIR" \
  --keep-newest "$keep_newest" --max-age-days "$max_age_days" --apply-policy
echo "MedAI backup cycle completed: $snapshot"
