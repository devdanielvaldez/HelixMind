#!/bin/sh
set -eu

failed_unit=${1:-medai-backup.service}
case "$failed_unit" in
  ''|*[!A-Za-z0-9_.@:-]*) failed_unit=unknown-unit ;;
esac
logger_bin=$(command -v logger 2>/dev/null || true)
[ -n "$logger_bin" ] || { echo "logger is required to record backup failures" >&2; exit 127; }
"$logger_bin" -p daemon.err -t medai-backup "ALERT: backup unit $failed_unit failed; inspect journal with: journalctl -u $failed_unit"

webhook=${MEDAI_BACKUP_ALERT_WEBHOOK:-}
[ -n "$webhook" ] || exit 0
case "$webhook" in
  https://*|http://127.0.0.1:*|http://localhost:*) ;;
  *) echo "MEDAI_BACKUP_ALERT_WEBHOOK must use HTTPS or loopback HTTP" >&2; exit 2 ;;
esac
command -v curl >/dev/null 2>&1 || { echo "curl is required to deliver MEDAI_BACKUP_ALERT_WEBHOOK" >&2; exit 127; }

timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
payload=$(printf '{"alerts":[{"labels":{"alertname":"MedAIBackupFailed","service":"medai-backup","severity":"critical"},"annotations":{"summary":"MedAI backup job failed","description":"%s failed; inspect the host journal for details."},"startsAt":"%s"}]}' "$failed_unit" "$timestamp")
status=$(curl --silent --show-error --max-time 10 --max-redirs 0 \
  --output /dev/null --write-out '%{http_code}' \
  --header 'Content-Type: application/json' --data "$payload" "$webhook") || {
  echo "Failed to deliver MedAI backup failure alert (HTTP transport error)" >&2
  exit 1
}
case "$status" in
  2??) echo "Delivered MedAI backup failure alert to configured monitoring webhook" ;;
  *) echo "Monitoring webhook rejected MedAI backup failure alert (HTTP $status)" >&2; exit 1 ;;
esac
