#!/usr/bin/env bash
set -euo pipefail

services=(
  "redis-server"
  "daymind-api"
  "daymind-fava"
)

status=0

echo "Service health summary:"
for service in "${services[@]}"; do
  if systemctl is-active --quiet "$service"; then
    echo "✅ $service is active"
  else
    echo "❌ $service is NOT active"
    journalctl -u "$service" -n 80 --no-pager || true
    status=1
  fi
done

APP_PORT="${APP_PORT:-8000}"
echo "Checking HTTP endpoints on port ${APP_PORT}"
if curl -fsS "http://127.0.0.1:${APP_PORT}/healthz" >/dev/null; then
  echo "✅ /healthz responded"
else
  echo "❌ /healthz failed"
  status=1
fi

if curl -fsS "http://127.0.0.1:${APP_PORT}/metrics" >/dev/null; then
  echo "✅ /metrics responded"
else
  echo "⚠️  /metrics not available (non-fatal)"
fi

exit "$status"
