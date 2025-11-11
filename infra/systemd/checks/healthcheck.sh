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
    status=1
  fi
done

exit "$status"
