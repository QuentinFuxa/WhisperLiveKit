#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Usage: remote_deploy.sh --host <ip> [--user root] [--key path] [--repo-path /opt/daymind] [--repo-url url] [--verify-only]

Deploys the current working tree to the remote host, ensures the git repo exists,
rebuilds a minimal torch-free runtime, seeds env defaults, enforces systemd units,
and verifies /healthz.
USAGE
}

HOST=""
USER="root"
KEY_PATH="${SSH_KEY_PATH:-}"
: "${REPO_URL:=https://github.com/noba-dkg-aion/daymind.git}"
: "${APP_DIR:=/opt/daymind}"
VERIFY_ONLY=false
declare -a CURL_HEADER_ARGS=()

wait_for_port() {
  local port="$1"
  local attempts=60
  for attempt in $(seq 1 "$attempts"); do
    if ss -lntp | grep -q ":${port} "; then
      echo "✅ Port ${port} is listening"
      return 0
    fi
    sleep 1
  done
  echo "::error::Port ${port} failed to open within ${attempts}s" >&2
  return 1
}

curl_with_retry() {
  local url="$1"
  local label="$2"
  local mode="${3:-plain}"
  local attempts=10
  for attempt in $(seq 1 "$attempts"); do
    if [ "$mode" = "head" ]; then
      if curl -fsS "${CURL_HEADER_ARGS[@]}" "$url" | head >/dev/null; then
        echo "✅ ${label}"
        return 0
      fi
    else
      if curl -fsS "${CURL_HEADER_ARGS[@]}" "$url" >/dev/null; then
        echo "✅ ${label}"
        return 0
      fi
    fi
    echo "Retry ${label} (${attempt}/${attempts})"
    sleep 2
  done
  echo "::error::${label} failed after ${attempts} attempts" >&2
  return 1
}

run_local_verify() {
  local env_file=/etc/default/daymind
  if [ ! -f "$env_file" ]; then
    echo "::warning::$env_file missing; creating placeholder"
    touch "$env_file"
  fi
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
  fi
  local app_host="${APP_HOST:-127.0.0.1}"
  local app_port="${APP_PORT:-8000}"
  local api_log="${APP_DIR}/api.log"

  systemctl daemon-reload || true
  if systemctl list-unit-files daymind-api.service >/dev/null 2>&1; then
    systemctl restart daymind-api.service || true
  else
    echo "daymind-api.service not found; starting uvicorn manually"
    pkill -f "uvicorn src.api.main" >/dev/null 2>&1 || true
    nohup "${APP_DIR}/venv/bin/uvicorn" src.api.main:app --host "$app_host" --port "$app_port" >> "$api_log" 2>&1 &
  fi

  wait_for_port "$app_port"

  CURL_HEADER_ARGS=()
  if [ -n "${DAYMIND_API_KEY:-}" ]; then
    CURL_HEADER_ARGS=(-H "X-API-Key: ${DAYMIND_API_KEY}")
  fi

  curl_with_retry "http://127.0.0.1:${app_port}/healthz" "/healthz"
  curl_with_retry "http://127.0.0.1:${app_port}/metrics" "/metrics" "head"
  echo "✅ Remote API verified on ${app_host}:${app_port}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --user)
      USER="${2:-}"
      shift 2
      ;;
    --key)
      KEY_PATH="${2:-}"
      shift 2
      ;;
    --repo-path)
      APP_DIR="${2:-}"
      shift 2
      ;;
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --verify-only)
      VERIFY_ONLY=true
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$VERIFY_ONLY" == false && -z "$HOST" ]]; then
  echo "Missing required --host." >&2
  usage
  exit 1
fi

if [[ "$VERIFY_ONLY" == true ]]; then
  run_local_verify
  exit 0
fi

if [[ -z "$KEY_PATH" ]]; then
  KEY_PATH="$HOME/.ssh/id_rsa"
fi

if [[ ! -f "$KEY_PATH" ]]; then
  echo "SSH key not found at $KEY_PATH" >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "==> Syncing local repo (git pull --rebase)"
  git pull --rebase >/dev/null 2>&1 || true
fi

SSH_OPTS=(-i "$KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
REMOTE="${USER}@${HOST}"
SUDO=""
if [[ "$USER" != "root" ]]; then
  SUDO="sudo"
fi

ssh_exec() {
# shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

run_remote_verify() {
  echo "==> Restarting services and verifying health"
  ssh_exec "
    set -euo pipefail
    APP_DIR='$APP_DIR'
    if [ -x \"\$APP_DIR/scripts/remote_deploy.sh\" ]; then
      APP_DIR='$APP_DIR' \"\$APP_DIR/scripts/remote_deploy.sh\" --verify-only
    else
      echo \"remote_deploy.sh not found at \$APP_DIR/scripts\" >&2
      exit 1
    fi
  "
}

echo "==> Ensuring daymind system account and repo"
ssh_exec "
  set -euo pipefail
  if ! command -v git >/dev/null 2>&1; then
    $SUDO apt-get update -y
    $SUDO apt-get install -y --no-install-recommends git
  fi
  if ! getent group daymind >/dev/null 2>&1; then
    $SUDO groupadd --system daymind
  fi
  if ! id -u daymind >/dev/null 2>&1; then
    $SUDO useradd --system --create-home --home '$APP_DIR' --shell /bin/bash -g daymind daymind
  fi
  $SUDO mkdir -p '$APP_DIR'
  if [ -d '$APP_DIR/.git' ]; then
    $SUDO -u daymind git -C '$APP_DIR' fetch --all -p || true
    $SUDO -u daymind git -C '$APP_DIR' checkout -f main
    $SUDO -u daymind git -C '$APP_DIR' reset --hard origin/main
  else
    $SUDO rm -rf '$APP_DIR'
    $SUDO -u daymind git clone '$REPO_URL' '$APP_DIR'
  fi
  $SUDO chown -R daymind:daymind '$APP_DIR'
"

echo "==> Rsyncing working tree"
RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.venv"
  "--exclude=venv"
  "--exclude=runtime"
  "--exclude=mobile/android/daymind/app/build"
  "--exclude=.gradle"
  "--exclude=dist"
)
rsync -az --delete "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE:$APP_DIR/"
ssh_exec "$SUDO chown -R daymind:daymind '$APP_DIR'"

echo "==> Ensuring Redis, runtime scaffolding, and ledger"
ssh_exec "
  set -euo pipefail
  if ! $SUDO systemctl is-enabled --quiet redis-server 2>/dev/null; then
    $SUDO apt-get update -y
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y redis-server
    $SUDO systemctl enable redis-server
  fi
  $SUDO systemctl restart redis-server
  $SUDO mkdir -p '$APP_DIR/runtime'
  $SUDO touch '$APP_DIR/runtime/ledger.beancount'
  $SUDO chown -R daymind:daymind '$APP_DIR/runtime'
"

echo "==> Rebuilding virtual environment (torch-free runtime)"
ssh_exec "
  set -euo pipefail
  $SUDO -u daymind bash -lc '
    set -euo pipefail
    cd \"$APP_DIR\"
    rm -rf venv
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip wheel
    ./venv/bin/pip install -r requirements_runtime.txt
    ./venv/bin/pip install -e . --no-deps
    ./venv/bin/python -c \"import src.api.main\"
  '
  $SUDO chown -R daymind:daymind '$APP_DIR'
"

echo "==> Writing /etc/default/daymind"
DAYMIND_API_KEY_VALUE="${DAYMIND_API_KEY:-}"
ssh_exec "
  set -euo pipefail
  DAYMIND_API_KEY_VALUE='${DAYMIND_API_KEY_VALUE}'
  $SUDO tee /etc/default/daymind >/dev/null <<EOF
APP_HOST=127.0.0.1
APP_PORT=8000
FAVA_PORT=8010
REDIS_URL=redis://127.0.0.1:6379
PYTHONPATH=/opt/daymind
DAYMIND_API_KEY=$DAYMIND_API_KEY_VALUE
# OPENAI_API_KEY is read from environment/CI secrets if present.
EOF
  $SUDO chmod 640 /etc/default/daymind
"

echo "==> Installing systemd units"
ssh_exec "
  set -euo pipefail
  $SUDO cp '$APP_DIR/infra/systemd/daymind-api.service' /etc/systemd/system/daymind-api.service
  $SUDO cp '$APP_DIR/infra/systemd/daymind-fava.service' /etc/systemd/system/daymind-fava.service
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable daymind-api.service daymind-fava.service
"

run_remote_verify
echo "==> Deployment summary"
ssh_exec "
  set -euo pipefail
  cd '$APP_DIR'
  COMMIT_SHA=\$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
  echo \"Deployed commit: \$COMMIT_SHA\"
  API_STATUS=\$($SUDO systemctl is-active daymind-api 2>/dev/null || true)
  FAVA_STATUS=\$($SUDO systemctl is-active daymind-fava 2>/dev/null || true)
  echo \"daymind-api: \$API_STATUS\"
  echo \"daymind-fava: \$FAVA_STATUS\"
"
