#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Usage: remote_deploy.sh --host <ip> [--user root] [--key path] [--repo-path /opt/daymind] [--repo-url url]

Deploys the current working tree to the remote host, ensures the git repo exists,
rebuilds a minimal torch-free runtime, seeds env defaults, enforces systemd units,
and verifies /healthz.
USAGE
}

HOST=""
USER="root"
KEY_PATH="${SSH_KEY_PATH:-}"
REMOTE_PATH="/opt/daymind"
REPO_URL="${REPO_URL:-https://github.com/noba-dkg-aion/daymind.git}"

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
      REMOTE_PATH="${2:-}"
      shift 2
      ;;
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
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

if [[ -z "$HOST" ]]; then
  echo "Missing required --host." >&2
  usage
  exit 1
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
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

echo "==> Ensuring daymind system account and repo"
ssh_exec "
  set -euo pipefail
  if ! getent group daymind >/dev/null 2>&1; then
    $SUDO groupadd --system daymind
  fi
  if ! id -u daymind >/dev/null 2>&1; then
    $SUDO useradd --system --create-home --home '$REMOTE_PATH' --shell /bin/bash -g daymind daymind
  fi
  $SUDO mkdir -p '$REMOTE_PATH'
  if [ ! -d '$REMOTE_PATH/.git' ]; then
    $SUDO rm -rf '$REMOTE_PATH'
    $SUDO -u daymind git clone '$REPO_URL' '$REMOTE_PATH'
  fi
  $SUDO -u daymind git -C '$REMOTE_PATH' fetch --all --tags || true
  $SUDO -u daymind git -C '$REMOTE_PATH' checkout main
  $SUDO -u daymind git -C '$REMOTE_PATH' pull --rebase origin main
  $SUDO chown -R daymind:daymind '$REMOTE_PATH'
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
rsync -az --delete "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE:$REMOTE_PATH/"
ssh_exec "$SUDO chown -R daymind:daymind '$REMOTE_PATH'"

echo "==> Ensuring Redis, runtime scaffolding, and ledger"
ssh_exec "
  set -euo pipefail
  if ! $SUDO systemctl is-enabled --quiet redis-server 2>/dev/null; then
    $SUDO apt-get update -y
    $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y redis-server
    $SUDO systemctl enable redis-server
  fi
  $SUDO systemctl restart redis-server
  $SUDO mkdir -p '$REMOTE_PATH/runtime'
  $SUDO touch '$REMOTE_PATH/runtime/ledger.beancount'
  $SUDO chown -R daymind:daymind '$REMOTE_PATH/runtime'
"

echo "==> Rebuilding virtual environment (torch-free runtime)"
ssh_exec "
  set -euo pipefail
  $SUDO -u daymind bash -lc '
    set -euo pipefail
    IFS=$'"'"'\n\t'"'"'
    cd \"$REMOTE_PATH\"
    rm -rf venv
    python3 -m venv venv
    ./venv/bin/pip install --upgrade pip wheel
    ./venv/bin/pip install -r requirements_runtime.txt
    ./venv/bin/pip install -e . --no-deps
    ./venv/bin/python -c \"import src.api.main\"
  '
  $SUDO chown -R daymind:daymind '$REMOTE_PATH'
"

echo "==> Writing /etc/default/daymind"
ssh_exec "
  set -euo pipefail
  $SUDO tee /etc/default/daymind >/dev/null <<'EOF'
APP_HOST=127.0.0.1
APP_PORT=8000
FAVA_PORT=8010
REDIS_URL=redis://127.0.0.1:6379
PYTHONPATH=/opt/daymind
# OPENAI_API_KEY is read from environment/CI secrets if present.
EOF
  $SUDO chmod 640 /etc/default/daymind
"

echo "==> Installing systemd units"
ssh_exec "
  set -euo pipefail
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-api.service' /etc/systemd/system/daymind-api.service
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-fava.service' /etc/systemd/system/daymind-fava.service
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable daymind-api.service daymind-fava.service
"

echo "==> Restarting services and verifying health"
ssh_exec "
  set -euo pipefail
  ENV_FILE=/etc/default/daymind
  if [ -f \"\$ENV_FILE\" ]; then
    # shellcheck disable=SC1091
    source \"\$ENV_FILE\"
  fi
  APP_HOST=\${APP_HOST:-127.0.0.1}
  APP_PORT=\${APP_PORT:-8000}
  API_LOG=/opt/daymind/api.log
  $SUDO systemctl restart daymind-api.service daymind-fava.service || true
  touch /etc/default/daymind
  grep -q "^APP_HOST=" /etc/default/daymind || echo "APP_HOST=127.0.0.1" >> /etc/default/daymind
  grep -q "^APP_PORT=" /etc/default/daymind || echo "APP_PORT=8000" >> /etc/default/daymind
  set -a
  . /etc/default/daymind
  set +a
  echo "🔄 Waiting up to 60s for API on \$\{APP_HOST}:\$\{APP_PORT}"
  for n in $(seq 1 60); do
    if ss -lnt | grep -q ":\$\{APP_PORT} "; then
      echo "✅ API socket ready"
      break
    fi
    sleep 1
  done
  API_HEADER=""
  if [ -n "\$\{DAYMIND_API_KEY:-}" ]; then
    API_HEADER="-H \"X-API-Key: \$\{DAYMIND_API_KEY}\""
  fi
  curl -fsS \$\{API_HEADER} "http://\$\{APP_HOST}:\$\{APP_PORT}/healthz" >/dev/null
  curl -fsS \$\{API_HEADER} "http://\$\{APP_HOST}:\$\{APP_PORT}/metrics" | head -n 3 >/dev/null

"

echo "==> Deployment summary"
ssh_exec "
  set -euo pipefail
  cd '$REMOTE_PATH'
  COMMIT_SHA=\$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
  echo \"Deployed commit: \$COMMIT_SHA\"
  API_STATUS=\$($SUDO systemctl is-active daymind-api 2>/dev/null || true)
  FAVA_STATUS=\$($SUDO systemctl is-active daymind-fava 2>/dev/null || true)
  echo \"daymind-api: \$API_STATUS\"
  echo \"daymind-fava: \$FAVA_STATUS\"
"
