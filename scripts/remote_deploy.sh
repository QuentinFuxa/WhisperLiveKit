#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: remote_deploy.sh --host <ip> [--user root] [--key path] [--repo-path /opt/daymind] [--repo-url url]

Deploys the current working tree to the remote host, ensures the git repo exists,
rsyncs code (excluding build artifacts), seeds the env file if missing, and restarts services.
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
    $SUDO useradd --system --home '$REMOTE_PATH' --shell /usr/sbin/nologin -g daymind daymind
  fi
  $SUDO mkdir -p '$REMOTE_PATH'
  if [ ! -d '$REMOTE_PATH/.git' ]; then
    $SUDO rm -rf '$REMOTE_PATH'
    $SUDO -u daymind git clone '$REPO_URL' '$REMOTE_PATH' || true
  fi
  if [ -d '$REMOTE_PATH/.git' ]; then
    $SUDO -u daymind git -C '$REMOTE_PATH' fetch origin main || true
    $SUDO -u daymind git -C '$REMOTE_PATH' checkout main || true
    $SUDO -u daymind git -C '$REMOTE_PATH' pull --ff-only origin main || true
  fi
  $SUDO chown -R daymind:daymind '$REMOTE_PATH'
"

echo "==> Rsyncing working tree"
RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.venv"
  "--exclude=mobile/android/daymind/app/build"
  "--exclude=.gradle"
  "--exclude=dist"
)
rsync -az --delete "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE:$REMOTE_PATH/"

echo "==> Installing Python dependencies"
VENV_PATH="${REMOTE_PATH}/venv"
ssh_exec "
  set -euo pipefail
  cd '$REMOTE_PATH'
  python3 -m venv '$VENV_PATH'
  '$VENV_PATH/bin/pip' install --upgrade pip wheel
  '$VENV_PATH/bin/pip' install -r requirements.txt
  '$VENV_PATH/bin/pip' install -e .
  chmod +x scripts/start_fava.sh
  $SUDO chown -R daymind:daymind '$REMOTE_PATH' '$VENV_PATH'
"

echo "==> Writing /etc/default/daymind"
ssh_exec "
  set -euo pipefail
  $SUDO tee /etc/default/daymind >/dev/null <<'EOF'
REDIS_URL=redis://127.0.0.1:6379/0
APP_PORT=8000
APP_WORKERS=1
FAVA_PORT=5000
FAVA_LEDGER_PATH=/opt/daymind/ledger/main.beancount
EOF
  $SUDO chmod 640 /etc/default/daymind
"

echo "==> Installing systemd units"
ssh_exec "
  set -euo pipefail
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-api.service' /etc/systemd/system/daymind-api.service
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-fava.service' /etc/systemd/system/daymind-fava.service
"

echo "==> Restarting services"
ssh_exec "
  set -euo pipefail
  APP_PORT=8000
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now daymind-api.service daymind-fava.service
  sleep 5
  $SUDO systemctl is-active daymind-api.service
  $SUDO systemctl is-active daymind-fava.service
  curl -fsS \"http://127.0.0.1:\${APP_PORT}/healthz\" >/dev/null
  curl -fsS \"http://127.0.0.1:\${APP_PORT}/metrics\" >/dev/null || true
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
