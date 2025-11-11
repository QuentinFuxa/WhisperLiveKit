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

echo "==> Ensuring remote repository at ${REMOTE_PATH}"
ssh_exec "
  set -euo pipefail
  $SUDO mkdir -p '$REMOTE_PATH'
  if [ ! -d '$REMOTE_PATH/.git' ]; then
    $SUDO rm -rf '$REMOTE_PATH'
    $SUDO git clone '$REPO_URL' '$REMOTE_PATH'
  fi
  if [ \"$USER\" != \"root\" ]; then
    $SUDO chown -R '$USER':'$USER' '$REMOTE_PATH'
  fi
  cd '$REMOTE_PATH'
  git fetch origin main || true
  git checkout main || true
  git pull --ff-only origin main || true
"

echo "==> Ensuring daymind system account"
ssh_exec "
  set -euo pipefail
  if ! getent group daymind >/dev/null 2>&1; then
    $SUDO groupadd --system daymind
  fi
  if ! id -u daymind >/dev/null 2>&1; then
    $SUDO useradd --system --home '$REMOTE_PATH' --shell /usr/sbin/nologin -g daymind daymind
  fi
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

echo "==> Installing Python dependencies and systemd units"
ssh_exec "
  set -euo pipefail
  cd '$REMOTE_PATH'
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install -e .
  $SUDO cp infra/systemd/daymind-api.service /etc/systemd/system/daymind-api.service
  $SUDO cp infra/systemd/daymind-fava.service /etc/systemd/system/daymind-fava.service
  $SUDO chown -R daymind:daymind '$REMOTE_PATH'
"

echo "==> Seeding env file (if missing)"
ssh_exec "
  set -euo pipefail
  $SUDO mkdir -p /etc/daymind
  if [ ! -f /etc/daymind/daymind.env ]; then
    $SUDO cp '$REMOTE_PATH/infra/systemd/daymind.env.example' /etc/daymind/daymind.env
    $SUDO chown root:daymind /etc/daymind/daymind.env 2>/dev/null || true
    $SUDO chmod 640 /etc/daymind/daymind.env
  fi
"

echo "==> Restarting services"
ssh_exec "
  set -euo pipefail
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable daymind-api daymind-fava >/dev/null 2>&1 || true
  $SUDO systemctl restart daymind-api daymind-fava
"

echo "==> Deployment summary"
ssh_exec "
  set -euo pipefail
  cd '$REMOTE_PATH'
  COMMIT_SHA=\$(git rev-parse HEAD 2>/dev/null || echo 'unknown')
  echo \"Deployed commit: \$COMMIT_SHA\"
  API_STATUS=\$($SUDO systemctl is-active daymind-api || true)
  FAVA_STATUS=\$($SUDO systemctl is-active daymind-fava || true)
  echo \"daymind-api: \$API_STATUS\"
  echo \"daymind-fava: \$FAVA_STATUS\"
"
