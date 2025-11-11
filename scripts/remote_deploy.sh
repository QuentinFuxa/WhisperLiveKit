#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

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
  $SUDO mkdir -p '$REMOTE_PATH' '$REMOTE_PATH/runtime'
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
  "--exclude=runtime"
  "--exclude=mobile/android/daymind/app/build"
  "--exclude=.gradle"
  "--exclude=dist"
)
rsync -az --delete "${RSYNC_EXCLUDES[@]}" -e "ssh ${SSH_OPTS[*]}" ./ "$REMOTE:$REMOTE_PATH/"
ssh_exec "$SUDO chown -R daymind:daymind '$REMOTE_PATH'"

echo "==> Rebuilding virtual environment"
ssh_exec "
  set -euo pipefail
  $SUDO -u daymind bash -lc '
    set -euo pipefail
    IFS=$'"'"'\n\t'"'"'
    cd \"$REMOTE_PATH\"
    if [ ! -d venv ]; then
      python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -U pip wheel
    pip install -r requirements.txt
    pip install -e .
    python -c \"import src, src.api.main\"
  '
  for bin in uvicorn fava; do
    if [ ! -x \"$REMOTE_PATH\"/venv/bin/\"$bin\" ]; then
      echo \"::error::missing $bin in venv\" >&2
      exit 1
    fi
  done
  $SUDO chown -R daymind:daymind '$REMOTE_PATH'
"

echo "==> Writing /etc/default/daymind"
ssh_exec "
  set -euo pipefail
  ENV_FILE=/etc/default/daymind
  if [ ! -f \"\$ENV_FILE\" ]; then
    $SUDO tee \"\$ENV_FILE\" >/dev/null <<'EOF'
APP_ENV=production
APP_HOST=127.0.0.1
APP_PORT=8000
APP_WORKERS=1
REDIS_URL=redis://127.0.0.1:6379
REDIS_URI=redis://127.0.0.1:6379/0
PYTHONPATH=/opt/daymind
FAVA_HOST=127.0.0.1
FAVA_PORT=8010
LEDGER_FILE=/opt/daymind/runtime/ledger.beancount
EOF
    $SUDO chmod 640 \"\$ENV_FILE\"
  fi
  LEDGER_FILE_PATH=\$($SUDO awk -F'=' '/^LEDGER_FILE=/{print \$2}' \"\$ENV_FILE\" | tail -n1)
  if [[ -z \"\$LEDGER_FILE_PATH\" ]]; then
    LEDGER_FILE_PATH=/opt/daymind/runtime/ledger.beancount
  fi
  $SUDO mkdir -p \"\$(dirname \"\$LEDGER_FILE_PATH\")\"
  if [ ! -s \"\$LEDGER_FILE_PATH\" ]; then
    $SUDO tee \"\$LEDGER_FILE_PATH\" >/dev/null <<'LEDGER'
option \"title\" \"DayMind Ledger\"
option \"operating_currency\" \"USD\"

1970-01-01 * \"Bootstrap\" \"Ledger initialized\"
  equity:opening-balances  0 USD
LEDGER
  fi
  $SUDO chown daymind:daymind \"\$LEDGER_FILE_PATH\"
"

echo "==> Installing systemd units"
ssh_exec "
  set -euo pipefail
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-api.service' /etc/systemd/system/daymind-api.service
  $SUDO cp '$REMOTE_PATH/infra/systemd/daymind-fava.service' /etc/systemd/system/daymind-fava.service
  $SUDO chown daymind:daymind '$REMOTE_PATH/scripts/start_fava.sh'
"

echo "==> Restarting services"
ssh_exec "
  set -euo pipefail
  ENV_FILE=/etc/default/daymind
  if [ -f \"\$ENV_FILE\" ]; then
    # shellcheck disable=SC1090
    source \"\$ENV_FILE\"
  fi
  APP_PORT=\"\${APP_PORT:-8000}\"
  $SUDO systemctl daemon-reload
  if ! $SUDO systemctl enable --now daymind-api daymind-fava; then
    for svc in daymind-api daymind-fava; do
      $SUDO journalctl -u \"\$svc\" -n 120 --no-pager || true
    done
    exit 1
  fi
  sleep 5
  for svc in daymind-api daymind-fava; do
    if ! $SUDO systemctl is-active --quiet \"\$svc\"; then
      echo \"::error::\$svc failed to start\" >&2
      $SUDO journalctl -u \"\$svc\" -n 120 --no-pager || true
      exit 1
    fi
  done
  if ! curl -fsS \"http://127.0.0.1:\${APP_PORT}/healthz\" >/dev/null; then
    echo \"::error::/healthz failed\" >&2
    for svc in daymind-api daymind-fava; do
      $SUDO journalctl -u \"\$svc\" -n 120 --no-pager || true
    done
    exit 1
  fi
  if curl -fsS \"http://127.0.0.1:\${APP_PORT}/metrics\" >/dev/null; then
    echo \"✅ /metrics responded\"
  else
    echo \"⚠️  /metrics failed\" >&2
  fi
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
