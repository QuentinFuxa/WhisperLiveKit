#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/mobile/daymind"
VENV_DIR="${APP_DIR}/.venv-buildozer"

run() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

echo "[DayMind] Installing Buildozer host prerequisites..."
run apt-get update
run apt-get install -y \
  build-essential \
  git \
  zip \
  unzip \
  openjdk-17-jdk \
  python3 \
  python3-pip \
  python3-venv \
  libffi-dev \
  libssl-dev \
  libbz2-dev \
  libsqlite3-dev \
  zlib1g-dev \
  libncurses5-dev \
  libtinfo6

echo "[DayMind] Creating Buildozer virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip cython==0.29.36 buildozer

cat <<'MSG'
[DayMind] Buildozer host ready.
To build the APK on this machine run:

  source mobile/daymind/.venv-buildozer/bin/activate
  cd mobile/daymind
  buildozer -v android debug

Or simply execute `scripts/build_apk.sh` from the repo root
after activating the virtualenv above.
MSG
