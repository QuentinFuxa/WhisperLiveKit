#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/mobile/daymind"
DIST_DIR="${ROOT_DIR}/dist"

echo "[DayMind] Building debug APK via Buildozer..."
cd "${APP_DIR}"
buildozer -v android debug

latest_apk="$(ls -t "${APP_DIR}/bin"/*.apk | head -n 1)"
mkdir -p "${DIST_DIR}"
cp "${latest_apk}" "${DIST_DIR}/daymind-debug.apk"

echo "[DayMind] APK copied to ${DIST_DIR}/daymind-debug.apk"
