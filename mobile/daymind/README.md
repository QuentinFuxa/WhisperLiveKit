# DayMind Android Client

The DayMind mobile app is a lightweight Kivy client that records short audio chunks, queues them offline, and streams them to the `/v1/transcribe` API while keeping summaries and logs at your fingertips.

## Feature Highlights
- **Start / Stop toggle** with visible recording state (`● Recording` vs `◼ Idle`).
- **Chunked capture**: 6 s WAV segments (16 kHz mono) saved in app-private storage.
- **Offline-first queue**: `ChunkQueue` durably persists pending uploads and retries with exponential backoff until the server acknowledges them.
- **Settings screen**: Stores Server URL + API Key, supports “Test connection” (calls `/healthz`), and hides the key input.
- **Summary screen**: Pulls `/v1/summary?date=<today>` off the UI thread, shows friendly errors, and allows manual refresh.
- **Log window + queue counter**: Recent events (recording, uploads, retries) are always visible; “Clear queue” removes pending files plus local copies.

## Configuration
1. Launch the app (desktop preview: `python -m mobile.daymind.main`).
2. Open **Settings** → enter `Server URL` (e.g., `https://api.daymind.dev`) and `API Key`.  
   > Every backend call includes `X-API-Key`; requests are rejected until these fields are filled.
3. Tap **Save**, then **Test Connection**. Success returns “Connection OK” in the log panel; failures log the error (missing key, 401, timeouts, etc.).

## Recording & Privacy
- Press **Start Recording**; the button text + indicator switch to “Stop Recording” and `● Recording`.  
- The recorder writes 6 s chunks to `<app data>/chunks/` and immediately enqueues them.
- **Stop Recording** halts capture; background uploads still proceed.
- Use **Clear Queue** at any time to purge pending files (local WAVs are deleted before removal).

## Summary Screen
- Tap **Summary** → **Refresh** to fetch the latest markdown summary for today (UTC).  
- Network work happens on a worker thread; the UI stays responsive.  
- Errors (404, timeout, auth) surface in both the summary view (“Error: …”) and the log window so the operator knows what failed.

## Offline Queue Behavior
- When the API is unreachable, uploads fail with exponential backoff (2 s → 4 s … max 60 s).  
- Pending chunks stay on disk (`chunk_queue.json`) across restarts.  
- Once connectivity returns, the `UploadWorker` drains the queue automatically.  
- Inspect queue length + log lines on the Record screen; use “Clear queue” to remove everything.

## Minimal Observability
- The in-app log retains the last 200 events (`mobile/daymind/services/logger.py`).
- Messages include timestamps for debugging uploads, retries, summaries, and settings actions.

## Building a Debug APK (Buildozer)

### Prerequisites
- Python 3.10+ on Linux (WSL/macOS require an Ubuntu container).
- Android SDK/NDK managed by Buildozer (it will download them on the first build).
- System packages: `build-essential`, `git`, `zip`, `unzip`, `openjdk-17-jdk`, `python3-venv`.

### Manual Build
If this is your first Buildozer run on a host, execute `scripts/setup_buildozer_host.sh`
from the repo root (works locally or on a Terraform-provisioned droplet). It installs
OpenJDK 17, the Android SDK/NDK helper stack, and a dedicated virtualenv so the build
command below succeeds.

```bash
cd mobile/daymind
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip buildozer
buildozer -v android debug
```

The APK appears under `mobile/daymind/bin/`. Install via `adb install bin/daymind-<version>-debug.apk`.

### Helper Script
From the repo root:
```bash
scripts/build_apk.sh
```

This runs Buildozer and copies the newest artifact to `dist/daymind-debug.apk` for easy sharing.

### Terraform Builder
Need more CPU/RAM for Buildozer? Reuse the DigitalOcean droplet defined under
`infra/terraform`. See [`docs/buildozer_server.md`](../../docs/buildozer_server.md)
for the full workflow (Terraform apply → rsync repo → `scripts/setup_buildozer_host.sh`
→ `scripts/build_apk.sh`).

### Required Permissions
Declared in `mobile/daymind/buildozer.spec`:
- `RECORD_AUDIO` – microphone access.
- `INTERNET` + `ACCESS_NETWORK_STATE` – API calls + connectivity checks.
- `WAKE_LOCK` – keeps CPU alive while recording/uploading.
- `FOREGROUND_SERVICE` – allows future background service hooks.

## Testing Checklist
1. **Settings**: Empty fields → “Test connection” fails; fill values → expect HTTP 200.  
2. **Recording**: Run for ~20 s; observe ≥2 queued chunks + uploads draining when online.  
3. **Offline queue**: Disable network, capture audio, confirm retries + persistence; re-enable network and verify automatic draining.  
4. **Summary**: Refresh to fetch today’s summary; errors display gracefully.  
5. **Persistence**: Restart app; settings + queue remain; recording starts OFF.  
6. **Stability**: Long recording (10 min) keeps memory bounded; log and queue stay coherent even after force-closing the app.

## Troubleshooting
- **401 Unauthorized** → verify API key; log shows “Unauthorized”.
- **Summary 404** → backend has not published a summary for that date yet.
- **Build failures** → remove `.buildozer/` + `bin/`, rerun `buildozer -v android debug`.
- **No audio on desktop** → install `sounddevice`; on Android, ensure microphone permission is granted in Settings.

Once the APK is installed, open **Settings** first, set credentials, then start recording. All uploads reference the backend you configure—no secrets are baked into the binary.
