# DayMind Android Alpha Test Plan

## Scope
Verify that the Android Kotlin client records, chunks, uploads, and communicates correctly with the FastAPI backend over HTTPS.

### 1. Environment Setup
- Confirm server is running (e.g., `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`).
- Ensure `/v1/transcribe` responds over HTTPS (`curl https://$BASE_URL/v1/transcribe -H "X-API-Key: <key>"`).
- Confirm the valid `X-API-Key` is configured on the device (via `local.properties` or EncryptedSharedPreferences) and matches `.env`.
- Verify Redis, OpenAI, and TLS health checks are green (`/healthz` returns `true` for redis/disk/openai/tls).

### 2. Build & Install
```bash
cd mobile/android/daymind
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```
Launch the DayMind Alpha app on the device.

### 3. Functional Tests
| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Tap **Record** | Foreground notification “DayMind is recording…” appears and service stays alive. |
| 2 | Wait ~6 s | A WAV chunk appears under `cacheDir/chunks/` (verify via `adb shell ls /storage/.../cache/chunks`). |
| 3 | Tap **Play Last Chunk** | Trimmed clip plays locally (button label switches to **Stop Playback**) and stops automatically after playback. |
| 4 | Ensure network is ON | Chunk upload job runs automatically; status message becomes “Uploaded ...”. |
| 5 | Toggle network OFF | Chunk file remains; queue shows “Waiting for network”/upload status persists. |
| 6 | Restore network | WorkManager retries and uploads chunk automatically; file deleted after 200 response. |
| 7 | Set invalid API key | `/v1/transcribe` returns 401/403, upload status shows auth error, chunk remains queued. |
| 8 | Tap **Stop Recording** | Foreground service stops without crash; no new chunks created afterward. |
| 9 | Re-open app | Pending chunks automatically enqueued (UploadStatus shows resumed/Retry). |
|10 | Check server logs | `/v1/transcribe` receives multipart form data with `file=@chunk.wav`, plus `session_ts`, `device_id`, `sample_rate`, `format`. |
|11 | Inspect backend | `data/ledger.jsonl` gains a new transcript line timestamped from the chunk session. |
|12 | Inspect metadata | Optional `speech_segments` payload arrives with `{start_ms,end_ms}` windows matching the periods where speech was detected (verify by pausing/muting during parts of the recording). |

### 4. Reliability & Edge Cases
- Repeat Record/Stop sequence 5×; verify no crashes, no duplicate chunks, and UI remains responsive.
- Simulate airplane mode mid-recording; after reconnect, WorkManager retries until upload succeeds.
- Record until battery drops below 20 % (if possible); ensure recording stops gracefully and WAV files remain valid.
- Rotate the screen while recording; observe the service continues, chunks remain consistent, and UI state persists.
- Revoke microphone permission while recording; app surfaces error and stops cleanly.

### 5. Security
- Confirm all requests target `https://` endpoints (Caddy/TLS) and not plain HTTP.
- Inspect logs/adb logcat for absence of plaintext API keys or sensitive metadata.
- Validate that EncryptedSharedPreferences (or `local.properties`) hides the API key and that `BuildConfig` values can be overridden via secure storage.

### 6. Performance
- Measure average upload latency per chunk on Wi-Fi (target < 2 s from request to 200 response). If possible, capture timestamps in logs.
- Monitor CPU usage; ensure background recording stays < 15 % and battery drain remains reasonable (< 5 %/h). Use `adb shell top`/`dumpsys batterystats` as needed.
- Confirm WorkManager retries up to 5× with exponential backoff (observe `logcat` or worker metadata after network flaps).

### 7. Post-Test Validation
- Ensure `data/ledger.jsonl` entries match every recorded chunk (timestamps line up with sessions).
- Query `/v1/summary`; the daily summary reflects the new transcript data.
- Call `/v1/finance` and `/v1/usage` to confirm endpoints stay healthy after uploads.

### 8. Reporting
- Mark each checkbox with ✅/❌ and note timestamps in `logs/test_android_alpha.jsonl` (fields: timestamp, scenario, expected, actual). Keep the file portable.
- For any failure, capture `adb logcat` snippets and server logs to support triage.
- Final pass ⇒ tag release `v1.9.0-EPIC-13-ANDROID-ALPHA` once upload behavior is stable.
