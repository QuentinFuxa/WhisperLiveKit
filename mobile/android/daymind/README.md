# DayMind Android Client (EPIC-13 / US-13.1)

Foreground-only Kotlin/Compose app that records mono 16 kHz WAV chunks, queues them locally, and lets WorkManager upload each file to `/v1/transcribe` with the `X-API-Key` header. All AI and finance logic remains on the server; the client only captures and transports audio.

## Local setup
1. Copy `local.properties.sample` to `local.properties` and add your Android SDK path plus DayMind endpoint secrets:

   ```properties
   sdk.dir=/Users/you/Library/Android/sdk
   BASE_URL=https://api.your-daymind-host
   API_KEY=dev-demo-key
   ```

2. Optional (production override): populate `EncryptedSharedPreferences` via a secured settings screen or adb shell so `SERVER_URL` / `API_KEY` keys take precedence over `BuildConfig` fallbacks.

3. Build the APK:

   ```bash
   cd mobile/android/daymind
   ./gradlew assembleDebug
   ```

4. Install on a device or emulator:

   ```bash
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

## Runtime behavior
- The `Record` toggle starts a foreground `AudioRecord` service (PCM 16‑bit, 16 kHz mono) that writes 30 s WAV chunks into `cacheDir/chunks`.
- Each finalized chunk schedules an `UploadChunkWorker` job with network constraints and exponential backoff.
- Successful uploads delete the chunk and clear any pause flags. Auth failures (401/403) pause the queue until the operator taps **Retry uploads** and fixes the key.
- Metadata sent along with the multipart payload: `session_ts`, `device_id`, `sample_rate`, and `format`.

## Privacy notes
- Audio chunks stay on-device (cache directory) until they upload over HTTPS.
- No AI inference, ledgers, or summaries run locally; the FastAPI backend continues to own all Text-First artifacts.
- Delete the app data to purge cached chunks if needed.

## CI
`.github/workflows/android.yml` assembles `app-debug.apk` on every push/PR touching `mobile/android/daymind/**` and publishes the artifact at `mobile/android/daymind/app/build/outputs/apk/debug/app-debug.apk`.
