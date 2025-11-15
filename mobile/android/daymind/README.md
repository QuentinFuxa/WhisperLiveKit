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
- The `Record` toggle starts a foreground `AudioRecord` service (PCM 16‑bit, 16 kHz mono) that writes 6 s WAV chunks into `cacheDir/chunks`.
- Each finalized chunk schedules an `UploadChunkWorker` job with network constraints and exponential backoff.
- Successful uploads delete the chunk and clear any pause flags. Auth failures (401/403) pause the queue until the operator taps **Retry uploads** and fixes the key.
- Metadata sent along with the multipart payload: `session_ts`, `device_id`, `sample_rate`, and `format`.

## Privacy notes
- Audio chunks stay on-device (cache directory) until they upload over HTTPS.
- No AI inference, ledgers, or summaries run locally; the FastAPI backend continues to own all Text-First artifacts.
- Delete the app data to purge cached chunks if needed.

## Build requirements
- JDK 17 (Temurin/Azul/OpenJDK)
- Android SDK Platform 34 + Build Tools 34.0.0 (install via `sdkmanager "platforms;android-34" "build-tools;34.0.0" "platform-tools"`)
- Gradle wrapper bundled in this repo (`./gradlew`)

Local builds stay deterministic with:
```bash
./gradlew assembleDebug               # debug (default local target)
./gradlew assembleRelease             # release (unsigned unless signing props configured)
```
Copy `gradle.properties.template` to `gradle.properties` (or export `ORG_GRADLE_PROJECT_*` vars) when providing signing credentials locally.

## CI
`.github/workflows/android_build.yml` assembles debug/release APKs on pushes to `main`, pull requests, tags, and manual dispatches. Manual triggers stay CLI-first:
```bash
gh workflow run android_build.yml -f build_type=debug -f runner=gh --ref main
gh workflow run android_build.yml -f build_type=release -f runner=gh --ref main
gh workflow run android_build.yml -f build_type=both -f runner=self -f ref=feature/android-ci
```
Artifacts land as `daymind-android-*` on each run; tag builds also attach the APKs to the GitHub Release. Set the optional signing secrets (`ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_ALIAS_PASSWORD`) or matching `gradle.properties` entries to emit `app-release-signed.apk` in addition to the default debug + unsigned release packages.

### UI — True Black + Logo
- Backgrounds and surfaces default to `#000000` (true black) for legacy and Android 12+ splash flows.
- Primary blue (`#375DFB`) remains the accent color for buttons and interactive elements.
- Splash/icon art lives in `app/src/main/res/drawable/daymind_logo.xml` (vector) with the source SVG mirrored under `mobile/android/daymind/art/`.
