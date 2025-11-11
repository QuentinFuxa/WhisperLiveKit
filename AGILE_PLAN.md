# Symbioza-DayMind Agile Plan

## Sprint 1 – MVP Foundation
- **Length:** 2 weeks (now ongoing governance into commercial readiness)
- **Objectives:** Stand up audio → text → structured ledger, resilience, and commercial deployment.
- **Definition of Done:** CI/tests green, docs/operational runbooks updated, planning artifacts synchronized.
- **Status:** ✅ EPIC-1 (STT Core), ✅ EPIC-2 (GPT pipeline), ✅ EPIC-3 (Infra), ✅ EPIC-4 (API bridge), ✅ EPIC-5 (Android client), 🟢 EPIC-6 (Finance track), 🟡 EPIC-11 (server MVP), ✅ EPIC-12 (Commercial readiness), 🟠 EPIC-13 (Android Kotlin integration).
- **Note:** Redis/JSONL, Text-First GPT, Terraform+CI, FastAPI auth/metrics, Kivy mobile, finance exporter, Fava/UI, systemd + TLS, CLI onboarding, and landing site are live or in motion.

### System Architecture
```
[Android App (Kotlin/Compose)] --HTTPS/API key--> [FastAPI Backend (Python)]
    |                                             |-> STT (WhisperLiveKit + VAD)
    |                                             |-> GPT Postproc (session-aware, JSONL ledger)
    |                                             |-> Finance (Beancount export + Fava dashboard)
    |                                             '--> Storage: transcripts.jsonl, ledger.jsonl, ledger.beancount, summaries/metadata
```

-### Milestones
- `v1.0-EPIC-1-STT_CORE` — WhisperLiveKit + VAD streaming foundation
- `v1.1.x` — GPT postproc, session-aware summaries, `safe_json_parse`
- `v1.3-EPIC-3-INFRA` — Terraform droplet + CI/CD automation
- `v1.4-EPIC-4-API` — FastAPI bridge (auth, metrics, ledger/summary)
- `v1.5-EPIC-5-ANDROID` — Kivy mobile companion with offline queue + Buildozer APK
- `v1.6.1` / `v1.6.2` — Finance Beancount export + Fava `/finance` bridge
- `v1.7.0-EPIC-11-MVP_SERVER` — Systemd services, TLS-aware health, LEGAL/NOTICE compliance
- `v1.8-EPIC-12-COMMERCIAL` — full auth, billing, TLS, onboarding site+docs delivered and pytest verified
- *To-tag:* `v1.7.0` release pending final deploy verification for EPIC-11 hardening.
- *Next tag planned:* `v1.9-EPIC-13-ANDROID` on the first Compose build + upload verification.

### Design Principles
- **Text-First Storage** — Every audio capture/transcript/summary/ledger entry persists as human-readable text/JSONL for auditability and inter-agent reuse (gated in EPIC-11). 
- **API-first Modular Mesh** — Interfaces (FastAPI, `/v1/transcribe`, `/v1/finance`, `/v1/usage`, `/welcome`) define contracts; clients (Mobile Kivy, Kotlin) consume them without sharing binaries.

### Kanban Snapshot

| Backlog | Next | In Progress | Done |
|---------|------|-------------|------|
| EPIC-10 – LangGraph DAG + Redis streams<br>EPIC-8 – Automations & notifications backlog<br>EPIC-9 – Release management automation | EPIC-13 – US-13.2 Summary + Finance views<br>EPIC-13 – US-13.3 Settings + diagnostics | EPIC-13 – US-13.1 Recording + chunk uploader (Kotlin client)<br>EPIC-13 – full Android Kotlin client integration | EPIC-1 .. EPIC-6 stories<br>EPIC-11 US-11.1..US-11.4<br>EPIC-12 US-12.1..US-12.3<br>EPIC-13 – US-13.CI Terraform apply + health gates<br>EPIC-13 – US-13.4 Android CI build pipeline |

### Epic Tracker

#### EPIC-1 — Real-Time STT Core (✅ Complete — tag `v1.0-EPIC-1-STT_CORE`)
- **US-1.1** WhisperLiveKit + VAD integration.
- **US-1.2** Transcript streaming to Redis + JSONL buffer.
- **US-1.3** STT e2e tests (noise sample, pytest assets).
> **Acceptance Gates:** Runner prints live transcripts; Redis + buffer verified in CI.

#### EPIC-2 — GPT-4o-mini Post-Processing (✅ Complete — tag `v1.1-EPIC-2-GPT_POSTPROC`)
- **US-2.1** GPT API pipeline.
- **US-2.2** Session-aware prompts + JSON extraction.
- **US-2.3** Daily summaries + safe JSON parser.
> **Acceptance Gates:** `data/ledger.jsonl` grows per transcript; `summary_<date>.md` + `ledger_<date>.jsonl` produced even when GPT output is malformed.

#### EPIC-3 — CI/CD & Deployment (✅ Complete — tag `v1.3-EPIC-3-INFRA`)
- **US-3.1** Docker/CI scaffolding.
- **US-3.2** Terraform droplet + Redis.
- **US-3.3** Auto summary/deploy workflow.
> **Acceptance Gates:** `ci_cd.yml` runs pytest + Terraform; infra outputs (IP, Redis URI) documented.

#### EPIC-4 — API Bridge (✅ Complete — tag `v1.4-EPIC-4-API`)
- **US-4.1** FastAPI skeleton + auth middleware.
- **US-4.2** `/v1/transcribe` + `/v1/ingest-transcript`.
- **US-4.3** `/v1/ledger` + `/v1/summary`.
- **US-4.4** `/healthz` + `/metrics` observability.
> **Acceptance Gates:** All routes require `X-API-Key`; Prometheus counters scraped in CI smoke tests.

#### EPIC-5 — Android Companion (✅ Complete — tag `v1.5-EPIC-5-ANDROID`)
- **US-5.1** Recorder + chunk queue.
- **US-5.2** Summary tab + offline persistence.
- **US-5.3** Settings/log view + Buildozer path.
> **Acceptance Gates:** Buildozer debug APK (`scripts/build_apk.sh`) verified; emulator run confirms queue persistence, log view, “Clear queue,” and summary refresh.

#### EPIC-6 — Finance / Ledger Analytics (🟢 Rolling Delivery)
- **US-6.1 – JSONL→Beancount exporter** — ✅ CLI + helper script generate `finance/ledger.beancount` deterministically from `data/ledger*.jsonl`.
- **US-6.2 – Fava dashboard service + `/v1/finance`** — ✅ `python -m src.finance.fava_runner` runs externally on port 5000, FastAPI redirect + summary endpoint serve category/date totals (tests in `tests/test_finance_api.py`).
- **US-6.3 – Finance aggregates API enhancements** — ⏭ (Next) add richer filters, pagination, CSV/endpoint parity; success = `/v1/finance` powers dashboards + automation jobs.
> **Acceptance Gates:** Daily exporter cron, `/finance` redirect secured, `/v1/finance` validated against golden ledger snapshots, CSV/JSON outputs ready for BI.

#### EPIC-7 — Long-Term Memory / Anki (⏳ Backlog)
- **US-7.1 – Deck builder from “remember” intents** — Next (daily `.apkg` named `Memory::DayMind::<YYYY-MM-DD>`).
- **US-7.2 – CI artifact export** — Backlog (scheduled workflow uploads `.apkg`).
- **US-7.3 – Schema & QA guard** — Backlog (front/back templates validated via genanki + sample import).
> **Acceptance Gates:** Deck artifacts import on AnkiDesktop/AnkiDroid, CI attaches them, schema tests prevent malformed cards.

#### EPIC-8 — Automation & Daily Report (⏳ Backlog)
- **US-8.1 – Daily cron workflow** — Next (GH Actions schedule runs exporter + summarizer + ledger refresh; artifacts attached).
- **US-8.2 – Notifications via Apprise** — Backlog (Telegram/email payload linking summary + CSV).
- **US-8.3 – Health/report metrics snapshot** — Backlog (daily status JSON with request counts/errors).
> **Acceptance Gates:** Cron finishes within 15 min, notifications delivered with success/failure flag, artifacts stored for 30 days.

#### EPIC-9 — Release Management (⏳ Backlog)
- **US-9.1 – Configure Release Please** — Backlog (conventional commits → auto PR + changelog).
- **US-9.2 – EPIC tag integration** — Backlog (tags `v1.6-EPIC-6-FINANCE` etc. produce per-epic sections).
> **Acceptance Gates:** Release PR opens on every merge, changelog groups entries by epic, tags pushed automatically on approval.

#### EPIC-10 — LangGraph Orchestration (⏳ Backlog)
- **US-10.1 – DAG skeleton** — Backlog (nodes for STT, GPT, Finance, Memory, Reporter with mocks/tests).
- **US-10.2 – Redis Streams wiring** — Backlog (XADD/XREADGROUP flows with metrics + replay instructions).
- **US-10.3 – Runbook & contracts** — Backlog (document node APIs, retry/backoff, error semantics).
> **Acceptance Gates:** LangGraph demo runs end-to-end with mocks, Redis streams monitored, runbook published with node contracts.

#### EPIC-11 — Serverized MVP Release (🟡 In Progress — target `v1.7.0-EPIC-11-MVP_SERVER`)
- **US-11.1 – Legal compliance & external services** — ✅ Fava/Beancount run strictly as HTTP/file-based externals; MIT LICENSE + NOTICE.md committed; README documents compliance + Text-First rule.
- **US-11.2 – Systemd services & env templates** — ✅ `daymind-api` + `daymind-fava` units share `/etc/daymind/daymind.env`; firewall guidance + reverse-proxy configs documented.
- **US-11.3 – CI/CD deploy job & docs** — ✅ GitHub Actions `deploy_app` rsyncs code, installs deps, restarts services; `DEPLOY.md` + `API_REFERENCE.md` refreshed.
- **US-11.4 – Hardening & release tag** — 🚧 TLS field added to `/healthz`, Prometheus counters expanded, release tag `v1.7.0` pending final smoke tests.
> **Acceptance Gates:** Droplet reboot → services active; `/healthz` fails if disk/redis/openai/tls broken; CI deploy job green; transcription always writes `data/transcript_<day>.jsonl` before GPT/Finance stages; `v1.7.0-EPIC-11-MVP_SERVER` tag cut with verified health + metrics.

#### EPIC-12 — Commercial Readiness (✅ Done — tag `v1.8-EPIC-12-COMMERCIAL`)
- **US-12.1 – Auth & Billing layer** — ✅ API key metadata store + `/v1/usage`, CLI for create/revoke, Stripe/Paddle stubs, docs in `BILLING.md`.
- **US-12.2 – Security hardening** — ✅ IP + key rate limiting middleware, TLS proxy configs (Caddy/Nginx), new `/healthz` fields, `SECURITY.md`, `DEPLOY.md` HTTPS guidance.
- **US-12.3 – Landing + onboarding** — ✅ `landing/` static site deployed via GitHub Pages, `/welcome` onboarding endpoint, `ONBOARDING.md`, `scripts/setup_daymind.sh` helper.
> **Acceptance Gates:** `/v1/usage` reports live stats, HTTPS enforced via reverse proxy (`infra/caddy/Caddyfile`), firewall docs updated, landing site auto publishes from CI, onboarding doc validated by new operator walkthrough, `v1.5+` customers can self-serve API keys + billing data.
> **Success:** `v1.8-EPIC-12-COMMERCIAL` release delivered with auth/billing, TLS, onboarding docs, and pytest verification.

#### EPIC-13 — Android Kotlin Client (🟠 In Progress)
- **US-13.1 – Recording + Chunk Uploader** — 🛠 In Progress (Jetpack Compose MVP screen with record toggle/pending queue counter, notification-backed ForegroundService writing mono 16 kHz ~30 s WAV chunks, Retrofit + WorkManager uploader w/ exponential backoff and `X-API-Key`, EncryptedSharedPreferences/`local.properties` config fallbacks).
- **US-13.2 – Summary & Finance Views** — Next (summary fetch, category totals chart, offline/refresh states using Compose/MPAndroidChart). 
- **US-13.3 – Settings & Key Storage** — Next (EncryptedSharedPreferences server URL + API key, diagnostics ping `/healthz`, persistent config survives restarts).
- **US-13.CI – Terraform apply + remote deploy health gates** — ✅ `terraform_apply` now runs pytest + Terraform against DO via secrets, `deploy_app` syncs `/opt/daymind` through `scripts/remote_deploy.sh`, and `verify_services` enforces Redis/API/Fava health plus `/healthz` + `/metrics`.
- **US-13.4 – Android CI build pipeline** — ✅ `.github/workflows/android_build.yml` compiles Compose debug + release variants on GitHub-hosted or self-hosted runners, publishes unsigned + optionally signed APKs, and uploads tag artifacts to GitHub Releases.
- **Note:** LiveKit STT is optional for CI; production runners install it via the `stt_livekit` extra or bespoke wheels before starting realtime capture.
- **US-13.CD – Service ops note:** `/etc/default/daymind` now carries deploy-time env (APP_PORT/FAVA_PORT/etc.), systemd units run `daymind-api` + `daymind-fava`, and CI verify jobs curl `/healthz` + `/metrics` after checking `systemctl is-active`.
> **US-13.1 Acceptance Gates:** Record toggle spins a notification-backed foreground service, chunk WAV files appear in cache, WorkManager auto-uploads queued chunks with metadata (`session_ts`, `device_id`, `sample_rate`, `format`), UI surfaces pending chunk count + last upload result and exposes “Retry uploads” for auth lockouts, HTTP 401/403 pauses the queue without deleting files, `/v1/transcribe` requests include `X-API-Key` + multipart `file=@chunk.wav`, CI builds/publishes `app-debug.apk`, and no real secrets are committed (keys supplied via `local.properties` or EncryptedSharedPreferences).
> **US-13.CI Acceptance Gates:** `terraform_apply` must init/validate/apply Terraform via repo secrets while uploading `terraform.tfstate`, `deploy_app` must invoke `scripts/remote_deploy.sh` for an idempotent `/opt/daymind` sync + service restart, and `verify_services` must run the Redis/API/Fava healthcheck script plus `/healthz` + `/metrics` curls before succeeding.
> **US-13.4 Acceptance Gates:** `android_build.yml` exposes `workflow_dispatch` inputs (`build_type`, `runner`, `ref`), `build_gh` (ubuntu-latest) and `build_self` (self-hosted) jobs assemble debug/release variants per input, artifacts (`daymind-android-*`) upload for every run, optional signing secrets emit `app-release-signed.apk`, and tag pushes attach all generated APKs to the corresponding GitHub Release.
> **Epic Acceptance Gates:** Kotlin app uploads chunks with exponential retry, summary and finance tabs render real data with errors handled, stored credentials survive reboots, health indicator toggles based on `/healthz` response, and the CI pipeline (`terraform_apply → deploy_app → verify_services`) must pass to certify infra, deploy, and healthcheck steps.

#### EPIC-14 — Deployable Services (🟢 Active)
- **US-14.1 – Systemd units + env** — ✅ Added `infra/systemd/daymind-api.service`, `infra/systemd/daymind-fava.service`, `/etc/default/daymind` wiring, and `scripts/start_fava.sh` so API + Fava boot as first-class services.
- **US-14.2 – Remote deploy hardening** — ✅ `scripts/remote_deploy.sh` now ensures the daymind user, rsyncs `/opt/daymind`, rebuilds `/opt/daymind/venv`, rewrites `/etc/default/daymind`, installs units, and restarts services with curl-based smoke tests.
- **US-14.3 – CI guardrails** — ✅ `infra/systemd/checks/healthcheck.sh` curls `/healthz` + `/metrics` after status checks and `tests/test_systemd_units_exist.py` proves the ExecStart anchors stay intact.
> **Acceptance Gates:** Both services stay `active` after deploy, `/healthz` returns 200 directly on the droplet, `/opt/daymind/ledger/main.beancount` bootstraps automatically, and CI verify jobs pass with no manual intervention.

### Operational Gates
- `/healthz` must expose `redis`, `disk`, `openai`, and `tls`; fail-fast when critical checks fail and return JSON for monitoring.
- Fava runs strictly as an external HTTP service; FastAPI only redirects/proxies (`/finance`, `FAVA_BASE_URL`) and never embeds GPL binaries.
- Text-First Storage invariant enforced: every pipeline writes transcripts/summaries/ledger entries to JSONL/MD before downstream processing or UI surfaces data.
- CI can be manually invoked via workflow_dispatch.
