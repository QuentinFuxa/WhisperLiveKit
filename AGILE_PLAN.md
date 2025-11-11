# Symbioza-DayMind Agile Plan

## Sprint 1 – MVP Foundation
- **Length:** 2 weeks
- **Objectives:** Stand up an end-to-end audio → text → structured ledger flow, basic CI/CD, and an Android capture client.
- **Definition of Done:** Tested components, documented interfaces, and updated Kanban status with links to commits.
- **Status:** ✅ EPIC-1 (STT Core), ✅ EPIC-2 (GPT pipeline), ✅ EPIC-3 (Infra), ✅ EPIC-4 (API bridge), ✅ EPIC-5 (Android client) — released through tags `v1.0`, `v1.1`, `v1.3`, `v1.4`, and `v1.5-EPIC-5-ANDROID`.
- **Note:** Continuous transcript sinks (Redis + JSONL), session-aware GPT processing with daily summaries + robustness patch, automated Terraform + CI/CD, FastAPI bridge, and the Android client are all live.

### Milestones
- `v1.0-EPIC-1-STT_CORE` — WhisperLiveKit integration + STT loop
- `v1.1-EPIC-2-GPT_POSTPROC` — GPT ledger/summarizer pipeline
- `v1.3-EPIC-3-INFRA` — Terraform + CI/CD automation
- `v1.4-EPIC-4-API` — FastAPI bridge (health, metrics, auth)
- `v1.5-EPIC-5-ANDROID` — Android client MVP (Buildozer APK)

### Epics & User Stories

#### EPIC-1 — Real-Time STT Core (✅ Complete — tag `v1.0-EPIC-1-STT_CORE`)
Goal: Port WhisperLiveKit fork, wrap VAD, and deliver resilient streaming transcripts.
- **US-1.1 – Integrate WhisperLiveKit fork & configure base VAD** — ✅ Done (real-time loop + config in `src/stt_core`).
- **US-1.2 – Add transcript streaming & local buffer** — ✅ Done (Redis Streams publisher + rolling JSONL buffer).
- **US-1.3 – Unit tests for audio → text → file output** — ✅ Done (pytest assets + CI gating).
> **Acceptance Gates**
> - LiveKit runner boots with configured backend/VAD and prints transcript segments.
> - Redis + buffer sinks persist and are asserted in CI (`tests/test_stt_*`).

#### EPIC-2 — GPT-4o-mini Post-Processing (✅ Complete — tag `v1.1-EPIC-2-GPT_POSTPROC`)
Goal: Transform transcripts into structured knowledge artifacts.
- **US-2.1 – Send transcripts to GPT-4o-mini via API** — ✅ Done (async OpenAI client + ledger appends).
- **US-2.2 – Extract structured JSON (transactions, events, notes)** — ✅ Done (session-aware prompts + ledger metadata).
- **US-2.3 – Store JSONL logs in `data/ledger/`** — ✅ Done (daily summary generator + robustness patch for GPT output).
> **Acceptance Gates**
> - `data/ledger.jsonl` grows per transcript with session metadata.
> - Daily summary generator produces markdown + structured JSON outputs without crashing on malformed GPT output (`safe_json_parse` tests).

#### EPIC-3 — CI/CD + Deployment (✅ Complete — tag `v1.3-EPIC-3-INFRA`)
Goal: Provide reproducible builds and automated deployment.
- **US-3.1 – Add Dockerfile + GitHub Actions workflow** — ✅ Done (multi-stage builds + pytest CI).
- **US-3.2 – Terraform DigitalOcean droplet setup** — ✅ Done (Droplet + Redis + outputs).
- **US-3.3 – Deploy daily auto-summary job** — ✅ Done (CI trigger calling summarizer, notifications wired).
> **Acceptance Gates**
> - `ci_cd.yml` executes lint/tests on every push + PR.
> - `infra/terraform` applies cleanly with documented variables and outputs (droplet IP + Redis URI).

#### EPIC-4 — API Bridge (FastAPI) (✅ Complete — tag `v1.4-EPIC-4-API`)
Goal: Ship a versioned API bridge for clients.
- **US-4.1 – FastAPI skeleton + auth** — ✅ Done (versioned router, API-key guard, error handling).
- **US-4.2 – `/v1/transcribe` & `/v1/ingest-transcript`** — ✅ Done (audio uploads + JSON ingestion wired to Redis/JSONL sinks).
- **US-4.3 – `/v1/ledger` & `/v1/summary`** — ✅ Done (daily summaries, ledger pagination, on-demand generation).
- **US-4.4 – `/healthz` & `/metrics` observability** — ✅ Done (disk + Redis checks, Prometheus counters).
> **Acceptance Gates**
> - Auth enforced via `X-API-Key` on every route; 401 tested.
> - `/metrics` emits Prometheus counters and is scraped in CI smoke tests.

#### EPIC-5 — Android DayMind Companion (✅ Complete — tag `v1.5-EPIC-5-ANDROID`)
Goal: Provide a background-friendly recorder with offline queue + summaries.
- **US-5.1 – Recording + chunker** — ✅ Done (Start/Stop toggle, 6 s WAV chunks, visual indicator).
- **US-5.2 – Settings + summary viewer** — ✅ Done (persisted settings, summary refresh, test connection UX).
- **US-5.3 – Offline queue + retries** — ✅ Done (durable queue, exponential backoff, Buildozer packaging + README).
Release recap: Android MVP verified via 24 green pytest suites, manual desktop preview, and Buildozer debug builds (`scripts/build_apk.sh` → `dist/daymind-debug.apk`). Recording indicator, offline queue, summary refresh, “Test connection,” log view, and “Clear queue” all confirmed on emulator.
> **Acceptance Gates**
> - `python -m mobile.daymind.main` demonstrates UX parity with Android build.
> - Buildozer spec + README instructions reproducibly generate a debug APK; queue persistence tested across restarts.

#### EPIC-6 — Finance / Ledger Analytics (Beancount + Fava) (🟡 In Progress)
Goal: turn GPT ledger events into double-entry books and dashboards.
- **US-6.1 – JSONL→Beancount exporter** — 🚧 In progress. Produce deterministic mappings of categories/currencies/time into `ledger.beancount`; cron runs daily straight from `data/ledger*.jsonl`. Success: `ledger.beancount` regenerates without manual edits.
- **US-6.2 – Fava dashboard service** — Wrap Fava under `/finance` with project ledger mounted and auth aligned with API keys. Success: charts/filters render for current dataset.
- **US-6.3 – Finance aggregates endpoint** — `GET /v1/finance` surfaces totals grouped by date/category with tests covering edge cases. Success: regression tests assert schema + calculations.
> **Acceptance Gates**
> - Exporter CI test compares known JSONL sample to `ledger.beancount`.
> - Fava health endpoint returns 200 and respects API auth.
> - `/v1/finance` documented and exercised in pytest (offline fixtures).

#### EPIC-7 — Long-Term Memory / Anki (genanki) (📥 Backlog)
Goal: capture “remember this” moments into spaced-repetition decks.
- **US-7.1 – Deck builder from memory commands** — Parse ledger/session directives and emit daily `Memory::DayMind::<YYYY-MM-DD>.apkg`. Success: deck artifact shows expected cards when imported.
- **US-7.2 – CI artifact export** — Nightly workflow uploads `.apkg` (and optional AnkiConnect note). Success: workflow summary links deck download.
- **US-7.3 – Schema & QA guard** — Define basic card templates, add smoke tests verifying round-trip (import/export). Success: sample AnkiDroid/Desktop import documented.
> **Acceptance Gates**
> - Deck metadata lists date stamp + tag set.
> - CI run stores `.apkg` artifact and surfaces checksum.
> - Automated test ensures at least one card renders with both front/back templates.

#### EPIC-8 — Automation & Daily Report (GitHub Actions schedule) (📥 Backlog)
Goal: autonomously regenerate data products and notify stakeholders each day.
- **US-8.1 – Daily cron workflow** — GitHub Actions schedule triggers summary refresh, JSONL→Beancount exporter, and ledger rollups. Success: workflow history shows daily success with attached artifacts.
- **US-8.2 – Apprise notifications** — Send Telegram/email message linking summary markdown + CSV. Success: notifications logged; secrets managed via GH.
- **US-8.3 – Health/report metrics snapshot** — Capture request counts/errors and publish inline with notification. Success: job output includes metrics JSON snippet.
> **Acceptance Gates**
> - Cron run recorded in Actions with retention of artifacts/logs.
> - Apprise dry-run test executed in CI using mock transports.
> - Metrics snippet validated via pytest fixture.

#### EPIC-9 — Release Management (Release Please) (📥 Backlog)
Goal: automate semantic versioning and changelog generation tied to epics.
- **US-9.1 – Configure Release-Please Action** — Conventional commits trigger version bumps + release PRs. Success: autop-run merges produce GitHub releases with assets.
- **US-9.2 – EPIC tag integration** — Release Please template references tags like `v1.6-EPIC-6-FINANCE` and groups changes per epic. Success: changelog includes epic headers + links.
> **Acceptance Gates**
> - Dry-run release shows correct next version.
> - Tagging workflow documented; governance notes updated for ReleaseAgent.

#### EPIC-10 — Orchestration (LangGraph) (📥 Backlog)
Goal: model DayMind as a LangGraph DAG stitched via Redis Streams for observability and retries.
- **US-10.1 – DAG definition** — Nodes for STT, GPT postproc, Finance exporter, Memory deck, Reporter. Success: runnable mock graph with state transitions logged.
- **US-10.2 – Redis Streams wiring** — Use XADD/XREADGROUP for events, including metrics on throughput/latency. Success: minimal harness demonstrates event handoffs locally.
- **US-10.3 – Runbook & contracts** — Document node interfaces, retries, backoff policies, and failure handling for OrchestratorAgent. Success: runbook stored in `docs/`.
> **Acceptance Gates**
> - Graph unit test asserts ordering + conditional branching.
> - Redis stream consumer benchmark recorded with lat/throughput metrics.
> - Runbook reviewed by Integrator + Automator agents.

### Kanban – Sprint 1
| Backlog | Next | In Progress | Done |
|---------|------|-------------|------|
| US-6.2 – Fava dashboard<br>US-6.3 – Finance aggregates endpoint<br>US-7.1 – Memory deck builder<br>US-7.2 – CI deck artifact<br>US-7.3 – Schema & QA<br>US-8.1 – Daily cron workflow<br>US-8.2 – Apprise notifications<br>US-8.3 – Health metrics snapshot<br>US-9.1 – Release-Please config<br>US-9.2 – Epic-aware tagging<br>US-10.1 – LangGraph DAG nodes<br>US-10.2 – Redis Streams wiring<br>US-10.3 – Runbook & contracts | — | **US-6.1 – JSONL→Beancount exporter** | US-1.1 / 1.2 / 1.3<br>US-2.1 / 2.2 / 2.3<br>US-3.1 / 3.2 / 3.3<br>US-4.1 / 4.2 / 4.3 / 4.4<br>US-5.1 / 5.2 / 5.3 |
