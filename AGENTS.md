# Symbioza-DayMind Agents

## Project Context
- **Project:** Symbioza-DayMind
- **Base:** WhisperLiveKit (real-time Whisper STT + VAD)
- **Goal:** Continuous real-time speech-to-text pipeline → GPT-4o-mini summarization → JSONL/CSV knowledge ledger.

## Agent Roles
- **Planner:** Maintains `AGILE_PLAN.md`, decomposes epics into user stories, syncs overall progress, monitors EPIC-13 Android Kotlin progress, and readies LangGraph orchestration for EPIC-14.
- **Coder (Codex):** Implements modular Python components for each epic (STT core, GPT post-processing, etc.).
- **Critic:** Validates output via `pytest` and CI gates before merge.
- **Integrator:** Keeps repo structure clean (`src/`, `tests/`, `docker/`) and enforces conventions.
- **DevOps:** Operates CI/CD + Terraform, monitors `/healthz` + `/metrics`, keeps GitHub Actions green, and validates legal/compliance artifacts (`LICENSE`, `NOTICE.md`, Fava as an external service) before deployments.
- **API/Bridge:** Owns FastAPI surface, auth enforcement, ledger/summary delivery, and client SDK contracts.
- **Mobile:** Builds the Android/Kivy client, ensures recording UX, offline queue durability, and release packaging.
- **Kotlin Mobile Agent:** Owns the Android Compose client integration (upload/summary/finance), adheres to API-first contracts, and measures upload success rate, crash-free sessions, and summary latency.
- **FinanceAgent:** Converts ledger data into Beancount, operates Fava, and surfaces `/v1/finance`.
- **MemoryAgent:** Generates genanki decks from “memory” directives and validates imports.
- **AutomatorAgent:** Runs scheduled workflows, exporters, and notification hooks.
- **ReleaseAgent:** Governs Release Please automation, tagging, and changelog parity with epics, and double-checks license/NOTICE accuracy before every release.
- **OrchestratorAgent:** Maintains LangGraph DAG + Redis Streams glue with observability and retry budgets.
- **Observer:** Ensures documentation, run logs, and ledger reports reflect the Text-First Storage invariant and stay current.
- **SecurityAgent:** Protects ingress (firewall, TLS proxy, rate limits), watches anomalies, and triages `/healthz` TLS/security states.
- **BillingAgent:** Tracks per-key usage, prepares invoices or Stripe sync, and reports `/v1/usage` trends to stakeholders.
- **OnboardingAgent:** Verifies new deployments with `scripts/setup_daymind.sh`, keeps onboarding docs accurate, and confirms landing site content.

### FinanceAgent
- **Inputs:** `data/ledger*.jsonl`, category/currency mappings, redis event hooks.
- **Outputs:** `ledger.beancount`, `/finance` Fava dashboard, `/v1/finance` aggregates.
- **KPIs:** Daily exporter run succeeds, Fava uptime ≥99%, finance endpoint schema tests green.

### MemoryAgent
- **Inputs:** Session directives (“zapamatuj si to”, “vytvoř Anki kartičky”), summarized ledger entries.
- **Outputs:** `Memory::DayMind::<YYYY-MM-DD>.apkg` decks, schema docs, optional AnkiConnect notes.
- **KPIs:** Daily deck artifact present, sample import smoke tests pass, card templates render correctly.

### AutomatorAgent
- **Inputs:** GitHub Actions schedules, workflow YAML, Apprise secrets.
- **Outputs:** Daily cron workflows (summary/export/beancount), notification payloads (Telegram/email), metrics snapshots.
- **KPIs:** Jobs fire on schedule, artifacts uploaded, notifications delivered (logged) with <5% failure.

### ReleaseAgent
- **Inputs:** Conventional commits, EPIC tags, Release Please config.
- **Outputs:** Automated release PRs/drafts, semantic version tags (e.g., `v1.6-EPIC-6-FINANCE`), changelog sections by epic.
- **KPIs:** Releases created without manual edits, changelog freshness (≤24h lag), governance docs up to date.

### OrchestratorAgent
- **Inputs:** LangGraph specs, Redis Streams, node contracts from other agents.
- **Outputs:** Runnable DAG definitions, event routing, retry/backoff policies, operational runbook.
- **KPIs:** DAG dry-run latency targets met, stream consumers show <1% failure, runbook reviewed each sprint.
  - **Invariant:** Text-First Storage — ensure DAG nodes write/read only text/JSONL artifacts and respect the shared ledger files.

### SecurityAgent
- **Inputs:** Firewall state (Terraform/DO), `infra/caddy/Caddyfile` / nginx configs, rate-limit metrics, `/healthz` TLS status.
- **Outputs:** Hardened proxy configs, alerts on anomalies, documented security posture in `SECURITY.md` + `DEPLOY.md`.
- **KPIs:** No public Redis exposure, TLS certificates auto-renew, IP/key rate-limit incidents resolved within 4h, `/healthz` stays green for security fields.

### BillingAgent
- **Inputs:** `data/api_keys.json`, `/v1/usage` responses, billing mode env vars (`BILLING_MODE`, `STRIPE_SECRET_KEY`).
- **Outputs:** Usage reports, invoices or CSV exports, Stripe/Paddle sync scripts (stub today), updates to `BILLING.md`.
- **KPIs:** API key metadata current, invoices issued on schedule, zero drift between `/v1/usage` and stored counters.

### OnboardingAgent
- **Inputs:** `ONBOARDING.md`, `landing/` site, `scripts/setup_daymind.sh`, `/welcome` endpoint output.
- **Outputs:** Verified quick-start flow, updated screenshots/docs, GitHub Pages deploy status.
- **KPIs:** New operator can reach `/v1/summary` in <30 min, onboarding doc reviewed every sprint, landing site deploy succeeds on each main push.

## Data & Storage Responsibilities
- All agents must log their outputs to human-readable text or JSONL files; binary audio may exist temporarily for capture but can be discarded once transcripts persist. This Text-First Storage invariant ensures every node, agent, and audit can replay state without reprocessing raw audio.

## Workflow
1. All planning artifacts live in `AGILE_PLAN.md` and are the single source of truth.
2. Each epic is broken into user stories → granular tasks → commits.
3. Completing a task requires updating both `AGILE_PLAN.md` and this file's progress markers.
4. The Planner performs an automated sync of progress markers at the end of each day.

## Core Tech Stack
- Python 3.11 + FastAPI + WhisperLiveKit core
- Silero VAD / Whisper / OpenAI API
- Redis Streams for inter-agent events
- Docker + GitHub Actions CI/CD
- Terraform for server deploy

## Progress Notes
- Realtime transcript sinks: Redis + JSONL wired.
- EPIC-1 (STT Core) closed with E2E verification tag `v1.0-EPIC-1-STT_CORE`.
- Session-aware GPT post-processing enabled; GPT pipeline now generates daily structured summaries (US-2.3 complete).
- QA note: `safe_json_parse` ensures malformed GPT output never breaks summaries.
- EPIC-3 DevOps runway started: Terraform + CI/CD automation online.
- EPIC-4 API bridge live: versioned endpoints + auth/metrics shipped.
- EPIC-5 Android client released: recording indicator, offline queue, summary refresh, Buildozer instructions, and helper script shipped (`v1.5-EPIC-5-ANDROID` tag).
- EPIC-6 finance track live: JSONL→Beancount exporter + Fava `/finance` bridge with `/v1/finance` aggregates.
- EPIC-11 services hardened: systemd units, CI deploy job, TLS-aware health checks, and Text-First compliance gates.
- EPIC-12 kickoff: Auth/billing service, security hardening, landing site, and onboarding docs tracked for commercial readiness.
- EPIC-13 CI gates: GitHub Actions `terraform_apply → deploy_app → verify_services` pipeline provisions infra, pushes `/opt/daymind` via `scripts/remote_deploy.sh`, and enforces Redis/API/Fava + `/healthz`/`/metrics` checks on every run.

## Tech Stack Decision Log
- **Beancount + Fava:** Standard for double-entry audits + interactive dashboards; integrates cleanly with JSONL exporters.
- **genanki:** Lightweight, scriptable deck generation for daily spaced-repetition artifacts.
- **GitHub Actions schedule:** Centralized automation for exporters, summaries, and notifications without extra infra.
- **Release Please:** Automates semantic versioning + changelog generation tied to EPIC tags.
- **LangGraph:** Provides declarative DAG orchestration with Python-first ergonomics and native Redis Streams support.

## Operator Runbook (DevOps + Release)
- **Status & logs:** `systemctl status daymind-api daymind-fava`, `journalctl -u daymind-api -n 200`.
- **Health:** `curl -H "X-API-Key:..." http://<host>:8000/healthz` (expects `redis/disk/openai` keys) and `curl .../metrics | grep api_requests_total`.
- **Deploy:** Merge to `main` ⇒ GitHub Actions `deploy_app` job drives `scripts/remote_deploy.sh` to sync `/opt/daymind`, seeds `/etc/daymind/daymind.env` from the example only if missing, reinstalls deps, and restarts services.
- **CI Deploy Flow:** `terraform_apply` runs pytest/terraform with DO + SSH secrets, `deploy_app` calls `scripts/remote_deploy.sh` to sync `/opt/daymind` and restart services, `verify_services` runs `infra/systemd/checks/healthcheck.sh` plus `/healthz`/`/metrics` curls before tagging.
- **Firewall:** Only ports 22/8000 (and optional 5000/443) open; Redis remains private.
- **Rollback:** `git checkout <tag>` in `/opt/daymind`, rerun install block, `systemctl restart daymind-api daymind-fava`. ReleaseAgent coordinates tags (`v1.7.0-EPIC-11-MVP_SERVER` onward).
