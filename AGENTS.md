# Symbioza-DayMind Agents

## Project Context
- **Project:** Symbioza-DayMind
- **Base:** WhisperLiveKit (real-time Whisper STT + VAD)
- **Goal:** Continuous real-time speech-to-text pipeline → GPT-4o-mini summarization → JSONL/CSV knowledge ledger.

## Agent Roles
- **Planner:** Maintains `AGILE_PLAN.md`, decomposes epics into user stories, and syncs overall progress.
- **Coder (Codex):** Implements modular Python components for each epic (STT core, GPT post-processing, etc.).
- **Critic:** Validates output via `pytest` and CI gates before merge.
- **Integrator:** Keeps repo structure clean (`src/`, `tests/`, `docker/`) and enforces conventions.
- **DevOps:** Operates CI/CD + Terraform, monitors `/healthz` + `/metrics`, and keeps GitHub Actions green.
- **API/Bridge:** Owns FastAPI surface, auth enforcement, ledger/summary delivery, and client SDK contracts.
- **Mobile:** Builds the Android/Kivy client, ensures recording UX, offline queue durability, and release packaging.
- **FinanceAgent:** Converts ledger data into Beancount, operates Fava, and surfaces `/v1/finance`.
- **MemoryAgent:** Generates genanki decks from “memory” directives and validates imports.
- **AutomatorAgent:** Runs scheduled workflows, exporters, and notification hooks.
- **ReleaseAgent:** Governs Release Please automation, tagging, and changelog parity with epics.
- **OrchestratorAgent:** Maintains LangGraph DAG + Redis Streams glue with observability and retry budgets.
- **Observer:** Ensures documentation, run logs, and ledger reports are up to date.

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

## Tech Stack Decision Log
- **Beancount + Fava:** Standard for double-entry audits + interactive dashboards; integrates cleanly with JSONL exporters.
- **genanki:** Lightweight, scriptable deck generation for daily spaced-repetition artifacts.
- **GitHub Actions schedule:** Centralized automation for exporters, summaries, and notifications without extra infra.
- **Release Please:** Automates semantic versioning + changelog generation tied to EPIC tags.
- **LangGraph:** Provides declarative DAG orchestration with Python-first ergonomics and native Redis Streams support.
