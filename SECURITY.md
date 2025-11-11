# DayMind Security Guide

DayMind is designed to run on a single droplet with optional managed services (Redis, OpenAI). This guide summarizes the controls shipped with EPIC‑12 and how to operate them.

## 1. Authentication

- All endpoints except `/welcome` demand `X-API-Key`.
- Keys live in `API_KEY_STORE_PATH` (JSON) and track usage metadata.
- Rotate keys via `python -m src.api.services.auth_service ...` and distribute them securely.
- `/v1/usage` lets customers self-check their consumption.

## 2. Rate Limiting

- Per-key limits: `API_RATE_LIMIT_PER_MINUTE` (Redis-backed token bucket; falls back to in-memory).
- Global/IP limits: `IP_RATE_LIMIT_PER_MINUTE` (middleware defined in `src/api/deps/security.py`). Set to `0` to disable.
- Exceeding a limit returns HTTP `429` with `{"detail": "Rate limit exceeded"}` or `"Too many requests"` (IP guard).

## 3. TLS & Reverse Proxy

- Use the provided `infra/caddy/Caddyfile` to terminate HTTPS with Let’s Encrypt on Caddy. Set `DAYMIND_DOMAIN` env var before running Caddy.
- Alternatively, adapt the config to Nginx. Ensure `/v1/*` proxies to port 8000 and `/finance` proxies to 5000 (or hide it entirely).
- `TLS_REQUIRED=true` forces `/healthz` to report `tls="error"` until `TLS_PROXY_HOST` is set, helping CI detect misconfigurations.

## 4. Firewalls & Ports

- Allow inbound: SSH (22), HTTPS (443 via proxy), optional HTTP (80 for ACME), API (8000) only if not behind proxy, Fava (5000) only for internal/VPN access.
- Block Redis (6379) from WAN; keep it bound to `127.0.0.1` or a private subnet.

## 5. Logging & Storage

- All critical data (transcripts, GPT outputs, finance ledgers, API key metadata) are persisted as text/JSONL files—see the Text-First Storage principle in `README.md`.
- When logging potentially sensitive payloads, call `anonymize_text` from `src/api/deps/security.py` to mask long numeric sequences.
- Journald rotation handles service logs; tighten retention via `/etc/systemd/journald.conf` if needed.

## 6. Health Monitoring

- `/healthz` reports `redis`, `disk`, `openai`, and `tls` states. Wire it into Prometheus or UptimeRobot with the appropriate API key.
- `/metrics` exposes Prometheus counters/histograms with labels `{path,method,status}`.

## 7. Incident Response

1. `journalctl -u daymind-api -n 200` to inspect errors.
2. Revoke suspicious API keys and rotate them via the CLI.
3. Restore clean ledger/transcript files from backups (they are plain text for easy diffing).
4. Reissue TLS certificates if keys are compromised; restart the proxy.

## 8. Compliance Checklist

- LICENSE (MIT) + NOTICE.md document dependency obligations (GPL components run as external services only).
- `BILLING.md` covers per-key usage reports and billing-ready metadata.
- `DEPLOY.md` includes firewall and rollback steps; keep it handy for audits.
