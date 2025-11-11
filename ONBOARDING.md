# DayMind Onboarding Guide

Use this checklist to bring a new teammate, customer, or droplet online in less than 30 minutes.

## 1. Clone & Install

```bash
git clone https://github.com/your-org/daymind.git
cd daymind
scripts/setup_daymind.sh
```

The helper creates `.venv`, installs requirements, and reminds you how to start `uvicorn` locally.

## 2. Configure Environment

1. Copy `.env.example` to `.env` and populate:
   - `API_KEYS` or generate one via the auth CLI
   - `OPENAI_API_KEY`
   - `REDIS_URL` (optional but recommended)
2. (Optional) Update `API_KEY_STORE_PATH` if you want keys stored outside `data/`.

## 3. Generate an API Key

```bash
python -m src.api.services.auth_service --store data/api_keys.json create demo-user
```

Share the resulting key securely. Every request increments counters visible via `/v1/usage`.

## 4. Run the API Locally

```bash
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Smoke tests:

```bash
curl http://localhost:8000/welcome
curl -H "X-API-Key: $KEY" http://localhost:8000/healthz
curl -H "X-API-Key: $KEY" http://localhost:8000/v1/usage
```

## 5. Mobile App (Optional)

- Follow `mobile/daymind/README.md` to build the Kivy client via Buildozer.
- Enter the server URL + API key in the Settings tab, tap “Test connection”, then start recording.

## 6. Deployment Path

- For production, follow `DEPLOY.md` (systemd-first) or `docker-compose.prod.yml`.
- Configure HTTPS via the provided `infra/caddy/Caddyfile` or your preferred reverse proxy.

## 7. Common Issues

| Symptom | Fix |
|---------|-----|
| `401 Unauthorized` | Key missing or revoked. Re-run the auth CLI and restart the API if needed. |
| `429 Too Many Requests` | Per-key rate limit reached. Increase `API_RATE_LIMIT_PER_MINUTE` or throttle clients. |
| `/finance` 404 | Run `python -m src.finance.export_beancount` and ensure Fava service is active. |
| `TLS` health check = `error` | Set `TLS_PROXY_HOST` or disable `TLS_REQUIRED` for local dev. |

## 8. Hand-off Artifacts

- Share `ONBOARDING.md`, `API_REFERENCE.md`, `SECURITY.md`, and `BILLING.md` with new operators.
- Record API keys and ledger artifacts in a secure vault to maintain compliance.
