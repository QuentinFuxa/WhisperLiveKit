# DayMind Deployment Runbook

This guide promotes a clean, systemd-first deployment on a single DigitalOcean droplet. The optional Docker Compose path mirrors the same topology for teams that prefer containers, but systemd remains the supported baseline for production.

## 1. Prerequisites
- Ubuntu/Debian host with Python 3.11+, git, ffmpeg, and Redis (can be remote).
- DNS entry or public IP.
- Firewall that only exposes SSH (22), API (8000), and optionally Fava (5000) or a reverse proxy (Caddy/Nginx).
- GitHub secrets for CI deploys:
  - `DO_TOKEN`, `SSH_FINGERPRINT` (Terraform)
  - `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PATH`
  - `DEPLOY_SSH_KEY` (private key with access to the droplet)
  - `DAYMIND_ENV` (entire contents of the environment file described below)

## 2. Provisioning
1. Apply the Terraform stack (`infra/terraform`) to create the droplet + Redis.
2. SSH into the host and install base packages:
   ```bash
   sudo apt-get update
   sudo apt-get install -y git python3 python3-venv ffmpeg rsync
   ```
3. Create a service user and directories:
   ```bash
   sudo useradd --system --shell /usr/sbin/nologin --home /opt/daymind daymind
   sudo mkdir -p /opt/daymind
   sudo chown -R daymind:daymind /opt/daymind
   ```
4. Clone the repository into `/opt/daymind` and create a virtualenv (or run `scripts/setup_daymind.sh` locally for parity):
   ```bash
   sudo -u daymind git clone https://github.com/<org>/daymind.git /opt/daymind
   cd /opt/daymind
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install -r requirements.txt
   .venv/bin/pip install -e .
   ```

## 3. Environment file
Copy `infra/systemd/daymind.env.example` to `/etc/daymind/daymind.env` and populate:
```
API_KEYS=prod-key-1,prod-key-2
API_KEY_STORE_PATH=/opt/daymind/data/api_keys.json
API_RATE_LIMIT_PER_MINUTE=120
IP_RATE_LIMIT_PER_MINUTE=240
OPENAI_API_KEY=sk-...
REDIS_URL=redis://10.0.0.5:6379/0
REDIS_STREAM=daymind:transcripts
TRANSCRIPT_PATH=/opt/daymind/data/transcripts.jsonl
LEDGER_PATH=/opt/daymind/data/ledger.jsonl
SUMMARY_DIR=/opt/daymind/data
FINANCE_LEDGER_PATH=/opt/daymind/finance/ledger.beancount
FAVA_HOST=127.0.0.1
FAVA_PORT=5000
TLS_REQUIRED=true
TLS_PROXY_HOST=daymind.example.com
STRIPE_SECRET_KEY=
BILLING_MODE=local
SESSION_GAP_SEC=45
```
Ensure the file is readable by the `daymind` group only:
```bash
sudo chown root:daymind /etc/daymind/daymind.env
sudo chmod 640 /etc/daymind/daymind.env
```

## 4. systemd services
Install the provided units:
```bash
sudo cp infra/systemd/daymind-api.service /etc/systemd/system/
sudo cp infra/systemd/daymind-fava.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now daymind-api daymind-fava
```

Check status/logs:
```bash
systemctl status daymind-api daymind-fava
journalctl -u daymind-api -n 200
```

## 5. Health, metrics, and smoke tests
```bash
curl -H "X-API-Key: <key>" http://<host>:8000/healthz
curl -H "X-API-Key: <key>" http://<host>:8000/metrics | grep api_requests_total
curl -H "X-API-Key: <key>" "http://<host>:8000/v1/summary?date=$(date +%F)"
curl -H "X-API-Key: <key>" "http://<host>:8000/v1/finance"
curl -H "X-API-Key: <key>" http://<host>:8000/v1/usage
curl http://<host>:8000/welcome
```

b. For Fava, either expose port 5000 (restricted to VPN/office IPs) or forward `/finance` through a reverse proxy.

## 6. Fava as an external service

- Fava runs independently via `daymind-fava.service`, reading `finance/ledger.beancount` and serving dashboards on port 5000.
- The FastAPI bridge interacts with it solely over HTTP (`/finance` redirect) and never links against Fava/Beancount binaries (GPL tools remain external).
- Environment file `infra/systemd/daymind.env` contains the shared configuration; drop it into `/etc/daymind/daymind.env`.
- Diagram:
  ```
  [Audio → STT → data/transcript_<date>.jsonl]
                   ↓
          [FastAPI Bridge on 8000]
                   ↓ (HTTP)
            [Fava Dashboard on 5000]
  ```

## 7. Security, TLS & firewall
- Apply a DigitalOcean firewall (or `ufw`) that allows: SSH (22), API (8000), optional HTTPS (443) if reverse proxying, and blocks Redis from WAN access.
- Optional TLS: deploy Caddy (`infra/caddy/Caddyfile`) or Nginx to terminate HTTPS and proxy `/v1/*` → 8000, `/finance` → 5000. Set `DAYMIND_DOMAIN` before starting Caddy to enable automatic Let’s Encrypt certificates.
- Rotate API keys regularly and keep `/etc/daymind/daymind.env` out of backups.

## 8. Optional Docker Compose path
Build and run containers locally or on the server:
```bash
cp infra/systemd/daymind.env.example .env.daymind   # edit values
docker compose -f docker-compose.prod.yml --env-file .env.daymind up -d --build
```
Services:
- `api` – FastAPI bridge (port 8000)
- `fava` – Finance dashboard (port 5000)
- `redis` – local Redis for STT buffering (optional if using managed Redis)

## 9. Static Landing & CI/CD
- `landing/` publishes to GitHub Pages after every successful `main` build.
- `.github/workflows/ci_cd.yml` includes `landing`, `deploy`, and `deploy_app` jobs: static site → Terraform → rsync/systemd restart.
`.github/workflows/ci_cd.yml` now contains `deploy_app`:
- After tests + Terraform, GitHub Actions rsyncs the repo to `$DEPLOY_PATH`, writes the env file from `DAYMIND_ENV`, installs dependencies, and restarts the services via systemd.
- Secrets required: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_PATH`, `DEPLOY_SSH_KEY`, `DAYMIND_ENV`.
- Logs appear in the workflow summary; remote logs remain accessible via `journalctl`.

## 10. Rollback
1. `git checkout <previous-tag>` in `/opt/daymind`.
2. Re-run the deployment block (`pip install -e .` + `systemctl restart daymind-api daymind-fava`).
3. If an emergency stop is needed:
   ```bash
   sudo systemctl stop daymind-api daymind-fava
   sudo systemctl disable daymind-api daymind-fava
   ```

## 11. Release checklist
- Tests green (`pytest -q`).
- `python -m src.finance.export_beancount` regenerated `finance/ledger.beancount`.
- `python -m src.finance.fava_runner` reachable locally.
- `/healthz`, `/metrics`, `/v1/summary`, `/v1/finance` verified on the droplet.
- Tag the repo: `git tag -a v1.7.0-EPIC-11-MVP_SERVER -m "Serverized MVP release"` and push.
