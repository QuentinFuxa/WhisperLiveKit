# DayMind Billing & Usage Tracking

DayMind ships with a lightweight authentication + billing scaffold so you can track per-customer usage on self-hosted or SaaS deployments. The defaults run entirely offline and write JSON metadata that follows the Text-First Storage rule.

## Key Components

| Component | Path | Purpose |
|-----------|------|---------|
| API key store | `data/api_keys.json` (configurable via `API_KEY_STORE_PATH`) | Persists `key`, `owner`, `created_at`, `usage_count`, `requests_today`, `last_used`, `revoked` flags. |
| Auth service | `src/api/services/auth_service.py` | Validates keys, enforces per-key rate limits (Redis-backed if available), records usage on every request. |
| `/v1/usage` | FastAPI endpoint | Returns live counters for the calling API key so customers can self-serve usage metrics. |
| Billing stub | `src/billing/stripe_stub.py` | Placeholder for Stripe/Paddle integrations; swap it for a real client when you are ready to charge. |

## Environment Variables

```
API_KEY_STORE_PATH=data/api_keys.json
API_RATE_LIMIT_PER_MINUTE=120
IP_RATE_LIMIT_PER_MINUTE=240
BILLING_MODE=local        # or "stripe" to enable Stripe integration hooks
STRIPE_SECRET_KEY=sk_live_...
```

`API_KEYS` remains supported for bootstrapping, but using the JSON store allows DayMind to track metadata for each key and expose it via `/v1/usage`.

## Managing API Keys

Use the built-in CLI helper:

```bash
python -m src.api.services.auth_service --store data/api_keys.json list
python -m src.api.services.auth_service --store data/api_keys.json create team-alpha
python -m src.api.services.auth_service --store data/api_keys.json revoke dm_xxx
```

Keys generated via CLI immediately become active (rate limited and logged). When running behind systemd, point the tool to `/etc/daymind/daymind.env`'s `API_KEY_STORE_PATH`.

## Usage Tracking Format

Each record in `data/api_keys.json` follows:

```json
{
  "key": "dm_abc...",
  "owner": "team-alpha",
  "created_at": 1731300000.0,
  "usage_count": 1245,
  "requests_today": 34,
  "requests_day": 20241111,
  "last_used": 1731333333.0,
  "revoked": false
}
```

These fields drive `/v1/usage` and any downstream billing automation (Stripe/Paddle). Because the file is JSON, you can tail it, commit sanitized snapshots, or feed it into custom analytics jobs.

## Stripe / Paddle Hooks

- Set `BILLING_MODE=stripe` and provide `STRIPE_SECRET_KEY` to signal that DayMind should use the Stripe stub. The current stub simply prepares data structures; replace `StripeBillingStub` with the official SDK when you are ready.
- Extend `src/billing/stripe_stub.py` or drop in `src/billing/stripe_client.py` to call Stripe APIs for invoices, subscriptions, and metered billing.

## Integrating With CI / Reports

- Schedule a GitHub Actions workflow (see EPIC-8) to call `/v1/usage` for each key and archive the JSON output.
- Publish customer-facing usage dashboards or email summaries by reading `data/api_keys.json` and `data/ledger*.jsonl`.

## Support Scripts

- `scripts/setup_daymind.sh` – bootstraps a virtualenv and installs deps.
- `scripts/export_finance.sh` – runs the JSONL→Beancount exporter; chain it with billing tasks for finance-ready snapshots.
