# DayMind API Reference

All endpoints require an `X-API-Key` header that matches one of the values in `API_KEYS`/`API_KEY`. Responses are JSON unless stated otherwise.

## Health & Metrics

### `GET /healthz`
Returns overall service status plus TLS state.
```json
{
  "ok": true,
  "redis": "ok",
  "disk": "ok",
  "openai": "skip",
  "tls": "ok",
  "timestamp": "2024-11-01T12:00:00Z"
}
```
- `redis`/`openai`/`tls` report `ok`, `skip`, or `error`.

### `GET /metrics`
Prometheus exposition of `api_requests_total{path,method,status}` and `api_request_latency_seconds{path,method}`. Scrape with Prometheus or `curl` (header auth required).

## Speech & Transcript Endpoints

### `POST /v1/transcribe`
- **Request:** `multipart/form-data`
  - `file`: WAV/FLAC/MP3 chunk (≤10 s ideal)
  - `lang` (optional): override language autodetect
  - `speech_segments` (optional): JSON array with `{start_ms,end_ms}` windows describing when speech was detected inside the clip; helps GPT summarize pauses vs. active speech.
- **Response:**
```json
{
  "text": "ahoj světe",
  "lang": "cs",
  "start": 0.0,
  "end": 5.8,
  "confidence": 0.92,
  "session_id": 3
}
```
- **Side effects:** uploads are published to Redis (`REDIS_STREAM`) and appended to the local transcripts buffer.

### `POST /v1/ingest-transcript`
Bypasses audio and stores JSON directly.
```json
{
  "text": "Nakup za 120 Kč",
  "start": 0.0,
  "end": 5.0,
  "lang": "cs",
  "meta": {"source": "import"}
}
```
Response:
```json
{ "status": "ok", "stored_at": 1731100000.0 }
```

## Knowledge Ledger

### `GET /v1/ledger`
Query stored ledger entries for a single day.
- Query params: `date=YYYY-MM-DD` (required)
- Response:
```json
{
  "date": "2024-11-01",
  "count": 12,
  "entries": [
    {"session_id": 2, "input": "Koupit kávu", "start": 1731100000.0, "end": 1731100005.0},
    ...
  ]
}
```

### `GET /v1/summary`
Returns markdown summary for a day, generating it on demand if missing.
- Query params: `date=YYYY-MM-DD`
- Response:
```json
{
  "date": "2024-11-01",
  "summary_md": "## Shrnutí\n- ...\n"
}
```

## Finance

### `GET /v1/finance`
Aggregates Beancount transactions by date + category.
- Optional `date=YYYY-MM-DD` filter.
- Response:
```json
{
  "count": 3,
  "items": [
    {"date": "2024-11-01", "category": "Expenses:Food", "total": 120.0, "currency": "CZK"},
    {"date": "2024-11-01", "category": "Income:Salary", "total": 5000.0, "currency": "CZK"}
  ]
}
```

### `GET /finance`
Redirects authenticated users to the live Fava dashboard (`http://<host>:5000` or `FAVA_BASE_URL`).

## Auth & Usage

### `GET /v1/usage`
Returns usage stats for the calling API key.
```json
{
  "owner": "team-alpha",
  "created_at": 1731300000.0,
  "usage_count": 1245,
  "requests_today": 34,
  "last_used": 1731333333.0
}
```

### `GET /welcome`
Public endpoint that confirms the API is live and returns links to onboarding, security, and billing docs.

## System Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Disk/Redis/OpenAI diagnostics |
| GET | `/metrics` | Prometheus counters/histograms |
| POST | `/v1/transcribe` | Upload audio chunk |
| POST | `/v1/ingest-transcript` | Ingest raw transcript |
| GET | `/v1/ledger` | Retrieve ledger entries for a day |
| GET | `/v1/summary` | Fetch markdown summary |
| GET | `/v1/finance` | Finance aggregates |
| GET | `/finance` | Redirect to Fava UI |
| GET | `/v1/usage` | Usage stats for current API key |
| GET | `/welcome` | Public welcome/docs pointer |

## Error Handling
- `401 Unauthorized` – missing/invalid API key.
- `404 Not Found` – summary or ledger not ready yet.
- `422 Unprocessable Entity` – invalid payloads (FastAPI validation).
- `500 Internal Server Error` – logged with stack traces; inspect `journalctl -u daymind-api`.

## Examples

```bash
curl -H "X-API-Key: $API_KEY" \
     -F file=@tests/assets/sample_cs.wav \
     http://localhost:8000/v1/transcribe

curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/v1/summary?date=$(date +%F)"

curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/v1/finance"
```
