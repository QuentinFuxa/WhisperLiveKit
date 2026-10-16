# Security guard

An optional application-layer security layer built on [fastapi-guard](https://github.com/Guard-Core/fastapi-guard). It runs inside the server, so it knows WhisperLiveKit's routes, and it complements a reverse proxy, WAF or firewall in front of the server (TLS stays there). Where nothing sits in front of the server, it is the protection. It is off by default: with it off, `guard` is never imported and the application is built exactly as before.

Statements about engine behavior below hold with fastapi-guard 8.1.1 and guard-core 4.4.0.

## What it adds

- Attack-pattern screening of the path, query string and headers (path traversal, injection, XSS, scanner probes, with semantic analysis) on the HTTP routes and on the WebSocket handshake of `/asr` and `/v1/listen`, with automatic bans for clients that keep probing.
- A banned or refused client never reaches `accept()`: no `AudioProcessor`, FFmpeg process or decoder is started for it. Uvicorn answers the handshake with HTTP 403.
- A per-client budget on new WebSocket sessions and on HTTP requests, and a stricter budget, a size limit and a content-type filter on `POST /v1/audio/transcriptions`.
- A rule that bans a client after repeated 401 responses from the HTTP API.
- IP and CIDR allow and block lists, country allow or block lists, blocking of cloud-provider IP ranges and user-agent filtering, all enforced at the WebSocket handshake as well.
- Emergency mode, which locks the server down to a short allowlist during an incident, and passive mode, which only logs, to try the rules before enforcing them.

Request bodies (audio and form fields) are never scanned. State lives in memory, so nothing else has to be deployed.

## Enable it

```bash
uv pip install 'whisperlivekit[security]'
wlk --security-guard
```

`WLK_SECURITY_GUARD=1` enables it too, including when the app is started as `uvicorn whisperlivekit.basic_server:app`. If it is enabled and fastapi-guard is not installed, the server stops at startup with an error that names the extra to install.

| Option | Purpose | Default |
|---|---|---|
| `--security-guard` | Enable the guard | off |
| `--security-guard-allow-ips` | Only these IPs or CIDR ranges may connect (comma-separated) | unset |
| `--security-guard-block-ips` | IPs or CIDR ranges that are always refused | unset |
| `--security-guard-allow-countries` | Only these country codes may connect (needs `IPINFO_TOKEN`) | unset |
| `--security-guard-block-countries` | Country codes that are refused (needs `IPINFO_TOKEN`) | unset |
| `--security-guard-block-clouds` | Cloud providers whose IP ranges are refused, for example `AWS,GCP,Azure` | unset |
| `--security-guard-block-user-agents` | User-agent patterns that are refused | unset |
| `--security-guard-rate-limit` | Per client and window: HTTP requests, and separately new WebSocket sessions | `60` |
| `--security-guard-rate-window` | Window in seconds for the budgets | `60` |
| `--security-guard-upload-rate-limit` | Per client and window: `POST /v1/audio/transcriptions` requests | `10` |
| `--security-guard-max-upload-mb` | Largest accepted `/v1/audio/transcriptions` request | `512` |
| `--security-guard-emergency` | Emergency mode: refuse everyone not in the emergency list | off |
| `--security-guard-emergency-allow-ips` | IPs or CIDR ranges admitted in emergency mode | unset |
| `--security-guard-passive` | Log without blocking | off |

The same settings exist as `security_guard*` fields of `WhisperLiveKitConfig`. Country rules use an IPinfo token read from the `IPINFO_TOKEN` environment variable; the token is never accepted on the command line. Cloud-provider ranges are downloaded when the server starts.

## Sizing the budget

`--security-guard-rate-limit` is counted per client address. HTTP requests and WebSocket handshakes have separate counters, so a busy browser tab does not use up the session budget. The default of 60 per 60 seconds follows from what a client does:

- A cold load of the browser UI is one request, `GET /`, because the page, script and icons are served inline, plus the `/favicon.ico` request browsers add on their own. The page makes no other HTTP request.
- The UI opens one WebSocket per recording. After an unexpected disconnect it reconnects with a backoff of 1 s doubling up to 30 s, which is about six attempts in a minute during an outage.
- fastapi-guard's own default of 10 per 60 seconds would be used up by five page loads.

At 60 per minute a single user has roughly ten times the headroom they use, several users behind one address still fit, and a client that opens a session every second is stopped. Raise it for many users behind one NAT or proxy address, and lower it on a small GPU box.

## Behavior to know

- **Bans.** A client that triggers 10 detections (fastapi-guard's `auto_ban_threshold`) is banned for an hour, and so is a client that returns more than five 401 responses within five minutes from `POST /v1/audio/transcriptions`. Bans apply to HTTP and to WebSocket handshakes. The engine never bans a loopback address or a trusted proxy, so a client connecting from `127.0.0.1` can be rate limited but not banned. Bans are held in memory per process and are lost on restart; with several workers each counts separately.
- **401 rule.** It counts the HTTP API only. A wrong token on a WebSocket closes the session with 4401 after the handshake and does not count.
- **Detection scope.** The free-text `context` query parameter is excluded, because free text trips the SQL injection patterns. The `ssrf` category is disabled: with guard-core 4.4.0 a `Referer` pointing at localhost or a LAN address is flagged, and the browser UI sends exactly that when the server is reached through them. A `Referer` with a script payload is still refused.
- **Excluded paths.** The paths in fastapi-guard's `exclude_paths` (`/docs`, `/openapi.json`, `/favicon.ico` and others) skip detection only; rate limits and IP rules still apply to them.
- **Allowlists.** A client matching `--security-guard-allow-ips` also skips detection and rate limiting.
- **Passive mode.** It logs instead of blocking on the HTTP routes, and for attack screening, user-agent and emergency rules at the WebSocket handshake. A WebSocket handshake from a blocklisted address, a banned address or a client over its session budget is still refused.
- **Headers.** The guard adds no response headers and no CORS handling: `CORSMiddleware` stays outermost, so a 403 or 429 from the guard still carries its CORS headers.

## What it does not do

[#455](https://github.com/QuentinFuxa/WhisperLiveKit/issues/455): the guard limits how fast one client can open sessions and shuts out abusers, but it does not cap concurrent sessions or close idle ones. Those two belong in the server itself.

## Client address and proxies

Uvicorn rewrites the client address from `X-Forwarded-For` before any middleware runs, so the guard and Uvicorn must trust the same proxies. With the guard enabled, `wlk` passes `--forwarded-allow-ips` to both. Without that option, proxy headers are turned off in Uvicorn, even when `FORWARDED_ALLOW_IPS` is set in the environment, so a client cannot change its identity by sending its own `X-Forwarded-For`.

Behind a reverse proxy, set `--forwarded-allow-ips` to the proxy address, and every client is then counted by the address the proxy reports.

When the app is started directly with `uvicorn whisperlivekit.basic_server:app`, the guard trusts no proxy, so pass `--no-proxy-headers` to Uvicorn. Uvicorn's default of trusting only `127.0.0.1` is also safe for a proxy on the same host. Never set `FORWARDED_ALLOW_IPS=*` there: it would let any client choose its address.

## Embedding

```python
from whisperlivekit.basic_server import create_app
from whisperlivekit.config import WhisperLiveKitConfig

app = create_app(WhisperLiveKitConfig(security_guard=True, security_guard_allow_ips="10.0.0.0/8"))
```
