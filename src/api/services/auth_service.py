"""API key management, usage tracking, and rate limiting helpers."""

from __future__ import annotations

import argparse
import json
import secrets
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Optional

from redis.asyncio import Redis, from_url


@dataclass
class APIKeyRecord:
    key: str
    owner: str
    created_at: float
    usage_count: int = 0
    last_used: float | None = None
    revoked: bool = False
    requests_today: int = 0
    requests_day: int = 0  # YYYYMMDD for quick resets


class RateLimitError(Exception):
    """Raised when a key exceeds the configured rate limit."""


class APIKeyStore:
    """Simple JSON-backed key registry following the Text-First Storage rule."""

    def __init__(self, path: Path):
        self.path = path
        self._records: Dict[str, APIKeyRecord] = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._records = {}
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
        if isinstance(data, dict) and "keys" in data:
            data = data.get("keys", [])
        records: Dict[str, APIKeyRecord] = {}
        now = time.time()
        for item in data or []:
            record = APIKeyRecord(
                key=item.get("key"),
                owner=item.get("owner", "unknown"),
                created_at=item.get("created_at", now),
                usage_count=item.get("usage_count", 0),
                last_used=item.get("last_used"),
                revoked=item.get("revoked", False),
                requests_today=item.get("requests_today", 0),
                requests_day=item.get("requests_day", 0),
            )
            if record.key:
                records[record.key] = record
        self._records = records

    def save(self) -> None:
        with self._lock:
            payload = [asdict(record) for record in self._records.values()]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def upsert(self, record: APIKeyRecord) -> None:
        self._records[record.key] = record
        self.save()

    def get(self, key: str) -> APIKeyRecord | None:
        return self._records.get(key)

    def list(self) -> List[APIKeyRecord]:
        return sorted(self._records.values(), key=lambda item: item.created_at)


class RateLimiter:
    """Token bucket backed by Redis when available, in-memory otherwise."""

    def __init__(self, redis_url: str | None, limit_per_minute: int):
        self.redis_url = redis_url
        self.limit = max(0, limit_per_minute)
        self._client: Redis | None = None
        self._lock = Lock()
        self._buckets: Dict[str, tuple[int, int]] = {}

    async def allow(self, key: str) -> bool:
        if self.limit == 0:
            return True
        bucket = int(time.time() // 60)
        if self.redis_url:
            client = await self._get_client()
            counter_key = f"daymind:ratelimit:{key}:{bucket}"
            count = await client.incr(counter_key)
            if count == 1:
                await client.expire(counter_key, 65)
            return count <= self.limit
        with self._lock:
            previous_bucket, previous_count = self._buckets.get(key, (bucket, 0))
            if previous_bucket != bucket:
                previous_bucket, previous_count = bucket, 0
            previous_count += 1
            self._buckets[key] = (previous_bucket, previous_count)
            return previous_count <= self.limit

    async def _get_client(self) -> Redis:
        if self._client is None:
            self._client = from_url(self.redis_url, decode_responses=False)
        return self._client


class AuthService:
    """Top-level orchestrator for API key validation and usage tracking."""

    def __init__(
        self,
        store_path: Path,
        fallback_keys: Iterable[str],
        redis_url: str | None,
        rate_limit_per_minute: int,
    ):
        self.store = APIKeyStore(store_path)
        self.fallback_keys = set(fallback_keys)
        self.rate_limiter = RateLimiter(redis_url, rate_limit_per_minute)

    def _ensure_record(self, key: str, owner: str = "env") -> APIKeyRecord:
        record = self.store.get(key)
        if record is None:
            record = APIKeyRecord(key=key, owner=owner, created_at=time.time())
            self.store.upsert(record)
        return record

    async def validate_and_track(self, key: str) -> APIKeyRecord:
        record = self.store.get(key)
        if record is None and key in self.fallback_keys:
            record = self._ensure_record(key, owner="env")
        if record is None or record.revoked:
            raise ValueError("unknown_api_key")
        allowed = await self.rate_limiter.allow(key)
        if not allowed:
            raise RateLimitError("rate_limit_exceeded")
        self._record_usage(record)
        return record

    def _record_usage(self, record: APIKeyRecord) -> None:
        now = time.time()
        today = time.strftime("%Y%m%d", time.gmtime(now))
        if record.requests_day != int(today):
            record.requests_day = int(today)
            record.requests_today = 0
        record.requests_today += 1
        record.usage_count += 1
        record.last_used = now
        self.store.upsert(record)

    def usage_snapshot(self, key: str) -> Dict[str, float | int | str | None]:
        record = self.store.get(key)
        if not record:
            return {}
        return {
            "owner": record.owner,
            "created_at": record.created_at,
            "usage_count": record.usage_count,
            "requests_today": record.requests_today,
            "last_used": record.last_used,
        }

    def create_key(self, owner: str, key: str | None = None) -> APIKeyRecord:
        token = key or self._generate_key()
        record = APIKeyRecord(key=token, owner=owner, created_at=time.time())
        self.store.upsert(record)
        return record

    def revoke_key(self, key: str) -> bool:
        record = self.store.get(key)
        if not record:
            return False
        record.revoked = True
        self.store.upsert(record)
        return True

    def list_keys(self) -> List[APIKeyRecord]:
        return self.store.list()

    @staticmethod
    def _generate_key(length: int = 40) -> str:
        alphabet = string.ascii_letters + string.digits
        return "dm_" + "".join(secrets.choice(alphabet) for _ in range(length))


def build_auth_service(
    store_path: Path,
    fallback_keys: Iterable[str],
    redis_url: str | None,
    rate_limit_per_minute: int,
) -> AuthService:
    return AuthService(store_path, fallback_keys, redis_url, rate_limit_per_minute)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="DayMind API key manager")
    parser.add_argument("--store", default="data/api_keys.json", help="Path to API key JSON store")
    sub = parser.add_subparsers(dest="command", required=True)

    create_cmd = sub.add_parser("create", help="Create a new API key")
    create_cmd.add_argument("owner", help="Owner or label for the key")
    create_cmd.add_argument("--key", help="Optional custom key value")

    revoke_cmd = sub.add_parser("revoke", help="Revoke an API key")
    revoke_cmd.add_argument("key", help="Key to revoke")

    sub.add_parser("list", help="List existing keys")

    args = parser.parse_args()
    store = APIKeyStore(Path(args.store))

    if args.command == "create":
        service = AuthService(Path(args.store), [], None, 0)
        record = service.create_key(owner=args.owner, key=args.key)
        print(f"created key for {record.owner}: {record.key}")
    elif args.command == "revoke":
        service = AuthService(Path(args.store), [], None, 0)
        ok = service.revoke_key(args.key)
        if not ok:
            raise SystemExit("key not found")
        print("revoked", args.key)
    elif args.command == "list":
        for record in store.list():
            status = "revoked" if record.revoked else "active"
            print(f"{record.key}\t{record.owner}\t{status}\tused={record.usage_count}")


if __name__ == "__main__":  # pragma: no cover - manual CLI helper
    _cli()
