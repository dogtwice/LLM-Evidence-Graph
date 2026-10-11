from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class SchedulerConfig:
    base_url: str
    api_key: str
    model: str
    rpm_limit: int = 500
    tpm_limit: int = 2_000_000
    rpm_utilization: float = 0.80
    tpm_utilization: float = 0.85
    retry_guard: float = 1.20
    concurrency_cap: int = 64
    timeout_seconds: float = 120.0
    retry_limit: int = 2
    max_tokens: int = 512
    max_connections: int | None = None
    """httpx connection-pool limit. None now means `concurrency_cap`, not httpx's
    ~100 default.

    httpx caps the pool at ~100 sockets, independently of this scheduler's
    semaphore, so a runner asking for 800 concurrency but leaving this unset gets
    800 permits queued behind 100 connections. Measured cost of that omission
    (S7.50, E4B, identical payloads): 8.8 RPS, against 22.3 RPS once the pool was
    raised. Defaulting to `concurrency_cap` makes the pool exactly as wide as the
    semaphore allows in flight, so it can never be the binding constraint -- and
    older runners that predate this field (s731/s739/s740/s743/s744, all of which
    omit it) get the fix without being edited. Set an explicit value only to
    deliberately constrain sockets below the concurrency cap."""


class _Bucket:
    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = float(rate), float(capacity)
        self.tokens, self.last = float(capacity), time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, amount: float) -> None:
        if not self.lock.locked():
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= amount:
                self.tokens -= amount
                return
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await asyncio.sleep((amount - self.tokens) / self.rate)


class AsyncLLMScheduler:

    def __init__(self, config: SchedulerConfig, checkpoint: str | Path):
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        rpm = config.rpm_limit * config.rpm_utilization / config.retry_guard / 60.0
        tpm = config.tpm_limit * config.tpm_utilization / config.retry_guard / 60.0
        reserve = max(1, config.max_tokens)
        self.req_bucket = _Bucket(rpm, max(1.0, rpm))
        self.tok_bucket = _Bucket(tpm, max(float(reserve), tpm * 20.0))
        self.semaphore = asyncio.Semaphore(config.concurrency_cap)

    @staticmethod
    def request_id(payload: dict[str, Any]) -> str:
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()

    def _completed(self) -> dict[str, dict[str, Any]]:
        if not self.checkpoint.exists():
            return {}
        out = {}
        for line in self.checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("ok"):
                    out[str(row["request_id"])] = row
        return out

    async def _one(self, client: httpx.AsyncClient, payload: dict[str, Any]) -> dict[str, Any]:
        rid = self.request_id(payload)
        prompt_tokens = max(1, int(payload.get("prompt_tokens_est", 1)))
        reservation = prompt_tokens + self.config.max_tokens
        for attempt in range(self.config.retry_limit + 1):
            await self.req_bucket.acquire(1.0)
            await self.tok_bucket.acquire(float(reservation))
            try:
                response = await client.post(f"{self.config.base_url.rstrip('/')}/chat/completions",
                                             headers={"Authorization": f"Bearer {self.config.api_key}"},
                                             json={"model": self.config.model, **payload},
                                             timeout=self.config.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                return {"request_id": rid, "ok": True, "attempt": attempt, "response": data}
            except Exception as exc:
                if attempt >= self.config.retry_limit:
                    return {"request_id": rid, "ok": False, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
                await asyncio.sleep(min(30.0, 2.0 ** attempt))
        raise AssertionError("unreachable")

    async def run(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed = self._completed()
        pending = [p for p in payloads if self.request_id(p) not in completed]
        writer_lock = asyncio.Lock()
        pool = self.config.max_connections or self.config.concurrency_cap


        client_ctx = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=pool, max_keepalive_connections=100),
            trust_env=False)


        with self.checkpoint.open("a", encoding="utf-8") as handle:
            async with client_ctx as client:
                async def wrapped(payload: dict[str, Any]):
                    async with self.semaphore:
                        row = await self._one(client, payload)
                        async with writer_lock:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                        return row
                fresh = await asyncio.gather(*(wrapped(p) for p in pending))


        by_id = {row["request_id"]: row for row in fresh}
        out: list[dict[str, Any]] = []
        for payload in payloads:
            rid = self.request_id(payload)
            row = completed.get(rid) or by_id.get(rid)
            if row is None:
                raise KeyError(f"no result for request {rid}")
            out.append(row)
        return out
