from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .config import APIConfig
from .llm.scheduler import AsyncLLMScheduler, SchedulerConfig


def sched_config(cfg: APIConfig) -> SchedulerConfig:
    return SchedulerConfig(
        base_url=cfg.base_url,
        api_key=cfg.api_key(),
        model=cfg.model,
        rpm_limit=cfg.rpm_limit,
        tpm_limit=cfg.tpm_limit,
        rpm_utilization=cfg.rpm_utilization,
        tpm_utilization=cfg.tpm_utilization,
        retry_guard=cfg.retry_guard,
        concurrency_cap=cfg.concurrency,
        timeout_seconds=cfg.timeout_seconds,
        retry_limit=cfg.retry_limit,
        max_tokens=cfg.max_tokens,
        max_connections=cfg.concurrency + 64,
    )


def read_checkpoint(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("request_id"))] = row
    return out


def fetch(payloads: list[dict], cfg: APIConfig, checkpoint: str | Path) -> dict[str, dict]:
    sched = AsyncLLMScheduler(sched_config(cfg), checkpoint)
    ids = [AsyncLLMScheduler.request_id(p) for p in payloads]
    done = {rid for rid, row in read_checkpoint(checkpoint).items() if row.get("ok")}
    todo = [p for p, rid in zip(payloads, ids) if rid not in done]
    if todo:
        print(f"requests: {len(payloads)} total, {len(done)} cached, {len(todo)} to fetch",
              flush=True)
        t0 = time.perf_counter()
        asyncio.run(sched.run(todo))
        dt = time.perf_counter() - t0
        print(f"fetched {len(todo)} in {dt:.1f}s ({len(todo) / max(dt, 1e-9):.1f} req/s)",
              flush=True)
    else:
        print(f"requests: {len(payloads)} total, all cached", flush=True)
    return read_checkpoint(checkpoint)
