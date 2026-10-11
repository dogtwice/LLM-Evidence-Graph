from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


@dataclass
class APIConfig:
    base_url: str = "https://your-endpoint.example.com/v1"
    api_key_env: str = "LLM_API_KEY"
    model: str = "your-model-id"
    rpm_limit: int = 500
    tpm_limit: int = 2_000_000
    rpm_utilization: float = 0.99
    tpm_utilization: float = 0.99
    retry_guard: float = 1.0
    concurrency: int = 64
    timeout_seconds: float = 180.0
    retry_limit: int = 2
    max_tokens: int = 900
    temperature: float = 0.0
    pass_seed: bool = False

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"environment variable {self.api_key_env} is empty; "
                f"copy .env.example to .env and fill in your key"
            )
        return key


@dataclass
class EmbeddingConfig:
    provider: str = "placeholder"
    model: str = "your-embedding-model"
    dimension: int = 384
    batch_size: int = 256
    cache: str = "data/bank77/embeddings.npy"
    instruction: str = ""


@dataclass
class RetrievalConfig:
    pool_depth: int = 6
    candidates_per_block: int = 10

    @property
    def total_candidates(self) -> int:
        return self.pool_depth * self.candidates_per_block

    @property
    def requests_per_target(self) -> int:
        return self.pool_depth


@dataclass
class GraphConfig:
    tau: float = 0.4
    per_target_edge_cap: int = 0
    beta_prior_epsilon: float = 1.0
    split_seed: int = 0
    train_fraction: float = 0.9
    encoder_variant: str = "Full"


@dataclass
class TrainConfig:
    seeds: list[int] = field(default_factory=lambda: [201, 202, 203, 204, 205])
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 80
    patience: int = 12


@dataclass
class ClusterConfig:
    algorithm: str = "kmeans"
    n_init: int = 10
    max_iter: int = 300
    random_state: int = 1
    eval_seeds: list[int] = field(default_factory=lambda: [1])


@dataclass
class Config:
    dataset: str = "bank77"
    data_dir: str = "data/bank77"
    prompt_file: str = "prompts/bank77.txt"
    cache_dir: str = "cache"
    api: APIConfig = field(default_factory=APIConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


def _build(cls, raw: Any):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"expected mapping for {cls.__name__}, got {type(raw).__name__}")
    names = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {f.name for f in Config.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"unknown top-level keys in {path}: {sorted(unknown)}")
    return Config(
        dataset=raw.get("dataset", "bank77"),
        data_dir=raw.get("data_dir", "data/bank77"),
        prompt_file=raw.get("prompt_file", "prompts/bank77.txt"),
        cache_dir=raw.get("cache_dir", "cache"),
        api=_build(APIConfig, raw.get("api")),
        embedding=_build(EmbeddingConfig, raw.get("embedding")),
        retrieval=_build(RetrievalConfig, raw.get("retrieval")),
        graph=_build(GraphConfig, raw.get("graph")),
        train=_build(TrainConfig, raw.get("train")),
        cluster=_build(ClusterConfig, raw.get("cluster")),
    )
