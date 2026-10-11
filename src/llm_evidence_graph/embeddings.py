from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import EmbeddingConfig


def _placeholder(texts: list[str], cfg: EmbeddingConfig, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((len(texts), cfg.dimension), dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _sentence_transformers(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.model)
    prefixed = [cfg.instruction + t for t in texts] if cfg.instruction else list(texts)
    vecs = model.encode(prefixed, batch_size=cfg.batch_size, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def compute_embeddings(texts: list[str], cfg: EmbeddingConfig) -> np.ndarray:
    if cfg.provider == "placeholder":
        return _placeholder(texts, cfg)
    if cfg.provider == "sentence-transformers":
        return _sentence_transformers(texts, cfg)
    raise ValueError(
        f"unknown embedding provider {cfg.provider!r}; "
        f"use 'placeholder' or 'sentence-transformers'"
    )


def load_or_compute(texts: list[str], cfg: EmbeddingConfig, cache_path: str | Path) -> np.ndarray:
    p = Path(cache_path)
    if p.exists():
        x = np.load(p)
        if x.shape[0] == len(texts):
            return np.asarray(x, dtype=np.float32)
        print(f"embedding cache {p} has {x.shape[0]} rows, expected {len(texts)}; "
              f"recomputing", flush=True)
    x = compute_embeddings(texts, cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, x)
    return x
