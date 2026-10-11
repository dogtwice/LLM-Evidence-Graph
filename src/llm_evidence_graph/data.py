from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    name: str
    texts: list[str]
    labels: np.ndarray
    k: int

    def __len__(self) -> int:
        return len(self.texts)


def _read_texts(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw["texts"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON list of strings")
    out = [str(t) for t in raw]
    if any(not t.strip() for t in out):
        raise ValueError(f"{path}: contains an empty text")
    return out


def load_dataset(data_dir: str | Path, name: str | None = None) -> Dataset:
    d = Path(data_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {d}")
    texts = _read_texts(d / "texts.json")
    labels = np.load(d / "labels.npy")
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if labels.shape[0] != len(texts):
        raise ValueError(f"labels {labels.shape} does not match {len(texts)} texts")
    unique = np.unique(labels)
    if unique[0] != 0 or unique[-1] != unique.size - 1:
        remap = {int(v): i for i, v in enumerate(unique.tolist())}
        labels = np.asarray([remap[int(v)] for v in labels.tolist()], dtype=np.int64)
    return Dataset(name=name or d.name, texts=texts, labels=labels,
                   k=int(np.unique(labels).size))
