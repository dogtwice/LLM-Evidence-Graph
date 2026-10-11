from __future__ import annotations

import numpy as np
import torch


def topk_cosine(query: np.ndarray, reference: np.ndarray, k: int,
                exclude_self: bool = False) -> np.ndarray:
    q = torch.as_tensor(np.ascontiguousarray(np.asarray(query, dtype=np.float32)))
    r = torch.as_tensor(np.ascontiguousarray(np.asarray(reference, dtype=np.float32)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = q.to(device)
    r = r.to(device)
    scores = q @ r.T
    if exclude_self:
        scores.fill_diagonal_(-torch.inf)
    return torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices.cpu().numpy()


def build_manifests(x: np.ndarray, name: str, pool_depth: int,
                    candidates_per_block: int) -> list[dict]:
    n = x.shape[0]
    total = pool_depth * candidates_per_block
    top = topk_cosine(x, x, min(total, n - 1), exclude_self=True)
    prefix = f"{name}_"
    mans: list[dict] = []
    for i in range(n):
        cands = [int(j) for j in top[i]]
        for b in range(pool_depth):
            block = cands[b * candidates_per_block:(b + 1) * candidates_per_block]
            if block:
                mans.append({"context_id": f"{prefix}t{i:04d}_b{b}", "target": i,
                             "candidate_ids": block, "form": "cand10xB", "block": b})
    return mans
