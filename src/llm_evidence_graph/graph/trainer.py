from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 3080
HIDDEN = 128
LATENT = 64
EPOCHS = 80
PATIENCE = 12
BATCH = 8192
SEED = 201
DROP_RATE = 0.10
GC_ALPHA = 0.10
APP_K = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ACTIVE_SEED = SEED


def set_seed(seed: int = SEED) -> None:
    global _ACTIVE_SEED
    _ACTIVE_SEED = int(seed)
    np.random.seed(_ACTIVE_SEED)
    torch.manual_seed(_ACTIVE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_ACTIVE_SEED)
    torch.use_deterministic_algorithms(True)


def active_seed() -> int:
    return int(_ACTIVE_SEED)


def dense_adj(edges, n: int = N, device=DEVICE, weights=None) -> torch.Tensor:
    if len(edges) == 0:
        return torch.eye(n, device=device, dtype=torch.float32)
    e = np.asarray(edges, dtype=np.int64)
    src = e[:, 0]
    dst = e[:, 1]
    w = np.ones(len(e), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    rows = np.concatenate([src, np.arange(n, dtype=np.int64)])
    cols = np.concatenate([dst, np.arange(n, dtype=np.int64)])
    vals = np.concatenate([w, np.ones(n, dtype=np.float32)])
    deg = np.bincount(rows, weights=vals, minlength=n).astype(np.float32)
    vals = vals / np.maximum(deg[rows], 1e-8)
    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]
    starts = np.r_[0, 1 + np.flatnonzero((rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1]))]
    rows_u = rows[starts]
    cols_u = cols[starts]
    vals_u = np.add.reduceat(vals, starts).astype(np.float32, copy=False)
    a = torch.zeros((n, n), device=device, dtype=torch.float32)
    a[torch.as_tensor(rows_u, device=device), torch.as_tensor(cols_u, device=device)] = torch.as_tensor(vals_u, device=device)
    return a


def graph_mm(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, x) if a.layout == torch.strided else torch.sparse.mm(a, x)


class RobustEncoder(nn.Module):

    def __init__(self, din: int, variant: str):
        super().__init__()
        self.variant = variant
        self.gcnii = variant in ("GCNII-Res", "Full")
        self.appnp = variant in ("APPNP-GPR", "Full")
        self.pairnorm = variant in ("PairNorm", "Full")
        self.s1 = nn.Linear(din, HIDDEN)
        self.n1 = nn.Linear(din, HIDDEN, bias=False)
        self.s2 = nn.Linear(HIDDEN, LATENT)
        self.n2 = nn.Linear(HIDDEN, LATENT, bias=False)
        if self.gcnii:
            self.i1 = nn.Linear(din, HIDDEN)
            self.i2 = nn.Linear(din, LATENT)
        if self.appnp:
            p = np.asarray([0.10, 0.09, 0.081, 0.0729, 0.6561], dtype=np.float32)
            p = p / p.sum()
            self.hop_logits = nn.Parameter(torch.tensor(np.log(p), dtype=torch.float32))
        self.drop = nn.Dropout(DROP_RATE)

    def pn(self, h: torch.Tensor) -> torch.Tensor:
        if not self.pairnorm:
            return h
        h = h - h.mean(dim=0, keepdim=True)
        norm = torch.linalg.vector_norm(h)
        return h * (np.sqrt(N) / (norm + 1e-8))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        ax = graph_mm(a, x)
        h = self.s1(x) + self.n1(ax)
        if self.gcnii:
            h = (1.0 - GC_ALPHA) * h + GC_ALPHA * self.i1(x)
        h = F.relu(h)
        h = self.pn(h)
        h = self.drop(h)
        ah = graph_mm(a, h)
        z = self.s2(h) + self.n2(ah)
        if self.gcnii:
            z = (1.0 - GC_ALPHA) * z + GC_ALPHA * self.i2(x)
        z = self.pn(z)
        if self.appnp:
            states = [z]
            cur = z
            for _ in range(APP_K):
                cur = graph_mm(a, cur)
                states.append(cur)
            co = F.softmax(self.hop_logits, dim=0)
            z = sum(c * s for c, s in zip(co, states))
        return F.normalize(z, p=2, dim=1)


class Decoder(nn.Module):

    def __init__(self, evidential: bool):
        super().__init__()
        self.evidential = evidential
        self.f1 = nn.Linear(LATENT * 4, 64)
        self.f2 = nn.Linear(64, 2 if evidential else 1)

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        q = torch.cat([zi, zj, (zi - zj).abs(), zi * zj], dim=1)
        return self.f2(F.relu(self.f1(q)))


def pair_logits(dec: Decoder, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    zi = z[pairs[:, 0]]
    zj = z[pairs[:, 1]]
    return dec(zi, zj)


def _context_key(ctx) -> str:
    return str(ctx.get("context_id") or ctx.get("logical_request_id"))


def selected_edges(contexts, selected) -> np.ndarray:
    edges = np.asarray(
        [(int(ctx["target"]), int(j))
         for ctx in contexts
         for j in sorted(selected.get(_context_key(ctx), set()))],
        dtype=np.int64,
    )
    if len(edges) == 0:
        return edges.reshape(0, 2)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]
