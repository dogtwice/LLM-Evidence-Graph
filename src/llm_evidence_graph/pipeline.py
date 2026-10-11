from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .graph import dense_adj, pair_logits, set_seed, active_seed
from .graph.trainer import APP_K, BATCH, DROP_RATE, EPOCHS, PATIENCE
from .graph.trainer import Decoder, RobustEncoder

import torch
import torch.nn.functional as F


def beta_kl_loss(a, b, A, B):
    logB_ab = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    logB_AB = torch.lgamma(A) + torch.lgamma(B) - torch.lgamma(A + B)
    sA = torch.digamma(A) - torch.digamma(A + B)
    sB = torch.digamma(B) - torch.digamma(A + B)
    return logB_ab - logB_AB + (A - a) * sA + (B - b) * sB


def contexts_of(manifests: list[dict]) -> list[dict]:
    ctx = []
    for m in manifests:
        cl = [int(c) for c in m["candidate_ids"] if int(c) != int(m["target"])]
        ctx.append({"context_id": m["context_id"], "target": int(m["target"]),
                    "candidate_ids": cl, "logical_request_id": m["context_id"]})
    return ctx


def select_by_threshold(contexts: list[dict], scored: list[dict | None],
                        tau: float, per_target_cap: int = 0) -> dict[str, set[int]]:
    if per_target_cap:
        ranked: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for ctx, sc in zip(contexts, scored):
            if not sc:
                continue
            t = int(ctx["target"])
            for c, p in sc.items():
                if c != t:
                    ranked[t].append((float(p), int(c)))
        allowed: dict[int, set[int]] = {}
        for t, items in ranked.items():
            items.sort(key=lambda kv: (-kv[0], kv[1]))
            allowed[t] = {c for p, c in items[:per_target_cap] if p >= tau}
        selected: dict[str, set[int]] = {}
        for ctx in contexts:
            t = int(ctx["target"])
            keep = allowed.get(t, set())
            selected[str(ctx["context_id"])] = {
                int(c) for c in ctx["candidate_ids"] if int(c) in keep}
        return selected

    selected = {}
    for ctx, sc in zip(contexts, scored):
        if not sc:
            selected[str(ctx["context_id"])] = set()
            continue
        t = int(ctx["target"])
        selected[str(ctx["context_id"])] = {
            int(c) for c in ctx["candidate_ids"]
            if int(c) != t and float(sc.get(int(c), 0.0)) >= tau}
    return selected


def build_targets(contexts: list[dict], scored: list[dict | None],
                  n: int, split_seed: int, train_fraction: float,
                  epsilon: float) -> dict:
    obs: dict[tuple[int, int], list[float]] = defaultdict(list)
    for ctx, sc in zip(contexts, scored):
        if not sc:
            continue
        t = int(ctx["target"])
        for c, p in sc.items():
            c = int(c)
            if c == t:
                continue
            key = (t, c) if t < c else (c, t)
            obs[key].append(float(p))
    keys = sorted(obs)
    if not keys:
        raise RuntimeError("no pair observations were collected")
    pairs = np.asarray(keys, dtype=np.int64)
    n_obs = np.asarray([len(obs[k]) for k in keys], dtype=np.float64)
    sum_p = np.asarray([sum(obs[k]) for k in keys], dtype=np.float64)
    A = epsilon + sum_p
    B = epsilon + (n_obs - sum_p)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(pairs))
    cut = int(train_fraction * len(pairs))
    tr, va = perm[:cut], perm[cut:]
    return {"pairs": pairs, "A": A, "B": B, "tr": tr, "va": va,
            "n_obs": n_obs, "sum_p": sum_p,
            "stats": {"pairs": int(len(pairs)),
                      "n_obs_max": float(n_obs.max()),
                      "n_obs_mean": float(n_obs.mean()),
                      "target_mean": float((A / (A + B)).mean())}}


def train_cell(x: np.ndarray, targets: dict, edges: np.ndarray, seed: int):
    n = x.shape[0]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trp, vap = targets["tr"], targets["va"]
    xt = torch.as_tensor(x, device=device, dtype=torch.float32)
    trp_t = torch.as_tensor(targets["pairs"][trp], device=device, dtype=torch.long)
    At = torch.as_tensor(targets["A"][trp], device=device, dtype=torch.float32)
    Bt = torch.as_tensor(targets["B"][trp], device=device, dtype=torch.float32)
    vap_t = torch.as_tensor(targets["pairs"][vap], device=device, dtype=torch.long)
    Av = torch.as_tensor(targets["A"][vap], device=device, dtype=torch.float32)
    Bv = torch.as_tensor(targets["B"][vap], device=device, dtype=torch.float32)

    enc = RobustEncoder(x.shape[1], "Full").to(device)
    dec = Decoder(evidential=True).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=1e-3, weight_decay=1e-5)
    train_edges = np.asarray(edges, dtype=np.int64)
    rng = np.random.default_rng(active_seed())
    best, state, wait, hist = None, None, 0, []
    adj_val = dense_adj(train_edges, n, device)

    def fwd_loss(z, pairs, Aa, Bb):
        out = pair_logits(dec, z, pairs)
        alpha = F.softplus(out[:, 0]) + 1.0
        beta = F.softplus(out[:, 1]) + 1.0
        return beta_kl_loss(alpha, beta, Aa, Bb).mean()

    for _ in range(EPOCHS):
        enc.train()
        dec.train()
        use = train_edges
        if len(train_edges) > 1:
            keep = rng.random(len(train_edges)) >= DROP_RATE
            if keep.sum() == 0:
                keep[rng.integers(len(train_edges))] = True
            use = train_edges[keep]
        adj_train = dense_adj(use, n, device)
        perm = torch.randperm(len(trp_t), device=device)
        total = 0.0
        for st in range(0, len(trp_t), BATCH):
            opt.zero_grad()
            z = enc(xt, adj_train)
            ix = perm[st:st + BATCH]
            loss = fwd_loss(z, trp_t[ix], At[ix], Bt[ix])
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(ix)
        enc.eval()
        dec.eval()
        with torch.no_grad():
            zv = enc(xt, adj_val)
            vl = fwd_loss(zv, vap_t, Av, Bv) if vap_t.numel() else torch.tensor(float("inf"))
        v = float(vl.cpu())
        hist.append({"epoch": len(hist) + 1, "train_loss": total / max(len(trp_t), 1),
                     "val_loss": v})
        if best is None or v < best - 1e-7:
            best, wait = v, 0
            state = {"enc": {k: t.detach().cpu().clone() for k, t in enc.state_dict().items()},
                     "dec": {k: t.detach().cpu().clone() for k, t in dec.state_dict().items()}}
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if state is not None:
        enc.load_state_dict(state["enc"])
        dec.load_state_dict(state["dec"])
    enc.eval()
    dec.eval()
    with torch.no_grad():
        z = enc(xt, adj_val).detach().cpu().numpy()
    return z, {"epochs": len(hist), "best_val_loss": best}


def dump_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
