from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .client import fetch, read_checkpoint
from .clustering import evaluate, summarize
from .config import Config, load_config
from .data import load_dataset
from .embeddings import load_or_compute
from .graph import selected_edges
from .pipeline import build_targets, contexts_of, select_by_threshold, train_cell
from .prompts import build_payload, load_system_prompt, parse_ratings
from .retrieval import build_manifests
from .llm.scheduler import AsyncLLMScheduler


def _cache(cfg: Config, *parts: str) -> Path:
    p = cfg.path(cfg.cache_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_embed(cfg: Config) -> int:
    ds = load_dataset(cfg.path(cfg.data_dir), cfg.dataset)
    x = load_or_compute(ds.texts, cfg.embedding, cfg.path(cfg.embedding.cache))
    print(f"{ds.name}: {len(ds)} texts, K={ds.k}, embedding {x.shape}", flush=True)
    return 0


def cmd_manifest(cfg: Config) -> int:
    ds = load_dataset(cfg.path(cfg.data_dir), cfg.dataset)
    x = load_or_compute(ds.texts, cfg.embedding, cfg.path(cfg.embedding.cache))
    mans = build_manifests(x, ds.name, cfg.retrieval.pool_depth,
                           cfg.retrieval.candidates_per_block)
    out = _cache(cfg, "manifests.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for m in mans:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"{ds.name}: {len(mans)} requests "
          f"({cfg.retrieval.requests_per_target} per target x {len(ds)} targets)",
          flush=True)
    print(f"written to {out}", flush=True)
    return 0


def _load_manifests(cfg: Config) -> list[dict]:
    p = _cache(cfg, "manifests.jsonl")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `leg manifest` first")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _payloads(cfg: Config, ds, mans: list[dict], seed: int) -> list[dict]:
    system = load_system_prompt(cfg.path(cfg.prompt_file))
    sampling_seed = seed if cfg.api.pass_seed else None
    return [build_payload(system, ds.texts, int(m["target"]),
                          [int(c) for c in m["candidate_ids"]],
                          cfg.api.max_tokens, cfg.api.temperature,
                          sampling_seed) for m in mans]


def cmd_request(cfg: Config, seed: int) -> int:
    ds = load_dataset(cfg.path(cfg.data_dir), cfg.dataset)
    mans = _load_manifests(cfg)
    payloads = _payloads(cfg, ds, mans, seed)
    ck = _cache(cfg, f"raw_seed{seed}.jsonl")
    fetch(payloads, cfg.api, ck)
    rows = read_checkpoint(ck)
    ok = sum(1 for r in rows.values() if r.get("ok"))
    print(f"{ds.name} seed {seed}: {ok}/{len(payloads)} requests ok, checkpoint {ck}",
          flush=True)
    if ok < len(payloads):
        print(f"WARNING: seed {seed}: {len(payloads) - ok} requests missing or failed",
              file=sys.stderr)
        return 1
    return 0


def cmd_train(cfg: Config, seed: int) -> int:
    ds = load_dataset(cfg.path(cfg.data_dir), cfg.dataset)
    x = load_or_compute(ds.texts, cfg.embedding, cfg.path(cfg.embedding.cache))
    mans = _load_manifests(cfg)
    ck = _cache(cfg, f"raw_seed{seed}.jsonl")
    rows = read_checkpoint(ck)

    scored: list[dict | None] = []
    missing = 0
    payloads = _payloads(cfg, ds, mans, seed)
    for m, payload in zip(mans, payloads):
        rid = AsyncLLMScheduler.request_id(payload)
        row = rows.get(rid)
        sc = None
        if row is not None and row.get("ok"):
            sc = parse_ratings(row, [int(c) for c in m["candidate_ids"]])
        if sc is None:
            missing += 1
        scored.append(sc)
    if missing:
        raise RuntimeError(f"seed {seed}: {missing}/{len(mans)} responses unavailable; "
                           f"run `leg request --seed {seed}` first")

    contexts = contexts_of(mans)
    selected = select_by_threshold(contexts, scored, cfg.graph.tau,
                                   cfg.graph.per_target_edge_cap)
    edges = selected_edges(contexts, selected)
    targets = build_targets(contexts, scored, len(ds), cfg.graph.split_seed,
                            cfg.graph.train_fraction, cfg.graph.beta_prior_epsilon)
    print(f"{ds.name} seed {seed}: {len(contexts)} contexts, {len(edges)} directed edges, "
          f"{targets['stats']['pairs']} pairs "
          f"(tau={cfg.graph.tau}, cap={cfg.graph.per_target_edge_cap})", flush=True)

    t0 = time.perf_counter()
    z, info = train_cell(x, targets, edges, seed)
    np.save(_cache(cfg, f"z_seed{seed}.npy"), z)
    metrics = evaluate(z, ds.labels, ds.k, cfg.cluster.eval_seeds)
    rec = {"dataset": ds.name, "seed": seed, "N": len(ds), "K": ds.k,
           "edges": int(len(edges)), "pairs": targets["stats"]["pairs"],
           "tau": cfg.graph.tau, "cap": cfg.graph.per_target_edge_cap,
           "epochs": info["epochs"], "best_val_loss": info["best_val_loss"],
           **metrics, "seconds": round(time.perf_counter() - t0, 2)}
    _write_json(_cache(cfg, f"results_seed{seed}.json"), rec)
    print(f"  seed {seed}: ACC={rec['acc']:.6f} NMI={rec['nmi']:.6f} "
          f"epochs={info['epochs']} ({rec['seconds']:.1f}s)", flush=True)
    return 0


def cmd_summary(cfg: Config) -> int:
    rows = []
    for seed in cfg.train.seeds:
        p = _cache(cfg, f"results_seed{seed}.json")
        if p.exists():
            rows.append(_read_json(p))
    if not rows:
        raise FileNotFoundError(f"no results_seed*.json under {cfg.path(cfg.cache_dir)}")
    s = summarize(rows)
    print(f"{cfg.dataset}: {s['n_seeds']} seeds, "
          f"ACC={s['acc']:.6f} (+/-{s['acc_std_pp']:.3f}pp), "
          f"NMI={s['nmi']:.6f} (+/-{s['nmi_std_pp']:.3f}pp)", flush=True)
    for r in rows:
        print(f"  seed {r['seed']}: ACC={r['acc']:.6f} NMI={r['nmi']:.6f}", flush=True)
    _write_json(_cache(cfg, "summary.json"), s)
    return 0


def cmd_run(cfg: Config) -> int:
    rc = cmd_embed(cfg)
    if rc:
        return rc
    rc = cmd_manifest(cfg)
    if rc:
        return rc
    for seed in cfg.train.seeds:
        rc = cmd_request(cfg, seed)
        if rc:
            return rc
        rc = cmd_train(cfg, seed)
        if rc:
            return rc
    return cmd_summary(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="leg",
        description="LLM-Evidence-Graph: retrieve -> judge -> train -> cluster")
    ap.add_argument("--config", default=None,
                    help="path to a YAML config (default: configs/default.yaml)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("embed", help="compute or load the frozen text embeddings")
    sub.add_parser("manifest", help="build the retrieval manifests (the request list)")
    for name, helptext in (("request", "query the judge for one seed's experiment"),
                           ("train", "train, cluster and evaluate one seed's experiment")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--seed", type=int, default=None,
                       help="experiment seed (default: every seed in the config)")
    sub.add_parser("summary", help="aggregate the per-seed results")
    sub.add_parser("run", help="embed -> manifest -> per-seed judge and train")
    a = ap.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()

    if a.command in ("request", "train"):
        seeds = [a.seed] if a.seed is not None else list(cfg.train.seeds)
        for seed in seeds:
            rc = cmd_request(cfg, seed) if a.command == "request" else cmd_train(cfg, seed)
            if rc:
                return rc
        return 0
    handlers = {"embed": cmd_embed, "manifest": cmd_manifest,
                "summary": cmd_summary, "run": cmd_run}
    return handlers[a.command](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
