from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits

THREADS = 4


def l2_normalize(x: np.ndarray) -> np.ndarray:
    h = np.asarray(x, dtype=np.float64)
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)


def hungarian_accuracy(pred: np.ndarray, labels: np.ndarray, k: int) -> float:
    pred = np.asarray(pred, dtype=np.int64).ravel()
    m = max(k, int(pred.max()) + 1, int(labels.max()) + 1)
    cm = np.zeros((m, m), dtype=np.int64)
    np.add.at(cm, (pred, labels), 1)
    rr, cc = linear_sum_assignment(-cm)
    return float(cm[rr, cc].sum() / len(labels))


def evaluate(z: np.ndarray, labels: np.ndarray, k: int,
             seeds: list[int] | tuple[int, ...] = (1,)) -> dict:
    h = l2_normalize(z)
    accs, nmis = [], []
    for s in seeds:
        with threadpool_limits(limits=THREADS):
            pred = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300,
                          random_state=int(s), algorithm="lloyd").fit_predict(h)
        accs.append(hungarian_accuracy(pred, labels, k))
        nmis.append(float(normalized_mutual_info_score(labels, pred)))
    a, n = np.asarray(accs), np.asarray(nmis)
    return {"acc": round(float(a.mean()), 6),
            "acc_std_pp": round(float(a.std(ddof=0)) * 100, 3),
            "nmi": round(float(n.mean()), 6),
            "nmi_std_pp": round(float(n.std(ddof=0)) * 100, 3),
            "acc_all": [round(float(v), 6) for v in a],
            "nmi_all": [round(float(v), 6) for v in n]}


def summarize(rows: list[dict]) -> dict:
    acc = np.asarray([r["acc"] for r in rows])
    nmi = np.asarray([r["nmi"] for r in rows])
    return {"n_seeds": len(rows),
            "acc": round(float(acc.mean()), 6),
            "acc_std_pp": round(float(acc.std(ddof=0)) * 100, 3),
            "nmi": round(float(nmi.mean()), 6),
            "nmi_std_pp": round(float(nmi.std(ddof=0)) * 100, 3)}
