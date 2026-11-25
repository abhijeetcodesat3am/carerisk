"""Population Stability Index drift monitoring (reference = training population)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(ref, cur, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI = sum((cur% - ref%) * ln(cur% / ref%)). Quantile bins from the reference;
    low-cardinality features use their categories; missing values get their own bin."""
    ref, cur = np.asarray(ref, dtype=float), np.asarray(cur, dtype=float)
    r_nan, c_nan = np.isnan(ref).mean(), np.isnan(cur).mean()
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    uniq = np.unique(ref)
    if len(uniq) <= bins:
        r = np.array([(ref == u).mean() for u in uniq])
        c = np.array([(cur == u).mean() for u in uniq])
        r, c = np.append(r, 0.0), np.append(c, 1.0 - c.sum())        # values unseen in reference
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
        c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r = np.append(r * (1 - r_nan), r_nan)
    c = np.append(c * (1 - c_nan), c_nan)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float, warn: float, alert: float) -> str:
    if np.isnan(value):
        return "n/a"
    return "alert" if value >= alert else "warn" if value >= warn else "stable"


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, columns, cfg: dict) -> pd.DataFrame:
    d = cfg["drift"]
    rows = []
    for c in columns:
        v = psi(ref[c], cur[c], bins=d["bins"])
        rows.append({"feature": c, "psi": v, "status": status(v, d["warn"], d["alert"]),
                     "ref_mean": float(ref[c].mean()), "cur_mean": float(cur[c].mean())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
