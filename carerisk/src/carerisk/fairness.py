"""Subgroup performance / fairness audit across age band and sex."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe(fn, y, p):
    return float(fn(y, p)) if len(np.unique(y)) == 2 else np.nan


def group_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flag = p >= threshold
    pos, neg = y == 1, y == 0
    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else np.nan,
        "mean_score": float(p.mean()) if len(p) else np.nan,
        "auroc": _safe(roc_auc_score, y, p),
        "auprc": _safe(average_precision_score, y, p),
        "selection_rate": float(flag.mean()) if len(y) else np.nan,
        "tpr": float(flag[pos].mean()) if pos.any() else np.nan,   # equal opportunity
        "fpr": float(flag[neg].mean()) if neg.any() else np.nan,
        "ppv": float(y[flag].mean()) if flag.any() else np.nan,
    }


def audit(df: pd.DataFrame, score_col: str, label_col: str, threshold: float,
          attributes=("age_band", "sex"), min_n: int = 30) -> pd.DataFrame:
    y_all, p_all = df[label_col].to_numpy(), df[score_col].to_numpy()
    rows = [{"attribute": "overall", "group": "all", **group_metrics(y_all, p_all, threshold)}]
    for attr in attributes:
        for g, sub in df.groupby(attr, observed=True):
            rows.append({"attribute": attr, "group": str(g),
                         **group_metrics(sub[label_col].to_numpy(), sub[score_col].to_numpy(), threshold)})
    out = pd.DataFrame(rows)
    out["calibration_ratio"] = out["mean_score"] / out["prevalence"]        # 1.0 = well calibrated
    for attr in attributes:
        m = out["attribute"] == attr
        best_tpr = out.loc[m, "tpr"].max()
        out.loc[m, "tpr_ratio"] = out.loc[m, "tpr"] / best_tpr
        out.loc[m, "auroc_gap"] = out.loc[m, "auroc"] - out.loc[m, "auroc"].max()
    out["flag"] = np.where(
        (out["attribute"] != "overall") & (out["n"] >= min_n)
        & ((out["tpr_ratio"] < 0.8) | (out["auroc_gap"] < -0.05)
           | ((out["calibration_ratio"] - 1).abs() > 0.2)),
        "review", "ok")
    out.loc[out["n"] < min_n, "flag"] = "small_n"
    return out
