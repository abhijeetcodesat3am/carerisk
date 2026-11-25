"""SHAP explanations: global importance and per-patient reason codes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> np.ndarray:
    """SHAP values (n_rows, n_features) for the positive class of a fitted sklearn Pipeline(prep, clf)."""
    prep, clf = model[:-1], model[-1]
    Xt = pd.DataFrame(np.asarray(prep.transform(X), dtype=float), columns=X.columns, index=X.index)
    if isinstance(clf, LogisticRegression):
        bg = Xt if background is None else pd.DataFrame(
            np.asarray(prep.transform(background), dtype=float), columns=X.columns)
        sv = shap.LinearExplainer(clf, bg).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(clf).shap_values(Xt, check_additivity=False)
    if isinstance(sv, list):          # older shap: list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, f, classes)
        sv = sv[:, :, 1]
    return sv


def reason_codes(sv: np.ndarray, X: pd.DataFrame, k: int = 3, min_shap: float = 0.01) -> list[list[dict]]:
    """Top-k features pushing each patient's risk *up* (by at least `min_shap`), with the raw value."""
    cols = np.array(X.columns)
    order = np.argsort(-sv, axis=1)[:, :k]
    vals = X.to_numpy()
    out = []
    for i, idx in enumerate(order):
        out.append([
            {"feature": cols[j], "value": None if pd.isna(vals[i, j]) else float(vals[i, j]),
             "shap": float(sv[i, j])}
            for j in idx if sv[i, j] >= min_shap
        ])
    return out


def format_reason(r: dict) -> str:
    v = r["value"]
    v = "NA" if v is None else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return f"{r['feature']}={v} (+{r['shap']:.3f})"


def global_importance(sv: np.ndarray, columns) -> pd.DataFrame:
    return (pd.DataFrame({"feature": list(columns), "mean_abs_shap": np.abs(sv).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def summary_plot(sv: np.ndarray, X: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(sv, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close("all")
