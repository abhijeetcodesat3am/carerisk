"""Benchmark LR / RF / XGBoost, track in MLflow, register the champion, explain it and audit fairness.

Protocol
  1. Every candidate config is fit on TRAIN and scored on VAL (nested MLflow runs).
  2. The best config per model family (by VAL AUPRC) is evaluated once on TEST, with bootstrap
     95% CIs and a paired bootstrap delta versus the logistic-regression baseline.
  3. The family with the best VAL AUPRC becomes the champion -> registered, alias `champion`.
"""
from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from . import explain, fairness, tracking  # noqa: E402
from .config import reports_dir  # noqa: E402
from .features import FEATURES, LABEL  # noqa: E402

# numpy + Apple Accelerate emit spurious matmul overflow warnings on recent macOS; results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.utils.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model")

BASELINE = "logreg_baseline"
CANDIDATES = {
    BASELINE: [{"C": 1.0}, {"C": 0.1}],
    "random_forest": [
        {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"n_estimators": 300, "min_samples_leaf": 50, "max_features": 0.5},
    ],
    "xgboost": [
        {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
        {"max_depth": 5, "learning_rate": 0.03, "min_child_weight": 10},
    ],
}


def make_model(family: str, params: dict, seed: int = 42) -> Pipeline:
    if family == BASELINE:
        return Pipeline([
            ("prep", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])),
            ("clf", LogisticRegression(max_iter=3000, random_state=seed, **params)),
        ])
    if family == "random_forest":
        return Pipeline([
            ("prep", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_jobs=-1, random_state=seed, **params)),
        ])
    if family == "xgboost":  # handles missing values natively
        return Pipeline([
            ("prep", "passthrough"),
            ("clf", XGBClassifier(n_estimators=2000, subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="aucpr", early_stopping_rounds=100, tree_method="hist",
                                  random_state=seed, n_jobs=-1, **params)),
        ])
    raise ValueError(family)


def fit(model: Pipeline, family: str, Xtr, ytr, Xva, yva) -> Pipeline:
    if family == "xgboost":
        return model.fit(Xtr, ytr, clf__eval_set=[(Xva, yva)], clf__verbose=False)
    return model.fit(Xtr, ytr)


def metrics(y, p) -> dict:
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "prevalence": float(np.mean(y))}


def bootstrap(y, p, groups, p_base=None, n=500, seed=0) -> dict:
    """Patient-clustered bootstrap CIs (resample patients, keep all their stays);
    if p_base is given, also paired CIs for the delta vs the baseline."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx_by_patient = list(pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices.values())
    res = {k: [] for k in ("auroc", "auprc", "d_auroc", "d_auprc")}
    for _ in range(n):
        pick = rng.integers(0, len(idx_by_patient), len(idx_by_patient))
        i = np.concatenate([idx_by_patient[j] for j in pick])
        if len(np.unique(y[i])) < 2:
            continue
        a, b = roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i])
        res["auroc"].append(a); res["auprc"].append(b)
        if p_base is not None:
            res["d_auroc"].append(a - roc_auc_score(y[i], p_base[i]))
            res["d_auprc"].append(b - average_precision_score(y[i], p_base[i]))
    return {f"{k}_ci": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in res.items() if v}


def _plots(y, preds: dict[str, np.ndarray], out: Path) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})")
        ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y, p):.3f})")
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[2].plot(mean_p, frac, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR")
    ax[1].axhline(np.mean(y), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-Recall (test)", xlabel="Recall", ylabel="Precision")
    lim = max(max(p.max() for p in preds.values()), np.mean(y) * 2)
    ax[2].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[2].set(title="Calibration (test)", xlabel="Mean predicted", ylabel="Observed rate")
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    p = out / "test_curves.png"; fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)
    return paths


def run(store, cfg: dict) -> dict:
    model_name = tracking.setup(cfg)
    df = store.read("gold_features").where("split IS NOT NULL").toPandas()
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    X = {s: d[FEATURES].astype(float) for s, d in parts.items()}
    y = {s: d[LABEL].to_numpy() for s, d in parts.items()}
    # Guard: patient-grouped split must have zero patient overlap.
    ids = {s: set(d["DESYNPUF_ID"]) for s, d in parts.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    out = reports_dir(cfg)
    with mlflow.start_run(run_name="benchmark") as parent:
        mlflow.log_params({"n_features": len(FEATURES), **{f"n_{s}": len(d) for s, d in parts.items()},
                           **{f"readmit_rate_{s}": round(float(y[s].mean()), 4) for s in parts}})
        best: dict[str, dict] = {}
        for family, grid in CANDIDATES.items():
            for i, params in enumerate(grid):
                with mlflow.start_run(run_name=f"{family}_{i}", nested=True) as r:
                    model = fit(make_model(family, params), family, X["train"], y["train"], X["val"], y["val"])
                    m = metrics(y["val"], model.predict_proba(X["val"])[:, 1])
                    mlflow.log_params({"family": family, **params})
                    mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
                    if family == "xgboost":
                        mlflow.log_metric("best_iteration", model[-1].best_iteration)
                    print(f"  {family:16s} cfg{i}  val AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}")
                    if family not in best or m["auprc"] > best[family]["val"]["auprc"]:
                        best[family] = {"model": model, "params": params, "val": m, "run_id": r.info.run_id}

        # --- one-shot test evaluation of the best config per family
        p_test = {f: b["model"].predict_proba(X["test"])[:, 1] for f, b in best.items()}
        rows = []
        for f, b in best.items():
            tm = metrics(y["test"], p_test[f])
            ci = bootstrap(y["test"], p_test[f], parts["test"]["DESYNPUF_ID"], None if f == BASELINE else p_test[BASELINE])
            b["test"] = tm
            rows.append({"model": f, **{f"val_{k}": v for k, v in b["val"].items() if k != "prevalence"},
                         **{f"test_{k}": v for k, v in tm.items()}, **{k: str(tuple(round(x, 4) for x in v))
                                                                       for k, v in ci.items()}})
            with mlflow.start_run(run_id=b["run_id"], nested=True):
                mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        bench = pd.DataFrame(rows).sort_values("val_auprc", ascending=False)
        base_test = best[BASELINE]["test"]
        bench["test_auroc_lift_vs_lr"] = bench["test_auroc"] - base_test["auroc"]
        bench["test_auprc_lift_vs_lr"] = bench["test_auprc"] - base_test["auprc"]
        bench.to_csv(out / "benchmark.csv", index=False)
        mlflow.log_artifact(str(out / "benchmark.csv"))
        for p in _plots(y["test"], p_test, out):
            mlflow.log_artifact(str(p))

        champ_family = max(best, key=lambda f: best[f]["val"]["auprc"])
        champ = best[champ_family]
        model = champ["model"]

        # --- capacity-based risk tiers fixed on VAL
        p_val = model.predict_proba(X["val"])[:, 1]
        t = cfg["risk_tiers"]
        thr_high = float(np.quantile(p_val, 1 - t["high_top_frac"]))
        thr_med = float(np.quantile(p_val, 1 - t["medium_top_frac"]))

        # --- SHAP (global on a test sample) for the champion
        sample = X["test"].sample(min(2000, len(X["test"])), random_state=0)
        sv = explain.shap_values(model, sample, background=X["train"].sample(min(500, len(X["train"])),
                                                                             random_state=0))
        imp = explain.global_importance(sv, sample.columns)
        imp.to_csv(out / "shap_importance.csv", index=False)
        explain.summary_plot(sv, sample, out / "shap_summary.png")

        # --- fairness audit on TEST at the high-risk threshold
        test_df = parts["test"].assign(score=p_test[champ_family])
        fair = fairness.audit(test_df, "score", LABEL, thr_high)
        fair.to_csv(out / "fairness_audit.csv", index=False)
        _fairness_plot(fair, out / "fairness_audit.png")

        # --- log + register champion
        with mlflow.start_run(run_name=f"champion_{champ_family}", nested=True) as cr:
            mlflow.log_params({"family": champ_family, **champ["params"],
                               "threshold_high": thr_high, "threshold_medium": thr_med})
            mlflow.log_metrics({**{f"val_{k}": v for k, v in champ["val"].items()},
                                **{f"test_{k}": v for k, v in champ["test"].items()}})
            for f in ("shap_importance.csv", "shap_summary.png", "fairness_audit.csv", "fairness_audit.png"):
                mlflow.log_artifact(str(out / f))
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "features.json").write_text(json.dumps(FEATURES))
                mlflow.log_artifact(str(Path(tmp) / "features.json"))
            sig = infer_signature(X["val"].head(100), model.predict_proba(X["val"].head(100))[:, 1])
            mlflow.sklearn.log_model(model, "model", signature=sig, input_example=X["val"].head(3))
            champ_run = cr.info.run_id
        mv = mlflow.register_model(f"runs:/{champ_run}/model", model_name)
        client = MlflowClient()
        tags = {"model_family": champ_family, "features": json.dumps(FEATURES),
                "threshold_high": str(thr_high), "threshold_medium": str(thr_med),
                "val_auprc": f"{champ['val']['auprc']:.4f}", "test_auprc": f"{champ['test']['auprc']:.4f}",
                "test_auroc": f"{champ['test']['auroc']:.4f}", "benchmark_run_id": parent.info.run_id}
        for k, v in tags.items():
            client.set_model_version_tag(model_name, mv.version, k, v)
        client.set_registered_model_alias(model_name, cfg["mlflow"]["champion_alias"], mv.version)

    return {"champion": champ_family, "model": model_name, "version": mv.version,
            "threshold_high": thr_high, "benchmark": bench, "fairness": fair, "shap_top": imp.head(10)}


def _fairness_plot(fair: pd.DataFrame, path: Path) -> None:
    sub = fair[fair["attribute"] != "overall"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    labels = sub["attribute"] + ": " + sub["group"]
    for a, col, title in zip(ax, ["auroc", "tpr", "calibration_ratio"],
                             ["AUROC by group", "TPR at high-risk threshold", "Calibration (pred / obs)"]):
        a.barh(labels, sub[col]); a.set_title(title)
        ref = fair.loc[fair["attribute"] == "overall", col].iloc[0]
        a.axvline(1.0 if col == "calibration_ratio" else ref, color="k", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
