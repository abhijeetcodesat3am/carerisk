"""Entry point: `python -m carerisk.pipeline [ingest|features|train|score|all]` (also used by the notebooks)."""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from . import features, ingest, score, train
from .config import TableStore, get_spark, load_config

STAGES = ["ingest", "features", "train", "score"]
pd.set_option("display.width", 200, "display.max_columns", 30)


def run_stage(stage: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    spark = get_spark()
    store = TableStore(spark, cfg)
    t0 = time.time()
    print(f"\n=== {stage} ===")
    if stage == "ingest":
        res = ingest.run(spark, store)
        print(json.dumps(res, indent=2))
    elif stage == "features":
        _, res = features.build(spark, store, cfg)
        print(json.dumps(res, indent=2))
    elif stage == "train":
        res = train.run(store, cfg)
        print(f"\nChampion: {res['champion']}  ->  {res['model']} v{res['version']} @champion")
        cols = ["model", "val_auprc", "test_auroc", "test_auprc", "auroc_ci", "auprc_ci",
                "test_auroc_lift_vs_lr", "test_auprc_lift_vs_lr", "d_auprc_ci"]
        print(res["benchmark"][[c for c in cols if c in res["benchmark"]]].round(4).to_string(index=False))
        print("\nFairness audit (test, high-risk threshold):")
        print(res["fairness"][["attribute", "group", "n", "prevalence", "auroc", "tpr", "calibration_ratio",
                               "flag"]].round(3).to_string(index=False))
        print("\nTop SHAP features:\n" + res["shap_top"].round(4).to_string(index=False))
    elif stage == "score":
        res = score.run(spark, store, cfg)
        print(f"Scored {res['n_scored']} discharges with model v{res['model_version']}: {res['tiers']}")
        print(f"Score PSI = {res['score_psi']:.4f}; alerts={res['alerts']} warnings={res['warnings']}")
        print(res["drift_top"].round(4).to_string(index=False))
    else:
        raise ValueError(f"unknown stage {stage}")
    print(f"--- {stage} done in {time.time() - t0:.1f}s")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES + ["all"])
    a = ap.parse_args()
    cfg = load_config()
    for s in STAGES if a.stage == "all" else [a.stage]:
        run_stage(s, cfg)


if __name__ == "__main__":
    main()
