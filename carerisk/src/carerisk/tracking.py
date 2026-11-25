"""MLflow setup that works both locally (SQLite registry) and on Databricks (Unity Catalog registry)."""
from __future__ import annotations

import json

import mlflow
from mlflow.tracking import MlflowClient

from .config import ROOT, on_databricks


def setup(cfg: dict) -> str:
    m = cfg["mlflow"]
    if on_databricks():
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(m["databricks_experiment"])
    else:
        mlflow.set_tracking_uri(m["local_tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/"))
        name = m["local_experiment"]
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts").as_uri())
        mlflow.set_experiment(name)
    return registered_name(cfg)


def registered_name(cfg: dict) -> str:
    name = cfg["mlflow"]["model_name"]
    if on_databricks():
        db = cfg["databricks"]
        return f"{db['catalog']}.{db['schema']}.{name}"
    return name


def load_champion(cfg: dict):
    """Returns (sklearn pipeline, metadata dict) for the model version behind the champion alias."""
    name = setup(cfg)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, cfg["mlflow"]["champion_alias"])
    model = mlflow.sklearn.load_model(f"models:/{name}@{cfg['mlflow']['champion_alias']}")
    tags = mv.tags
    meta = {
        "model_name": name,
        "model_version": str(mv.version),
        "run_id": mv.run_id,
        "features": json.loads(tags["features"]),
        "threshold_high": float(tags["threshold_high"]),
        "threshold_medium": float(tags["threshold_medium"]),
        "model_family": tags.get("model_family", ""),
    }
    return model, meta
