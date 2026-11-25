"""Configuration, environment detection and the table store (Delta on Databricks, Parquet locally)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = path or os.environ.get("CARERISK_CONFIG", ROOT / "conf" / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_spark():
    from pyspark.sql import SparkSession

    if on_databricks():
        return SparkSession.builder.getOrCreate()
    return (
        SparkSession.builder.master("local[*]")
        .appName("carerisk")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", os.environ.get("CARERISK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def reports_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["reports_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


class TableStore:
    """Reads/writes named tables. Unity Catalog Delta tables on Databricks, Parquet folders locally."""

    def __init__(self, spark, cfg: dict):
        self.spark = spark
        self.cfg = cfg
        if on_databricks():
            db = cfg["databricks"]
            self.catalog, self.schema = db["catalog"], db["schema"]
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")
        else:
            self.tables_dir = ROOT / cfg["paths"]["tables_dir"]
            self.tables_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> str:
        if on_databricks():
            return self.cfg["databricks"]["raw_dir"]
        return str(ROOT / self.cfg["paths"]["raw_dir"])

    def fqn(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}" if on_databricks() else str(self.tables_dir / name)

    def write(self, df, name: str, mode: str = "overwrite") -> None:
        if on_databricks():
            (df.write.format("delta").mode(mode).option("overwriteSchema", "true")
             .saveAsTable(self.fqn(name)))
        else:
            df.write.mode(mode).parquet(self.fqn(name))

    def read(self, name: str):
        if on_databricks():
            return self.spark.table(self.fqn(name))
        return self.spark.read.parquet(self.fqn(name))

    def exists(self, name: str) -> bool:
        if on_databricks():
            return self.spark.catalog.tableExists(self.fqn(name))
        return Path(self.fqn(name)).exists()
