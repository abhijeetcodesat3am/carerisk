# Databricks notebook source
# MAGIC %md
# MAGIC # CareRisk · 04 Batch scoring + PSI drift
# MAGIC Thin wrapper around `src/carerisk` so the same code runs locally and as a Databricks job task.

# COMMAND ----------
# MAGIC %pip install -q "xgboost>=2.0" "shap>=0.45" pyyaml
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
from carerisk.pipeline import run_stage

result = run_stage("score")
