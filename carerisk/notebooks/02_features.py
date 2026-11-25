# Databricks notebook source
# MAGIC %md
# MAGIC # CareRisk · 02 Cohort, label & features (gold)
# MAGIC Thin wrapper around `src/carerisk` so the same code runs locally and as a Databricks job task.

# COMMAND ----------
# MAGIC %pip install -q pyyaml
# COMMAND ----------

import os, sys
sys.path.insert(0, os.path.abspath("../src"))
from carerisk.pipeline import run_stage

result = run_stage("features")
