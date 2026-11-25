"""Bronze -> silver ingestion of CMS DE-SynPUF beneficiary, inpatient and outpatient files.

Expected raw file names (as distributed by CMS, sample 1 shown):
    DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv   (also 2009, 2010)
    DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv
    DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

CHRONIC_FLAGS = {
    "SP_ALZHDMTA": "cc_alzheimers",
    "SP_CHF": "cc_chf",
    "SP_CHRNKIDN": "cc_ckd",
    "SP_CNCR": "cc_cancer",
    "SP_COPD": "cc_copd",
    "SP_DEPRESSN": "cc_depression",
    "SP_DIABETES": "cc_diabetes",
    "SP_ISCHMCHT": "cc_ihd",
    "SP_OSTEOPRS": "cc_osteoporosis",
    "SP_RA_OA": "cc_ra_oa",
    "SP_STRKETIA": "cc_stroke_tia",
}
IP_DX = ["ADMTNG_ICD9_DGNS_CD"] + [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]
IP_PX = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]
OP_DX = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]


def _d(col: str):
    return F.to_date(F.col(col), "yyyyMMdd")


def _codes(cols: list[str], available: list[str]):
    present = [F.trim(F.col(c)) for c in cols if c in available]
    return F.filter(F.array(*present), lambda x: x.isNotNull() & (x != ""))


def _read(spark: SparkSession, path: str) -> DataFrame:
    # Everything as string: SynPUF codes have leading zeros / letters (V-, E-codes).
    return spark.read.option("header", True).option("inferSchema", False).csv(path)


def silver_beneficiary(spark: SparkSession, raw_dir: str) -> DataFrame:
    df = _read(spark, f"{raw_dir}/*Beneficiary_Summary_File*.csv")
    df = df.withColumn(
        "year", F.regexp_extract(F.col("_metadata.file_path"), r"DE1_0_(\d{4})_Beneficiary", 1).cast("int")
    )
    flags = [F.when(F.col(src) == "1", 1).otherwise(0).alias(dst) for src, dst in CHRONIC_FLAGS.items()]
    money = [F.coalesce(F.col(c).cast("double"), F.lit(0.0)).alias(c.lower())
             for c in ("MEDREIMB_IP", "MEDREIMB_OP", "MEDREIMB_CAR")]
    return df.select(
        "DESYNPUF_ID",
        "year",
        _d("BENE_BIRTH_DT").alias("birth_dt"),
        _d("BENE_DEATH_DT").alias("death_dt"),
        F.when(F.col("BENE_SEX_IDENT_CD") == "2", "F").otherwise("M").alias("sex"),
        F.col("BENE_RACE_CD").cast("int").alias("race_cd"),
        F.when(F.col("BENE_ESRD_IND") == "Y", 1).otherwise(0).alias("esrd"),
        F.col("SP_STATE_CODE").alias("state_cd"),
        *flags,
        *money,
    ).dropDuplicates(["DESYNPUF_ID", "year"])


def silver_ip_claims(spark: SparkSession, raw_dir: str) -> DataFrame:
    df = _read(spark, f"{raw_dir}/*Inpatient_Claims*.csv")
    cols = df.columns
    claims = df.select(
        "DESYNPUF_ID",
        "CLM_ID",
        F.coalesce(_d("CLM_ADMSN_DT"), _d("CLM_FROM_DT")).alias("admit_dt"),
        F.coalesce(_d("NCH_BENE_DSCHRG_DT"), _d("CLM_THRU_DT")).alias("discharge_dt"),
        F.col("CLM_PMT_AMT").cast("double").alias("pmt"),
        _codes(IP_DX, cols).alias("dx"),
        F.size(_codes(IP_PX, cols)).alias("n_proc"),
    )
    # A claim can span several SEGMENT rows: collapse to one row per claim.
    claims = claims.groupBy("DESYNPUF_ID", "CLM_ID").agg(
        F.min("admit_dt").alias("admit_dt"),
        F.max("discharge_dt").alias("discharge_dt"),
        F.sum("pmt").alias("pmt"),
        F.array_distinct(F.flatten(F.collect_list("dx"))).alias("dx"),
        F.max("n_proc").alias("n_proc"),
    )
    return claims.filter(
        F.col("admit_dt").isNotNull() & F.col("discharge_dt").isNotNull()
        & (F.col("discharge_dt") >= F.col("admit_dt"))
    )


def build_episodes(ip_claims: DataFrame) -> DataFrame:
    """Merge overlapping / same-day-transfer claims into one hospital episode."""
    w = Window.partitionBy("DESYNPUF_ID").orderBy("admit_dt", "discharge_dt", "CLM_ID")
    prev_max_dis = F.max("discharge_dt").over(w.rowsBetween(Window.unboundedPreceding, -1))
    new_ep = F.when(prev_max_dis.isNull() | (F.col("admit_dt") > prev_max_dis), 1).otherwise(0)
    seq = (ip_claims.withColumn("_new", new_ep)
           .withColumn("ep_seq", F.sum("_new").over(w.rowsBetween(Window.unboundedPreceding, 0))))
    return seq.groupBy("DESYNPUF_ID", "ep_seq").agg(
        F.min("admit_dt").alias("admit_dt"),
        F.max("discharge_dt").alias("discharge_dt"),
        F.sum("pmt").alias("ip_payment"),
        F.array_distinct(F.flatten(F.collect_list("dx"))).alias("dx"),
        F.sum("n_proc").alias("n_proc"),
        F.count("*").alias("n_claims"),
    ).withColumn("episode_id", F.concat_ws("_", "DESYNPUF_ID", F.col("ep_seq").cast("string"))).drop("ep_seq")


def silver_op_claims(spark: SparkSession, raw_dir: str) -> DataFrame:
    df = _read(spark, f"{raw_dir}/*Outpatient_Claims*.csv")
    cols = df.columns
    op = df.select(
        "DESYNPUF_ID",
        "CLM_ID",
        _d("CLM_FROM_DT").alias("from_dt"),
        F.col("CLM_PMT_AMT").cast("double").alias("pmt"),
        _codes(OP_DX, cols).alias("dx"),
    ).filter(F.col("from_dt").isNotNull())
    return op.groupBy("DESYNPUF_ID", "CLM_ID").agg(
        F.min("from_dt").alias("from_dt"),
        F.sum("pmt").alias("pmt"),
        F.array_distinct(F.flatten(F.collect_list("dx"))).alias("dx"),
    )


def dx_events(ip_claims: DataFrame, op_claims: DataFrame) -> DataFrame:
    """Long table of (patient, date, ICD-9 code) from both settings, used for the comorbidity index."""
    ip = ip_claims.select("DESYNPUF_ID", F.col("admit_dt").alias("event_dt"), F.explode("dx").alias("code"))
    op = op_claims.select("DESYNPUF_ID", F.col("from_dt").alias("event_dt"), F.explode("dx").alias("code"))
    return ip.unionByName(op).distinct()


def run(spark, store) -> dict:
    raw = store.raw_dir
    bene = silver_beneficiary(spark, raw)
    ip = silver_ip_claims(spark, raw).cache()
    op = silver_op_claims(spark, raw).cache()
    store.write(bene, "silver_beneficiary")
    store.write(ip, "silver_ip_claims")
    store.write(build_episodes(ip), "silver_ip_episodes")
    store.write(op, "silver_op_claims")
    store.write(dx_events(ip, op), "silver_dx_events")
    counts = {t: store.read(t).count() for t in
              ("silver_beneficiary", "silver_ip_claims", "silver_ip_episodes", "silver_op_claims")}
    ip.unpersist(); op.unpersist()
    return counts
