"""Index-stay cohort, 30-day readmission label, patient-level features and the patient-grouped/time split.

Every feature for an index stay uses only information available at its discharge date:
  * utilisation counts use strictly earlier admissions / visits inside the look-back window,
  * the comorbidity index uses diagnoses from the look-back window up to the index discharge,
  * beneficiary-file chronic flags and annual costs come from the *previous* calendar year.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .comorbidity import CHARLSON, categorize_codes, charlson_score
from .ingest import CHRONIC_FLAGS

CC_COLS = list(CHRONIC_FLAGS.values())

FEATURES = [
    # demographics
    "age", "sex_female", "race_black", "race_hispanic", "race_other", "esrd",
    # index stay
    "los", "n_claims", "n_dx", "n_proc", "log_ip_payment",
    # prior utilisation
    "prior_admits_365", "prior_admits_180", "prior_ip_days_365", "days_since_last_discharge",
    "op_visits_365", "op_visits_30",
    # comorbidity burden
    "cci", "cci_chf", "cci_copd", "cci_renal", "cci_cancer", "cci_mets", "cci_dm_comp",
    *CC_COLS, "n_chronic",
    # prior-year cost
    "log_prior_ip_reimb", "log_prior_op_reimb", "log_prior_car_reimb",
]
LABEL = "readmit_30"
KEYS = ["DESYNPUF_ID", "episode_id", "admit_dt", "discharge_dt"]
GROUP_COLS = ["age_band", "sex", "state_cd"]


def _day(col):
    return F.datediff(F.col(col), F.lit("1970-01-01"))


def label_and_history(episodes: DataFrame, cohort: dict) -> DataFrame:
    """Readmission label + prior inpatient utilisation, computed over the full episode history."""
    lb = int(cohort["lookback_days"])
    win = int(cohort["label_window_days"])
    w = Window.partitionBy("DESYNPUF_ID").orderBy("admit_day")
    rng = lambda days: w.rangeBetween(-days, -1)  # noqa: E731  admissions strictly before this one
    ep = episodes.withColumn("admit_day", _day("admit_dt")).withColumn(
        "los", F.datediff("discharge_dt", "admit_dt"))
    ep = (
        ep.withColumn("next_admit_dt", F.lead("admit_dt").over(w))
        .withColumn("days_to_readmit", F.datediff("next_admit_dt", "discharge_dt"))
        .withColumn(LABEL, F.when(F.col("days_to_readmit") <= win, 1).otherwise(0))
        .withColumn("prior_admits_365", F.count("*").over(rng(lb)))
        .withColumn("prior_admits_180", F.count("*").over(rng(180)))
        .withColumn("prior_ip_days_365", F.coalesce(F.sum("los").over(rng(lb)), F.lit(0)))
        .withColumn("days_since_last_discharge",
                    F.least(F.coalesce(F.datediff("admit_dt", F.lag("discharge_dt").over(w)), F.lit(730)),
                            F.lit(730)))
    )
    return ep


def outpatient_features(index: DataFrame, op: DataFrame, lookback: int) -> DataFrame:
    i = index.select("episode_id", "DESYNPUF_ID", "admit_dt")
    o = op.select("DESYNPUF_ID", "from_dt")
    j = i.join(o, "DESYNPUF_ID").where(
        (F.col("from_dt") < F.col("admit_dt")) & (F.col("from_dt") >= F.date_sub("admit_dt", lookback)))
    return j.groupBy("episode_id").agg(
        F.count("*").alias("op_visits_365"),
        F.sum(F.when(F.col("from_dt") >= F.date_sub("admit_dt", 30), 1).otherwise(0)).alias("op_visits_30"),
    )


def comorbidity_features(spark, index: DataFrame, dxe: DataFrame, lookback: int) -> DataFrame:
    i = index.select("episode_id", "DESYNPUF_ID", "admit_dt", "discharge_dt")
    j = i.join(dxe, "DESYNPUF_ID").where(
        (F.col("event_dt") <= F.col("discharge_dt")) & (F.col("event_dt") >= F.date_sub("admit_dt", lookback)))
    cats = categorize_codes(spark, j.select("episode_id", "code"))
    return charlson_score(spark, cats, "episode_id")


def beneficiary_features(bene: DataFrame) -> tuple[DataFrame, DataFrame]:
    demo = bene.groupBy("DESYNPUF_ID").agg(
        F.min("birth_dt").alias("birth_dt"),
        F.max("death_dt").alias("death_dt"),
        F.last("sex").alias("sex"),
        F.last("race_cd").alias("race_cd"),
        F.max("state_cd").alias("state_cd"),
    )
    yearly = bene.select(
        "DESYNPUF_ID", (F.col("year") + 1).alias("feature_year"), "esrd", *CC_COLS,
        F.log1p("medreimb_ip").alias("log_prior_ip_reimb"),
        F.log1p("medreimb_op").alias("log_prior_op_reimb"),
        F.log1p("medreimb_car").alias("log_prior_car_reimb"),
    )
    return demo, yearly


def assign_split(df: DataFrame, cohort: dict, split: dict) -> DataFrame:
    """Patient-grouped + time-based split.

    * Each beneficiary is hashed to exactly one of train / val / test patients (no patient overlap).
    * Train and val patients contribute index stays before `test_start`.
    * Test patients contribute only index stays on/after `test_start` -> unseen patients, future period.
    """
    bucket = F.pmod(F.xxhash64("DESYNPUF_ID", F.lit(int(split["seed"]))), F.lit(100))
    tr, va = int(split["train_pct"]), int(split["val_pct"])
    group = F.when(bucket < tr, "train").when(bucket < tr + va, "val").otherwise("test")
    in_test_period = F.col("admit_dt") >= F.lit(cohort["test_start"]).cast("date")
    return (df.withColumn("_g", group)
            .withColumn("split", F.when((F.col("_g") == "test") & in_test_period, "test")
                        .when((F.col("_g") != "test") & ~in_test_period, F.col("_g")))
            .drop("_g"))


def build(spark, store, cfg: dict) -> tuple[DataFrame, dict]:
    cohort, lb = cfg["cohort"], int(cfg["cohort"]["lookback_days"])
    episodes = store.read("silver_ip_episodes")
    hist = label_and_history(episodes, cohort)

    start = F.lit(cohort["index_start"]).cast("date")
    censor = F.date_sub(F.lit(cohort["data_end"]).cast("date"), int(cohort["label_window_days"]))
    demo, yearly = beneficiary_features(store.read("silver_beneficiary"))

    idx = hist.where(F.col("admit_dt") >= start).join(demo, "DESYNPUF_ID", "left")
    win = int(cohort["label_window_days"])
    died_in_hosp = F.col("death_dt").isNotNull() & (F.col("death_dt") <= F.col("discharge_dt"))
    died_no_readmit = (F.col("death_dt").isNotNull() & (F.col(LABEL) == 0)
                       & (F.col("death_dt") <= F.date_add("discharge_dt", win)))
    censored = F.col("discharge_dt") > censor
    stats = {"index_stays_raw": idx.count()}
    idx = idx.withColumn("_excl", F.when(died_in_hosp, "died_in_hospital")
                         .when(died_no_readmit, "died_within_window_no_readmit")
                         .when(censored, "insufficient_follow_up"))
    for r in idx.groupBy("_excl").count().collect():
        if r["_excl"]:
            stats[f"excluded_{r['_excl']}"] = r["count"]
    idx = idx.where(F.col("_excl").isNull()).drop("_excl").cache()

    op_f = outpatient_features(idx, store.read("silver_op_claims"), lb)
    cci_f = comorbidity_features(spark, idx, store.read("silver_dx_events"), lb)

    age = F.floor(F.datediff("admit_dt", "birth_dt") / 365.25)
    feats = (
        idx.withColumn("feature_year", F.year("admit_dt"))
        .join(yearly, ["DESYNPUF_ID", "feature_year"], "left")
        .join(op_f, "episode_id", "left")
        .join(cci_f, "episode_id", "left")
        .withColumn("age", age)
        .withColumn("age_band", F.when(age < 65, "<65").when(age < 75, "65-74").when(age < 85, "75-84")
                    .otherwise("85+"))
        .withColumn("sex_female", (F.col("sex") == "F").cast("int"))
        .withColumn("race_black", (F.col("race_cd") == 2).cast("int"))
        .withColumn("race_hispanic", (F.col("race_cd") == 5).cast("int"))
        .withColumn("race_other", (F.col("race_cd") == 3).cast("int"))
        .withColumn("n_dx", F.size("dx"))
        .withColumn("log_ip_payment", F.log1p(F.greatest(F.col("ip_payment"), F.lit(0.0))))
        .na.fill(0, subset=["op_visits_365", "op_visits_30", "cci", "esrd", *CC_COLS,
                            *[f"cci_{c}" for c in CHARLSON]])
        .withColumn("n_chronic", sum(F.col(c) for c in CC_COLS))
    )
    feats = assign_split(feats, cohort, cfg["split"])
    out = feats.select(*KEYS, *GROUP_COLS, "split", *[F.col(c).cast("double") for c in FEATURES],
                       F.col(LABEL).cast("int"), "days_to_readmit")
    store.write(out, "gold_features")
    idx.unpersist()
    g = store.read("gold_features")
    for r in g.groupBy("split").agg(F.count("*").alias("n"), F.avg(LABEL).alias("rate")).collect():
        stats[f"split_{r['split']}"] = {"n": r["n"], "readmit_rate": round(r["rate"], 4)}
    return g, stats
