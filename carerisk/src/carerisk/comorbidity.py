"""Charlson Comorbidity Index from ICD-9-CM codes (Quan et al., Med Care 2005, enhanced ICD-9 coding).

SynPUF stores codes without the decimal point (e.g. "4280"), so every rule is a code prefix.
Ranges such as "4254-4259" expand to all same-length prefixes in the range.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# category: (weight, prefixes / ranges)
CHARLSON = {
    "mi": (1, ["410", "412"]),
    "chf": (1, ["39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413", "40491", "40493",
                "4254-4259", "428"]),
    "pvd": (1, ["0930", "4373", "440", "441", "4431-4439", "4471", "5571", "5579", "V434"]),
    "cevd": (1, ["36234", "430-438"]),
    "dementia": (1, ["290", "2941", "3312"]),
    "copd": (1, ["4168", "4169", "490-505", "5064", "5081", "5088"]),
    "rheum": (1, ["4465", "7100-7104", "7140-7142", "7148", "725"]),
    "pud": (1, ["531-534"]),
    "liver_mild": (1, ["07022", "07023", "07032", "07033", "07044", "07054", "0706", "0709", "570", "571",
                       "5733", "5734", "5738", "5739", "V427"]),
    "dm": (1, ["2500-2503", "2508", "2509"]),
    "dm_comp": (2, ["2504-2507"]),
    "plegia": (2, ["3341", "342", "343", "3440-3446", "3449"]),
    "renal": (2, ["40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493", "582",
                  "5830-5837", "585", "586", "5880", "V420", "V451", "V56"]),
    "cancer": (2, ["140-172", "1740-1958", "200-208", "2386"]),
    "liver_severe": (3, ["4560-4562", "5722-5728"]),
    "mets": (6, ["196-199"]),
    "hiv": (6, ["042-044"]),
}
# If the more severe form is present, the milder one is not counted.
HIERARCHY = {"dm": "dm_comp", "liver_mild": "liver_severe", "cancer": "mets"}


def _expand(rule: str) -> list[str]:
    if "-" not in rule:
        return [rule]
    lo, hi = rule.split("-")
    width = len(lo)
    prefix = "".join(ch for ch in lo if not ch.isdigit())  # e.g. "V" codes
    a, b = int(lo[len(prefix):]), int(hi[len(prefix):])
    return [f"{prefix}{n:0{width - len(prefix)}d}" for n in range(a, b + 1)]


def prefix_table() -> list[tuple[str, str]]:
    rows = []
    for cat, (_, rules) in CHARLSON.items():
        for r in rules:
            rows += [(p, cat) for p in _expand(r)]
    return rows


def categorize_codes(spark: SparkSession, codes: DataFrame, code_col: str = "code") -> DataFrame:
    """Adds a `charlson_cat` column (one row per matching category) via equi-join on 3/4/5-char prefixes."""
    lookup = spark.createDataFrame(prefix_table(), ["prefix", "charlson_cat"])
    with_prefix = codes.withColumn(
        "prefix", F.explode(F.array_distinct(F.array(*[F.substring(F.col(code_col), 1, n) for n in (3, 4, 5)])))
    )
    return with_prefix.join(F.broadcast(lookup), "prefix").drop("prefix")


def charlson_score(spark: SparkSession, cats: DataFrame, key: str) -> DataFrame:
    """cats: (key, charlson_cat) rows -> (key, cci, cci_<cat> flags) with hierarchy applied."""
    flags = cats.select(key, "charlson_cat").distinct().groupBy(key).pivot(
        "charlson_cat", list(CHARLSON)).agg(F.max(F.lit(1))).na.fill(0)
    for mild, severe in HIERARCHY.items():
        flags = flags.withColumn(mild, F.when(F.col(severe) == 1, 0).otherwise(F.col(mild)))
    score = sum(F.col(c) * w for c, (w, _) in CHARLSON.items())
    return flags.select(key, score.alias("cci"), *[F.col(c).alias(f"cci_{c}") for c in CHARLSON])
