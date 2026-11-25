"""Generate a small dataset in the exact DE-SynPUF file layout, for local smoke-testing.

It is NOT the CMS data: it plants a known readmission signal (frailty, chronic burden, prior
admissions, LOS) and a deliberate 2010 shift (more outpatient visits, shorter stays) so the
drift monitor has something to find. Use scripts/download_synpuf.sh for the real files.

    python scripts/make_sample_data.py --patients 20000 --out data/raw
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START, END = date(2008, 1, 1), date(2010, 12, 31)
FLAGS = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD", "SP_DEPRESSN", "SP_DIABETES",
         "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA"]
FLAG_BASE = [-3.2, -1.6, -2.0, -2.6, -2.0, -1.8, -1.0, -0.9, -2.0, -1.6, -3.0]
FLAG_CODES = {  # ICD-9 codes that map to Charlson categories
    "SP_ALZHDMTA": ["3310", "2900"], "SP_CHF": ["4280", "42833", "4254"], "SP_CHRNKIDN": ["5859", "5853", "40391"],
    "SP_CNCR": ["1749", "1629", "1890", "1970"], "SP_COPD": ["496", "4912", "4928"], "SP_DEPRESSN": ["311", "29630"],
    "SP_DIABETES": ["25000", "25002", "25040", "25060"], "SP_ISCHMCHT": ["41401", "41071", "412"],
    "SP_OSTEOPRS": ["73300"], "SP_RA_OA": ["7140", "7150"], "SP_STRKETIA": ["43491", "4359", "3420"],
}
NOISE = ["4019", "2724", "V5869", "27800", "4011", "2449", "53081", "78650", "7802", "5990", "486", "42731", "2859"]
EXTRA = ["5715", "5722", "1985", "042", "4439", "5330", "71590"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fmt(d: date | None) -> str:
    return "" if d is None else d.strftime("%Y%m%d")


def rand_date(rng, a: date, b: date) -> date:
    return a + timedelta(days=int(rng.integers(0, max((b - a).days, 1))))


def main(n: int, out: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ids = [f"{x:016X}" for x in rng.integers(0, 2**63, n)]
    frail = rng.normal(0, 1, n)
    young = rng.random(n) < 0.15
    birth = [date(int(rng.integers(1950, 1984)) if y else int(rng.integers(1912, 1943)),
                  int(rng.integers(1, 13)), int(rng.integers(1, 29))) for y in young]
    sex = np.where(rng.random(n) < 0.45, 1, 2)
    race = rng.choice([1, 2, 3, 5], n, p=[0.83, 0.10, 0.04, 0.03])
    state = rng.integers(1, 55, n)
    age08 = np.array([(START - b).days / 365.25 for b in birth])

    # chronic flags per year (persistent once acquired)
    flags = np.zeros((3, n, len(FLAGS)), dtype=int)
    for k, base in enumerate(FLAG_BASE):
        p0 = sigmoid(base + 0.9 * frail + 0.02 * (age08 - 75))
        flags[0, :, k] = rng.random(n) < p0
        for y in (1, 2):
            flags[y, :, k] = flags[y - 1, :, k] | (rng.random(n) < p0 * 0.15)
    esrd = (flags[0, :, 2] == 1) & (rng.random(n) < 0.12)

    ip_rows, op_rows, death = [], [], [None] * n
    clm = 100000
    for i in range(n):
        nchron = flags[0, i].sum()
        # death
        hz = sigmoid(-4.2 + 0.7 * frail[i] + 0.05 * (age08[i] - 75) + 0.15 * nchron)
        for y in range(3):
            if rng.random() < hz:
                death[i] = rand_date(rng, date(2008 + y, 1, 1), date(2008 + y, 12, 31)); break
        last = death[i] or END
        patient_codes = [c for k, f in enumerate(FLAGS) if flags[0, i, k] for c in FLAG_CODES[f]]
        # outpatient visits (2010 rate +35% -> drift)
        for y in range(3):
            lam = np.exp(1.3 + 0.45 * frail[i] + 0.12 * nchron) * (1.35 if y == 2 else 1.0)
            for _ in range(rng.poisson(lam)):
                d = rand_date(rng, date(2008 + y, 1, 1), date(2008 + y, 12, 31))
                if d > last:
                    continue
                dx = list(rng.choice(patient_codes or NOISE, min(2, len(patient_codes or NOISE)), replace=False)) \
                    + list(rng.choice(NOISE, 2, replace=False))
                clm += 1
                op_rows.append({"DESYNPUF_ID": ids[i], "CLM_ID": str(clm), "SEGMENT": "1", "CLM_FROM_DT": fmt(d),
                                "CLM_THRU_DT": fmt(d), "CLM_PMT_AMT": str(int(rng.gamma(2, 60))),
                                **{f"ICD9_DGNS_CD_{j + 1}": dx[j] if j < len(dx) else "" for j in range(10)}})
        # inpatient: Poisson admissions + planted 30-day readmission mechanism
        rate = np.exp(-2.0 + 0.85 * frail[i] + 0.18 * nchron + 0.01 * (age08[i] - 75))
        t = START + timedelta(days=int(rng.exponential(365 / rate)))
        recent = 0
        while t <= last:
            shift = 0.8 if t.year == 2010 else 1.0
            los = max(0, int(rng.poisson(np.exp(1.2 + 0.25 * frail[i]) * shift)))
            dis = min(t + timedelta(days=los), last)
            dx = list(rng.choice(patient_codes, min(len(patient_codes), int(rng.integers(1, 5))), replace=False)) \
                if patient_codes else []
            dx += list(rng.choice(NOISE, int(rng.integers(1, 4)), replace=False))
            if rng.random() < 0.05:
                dx.append(str(rng.choice(EXTRA)))
            clm += 1
            ip_rows.append({"DESYNPUF_ID": ids[i], "CLM_ID": str(clm), "SEGMENT": "1", "CLM_FROM_DT": fmt(t),
                            "CLM_THRU_DT": fmt(dis), "PRVDR_NUM": f"{rng.integers(10000, 99999)}",
                            "CLM_PMT_AMT": str(int(rng.gamma(3, 2500) * (1 + 0.1 * los))),
                            "CLM_ADMSN_DT": fmt(t), "ADMTNG_ICD9_DGNS_CD": dx[0],
                            "CLM_UTLZTN_DAY_CNT": str(los), "NCH_BENE_DSCHRG_DT": fmt(dis),
                            "CLM_DRG_CD": f"{rng.integers(1, 999):03d}",
                            **{f"ICD9_DGNS_CD_{j + 1}": dx[j + 1] if j + 1 < len(dx) else "" for j in range(10)},
                            **{f"ICD9_PRCDR_CD_{j + 1}": str(rng.integers(3000, 9999)) if rng.random() < 0.3 / (j + 1)
                               else "" for j in range(6)}})
            if rng.random() < 0.03:  # same-day transfer -> second claim, should be merged into one episode
                clm += 1
                t2 = dis + timedelta(days=int(rng.integers(1, 4)))
                ip_rows.append({**ip_rows[-1], "CLM_ID": str(clm), "CLM_FROM_DT": fmt(dis), "CLM_ADMSN_DT": fmt(dis),
                                "CLM_THRU_DT": fmt(t2), "NCH_BENE_DSCHRG_DT": fmt(t2)})
                dis = t2
            p_re = sigmoid(-2.9 + 0.55 * frail[i] + 0.12 * nchron + 0.06 * min(los, 15) + 0.45 * recent
                           + 0.4 * esrd[i] + (0.25 if young[i] else 0))
            recent = min(recent + 1, 3)
            if rng.random() < p_re:
                t = dis + timedelta(days=int(rng.integers(1, 31)))
            else:
                recent = max(recent - 1, 0)
                t = dis + timedelta(days=int(31 + rng.exponential(365 / rate)))

    ip = pd.DataFrame(ip_rows)
    for y in range(3):
        yr = 2008 + y
        ipy = ip[ip["CLM_FROM_DT"].str[:4] == str(yr)].assign(p=lambda d: d["CLM_PMT_AMT"].astype(float))
        ip_reimb = ipy.groupby("DESYNPUF_ID")["p"].sum()
        opd = pd.DataFrame(op_rows)
        opd = opd[opd["CLM_FROM_DT"].str[:4] == str(yr)].assign(p=lambda d: d["CLM_PMT_AMT"].astype(float))
        op_reimb = opd.groupby("DESYNPUF_ID")["p"].sum()
        alive = [i for i in range(n) if death[i] is None or death[i].year >= yr]
        bene = pd.DataFrame({
            "DESYNPUF_ID": [ids[i] for i in alive],
            "BENE_BIRTH_DT": [fmt(birth[i]) for i in alive],
            "BENE_DEATH_DT": [fmt(death[i]) if death[i] and death[i].year == yr else "" for i in alive],
            "BENE_SEX_IDENT_CD": sex[alive], "BENE_RACE_CD": race[alive],
            "BENE_ESRD_IND": np.where(esrd[alive], "Y", "0"), "SP_STATE_CODE": [f"{s:02d}" for s in state[alive]],
            "BENE_COUNTY_CD": "100", "BENE_HI_CVRAGE_TOT_MONS": 12, "BENE_SMI_CVRAGE_TOT_MONS": 12,
            **{f: np.where(flags[y, alive, k] == 1, 1, 2) for k, f in enumerate(FLAGS)},
            "MEDREIMB_IP": [int(ip_reimb.get(ids[i], 0)) for i in alive],
            "MEDREIMB_OP": [int(op_reimb.get(ids[i], 0)) for i in alive],
            "MEDREIMB_CAR": [int(rng.gamma(2, 400 + 150 * max(frail[i], 0))) for i in alive],
        })
        bene.to_csv(out / f"DE1_0_{yr}_Beneficiary_Summary_File_Sample_1.csv", index=False)
    ip.to_csv(out / "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", index=False)
    pd.DataFrame(op_rows).to_csv(out / "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv", index=False)
    print(f"wrote {n} beneficiaries, {len(ip)} inpatient claims, {len(op_rows)} outpatient claims -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.patients, a.out, a.seed)
