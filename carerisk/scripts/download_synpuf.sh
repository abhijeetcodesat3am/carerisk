#!/usr/bin/env bash
# Downloads CMS DE-SynPUF Sample 1 (beneficiary 2008-2010, inpatient, outpatient) and unzips to data/raw.
# Source: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
# Total ≈ 120 MB zipped / ≈ 1.1 GB unzipped. Change SAMPLE=2..20 for other samples.
set -euo pipefail
SAMPLE=${SAMPLE:-1}
OUT=${1:-"$(cd "$(dirname "$0")/.." && pwd)/data/raw"}
BASE="https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads"
mkdir -p "$OUT"; cd "$OUT"
rm -f DE1_0_*Sample_*.csv   # remove the generated sample data
for f in \
  "DE1_0_2008_Beneficiary_Summary_File_Sample_${SAMPLE}.zip" \
  "DE1_0_2009_Beneficiary_Summary_File_Sample_${SAMPLE}.zip" \
  "DE1_0_2010_Beneficiary_Summary_File_Sample_${SAMPLE}.zip" \
  "DE1_0_2008_to_2010_Inpatient_Claims_Sample_${SAMPLE}.zip" \
  "DE1_0_2008_to_2010_Outpatient_Claims_Sample_${SAMPLE}.zip"; do
  echo "↓ $f"; curl -fSL --retry 3 -o "$f" "$BASE/$f"; unzip -o -q "$f" && rm "$f"
done
ls -lh "$OUT"
# On Databricks: upload these CSVs to the Volume in conf/config.yaml (databricks.raw_dir), e.g.
#   databricks fs cp -r "$OUT" dbfs:/Volumes/main/carerisk/raw
