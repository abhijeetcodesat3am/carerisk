PY      := .venv/bin/python
export JAVA_HOME ?= $(shell /usr/libexec/java_home -v 21 2>/dev/null || /usr/libexec/java_home -v 17 2>/dev/null)
export PYTHONPATH := src
export PYSPARK_PYTHON := $(PY)

.PHONY: setup sample data ingest features train score all test mlflow clean
setup:      ; uv venv -p 3.11 .venv && uv pip install -p $(PY) -r requirements.txt
sample:     ; $(PY) scripts/make_sample_data.py --patients 20000
data:       ; ./scripts/download_synpuf.sh
ingest features train score all: ; $(PY) -m carerisk.pipeline $@
test:       ; $(PY) -m pytest -q tests
mlflow:     ; .venv/bin/mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
clean:      ; rm -rf data/tables reports mlruns.db mlartifacts spark-warehouse
