PY := .venv/Scripts/python.exe
VW := $(PY) -m value_wage.cli

.PHONY: help install data features train baselines tune evaluate compare target-compare explain mispricing export serve test lint type clean all

help:
	@echo "Targets:"
	@echo "  install     Install package in editable mode"
	@echo "  data        Build processed dataset (FM master + Transfermarkt join)"
	@echo "  features    Build and save feature matrix"
	@echo "  train       Train one booster per target with default params (quick smoke run)"
	@echo "  tune TARGET=value MODEL=lgbm TRIALS=30   Optuna tune a single model/target"
	@echo "  tune-all    Tune all boosters for both targets (slow)"
	@echo "  baselines   Fit ElasticNet + Random Forest reference models (both targets)"
	@echo "  evaluate    Evaluate the selected winning models on the test split"
	@echo "  compare     Baseline vs production comparison table (both targets)"
	@echo "  target-compare TARGET=value MODEL=lgbm   log1p vs direct target experiment"
	@echo "  explain     Produce SHAP global + local plots"
	@echo "  mispricing  Produce the mispricing board CSVs"
	@echo "  export      Produce the JSON bundle for the web UI"
	@echo "  serve       Serve the web UI at http://localhost:8765"
	@echo "  test        Run pytest"
	@echo "  lint        Run ruff"
	@echo "  type        Run mypy"
	@echo "  all         test + lint + type + data + train (quick) + evaluate"

install:
	$(PY) -m pip install -e ".[dev]"

data:
	$(VW) data build

features:
	$(VW) features build

train:
	$(VW) train quick --target value
	$(VW) train quick --target wage

tune:
	$(VW) tune one --target $(TARGET) --model $(MODEL) --trials $(TRIALS)

tune-all:
	$(VW) tune all --trials 50

baselines:
	$(VW) train baseline --target value --model elasticnet
	$(VW) train baseline --target value --model rf
	$(VW) train baseline --target wage --model elasticnet
	$(VW) train baseline --target wage --model rf

evaluate:
	$(VW) evaluate run --target value
	$(VW) evaluate run --target wage

compare:
	$(VW) compare run --target value
	$(VW) compare run --target wage

TARGET ?= value
MODEL ?= lgbm
target-compare:
	$(PY) experiments/target_transform_compare.py --target $(TARGET) --model $(MODEL)

explain:
	$(VW) explain run --target value
	$(VW) explain run --target wage

mispricing:
	$(VW) mispricing build --target value
	$(VW) mispricing build --target wage

export:
	$(VW) export web

serve:
	cd web && $(PY) -m http.server 8765

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

type:
	$(PY) -m mypy

all: test lint data train evaluate

clean:
	rm -rf data/processed/* mlruns/* optuna.db reports/figures/* reports/*.html reports/*.pdf
