PY := .venv/Scripts/python.exe
VW := $(PY) -m value_wage.cli

.PHONY: help install data features train tune evaluate explain mispricing calibrate export serve test lint type clean all

help:
	@echo "Targets:"
	@echo "  install     Install package in editable mode"
	@echo "  data        Build processed dataset (FM master + Transfermarkt join)"
	@echo "  features    Build and save feature matrix"
	@echo "  train       Train one booster per target with default params (quick smoke run)"
	@echo "  tune TARGET=value MODEL=lgbm TRIALS=30   Optuna tune a single model/target"
	@echo "  tune-all    Tune all boosters for both targets (slow)"
	@echo "  evaluate    Evaluate the selected winning models on the test split"
	@echo "  calibrate   Fit MAPIE conformal intervals (--method cross)"
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

evaluate:
	$(VW) evaluate run --target value
	$(VW) evaluate run --target wage

explain:
	$(VW) explain run --target value
	$(VW) explain run --target wage

mispricing:
	$(VW) mispricing build --target value
	$(VW) mispricing build --target wage

calibrate:
	$(VW) calibrate run --target value --model lgbm --method cross
	$(VW) calibrate run --target wage  --model lgbm --method cross

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
