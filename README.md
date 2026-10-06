![CI](https://github.com/ashutosh-kulkarni-dev/premier-league-value-modelling/actions/workflows/ci.yml/badge.svg)
# PL Insight — Premier League Player Value & Wage Model 

A regression model and static web UI that predict Premier League players' market
values (Transfermarkt €) and wages (FM-estimated £/wk) from on-pitch performance
and scouted attributes, and flag players where the actual market valuation
diverges most from what the measurable features alone would predict.

> **Live demo:** https://premier-league-value-modelling-web.vercel.app/
> The project treats the model's **residuals as the deliverable**, not as errors
> to minimise. Residuals quantify the "human premium" that mechanical features
> can't see: manager demand, auction dynamics, expiring contracts, injury
> rumours, hype. The mispricing board is a tool for investigation, not a
> declaration of market error.

---

## Highlights

| | Value model | Wage model |
|---|---|---|
| **R² (log)**              | 0.78 | 0.86 |
| **MAE (log)**             | 0.40 | 0.26 |
| **Spearman ρ**            | 0.88 | 0.94 |
| **80% interval coverage** | 90% | 90% |
| **Winner**                | LightGBM | LightGBM |

- **Three boosters tuned under one Optuna recipe** (LightGBM, XGBoost, HGBR; CatBoost dropped for compute budget).
- **Season-based split** with GroupKFold on `player_id` inside training — no cross-season leakage.
- **MAPIE cross-conformal** 80% prediction intervals (distribution-free coverage guarantee).
- **SHAP** on every prediction, globally and per-player.
- **Three-layer leakage guard** — yaml allowlist + runtime assertion + unit test — stops FM's own `value_high` from tautologically dominating the model.

Full write-up: [`reports/REPORT.md`](reports/REPORT.md).

---

## Data sources

| Source | What it brings | Notes |
|---|---|---|
| **Transfermarkt** (Kaggle `davidcariboo/player-scores`) | real market values, bio, career, photos, per-match stats | **not committed** — download yourself (see Setup) |
| **Understat** | per-season xG, xA, xG chain/buildup | ingested upstream; the processed master is committed |
| **Football Manager 26** (FMinside) | 40+ scouted attributes, `current_ability`, `potential`, contract | one snapshot |

Covers **3 Premier League seasons: 2023-24, 2024-25, 2025-26**.

---

## Repository layout

```
pl-insight/
├── src/value_wage/            # importable package (14 modules, mypy-ready)
│   ├── config.py              # paths, seeds, feature lists, hyperparam spaces
│   ├── data.py                # loader + pandera validation + SHA-256 provenance
│   ├── features.py            # per-90, age², contract, FM aggregates
│   ├── splits.py              # season split + GroupKFold inner CV
│   ├── preprocess.py          # ColumnTransformers
│   ├── models/                # baselines + 4 boosters + quantile companions
│   ├── tuning.py              # Optuna harness, MLflow logging
│   ├── calibration.py         # MAPIE conformal wrapper
│   ├── train.py               # fit → artifact
│   ├── evaluate.py            # metrics, bootstrap CIs, diagnostic plots
│   ├── explain.py             # SHAP global + waterfalls
│   ├── mispricing.py          # ranked board + comparables
│   ├── export_web.py          # produces the JSON bundle for the UI
│   ├── sources/
│   │   ├── transfermarkt.py          # polite TM scraper (unused — Kaggle preferred)
│   │   └── transfermarkt_kaggle.py   # Kaggle ingest + 4-pass fuzzy join
│   └── cli.py                 # typer CLI
├── config/features.yaml       # feature allowlist / excluded list
├── tests/                     # pytest + hypothesis (28 passing)
├── data/
│   ├── raw/                   # master_fm26_pl.parquet (committed) + Kaggle (gitignored)
│   └── processed/             # features + model artefacts + predictions
├── reports/                   # metrics JSON, SHAP CSVs, figures, REPORT.md
├── web/                       # Deployed to Vercel
│   ├── index.html             # single-file UI, Chart.js from CDN
│   ├── placeholder.webp
│   ├── README.md              # JSON data contract
│   └── data/
│       ├── players.json       # per-player predictions + SHAP + bio + career
│       ├── bio.json           # compact bio-only lookup
│       ├── meta.json          # model metadata + global SHAP
│       └── mispricing.json    # top-20 under/over per target
├── C/                         # Deferred extension: transfer-fee model plan
├── plan.md                    # Original project plan
├── techstack.md               # Tool choices + rationale
├── pyproject.toml
├── Makefile
├── LICENSE                    # MIT; dataset attributions inside
└── README.md                  # this file
```

---

## Setup

**Requirements:** Python 3.11+, Git. ~2 GB free disk (mostly for the Kaggle dump).

```bash
git clone https://github.com/<you>/pl-insight.git
cd pl-insight

# Fresh venv
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"     # Windows
# source .venv/bin/activate && pip install -e ".[dev]"  # macOS/Linux
```

### Get the dataset

1. Download `davidcariboo/player-scores` from
   <https://www.kaggle.com/datasets/davidcariboo/player-scores>
2. Extract into `data/raw/transfermarkt/` so you have
   `players.csv`, `transfers.csv`, `player_valuations.csv`, `appearances.csv`,
   `games.csv`, `clubs.csv`, `competitions.csv`.

### Run the pipeline from scratch

```bash
# Data
.venv/Scripts/python.exe -m value_wage.cli data build
.venv/Scripts/python.exe -m value_wage.cli features build

# Tune each (model, target) pair separately (keeps each run under 10 min)
for M in lgbm xgb hgbr; do
  .venv/Scripts/python.exe -m value_wage.cli tune one --target value --model $M --trials 30
  .venv/Scripts/python.exe -m value_wage.cli tune one --target wage  --model $M --trials 30
done

# Train the winners
.venv/Scripts/python.exe -m value_wage.cli train tuned --target value --model lgbm --params-json data/processed/models/value__lgbm__best_params/params.json
.venv/Scripts/python.exe -m value_wage.cli train tuned --target wage  --model lgbm --params-json data/processed/models/wage__lgbm__best_params/params.json

# Evaluate, calibrate, SHAP, mispricing
.venv/Scripts/python.exe -m value_wage.cli evaluate run  --target value
.venv/Scripts/python.exe -m value_wage.cli evaluate run  --target wage
.venv/Scripts/python.exe -m value_wage.cli calibrate run --target value --model lgbm --method cross
.venv/Scripts/python.exe -m value_wage.cli calibrate run --target wage  --model lgbm --method cross
.venv/Scripts/python.exe -m value_wage.cli explain  run  --target value
.venv/Scripts/python.exe -m value_wage.cli explain  run  --target wage
.venv/Scripts/python.exe -m value_wage.cli mispricing build --target value
.venv/Scripts/python.exe -m value_wage.cli mispricing build --target wage

# Export JSON bundle for the web UI
.venv/Scripts/python.exe -m value_wage.cli export web
```

Every run is seeded (`SEED=42`); the raw inputs are SHA-256 hashed on
ingestion; the same inputs produce bit-identical outputs.

### Run the UI locally

```bash
cd web
python -m http.server 8765
# open http://localhost:8765
```

### Run tests

```bash
.venv/Scripts/python.exe -m pytest
```

---

## Methodology at a glance

- **Target:** `log1p(market_value_in_eur)` from Transfermarkt. Two snapshots per
  `(player, season)` — 01 Aug (start) and 31 May (end) — so the model learns
  within-season aging and contract decay.
- **Features (48):** biographical + per-90 performance + volume + 29 FM
  attributes + 5 FM aggregates + contract months remaining + primary position.
- **Validation:** train 2023-24, validate 2024-25, test 2025-26.
- **GroupKFold on `player_id`** inside training folds — same player cannot
  appear in both inner-train and inner-val.
- **Pre-declared selection rule:** tune all three boosters under one Optuna
  recipe (MAE on log target, TPE + MedianPruner), pick the winner on inner-CV
  log-MAE.
- **Calibration:** MAPIE cross-conformal (CV+) with GroupKFold → 80% intervals
  that satisfy the coverage guarantee without sacrificing training data.
- **SHAP TreeExplainer** on every scored row; global + local attributions
  shipped in the UI.

See [`reports/REPORT.md`](reports/REPORT.md) for the full write-up including
validation design, baselines, SHAP analysis, mispricing case studies, and
limitations.

---

## Honest limitations

1. **CatBoost was dropped mid-sweep** for compute budget — a re-run with longer
   runtime should include it.
2. **Wage target is still FM-estimated.** Real wages are paywalled (Capology);
   the wage model is honestly labelled as "FM-estimated wage prediction".
3. **91% match rate** between the master and Transfermarkt. The 9% unmatched
   are mostly youth / reserves outside TM's coverage; they remain in the dataset
   with `null` target.
4. **Interval widths are wide by design.** Real market values for players with
   identical measurable features really do vary ~5× multiplicatively. The UI
   converts the raw width into a confidence tier (High / Moderate / Low)
   because a scout needs a decision, not a disclaimer.
5. **Three seasons, one league.** Extending to La Liga or Serie A would require
   re-tuning the primary-position parser and the club alias dictionary.

---

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset attributions and third-party
trademarks in the same file.
