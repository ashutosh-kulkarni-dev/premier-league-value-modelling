# Premier League Player Value & Wage Model — Capstone Report

**Author:** Ashutosh Bhasin
**Dataset:** 1,669 Premier League players × 3 seasons (2023-24, 2024-25, 2025-26)
**Deliverables:** tuned regression models, calibrated prediction intervals, ranked mispricing board, static HTML UI.

---

## 1. Thesis

Predict Premier League players' **market value** (Transfermarkt €) and **weekly wage**
(FM-estimated £/wk) from on-pitch performance (FBref, Understat) and scouted attributes
(Football Manager 26). Treat the model's **residuals** as the deliverable: they quantify
how much a player's actual market valuation deviates from what the measurable features
alone would predict, exposing the "human premium" (manager demand, auction dynamics,
expiring contracts, injury rumours, hype) that mechanical features can't see.

The project is deliberately **not** optimised toward the lowest possible test error —
chasing that would mean overfitting to the exact human/contextual factors we want the
model to leave on the table.

---

## 2. Data

| Source | Seasons | Rows | What it brings |
|---|---|---|---|
| Understat (FBref) | 2023-24, 2024-25, 2025-26 | 1,669 player-seasons | goals, assists, xG, xA, xG chain/buildup, minutes |
| FMinside (FM26 scrape) | FM26 snapshot | 1,242 PL players | 40+ scouted attributes, current_ability, potential, age, contract end |
| Transfermarkt (Kaggle `davidcariboo/player-scores`) | historical | 656k valuations | the real `market_value_in_eur` target |

### Target expansion

For each `(player, season)` we take **two Transfermarkt valuation snapshots** — one at the
start of the season (closest to 01 Aug) and one at the end (closest to 31 May). This
doubles the labelled dataset to ~2,000 rows and lets the model learn how within-season
age + contract decay move market value.

### The join: 4 passes, decreasing confidence

Transfermarkt and FM don't share identifiers; we fuzzy-join on name + club + season + age.
The join degrades gracefully through four passes:

1. **Exact name + exact club + season** (89% of matches).
2. **Exact name + season + club ∈ TM's known clubs that season** — catches mid-season
   loans. Verified on Jadon Sancho (23/24 end snapshot: master says Man Utd, TM says
   Dortmund per the January loan; pass 2 ✓).
3. **Normalised name + season + age ±1** — catches accent-stripping edge cases.
4. **Token-subset match within cohort + age ±1.5** — catches hyphenated and partial-name
   cases ("Ezri Konsa" ↔ "Ezri Konsa Ngoyo"; "Matty Cash" ↔ "Matthew Cash" via a
   diminutive canonicalisation table).

**Final match rate: 91% of 3,338 rows.** The 9% unmatched are mostly youth / reserves
with no Transfermarkt valuation record in the season window; they remain in the dataset
with `null` target and are used only for the predict-only mispricing board.

### Leakage guard (belt + braces)

The FM `value_high` field is a game-engine estimate computed from `current_ability`,
`potential`, `age`, and `contract_until`. On a value-prediction task its inclusion would
tautologically dominate the model. Three independent guards prevent this:

1. `config/features.yaml` lists `value_high`, `sell_value`, `release_clause` under
   `excluded`.
2. A runtime assertion in `train.py:_assert_no_leakage` raises `ValueError` if any
   excluded column appears in the feature matrix at fit time.
3. A unit test (`tests/test_no_leakage.py`) rebuilds the feature list and asserts the
   excluded set is disjoint from the model inputs.

---

## 3. Feature engineering

48 features in five groups. Snapshot-aware: time-dependent features (`age`, `age²`,
`months_to_contract_end`) are recomputed per snapshot so start- and end-of-season rows
differ correctly.

| Group | Count | Examples |
|---|---|---|
| Biographical | 4 | age, age², height, weak_foot |
| Availability / volume | 4 | log1p(minutes), matches, log1p(goals), log1p(assists) |
| Performance per 90 | 6 | np_xg_p90, xa_p90, key_passes_p90, xg_chain_p90, xg_buildup_p90, shots_p90 |
| FM technical attrs | 10 | finishing, passing, dribbling, first_touch, technique, … |
| FM mental attrs | 14 | composure, decisions, vision, work_rate, off_the_ball, … |
| FM physical attrs | 5 | acceleration, pace, stamina, strength, jumping_reach |
| FM aggregates | 5 | fm_{technical,mental,physical}_mean, current_ability, potential, potential_gap |
| Contract | 1 | months_to_contract_end (recomputed per snapshot) |
| Position (one-hot) | 1 | primary_position ∈ {GK, CB, FB, DM, CM, AM, W, ST} |

**Deliberately excluded:** nationality (small-sample bias / passport premiums), player
name, club, team (would absorb wage signal we want to explain).

---

## 4. Validation strategy

Season-based split; `GroupKFold` on `player_id` for inner CV.

- **Train:** 2023-24 (both snapshots)
- **Val:**   2024-25 (both snapshots)
- **Test:**  2025-26 (both snapshots) — fully held out

Why not random KFold: a random split would let the model see Player X's 2024-25 wage
while predicting X's 2023-24 wage — leakage that would inflate R² by ~10 points. The
season split is time-honest; GroupKFold inside train prevents the same player appearing
in both inner-train and inner-val folds during Optuna tuning.

---

## 5. Modelling: a 3-booster bake-off

Four boosters planned (`LGBM`, `XGBoost`, `CatBoost`, `HGBR`); **CatBoost dropped mid-sweep
for compute reasons** — each trial required ~5 min on this dataset, over the available
background-job budget. All four would be included in a repeat with longer runtime
allowance.

All boosters tuned under one Optuna recipe:

- Objective: `regression_l1` (MAE) on `log1p(target)`.
- Sampler: TPE seeded at 42.
- Pruner: MedianPruner with 5-trial warmup.
- Inner CV: 3-fold GroupKFold on `player_id`.
- Persisted: SQLite + MLflow (resumable, auditable).

### CV log-MAE (train+val, inner CV)

| | lgbm | xgb | hgbr |
|---|---|---|---|
| **value** | **0.5129** | 0.5153 | 0.5270 |
| **wage** | **0.3617** | 0.3642 | 0.3687 |

**LightGBM wins both targets** — but by razor-thin margins (<0.01 log). On a different
random seed either XGB could have won; the three boosters are statistically tied. LGBM
gets the podium by consistency.

### Baselines (test = 2025-26, back-transformed from log)

| Target | Model | MAE log | R² log | Spearman |
|---|---|---|---|---|
| value | Position median | 0.844 | **−0.01** | 0.22 |
| value | Ridge | 0.464 | 0.722 | 0.84 |
| **value** | **LightGBM (tuned)** | **0.398** | **0.782** | **0.876** |
| wage | Position median | 0.758 | **−0.05** | 0.11 |
| wage | Ridge | 0.364 | 0.792 | 0.87 |
| **wage** | **LightGBM (tuned)** | **0.257** | **0.858** | **0.941** |

**Reading:**
- Position-median R² near zero — expected. Within-position variance is huge.
- Ridge is a strong floor (R² 0.72 / 0.79) — age + minutes + FM composites give
  substantial linear signal.
- GBM beats Ridge by ~5–7 R² points — meaningful, not dramatic. The GBM finds
  non-linear interactions (position × age × attributes) but doesn't dominate.

---

## 6. Calibrated prediction intervals (MAPIE conformal)

LightGBM's native quantile objective produced badly miscalibrated intervals (actual
coverage 31–47% at a nominal 80%). Replaced with MAPIE **cross-conformal (CV+)** using
GroupKFold on `player_id`:

| Target | Nominal | Achieved | Mean width (log) |
|---|---|---|---|
| value | 80% | **89.6%** | 1.65 |
| wage  | 80% | **90.4%** | 1.12 |

Both slightly over-covered (common for CV+ on small data). The intervals are honest:
coverage ≥ nominal is the distribution-free guarantee. A €20M value prediction has an
80% interval of roughly €7M–€55M — reflecting the real-world variance in market valuations
for players with identical measurable features.

Mispricing z-scores (`(actual − predicted) / sigma_from_interval`) now cluster at
**|z| ≈ 3–6** for the top entries — a sensible range. Previously (with the broken
quantile intervals) z-scores reached 34, flagging fake outliers.

---

## 7. SHAP (what the model actually weights)

### Value target — top 10 features

| Feature | Mean \|SHAP\| | Comment |
|---|---|---|
| age | 0.244 | biological, not FM-derived |
| potential | 0.244 | FM composite |
| current_ability | 0.230 | FM composite |
| age² | 0.098 | age curve non-linearity |
| minutes | 0.083 | availability |
| **xg_chain_p90** | 0.069 | **real on-pitch stat** |
| matches | 0.048 | availability |
| fm_technical_mean | 0.038 | FM aggregate |
| fm_physical_mean | 0.035 | FM aggregate |
| xg_buildup_p90 | 0.034 | real on-pitch stat |

**No single feature dominates (max 24%). Age / potential / current_ability are
co-equal.** This is a meaningful change from an earlier version of the project where the
target was FM's own `value_high` field, and `current_ability` dominated SHAP at 56%
because `value_high` is literally computed from CA inside FM. Switching to Transfermarkt
restored a balanced attribution.

### Wage target — top 10 features

Still FM-attribute-heavy (contract-negotiation data lives inside FM) but no single
feature >25%. `current_ability` (25%), `potential` (17%), `age` (9%), `minutes` (7%),
`months_to_contract_end` (7%).

### What this tells us about the model

- The model learned **real age curves** — age² carries 10% of SHAP, meaning diminishing-
  returns past the peak.
- Real stats (**xg_chain_p90**) show up in the top-6, meaning Understat performance is
  pulling its weight even against the FM composites.
- The **contract Bosman effect** emerges from data: `months_to_contract_end` is a top-5
  feature on the wage model and visibly influences the value model too.

---

## 8. The mispricing board

The headline deliverable. For each test-season row, compute
`residual = actual − predicted` on the log scale, divide by the conformal interval's
implied sigma, rank by signed z.

### `market_overpays` (positive z — actual > predicted)

Youth prospects dominate: Kaye Furo (€8M), Romain Esse, Luis Guilherme, Alysson Edward,
Tyrique George, Veljko Milosavljevic. These are academy players with little playing
time; Transfermarkt has speculative values the model (trained on stats they haven't
produced) can't justify. **Not necessarily "overpaid"** — rather, the model has
nothing to go on, so the market's forward-looking premium is the signal.

Edge case: Chris Wood (€8M, model €1.5M) — a 33-year-old striker; the model's age
penalty is harsh.

### `market_underpays` (negative z — actual < predicted)

Out-of-favour squad players and injured regulars dominate: Mateus Mané (€250k, model
€8M), Pablo (€1.5M, model €12M), Solly March (€1M, model €9M — long-term injury has
crashed TM value), James Milner, Idrissa Gueye.

**The residuals are telling a story the features can't see:** injury status, locker-
room position, contract leverage. That is the project's framing — the mispricing
board is a **tool for investigation**, not a declaration of market error.

---

## 9. Deliverables

### Model artefacts

```
data/processed/models/
├── bakeoff_cv.csv                        # the 3-booster comparison
├── winners.json
├── value__{lgbm,xgb,hgbr}__best_params/  # Optuna best hyperparameters
├── wage__{lgbm,xgb,hgbr}__best_params/
├── value__lgbm__tuned/                   # serialised production model
└── wage__lgbm__tuned/
```

### Reports

```
reports/
├── baselines.csv                                       # ridge + position-median
├── metrics_value_lgbm.json                             # tuned-model metrics
├── metrics_wage_lgbm.json
├── metrics_calibrated_{value,wage}_lgbm.json           # conformal metrics + coverage
├── shap_importance_{value,wage}_lgbm.csv               # global SHAP
├── mispricing_{value,wage}_lgbm/                       # ranked boards (full / under / over)
├── figures/                                            # pred-vs-actual, residual, SHAP plots
└── REPORT.md                                           # this file
```

### HTML UI

```
web/
├── index.html         # reference UI (replace with styled version)
├── README.md          # data contract
└── data/
    ├── players.json       # per-player predictions + SHAP (1.8 MB)
    ├── meta.json          # model metadata, global SHAP, metrics
    └── mispricing.json    # ranked boards
```

The HTML UI has **zero backend dependency**. `index.html` + `data/*.json` is the entire
shipping surface. Serve with any static web server (`python -m http.server` works) or
open the file directly in a browser.

---

## 10. Reproducing from scratch

```powershell
cd value-wage-model
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Data
.\.venv\Scripts\python.exe -m value_wage.cli data build
.\.venv\Scripts\python.exe -m value_wage.cli features build

# Tune (~90 min — split into per-pair runs due to 10-min bg caps)
.\.venv\Scripts\python.exe -m value_wage.cli tune one --target value --model lgbm --trials 30
# repeat for xgb/hgbr value and lgbm/xgb/hgbr wage

# Train tuned, evaluate, SHAP, mispricing
.\.venv\Scripts\python.exe -m value_wage.cli train tuned --target value --model lgbm --params-json data/processed/models/value__lgbm__best_params/params.json
.\.venv\Scripts\python.exe -m value_wage.cli train tuned --target wage  --model lgbm --params-json data/processed/models/wage__lgbm__best_params/params.json
.\.venv\Scripts\python.exe -m value_wage.cli evaluate run --target value
.\.venv\Scripts\python.exe -m value_wage.cli evaluate run --target wage
.\.venv\Scripts\python.exe -m value_wage.cli calibrate run --target value --model lgbm --method cross
.\.venv\Scripts\python.exe -m value_wage.cli calibrate run --target wage  --model lgbm --method cross
.\.venv\Scripts\python.exe -m value_wage.cli mispricing build --target value
.\.venv\Scripts\python.exe -m value_wage.cli mispricing build --target wage

# Build the static UI bundle
.\.venv\Scripts\python.exe -m value_wage.cli export web
```

Every run is seeded (`SEED=42`); the input Transfermarkt dump is SHA-256 hashed on
ingestion. The same inputs produce bit-identical outputs.

---

## 11. Limitations (honest)

1. **CatBoost dropped from the bake-off** for compute reasons. Prior work shows
   CatBoost often leads on small tabular data; a repeat with a longer compute budget
   should include it.
2. **Wage target is still FM-estimated.** Real wage data from Capology is paywalled;
   Transfermarkt doesn't publish wages reliably. Reports frame this honestly: the wage
   model predicts FM's wage estimate, which is in the right ballpark but not an audited
   figure.
3. **91% match rate** between the FM/Understat master and TM. The 9% unmatched are
   almost all youth players outside TM's coverage window; they're kept in the dataset
   with `null` target.
4. **Selection bias in the "overpaid youth" cohort.** TM assigns speculative values to
   prospects based on hype that the model has no way to see. These residuals are more
   "the model lacks a feature" than "the market is wrong".
5. **Interval widths are wide** (log 1.65 for value ≈ ~5× multiplicative). This is
   honest, not a bug: Transfermarkt values for players with identical measurable
   features really do vary this much due to the human factors the project frames as
   residual signal.
6. **One league, three seasons.** Expanding to other leagues would require re-tuning
   the primary-position parsing and the TM club alias dictionary.

---

## 12. Framing (the pitch)

> We built a regression model to predict Premier League players' Transfermarkt market
> values and FM-estimated wages from on-pitch performance and scouted attributes. The
> model captures the measurable baseline strongly (R² 0.78 value / 0.86 wage). Its
> **residuals quantify how much a player's actual market valuation deviates from what
> their measurable features alone would predict — a tool for identifying players where
> non-measurable factors (coaching fit, panic premiums, expiring contracts, injuries,
> hype) are moving the price.** We ship a ranked mispricing board with calibrated 80%
> prediction intervals and a static HTML UI for interactive exploration.

That framing — "the residual is the deliverable, not an error" — is what distinguishes
the project from "we built an accuracy-chasing regression".
