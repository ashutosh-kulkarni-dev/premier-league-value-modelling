# Capstone Plan — Premier League Player Value & Wage Model

**Thesis.** Build a reproducible, well-evaluated regression system that predicts a Premier League player's **market value** and **weekly wage** from on-pitch performance (FBref, Understat) and scouted attributes (Football Manager 26). Use the model's residuals to surface **under- and over-valued players** — the output a recruitment analyst would actually consume.

**Why this problem.**
- Clear, numeric targets (`value_high`, `wage_pw`) with real-world meaning → unambiguous success metrics.
- Two complementary targets let you tell a richer story (market valuation vs. club wage policy diverge).
- Rich feature space (performance + attributes + biographical) → non-trivial feature engineering + interpretability work.
- Natural downstream artefact (mispricing board) → portfolio-ready demo, not just a notebook.

**Scope guard-rails.**
- Premier League only (2023-24 → 2025-26 Understat; FM26 attribute snapshot).
- One model family per target, taken end-to-end with care, instead of a shallow bake-off.
- No scraping work in this phase — treat `data/interim/*.parquet` as the frozen input.

---

## 1. Dataset & target

Primary table: `scout-pipeline/data/interim/master_fm26_pl.parquet` — 1,669 rows × 89 cols, FM attributes joined to Understat performance per player.

| Target | n (non-null) | Median | Max | Transform |
|---|---|---|---|---|
| `value_high` (€, transfermarkt-style) | 1,162 | €60M | €476M | `log1p` |
| `wage_pw` (£/week) | 1,242 | £81k | £521k | `log1p` |

**Why log-transform.** Both targets are right-skewed over ~3 orders of magnitude. Modelling in log-space makes the loss care about *multiplicative* error ("20% off" is the same mistake for a youth-team player and a superstar), which is how the market itself behaves and how an analyst would critique the model. Reported errors convert back to the original scale for the business audience.

**Why ~500 missing targets are fine, not a blocker.** Missingness is almost entirely squad/youth players without a public market value — not informative-missing on the features. We train on the labelled rows and *predict* for the unlabelled ones; those predictions are the mispricing board for players the market hasn't priced.

---

## 2. Objectives & success criteria

**Modelling objective.** Minimise MAE on `log1p(target)` with time-honest validation. Secondary: calibration (predicted vs. actual deciles) and residual stability across positions.

**Target metrics (held-out test, back-transformed to € / £):**

| Metric | Value model | Wage model | Why this bar |
|---|---|---|---|
| MAE (log) | ≤ 0.35 | ≤ 0.30 | ≈ ±35% / ±30% median multiplicative error — competitive with public baselines for transfer-value models. |
| R² (log) | ≥ 0.75 | ≥ 0.80 | Age + minutes alone get you ~0.5; beating that materially shows the attributes earn their keep. |
| Rank correlation (Spearman) | ≥ 0.85 | ≥ 0.88 | Scouts care about ordering more than absolute €; this is the metric a user actually feels. |

**Deliverable objective.** A mispricing report + a lightweight Streamlit app where the user picks a player and sees: predicted value, actual value, residual percentile, top SHAP drivers, and 5 nearest comparables.

---

## 3. Data pipeline

```
raw parquet  →  load & validate  →  feature engineering  →  model-ready matrix
     │                                      │
     │                                      ├── per-90 performance features
     │                                      ├── FM attribute groups (technical, mental, physical)
     │                                      ├── primary position (parsed from CSV string)
     │                                      ├── age-curve features (age, age², years_to_peak)
     │                                      └── contract_remaining (months)
     ↓
train / val / test split by SEASON (not random)
```

**Validation strategy — group-aware, time-aware.**
- **Test**: 2025-26 season rows (most recent, mimics "predict for the upcoming window").
- **Validation**: 2024-25.
- **Train**: 2023-24 and prior.
- A player appearing in multiple seasons is kept in a single fold (GroupKFold on `player_id` *within* train) to prevent the model from memorising a specific player's price point.

**Why not random KFold.** Random splits let the model peek at a player's own 2024-25 wage when predicting their 2023-24 wage — leakage that would inflate R² by 10+ points and lie about generalisation.

---

## 4. Feature engineering

| Group | Examples | Rationale |
|---|---|---|
| **Performance per 90** | `np_xg_p90`, `xa_p90`, `key_passes_p90`, `xg_chain_p90` | Rate stats, not totals — a bench player with 90 min and 1 goal shouldn't look like Haaland. |
| **Volume** | `log1p(minutes)`, `matches` | The market pays for sustained availability; log because the first 500 min matter far more than the last 500. |
| **FM technical** | finishing, passing, dribbling, first_touch, technique, long_shots, heading, crossing, tackling, marking | Captures skills performance stats under-measure (e.g. a backup with elite finishing scored on limited minutes). |
| **FM mental** | composure, decisions, vision, off_the_ball, work_rate, anticipation | Known to drive wage/value more than physicals for midfielders. |
| **FM physical** | acceleration, pace, stamina, strength, jumping | Position-dependent weight; pace is a known value multiplier for wingers/forwards. |
| **Attribute aggregates** | mean/max within group, `current_ability`, `potential`, `potential − CA` (growth headroom) | Non-linear interactions are easier for a tree to find when the summary is pre-computed. |
| **Biographical** | `age`, `age²`, `height_cm`, `foot` (one-hot), `weak_foot` | Age is the single strongest predictor of value — the age curve peaks around 26-27 for value, later for wage. |
| **Contract** | `months_to_contract_end` | Expiring contracts suppress market value sharply (Bosman); including it isolates "value given a fair contract situation". |
| **Position** | parsed primary role from `positions` CSV → `{GK, CB, FB, DM, CM, AM, W, ST}` | Positional baselines differ by an order of magnitude; one-hot lets the tree split on them cleanly. |

**Not using.** Nationality (small-sample bias, risks encoding passport premiums the project shouldn't amplify), player name, club (too close to the wage target — a club dummy would absorb most of the signal we want to explain).

---

## 5. Modelling approach

### 5.1 Baselines (must be written first — the model has to beat them)

1. **Median by position** — a sanity floor.
2. **Linear regression on age + log(minutes) + position** — the "obvious" model.
3. **Ridge on all features (standardised)** — a competent linear benchmark.

If LightGBM doesn't materially beat ridge, something is wrong with the features or the split, and that's a finding worth reporting.

### 5.2 Main models — a disciplined boosting bake-off

Rather than committing to one booster up-front, fit four and let the validation set decide. All four are tree-based gradient boosters; they differ in how they split, how they handle categoricals/NaN, and how they regularise — exactly the axes on which a fair comparison teaches something.

| Model | Why it's in the bake-off |
|---|---|
| **LightGBM** | Leaf-wise growth, native NaN + categorical handling, fastest tuning at this size. Expected to be strong on the FM attribute block (lots of missing values). |
| **XGBoost** | The reference boosting implementation; level-wise growth and strong regularisation make it the usual winner on small, noisy datasets like ours (~1k labelled rows). Native NaN handling; categorical support is newer but functional. |
| **CatBoost** | Ordered boosting reduces target leakage on small samples; best-in-class native categorical handling (useful for `primary_position` and `foot`). Typically the most forgiving default — low tuning effort to a decent score. |
| **HistGradientBoostingRegressor (sklearn)** | A boring, fully-sklearn, in-process baseline. Fast, native NaN, no extra install. Keeps the other three honest — if nothing beats it meaningfully, the fancy libraries aren't earning their place. |

**What is *not* in the bake-off and why.**
- **AdaBoost / plain GradientBoostingRegressor** — strictly dominated by the four above on tabular regression; included only in a one-cell appendix for completeness.
- **Deep models (TabNet, FT-Transformer, MLP)** — 1k–1.6k labelled rows is well below the regime where DL beats GBMs on tabular (Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning on typical tabular data?"); interpretability story is harder; adds a GPU dependency for no measurable gain.
- **Stacking / blending of the four** — fit as a tiebreaker *only* if the top two models are within 1 MAE-point of each other on validation. Marginal gain, large interpretability cost; included honestly rather than hidden.

**Shared training recipe (identical across models for a fair comparison).**
- **Objective:** `regression_l1` / MAE on `log1p(target)`. Robust to the handful of extreme-outlier contracts (Haaland, De Bruyne) without needing to winsorise.
- **Preprocessing:** the same sklearn `ColumnTransformer` for all four — median imputation for linear baselines only; for the boosters, NaNs are passed through where supported (LightGBM, XGBoost, CatBoost, HGBR all handle them natively).
- **Hyperparameter search:** Optuna, 50 trials per model, Bayesian TPE sampler + MedianPruner, inner 3-fold GroupKFold on the train+val slab, scoring on log-MAE. Per-library search spaces (standard ranges from the Optuna + library docs):
  - LightGBM: `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`.
  - XGBoost: `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`.
  - CatBoost: `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `border_count`.
  - HGBR: `max_leaf_nodes`, `min_samples_leaf`, `learning_rate`, `l2_regularization`, `max_iter`.
- **Early stopping** on the val fold with 100 rounds patience for all libraries that support it (LGBM, XGB, CatBoost; HGBR uses `early_stopping=True`).
- **Seeds:** single `SEED=42` passed into every stochastic call per library — results are bit-reproducible run-to-run.

**Model-selection rule (decided before results, to avoid cherry-picking).**
1. Rank the four tuned models by held-out **test MAE on log-target** (primary metric).
2. If the top model's advantage over #2 is ≥ 0.02 log-MAE (~2% multiplicative error), it wins outright and is the "main model" everything downstream (SHAP, mispricing board, Streamlit) uses.
3. Otherwise, prefer the model with (a) tighter residual bands by position, then (b) faster inference, then (c) simpler install footprint. All four are tested; one is promoted. The others are reported in the comparison table and otherwise set aside.
4. The **quantile intervals** (§5.3) are refit using the winning model's library — kept in-framework rather than mixing stacks.

**Why this is more defensible than picking LightGBM up-front.** A single-model project invites the reviewer's question "did you try X?"; a disciplined four-way bake-off with a pre-declared selection rule answers it before it's asked. The comparison table itself is a portfolio asset — it shows understanding of *why* these libraries differ, not just that they exist.

### 5.3 Uncertainty — quantile regression

Fit two additional models at α=0.1 and α=0.9 using the **winning library's** quantile objective (LightGBM `quantile`, XGBoost `reg:quantileerror`, CatBoost `Quantile:alpha=`, or HGBR `loss="quantile"`). If the winning library ever lacks quantile support, fall back to a **conformal prediction** wrapper (`mapie`) around point predictions — distribution-free, library-agnostic, and still interval-valid.

Report a prediction interval per player. This matters for the mispricing use case: a player €20M below prediction with a tight interval is a real signal; the same gap with a wide interval is noise.

---

## 6. Interpretability & residual analysis

- **Global SHAP** — mean absolute SHAP per feature, with a positional breakdown (feature importance for strikers ≠ for CBs).
- **Local SHAP** — per-player waterfall for the top 20 mispriced players, used as evidence in the final report.
- **Residual diagnostics** — residuals vs. age, vs. minutes, vs. position; QQ plot; no autocorrelation check needed (no time series within a player at this stage).
- **Mispricing board** — rank players by `(actual − predicted) / predicted_sigma`; split into "undervalued" (market < model) and "overvalued" (market > model). Case-study the top 5 of each in the report.

**Honesty clause.** The report must call out that "undervalued by the model" ≠ "actually undervalued by the market" — the model is one view, and systematic residuals (e.g. the market consistently paying more for South American wingers than our model expects) are often **the model missing a real factor**, not the market being wrong. This framing is what separates a credible analyst from a hype piece.

---

## 7. Project layout

```
value-wage-model/
├── data/
│   ├── raw/            # symlink or copy of scout-pipeline/data/interim
│   └── processed/      # model-ready parquet, versioned by date
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baselines.ipynb
│   ├── 04_lgbm_tuning.ipynb
│   ├── 05_shap_and_residuals.ipynb
│   └── 06_mispricing_report.ipynb
├── src/value_wage/
│   ├── config.py           # paths, seeds, feature lists
│   ├── data.py             # load + split
│   ├── features.py         # pure functions, unit-tested
│   ├── models.py           # baseline + lgbm wrappers
│   ├── train.py            # CLI entry — reproducible training run
│   ├── evaluate.py         # metrics + plots
│   └── explain.py          # SHAP + mispricing board
├── tests/                  # pytest; feature engineering is pure → easy to test
├── reports/
│   ├── figures/
│   └── capstone_report.pdf
├── app/streamlit_app.py    # final demo
├── pyproject.toml
├── Makefile                # make data, make train, make report, make app
├── README.md
└── techstack.md
```

**Why a `src/` package, not notebook-only.** Notebooks for narrative, package for the code that actually runs. Any function used in two notebooks lives in `src/`. This is table-stakes for anything that calls itself "professional".

---

## 8. Execution timeline (indicative, ~4 weeks part-time)

| Week | Milestone | Done when |
|---|---|---|
| **1** | EDA + feature spec frozen | `01_eda.ipynb` + `02_feature_engineering.ipynb` reviewed; feature list in `config.py` |
| **2** | Baselines + untuned run of all four boosters | Metrics table for 3 baselines + untuned LGBM / XGB / CatBoost / HGBR on val; split audited for leakage |
| **3** | Full Optuna tuning on each booster + select winner + quantile models + SHAP | Bake-off table in report; winner selected by pre-declared rule; SHAP plots saved; quantile intervals calibrated (coverage ≈ 80%) |
| **4** | Mispricing report + Streamlit app + writeup | PDF report + running app; README reproduces end-to-end with `make all` |

Buffer week assumed for the inevitable data-quality discovery.

---

## 9. Risks and how they're handled

| Risk | Mitigation |
|---|---|
| **Target leakage via club dummies / wage into value model** | Explicit feature allowlist in `config.py`; `wage_pw` is never a feature for the value model and vice versa. |
| **Small test set (one season)** | Report bootstrap CIs on test metrics; show stability across the 3 Understat seasons via rolling-origin eval. |
| **FM attributes are subjective and from one snapshot** | Report model performance *with* and *without* FM attributes; if gain is small, honest finding; if large, caveat that it depends on FM's scouting quality. |
| **"Value" is a transfermarkt estimate, not a transaction** | State this prominently; frame the model as "predicting the public valuation", not "predicting what a club would pay". |
| **1.6k rows is modest** | Favour regularisation, cross-validated tuning, honest intervals; avoid flashy deep models that will overfit. |
| **Position parsing from `AMC,AMR,DC,DR,...` strings** | Deterministic rules + unit tests; a handful of edge cases hand-reviewed. |

---

## 10. Definition of done

- [ ] `make all` reproduces every number in the report from the frozen parquet.
- [ ] Test metrics meet the targets in §2, or the gap is honestly explained.
- [ ] SHAP global + local plots exist for both targets.
- [ ] Mispricing board (top-20 under, top-20 over) exported as CSV + rendered in the app.
- [ ] Streamlit app runs locally with one command; player picker + prediction + interval + SHAP + comparables all work.
- [ ] README has: problem statement, data provenance, how to run, results table, honest limitations.
- [ ] Code passes `ruff` + `mypy --strict` on `src/`; tests pass.
