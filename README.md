![CI](https://github.com/ashutosh-kulkarni-dev/premier-league-value-modelling/actions/workflows/ci.yml/badge.svg)
# PL Insight — Premier League Player Value & Wage Model

A regression pipeline and static web app that predict Premier League players' **market value** (Transfermarkt €) and **weekly wage** from on-pitch performance and scouted football attributes, then flag players where the market diverges most from what measurable features predict.

> **Live demo:** https://premier-league-value-modelling-web.vercel.app/
---

## Headline metrics (season holdout)

Train on **2023-24 + 2024-25**, test on the held-out **2025-26** season — the model has never seen these players at that time point.

| Metric | Value model | Wage model |
|---|---:|---:|
| **R² (log target)** | **0.784** | **0.839** |
| MAE (log target) | 0.392 | 0.264 |
| MAE (native) | €7.3M | €15k / wk |
| **Median absolute % error** | **27.8%** | **13.5%** |
| Mean absolute % error | 49.0% | 51.0% |
| Spearman rank corr. | 0.874 | 0.934 |
| Test players (2025-26) | 1,015 | 1,008 |
| Winning algorithm | LightGBM | LightGBM |

Mean APE is tail-heavy — a small subset of low-value players produce extreme relative errors — so the **median is the honest central-tendency measure**. Rank correlation (Spearman) is high, which is what matters for a mispricing ranking.

---

## What the app shows

A single-page web app (shareable hash URLs, light/dark mode) with these views:

- **Dashboard** — held-out R², MAE and typical % error; the verdict and confidence breakdowns; a predicted-vs-actual scatter; and a live mispricing feed.
- **Players** — a searchable, sortable, position-filterable index of the squad.
- **Player profile** — point prediction, confidence tier, SHAP top drivers, market-vs-model residual, full Transfermarkt bio, value-over-time chart, season-by-season stats, and career history.
- **Underpays / Overpays** — players the market prices most below / above the model, with filters (club, position, verdict, minutes, mispricing %).
- **Leaders** — highest model-valued players by positional area.
- **Teams** — squad value totals per club, and a per-club squad page.
- **Compare** — two players side by side, including a value-over-time overlay.
- **Insights / About** — global SHAP importance, how the model is built and validated, and what confidence means.

Each scored player (2025-26) carries a **point prediction**, a **confidence tier**, the **SHAP drivers** of that prediction, a **verdict label** (excellent / good / fair / poor from the |% error|), and the **market-vs-model residual** that drives the mispricing boards.

---

## Confidence

Each prediction shows a confidence tier — the model's **backtested typical error** for players in the same **positional area** (GK / defence / midfield / attack) and **value tier**, measured on the holdout. A scout reads it as "for players like this, the model is usually within X%." High / Medium / Low thresholds are calibrated **per area**, because attackers and goalkeepers are inherently harder to value than defenders. The confidence table ships in `meta.json` and is applied client-side.

---

## Data pipeline

Three data sources feed a per-season, per-snapshot master (two snapshots per season — pre-season "start" and end-of-season "end"). Only the **2025-26 Premier League** roster (20 clubs) is published to the app.

### Football Manager attributes (per season)
- **2023-24** → FMInside FM24 scrape of the PL cohort
- **2024-25** → FM24 in-game HTML export after mid-season editor-data updates (transfers, CA/PA changes, aging)
- **2025-26** → FMInside FM26 scrape of the PL cohort

Technical, mental, physical and goalkeeping attributes plus `current_ability` / `potential` are carried per season. Goalkeeping attributes apply to GK rows only.

### Understat (per season)
- Minutes, matches, goals, assists, xG, non-penalty xG, xA, xG chain, xG buildup, key passes, shots — converted to per-90 rates during feature engineering.

### Transfermarkt (per snapshot date)
- **`players.csv`** → bio, primary position (`sub_position`), nationality
- **`player_valuations.csv`** → market value and current club at each snapshot (nearest datapoint within the season window), plus the full value-over-time trajectory
- **Target:** value = Transfermarkt `market_value_in_eur`; wage = FM `wage_pw` (TM doesn't publish wages reliably)
- **Primary position:** sourced from TM `sub_position`, mapped to the 8-bucket vocabulary {GK, CB, FB, DM, CM, AM, W, ST}, with a profile-scraper fallback for the handful TM misses

### FM ↔ Transfermarkt join
The FM/Understat master is joined to Transfermarkt by a **4–5 pass fuzzy matcher on normalised name + club** (`src/value_wage/sources/transfermarkt_kaggle.py`), progressively loosening criteria: exact name + club → club-set membership (handles loans) → unique name + age → token-subset (middle names) → distinctive-name, each with **per-season uniqueness guards** so common names (Emerson, Gabriel, Thiago) can't collide.

Name normalisation ASCII-folds **and transliterates baked-in Latin-extended letters** (Ø→o, æ→ae, ł→l, dotless ı→i, …), so players like "Ødegaard" are not silently dropped by naïve `encode("ascii","ignore")`. Unmatched rows (academy / fringe players with no TM Premier-League valuation) are kept as **predict-only** (NaN target) and shown with a "limited data" note.

---

## Feature set

The model consumes an explicit allowlist (`config/features.yaml`) — nothing enters the matrix via star-select. Current set: **54 features** (52 numeric + 2 categorical).

| Group | Count | Examples |
|---|---:|---|
| Biographical & availability | 7 | age, age_sq, height_cm, weak_foot, minutes, matches, log1p_minutes |
| Performance (Understat) | 8 | np_xg_p90, xa_p90, key_passes_p90, xg_chain_p90, xg_buildup_p90, shots_p90, log1p_goals, log1p_assists |
| FM technical | 10 | finishing, passing, dribbling, technique, marking, tackling, heading, long_shots, crossing, first_touch |
| FM mental | 14 | composure, decisions, vision, work_rate, positioning, teamwork, anticipation, leadership, concentration, determination, bravery, flair, aggression, off_the_ball |
| FM physical | 5 | pace, acceleration, strength, stamina, jumping_reach |
| FM aggregates | 6 | current_ability, potential, potential_gap, fm_technical_mean, fm_mental_mean, fm_physical_mean |
| Contract | 1 | months_to_contract_end |
| Market inflation (exogenous) | 1 | market_inflation_index |
| Categorical | 2 | primary_position, foot |

**Market inflation, done non-leakily.** `market_inflation_index` is a per-season transfer-market inflation factor (base 2023-24 = 1.0 → ~1.0 / 1.4 / 1.6) computed as the median market value across the **big-5 European leagues *excluding* the Premier League**. Because the reference population is disjoint from the PL players being predicted, it tracks market-wide inflation **without** leaking the held-out season's PL values.

The target and the FM money fields (`value_high`, `sell_value`, `release_clause`), along with any PL-cohort season aggregates derived from the targets, are **excluded** to prevent label leakage. The exclusion is enforced in `config/features.yaml`, in the training path (`_assert_no_leakage`), and by `tests/test_no_leakage.py`. Club and season identifiers are also excluded — they'd absorb the budget/market signal the project exists to explain.

---

## Modelling

- **Targets**: `log1p(market_value_eur)` and `log1p(wage_pw)` — right-skewed targets benefit from log-space training.
- **Algorithm**: LightGBM, selected over XGBoost, CatBoost and HistGradientBoosting in an Optuna bake-off.
- **Split**: season-based — train **2023-24 + 2024-25**, test **2025-26** (held out). A **GroupKFold on `player_id`** for the inner CV stops the same player landing in both inner-train and inner-val.
- **Hyperparameters**: Optuna TPE sampler + MedianPruner; best parameters pinned.
- **Confidence**: backtested typical % error by positional area × value tier (see **Confidence** above).
- **Leakage guards**: the target, FM money fields, and PL-derived season factors are excluded and asserted by both the training path and the test suite.

### Baselines

The production GBM is benchmarked against two simpler reference models on the same features, split, and `log1p` target, plus a per-position-median sanity floor:

- **ElasticNet** (regularized linear, self-tuned `alpha` + `l1_ratio` over GroupKFold) — L2 absorbs the many correlated features; L1 zeros out irrelevant ones.
- **Random Forest** (bagged trees) — nonlinearity without boosting.

Held-out 2025-26 R² (log target) — the ladder behaves as expected, and LightGBM earns its place:

| Model | Value R² | Wage R² |
|---|---:|---:|
| Median-by-position (floor) | 0.05 | −0.07 |
| ElasticNet | 0.73 | 0.75 |
| Random Forest | 0.70 | 0.80 |
| **LightGBM (production)** | **0.78** | **0.84** |

Reproduce with `vw compare run --target value|wage` (writes `reports/model_comparison_{target}.{csv,md}`).

---

## Known limitations

1. **Transfermarkt noise ceiling** — TM market values are crowdsourced; published studies place their inherent variance at ~20–30%. Part of the model's residual is this noise, not modelling error.
2. **Non-uniform market drift** — the market inflated materially from 2024-25 to 2025-26, and *unevenly*: the most expensive players inflated more than squad players. The exogenous `market_inflation_index` lets the tree track this per-player, but a single market has no perfect scalar; typical players are well-centred while the very top end still reads slightly low. Future seasons require the index to be recomputed.
3. **Feature coverage** — academy / fringe players absent from Transfermarkt's directory are kept as predict-only (no labelled target) and don't contribute to the metrics.

---

## Repository layout

```
├── config/            # YAML configs (feature allowlist, splits, model params)
├── data/
│   ├── raw/           # Source datasets — Transfermarkt Kaggle dump is gitignored
│   └── processed/     # Trained models, feature matrices, bridge/lookup tables
├── reports/           # SHAP CSVs, metrics, mispricing exports, figures
├── src/value_wage/    # Library code — data, sources, features, train, confidence, evaluate, export
├── tests/             # Pytest suite — split integrity, feature pipeline, join canaries, no-leakage guards
├── web/               # Static single-page app (vanilla HTML + JS, charts from CDN)
│   └── data/          # players.json (slim index), player/<id>.json (lazy detail), meta.json, mispricing.json
├── Makefile           # One-command targets: data, features, tune, train, evaluate, export, serve
├── pyproject.toml     # Project metadata and dependencies
└── vercel.json        # Static deploy config (serves web/)
```

### Web data contract
- **`players.json`** — a slim per-season row for every player: predictions, verdict, scalars, and a small bio subset (image, position, contract, TM values). Loaded on first paint.
- **`player/<id>.json`** — lazy per-player detail (full bio, career, season stats, value trajectory, SHAP), fetched on demand when a profile or comparison opens.
- **`meta.json`** — model metadata, metrics, global SHAP, feature list, and the per-area confidence table.
- **`mispricing.json`** — ranked under/over-paid boards per target.

See `web/README.md` for the full schema.

---

## Local development

```bash
# install
pip install -e ".[dev]"

# run the test suite
make test            # or: python -m pytest

# preview the app (serves web/ with the committed JSON bundle)
make serve           # or: python -m http.server 8000 --directory web
# then open http://localhost:8000/
```

Rebuilding the pipeline from raw data (`make data features tune train evaluate export`, or the `vw` CLI) requires the Transfermarkt Kaggle dump (`https://www.kaggle.com/datasets/davidcariboo/player-scores`), the FMInside pages for the three FM editions, and the Understat per-season aggregates. See the `Makefile` for the end-to-end targets.

---

## Tech stack

Python 3.13 · pandas · scikit-learn · LightGBM · SHAP · Optuna · pydantic · Typer CLI · requests + BeautifulSoup for scraping · pytest · vanilla HTML/JS for the app · Vercel for static hosting.

---

## Licence

See `LICENSE`.
