![CI](https://github.com/ashutosh-kulkarni-dev/premier-league-value-modelling/actions/workflows/ci.yml/badge.svg)
# PL Insight — Premier League Player Value & Wage Model

A regression pipeline and static web UI that predict Premier League players' **market value** (Transfermarkt €) and **weekly wage** from on-pitch performance and scouted football attributes, then flag players where the market diverges most from what measurable features predict.

> **Live demo:** https://premier-league-value-modelling-web.vercel.app/

The project treats the model's **residuals as insight**, not as errors to minimise — residuals quantify the "human premium" that mechanical features can't see: manager demand, auction dynamics, expiring contracts, injury rumours, hype. The mispricing board is a tool for investigation, not a declaration of market error.

---

## Headline metrics (honest season holdout)

Train on **2023-24 + 2024-25**, test on **2025-26**. Metrics are computed on the held-out season — the model has never seen these players at that time point. Intervals are MAPIE cross-conformal (CV+).

| Metric | Value model | Wage model |
|---|---:|---:|
| **R² (log target)** | **0.779** | **0.839** |
| MAE (log target) | 0.396 | 0.276 |
| MAE (native) | €7.3M | €17k / wk |
| **Median absolute % error** | **27.5%** | **14.9%** |
| Mean absolute % error | 49.3% | 51.1% |
| Spearman rank corr. | 0.874 | 0.929 |
| **80% interval coverage** | **89.6%** | **88.0%** |
| Test players (2025-26) | 1,015 | 1,008 |
| Winning algorithm | LightGBM | LightGBM |

Mean APE is tail-heavy — a small subset of low-value players produce extreme relative errors — so the **median is the honest central-tendency measure**. Rank correlation (Spearman) is high, which is what matters for a mispricing ranking.

> **Note on honesty:** earlier versions of this README reported a value R² of 0.91 / 9.6% median APE. Those figures were inflated by a season-level market-value feature derived from the held-out season's own targets (label leakage). That feature has been removed and banned (see **Feature set**); the numbers above are the leakage-free result.

---

## What the UI shows

Each scored player (2025-26) carries:
- **Point prediction** — market value (€) and weekly wage
- **80% prediction interval** — conformal lower/upper bounds
- **SHAP top drivers** — the features most pushing that specific prediction up and down
- **Verdict label** — excellent / good / fair / poor, from the actual-vs-prediction gap and whether the actual falls inside the interval
- **Market-vs-model residual** — the headline mispricing signal
- **Context from Transfermarkt** — bio, transfer/career history, season-by-season appearances, and the full market-value trajectory

The dashboard also surfaces:
- Live held-out R², MAE, and interval coverage
- Most under-paid and most over-paid players, with filters (club, position, verdict, minutes, mispricing %)
- Global SHAP feature importance

---

## Data pipeline

Three data sources feed a per-season, per-snapshot master (two snapshots per season — pre-season "start" and end-of-season "end").

### Football Manager attributes (per season)
- **2023-24** → FMInside FM24 scrape of the PL cohort
- **2024-25** → FM24 in-game HTML export after mid-season editor-data updates (transfers, CA/PA changes, aging)
- **2025-26** → FMInside FM26 scrape of the PL cohort

Technical, mental, physical and goalkeeping attributes plus `current_ability` / `potential` are carried per season. Goalkeeping attributes apply to GK rows only.

### Understat (per season)
- Minutes, matches, goals, assists, xG, non-penalty xG, xA, xG chain, xG buildup, key passes, shots — converted to per-90 rates during feature engineering.

### Transfermarkt (per snapshot date)
- **`players.csv`** → bio, primary position (`sub_position`), nationality
- **`player_valuations.csv`** → market value and current club at each snapshot (nearest datapoint within the season window)
- **Target:** value = Transfermarkt `market_value_in_eur`; wage = FM `wage_pw` (TM doesn't publish wages reliably)
- **Primary position:** sourced from TM `sub_position`, mapped to the 8-bucket vocabulary {GK, CB, FB, DM, CM, AM, W, ST}, with a profile-scraper fallback for the handful TM misses

### FM ↔ Transfermarkt join
The FM/Understat master is joined to Transfermarkt by a **4–5 pass fuzzy matcher on normalised name + club** (`src/value_wage/sources/transfermarkt_kaggle.py`), progressively loosening criteria: exact name + club → club-set membership (handles loans) → unique name + age → token-subset (middle names) → distinctive-name, each with **per-season uniqueness guards** so common names (Emerson, Gabriel, Thiago) can't collide.

Name normalisation ASCII-folds **and transliterates baked-in Latin-extended letters** (Ø→o, æ→ae, ł→l, dotless ı→i, …). Without that step `encode("ascii","ignore")` silently *deletes* those letters — e.g. "Ødegaard" → "degaard" — which previously caused Ødegaard and ~90 other players to miss the join and drop out of the dataset entirely. Unmatched rows (academy / fringe players with no TM Premier-League valuation) are kept as **predict-only** (NaN target).

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

> **Removed for leakage:** `season_mv_inflation_factor`, `season_position_mv_inflation_factor`, `season_fee_mv_premium`. These were PL-cohort season aggregates derived from the targets themselves (they encoded the held-out season's own price level). They are banned in `config/features.yaml` and the ban is enforced by `tests/test_no_leakage.py`.

Club and season identifiers are deliberately **excluded** — they'd absorb the budget/market signal the project exists to explain.

---

## Modelling

- **Targets**: `log1p(market_value_eur)` and `log1p(wage_pw)` — right-skewed targets benefit from log-space training.
- **Algorithm**: LightGBM, selected over XGBoost, CatBoost and HistGradientBoosting in an Optuna bake-off.
- **Split**: season-based — train **2023-24 + 2024-25**, test **2025-26** (held out). A **GroupKFold on `player_id`** for the inner CV stops the same player landing in both inner-train and inner-val.
- **Hyperparameters**: Optuna TPE sampler + MedianPruner; best parameters pinned.
- **Prediction intervals**: MAPIE **cross-conformal (CV+)** over train+val at α = 0.20 — distribution-free 80% intervals that actually achieve ~88–90% empirical coverage (the raw LightGBM quantile intervals under-covered at ~40%).
- **Leakage guards**: the target and FM money fields (`value_high`, `sell_value`, `release_clause`) and the PL-derived season factors are all excluded and asserted by both the training path (`_assert_no_leakage`) and the test suite.

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
├── src/value_wage/    # Library code — data, sources, features, train, calibrate, evaluate, export
├── tests/             # Pytest suite — split integrity, feature pipeline, join canaries, no-leakage guards
├── web/               # Static single-page UI (vanilla HTML + JS, charts from CDN)
│   └── data/          # players.json, mispricing.json, meta.json, bio.json consumed by the UI
├── Makefile           # One-command targets: data, features, tune, train, evaluate, calibrate, export
├── pyproject.toml     # Project metadata and dependencies
└── vercel.json        # Static deploy config
```

---

## Local development

```bash
# install
pip install -e ".[dev]"

# run the test suite
make test            # or: python -m pytest

# preview the UI (serves web/ with the committed JSON bundle)
python -m http.server 8000 --directory web
# then open http://localhost:8000/
```

Rebuilding the full pipeline from raw data (`make data features tune train evaluate calibrate export`, or the `vw` CLI) requires the Transfermarkt Kaggle dump (`https://www.kaggle.com/datasets/davidcariboo/player-scores`), the FMInside pages for the three FM editions, and the Understat per-season aggregates. See the `Makefile` for the end-to-end targets.

---

## Tech stack

Python 3.13 · pandas · scikit-learn · LightGBM · SHAP · Optuna · MAPIE · pydantic · Typer CLI · requests + BeautifulSoup for scraping · pytest · vanilla HTML/JS for the UI · Vercel for static hosting.

---

## Licence

See `LICENSE`.
