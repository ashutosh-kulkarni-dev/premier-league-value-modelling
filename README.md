# PL Insight — Premier League Player Value Model

A regression pipeline and static web UI that predict Premier League players' market values (Transfermarkt €) from on-pitch performance and scouted football attributes, then flag players where the market valuation diverges most from what measurable features predict.

> **Live demo:** https://premier-league-value-modelling-web.vercel.app/

The project treats the model's **residuals as insight**, not as errors to minimise — residuals quantify the "human premium" that mechanical features can't see: manager demand, auction dynamics, expiring contracts, injury rumours, hype. The mispricing board is a tool for investigation, not a declaration of market error.

---

## Headline metrics (honest fold-2 holdout)

Train on 2023-24 + 2024-25, test on 2025-26. Metrics computed on the held-out season — the model has never seen these players at that time point.

| Metric | Value |
|---|---:|
| **R² (log target)** | **0.865** |
| **MAE (log target)** | **0.252** |
| **MAE (native €)** | €4.21M |
| **Median absolute % error** | **14.4%** |
| **Mean absolute % error** | 43.6% |
| **80% interval coverage** | 72.8% |
| **Winning algorithm** | LightGBM |

Mean APE is tail-heavy because a small subset of low-value players produce extreme relative errors; the median is the honest central-tendency measure for most of the dataset.

---

## What the UI shows

Each player page carries:
- **Point prediction** (market value in €)
- **Confidence tier** — SUPERSTAR / HIGH / MEDIUM / SPECULATIVE, driven by predicted value, minutes played, age, and position availability
- **SHAP top drivers** — the 3 features most pulling the prediction up and the 3 pulling it down for that specific player
- **Comparable players** — 3-5 nearest neighbours in feature space within the same primary position, with their actual market values for calibration
- **Market vs model residual** — the headline mispricing signal

The dashboard also surfaces:
- Current fold-2 R², MAE, and interval coverage
- Top 20 most under-paid and 20 most over-paid players, filtered to meaningful cohorts (actual ≥ €2M, minutes ≥ 500, excluding SPECULATIVE)
- Global SHAP feature importance

---

## Data pipeline

Three data sources, joined by stable identifiers (`fmi_player_id` for FM, `player_tm_id` for Transfermarkt) rather than fuzzy name matching:

### Football Manager attributes (per season)
- **2023-24 season rows** → FMInside FM24 24.3.0 scrape (1,670 PL players, scraped via `scrape_fminside.py` with 3-second rate limit and raw-HTML caching)
- **2024-25 season rows** → FM24 in-game HTML export after loading May 2025 editor-data updates (transfers, CA/PA changes, aging) — gives post-summer-window + January-window attribute state
- **2025-26 season rows** → FMInside FM26 scrape (1,673 PL players)

All 46 individual attributes (14 technical, 14 mental, 8 physical, 10 GK) are joined per-season per-snapshot. Goalkeeper attributes are gated to GK rows only. For 2024-25, `current_ability` and `potential` are imputed by midpoint of the 2023-24 and 2025-26 anchors when both exist (pass-through when only one anchor exists).

### Transfermarkt (per snapshot date)
- **`players.csv`** → player bio, primary position (`sub_position`), nationality (read as UTF-8 to preserve diacritics)
- **`player_valuations.csv`** → market value and current club at each snapshot date (nearest datapoint within ±120 days)
- Primary position sourced from TM `sub_position` mapped to the 8-bucket vocabulary {GK, CB, FB, DM, CM, AM, W, ST}
- A dedicated bridge (`data/processed/tm_fmi_bridge.csv`) resolves TM ↔ FMI id pairs, with fallback name normalisation (unidecode) and uniqueness guards so that short/common names (Emerson, Gabriel, Thiago, etc.) can't collide across players

### Understat (per season)
- Minutes, matches, goals, assists, xG, non-penalty xG, xA, xG chain, xG buildup, key passes
- Converted to per-90 rates during feature engineering

### Position recovery scraper
A separate TM profile scraper (`tm_scrape_positions.py`) resolves primary position for the ~40 players whose TM `sub_position` wasn't initially joined — reads the "Main position" line directly from each profile page and maps to the 8-bucket vocabulary. Raw HTML is cached for free re-runs.

---

## Feature set (v8slim)

| Group | Count | Examples |
|---|---:|---|
| Core numeric | 15 | age, minutes, matches, goals, assists, np_xg_p90, xg_chain_p90, xg_buildup_p90, key_passes_p90, current_ability, potential, months_to_contract_end, n_fm_positions, height_cm, weak_foot |
| FM technical attributes | 14 | finishing, passing, dribbling, technique, marking, tackling, heading, long_shots, crossing, corners, free_kick_taking, penalty_taking, long_throws, first_touch |
| FM mental attributes | 14 | composure, decisions, vision, work_rate, positioning, teamwork, anticipation, leadership, concentration, determination, bravery, flair, aggression, off_the_ball |
| FM physical attributes | 8 | pace, acceleration, strength, stamina, agility, balance, jumping_reach, natural_fitness |
| FM goalkeeping attributes (GK-gated) | 10 | handling, reflexes, aerial_reach, command_of_area, communication, kicking, one_on_ones, rushing_out, throwing, eccentricity |
| Age buckets | 2 | is_prime (22-29), is_decline (≥30) |
| Market-drift features (from PL transfer-window data) | 3 | season_mv_inflation_factor, season_position_mv_inflation_factor, season_fee_mv_premium |
| Categorical | 3 | primary_position, club, season |

Explicit interaction features (`finishing × is_striker` style) were tested and dropped after SHAP showed the tree ensemble was already capturing them via position-conditional splits.

---

## Modelling

- **Target transform**: `log1p(market_value_eur)` — right-skewed targets benefit from log-space training
- **Algorithm**: LightGBM, selected over XGBoost and HistGradientBoosting in an Optuna bake-off
- **Validation**: rolling-window time-series cross-validation — Fold 1 trains on 2023-24 and tests on 2024-25; Fold 2 trains on 2023-24 + 2024-25 and tests on 2025-26
- **Hyperparameters**: Optuna TPE sampler + MedianPruner, best parameters pinned
- **Serving model**: trained on all three seasons combined — standard industry practice for time-evolving domains after honest CV, with the acknowledged limitation that performance on a future unseen season would require recalibration if seasonal market dynamics drift materially
- **Prediction intervals**: conformal recalibration on fold-2 residuals (α=0.20)
- **Confidence tiers**: UI-level categorisation based on predicted value, player minutes, age, and position availability

---

## Known limitations

1. **Transfermarkt noise ceiling**: TM market values are crowdsourced. Published studies place TM's own variance at ~20-30%. The model's residuals partly reflect this inherent noise rather than modelling error.
2. **Market drift**: observed +15-25% median market-value inflation from 2024-25 to 2025-26 (confirmed via real PL transfer-window data). Captured partially via drift features; future seasons will need re-recalibration.
3. **Feature coverage**: a small residual of academy / fringe players are not carried by Transfermarkt's player directory and are excluded from the serving dataset.

---

## Repository layout

```
├── config/              # YAML configs (feature list, splits, model params)
├── data/
│   ├── raw/             # Source datasets — Transfermarkt dump is gitignored
│   └── processed/       # Trained models, feature matrices, bridge tables
├── reports/             # SHAP CSVs, comparison docs, mispricing exports
├── src/value_wage/      # Library code — features, sources, train, evaluate
├── tests/               # Pytest suite — split integrity, feature pipeline, no-leakage guards
├── web/                 # Static single-page UI (vanilla HTML + JS, Chart.js from CDN)
│   └── data/            # players.json, mispricing.json, meta.json consumed by the UI
├── Makefile             # One-command targets: ingest, build-features, train, export-web
├── pyproject.toml       # Project metadata and dependencies
└── vercel.json          # Static deploy config
```

---

## Local development

```bash
# install
pip install -e ".[dev]"

# preview the UI
python -m http.server 8000 --directory web
# then open http://localhost:8000/
```

Rebuilding the full pipeline from raw data requires the Transfermarkt Kaggle dump (`https://www.kaggle.com/datasets/davidcariboo/player-scores`), FMInside pages for the three FM editions, and Understat per-season aggregates. See `Makefile` for the end-to-end targets.

---

## Tech stack

Python 3.13 · pandas · scikit-learn · LightGBM · SHAP · Optuna · MAPIE · requests + BeautifulSoup for scraping · pytest for tests · vanilla HTML/JS + Chart.js for the UI · Vercel for static hosting.

---

## Licence

See `LICENSE`.
