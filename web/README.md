# Static Web Bundle

This folder is the entire app dependency surface. No backend; `index.html` + `data/*.json` is
all that's shipped. Serve the folder with any static server (e.g. `python -m http.server`, or
`make serve` from the project root) and the page runs. Vercel serves this `web/` directory
directly (`vercel.json` → `outputDirectory: "web"`).

The data is split into a **slim index** loaded on first paint and **lazy per-player detail**
files fetched on demand, so the initial payload stays small.

## Rebuilding the data

```powershell
cd value-wage-model
.\.venv\Scripts\python.exe -m value_wage.cli export web
```

Writes:

- `data/players.json`     - slim per-`(player, season, snapshot)` row: predictions, verdict, scalars, small bio subset (~1.5 MB)
- `data/player/<id>.json` - lazy per-player detail: full bio, career, season stats, value trajectory, per-season SHAP (one small file per player)
- `data/meta.json`        - model metadata, metrics, global SHAP, confidence table (~10 KB)
- `data/mispricing.json`  - top-20 under/over mispriced per target (~40 KB)

Only players in the **2025-26 Premier League** roster are exported; club names are normalised to
one canonical name per club.

## Data contract

### `players.json` - array of slim player-season-snapshot objects

```json
[
  {
    "player_id":    7322,
    "player_tm_id": 433177,
    "player":       "Bukayo Saka",
    "season":       "2025-26",
    "snapshot":     "end",
    "club":         "Arsenal",
    "position":     "W",
    "age":          24.73,
    "minutes":      2239,

    "bio": {
      "image_url":                    "https://img.a.transfermarkt.technology/portrait/header/433177-...",
      "sub_position":                 "Right Winger",
      "country_of_citizenship":       "England",
      "contract_expiration_date":     "2030-06-30",
      "tm_current_market_value_eur":  110000000,
      "tm_highest_market_value_eur":  150000000
    },

    "value": {
      "actual":       110000000,
      "predicted":    113987866,
      "residual_pct":    0.036,
      "verdict": { "label": "excellent", "within_20_pct": true, "pct_err": 0.036 }
    },
    "wage": { "...": "same shape as value, in GBP/wk" }
  }
]
```

#### Notes

- `bio` here is a **small subset** for lists/thumbnails; the full bio lives in the per-player detail file.
- `value` / `wage` are `null` on rows with no scored prediction; `actual` is `null` when the Transfermarkt join found no match.
- `predicted` is on the back-transformed scale (EUR for value, GBP/wk for wage).

#### Verdict rule (from |% error|)

| Rule | Verdict |
|---|---|
| \|% error\| ≤ 20% | `excellent` |
| \|% error\| ≤ 35% | `good` |
| \|% error\| ≤ 50% | `fair` |
| \|% error\| > 50% | `poor` |

### `player/<id>.json` - lazy per-player detail

Keyed by `player_id`. Fetched on demand when a profile or comparison opens.

```json
{
  "bio": { "full_name": "Bukayo Saka", "date_of_birth": "2001-09-05", "foot": "left",
           "height_cm": 178, "agent_name": null, "international_caps": 54, "...": "full bio" },
  "career": [
    {"date": "2019-07-01", "season": "19/20", "from_club": "Arsenal U23", "to_club": "Arsenal", "fee_eur": null}
  ],
  "season_stats": [
    {"season": "2023", "matches": 46, "minutes": 3841, "goals": 20, "assists": 14, "yellow": 4, "red": 0}
  ],
  "value_trajectory": [
    {"date": "2023-06-20", "market_value_eur": 90000000, "club": "Arsenal FC"}
  ],
  "shap": {
    "2025-26|end": {
      "value": [{"feature": "potential", "contribution_log": 0.515}, {"feature": "current_ability", "contribution_log": 0.504}],
      "wage":  [{"feature": "current_ability", "contribution_log": 0.33}]
    }
  }
}
```

- `shap` is keyed by `"<season>|<snapshot>"`; each target has its top-6 features by `|contribution_log|` (log-scale; positive pushes the prediction up).
- `value_trajectory` spans the player's full TM history.
- `season_stats` is per-season aggregates across all competitions (TM `games.csv` season label).

### `meta.json`

```json
{
  "targets": {
    "value": {
      "model": "lgbm",
      "metrics": {
        "point_r2_log":   0.784,
        "point_mae_log":  0.392,
        "point_spearman": 0.874,
        "median_ape_pct": 27.8,
        "n_test":         1015
      },
      "confidence": {
        "tier_bounds":      [5000000, 15000000, 40000000],
        "tier_labels":      ["<5M", "5-15M", "15-40M", "40M+"],
        "area_of_position": {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "...": "..."},
        "area_thresholds":  {"GK": [30, 36], "DEF": [25, 40], "MID": [25, 40], "ATT": [30, 45]},
        "lookup":           {"DEF": {"<5M": 59, "5-15M": 24, "15-40M": 21, "40M+": 25}, "...": "..."},
        "tier_fallback":    {"<5M": 62, "5-15M": 27, "15-40M": 24, "40M+": 31}
      },
      "shap_global_top15": [{"feature": "age", "mean_abs_shap": 0.244}],
      "features":          ["age", "age_sq", "minutes", "..."]
    },
    "wage": { "...": "same shape (confidence tiers are in GBP/wk)" }
  }
}
```

**Confidence.** For a player, pick the `area` from `area_of_position[position]` and the tier by
bucketing their predicted value with `tier_bounds`. Look up the cohort's typical % error in
`lookup[area][tier]` (falling back to `tier_fallback[tier]`), then classify against that area's
`area_thresholds` `[high, low]`: below `high` → High, between → Medium, above `low` → Low.

### `mispricing.json`

```json
{
  "value": {
    "market_underpays": [
      {"player": "...", "club": "...", "season": "2025-26",
       "actual": 500000, "pred": 3153000, "residual_pct": 5.3, "z": -3.26}
    ],
    "market_overpays": [ { "..." : "same shape" } ]
  },
  "wage": { "..." : "same shape" }
}
```

- `market_underpays`: actual < predicted — the market pays LESS than the model expects (bargains, or a player written down for an invisible reason).
- `market_overpays`: actual > predicted — the market pays MORE (youth premiums, "manager wanted them").
- `z` standardises the log residual by a single global sigma, so it ranks consistently across the value range.

## Framing

The model predicts the baseline value a player's measurable features (age, performance, scouted
attributes, contract) can support. **Residuals are the deliberate signal**: they quantify how
much the actual market valuation deviates from that baseline. Positive residuals are the "human
premium"; negative residuals are the "human discount". The mispricing board is a tool for
identifying players worth a conversation, not a declaration of market error.

## Minimal UI contract

Your `index.html` only needs to:

1. `fetch('./data/meta.json')`       - once, for metadata, metrics, confidence table, global SHAP.
2. `fetch('./data/players.json')`    - once, for the slim player index + predictions.
3. `fetch('./data/mispricing.json')` - once, for the ranked boards.
4. `fetch('./data/player/<id>.json')` - on demand, when a profile or comparison opens.
