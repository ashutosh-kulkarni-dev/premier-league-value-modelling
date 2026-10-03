# Static Web Bundle

This folder is the entire UI dependency surface. No backend; `index.html` + `data/*.json` is
all that's shipped. Open `index.html` in a browser (or serve the folder with any static
server such as `python -m http.server`) and the page runs.

Replace `index.html` with whatever UI you build. The data contract below is stable.

## Rebuilding the data

```powershell
cd value-wage-model
.\.venv\Scripts\python.exe -m value_wage.cli export web
```

Writes:

- `data/players.json`     - per `(player, season, snapshot)` row with predictions, SHAP, bio, career, season stats, value trajectory, verdict (~15 MB)
- `data/meta.json`        - model metadata, global SHAP, metrics (~7 KB)
- `data/mispricing.json`  - top-20 under/over mispriced per target (~40 KB)
- `data/bio.json`         - bio only, keyed by `player_tm_id` (~500 KB) - lighter alternative to the full `players.json` for first-paint

## Data contract

### `players.json` - array of player-season-snapshot objects

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
      "full_name":                "Bukayo Saka",
      "first_name":               "Bukayo",
      "last_name":                "Saka",
      "date_of_birth":            "2001-09-05",
      "country_of_birth":         "England",
      "country_of_citizenship":   "England",
      "sub_position":             "Right Winger",
      "position":                 "Attack",
      "foot":                     "left",
      "height_cm":                178,
      "contract_expiration_date": "2030-06-30",
      "agent_name":               null,
      "image_url":                "https://img.a.transfermarkt.technology/portrait/header/433177-...",
      "international_caps":       54,
      "international_goals":      14,
      "current_club":             "Arsenal FC",
      "tm_current_market_value_eur":  110000000,
      "tm_highest_market_value_eur":  150000000
    },

    "career": [
      {"date": "2008-07-01", "season": "08/09", "from_club": "Watford Yth.", "to_club": "Arsenal Youth", "fee_eur": null},
      {"date": "2019-07-01", "season": "19/20", "from_club": "Arsenal U23",  "to_club": "Arsenal",       "fee_eur": null}
    ],

    "season_stats": [
      {"season": "2023", "matches": 46, "minutes": 3841, "goals": 20, "assists": 14, "yellow": 4, "red": 0},
      {"season": "2024", "matches": 32, "minutes": 2502, "goals": 10, "assists":  9, "yellow": 2, "red": 0}
    ],

    "value_trajectory": [
      {"date": "2023-06-20", "market_value_eur":  90000000, "club": "Arsenal FC"},
      {"date": "2023-12-19", "market_value_eur": 110000000, "club": "Arsenal FC"},
      {"date": "2024-05-29", "market_value_eur": 140000000, "club": "Arsenal FC"}
    ],

    "value": {
      "actual":       110000000,
      "predicted":    113987866,
      "lower":         50865694,
      "upper":        259503319,
      "residual_pct":    0.036,
      "shap_top": [
        {"feature": "potential",       "contribution_log": 0.515},
        {"feature": "current_ability", "contribution_log": 0.504},
        {"feature": "age",             "contribution_log": 0.196},
        {"feature": "xg_chain_p90",    "contribution_log": 0.135},
        {"feature": "minutes",         "contribution_log": 0.104},
        {"feature": "log1p_goals",     "contribution_log": 0.086}
      ],
      "verdict": {
        "label":           "excellent",
        "within_interval": true,
        "within_20_pct":   true,
        "pct_err":         0.036
      }
    },
    "wage": { "...": "same shape as value, in GBP/wk" }
  }
]
```

#### Notes

- `value` / `wage` are `null` on rows that fell outside the test season (unlabelled, no scored prediction).
- `actual` is `null` when the Transfermarkt join found no match for that row.
- `predicted`, `lower`, `upper` are on the back-transformed scale (EUR for value, GBP/wk for wage).
- `contribution_log` is on the `log1p(target)` scale - positive values push the prediction up.
- `shap_top` is the 6 features with highest `|contribution|` for that specific prediction.
- `bio.agent_name` is often `null` - TM frequently withholds agent data.
- `value_trajectory` spans the player's full TM history (sometimes 20+ snapshots across years).
- `season_stats` is per-season aggregates across ALL competitions the player appeared in (not PL-only), using TM's `games.csv` season label.

#### Verdict rule

| Rule | Verdict |
|---|---|
| |residual| <= 20% | `excellent` |
| within 80% interval AND |residual| <= 50% | `good` |
| within 80% interval AND |residual| > 50% | `fair` |
| outside 80% interval | `poor` (likely driven by non-measurable "human" factors - the mispricing-board story) |

### `bio.json` - compact map `{player_tm_id: bio}`

Same `bio` shape as above, keyed by stringified `player_tm_id`. Useful if your UI wants to show a bio sidebar without parsing the full 15MB players.json upfront. Lazy-load the full row from players.json on click.

### `meta.json`

```json
{
  "targets": {
    "value": {
      "model": "lgbm",
      "metrics": {
        "nominal_coverage":        0.8,
        "achieved_coverage":       0.896,
        "mean_interval_width_log": 1.65,
        "point_mae_log":           0.398,
        "point_r2_log":            0.782,
        "point_spearman":          0.876,
        "n_test":                  978,
        "method":                  "cross"
      },
      "shap_global_top15": [{"feature": "age", "mean_abs_shap": 0.244}],
      "features":          ["age", "age_sq", "minutes", "..."]
    },
    "wage": { "...": "same shape" }
  }
}
```

### `mispricing.json`

```json
{
  "value": {
    "market_underpays": [
      {"player": "...", "club": "...", "season": "2025-26",
       "actual": 500000, "pred": 3153000, "lower": 1529988, "upper": 6499048,
       "residual_pct": 5.3, "z": -3.26}
    ],
    "market_overpays": [ { "..." : "same shape" } ]
  },
  "wage": { "..." : "same shape" }
}
```

Signs:
- `market_underpays`: actual < predicted. The market pays LESS than the model expects given the player's measurable features. "Bargains" (if you trust the model) or "players the market has written down for an invisible reason" (if you trust the market).
- `market_overpays`: actual > predicted. The market pays MORE than the model expects. Often youth prospects with speculative premiums or players with a "manager wanted them" premium.

## Framing

The model predicts the baseline value a player's measurable features (age, performance, scouted attributes, contract) can support. **Residuals are the deliberate signal**: they quantify how much the actual market valuation deviates from the measurable baseline. Positive residuals are the "human premium"; negative residuals are the "human discount". The mispricing board is a tool for identifying players worth a conversation, not a declaration of market error.

## Minimal UI contract

Your `index.html` only needs to:

1. `fetch('./data/meta.json')`     - once, for model metadata + global SHAP.
2. `fetch('./data/players.json')`  - once, for the player list + enrichment.
3. `fetch('./data/mispricing.json')` - once, for the ranked boards.
4. User picks a target (`value` / `wage`), a player, a season, a snapshot.
5. Render:
   - header: player photo (`bio.image_url`), name, nationality, DOB, position, foot, height
   - predicted vs actual with 80% interval + verdict chip
   - top SHAP drivers for that specific prediction
   - career timeline from `career[]`
   - stats-per-season bar/line chart from `season_stats[]`
   - value-over-time line chart from `value_trajectory[]`
   - mispricing boards from `mispricing.json`
