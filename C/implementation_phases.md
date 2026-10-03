# Path C — Implementation Phases

For when Path A has shipped and we come back to this.

Prerequisite: Path A's codebase (`value-wage-model/`) is complete and the
boosters/tuning/eval/SHAP infrastructure is working. We fork it, not rebuild.

---

## Phase 0 — Scaffold the fork

Clone `value-wage-model/` → `fee-model/`. Rename package to `fee_model`
(so both projects can be installed in the same env side-by-side). Update
`pyproject.toml` name, console-script entry, and `README.md` thesis
paragraph. **Done when:** `pip install -e .` succeeds, `pytest` passes on
the inherited tests.

## Phase 1 — Data ingest

Source: **Kaggle `davidcariboo/player-scores`** (also mirrored to GitHub at
`dcaribou/transfermarkt-datasets` — download works without Kaggle
credentials via its weekly release zip). Pull:

- `transfers.csv`              — ~500k historical transfers
- `players.csv`                — bio + current market value
- `player_valuations.csv`      — historical market-value timeline
- `appearances.csv`            — minutes / goals / cards per match
- `game_events.csv`            — goals, assists, cards with timestamps
- `clubs.csv`                  — club metadata (league, squad size)
- `club_games.csv`             — per-match club results (for standings)
- `competitions.csv`           — league tier reference

Write `src/fee_model/sources/transfermarkt.py`: single `download()`
function that fetches the latest release, caches to `data/raw/tm/`, and
hashes each file. **Done when:** `fee data pull` writes 8 CSVs with
recorded SHA-256s.

## Phase 2 — Transfer-row filter & label build

Build the master transfer table:

```python
transfers
  .filter(fee_eur > 0 & fee_eur IS NOT NULL)              # drops ~30% undisclosed
  .filter(type IN {"permanent", "loan_with_obligation"})
  .filter(date >= 2015-07-01)                             # 10 year window
  .filter(buyer_league IN TOP7 OR seller_league IN TOP7)
```

Target: `log1p(fee_eur)`. Expected row count: **2k–4k** after filters.

Write `src/fee_model/data.py` paralleling Path A's; pandera schema
enforces fee > 0, buyer_id and seller_id non-null, date parseable.
**Done when:** `fee data build` produces `data/processed/transfers.parquet`
with row-count report and non-null checks.

## Phase 3 — Player feature snapshot at transfer date

Each transfer row needs features **as of the transfer date**, not current.

- **Pre-transfer season performance** (last full season ending ≤ transfer
  date): minutes, goals, assists, xG per 90 if available (from Understat
  via a separate join; fall back to goals-based rate otherwise).
- **Age at transfer** = (`transfer_date` − `date_of_birth`) / 365.25.
- **Months to contract end at transfer**.
- **Position** from `players.csv.sub_position`.
- **FM scout attributes** — join FM26 snapshot *only for transfers in
  2024-25 or later*; set to NaN for older ones (boosters handle NaN).
  Honest limitation: older transfers lose this feature group.

Write `src/fee_model/features/player.py` as pure functions,
unit-tested on fixture rows.

## Phase 4 — Buyer / seller club context features

This is the key feature group that lets the model isolate the "Chelsea
tax" from player attributes.

- `buyer_league`, `seller_league` — categorical
- `buyer_wage_tier` — quintile of buyer's squad market value at transfer
  season (proxy for wage bill)
- `buyer_recent_spend_3y` — total fees paid by buyer in prior 3 years
- `buyer_recent_sales_3y` — total fees received
- `buyer_table_position_last_season` — ranking in domestic league
- `is_promotion_window` — buyer just promoted / seller just relegated
- `transfer_window` — summer vs winter dummy
- `year_of_transfer` — raw year, lets the tree learn market inflation

Write `src/fee_model/features/clubs.py`. **Done when:** every row has
a complete buyer + seller context feature set.

## Phase 5 — Time-based splits

Replace Path A's season splits:
- **Train:** transfers up to 2023-01-01
- **Val:** transfers 2023-01-01 to 2024-06-30
- **Test:** transfers from 2024-07-01 onward

Rationale: a model needs to generalise to **future** windows, so the
holdout is the most recent. GroupKFold inside train on `player_id` is
still appropriate for the same leakage reason as Path A.

Replace `src/fee_model/splits.py`'s season logic with
transfer-date logic. All downstream code (tuning, evaluate) is agnostic.

## Phase 6 — Bake-off reuse

Rerun Path A's 4-booster bake-off (LGBM / XGB / CatBoost / HGBR), same
Optuna recipe, same loss (MAE on log target). No code changes expected
in `tuning.py`. **Done when:** `fee tune all --trials 50` produces a
new `bakeoff_cv.csv` + `winners.json`.

## Phase 7 — Residual-based analytics (the real deliverable)

New module `src/fee_model/residuals.py`:

- `club_residual_scorecard(predictions, by="buyer")` — mean, median, n
  of `(fee − predicted_fee)` per buyer club, filterable by window.
- `club_residual_scorecard(predictions, by="seller")` — mirror.
- `transfer_residual_board(top_n=20)` — biggest overpays, biggest
  bargains, with model's SHAP breakdown of why the baseline was what
  it was.

Replaces Path A's `mispricing.py` — same shape, different interpretation.

## Phase 8 — Post-transfer realised-value check (optional stretch)

For transfers at least 1-2 seasons old, compute each player's realised
post-transfer value using next-season performance (xG+xA+minutes
converted to a notional € using Path A's model). Add a column to the
mispricing board:

```
Player X | Fee paid €65M | Model predicted €55M | Realised value €25M
       → "overpaid relative to model, underperformed vs realised"
```

This is the sentence that makes the project's output land.

## Phase 9 — Streamlit extension

Reuse the Path A app, add a new tab:

- Dropdown: pick a *player not currently at that club*.
- Dropdown: pick a *buyer club*.
- Date picker: hypothetical transfer window.
- Output: predicted fee + interval + SHAP breakdown (base + buyer
  premium + contract factor + inflation adjustment).
- Below: 5 nearest comparable *past transfers* with their actual fees.

## Phase 10 — Report chapter

One chapter added to the Path A report:
- Section 1: Why fees, not values (overpayment framing).
- Section 2: Data pipeline + filter rationale.
- Section 3: Headline metrics (MAE log, Spearman, interval coverage).
- Section 4: The club scorecards (two leaderboards).
- Section 5: Case studies (3 famous overpays, 3 famous bargains with
  SHAP breakdowns).
- Section 6: Limitations (selection bias, missing fees, inflation
  assumptions).

---

## Rough effort estimate

| Phase | Day-equivalents of work |
|---|---|
| 0 Scaffold fork | 0.25 |
| 1 Data ingest | 0.5 |
| 2 Filter & label build | 0.5 |
| 3 Player features | 1 |
| 4 Club context features | 1 |
| 5 Time splits | 0.25 |
| 6 Bake-off | 0.25 (compute, mostly) |
| 7 Residual analytics | 0.5 |
| 8 Post-transfer check | 0.75 |
| 9 Streamlit tab | 0.5 |
| 10 Report chapter | 0.75 |
| **Total** | **~5.25 days** |

Compared with Path A's ~1 extra day of work, C is a materially bigger
build. But almost all of it is new *data-engineering + domain features*,
not new ML — the ML pipeline is already written.
