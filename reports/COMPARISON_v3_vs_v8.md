# v3 vs v8 Comparison

## Fold-2 Metrics (train 2023-24+2024-25, test 2025-26)

### Value

| metric | prod_v3 | v8 | delta |
|---|---|---|---|
| mae_log | 0.3859 | 0.3824 | -0.0036 |
| r2_log | 0.7684 | 0.7742 | +0.0058 |
| mae_native | 6526141.7774 | 6559048.5168 | +32906.7394 |
| mean_ape_pct | 48.2882 | 48.9988 | +0.7106 |
| median_ape_pct | 24.6051 | 24.9490 | +0.3439 |

### Wage

| metric | prod_v3 | v8 | delta |
|---|---|---|---|
| mae_log | 0.4445 | 0.4767 | +0.0322 |
| r2_log | 0.7348 | 0.7150 | -0.0198 |
| mae_native | 28640.5405 | 30419.8395 | +1779.2990 |
| mean_ape_pct | 46.9447 | 47.1444 | +0.1997 |
| median_ape_pct | 30.3873 | 32.7666 | +2.3792 |

## Top-20 SHAP features (value model)

### Before (v3)

- current_ability: 0.3719
- age: 0.2654
- potential: 0.1108
- age_sq: 0.1013
- xg_chain_p90: 0.1001
- minutes: 0.0913
- xg_buildup_p90: 0.0844
- fm_physical_mean: 0.0511
- months_to_contract_end: 0.0485
- fm_technical_mean: 0.0483
- matches: 0.0442
- log1p_minutes: 0.0435
- goals: 0.0410
- key_passes_p90: 0.0407
- primary_position: 0.0379
- np_xg_p90: 0.0368
- potential_gap: 0.0242
- fm_mental_mean: 0.0220
- technique: 0.0161
- log1p_goals: 0.0158

### After (v8)

- current_ability: 0.3507
- age: 0.2617
- potential: 0.1212
- xg_chain_p90: 0.1008
- age_sq: 0.0775
- minutes: 0.0756
- matches: 0.0726
- fm_physical_mean: 0.0643
- months_to_contract_end: 0.0477
- xg_buildup_p90: 0.0465
- log1p_minutes: 0.0460
- goals: 0.0411
- primary_position: 0.0335
- is_decline: 0.0327
- np_xg_p90: 0.0321
- potential_gap: 0.0293
- fm_technical_mean: 0.0284
- key_passes_p90: 0.0257
- season_mv_inflation_factor: 0.0256
- is_prime: 0.0247

## Named spot-check (2025-26)

| player | actual | v3 pred | v8 pred |
|---|---|---|---|
| Erling Haaland | EUR 200.0M | EUR 182.2M | EUR 183.3M |
| Cole Palmer | EUR 120.0M | EUR 88.2M | EUR 90.5M |
| Antoine Semenyo | EUR 55.0M | EUR 71.2M | EUR 67.1M |
| Martin Ødegaard | N/A | N/A | N/A |
| James Trafford | EUR 30.0M | EUR 12.0M | EUR 11.8M |

## Verdict

- **Value**: flat (MAE_log delta -0.0036, R2 delta +0.0058)
- **Wage**: regressed (MAE_log delta +0.0322)
- New v8 features in top-20 SHAP: ['is_decline', 'season_mv_inflation_factor', 'is_prime']
- Dead new features (<0.005 SHAP): ['finishing_x_is_ST', 'finishing_x_is_AM', 'composure_x_is_ST', 'heading_x_is_CB', 'heading_x_is_ST', 'marking_x_is_CB', 'marking_x_is_FB', 'tackling_x_is_DM', 'tackling_x_is_CB', 'tackling_x_is_CM', 'pace_x_is_W', 'pace_x_is_FB', 'vision_x_is_AM', 'vision_x_is_CM', 'dribbling_x_is_W', 'handling_x_is_GK', 'reflexes_x_is_GK', 'is_youth']