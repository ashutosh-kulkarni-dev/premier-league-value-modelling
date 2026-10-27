# Target transform comparison — value (lgbm)

Same features, season splits, preprocessor and model; the only difference is whether the
model trains on `log1p(market_value_eur)` or on the raw value directly. All metrics
are scored on the raw-unit scale (test split). `€m` columns are in those units.

| target_transform | MAE_€m | RMSE_€m | R2_raw | Spearman | MedAPE | MAE_log | R2_log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log1p | 6.9708 | 11.0785 | 0.7650 | 0.8778 | 0.2707 | 0.3809 | 0.7902 |
| direct | 7.1120 | 11.0192 | 0.7675 | 0.8745 | 0.2869 | 0.5225 | -0.5409 |

Lower is better: MAE, RMSE, MedAPE, MAE_log. Higher is better: R2, Spearman, R2_log.

| Metric | Winner |
|---|---|
| MAE (€m) | log1p |
| RMSE (€m) | direct |
| R2 (raw) | direct |
| MedAPE | log1p |
| R2 (log1p) | log1p |
