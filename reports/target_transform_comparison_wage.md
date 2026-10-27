# Target transform comparison — wage (lgbm)

Same features, season splits, preprocessor and model; the only difference is whether the
model trains on `log1p(wage_pw)` or on the raw value directly. All metrics
are scored on the raw-unit scale (test split). `€k/wk` columns are in those units.

| target_transform | MAE_€k/wk | RMSE_€k/wk | R2_raw | Spearman | MedAPE | MAE_log | R2_log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| log1p | 12.1732 | 20.5772 | 0.9236 | 0.9534 | 0.1037 | 0.2233 | 0.8694 |
| direct | 12.5908 | 20.6795 | 0.9228 | 0.9440 | 0.0989 | 0.2811 | 0.6224 |

Lower is better: MAE, RMSE, MedAPE, MAE_log. Higher is better: R2, Spearman, R2_log.

| Metric | Winner |
|---|---|
| MAE (€k/wk) | log1p |
| RMSE (€k/wk) | log1p |
| R2 (raw) | log1p |
| MedAPE | direct |
| R2 (log1p) | log1p |
