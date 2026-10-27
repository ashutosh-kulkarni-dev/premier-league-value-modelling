# Model comparison — value

Held-out test season ('2025-26',) (n=1015). Trained on ('2023-24', '2024-25'). Same feature allowlist and `log1p` target for every model.

| model | r2_log | mae_log | mae_native | median_ape_pct | mean_ape_pct | spearman |
|---|---|---|---|---|---|---|
| lgbm (production) | 0.7836 | 0.3916 | 7278216.0 | 27.8 | 48.96 | 0.8738 |
| xgb | 0.7825 | 0.389 | 7165546.0 | 27.68 | 48.92 | 0.8746 |
| catboost | 0.7608 | 0.4061 | 7304017.0 | 28.39 | 55.06 | 0.8664 |
| elasticnet | 0.7334 | 0.4383 | 7979934.0 | 29.85 | 57.84 | 0.8424 |
| rf | 0.6985 | 0.4781 | 8788191.0 | 33.76 | 62.37 | 0.8495 |
| median_by_position | 0.047 | 0.8616 | 15550585.0 | 55.56 | 243.46 | 0.1966 |
