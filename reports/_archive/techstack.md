# Tech Stack — Value & Wage Model

Every choice below is picked for one reason: **it is the boring, defensible default for a tabular regression project at this scale, with a reproducibility and interpretability bar.** Novelty is a cost here, not a feature.

---

## Language & environment

| Tool | Choice | Why |
|---|---|---|
| Language | **Python 3.11** | Matches `scout-pipeline`'s existing `.venv`; 3.11 gives faster CPython and better error messages than 3.10, without the stdlib churn of 3.12/3.13. |
| Package / env manager | **uv** (with `pyproject.toml`) | 10–100× faster than pip for installs and lockfile resolution; produces a `uv.lock` that pins every transitive dep — exactly what "reproducible" means. Falls back to `pip` for anyone who refuses to install it. |
| Project layout | **src-layout** package `value_wage` | Prevents the classic "imports work in a notebook but break on install" trap. Signals that this is a project, not a script dump. |
| Config | **`pydantic-settings`** + a single `config.py` | One typed source of truth for paths, seeds, feature lists, hyperparameter grids. No scattered constants. |

---

## Data layer

| Tool | Choice | Why |
|---|---|---|
| Tabular engine | **pandas 2.x with the PyArrow backend** | Already used by `scout-pipeline`; dataset (~1.7k rows × ~90 cols) is far too small to justify Polars' learning curve for this project. PyArrow backend gives proper nullable dtypes, which matters because ~25% of FM rows have missing attributes. |
| On-disk format | **Parquet** | Preserves dtypes, is already the format of every upstream file, compresses well, is columnar (fast feature-subset reads). CSV is a worse default in every dimension except Excel-opens-it. |
| Validation | **pandera** schemas on load and after feature engineering | Catches silent regressions — e.g. a feature pipeline that starts producing nulls or shifts a dtype. Cheaper than a bug surfacing in week 4. |
| Great Expectations | **rejected** | Overkill for a single-project pipeline; pandera gives 90% of the value for 10% of the ceremony. |

---

## Modelling

| Tool | Choice | Why |
|---|---|---|
| Baselines | **scikit-learn** `DummyRegressor`, `Ridge`, `LinearRegression` | Universally understood, zero config, fits in two lines. The baseline exists to be beaten honestly, not to win. |
| Booster #1 | **LightGBM** | Leaf-wise growth; native categorical + native NaN handling (critical for FM features, ~25% missing); fastest tuning at this size. Expected strong performer on the attribute block. |
| Booster #2 | **XGBoost** | Reference gradient-boosting implementation; level-wise growth and strong L1/L2 regularisation tend to win on small, noisy datasets like ours (~1k labelled rows). Native NaN handling. |
| Booster #3 | **CatBoost** | Ordered boosting mitigates target leakage on small samples; best-in-class native categorical handling (`primary_position`, `foot`); the most forgiving default — lowest tuning effort to a decent score. |
| Booster #4 | **HistGradientBoostingRegressor (sklearn)** | A boring in-process baseline — fast, native NaN, no extra install, zero API drift. Keeps the three third-party libraries honest: if nothing beats HGBR materially, the fancy ones aren't earning their place. |
| Selection | **Pre-declared rule** (see [plan.md](plan.md) §5.2) | All four tuned under the identical Optuna recipe; the winner on held-out test log-MAE is promoted to the main model, the others are reported in the comparison table. Prevents post-hoc cherry-picking. |
| **Not** AdaBoost / plain `GradientBoostingRegressor` | Strictly dominated by the four above on tabular regression; included only as a one-cell appendix for completeness. |
| **Not** a neural net (TabNet, FT-Transformer, MLP) | 1k–1.6k labelled rows is far below the regime where DL beats GBMs on tabular; interpretability story is harder; adds a GPU dependency for no measurable gain. |
| Uncertainty | **Winning library's quantile objective** at α=0.1, 0.9 (LightGBM `quantile`, XGBoost `reg:quantileerror`, CatBoost `Quantile`, HGBR `loss="quantile"`) | Keeps intervals in the same framework as point predictions. Fallback: **`mapie`** conformal wrapper — distribution-free and library-agnostic — if the winner ever lacks quantile support. |
| Preprocessing | **scikit-learn `Pipeline` + `ColumnTransformer`** | Encodes the "fit on train only" discipline into code; serialises as a single artefact with the model. |
| Hyperparameter search | **Optuna** with TPE sampler + MedianPruner | Sample-efficient (TPE beats random/grid at the same budget); pruner kills bad trials early; results persist to SQLite so a run can be resumed. |
| Cross-validation | **scikit-learn `GroupKFold`** on `player_id` | Prevents a player appearing in both train and val, which would be the single most dangerous leakage in this project. |
| Experiment tracking | **MLflow (local file store)** | Logs params, metrics, artefacts, model binaries per run; no server to set up; the UI (`mlflow ui`) is good enough to swap between runs in the report. W&B is nicer but adds an account dependency the project doesn't need. |

---

## Interpretability

| Tool | Choice | Why |
|---|---|---|
| Feature attribution | **SHAP** with `TreeExplainer` | Exact for tree models (no sampling noise), fast on this scale. The deliverable literally depends on this. |
| Partial dependence | **scikit-learn `PartialDependenceDisplay`** | For 2-3 headline features (age, minutes, finishing) — PDPs are easier for non-technical readers than SHAP. |
| Permutation importance | **sklearn `permutation_importance`** on the held-out set | Cross-checks SHAP; disagreement between the two is itself a finding worth reporting. |

---

## Visualisation & reporting

| Tool | Choice | Why |
|---|---|---|
| Static plots | **matplotlib** + **seaborn** | What SHAP, sklearn, and every reviewer expect; publication-ready; no JS runtime. |
| Interactive plots (app only) | **Plotly Express** | Only inside Streamlit, where hover/zoom earns its weight. Not in the notebooks or report — static images render in GitHub and PDFs. |
| Report | **Quarto** → PDF + HTML | Lets the final report pull live from the notebooks, so numbers in the writeup can never go stale against the code. |
| Demo app | **Streamlit** | Fastest path from a trained model to a shareable UI; one Python file, no frontend code. Gradio would also work; Streamlit's `st.cache_data` + layout primitives fit this app better. |

---

## Code quality & tooling

| Tool | Choice | Why |
|---|---|---|
| Linter / formatter | **ruff** (lint + format) | Replaces black + isort + flake8 + pyupgrade in one tool, 100× faster. The new default. |
| Type checker | **mypy --strict** on `src/`, lenient on notebooks | Types in the library code catch feature-pipeline bugs at edit time; notebooks are exploratory and shouldn't be forced into strict typing. |
| Tests | **pytest** + **hypothesis** for property tests on feature functions | Feature engineering is pure data-in-data-out — exactly where property-based tests catch edge cases human-written cases miss (zero minutes, negative age typos, all-NaN rows). |
| Pre-commit | **pre-commit** running ruff + mypy + nbstripout | Keeps the repo clean automatically; `nbstripout` prevents checking in notebook outputs, which otherwise balloon the diff and leak PII-ish stuff. |
| Task runner | **Makefile** | `make data`, `make train`, `make report`, `make app`. Universally understood on macOS/Linux; one-line commands for the user to copy. |
| CI | **GitHub Actions** running lint + mypy + tests on push | Even a solo project benefits — proves on a fresh VM that `uv sync && make test` works, which is the real reproducibility check. |

---

## Determinism & reproducibility

| Concern | How it's handled |
|---|---|
| Random seeds | Single `SEED = 42` in `config.py`, passed explicitly to every stochastic call (numpy, lightgbm, optuna sampler, sklearn splitters). No implicit global seeding. |
| Dependency pinning | `uv.lock` committed; CI installs from the lockfile, not `pyproject.toml`. |
| Data pinning | Input parquet files hashed (SHA-256) at load time; hash recorded in the MLflow run. If the input changes, the run tag changes — no silent drift. |
| Compute | CPU only. Keeps the project runnable on any laptop and sidesteps the "did CUDA change the result?" rabbit hole. LightGBM is fast enough on CPU for this size. |

---

## Deliberately **not** in the stack

| Rejected | Why |
|---|---|
| Docker | A `uv.lock` + README install step covers reproducibility for a solo research project. Docker is overhead that doesn't earn its weight here. |
| DVC / LakeFS | 7 small parquet files, versioned by git-LFS or just committed. DVC's workflow cost outweighs the benefit at this scale. |
| Airflow / Prefect / Dagster | There is no recurring pipeline. One `make train` run per experiment. An orchestrator here would be theatre. |
| FastAPI backend | Streamlit *is* the UI and the backend. Adding a REST layer with one consumer (the Streamlit app) is pure friction. |
| Kubernetes / cloud deploy | Scope is a local demo + a PDF report. If the project later wants a public demo, Streamlit Community Cloud is one click; nothing in the stack blocks that. |
| PyTorch / TensorFlow | See §Modelling — no deep model is justified at this dataset size and interpretability bar. |
| AutoML (H2O, auto-sklearn, FLAML) | The project's value is in the thinking (features, validation, interpretation), not in finding the best model by brute force. An AutoML winner you can't explain is worth less than a tuned LightGBM you can. |

---

## One-screen summary

> **Python 3.11 + uv** for the environment.
> **pandas (arrow) + pandera + parquet** for data.
> **sklearn baselines + a disciplined bake-off of LightGBM / XGBoost / CatBoost / HistGradientBoosting (all tuned with Optuna under one shared recipe, grouped by player, split by season), with the winner promoted and its quantile variant used for intervals** for modelling.
> **SHAP + PDP + permutation importance** for interpretability.
> **MLflow** for tracking, **Quarto** for the report, **Streamlit** for the demo.
> **ruff + mypy + pytest + pre-commit + GitHub Actions** for the engineering floor.

Every element is the straightforward professional default for a small, interpretable, reproducible tabular ML project — chosen so the thinking in the report, not the tooling, is where this project earns its grade.
