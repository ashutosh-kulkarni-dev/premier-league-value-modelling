"""Typer CLI. Every stage of the pipeline has one subcommand.

Entry point: `vw` (declared in pyproject.toml). All commands exit non-zero on failure —
no silent swallowing of errors.
"""

from __future__ import annotations

import functools
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from value_wage.config import (
    ALL_BASELINES,
    ALL_BOOSTERS,
    ALL_TARGETS,
    TARGET_COLUMN,
    Model,
    Target,
    get_settings,
)
from value_wage.data import copy_from_scout_pipeline, load_master, load_master_with_tm
from value_wage.evaluate import (
    bootstrap_metric_ci,
    compute_metrics,
    plot_pred_vs_actual,
    plot_residuals,
    summarise_predictions,
)
from value_wage.explain import explain_tree_model, save_global_summary_plot, save_waterfall_plot
from value_wage.export_web import export as export_web_bundle
from value_wage.features import build_feature_matrix
from value_wage.mispricing import build_board
from value_wage.models.baselines import median_by_position
from value_wage.splits import make_splits
from value_wage.train import (
    TrainedArtifact,
    _normalize_na,
    predict,
    train_baseline,
    train_booster,
)
from value_wage.tuning import tune

app = typer.Typer(
    add_completion=False,
    help="Premier League player value & wage model.",
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)
data_app = typer.Typer(help="Data loading + validation.")
feat_app = typer.Typer(help="Feature engineering.")
train_app = typer.Typer(help="Training.")
tune_app = typer.Typer(help="Optuna tuning.")
eval_app = typer.Typer(help="Evaluation.")
explain_app = typer.Typer(help="SHAP explanations.")
misp_app = typer.Typer(help="Mispricing board.")

export_app = typer.Typer(help="Export results for a static HTML UI.")
compare_app = typer.Typer(help="Baseline vs production model comparison.")

app.add_typer(data_app, name="data")
app.add_typer(feat_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(tune_app, name="tune")
app.add_typer(eval_app, name="evaluate")
app.add_typer(explain_app, name="explain")
app.add_typer(misp_app, name="mispricing")
app.add_typer(export_app, name="export")
app.add_typer(compare_app, name="compare")

console = Console()


def _guard(fn):  # type: ignore[no-untyped-def]
    """Decorator: print full traceback and exit 1 on any exception. No silent failures.

    Uses `functools.wraps` so Typer can introspect the original signature.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return fn(*args, **kwargs)
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[bold red]ERROR[/bold red]: {exc}")
            traceback.print_exc()
            raise typer.Exit(code=1) from exc
    return wrapper


# ---------------- data ----------------

@data_app.command("build")
@_guard
def data_build(skip_tm: bool = typer.Option(False, help="Skip the TM join (debug only).")) -> None:
    """Copy the FM master, validate it, join Transfermarkt market values, persist expanded parquet."""
    settings = get_settings()
    dst = copy_from_scout_pipeline(settings)
    result = load_master(settings)
    console.print(f"[green]OK[/green] copied -> {dst}")
    console.print(f"  rows: {result.row_count}")
    console.print(f"  sha256: {result.sha256[:12]}...")
    console.print(f"  season_counts: {result.season_counts}")

    if skip_tm:
        console.print("[yellow]--skip-tm set; not building the TM-expanded master.[/yellow]")
        return

    tm_result = load_master_with_tm(settings)
    settings.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    tm_result.frame.to_parquet(settings.paths.processed_master_tm, index=False)
    console.print(f"[green]OK[/green] TM-expanded master -> {settings.paths.processed_master_tm}")
    console.print("[cyan]Join report:[/cyan]")
    for line in tm_result.join_report.summary().splitlines():
        console.print(f"  {line}")


# ---------------- features ----------------

def _features_df() -> pd.DataFrame:
    """Prefer the TM-expanded master (snapshot-aware, TM target) when present; else FM-only."""
    settings = get_settings()
    if settings.paths.processed_master_tm.exists():
        source = pd.read_parquet(settings.paths.processed_master_tm)
    else:
        source = load_master(settings).frame
    return build_feature_matrix(source, settings=settings)


@feat_app.command("build")
@_guard
def features_build() -> None:
    """Build the feature matrix and write it to data/processed/features.parquet."""
    settings = get_settings()
    settings.paths.ensure()
    df = _features_df()
    df.to_parquet(settings.paths.processed_features, index=False)
    console.print(
        f"[green]OK[/green] wrote {len(df)} rows x {df.shape[1]} cols -> "
        f"{settings.paths.processed_features}"
    )


def _load_features_or_build() -> pd.DataFrame:
    settings = get_settings()
    if settings.paths.processed_features.exists():
        return pd.read_parquet(settings.paths.processed_features)
    return _features_df()


# ---------------- train ----------------

@train_app.command("quick")
@_guard
def train_quick(
    target: Target = typer.Option("value", help="value or wage"),
    model: Model = typer.Option("lgbm", help="lgbm | xgb | catboost | hgbr"),
) -> None:
    """Fit one booster with default params (no tuning). Smoke-test harness."""
    settings = get_settings()
    features = _load_features_or_build()
    artifact = train_booster(model, target, features, settings=settings)
    out = settings.paths.models_dir / f"{target}__{model}__quick"
    artifact.to_dir(out)
    console.print(f"[green]OK[/green] saved -> {out}")


@train_app.command("baseline")
@_guard
def train_baseline_cmd(
    target: Target = typer.Option("value", help="value or wage"),
    model: Model = typer.Option("elasticnet", help="elasticnet | rf"),
) -> None:
    """Fit a reference baseline (ElasticNet / Random Forest) and persist it."""
    if model not in ALL_BASELINES:
        raise typer.BadParameter(f"--model must be one of {ALL_BASELINES}, got {model!r}.")
    settings = get_settings()
    features = _load_features_or_build()
    artifact = train_baseline(model, target, features, settings=settings)
    out = settings.paths.models_dir / f"{target}__{model}__baseline"
    artifact.to_dir(out)
    console.print(f"[green]OK[/green] {model}/{target} baseline -> {out}")
    console.print(f"  chosen params: {artifact.best_params}")


@train_app.command("tuned")
@_guard
def train_tuned(
    target: Target = typer.Option("value"),
    model: Model = typer.Option("lgbm"),
    params_json: Path = typer.Option(..., help="JSON file of hyperparameters (from `vw tune one`)."),
) -> None:
    """Fit using params from a previous Optuna run."""
    settings = get_settings()
    params = json.loads(params_json.read_text(encoding="utf-8"))
    features = _load_features_or_build()
    artifact = train_booster(model, target, features, params=params, settings=settings)
    out = settings.paths.models_dir / f"{target}__{model}__tuned"
    artifact.to_dir(out)
    console.print(f"[green]OK[/green] saved -> {out}")


# ---------------- tune ----------------

@tune_app.command("one")
@_guard
def tune_one(
    target: Target = typer.Option("value"),
    model: Model = typer.Option("lgbm"),
    trials: int = typer.Option(50),
) -> None:
    settings = get_settings()
    features = _load_features_or_build()
    result = tune(model, target, features, n_trials=trials, settings=settings)
    out_dir = settings.paths.models_dir / f"{target}__{model}__best_params"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "params.json").write_text(json.dumps(result.best_params, indent=2), encoding="utf-8")
    (out_dir / "cv_score.json").write_text(
        json.dumps({"best_cv_log_mae": result.best_value, "n_trials": result.n_trials}, indent=2),
        encoding="utf-8",
    )
    console.print(
        f"[green]OK[/green] {model}/{target}: CV log-MAE = {result.best_value:.4f} "
        f"(n_trials={result.n_trials})"
    )
    console.print(f"  params -> {out_dir / 'params.json'}")


@tune_app.command("all")
@_guard
def tune_all(trials: int = typer.Option(50)) -> None:
    """Tune every booster for every target. The pre-declared selection rule picks the winner."""
    settings = get_settings()
    features = _load_features_or_build()
    results: list[dict[str, object]] = []
    for tgt in ALL_TARGETS:
        for mdl in ALL_BOOSTERS:
            console.print(f"[cyan]Tuning {mdl} / {tgt}...[/cyan]")
            r = tune(mdl, tgt, features, n_trials=trials, settings=settings)
            out_dir = settings.paths.models_dir / f"{tgt}__{mdl}__best_params"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "params.json").write_text(json.dumps(r.best_params, indent=2), encoding="utf-8")
            results.append({"target": tgt, "model": mdl, "cv_log_mae": r.best_value})

    # Select winner per target by the pre-declared rule (lowest CV log-MAE).
    results_df = pd.DataFrame(results)
    results_df.to_csv(settings.paths.models_dir / "bakeoff_cv.csv", index=False)
    winners = results_df.sort_values("cv_log_mae").groupby("target").head(1)
    (settings.paths.models_dir / "winners.json").write_text(
        winners.to_json(orient="records", indent=2), encoding="utf-8"
    )
    console.print("[bold green]Bake-off results[/bold green]")
    tbl = Table()
    for col in results_df.columns:
        tbl.add_column(col)
    for _, row in results_df.sort_values(["target", "cv_log_mae"]).iterrows():
        tbl.add_row(*[str(row[c]) for c in results_df.columns])
    console.print(tbl)
    console.print(f"[green]Winners saved[/green] -> {settings.paths.models_dir / 'winners.json'}")


# ---------------- evaluate ----------------

def _winner_artifact_dir(target: Target) -> Path | None:
    settings = get_settings()
    winners_file = settings.paths.models_dir / "winners.json"
    if not winners_file.exists():
        return None
    winners = json.loads(winners_file.read_text(encoding="utf-8"))
    for w in winners:
        if w["target"] == target:
            return settings.paths.models_dir / f"{target}__{w['model']}__tuned"
    return None


def _resolve_artifact_dir(target: Target, model: Model | None) -> Path:
    settings = get_settings()
    if model:
        for suffix in ("tuned", "quick", "baseline"):
            cand = settings.paths.models_dir / f"{target}__{model}__{suffix}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No saved artifact for target={target}, model={model}.")
    for suffix in ("tuned", "quick", "baseline"):
        for mdl in ALL_BOOSTERS:
            cand = settings.paths.models_dir / f"{target}__{mdl}__{suffix}"
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"No saved artifact for target={target}. Run `vw train quick --target {target}` first."
    )


@eval_app.command("run")
@_guard
def evaluate_run(
    target: Target = typer.Option("value"),
    model: Model = typer.Option(None, help="Specific model name; defaults to any saved artifact."),
) -> None:
    settings = get_settings()
    art_dir = _resolve_artifact_dir(target, model)
    artifact = TrainedArtifact.from_dir(art_dir)
    features = _load_features_or_build()
    split = make_splits(features, target, cfg=settings.split)
    target_col = TARGET_COLUMN[target]

    preds = predict(artifact, split.test)
    y_true_log = np.log1p(split.test[target_col].to_numpy(dtype=float))

    metrics = compute_metrics(y_true_log, preds["pred"])
    mae_point, mae_lo, mae_hi = bootstrap_metric_ci(y_true_log, preds["pred"], metric="mae")
    r2_point, r2_lo, r2_hi = bootstrap_metric_ci(y_true_log, preds["pred"], metric="r2")

    row = {
        "target": target,
        "model": artifact.model_name,
        "n_test": metrics.n,
        "mae_log": round(metrics.mae_log, 4),
        "mae_log_ci": f"[{mae_lo:.4f}, {mae_hi:.4f}]",
        "r2_log": round(metrics.r2_log, 4),
        "r2_log_ci": f"[{r2_lo:.4f}, {r2_hi:.4f}]",
        "spearman": round(metrics.spearman, 4),
        "median_abs_pct_error": round(metrics.median_abs_pct_error, 4),
    }

    out_json = settings.paths.reports_dir / f"metrics_{target}_{artifact.model_name}.json"
    out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
    console.print_json(data=row)

    # Diagnostic plots.
    plot_pred_vs_actual(
        y_true_log,
        preds["pred"],
        title=f"{target} — {artifact.model_name}: pred vs actual (log1p)",
        out_path=settings.paths.figures_dir / f"pred_vs_actual_{target}_{artifact.model_name}.png",
    )
    plot_residuals(
        y_true_log,
        preds["pred"],
        covariate=split.test["age"].to_numpy(dtype=float),
        covariate_name="age",
        title=f"{target} — residuals vs age",
        out_path=settings.paths.figures_dir / f"resid_vs_age_{target}_{artifact.model_name}.png",
    )

    # Persist the per-row predictions for downstream stages (no intervals).
    preds_df = summarise_predictions(split.test, y_true_log, preds["pred"], target_col=target_col)
    preds_df.to_parquet(
        settings.paths.processed_dir / f"predictions_{target}_{artifact.model_name}.parquet",
        index=False,
    )
    console.print(f"[green]OK[/green] metrics -> {out_json}")


# ---------------- explain ----------------

@explain_app.command("run")
@_guard
def explain_run(
    target: Target = typer.Option("value"),
    model: Model = typer.Option(None),
    top_waterfalls: int = typer.Option(5),
) -> None:
    settings = get_settings()
    art_dir = _resolve_artifact_dir(target, model)
    artifact = TrainedArtifact.from_dir(art_dir)
    features = _load_features_or_build()
    split = make_splits(features, target, cfg=settings.split)
    feature_names = [*artifact.numeric_cols, *artifact.categorical_cols]
    X_test = pd.DataFrame(
        artifact.preprocessor.transform(_normalize_na(split.test[feature_names])),
        columns=feature_names,
        index=split.test.index,
    )
    result = explain_tree_model(artifact.model, X_test)
    importance = result.global_importance()
    importance_path = settings.paths.reports_dir / f"shap_importance_{target}_{artifact.model_name}.csv"
    importance.to_csv(importance_path, index=False)
    save_global_summary_plot(
        result,
        settings.paths.figures_dir / f"shap_summary_{target}_{artifact.model_name}.png",
    )

    # Top-N waterfalls by magnitude of prediction.
    preds = artifact.model.predict(X_test)
    order = np.argsort(-np.abs(preds - preds.mean()))[:top_waterfalls]
    for rank, idx in enumerate(order, start=1):
        save_waterfall_plot(
            result,
            int(idx),
            settings.paths.figures_dir
            / f"shap_waterfall_{target}_{artifact.model_name}_{rank:02d}.png",
        )
    console.print(f"[green]OK[/green] SHAP importance -> {importance_path}")
    console.print(f"[green]OK[/green] {top_waterfalls} waterfall plots -> {settings.paths.figures_dir}")


# ---------------- mispricing ----------------

@misp_app.command("build")
@_guard
def mispricing_build(
    target: Target = typer.Option("value"),
    model: Model = typer.Option(None),
    top_n: int = typer.Option(20),
) -> None:
    settings = get_settings()
    art_dir = _resolve_artifact_dir(target, model)
    artifact = TrainedArtifact.from_dir(art_dir)
    pred_file = (
        settings.paths.processed_dir / f"predictions_{target}_{artifact.model_name}.parquet"
    )
    if not pred_file.exists():
        raise FileNotFoundError(
            f"Expected {pred_file}. Run `vw evaluate run --target {target}` first."
        )
    preds = pd.read_parquet(pred_file)
    # Intervals were dropped; the board uses a global residual sigma (ranking is unchanged
    # from the old near-constant-width intervals).
    board = build_board(preds, top_n=top_n)
    out_dir = settings.paths.reports_dir / f"mispricing_{target}_{artifact.model_name}"
    board.write(out_dir)
    console.print(f"[green]OK[/green] mispricing board -> {out_dir}")


# ---------------- export ----------------

@export_app.command("web")
@_guard
def export_web(model_name: str = typer.Option("lgbm", help="Which tuned model to export.")) -> None:
    """Produce the JSON bundle under `web/data/` that any HTML UI can consume."""
    settings = get_settings()
    bundle = export_web_bundle(settings=settings, model_name=model_name)
    console.print(f"[green]OK[/green] players    -> {bundle.players_path}")
    console.print(f"[green]OK[/green] meta       -> {bundle.meta_path}")
    console.print(f"[green]OK[/green] mispricing -> {bundle.mispricing_path}")
    console.print(f"[green]OK[/green] detail     -> {bundle.detail_dir} ({bundle.n_detail} players)")


# ---------------- compare ----------------

@compare_app.command("run")
@_guard
def compare_run(
    target: Target = typer.Option("value", help="value or wage"),
) -> None:
    """Score the baselines and the production LightGBM on the held-out test season, side by side.

    Reference ladder: per-position median floor -> ElasticNet -> Random Forest -> LightGBM.
    Writes `reports/model_comparison_{target}.{csv,md}` and prints the table.
    """
    settings = get_settings()
    features = _load_features_or_build()
    target_col = TARGET_COLUMN[target]
    split = make_splits(features, target, cfg=settings.split)
    fit_frame = pd.concat([split.train, split.val], axis=0, ignore_index=True)
    y_true_log = np.log1p(split.test[target_col].to_numpy(dtype=float))
    actual = np.expm1(y_true_log)

    def _row(name: str, pred_log: np.ndarray) -> dict[str, object]:
        m = compute_metrics(y_true_log, np.asarray(pred_log, dtype=float))
        pred_native = np.expm1(np.asarray(pred_log, dtype=float))
        mean_ape = float(np.mean(np.abs(pred_native - actual) / np.maximum(actual, 1.0)))
        return {
            "model": name,
            "r2_log": round(m.r2_log, 4),
            "mae_log": round(m.mae_log, 4),
            "mae_native": round(float(np.mean(np.abs(pred_native - actual))), 0),
            "median_ape_pct": round(m.median_abs_pct_error * 100, 2),
            "mean_ape_pct": round(mean_ape * 100, 2),
            "spearman": round(m.spearman, 4),
        }

    rows: list[dict[str, object]] = []

    # 1) Sanity floor: per-position median of the log target.
    floor = median_by_position()
    floor.fit(fit_frame, np.log1p(fit_frame[target_col].to_numpy(dtype=float)))
    rows.append(_row("median_by_position", floor.predict(split.test)))

    # 2) Baselines (trained fresh on train+val).
    for mdl in ALL_BASELINES:
        console.print(f"[cyan]Fitting baseline {mdl}...[/cyan]")
        art = train_baseline(mdl, target, features, settings=settings)
        rows.append(_row(mdl, predict(art, split.test)["pred"]))

    # 3) Tuned boosters, if present on disk. lgbm is the production model; the rest
    #    (xgb, catboost, hgbr) are included whenever their tuned artifact exists.
    found_booster = False
    for mdl in ALL_BOOSTERS:
        booster_dir = settings.paths.models_dir / f"{target}__{mdl}__tuned"
        if not booster_dir.exists():
            continue
        found_booster = True
        art = TrainedArtifact.from_dir(booster_dir)
        label = f"{mdl} (production)" if mdl == "lgbm" else mdl
        rows.append(_row(label, predict(art, split.test)["pred"]))
    if not found_booster:
        console.print("[yellow]No tuned boosters found; comparing baselines only.[/yellow]")

    df = pd.DataFrame(rows).sort_values("r2_log", ascending=False).reset_index(drop=True)
    settings.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = settings.paths.reports_dir / f"model_comparison_{target}.csv"
    md_path = settings.paths.reports_dir / f"model_comparison_{target}.md"
    df.to_csv(csv_path, index=False)
    header = (
        f"# Model comparison — {target}\n\n"
        f"Held-out test season {settings.split.test_seasons} "
        f"(n={len(y_true_log)}). Trained on {settings.split.train_seasons + settings.split.val_seasons}. "
        "Same feature allowlist and `log1p` target for every model.\n\n"
    )
    cols = list(df.columns)
    md_lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()],
    ]
    md_path.write_text(header + "\n".join(md_lines) + "\n", encoding="utf-8")

    tbl = Table(title=f"Model comparison — {target} (test {settings.split.test_seasons[0]})")
    for col in df.columns:
        tbl.add_column(col)
    for _, r in df.iterrows():
        tbl.add_row(*[str(r[c]) for c in df.columns])
    console.print(tbl)
    console.print(f"[green]OK[/green] -> {csv_path}")
    console.print(f"[green]OK[/green] -> {md_path}")


if __name__ == "__main__":
    sys.exit(app())
