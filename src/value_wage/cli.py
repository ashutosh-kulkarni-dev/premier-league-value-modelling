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

from value_wage.config import ALL_BOOSTERS, ALL_TARGETS, TARGET_COLUMN, Model, Target, get_settings
from value_wage.data import copy_from_scout_pipeline, load_master, load_master_with_tm
from value_wage.evaluate import (
    bootstrap_metric_ci,
    compute_metrics,
    interval_coverage,
    plot_pred_vs_actual,
    plot_residuals,
    summarise_predictions,
)
from value_wage.explain import explain_tree_model, save_global_summary_plot, save_waterfall_plot
from value_wage.features import build_feature_matrix
from value_wage.mispricing import build_board
from value_wage.splits import make_splits
from value_wage.calibration import calibrate
from value_wage.export_web import export as export_web_bundle
from value_wage.train import TrainedArtifact, _normalize_na, predict, train_booster
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

calib_app = typer.Typer(help="Conformal interval calibration.")
export_app = typer.Typer(help="Export results for a static HTML UI.")

app.add_typer(data_app, name="data")
app.add_typer(feat_app, name="features")
app.add_typer(train_app, name="train")
app.add_typer(tune_app, name="tune")
app.add_typer(eval_app, name="evaluate")
app.add_typer(explain_app, name="explain")
app.add_typer(misp_app, name="mispricing")
app.add_typer(calib_app, name="calibrate")
app.add_typer(export_app, name="export")

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
        for suffix in ("tuned", "quick"):
            cand = settings.paths.models_dir / f"{target}__{model}__{suffix}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No saved artifact for target={target}, model={model}.")
    for suffix in ("tuned", "quick"):
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
    if "lower" in preds and "upper" in preds:
        cov = interval_coverage(y_true_log, preds["lower"], preds["upper"])
        row["interval_coverage_80"] = round(cov, 4)

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

    # Persist the per-row predictions for downstream stages.
    preds_df = summarise_predictions(split.test, y_true_log, preds["pred"], target_col=target_col)
    if "lower" in preds and "upper" in preds:
        preds_df["lower_log"] = preds["lower"]
        preds_df["upper_log"] = preds["upper"]
        preds_df["lower"] = np.expm1(preds["lower"])
        preds_df["upper"] = np.expm1(preds["upper"])
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
    lower = preds["lower_log"].to_numpy(dtype=float) if "lower_log" in preds.columns else None
    upper = preds["upper_log"].to_numpy(dtype=float) if "upper_log" in preds.columns else None
    board = build_board(preds, lower_log=lower, upper_log=upper, top_n=top_n)
    out_dir = settings.paths.reports_dir / f"mispricing_{target}_{artifact.model_name}"
    board.write(out_dir)
    console.print(f"[green]OK[/green] mispricing board -> {out_dir}")


# ---------------- calibrate ----------------

@calib_app.command("run")
@_guard
def calibrate_run(
    target: Target = typer.Option("value"),
    model: Model = typer.Option("lgbm"),
    confidence: float = typer.Option(0.8, help="Nominal coverage (e.g. 0.8 for 80% intervals)."),
    conformity: str = typer.Option(
        "absolute",
        help="MAPIE conformity score: 'absolute' (constant-width) or 'gamma' (log-scale, "
        "multiplicative error).",
    ),
    method: str = typer.Option(
        "split",
        help="'split' (train-only point model + val calibration — honest independence, less data) "
        "or 'cross' (CV+ on train+val via GroupKFold — more data, tighter intervals).",
    ),
) -> None:
    """Fit MAPIE split-conformal intervals and report achieved vs nominal coverage on test.

    Overwrites `predictions_{target}_{model}.parquet` with calibrated lower/upper columns and
    writes `metrics_calibrated_{target}_{model}.json`.
    """
    settings = get_settings()
    params_path = settings.paths.models_dir / f"{target}__{model}__best_params" / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"Tuned params not found at {params_path}. Run `vw tune one --target {target} "
            f"--model {model}` first."
        )
    params = json.loads(params_path.read_text(encoding="utf-8"))
    features = _load_features_or_build()
    predictor = calibrate(
        model, target, features, params,
        confidence_level=confidence, conformity_score=conformity,
        method=method, settings=settings,
    )

    split = make_splits(features, target, cfg=settings.split)
    target_col = TARGET_COLUMN[target]
    preds = predictor.predict(split.test)
    y_true = np.log1p(split.test[target_col].to_numpy(dtype=float))

    inside = (y_true >= preds["lower"]) & (y_true <= preds["upper"])
    coverage = float(inside.mean())
    width_log = float(np.mean(preds["upper"] - preds["lower"]))
    median_width_log = float(np.median(preds["upper"] - preds["lower"]))

    from value_wage.evaluate import compute_metrics
    m = compute_metrics(y_true, preds["pred"])

    if method == "cross":
        note = (
            "Cross-conformal (CV+) with GroupKFold on player_id over train+val. "
            "Point predictions ensemble K out-of-fold models; intervals satisfy the coverage "
            "guarantee without sacrificing training data."
        )
    else:
        note = (
            "Split-conformal: point model trained on TRAIN SEASON ONLY (not train+val) so val "
            "could be held out for calibration. Metrics therefore differ slightly from "
            "`evaluate run`."
        )
    out = {
        "target": target,
        "model": model,
        "method": method,
        "conformity_score": conformity,
        "nominal_coverage": confidence,
        "achieved_coverage": round(coverage, 4),
        "mean_interval_width_log": round(width_log, 4),
        "median_interval_width_log": round(median_width_log, 4),
        "point_mae_log": round(m.mae_log, 4),
        "point_r2_log": round(m.r2_log, 4),
        "point_spearman": round(m.spearman, 4),
        "n_test": m.n,
        "note": note,
    }
    out_path = settings.paths.reports_dir / f"metrics_calibrated_{target}_{model}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    console.print_json(data=out)

    # Overwrite the predictions parquet with the calibrated intervals so mispricing re-uses them.
    pred_df = split.test.loc[:, ["player", "player_id", "season", "club", target_col]].copy()
    pred_df["actual_log"] = y_true
    pred_df["pred_log"] = preds["pred"]
    pred_df["actual"] = np.expm1(y_true)
    pred_df["pred"] = np.expm1(preds["pred"])
    pred_df["lower_log"] = preds["lower"]
    pred_df["upper_log"] = preds["upper"]
    pred_df["lower"] = np.expm1(preds["lower"])
    pred_df["upper"] = np.expm1(preds["upper"])
    pred_df["residual_log"] = preds["pred"] - y_true
    pred_df["residual_pct"] = (pred_df["pred"] - pred_df["actual"]) / pred_df["actual"].replace(0, np.nan)
    pred_df.to_parquet(
        settings.paths.processed_dir / f"predictions_calibrated_{target}_{model}.parquet",
        index=False,
    )
    console.print(f"[green]OK[/green] calibrated predictions saved")


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
    console.print(f"[green]OK[/green] bio        -> {bundle.bio_path}")


if __name__ == "__main__":
    sys.exit(app())
