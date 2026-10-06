"""Export the model + results into a static JSON bundle any HTML UI can consume.

Produces four files under `web/data/`:
  - `players.json`  — per-player predictions, intervals, residuals, top-6 SHAP drivers,
                      verdict label, bio, career history, season stats timeline, value trajectory
  - `meta.json`     — model metadata, global SHAP, feature list, metrics
  - `mispricing.json` — ranked mispricing boards (under/over) per target
  - `bio.json`      — per-player bio keyed by player_id (also inlined into players.json;
                      shipped separately for apps that want a smaller first payload)

The HTML side has zero backend requirement. Load the JSON with `fetch`, build any UI.
See `web/README.md` for the data contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

from value_wage.config import ALL_TARGETS, TARGET_COLUMN, Settings, get_settings
from value_wage.splits import make_splits
from value_wage.train import TrainedArtifact, _normalize_na

SHAP_TOP_K = 6


@dataclass(frozen=True)
class WebBundle:
    players_path: Path
    meta_path: Path
    mispricing_path: Path
    bio_path: Path


# -------- Phase 1 enrichment: bio, career, stats timeline, value trajectory --------


def _load_bio(tm_dir: Path) -> dict[int, dict[str, Any]]:
    """TM players.csv → {tm_player_id: bio dict}. Only the subset of fields a UI needs."""
    cols = [
        "player_id", "first_name", "last_name", "name", "date_of_birth",
        "country_of_birth", "country_of_citizenship", "sub_position", "position",
        "foot", "height_in_cm", "contract_expiration_date", "agent_name",
        "image_url", "international_caps", "international_goals",
        "current_club_name", "market_value_in_eur", "highest_market_value_in_eur",
    ]
    df = pd.read_csv(tm_dir / "players.csv", usecols=cols, low_memory=False)
    out: dict[int, dict[str, Any]] = {}
    for _, r in df.iterrows():
        pid = int(r["player_id"])
        out[pid] = {
            "full_name": r.get("name"),
            "first_name": r.get("first_name"),
            "last_name": r.get("last_name"),
            "date_of_birth": None if pd.isna(r.get("date_of_birth")) else str(r["date_of_birth"])[:10],
            "country_of_birth": r.get("country_of_birth"),
            "country_of_citizenship": r.get("country_of_citizenship"),
            "sub_position": r.get("sub_position"),
            "position": r.get("position"),
            "foot": r.get("foot"),
            "height_cm": None if pd.isna(r.get("height_in_cm")) else int(r["height_in_cm"]),
            "contract_expiration_date": (
                None if pd.isna(r.get("contract_expiration_date"))
                else str(r["contract_expiration_date"])[:10]
            ),
            "agent_name": r.get("agent_name"),
            "image_url": r.get("image_url"),
            "international_caps": None if pd.isna(r.get("international_caps")) else int(r["international_caps"]),
            "international_goals": None if pd.isna(r.get("international_goals")) else int(r["international_goals"]),
            "current_club": r.get("current_club_name"),
            "tm_current_market_value_eur": (
                None if pd.isna(r.get("market_value_in_eur"))
                else float(r["market_value_in_eur"])
            ),
            "tm_highest_market_value_eur": (
                None if pd.isna(r.get("highest_market_value_in_eur"))
                else float(r["highest_market_value_in_eur"])
            ),
        }
    return out


def _load_career(tm_dir: Path, player_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    """Transfers history per player, chronological. Loans and permanents."""
    df = pd.read_csv(
        tm_dir / "transfers.csv",
        usecols=["player_id", "transfer_date", "transfer_season",
                 "from_club_name", "to_club_name", "transfer_fee"],
        low_memory=False,
    )
    df = df[df["player_id"].isin(player_ids)]
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df = df.sort_values("transfer_date")
    out: dict[int, list[dict[str, Any]]] = {}
    for pid, grp in df.groupby("player_id"):
        rows = []
        for _, r in grp.iterrows():
            rows.append({
                "date": None if pd.isna(r["transfer_date"]) else r["transfer_date"].date().isoformat(),
                "season": r.get("transfer_season"),
                "from_club": r.get("from_club_name"),
                "to_club": r.get("to_club_name"),
                "fee_eur": None if pd.isna(r.get("transfer_fee")) else float(r["transfer_fee"]),
            })
        out[int(pid)] = rows
    return out


def _load_season_stats(tm_dir: Path, player_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    """Aggregate appearances → per (player, season) totals using games.csv for the season label."""
    games = pd.read_csv(
        tm_dir / "games.csv",
        usecols=["game_id", "season", "competition_id"],
        low_memory=False,
    )
    season_lookup = dict(zip(games["game_id"], games["season"].astype(str), strict=True))

    appearances = pd.read_csv(
        tm_dir / "appearances.csv",
        usecols=["player_id", "game_id", "goals", "assists", "minutes_played",
                 "yellow_cards", "red_cards", "competition_id"],
        low_memory=False,
    )
    appearances = appearances[appearances["player_id"].isin(player_ids)]
    appearances["season"] = appearances["game_id"].map(season_lookup)
    appearances = appearances.dropna(subset=["season"])

    agg = (
        appearances.groupby(["player_id", "season"])
        .agg(
            matches=("game_id", "nunique"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow=("yellow_cards", "sum"),
            red=("red_cards", "sum"),
        )
        .reset_index()
        .sort_values(["player_id", "season"])
    )

    out: dict[int, list[dict[str, Any]]] = {}
    for pid, grp in agg.groupby("player_id"):
        out[int(pid)] = [
            {
                "season": str(r["season"]),
                "matches": int(r["matches"]),
                "minutes": int(r["minutes"]),
                "goals": int(r["goals"]),
                "assists": int(r["assists"]),
                "yellow": int(r["yellow"]),
                "red": int(r["red"]),
            }
            for _, r in grp.iterrows()
        ]
    return out


def _load_value_trajectory(tm_dir: Path, player_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    """Full TM market-value timeline per player."""
    df = pd.read_csv(
        tm_dir / "player_valuations.csv",
        usecols=["player_id", "date", "market_value_in_eur", "current_club_name"],
        low_memory=False,
    )
    df = df[df["player_id"].isin(player_ids)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    out: dict[int, list[dict[str, Any]]] = {}
    for pid, grp in df.groupby("player_id"):
        out[int(pid)] = [
            {
                "date": r["date"].date().isoformat(),
                "market_value_eur": float(r["market_value_in_eur"]),
                "club": r.get("current_club_name"),
            }
            for _, r in grp.iterrows()
            if pd.notna(r["date"]) and pd.notna(r["market_value_in_eur"])
        ]
    return out


# -------- Verdict --------


def _verdict(actual: float | None, predicted: float | None,
             lower: float | None, upper: float | None) -> dict[str, Any] | None:
    if actual is None or predicted is None:
        return None
    pct_err = (predicted - actual) / actual if actual else None
    within_interval = bool(lower is not None and upper is not None
                           and lower <= actual <= upper)
    abs_pct = abs(pct_err) if pct_err is not None else None
    if abs_pct is None:
        label = "unknown"
    elif abs_pct <= 0.20:
        label = "excellent"
    elif within_interval and abs_pct <= 0.50:
        label = "good"
    elif within_interval:
        label = "fair"
    else:
        label = "poor"
    return {
        "label": label,
        "within_interval": within_interval,
        "within_20_pct": abs_pct is not None and abs_pct <= 0.20,
        "pct_err": None if pct_err is None else round(pct_err, 4),
    }


def _resolve_artifact(settings: Settings, target: str, model_name: str = "lgbm") -> TrainedArtifact:
    path = settings.paths.models_dir / f"{target}__{model_name}__tuned"
    if not path.exists():
        raise FileNotFoundError(
            f"Tuned artifact missing at {path}. Run `vw train tuned --target {target} "
            f"--model {model_name} --params-json ...` first."
        )
    return TrainedArtifact.from_dir(path)


def _top_shap_rows(
    shap_values: np.ndarray,
    feature_names: list[str],
    row_idx: int,
    k: int,
) -> list[dict[str, Any]]:
    row = shap_values[row_idx]
    order = np.argsort(-np.abs(row))[:k]
    return [
        {"feature": feature_names[i], "contribution_log": float(row[i])}
        for i in order
    ]


def _build_per_player_rows(
    artifact: TrainedArtifact,
    frame: pd.DataFrame,
    predictions_df: pd.DataFrame,
    target_label: str,
) -> dict[int, dict[str, Any]]:
    """Return a {row_key: per-target block} mapping, keyed by (player_id, season, snapshot).

    Uses the predictions parquet (which already has calibrated intervals) as the source of
    truth for pred/lower/upper, then layers SHAP on top.
    """
    feature_names = [*artifact.numeric_cols, *artifact.categorical_cols]
    X_t = pd.DataFrame(
        artifact.preprocessor.transform(_normalize_na(frame[feature_names])),
        columns=feature_names,
        index=frame.index,
    )
    explainer = shap.TreeExplainer(artifact.model)
    shap_vals = np.asarray(explainer.shap_values(X_t))

    # Join predictions back by (player_id, season) — snapshot column may not survive the eval path
    # so we match on player_id + season, taking the end-of-season row where multiple snapshots exist.
    pred_lookup = {
        (int(r["player_id"]), r["season"]): r for _, r in predictions_df.iterrows()
    }

    out: dict[int, dict[str, Any]] = {}
    for i, row in frame.reset_index(drop=True).iterrows():
        key = (int(row["player_id"]), row["season"], row.get("snapshot", "end"))
        pred_row = pred_lookup.get((int(row["player_id"]), row["season"]))
        block: dict[str, Any] = {
            "actual": float(row[target_label]) if pd.notna(row[target_label]) else None,
            "shap_top": _top_shap_rows(shap_vals, feature_names, i, SHAP_TOP_K),
        }
        if pred_row is not None:
            block["predicted"] = float(pred_row["pred"])
            block["lower"] = float(pred_row["lower"]) if "lower" in pred_row else None
            block["upper"] = float(pred_row["upper"]) if "upper" in pred_row else None
            block["residual_pct"] = (
                float(pred_row["residual_pct"]) if pd.notna(pred_row.get("residual_pct")) else None
            )
        out[hash(key)] = {"key": {"player_id": key[0], "season": key[1], "snapshot": key[2]}, "block": block}
    return out


def export(settings: Settings | None = None, model_name: str = "lgbm") -> WebBundle:
    settings = settings or get_settings()
    web_dir = settings.paths.project_root / "web" / "data"
    web_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(settings.paths.processed_features)

    # -------- Phase 1 enrichment: load the extra Kaggle tables --------
    tm_dir = settings.paths.tm_dir
    # Resolve TM player ids for everyone in our master. Build a mapping once.
    tm_master = pd.read_parquet(settings.paths.processed_master_tm)
    tm_id_by_master_pid: dict[int, int] = {}
    for _, r in tm_master.dropna(subset=["player_tm_id", "player_id"]).iterrows():
        tm_id_by_master_pid[int(r["player_id"])] = int(r["player_tm_id"])
    tm_player_ids = set(tm_id_by_master_pid.values())

    bio_by_tm = _load_bio(tm_dir)
    career_by_tm = _load_career(tm_dir, tm_player_ids)
    stats_by_tm = _load_season_stats(tm_dir, tm_player_ids)
    valtraj_by_tm = _load_value_trajectory(tm_dir, tm_player_ids)

    per_player: dict[tuple[int, str, str], dict[str, Any]] = {}

    # Base identity rows across the whole feature matrix (not just test),
    # so the UI can list every player even if unlabelled.
    identity_cols = [c for c in [
        "player", "player_id", "season", "snapshot", "club", "team", "position",
        "age", "minutes", "primary_position",
    ] if c in features.columns]
    def _safe(x: Any) -> Any:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except (TypeError, ValueError):
            pass
        return x

    # Position label shown in the UI comes ONLY from Transfermarkt's
    # sub_position. FM's multi-code list is unreliable for primary position
    # (Luke Shaw's "DC,DL" misclassifies as CB; Šeško's "AMC,ST" as AM).
    # Rows with no TM sub_position are left as null — they won't match any
    # position chip, which is the honest outcome when TM has no data.
    _SUB_POSITION_TO_BUCKET: dict[str, str] = {
        "goalkeeper": "GK",
        "centre-back": "CB", "center-back": "CB",
        "left-back": "FB", "right-back": "FB",
        "left wing-back": "FB", "right wing-back": "FB",
        "defensive midfield": "DM",
        "central midfield": "CM",
        "left midfield": "W", "right midfield": "W",
        "attacking midfield": "AM",
        "left winger": "W", "right winger": "W",
        "centre-forward": "ST", "center-forward": "ST",
        "second striker": "ST",
    }

    def _canonical_position(sub_position: Any) -> Any:
        sub = _safe(sub_position)
        if not sub:
            return None
        return _SUB_POSITION_TO_BUCKET.get(str(sub).strip().lower())

    for _, r in features[identity_cols].iterrows():
        pid = int(r["player_id"])
        key = (pid, r["season"], _safe(r.get("snapshot")) or "end")
        if key not in per_player:
            tm_id = tm_id_by_master_pid.get(pid)
            bio_row = bio_by_tm.get(tm_id) if tm_id else None
            per_player[key] = {
                "player_id": pid,
                "player_tm_id": tm_id,
                "player": _safe(r["player"]),
                "season": r["season"],
                "snapshot": _safe(r.get("snapshot")) or "end",
                "club": _safe(r.get("club")) or _safe(r.get("team")),
                "position": _canonical_position((bio_row or {}).get("sub_position")),
                "age": None if pd.isna(r.get("age")) else float(r["age"]),
                "minutes": None if pd.isna(r.get("minutes")) else int(r["minutes"]),
                "bio": bio_row,
                "career": career_by_tm.get(tm_id, []) if tm_id else [],
                "season_stats": stats_by_tm.get(tm_id, []) if tm_id else [],
                "value_trajectory": valtraj_by_tm.get(tm_id, []) if tm_id else [],
                "value": None,
                "wage": None,
            }

    meta_blocks: dict[str, Any] = {"targets": {}}

    for target in ALL_TARGETS:
        artifact = _resolve_artifact(settings, target, model_name)
        target_col = TARGET_COLUMN[target]
        split = make_splits(features, target, cfg=settings.split)
        pred_path = (
            settings.paths.processed_dir / f"predictions_{target}_{artifact.model_name}.parquet"
        )
        if not pred_path.exists():
            raise FileNotFoundError(
                f"{pred_path} missing. Run `vw evaluate run --target {target}` (and optionally "
                f"`vw calibrate run --target {target}`) first."
            )
        predictions_df = pd.read_parquet(pred_path)

        # SHAP + predictions for the test slice (the "scored" rows).
        feature_names = [*artifact.numeric_cols, *artifact.categorical_cols]
        X_t = pd.DataFrame(
            artifact.preprocessor.transform(_normalize_na(split.test[feature_names])),
            columns=feature_names,
            index=split.test.index,
        )
        explainer = shap.TreeExplainer(artifact.model)
        shap_vals = np.asarray(explainer.shap_values(X_t))

        pred_lookup = {
            (int(r["player_id"]), r["season"]): r for _, r in predictions_df.iterrows()
        }

        for ridx, (_, r) in enumerate(split.test.reset_index(drop=True).iterrows()):
            key = (int(r["player_id"]), r["season"], r.get("snapshot", "end"))
            if key not in per_player:
                continue
            pred_row = pred_lookup.get((int(r["player_id"]), r["season"]))
            block: dict[str, Any] = {
                "actual": float(r[target_col]) if pd.notna(r[target_col]) else None,
                "shap_top": _top_shap_rows(shap_vals, feature_names, ridx, SHAP_TOP_K),
            }
            if pred_row is not None:
                block["predicted"] = float(pred_row["pred"])
                block["lower"] = (
                    float(pred_row["lower"]) if "lower" in pred_row and pd.notna(pred_row["lower"]) else None
                )
                block["upper"] = (
                    float(pred_row["upper"]) if "upper" in pred_row and pd.notna(pred_row["upper"]) else None
                )
                block["residual_pct"] = (
                    float(pred_row["residual_pct"]) if pd.notna(pred_row.get("residual_pct")) else None
                )
            block["verdict"] = _verdict(
                block.get("actual"), block.get("predicted"),
                block.get("lower"), block.get("upper"),
            )
            per_player[key][target] = block

        # Meta: global SHAP + metrics.
        importance = (
            np.abs(shap_vals).mean(axis=0).tolist()
        )
        global_shap = sorted(
            [{"feature": feature_names[i], "mean_abs_shap": float(importance[i])} for i in range(len(feature_names))],
            key=lambda d: -d["mean_abs_shap"],
        )[:15]
        metrics_file = settings.paths.reports_dir / f"metrics_calibrated_{target}_{artifact.model_name}.json"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
        meta_blocks["targets"][target] = {
            "model": artifact.model_name,
            "metrics": metrics,
            "shap_global_top15": global_shap,
            "features": feature_names,
        }

    # Mispricing (reads the CSVs produced by `vw mispricing build`).
    mispricing: dict[str, Any] = {}
    for target in ALL_TARGETS:
        dir_ = settings.paths.reports_dir / f"mispricing_{target}_{model_name}"
        if not dir_.exists():
            continue
        under = pd.read_csv(dir_ / "mispricing_undervalued.csv").head(20)
        over = pd.read_csv(dir_ / "mispricing_overvalued.csv").head(20)
        mispricing[target] = {
            "market_underpays": under.to_dict(orient="records"),
            "market_overpays": over.to_dict(orient="records"),
        }

    players_payload = sorted(
        per_player.values(),
        key=lambda d: (d["player"], d["season"], d["snapshot"]),
    )

    players_path = web_dir / "players.json"
    meta_path = web_dir / "meta.json"
    mispricing_path = web_dir / "mispricing.json"
    bio_path = web_dir / "bio.json"

    players_path.write_text(_dumps(players_payload), encoding="utf-8")
    meta_path.write_text(_dumps(meta_blocks, indent=2), encoding="utf-8")
    mispricing_path.write_text(_dumps(mispricing), encoding="utf-8")

    # Smaller companion payload: bio only, keyed by player_id. Lets a UI do a lighter first
    # fetch and lazy-load the full stats/trajectory from players.json on demand.
    bio_only = {str(k): v for k, v in bio_by_tm.items() if k in tm_player_ids}
    bio_path.write_text(_dumps(bio_only), encoding="utf-8")

    return WebBundle(players_path, meta_path, mispricing_path, bio_path)


def _json_safe(obj: Any) -> Any:
    """json.dump `default=` fallback. Catches numpy types and pandas NAs."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def _clean_nans(obj: Any) -> Any:
    """Recursively replace pandas/numpy NA with None so NO raw `NaN` ever hits the JSON output.

    `json.dumps` with `default=` only fires on unknown TYPES, not on float('nan') which Python
    treats as a valid float. So we scrub NaN proactively before dumping.
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def _dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(_clean_nans(obj), default=_json_safe, allow_nan=False, **kwargs)
