"""Typed configuration: paths, seeds, feature lists, hyperparameter search spaces.

Single source of truth. Nothing else in the package may hard-code a path or seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent  # the "New folder" that holds both scout-pipeline and this project


class Paths(BaseModel):
    """All filesystem locations the package reads or writes."""

    project_root: Path = PROJECT_ROOT
    scout_pipeline_interim: Path = REPO_ROOT / "scout-pipeline" / "data" / "interim"
    raw_master: Path = PROJECT_ROOT / "data" / "raw" / "master_fm26_pl.parquet"
    tm_dir: Path = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    processed_features: Path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    processed_master_tm: Path = PROJECT_ROOT / "data" / "processed" / "master_with_tm.parquet"
    models_dir: Path = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    mlruns_dir: Path = PROJECT_ROOT / "mlruns"
    optuna_db: Path = PROJECT_ROOT / "optuna.db"
    features_yaml: Path = PROJECT_ROOT / "config" / "features.yaml"

    def ensure(self) -> None:
        """Create every writeable directory. Idempotent."""
        for p in [
            self.processed_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.mlruns_dir,
            self.raw_master.parent,
        ]:
            p.mkdir(parents=True, exist_ok=True)


Target = Literal["value", "wage"]
Model = Literal["dummy", "ridge", "lgbm", "xgb", "catboost", "hgbr"]
ALL_TARGETS: tuple[Target, ...] = ("value", "wage")
ALL_BOOSTERS: tuple[Model, ...] = ("lgbm", "xgb", "catboost", "hgbr")

TARGET_COLUMN: dict[Target, str] = {"value": "market_value_eur", "wage": "wage_pw"}
# The value target is Transfermarkt's `market_value_in_eur` snapshot (joined via sources/transfermarkt_kaggle.py).
# The wage target remains FM's `wage_pw` — Transfermarkt doesn't publish wages reliably.
# FM's `value_high` is intentionally NOT a target: it's computed by the game engine from
# `current_ability` + `potential` + `age` + `contract_until`, so predicting it just inverts
# that formula instead of learning anything useful.


class SplitConfig(BaseModel):
    """Season-based splits. 2025-26 → test, 2024-25 → val, 2023-24 → train."""

    train_seasons: tuple[str, ...] = ("2023-24",)
    val_seasons: tuple[str, ...] = ("2024-25",)
    test_seasons: tuple[str, ...] = ("2025-26",)
    inner_cv_folds: int = 3


class FeatureLists(BaseModel):
    """Loaded from config/features.yaml. Enforces the allowlist discipline."""

    shared_numeric: list[str]
    shared_categorical: list[str]
    excluded: list[str]

    @classmethod
    def load(cls, path: Path) -> FeatureLists:
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"features.yaml is malformed: expected mapping, got {type(doc)}")
        return cls(**doc)

    def all_features(self) -> list[str]:
        return [*self.shared_numeric, *self.shared_categorical]


class OptunaConfig(BaseModel):
    """Shared Optuna recipe. Identical across all four boosters for a fair comparison."""

    n_trials: int = 50
    timeout_sec: int | None = None
    sampler_seed: int = 42
    pruner_warmup_trials: int = 5
    early_stopping_rounds: int = 100
    storage_uri: str = ""  # populated at runtime from Paths.optuna_db


class Settings(BaseSettings):
    """Top-level settings. Overridable via env vars prefixed VW_."""

    model_config = SettingsConfigDict(env_prefix="VW_", env_file=".env", extra="ignore")

    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    split: SplitConfig = Field(default_factory=SplitConfig)
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)
    mlflow_experiment: str = "value-wage"

    def load_features(self) -> FeatureLists:
        return FeatureLists.load(self.paths.features_yaml)


def get_settings() -> Settings:
    """Returns a Settings instance. Call this instead of instantiating directly."""
    return Settings()
