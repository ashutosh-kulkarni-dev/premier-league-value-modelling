"""Data loading, validation, and provenance tracking.

Contract:
- `load_master()` returns the raw master table or raises. Never returns a silently truncated frame.
- `copy_from_scout_pipeline()` is the only writer into `data/raw/`.
- Every load records the input file's SHA-256 for provenance.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from value_wage.config import Settings, get_settings
from value_wage.sources.transfermarkt_kaggle import TMJoinReport, build_tm_target_frame

MASTER_FILENAME = "master_fm26_pl.parquet"

# A tight schema for the columns the rest of the pipeline relies on.
# Columns outside this list are tolerated but not required; schema failures are hard errors.
MASTER_SCHEMA = DataFrameSchema(
    columns={
        "season": Column(str, nullable=False),
        "player": Column(str, nullable=False),
        "player_id": Column(pa.Int64, nullable=False),
        "team": Column(str, nullable=True),
        "position": Column(str, nullable=True),
        "minutes": Column(pa.Int64, nullable=False, checks=pa.Check.ge(0)),
        "matches": Column(pa.Int64, nullable=False, checks=pa.Check.ge(0)),
        "age": Column(float, nullable=True, checks=pa.Check.in_range(14, 50)),
        "goals": Column(pa.Int64, nullable=False, checks=pa.Check.ge(0)),
        "assists": Column(pa.Int64, nullable=False, checks=pa.Check.ge(0)),
        "xg": Column(float, nullable=True, checks=pa.Check.ge(0)),
        "xa": Column(float, nullable=True, checks=pa.Check.ge(0)),
        "np_xg": Column(float, nullable=True, checks=pa.Check.ge(0)),
        "value_high": Column(float, nullable=True, checks=pa.Check.ge(0)),
        "wage_pw": Column(float, nullable=True, checks=pa.Check.ge(0)),
        "current_ability": Column(float, nullable=True, checks=pa.Check.in_range(1, 220)),
        "potential": Column(float, nullable=True, checks=pa.Check.in_range(1, 220)),
        "positions": Column(str, nullable=True),
        "contract_until": Column(str, nullable=True),
    },
    strict=False,  # extra columns allowed
    coerce=True,
)


@dataclass(frozen=True)
class LoadResult:
    """Return value of `load_master`: the frame + its provenance."""

    frame: pd.DataFrame
    source_path: Path
    sha256: str
    row_count: int
    season_counts: dict[str, int]


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def copy_from_scout_pipeline(settings: Settings | None = None) -> Path:
    """Copy the master parquet from scout-pipeline/data/interim into data/raw.

    Raises:
        FileNotFoundError: if the source parquet does not exist.
    """
    settings = settings or get_settings()
    settings.paths.ensure()
    src = settings.paths.scout_pipeline_interim / MASTER_FILENAME
    if not src.exists():
        raise FileNotFoundError(
            f"Expected master parquet at {src}. "
            "Run scout-pipeline's build step first, or point VW_PATHS__SCOUT_PIPELINE_INTERIM "
            "at the correct directory."
        )
    dst = settings.paths.raw_master
    shutil.copy2(src, dst)
    return dst


def load_master(settings: Settings | None = None) -> LoadResult:
    """Load the master parquet with strict schema validation and provenance.

    Raises:
        FileNotFoundError: if the raw file is missing (did you run `make data`?).
        pandera.errors.SchemaError: on any schema violation.
        ValueError: on structural anomalies (empty frame, duplicate player-season rows).
    """
    settings = settings or get_settings()
    src = settings.paths.raw_master
    if not src.exists():
        raise FileNotFoundError(
            f"Raw master parquet not found at {src}. Run `make data` to populate it."
        )

    frame = pd.read_parquet(src)
    if frame.empty:
        raise ValueError(f"Master parquet at {src} is empty.")

    # Pandera reads some stringy columns as `object`; cast those we rely on.
    for col in ("season", "player", "team", "position", "positions", "contract_until"):
        if col in frame.columns:
            frame[col] = frame[col].astype("string")

    validated = MASTER_SCHEMA.validate(frame, lazy=True)

    # Enforce uniqueness of (player_id, season) — the row grain of the master.
    dup_mask = validated.duplicated(subset=["player_id", "season"], keep=False)
    if bool(dup_mask.any()):
        n_dups = int(dup_mask.sum())
        raise ValueError(
            f"Master parquet contains {n_dups} duplicate (player_id, season) rows. "
            "Expected exactly one row per player per season."
        )

    return LoadResult(
        frame=validated,
        source_path=src,
        sha256=sha256_file(src),
        row_count=len(validated),
        season_counts={str(k): int(v) for k, v in validated["season"].value_counts().items()},
    )


@dataclass(frozen=True)
class TMLoadResult:
    """Return of `load_master_with_tm`: expanded frame + join diagnostics."""

    frame: pd.DataFrame
    master_sha256: str
    tm_dir: Path
    join_report: TMJoinReport


def load_master_with_tm(settings: Settings | None = None) -> TMLoadResult:
    """Load the FM/Understat master and overlay the Transfermarkt per-snapshot target.

    The returned frame has TWO rows per (player, season), one per snapshot in {start, end},
    with `market_value_eur` populated where a TM match was found. Rows where the TM join
    missed are kept but have NaN target (predict-only).
    """
    settings = settings or get_settings()
    master_result = load_master(settings)
    if not settings.paths.tm_dir.exists():
        raise FileNotFoundError(
            f"TM Kaggle dump directory {settings.paths.tm_dir} missing. "
            "Extract https://www.kaggle.com/datasets/davidcariboo/player-scores there."
        )
    expanded, report = build_tm_target_frame(master_result.frame, settings.paths.tm_dir)
    return TMLoadResult(
        frame=expanded,
        master_sha256=master_result.sha256,
        tm_dir=settings.paths.tm_dir,
        join_report=report,
    )
