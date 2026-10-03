"""Pure feature-engineering functions.

Every function is `DataFrame -> DataFrame` (or `Series -> Series`) with no I/O and no mutation
of its input. This makes every feature unit-testable and recomposable.

The public entry point is `build_feature_matrix`, which composes the primitives into the
model-ready frame described in config/features.yaml.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd

from value_wage.config import FeatureLists, Settings, get_settings

# Canonical primary-position buckets derived from FM's `positions` CSV string.
# Order matters: the first bucket whose code appears in the player's position list wins.
# Rationale: a player listed "DC,DM,MC" is primarily a CB; "AMR,MR" is primarily a winger.
POSITION_BUCKETS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("GK", ("GK",)),
    ("CB", ("DC",)),
    ("FB", ("DL", "DR", "WBL", "WBR")),
    ("DM", ("DM",)),
    ("CM", ("MC",)),
    ("W", ("AML", "AMR", "ML", "MR")),
    ("AM", ("AMC",)),
    ("ST", ("ST",)),
)

FM_TECHNICAL: Final[tuple[str, ...]] = (
    "crossing", "dribbling", "finishing", "first_touch", "heading", "long_shots",
    "marking", "passing", "tackling", "technique",
)
FM_MENTAL: Final[tuple[str, ...]] = (
    "aggression", "anticipation", "bravery", "composure", "concentration",
    "decisions", "determination", "flair", "leadership", "off_the_ball",
    "positioning", "teamwork", "vision", "work_rate",
)
FM_PHYSICAL: Final[tuple[str, ...]] = (
    "acceleration", "pace", "stamina", "strength", "jumping_reach",
)


def parse_primary_position(positions: pd.Series) -> pd.Series:
    """Collapse FM's comma-separated position string into one canonical bucket.

    Rules:
    - Empty / null → NaN (will be imputed downstream or dropped).
    - First matching bucket in POSITION_BUCKETS wins.
    - Unknown codes → NaN (surfaces as a data-quality signal, not silently absorbed).
    """
    def classify(raw: object) -> str | None:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        s = str(raw).strip()
        if not s:
            return None
        codes = {code.strip() for code in s.split(",") if code.strip()}
        for bucket, members in POSITION_BUCKETS:
            if codes & set(members):
                return bucket
        return None

    return positions.map(classify).astype("string")


def months_to_contract_end(contract_until: pd.Series, *, reference: datetime | None = None) -> pd.Series:
    """Months between the reference date and `contract_until`.

    Null `contract_until` → NaN (not zero — "no known contract end" ≠ "expires today").
    Negative values (contract already ended) are clipped to 0.
    """
    reference = reference or datetime(2025, 7, 1)  # start of the 2025-26 season

    def parse(x: object) -> float:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float("nan")
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return float("nan")
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return float("nan")
        delta_days = (dt - reference).days
        return max(0.0, delta_days / 30.4375)

    return contract_until.map(parse).astype(float)


def per_90(numerator: pd.Series, minutes: pd.Series, *, min_minutes: int = 90) -> pd.Series:
    """Rate per 90 minutes. Players with `< min_minutes` → NaN (not an inflated rate from tiny samples)."""
    out = numerator.astype(float) * 90.0 / minutes.replace(0, np.nan).astype(float)
    out = out.where(minutes.astype(float) >= float(min_minutes), other=np.nan)
    return out


def encode_foot(foot: pd.Series) -> pd.Series:
    """Normalise the foot column to {right, left, either}. Unknown → NaN."""
    mapping = {
        "right": "right", "r": "right", "right only": "right",
        "left": "left", "l": "left", "left only": "left",
        "either": "either", "both": "either",
    }

    def classify(x: object) -> str | None:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        key = str(x).strip().lower()
        return mapping.get(key)

    return foot.map(classify).astype("string")


def build_feature_matrix(
    master: pd.DataFrame,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Produce the model-ready frame.

    Snapshot-aware: when the input has a `snapshot_date` column (post-TM join), time-varying
    features (age, contract months) are recomputed per snapshot so start-of-season and
    end-of-season rows differ correctly. Without `snapshot_date`, falls back to the old
    behaviour (one row per player-season, FM-only).
    """
    settings = settings or get_settings()
    features = settings.load_features()
    df = master.copy()

    has_snapshot = "snapshot_date" in df.columns
    minutes = df["minutes"].astype(float)

    # Per-90 performance (unchanged — Understat stats are per-season, not per-snapshot).
    df["np_xg_p90"] = per_90(df["np_xg"], minutes)
    df["xa_p90"] = per_90(df["xa"], minutes)
    df["key_passes_p90"] = per_90(df["key_passes"], minutes)
    df["xg_chain_p90"] = per_90(df["xg_chain"], minutes)
    df["xg_buildup_p90"] = per_90(df["xg_buildup"], minutes)
    df["shots_p90"] = per_90(df["shots"], minutes)

    # Volume.
    df["log1p_minutes"] = np.log1p(minutes)
    df["log1p_goals"] = np.log1p(df["goals"].astype(float))
    df["log1p_assists"] = np.log1p(df["assists"].astype(float))

    # Biographical.
    if has_snapshot and "age_at_snapshot" in df.columns:
        # Prefer the TM-derived, snapshot-exact age. Fall back to master age if missing.
        df["age"] = df["age_at_snapshot"].fillna(df["age"].astype(float))
    df["age_sq"] = df["age"].astype(float) ** 2

    # Contract — reference = snapshot_date when available, else the default (01 Jul 2025).
    if has_snapshot:
        df["months_to_contract_end"] = _months_to_contract_end_per_row(
            df["contract_until"], df["snapshot_date"]
        )
    else:
        df["months_to_contract_end"] = months_to_contract_end(df["contract_until"])

    # Position + foot.
    df["primary_position"] = parse_primary_position(df["positions"])
    df["foot"] = encode_foot(df["foot"])

    # FM aggregates and potential-gap.
    df["fm_technical_mean"] = df[list(FM_TECHNICAL)].mean(axis=1)
    df["fm_mental_mean"] = df[list(FM_MENTAL)].mean(axis=1)
    df["fm_physical_mean"] = df[list(FM_PHYSICAL)].mean(axis=1)
    df["potential_gap"] = df["potential"] - df["current_ability"]

    required = features.all_features()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Feature matrix is missing allowlisted columns {missing}. "
            "Either engineer them in features.py or remove them from config/features.yaml."
        )

    # Keep targets + identifiers + snapshot metadata + features.
    base_keep = [
        "player", "player_id", "season", "club", "team", "position", "positions",
        "snapshot", "snapshot_date", "actual_valuation_date",
        "market_value_eur", "wage_pw",
    ]
    keep = [c for c in base_keep if c in df.columns] + required
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    out_cols = [c for c in keep if not (c in seen or seen.add(c))]
    return df.loc[:, out_cols].copy()


def _months_to_contract_end_per_row(
    contract_until: pd.Series,
    snapshot_date: pd.Series,
) -> pd.Series:
    """Per-row version of months_to_contract_end: each row uses its own snapshot date as reference."""
    out = []
    for cu, sd in zip(contract_until, snapshot_date, strict=True):
        try:
            ref = datetime.fromisoformat(str(sd)) if pd.notna(sd) else datetime(2025, 7, 1)
        except ValueError:
            ref = datetime(2025, 7, 1)
        out.append(months_to_contract_end(pd.Series([cu]), reference=ref).iloc[0])
    return pd.Series(out, index=contract_until.index, dtype=float)


def feature_lists_for(_target: str, features: FeatureLists) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature-column names for the given target.

    Currently identical for both targets; split by argument so future per-target lists
    (e.g. GK-only features) have a hook without a refactor.
    """
    return list(features.shared_numeric), list(features.shared_categorical)
