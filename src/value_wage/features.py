"""Pure feature-engineering functions.

Position derivation (v6 regression fix):
    1. Authoritative lookup: TM `sub_position` keyed on `player_tm_id`.
    2. Fallback: scraped_positions.csv (name+club fuzzy) for players TM-join missed.
    3. No fallback to FM positions parser (confirmed in MEMORY note).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from value_wage.config import FeatureLists, Settings, get_settings

# TM sub_position → 8-bucket canonical vocab. Authoritative.
TM_SUBPOS_TO_BUCKET: Final[dict[str, str]] = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB",
    "Right-Back": "FB",
    "Defensive Midfield": "DM",
    "Central Midfield": "CM",
    "Attacking Midfield": "AM",
    "Left Winger": "W",
    "Right Winger": "W",
    "Left Midfield": "W",
    "Right Midfield": "W",
    "Centre-Forward": "ST",
    "Second Striker": "ST",
}

# Short-code (used by TM search pages, surfaced by the scraper).
TM_SHORT_TO_BUCKET: Final[dict[str, str]] = {
    "GK": "GK",
    "CB": "CB", "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "DM": "DM", "CM": "CM", "AM": "AM",
    "LW": "W", "RW": "W", "LM": "W", "RM": "W",
    "CF": "ST", "ST": "ST", "SS": "ST",
}

# Default location of the scraper CSV (fallback). Override via settings.paths.scratchpad or the
# SCRAPED_POSITIONS env var when the layout differs.
import os as _os
_DEFAULT_SCRAPED_POSITIONS = Path(
    _os.environ.get(
        "SCRAPED_POSITIONS",
        str(Path(__file__).resolve().parents[3] / "scratchpad" / "scraped_positions.csv"),
    )
)


def _normalise_name(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    norm = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    norm = norm.replace("�", " ").replace("-", " ").replace("'", "")
    return " ".join(norm.lower().split())


def _club_token(s: object) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    n = _normalise_name(s)
    # Strip common suffix tokens
    n = re.sub(r"\b(fc|afc|cf|united|utd|town|city|wanderers|hotspur|and|&|hove|albion)\b", " ", n)
    return " ".join(n.split())


def primary_position_from_tm(
    player_tm_id: pd.Series,
    tm_players_csv: Path | str,
) -> pd.Series:
    """Look up primary_position per row from TM `sub_position`, keyed on `player_tm_id`."""
    tm = pd.read_csv(tm_players_csv, low_memory=False, usecols=["player_id", "sub_position"])
    tm = tm.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])
    tm["bucket"] = tm["sub_position"].map(TM_SUBPOS_TO_BUCKET)
    lookup = dict(zip(tm["player_id"].astype("Int64").astype(str), tm["bucket"]))

    def classify(x: object) -> str | None:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            key = str(int(float(x)))
        except (TypeError, ValueError):
            return None
        v = lookup.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    return player_tm_id.map(classify).astype("string")


def apply_scraper_fallback(
    df: pd.DataFrame,
    scraper_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Fill NaN primary_position using scraped_positions.csv (name+club fuzzy match).

    The scraper CSV has columns name, club, season, bucket, note. Only `note=='ok'` rows
    with a non-null bucket are consulted. Matching is on (normalised name, club token
    overlap). Any still-null primary_position after this returns NaN.
    """
    path = Path(scraper_csv or _DEFAULT_SCRAPED_POSITIONS)
    if not path.exists():
        return df
    scr = pd.read_csv(path)
    scr = scr[scr["bucket"].notna()].copy()
    if scr.empty:
        return df
    scr["name_norm"] = scr["name"].map(_normalise_name)
    scr["club_tok"] = scr["club"].map(_club_token)

    # Index scraper hits by name_norm → list of (club_tok, bucket)
    by_name: dict[str, list[tuple[str, str]]] = {}
    for _, r in scr.iterrows():
        by_name.setdefault(r["name_norm"], []).append((r["club_tok"], str(r["bucket"])))

    out = df.copy()
    mask = out["primary_position"].isna()
    if not mask.any():
        return out

    sub = out.loc[mask, ["player", "team"]].copy()
    sub["_name_norm"] = sub["player"].map(_normalise_name)
    sub["_club_tok"] = sub["team"].map(_club_token)

    def pick(row) -> str | None:
        hits = by_name.get(row["_name_norm"])
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0][1]
        # Multiple hits for same name — pick by club token overlap
        rt = set(row["_club_tok"].split())
        best, best_score = None, -1
        for ct, bucket in hits:
            score = len(rt & set(ct.split()))
            if score > best_score:
                best_score = score
                best = bucket
        return best

    filled = sub.apply(pick, axis=1)
    out.loc[mask, "primary_position"] = filled.values
    out["primary_position"] = out["primary_position"].astype("string")
    return out


def count_fm_positions(positions: pd.Series) -> pd.Series:
    def _count(raw: object) -> float:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return float("nan")
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return float("nan")
        toks = {c.strip().upper() for c in re.split(r"[,/;|\s]+", s) if c.strip()}
        return float(len(toks)) if toks else float("nan")

    return positions.map(_count).astype(float)


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
    """Legacy FM parser — kept for back-compat but no longer the authoritative source."""
    POSITION_BUCKETS = (
        ("GK", ("GK", "G")),
        ("CB", ("DC", "D")),
        ("FB", ("DL", "DR", "WBL", "WBR", "WB")),
        ("DM", ("DM",)),
        ("CM", ("MC", "M")),
        ("W", ("AML", "AMR", "ML", "MR", "W")),
        ("AM", ("AMC", "AM")),
        ("ST", ("ST", "F", "S")),
    )
    def classify(raw: object) -> str | None:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        s = str(raw).strip()
        if not s:
            return None
        codes = {c.strip() for c in re.split(r"[,\s]+", s) if c.strip()}
        for bucket, members in POSITION_BUCKETS:
            if codes & set(members):
                return bucket
        return None
    return positions.map(classify).astype("string")


def months_to_contract_end(contract_until: pd.Series, *, reference: datetime | None = None) -> pd.Series:
    reference = reference or datetime(2025, 7, 1)
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
    out = numerator.astype(float) * 90.0 / minutes.replace(0, np.nan).astype(float)
    out = out.where(minutes.astype(float) >= float(min_minutes), other=np.nan)
    return out


def encode_foot(foot: pd.Series) -> pd.Series:
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
    settings = settings or get_settings()
    features = settings.load_features()
    df = master.copy()

    has_snapshot = "snapshot_date" in df.columns
    minutes = df["minutes"].astype(float)

    df["np_xg_p90"] = per_90(df["np_xg"], minutes)
    df["xa_p90"] = per_90(df["xa"], minutes)
    df["key_passes_p90"] = per_90(df["key_passes"], minutes)
    df["xg_chain_p90"] = per_90(df["xg_chain"], minutes)
    df["xg_buildup_p90"] = per_90(df["xg_buildup"], minutes)
    df["shots_p90"] = per_90(df["shots"], minutes)

    df["log1p_minutes"] = np.log1p(minutes)
    df["log1p_goals"] = np.log1p(df["goals"].astype(float))
    df["log1p_assists"] = np.log1p(df["assists"].astype(float))

    if has_snapshot and "age_at_snapshot" in df.columns:
        df["age"] = df["age_at_snapshot"].fillna(df["age"].astype(float))
    df["age_sq"] = df["age"].astype(float) ** 2

    if has_snapshot:
        df["months_to_contract_end"] = _months_to_contract_end_per_row(
            df["contract_until"], df["snapshot_date"]
        )
    else:
        df["months_to_contract_end"] = months_to_contract_end(df["contract_until"])

    # Position: TM sub_position lookup via player_tm_id, then scraper fallback.
    if "player_tm_id" in df.columns:
        df["primary_position"] = primary_position_from_tm(
            df["player_tm_id"], settings.paths.tm_dir / "players.csv"
        )
    else:
        df["primary_position"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    df = apply_scraper_fallback(df)

    # Secondary: FM positional versatility (not for primary bucket).
    df["n_fm_positions"] = count_fm_positions(df["positions"])

    df["foot"] = encode_foot(df["foot"])

    df["fm_technical_mean"] = df[list(FM_TECHNICAL)].mean(axis=1)
    df["fm_mental_mean"] = df[list(FM_MENTAL)].mean(axis=1)
    df["fm_physical_mean"] = df[list(FM_PHYSICAL)].mean(axis=1)
    df["potential_gap"] = df["potential"] - df["current_ability"]

    required = features.all_features()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Feature matrix is missing allowlisted columns {missing}."
        )

    base_keep = [
        "player", "player_id", "season", "club", "team", "position", "positions",
        "snapshot", "snapshot_date", "actual_valuation_date",
        "market_value_eur", "wage_pw", "player_tm_id",
    ]
    keep = [c for c in base_keep if c in df.columns] + required
    seen: set[str] = set()
    out_cols = [c for c in keep if not (c in seen or seen.add(c))]
    return df.loc[:, out_cols].copy()


def _months_to_contract_end_per_row(
    contract_until: pd.Series,
    snapshot_date: pd.Series,
) -> pd.Series:
    out = []
    for cu, sd in zip(contract_until, snapshot_date, strict=True):
        try:
            ref = datetime.fromisoformat(str(sd)) if pd.notna(sd) else datetime(2025, 7, 1)
        except ValueError:
            ref = datetime(2025, 7, 1)
        out.append(months_to_contract_end(pd.Series([cu]), reference=ref).iloc[0])
    return pd.Series(out, index=contract_until.index, dtype=float)


def feature_lists_for(_target: str, features: FeatureLists) -> tuple[list[str], list[str]]:
    return list(features.shared_numeric), list(features.shared_categorical)
