"""Ingest the davidcariboo/player-scores Kaggle dump and produce a tidy per-season
market-value snapshot table joined to the existing FM/Understat master.

Pipeline (pure functions where possible, no I/O below the top-level `build_tm_target_frame`):

    players.csv + player_valuations.csv + transfers.csv + clubs.csv + competitions.csv
                                 │
                                 ▼
                 (filter non-PL clubs out via GB1 competition id)
                                 │
                                 ▼
            (pick the valuation closest to 01-Aug-season and 31-May-season)
                                 │
                                 ▼
       (expand to one row per [player, season, snapshot] with current_club at that date)
                                 │
                                 ▼
                (join to the FM/Understat master via 4-pass fuzzy join)
                                 │
                                 ▼
                        tidy frame used by data.py

No silent failures: every step raises on structural anomalies and reports counts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

PL_COMPETITION_ID: Final[str] = "GB1"

# Each (snapshot_label, month, day) picks the valuation nearest to that date within the season.
# "start" = pre-season / first window; "end" = season climax, before summer window re-opens.
SEASON_SNAPSHOTS: Final[tuple[tuple[str, int, int], ...]] = (
    ("start", 8, 1),
    ("end", 5, 31),
)

# Our master uses "2023-24"; TM's transfer_season uses "23/24". Build the mapping once.
SEASON_RANGES: Final[dict[str, tuple[datetime, datetime]]] = {
    "2023-24": (datetime(2023, 7, 1), datetime(2024, 6, 30)),
    "2024-25": (datetime(2024, 7, 1), datetime(2025, 6, 30)),
    "2025-26": (datetime(2025, 7, 1), datetime(2026, 6, 30)),
}


@dataclass(frozen=True)
class TMJoinReport:
    """Diagnostic breakdown of the 4-pass fuzzy join. Printed by the CLI, logged by tests."""

    master_rows: int
    tm_candidates: int
    matched_by_pass: dict[int, int]  # pass_number → matched rows
    unmatched_examples: list[str]

    def summary(self) -> str:
        lines = [
            f"master rows (player, season): {self.master_rows}",
            f"TM snapshot candidates:        {self.tm_candidates}",
            "matched by pass:",
            *[f"  pass {k}: {v}" for k, v in sorted(self.matched_by_pass.items())],
            f"total matched: {sum(self.matched_by_pass.values())}",
            f"unmatched: {self.master_rows - sum(self.matched_by_pass.values())}",
        ]
        if self.unmatched_examples:
            lines.append("unmatched examples:")
            lines += [f"  - {ex}" for ex in self.unmatched_examples[:10]]
        return "\n".join(lines)


_NAME_DIMINUTIVES: Final[dict[str, str]] = {
    "matty": "matthew",
    "danny": "daniel",
    "tony": "anthony",
    "chrissy": "chris",
    "steve": "stephen",
    "steven": "stephen",
    "nick": "nicholas",
    "will": "william",
    "bill": "william",
    "billy": "william",
    "benjy": "benjamin",
    "ben": "benjamin",
    "sammy": "samuel",
    "sam": "samuel",
    "alex": "alexander",
    "jake": "jacob",
    "harry": "harold",
    "charlie": "charles",
    "kez": "keziah",
}


def _normalise_name(s: str | float) -> str:
    """ASCII-fold + lowercase + strip hyphens + collapse whitespace.

    'José Mourinho' → 'jose mourinho'
    'Smith-Rowe'    → 'smith rowe'
    'Ødegaard'      → 'odegaard'
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    norm = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    norm = norm.replace("-", " ").replace("'", "").replace(".", "")
    return " ".join(norm.lower().split())


def _name_tokens(s: str) -> frozenset[str]:
    """Set of lowercase tokens from a normalised name, with diminutives canonicalised."""
    if not s:
        return frozenset()
    tokens = [t for t in s.split() if t]
    canonical = [_NAME_DIMINUTIVES.get(t, t) for t in tokens]
    return frozenset(canonical)


_SUFFIX_PAT = re.compile(
    r"\b(fc|afc|cf|sc|fk|ec|if|bk|ac|aek|sv|ca|cd|fsv|rc|rcd|usc|eu|gf|cfr|ssc|ff|vfl|vfb|tsg|cp)\b"
)
_SUFFIX_PREFIX_PAT = re.compile(r"^(fc|afc|cf|ac|sc|rc|ssc|ssd|us|usc|eu|cs|ca|cd)\s+")

_CLUB_ALIASES: Final[dict[str, str]] = {
    "manchester united": "man utd",
    "man united": "man utd",
    "manchester city": "man city",
    "newcastle united": "newcastle",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "nottingham forest": "nottm forest",
    "nottm forest": "nottm forest",
    "brighton & hove albion": "brighton",
    "brighton and hove albion": "brighton",
    "west ham united": "west ham",
    "leeds united": "leeds",
    "leicester city": "leicester",
    "ipswich town": "ipswich",
    "bournemouth": "bournemouth",
    "sheffield united": "sheffield utd",
    "sheffield wednesday": "sheffield wed",
    "paris saint-germain": "psg",
    "paris saint germain": "psg",
    "bayern munich": "bayern",
    "bayern münchen": "bayern",
    "borussia dortmund": "dortmund",
    "borussia m'gladbach": "monchengladbach",
    "borussia monchengladbach": "monchengladbach",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "ac milan": "milan",
    "atletico de madrid": "atletico madrid",
    "atletico madrid": "atletico madrid",
    "fc barcelona": "barcelona",
    "real madrid cf": "real madrid",
}


def _club_alias(name: str | float) -> str:
    """Normalise club names for fuzzy equality.

    Pipeline: ascii-fold + lowercase → strip common suffix/prefix tokens
    (FC / AFC / CF / SC / etc.) → look up curated alias table → return.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = _normalise_name(name)
    # Strip "FC"-style suffix tokens anywhere in the name (bounded by \b).
    s = _SUFFIX_PAT.sub(" ", s)
    s = _SUFFIX_PREFIX_PAT.sub("", s)
    s = " ".join(s.split())
    return _CLUB_ALIASES.get(s, s)


def load_tm_raw(tm_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the five TM tables we rely on. Raises if any is missing or empty."""
    expected = ["players", "player_valuations", "transfers", "clubs", "competitions"]
    out: dict[str, pd.DataFrame] = {}
    for name in expected:
        path = tm_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected TM table '{name}.csv' at {path}. "
                "Extract the Kaggle `davidcariboo/player-scores` zip there."
            )
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            raise ValueError(f"TM table {name}.csv is empty.")
        out[name] = df
    return out


def _parse_tm_season(s: str) -> str | None:
    """'23/24' → '2023-24', with century heuristic (<=50 → 2000s, >50 → 1900s)."""
    if not isinstance(s, str) or "/" not in s:
        return None
    a, b = s.split("/")
    try:
        ai, bi = int(a), int(b)
    except ValueError:
        return None
    yr_a = 2000 + ai if ai <= 50 else 1900 + ai
    yr_b = yr_a + 1 if bi > ai or (ai == 99 and bi == 0) else yr_a + 1
    return f"{yr_a}-{yr_b % 100:02d}"


def pl_player_ids(tm: dict[str, pd.DataFrame]) -> set[int]:
    """Players who appeared at a GB1-registered club at any snapshot in 2023-26."""
    v = tm["player_valuations"].copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    mask = (
        (v["player_club_domestic_competition_id"] == PL_COMPETITION_ID)
        & (v["date"] >= datetime(2023, 6, 1))
        & (v["date"] <= datetime(2026, 7, 1))
    )
    return set(v.loc[mask, "player_id"].dropna().astype(int).unique().tolist())


def clubs_per_player_season(tm: dict[str, pd.DataFrame]) -> dict[tuple[int, str], set[str]]:
    """From transfers.csv, map (player_id, season) → set of club_alias strings they were at.

    Starts with the player's PRE-transfer club at season start (from `from_club_name`), then
    adds every destination club across all intra-season transfers (`to_club_name`).
    """
    t = tm["transfers"].copy()
    t["season"] = t["transfer_season"].map(_parse_tm_season)
    t = t[t["season"].isin(SEASON_RANGES.keys())]
    out: dict[tuple[int, str], set[str]] = {}
    for _, row in t.iterrows():
        key = (int(row["player_id"]), row["season"])
        bucket = out.setdefault(key, set())
        for col in ("from_club_name", "to_club_name"):
            alias = _club_alias(row[col])
            if alias:
                bucket.add(alias)
    return out


def build_snapshot_frame(
    tm: dict[str, pd.DataFrame],
    *,
    pl_only: bool = True,
) -> pd.DataFrame:
    """For each (player, season, snapshot in {start, end}), pick the TM valuation nearest the
    target date **within** that season's window.

    Returns one row per snapshot with:
        player_tm_id, player_tm_name, player_tm_name_norm, dob, season, snapshot,
        snapshot_date, market_value_eur, current_club_tm, current_club_alias
    """
    v = tm["player_valuations"].copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v.dropna(subset=["date", "market_value_in_eur", "player_id"])
    if pl_only:
        pl_ids = pl_player_ids(tm)
        v = v[v["player_id"].isin(pl_ids)]

    p = tm["players"][["player_id", "name", "date_of_birth"]].drop_duplicates("player_id")
    p["date_of_birth"] = pd.to_datetime(p["date_of_birth"], errors="coerce")

    out_rows: list[dict[str, object]] = []
    for season, (season_lo, season_hi) in SEASON_RANGES.items():
        season_vals = v[(v["date"] >= season_lo) & (v["date"] <= season_hi)]
        for snap_label, month, day in SEASON_SNAPSHOTS:
            # Target date for this snapshot, clamped to within the season window.
            target_year = season_lo.year if month >= 7 else season_hi.year
            target_date = datetime(target_year, month, day)
            # For each player with any valuation in this season, pick the one nearest target.
            season_vals = season_vals.assign(
                _delta=(season_vals["date"] - pd.Timestamp(target_date)).abs()
            )
            nearest = season_vals.loc[season_vals.groupby("player_id")["_delta"].idxmin()]
            for _, r in nearest.iterrows():
                out_rows.append(
                    {
                        "player_tm_id": int(r["player_id"]),
                        "season": season,
                        "snapshot": snap_label,
                        "snapshot_date": target_date.date().isoformat(),
                        "actual_valuation_date": r["date"].date().isoformat(),
                        "market_value_eur": float(r["market_value_in_eur"]),
                        "current_club_tm": r["current_club_name"],
                        "current_club_alias": _club_alias(r["current_club_name"]),
                    }
                )

    out = pd.DataFrame(out_rows)
    out = out.merge(p, how="left", left_on="player_tm_id", right_on="player_id").drop(
        columns="player_id"
    )
    out = out.rename(columns={"name": "player_tm_name", "date_of_birth": "player_dob"})
    out["player_tm_name_norm"] = out["player_tm_name"].map(_normalise_name)
    out["player_tm_tokens"] = out["player_tm_name_norm"].map(_name_tokens)
    out["age_at_snapshot"] = (
        (pd.to_datetime(out["snapshot_date"]) - out["player_dob"]).dt.days / 365.25
    )
    return out


def join_master_to_tm(
    master: pd.DataFrame,
    snapshots: pd.DataFrame,
    tm: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, TMJoinReport]:
    """Four-pass fuzzy join. Returns (expanded_master_with_target, join_report).

    The returned frame has TWO rows per (player, season) — one per snapshot — with the TM
    `market_value_eur` and `age_at_snapshot` populated where a match was found. Unmatched rows
    are kept with NaN target (predict-only).
    """
    m = master.copy()
    m["player_norm"] = m["player"].map(_normalise_name)
    m["player_tokens"] = m["player_norm"].map(_name_tokens)
    m["club_alias"] = m["team"].map(_club_alias)

    s = snapshots.copy()
    clubs_lookup = clubs_per_player_season(tm)

    # Expand master to 2 rows per (player, season), one per snapshot.
    expanded = m.loc[m.index.repeat(len(SEASON_SNAPSHOTS))].copy().reset_index(drop=True)
    expanded["snapshot"] = [lbl for _ in range(len(m)) for lbl, _, _ in SEASON_SNAPSHOTS]

    pass_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    tm_val_col = "market_value_eur"
    tm_age_col = "age_at_snapshot"
    tm_date_col = "snapshot_date"
    tm_actual_date_col = "actual_valuation_date"
    tm_id_col = "player_tm_id"
    tm_pass_col = "join_pass"
    tm_club_col = "current_club_tm"

    expanded[tm_val_col] = np.nan
    expanded[tm_age_col] = np.nan
    expanded[tm_date_col] = pd.NA
    expanded[tm_actual_date_col] = pd.NA
    expanded[tm_id_col] = pd.NA
    expanded[tm_club_col] = pd.NA
    expanded[tm_pass_col] = pd.NA

    # Pre-index the snapshot frame for fast lookups.
    snap_by_name = s.set_index(["player_tm_name_norm", "season", "snapshot"]).sort_index()
    # For pass 4: snapshots grouped by (season, snapshot) for token-subset search.
    snaps_by_cohort = {
        (season, snap): grp
        for (season, snap), grp in s.groupby(["season", "snapshot"])
    }

    def _subset_match(master_tokens: frozenset[str], cohort: pd.DataFrame) -> pd.DataFrame:
        """Match if master and TM token sets share ≥ 2 tokens and neither has >1 token
        the other lacks. Catches 'Ezri Konsa' ↔ 'Ezri Konsa Ngoyo' both directions,
        and 'Matty Cash' ↔ 'Matthew Cash' via the diminutive canonicalisation."""
        if not master_tokens:
            return cohort.iloc[0:0]

        def ok(tm_tokens: frozenset[str]) -> bool:
            if not tm_tokens:
                return False
            inter = master_tokens & tm_tokens
            if len(inter) < min(2, len(master_tokens), len(tm_tokens)):
                return False
            # Allow one side to have one extra token (middle names); reject bigger drifts.
            extra_master = master_tokens - tm_tokens
            extra_tm = tm_tokens - master_tokens
            return len(extra_master) <= 1 and len(extra_tm) <= 1

        return cohort[cohort["player_tm_tokens"].map(ok)]

    for i, row in expanded.iterrows():
        season, snap = row["season"], row["snapshot"]
        name_norm = row["player_norm"]
        key = (name_norm, season, snap)
        try:
            candidates = snap_by_name.loc[[key]].reset_index()
        except KeyError:
            candidates = pd.DataFrame(columns=snap_by_name.reset_index().columns)

        matched: pd.Series | None = None
        matched_pass: int | None = None

        # Pass 1: exact normalised-name + exact club alias.
        if len(candidates):
            p1 = candidates[candidates["current_club_alias"] == row["club_alias"]]
            if len(p1) == 1:
                matched = p1.iloc[0]
                matched_pass = 1

        # Pass 2: name match + master's club ∈ TM's known clubs for this player this season.
        if matched is None and len(candidates):
            def clubs_for_tm_id(tm_id: int) -> set[str]:
                return clubs_lookup.get((int(tm_id), season), set())
            p2 = candidates[
                candidates.apply(
                    lambda c: row["club_alias"] in clubs_for_tm_id(c["player_tm_id"]),
                    axis=1,
                )
            ]
            if len(p2) == 1:
                matched = p2.iloc[0]
                matched_pass = 2

        # Pass 3: unique name+season+snapshot with age ±1.
        if matched is None and len(candidates) == 1:
            cand_row = candidates.iloc[0]
            master_age = float(row["age"]) if pd.notna(row["age"]) else np.nan
            tm_age = (
                float(cand_row["age_at_snapshot"]) if pd.notna(cand_row["age_at_snapshot"]) else np.nan
            )
            if np.isnan(master_age) or np.isnan(tm_age) or abs(master_age - tm_age) <= 1.0:
                matched = cand_row
                matched_pass = 3

        # Pass 4: token-subset match within (season, snapshot) cohort + age ±1.
        # Catches partial-name cases ("Ezri Konsa" ⊂ "Ezri Konsa Ngoyo") and hyphenated/
        # diminutive variants after normalisation+diminutive canonicalisation.
        if matched is None:
            cohort = snaps_by_cohort.get((season, snap))
            if cohort is not None:
                p4 = _subset_match(row["player_tokens"], cohort)
                if len(p4) >= 1:
                    master_age = float(row["age"]) if pd.notna(row["age"]) else np.nan
                    if not np.isnan(master_age):
                        p4 = p4.loc[
                            (p4["age_at_snapshot"].sub(master_age).abs() <= 1.5)
                            | p4["age_at_snapshot"].isna()
                        ]
                    if len(p4) == 1:
                        matched = p4.iloc[0]
                        matched_pass = 4

        if matched is None or matched_pass is None:
            continue
        pass_counts[matched_pass] += 1
        expanded.at[i, tm_pass_col] = matched_pass

        expanded.at[i, tm_val_col] = float(matched["market_value_eur"])
        expanded.at[i, tm_age_col] = float(matched["age_at_snapshot"])
        expanded.at[i, tm_date_col] = str(matched["snapshot_date"])
        expanded.at[i, tm_actual_date_col] = str(matched["actual_valuation_date"])
        expanded.at[i, tm_id_col] = int(matched["player_tm_id"])
        expanded.at[i, tm_club_col] = matched["current_club_tm"]

    # Diagnostic report.
    matched_mask = expanded[tm_pass_col].notna()
    unmatched = expanded.loc[~matched_mask, ["player", "season", "snapshot", "team"]]
    examples = [
        f"{r.player} ({r.season}, {r.snapshot}, master team={r.team})"
        for r in unmatched.drop_duplicates(subset=["player", "season"]).head(20).itertuples()
    ]
    report = TMJoinReport(
        master_rows=len(expanded),
        tm_candidates=len(s),
        matched_by_pass=pass_counts,
        unmatched_examples=examples,
    )
    expanded = expanded.drop(
        columns=["player_norm", "club_alias", "player_tokens"], errors="ignore"
    )
    return expanded, report


def build_tm_target_frame(
    master: pd.DataFrame,
    tm_dir: Path,
) -> tuple[pd.DataFrame, TMJoinReport]:
    """Top-level entry: load TM, build snapshots, join to master, return expanded frame."""
    tm = load_tm_raw(tm_dir)
    snapshots = build_snapshot_frame(tm, pl_only=True)
    return join_master_to_tm(master, snapshots, tm)
