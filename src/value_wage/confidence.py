"""Prediction confidence from backtested typical error — no intervals, no MAPIE.

The old confidence signal came from the width of a conformal prediction interval. Those
intervals were near-homoscedastic (a roughly constant multiplicative band for every
player), so the width carried almost no per-player information. This module replaces it
with a signal that genuinely varies and is trivial to explain:

    confidence = the model's historical median |% error| for players in the same
                 positional AREA and SIZE TIER (measured on the holdout).

A scout reads it as "for players like this, the model is usually within X%." The breaks
are calibrated per area, because some areas (GK, attack) are inherently harder to pin than
others and should not be held to the same absolute bar. Both the size tiers and the
thresholds are target-specific: the value target buckets by market value (euros) and the
wage target by weekly wage, and wage is predicted far more accurately so its bars are
tighter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# The 8 sub-position buckets collapse into 4 scouting areas.
AREA_OF_POSITION: dict[str, str] = {
    "GK": "GK",
    "CB": "DEF", "FB": "DEF",
    "DM": "MID", "CM": "MID", "AM": "MID",
    "W": "ATT", "ST": "ATT",
}
AREAS: tuple[str, ...] = ("GK", "DEF", "MID", "ATT")

# Per-target config. `bounds` are the three cut points (in the target's own units) that
# define four size tiers; `thresholds` are the per-area (High<, Low>) cut points on typical
# % error. All calibrated on the 2025-26 holdout.
#  - value: market value in EUR. Sub-€5M players are hard to value in % terms (~60%),
#    mid-tier (€5-40M) best (~20%), €40M+ moderate (~30%).
#  - wage: weekly wage in EUR. Low earners (<€20k/wk) are noisy (~50%+), high earners
#    (€100k+/wk) very accurate (~7%); overall error is far lower than value.
TIER_CONFIG: dict[str, dict[str, Any]] = {
    "value": {
        "bounds": (5_000_000.0, 15_000_000.0, 40_000_000.0),
        "labels": ("<5M", "5-15M", "15-40M", "40M+"),
        "thresholds": {
            "GK": (30.0, 36.0),
            "DEF": (25.0, 40.0),
            "MID": (25.0, 40.0),
            "ATT": (30.0, 45.0),
        },
    },
    "wage": {
        "bounds": (20_000.0, 50_000.0, 100_000.0),
        "labels": ("<20k", "20-50k", "50-100k", "100k+"),
        "thresholds": {
            "GK": (16.0, 30.0),
            "DEF": (15.0, 30.0),
            "MID": (15.0, 30.0),
            "ATT": (18.0, 35.0),
        },
    },
}

# Below this many players, an area×tier cell is too thin to trust and falls back to the
# size-tier median pooled across all areas.
MIN_CELL_N: int = 10


def size_tier(value: float | None, target: str) -> str | None:
    """Bucket a value (euros for `value`, €/week for `wage`) into its tier label.

    Used on the *predicted* value at display time so even players without an observed
    market value / wage still get a confidence.
    """
    if value is None or not np.isfinite(value):
        return None
    labels = TIER_CONFIG[target]["labels"]
    b1, b2, b3 = TIER_CONFIG[target]["bounds"]
    if value < b1:
        return labels[0]
    if value < b2:
        return labels[1]
    if value < b3:
        return labels[2]
    return labels[3]


def area_of(position: str | None) -> str | None:
    if not position:
        return None
    return AREA_OF_POSITION.get(str(position).strip().upper())


def build_confidence_table(rows: pd.DataFrame, target: str) -> dict[str, Any]:
    """Compute the area×tier typical-error lookup for one target from scored holdout rows.

    `rows` needs columns: ``area`` (GK/DEF/MID/ATT), ``actual`` and ``pred`` in the
    target's units. Returns a JSON-serialisable dict with the lookup, the per-tier
    fallbacks, and all the constants the frontend needs to classify a live player.
    """
    labels = TIER_CONFIG[target]["labels"]
    df = rows.copy()
    df = df[(df["actual"] > 0) & df["area"].isin(AREAS)]
    df["ape"] = (df["pred"] - df["actual"]).abs() / df["actual"] * 100.0
    df["tier"] = df["actual"].apply(lambda v: size_tier(v, target))
    df = df[df["tier"].notna()]

    tier_fallback = {
        ti: (float(df.loc[df["tier"] == ti, "ape"].median()) if (df["tier"] == ti).any() else None)
        for ti in labels
    }
    lookup: dict[str, dict[str, float | None]] = {}
    for ar in AREAS:
        lookup[ar] = {}
        for ti in labels:
            cell = df[(df["area"] == ar) & (df["tier"] == ti)]["ape"]
            lookup[ar][ti] = float(cell.median()) if len(cell) >= MIN_CELL_N else tier_fallback[ti]

    return {
        "tier_bounds": list(TIER_CONFIG[target]["bounds"]),
        "tier_labels": list(labels),
        "area_of_position": AREA_OF_POSITION,
        "area_thresholds": {k: list(v) for k, v in TIER_CONFIG[target]["thresholds"].items()},
        "lookup": lookup,
        "tier_fallback": tier_fallback,
        "n_scored": len(df),
    }
