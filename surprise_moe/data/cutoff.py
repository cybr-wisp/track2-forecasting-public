"""Leakage guards used throughout SURPRISE-MoE."""

from __future__ import annotations

import pandas as pd


class CutoffViolation(RuntimeError):
    """Raised when information after the forecast origin is observed."""


def enforce_cutoff(frame: pd.DataFrame, asof: str, date_column: str = "date") -> pd.DataFrame:
    """Return rows available at or before ``asof`` and reject malformed dates."""
    if date_column not in frame.columns:
        raise ValueError(f"Missing required date column: {date_column}")

    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")

    if result[date_column].isna().any():
        raise CutoffViolation("Panel contains invalid timestamps.")

    cutoff = pd.Timestamp(asof)
    result = result.loc[result[date_column] <= cutoff].copy()

    if not result.empty and result[date_column].max() > cutoff:
        raise CutoffViolation("Future observation crossed the as-of boundary.")

    return result
