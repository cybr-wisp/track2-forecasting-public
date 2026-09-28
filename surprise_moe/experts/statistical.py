"""Robust numeric statistical expert for SURPRISE-MoE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from surprise_moe.config import DEFAULT_CONFIG


def differences_without_large_gaps(series: pd.Series) -> pd.Series:
    """Compute differences while avoiding jumps across large missing-history gaps."""
    if len(series) < 2:
        return pd.Series(dtype=float)

    values = series.astype(float)
    diffs = values.diff()

    dates = pd.DatetimeIndex(series.index)
    gaps = pd.Series(dates, index=dates).diff().dt.days
    finite_gaps = gaps.dropna()

    if finite_gaps.empty:
        return diffs.dropna()

    threshold = max(float(finite_gaps.median()) * 10.0, 5.0)

    return diffs.loc[(gaps <= threshold).fillna(False)].dropna()


def estimate_step_statistics(series: pd.Series) -> tuple[float, float, float]:
    """Estimate anchor, drift and volatility from cutoff-safe history."""
    if series.empty:
        return 0.0, 0.0, 1.0

    steps = differences_without_large_gaps(series)
    last = float(series.iloc[-1])

    if steps.empty:
        return (
            last,
            0.0,
            max(abs(last) * 1e-4, DEFAULT_CONFIG.volatility_floor),
        )

    drift = float(steps.mean())
    volatility = float(steps.std(ddof=1))

    if not np.isfinite(volatility):
        volatility = 0.0

    volatility = max(
        volatility,
        abs(last) * 1e-4,
        DEFAULT_CONFIG.volatility_floor,
    )

    return last, drift, volatility
