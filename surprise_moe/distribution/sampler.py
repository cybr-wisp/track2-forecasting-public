"""Validation utilities for probabilistic forecast samples."""

from __future__ import annotations

import numpy as np


class InvalidForecastSamples(ValueError):
    """Raised when forecast samples violate the Track 2 contract."""


def validate_samples(
    samples: np.ndarray,
    *,
    n_draws: int,
    n_assets: int,
    n_horizons: int,
) -> np.ndarray:
    """Validate and return a Track 2 Monte Carlo sample tensor.

    Expected axis order:

        [draw, asset, horizon]
    """
    samples = np.asarray(samples, dtype=np.float64)

    expected_shape = (n_draws, n_assets, n_horizons)

    if samples.shape != expected_shape:
        raise InvalidForecastSamples(
            f"Expected forecast shape {expected_shape}, got {samples.shape}."
        )

    if n_draws < 200:
        raise InvalidForecastSamples(
            f"At least 200 draws are required; received {n_draws}."
        )

    if not np.isfinite(samples).all():
        raise InvalidForecastSamples(
            "Forecast samples contain NaN or infinite values."
        )

    # Every cell should represent a distribution rather than a single
    # repeated point estimate.
    spread = np.std(samples, axis=0)

    if np.any(spread <= 0):
        bad = np.argwhere(spread <= 0)
        raise InvalidForecastSamples(
            f"Degenerate predictive distribution in {len(bad)} cell(s)."
        )

    return samples
