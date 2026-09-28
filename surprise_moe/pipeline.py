"""Top-level SURPRISE-MoE forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from surprise_moe.config import DEFAULT_CONFIG
from surprise_moe.distribution.sampler import validate_samples
from surprise_moe.experts.fallback import numeric_fallback


@dataclass(frozen=True)
class ForecastOutput:
    """Internal SURPRISE forecast representation."""

    samples: np.ndarray
    metadata: dict[str, Any]


def forecast(
    *,
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    target_type: str = "level",
    n_draws: int | None = None,
    seed: int | None = None,
    panel_steps: np.ndarray | None = None,
) -> ForecastOutput:
    """Produce a probabilistic forecast.

    Version 0 intentionally routes all requests through the robust numerical
    fallback. Subsequent commits add experts and routing without changing
    this external interface.
    """
    draws = DEFAULT_CONFIG.n_draws if n_draws is None else int(n_draws)

    result = numeric_fallback(
        panels=panels,
        assets=assets,
        horizons=horizons,
        asof=asof,
        target_type=target_type,
        n_draws=draws,
        seed=seed,
        panel_steps=panel_steps,
    )

    samples = validate_samples(
        result.samples,
        n_draws=draws,
        n_assets=len(assets),
        n_horizons=len(horizons),
    )

    metadata = {
        **result.metadata,
        "pipeline": "surprise-moe",
        "pipeline_version": "0.1",
        "primary_path": "numeric-fallback",
    }

    return ForecastOutput(
        samples=samples,
        metadata=metadata,
    )
