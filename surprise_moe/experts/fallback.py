"""Competition-safe numeric fallback for SURPRISE-MoE.

The fallback deliberately reuses the organizer reference sampler so that
failure of a higher-level SURPRISE component cannot break target semantics,
cross-asset dependence, or the required Monte Carlo shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.cli import _draw

from surprise_moe.config import DEFAULT_CONFIG
from surprise_moe.distribution.sampler import validate_samples


@dataclass(frozen=True)
class FallbackResult:
    """Output of the numeric emergency forecaster."""

    samples: np.ndarray
    metadata: dict[str, Any]


def numeric_fallback(
    *,
    panels: dict[str, pd.DataFrame],
    assets: list[str],
    horizons: list[int],
    asof: str,
    target_type: str = "level",
    n_draws: int | None = None,
    seed: int | None = None,
    panel_steps: np.ndarray | None = None,
) -> FallbackResult:
    """Generate a robust joint forecast using the reference statistical floor.

    This remains available even when text reasoning, learned experts,
    routing, or calibration fail.
    """
    draws = DEFAULT_CONFIG.n_draws if n_draws is None else int(n_draws)
    random_seed = DEFAULT_CONFIG.seed if seed is None else int(seed)

    samples, stats = _draw(
        panels=panels,
        assets=assets,
        horizons=horizons,
        asof=asof,
        n_draws=draws,
        seed=random_seed,
        target_type=target_type,
        panel_steps=panel_steps,
    )

    samples = validate_samples(
        samples,
        n_draws=draws,
        n_assets=len(assets),
        n_horizons=len(horizons),
    )

    metadata = {
        "model": "reference-joint-statistical-fallback",
        "fallback": True,
        "seed": random_seed,
        "n_draws": draws,
        "target_type": target_type,
        "statistics": stats,
    }

    return FallbackResult(
        samples=samples,
        metadata=metadata,
    )
