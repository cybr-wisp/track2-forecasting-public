import numpy as np
import pandas as pd

from surprise_moe.experts.fallback import numeric_fallback


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    rows = []

    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "asset": "A",
                "value": 100.0 + i * 0.1,
            }
        )
        rows.append(
            {
                "date": date,
                "asset": "B",
                "value": 80.0 + i * 0.08,
            }
        )

    return pd.DataFrame(rows)


def test_numeric_fallback_returns_joint_samples():
    panels = {"synthetic": _panel()}

    result = numeric_fallback(
        panels=panels,
        assets=["A", "B"],
        horizons=[1, 5],
        asof="2024-05-17",
        n_draws=500,
        seed=42,
    )

    assert result.samples.shape == (500, 2, 2)
    assert np.isfinite(result.samples).all()
    assert result.metadata["fallback"] is True


def test_numeric_fallback_is_reproducible():
    panels = {"synthetic": _panel()}

    first = numeric_fallback(
        panels=panels,
        assets=["A", "B"],
        horizons=[1],
        asof="2024-05-17",
        seed=7,
    )

    second = numeric_fallback(
        panels=panels,
        assets=["A", "B"],
        horizons=[1],
        asof="2024-05-17",
        seed=7,
    )

    np.testing.assert_array_equal(
        first.samples,
        second.samples,
    )
