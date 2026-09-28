import numpy as np
import pandas as pd

from surprise_moe.pipeline import forecast


def test_pipeline_produces_valid_forecast():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")

    panel = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "asset": ["A"] * len(dates) + ["B"] * len(dates),
            "value": (
                list(np.linspace(100, 110, len(dates)))
                + list(np.linspace(80, 86, len(dates)))
            ),
        }
    )

    result = forecast(
        panels={"synthetic": panel},
        assets=["A", "B"],
        horizons=[1, 5],
        asof="2024-05-17",
        n_draws=500,
        seed=42,
    )

    assert result.samples.shape == (500, 2, 2)
    assert np.isfinite(result.samples).all()

    assert result.metadata["pipeline"] == "surprise-moe"
    assert result.metadata["primary_path"] == "numeric-fallback"
