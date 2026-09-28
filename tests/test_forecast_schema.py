import numpy as np
import pytest

from surprise_moe.distribution.sampler import (
    InvalidForecastSamples,
    validate_samples,
)


def test_valid_joint_draw_tensor_passes():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(500, 4, 2))

    result = validate_samples(
        samples,
        n_draws=500,
        n_assets=4,
        n_horizons=2,
    )

    assert result.shape == (500, 4, 2)


def test_wrong_axis_shape_is_rejected():
    samples = np.ones((4, 500, 2))

    with pytest.raises(InvalidForecastSamples, match="Expected forecast shape"):
        validate_samples(
            samples,
            n_draws=500,
            n_assets=4,
            n_horizons=2,
        )


def test_nonfinite_samples_are_rejected():
    samples = np.random.default_rng(42).normal(size=(500, 2, 1))
    samples[0, 0, 0] = np.nan

    with pytest.raises(InvalidForecastSamples, match="NaN or infinite"):
        validate_samples(
            samples,
            n_draws=500,
            n_assets=2,
            n_horizons=1,
        )


def test_too_few_draws_are_rejected():
    samples = np.random.default_rng(42).normal(size=(100, 2, 1))

    with pytest.raises(InvalidForecastSamples, match="At least 200"):
        validate_samples(
            samples,
            n_draws=100,
            n_assets=2,
            n_horizons=1,
        )


def test_degenerate_distribution_is_rejected():
    samples = np.ones((500, 2, 1))

    with pytest.raises(InvalidForecastSamples, match="Degenerate"):
        validate_samples(
            samples,
            n_draws=500,
            n_assets=2,
            n_horizons=1,
        )
