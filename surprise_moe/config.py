"""Core configuration for SURPRISE-MoE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SurpriseConfig:
    """Competition-safe defaults for SURPRISE-MoE."""

    n_draws: int = 500
    seed: int = 42

    # Numerical robustness
    min_history: int = 30
    volatility_floor: float = 1e-6
    covariance_epsilon: float = 1e-8

    # Fallback behaviour
    enable_fallback: bool = True


DEFAULT_CONFIG = SurpriseConfig()
