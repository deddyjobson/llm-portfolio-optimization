"""Configuration loading and dataclasses for portfolio benchmark settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tomli


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    path: str = "data/processed/djia_baby.npz"
    lookback: int = 10


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    transaction_cost: float = 0.001
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str = "qwen2.5:1.5b-instruct"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    multiline: bool = False


@dataclass
class BootstrapConfig:
    """Bootstrap confidence interval configuration."""

    enabled: bool = False
    n_bootstrap: int = 1000
    block_size: Optional[int] = None  # None = auto (sqrt(n))
    confidence_level: float = 0.95
    seed: Optional[int] = 42


@dataclass
class RollingConfig:
    """Rolling window evaluation configuration."""

    enabled: bool = False
    train_size: int = 100
    val_size: int = 30
    test_size: int = 50
    step_size: Optional[int] = None  # Defaults to test_size (non-overlapping)
    confidence_level: float = 0.95


@dataclass
class DROConfig:
    """Wasserstein DRO configuration."""

    enabled: bool = True
    eta: float = 0.05
    epsilon: float = 0.1
    rho: float = 1.0
    support_radius: Optional[float] = None
    solver_method: str = "conic"
    solver: str = "ECOS"


@dataclass
class Config:
    """Main configuration container."""

    dataset: DatasetConfig
    backtest: BacktestConfig
    llm: LLMConfig
    bootstrap: BootstrapConfig
    rolling: RollingConfig
    dro: DROConfig


def load_config(path: str | Path) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Parsed Config object.
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = tomli.load(f)

    dataset_cfg = DatasetConfig(**data.get("dataset", {}))
    backtest_cfg = BacktestConfig(**data.get("backtest", {}))
    llm_cfg = LLMConfig(**data.get("llm", {}))
    bootstrap_cfg = BootstrapConfig(**data.get("bootstrap", {}))
    rolling_cfg = RollingConfig(**data.get("rolling", {}))
    dro_data = data.get("dro", {})
    if "eta" not in dro_data and "alpha" in dro_data:
        dro_data["eta"] = 1.0 - float(dro_data["alpha"])
    if "epsilon" not in dro_data and "epsilon_marginal" in dro_data:
        dro_data["epsilon"] = dro_data["epsilon_marginal"]
    if "rho" not in dro_data and "risk_weight" in dro_data:
        dro_data["rho"] = dro_data["risk_weight"]
    dro_cfg = DROConfig(**dro_data)

    return Config(
        dataset=dataset_cfg,
        backtest=backtest_cfg,
        llm=llm_cfg,
        bootstrap=bootstrap_cfg,
        rolling=rolling_cfg,
        dro=dro_cfg,
    )
