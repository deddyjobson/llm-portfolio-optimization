"""Data loading utilities for processed datasets."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_dataset(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load processed dataset from .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        Tuple of (price_relatives, log_relatives) arrays of shape (T, N).
    """
    path = Path(path)
    data = np.load(path)

    # Support both new (X, R) and legacy (price_relatives, log_relatives) keys
    if "X" in data:
        price_relatives = data["X"]
        log_relatives = data["R"]
    else:
        price_relatives = data["price_relatives"]
        log_relatives = data["log_relatives"]

    return price_relatives, log_relatives


def load_dataset_with_metadata(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load processed dataset with asset names from .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        Tuple of (price_relatives, log_relatives, asset_names) arrays.
        asset_names may be None if not present in the file.
    """
    path = Path(path)
    data = np.load(path)

    # Support both new (X, R) and legacy (price_relatives, log_relatives) keys
    if "X" in data:
        price_relatives = data["X"]
        log_relatives = data["R"]
    else:
        price_relatives = data["price_relatives"]
        log_relatives = data["log_relatives"]

    asset_names = data.get("asset_names", None)

    return price_relatives, log_relatives, asset_names


def make_baby_dataset(
    input_path: str | Path,
    output_path: str | Path,
    n_periods: int = 220,
    n_assets: int = 6,
) -> Path:
    """Create a smaller 'baby' dataset for quick testing.

    Args:
        input_path: Path to the full dataset .npz file.
        output_path: Path to save the baby dataset.
        n_periods: Number of time periods to include.
        n_assets: Number of assets to include.

    Returns:
        Path to the created baby dataset.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    price_relatives, log_relatives, asset_names = load_dataset_with_metadata(input_path)

    # Take first n_periods and first n_assets
    T, N = price_relatives.shape
    n_periods = min(n_periods, T)
    n_assets = min(n_assets, N)

    price_relatives_baby = price_relatives[:n_periods, :n_assets]
    log_relatives_baby = log_relatives[:n_periods, :n_assets]

    # Subset asset names if available
    asset_names_baby = None
    if asset_names is not None:
        asset_names_baby = asset_names[:n_assets]

    # Save with both new and legacy keys
    save_kwargs = {
        "X": price_relatives_baby,
        "R": log_relatives_baby,
        "price_relatives": price_relatives_baby,
        "log_relatives": log_relatives_baby,
    }
    if asset_names_baby is not None:
        save_kwargs["asset_names"] = asset_names_baby

    np.savez(output_path, **save_kwargs)

    return output_path


def split_dataset(
    data: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train/val/test sets.

    Args:
        data: Data array of shape (T, N).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.

    Returns:
        Tuple of (train, val, test) arrays.
    """
    T = len(data)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test
