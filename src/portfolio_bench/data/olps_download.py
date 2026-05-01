"""Download OLPS dataset from GitHub repository."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Dataset name mappings for OLPS files that don't match expected names
DATASET_NAME_MAP = {
    "nyse": "nyse-o",      # Default to nyse-o (larger, 36 assets)
    "nyse-n": "nyse-n",    # Nobel subset (23 assets)
    "nyse-o": "nyse-o",    # Original subset (36 assets)
}


def download_olps_data(dataset: str = "djia", output_dir: str = "data") -> Path:
    """Download and extract dataset from OLPS repository.

    Args:
        dataset: Name of dataset to download (djia, msci, nyse, sp500, tse).
        output_dir: Base output directory for data files.

    Returns:
        Path to the processed .npz file.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    external_dir = Path("references") / "external"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    # Clone OLPS repo to references/external/ (persistent, gitignored)
    repo_dir = external_dir / "OLPS"
    repo_url = "https://github.com/OLPS/OLPS.git"

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            capture_output=True,
        )

    # Find and copy the .mat file (apply name mapping for nyse variants)
    mat_filename = DATASET_NAME_MAP.get(dataset, dataset)
    mat_file = repo_dir / "Data" / f"{mat_filename}.mat"
    if not mat_file.exists():
        # Try lowercase
        mat_file = repo_dir / "Data" / f"{mat_filename.lower()}.mat"
    if not mat_file.exists():
        raise FileNotFoundError(f"Dataset {dataset} not found in OLPS repo")

    dest_mat = raw_dir / f"{dataset}.mat"
    shutil.copy(mat_file, dest_mat)

    # Process the .mat file - output to djia_full.npz
    output_path = process_mat_file(dest_mat, processed_dir / f"{dataset}_full.npz")
    return output_path


def process_mat_file(mat_path: Path, output_path: Path) -> Path:
    """Process .mat file to extract price relatives and log-relatives.

    Args:
        mat_path: Path to input .mat file.
        output_path: Path to output .npz file.

    Returns:
        Path to the created .npz file.
    """
    # Load .mat file
    mat_data = loadmat(str(mat_path))

    # Find the appropriate data variable
    # Skip metadata keys starting with __
    # Select first numeric 2D matrix with T > N, N >= 2
    data_key = None
    price_relatives = None

    for key in mat_data.keys():
        if key.startswith("__"):
            continue
        arr = mat_data[key]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2:
            continue
        T, N = arr.shape
        # Check: T > N, N >= 2, all values numeric
        if T > N and N >= 2 and np.issubdtype(arr.dtype, np.number):
            data_key = key
            price_relatives = arr.astype(np.float64)
            break

    if price_relatives is None:
        raise ValueError(f"No suitable data matrix found in {mat_path}")

    T, N = price_relatives.shape

    # Validate X > 0 (price relatives must be positive)
    if not np.all(price_relatives > 0):
        raise ValueError(f"Price relatives must be positive, found non-positive values in {mat_path}")

    # Compute log-relatives (log returns)
    log_relatives = np.log(price_relatives)

    # Generate asset names (Asset_1, Asset_2, etc.)
    asset_names = np.array([f"Asset_{i+1}" for i in range(N)])

    # Compute file hash for metadata
    with open(mat_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Save as .npz with X, R, asset_names
    np.savez(
        output_path,
        X=price_relatives,
        R=log_relatives,
        asset_names=asset_names,
        # Also save with legacy names for backward compatibility
        price_relatives=price_relatives,
        log_relatives=log_relatives,
    )

    # Create metadata.json
    metadata = {
        "source": "https://github.com/OLPS/OLPS",
        "source_file": mat_path.name,
        "source_key": data_key,
        "sha256": file_hash,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "shape": {"T": T, "N": N},
        "asset_names": asset_names.tolist(),
    }

    metadata_path = output_path.parent / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path
