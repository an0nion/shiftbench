"""Dataset loading and registry management for ShiftBench.

This module provides utilities to load datasets from the registry,
including features, labels, cohorts, and train/cal/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DatasetRegistry:
    """Manages the dataset registry (data/registry.json)."""

    def __init__(self, registry_path: Optional[Path] = None):
        """Initialize registry from JSON file.

        Args:
            registry_path: Path to registry.json. If None, uses default location.
        """
        if registry_path is None:
            # Default to shift-bench/data/registry.json
            registry_path = Path(__file__).parent.parent.parent / "data" / "registry.json"

        self.registry_path = Path(registry_path)
        self._registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from JSON."""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Registry not found at {self.registry_path}. "
                "Run dataset preparation scripts first."
            )

        with open(self.registry_path, "r") as f:
            return json.load(f)

    def list_datasets(self, domain: Optional[str] = None) -> List[str]:
        """List all dataset names in the registry.

        Args:
            domain: Optional filter by domain ("molecular", "text", "tabular")

        Returns:
            List of dataset names
        """
        datasets = self._registry["datasets"]
        if domain:
            datasets = [d for d in datasets if d["domain"] == domain]
        return [d["name"] for d in datasets]

    def get_dataset_info(self, name: str) -> Dict:
        """Get metadata for a specific dataset.

        Args:
            name: Dataset name (e.g., "bace", "imdb")

        Returns:
            Dictionary with dataset metadata

        Raises:
            KeyError: If dataset not found
        """
        for dataset in self._registry["datasets"]:
            if dataset["name"] == name:
                return dataset

        raise KeyError(f"Dataset '{name}' not found in registry")

    def get_domains(self) -> List[str]:
        """Get list of all domains."""
        return list(set(d["domain"] for d in self._registry["datasets"]))

    def get_metadata(self) -> Dict:
        """Get registry-level metadata."""
        return self._registry.get("metadata", {})


def load_dataset(
    name: str,
    data_dir: Optional[Path] = None,
    return_splits: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """Load a dataset from ShiftBench.

    Args:
        name: Dataset name from registry (e.g., "bace", "imdb")
        data_dir: Directory containing processed dataset files. If None, uses default.
        return_splits: Whether to return split assignments

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (n_samples,)
        cohort_ids: Cohort identifiers (n_samples,)
        splits: DataFrame with columns [uid, split] if return_splits=True, else None

    Raises:
        FileNotFoundError: If dataset files not found
        KeyError: If dataset not in registry

    Example:
        >>> X, y, cohorts, splits = load_dataset("bace")
        >>> cal_mask = (splits["split"] == "cal").values
        >>> X_cal, y_cal = X[cal_mask], y[cal_mask]
    """
    registry = DatasetRegistry()
    info = registry.get_dataset_info(name)

    if data_dir is None:
        # Default to shift-bench/data/processed/<name>/
        data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / name

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            f"Run preprocessing for {name} first."
        )

    # Load features, labels, cohorts
    X = np.load(data_dir / "features.npy")
    y = np.load(data_dir / "labels.npy")
    cohorts = np.load(data_dir / "cohorts.npy", allow_pickle=True)

    splits_df = None
    if return_splits:
        splits_path = data_dir / "splits.csv"
        if splits_path.exists():
            splits_df = pd.read_csv(splits_path)
        else:
            raise FileNotFoundError(f"Splits file not found: {splits_path}")

    return X, y, cohorts, splits_df


def get_registry() -> DatasetRegistry:
    """Get the default dataset registry.

    Returns:
        DatasetRegistry instance

    Example:
        >>> registry = get_registry()
        >>> print(registry.list_datasets(domain="molecular"))
        ['bace', 'bbbp', 'clintox', ...]
    """
    return DatasetRegistry()
