"""Helpers for loading the already split TEP parquet datasets."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class TEPDataset:
    """Container for a single TEP split."""

    features: pd.DataFrame
    labels: pd.Series | None = None
    fault_class: pd.Series | None = None
    name: str = ""


@dataclass(slots=True)
class TEPSplits:
    """The four canonical TEP parquet splits.

    For unsupervised training, use ``train_fault_free``.
    For evaluation, combine the testing splits into one labeled frame.
    """

    train_fault_free: TEPDataset
    train_faulty: TEPDataset
    test_fault_free: TEPDataset
    test_faulty: TEPDataset

    def training_dataset(self) -> TEPDataset:
        """Return the clean training split for unsupervised learning."""

        return self.train_fault_free

    def evaluation_dataset(self) -> TEPDataset:
        """Return the labeled test split with normal and faulty rows stacked together."""

        normal_features = self.test_fault_free.features.copy()
        faulty_features = self.test_faulty.features.copy()

        features = pd.concat([normal_features, faulty_features], ignore_index=True)
        labels = pd.Series(
            [0] * len(normal_features) + [1] * len(faulty_features),
            name="is_fault",
            dtype="int64",
        )

        return TEPDataset(features=features, labels=labels, name="tep_test")


def load_tep_dataset(
    path: str | Path,
    *,
    label_column: str = "is_fault",
    fault_column: str = "fault_class",
) -> TEPDataset:
    """Load a parquet file and split it into features and labels."""

    data_frame = pd.read_parquet(Path(path))
    data_frame.columns = data_frame.columns.str.lower()

    labels = data_frame[label_column] if label_column in data_frame.columns else None
    fault_class = data_frame[fault_column] if fault_column in data_frame.columns else None
    feature_columns = [
        column
        for column in data_frame.columns
        if column not in {label_column, fault_column}
    ]

    return TEPDataset(
        features=data_frame[feature_columns],
        labels=labels,
        fault_class=fault_class,
        name=Path(path).stem,
    )


def load_tep_splits(base_path: str | Path) -> TEPSplits:
    """Load the four pre-split TEP parquet files from ``base_path``."""

    base_dir = Path(base_path)

    return TEPSplits(
        train_fault_free=load_tep_dataset(base_dir / "TEP_FaultFree_Training.parquet"),
        train_faulty=load_tep_dataset(base_dir / "TEP_Faulty_Training.parquet"),
        test_fault_free=load_tep_dataset(base_dir / "TEP_FaultFree_Testing.parquet"),
        test_faulty=load_tep_dataset(base_dir / "TEP_Faulty_Testing.parquet"),
    )