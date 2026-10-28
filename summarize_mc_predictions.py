"""Summarize repeated stochastic localization predictions at image level."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = [
    "Cytoplasm",
    "Endoplasmic_reticulum",
    "Mitochondria",
    "Nucleus",
    "Plasma_membrane",
]
Z95 = 1.959963984540054


def load_inputs(folder: Path, expected_passes: int | None = None):
    folder = Path(folder)
    metadata_path = folder / "metadata.csv"
    predictions_path = folder / "mc_predictions.npy"
    if not metadata_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Expected metadata.csv and mc_predictions.npy in {folder}")

    metadata = pd.read_csv(metadata_path, dtype={"Protein Id": str})
    required = {"Protein Name", "Protein Id"}
    if metadata.empty or not required.issubset(metadata.columns):
        raise ValueError(f"metadata.csv must contain {sorted(required)}")
    if metadata[["Protein Name", "Protein Id"]].isna().any().any():
        raise ValueError("Protein Name and Protein Id must not contain missing values")

    predictions = np.load(predictions_path, mmap_mode="r")
    if predictions.ndim != 3 or predictions.shape[2] != len(CATEGORIES):
        raise ValueError("mc_predictions.npy must have shape [R, N, 5]")
    if predictions.shape[1] != len(metadata):
        raise ValueError("Prediction sample dimension does not match metadata row count")
    if expected_passes is not None and predictions.shape[0] != expected_passes:
        raise ValueError(f"Expected {expected_passes} stochastic passes, found {predictions.shape[0]}")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    return metadata, predictions


def summarize_folder(folder: Path, output_csv: Path, expected_passes: int | None = None) -> pd.DataFrame:
    metadata, predictions = load_inputs(folder, expected_passes)
    r = predictions.shape[0]
    mean = np.asarray(predictions.mean(axis=0), dtype=float)
    sd = np.asarray(predictions.std(axis=0, ddof=0), dtype=float)
    ci_width = 2.0 * Z95 * sd / np.sqrt(r)

    out = metadata.copy()
    for j, category in enumerate(CATEGORIES):
        out[f"Mean_{category}"] = mean[:, j]
        out[f"SD_{category}"] = sd[:, j]
        out[f"CI95_width_{category}"] = ci_width[:, j]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
