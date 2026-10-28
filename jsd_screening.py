"""Protein-level aggregation, relative-distribution normalization, and JSD screening."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from summarize_mc_predictions import CATEGORIES

MEAN_COLS = [f"Mean_{name}" for name in CATEGORIES]


def normalize_relative(scores, delta: float = 1e-12) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("delta must be a positive finite value")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Localization scores must be finite and nonnegative")
    smoothed = x + delta
    return smoothed / smoothed.sum(axis=-1, keepdims=True)


def js_divergence(p, q, delta: float = 1e-12) -> float:
    p = normalize_relative(p, delta)
    q = normalize_relative(q, delta)
    if p.shape != q.shape:
        raise ValueError("JSD inputs must have identical shapes")
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(max(0.0, 0.5 * kl_pm + 0.5 * kl_qm))


def _read_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Protein Id": str})
    required = {"Protein Name", "Protein Id", *MEAN_COLS}
    if df.empty or not required.issubset(df.columns):
        raise ValueError(f"Image-level summary must contain {sorted(required)}")
    if df[["Protein Name", "Protein Id", *MEAN_COLS]].isna().any().any():
        raise ValueError("Image-level summary contains missing required values")
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    names = df.groupby("Protein Id")["Protein Name"].nunique()
    if (names > 1).any():
        raise ValueError("A Protein Id maps to multiple Protein Name values")
    agg = df.groupby(["Protein Id", "Protein Name"], sort=False)[MEAN_COLS].mean().reset_index()
    counts = df.groupby(["Protein Id", "Protein Name"], sort=False).size().rename("Images").reset_index()
    return agg.merge(counts, on=["Protein Id", "Protein Name"], validate="one_to_one")


def analyze_tissue(normal_csv: Path, tumor_csv: Path, tissue: str,
                   cutoff: float = 0.1, delta: float = 1e-12):
    normal = _aggregate(_read_summary(normal_csv))
    tumor = _aggregate(_read_summary(tumor_csv))

    normal_ids = set(normal["Protein Id"])
    tumor_ids = set(tumor["Protein Id"])
    shared = normal_ids & tumor_ids
    excluded = []
    for pid in sorted(normal_ids - shared):
        excluded.append({"Tissue": tissue, "Protein Id": pid, "Reason": "normal_only"})
    for pid in sorted(tumor_ids - shared):
        excluded.append({"Tissue": tissue, "Protein Id": pid, "Reason": "tumor_only"})

    n = normal[normal["Protein Id"].isin(shared)].copy()
    t = tumor[tumor["Protein Id"].isin(shared)].copy()
    merged = n.merge(t, on="Protein Id", suffixes=("_N", "_T"), validate="one_to_one")
    if (merged["Protein Name_N"] != merged["Protein Name_T"]).any():
        raise ValueError(f"Protein Name mismatch between normal and tumor in {tissue}")

    rows = []
    for _, row in merged.iterrows():
        n_scores = np.array([row[f"{col}_N"] for col in MEAN_COLS], dtype=float)
        t_scores = np.array([row[f"{col}_T"] for col in MEAN_COLS], dtype=float)
        n_dist = normalize_relative(n_scores, delta)
        t_dist = normalize_relative(t_scores, delta)
        score = js_divergence(n_scores, t_scores, delta)
        n_max = np.max(n_scores)
        t_max = np.max(t_scores)
        n_idx = int(np.argmax(n_scores))
        t_idx = int(np.argmax(t_scores))
        record = {
            "Tissue": tissue,
            "Protein Name": row["Protein Name_N"],
            "Protein Id": row["Protein Id"],
            "Normal_images": int(row["Images_N"]),
            "Tumor_images": int(row["Images_T"]),
            "JSD": score,
            "Candidate": bool(score >= cutoff),
            "Normal_dominant": CATEGORIES[n_idx].replace("_", " "),
            "Tumor_dominant": CATEGORIES[t_idx].replace("_", " "),
            "Dominant_changed": bool(n_idx != t_idx),
            "Dominant_tie": bool((np.isclose(n_scores, n_max)).sum() > 1 or (np.isclose(t_scores, t_max)).sum() > 1),
        }
        for j, category in enumerate(CATEGORIES):
            clean = category.replace("_", " ")
            record[f"Normal_{clean}_mean"] = n_scores[j]
            record[f"Tumor_{clean}_mean"] = t_scores[j]
            record[f"Normal_{clean}_relative"] = n_dist[j]
            record[f"Tumor_{clean}_relative"] = t_dist[j]
        rows.append(record)

    result = pd.DataFrame(rows)
    return result, pd.DataFrame(excluded, columns=["Tissue", "Protein Id", "Reason"])


def run_screening(image_root: Path, output_dir: Path, tissues: list[str],
                  cutoff: float = 0.1, delta: float = 1e-12,
                  thresholds=(0.05, 0.10, 0.15, 0.20)):
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    all_results, all_excluded = [], []
    for tissue in tissues:
        normal_csv = image_root / tissue / "normal" / "image_level_summary.csv"
        tumor_csv = image_root / tissue / "tumor" / "image_level_summary.csv"
        result, excluded = analyze_tissue(normal_csv, tumor_csv, tissue, cutoff, delta)
        all_results.append(result)
        all_excluded.append(excluded)

    results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    excluded = pd.concat(all_excluded, ignore_index=True) if all_excluded else pd.DataFrame(
        columns=["Tissue", "Protein Id", "Reason"])

    results.to_csv(output_dir / "protein_jsd.csv", index=False)
    results[results["Candidate"]].to_csv(output_dir / "candidates.csv", index=False)
    excluded.to_csv(output_dir / "excluded_proteins.csv", index=False)

    paired_counts = results.groupby("Tissue").size().rename("Paired_proteins")
    candidate_rows = results[results["Candidate"]]
    candidate_counts = candidate_rows.groupby("Tissue").size().rename("Candidates")
    changed_counts = candidate_rows.groupby("Tissue")["Dominant_changed"].sum().rename("Dominant_changed")
    summary = pd.concat([paired_counts, candidate_counts, changed_counts], axis=1).fillna(0).reset_index()
    for col in ["Paired_proteins", "Candidates", "Dominant_changed"]:
        summary[col] = summary[col].astype(int)
    summary["Dominant_unchanged"] = summary["Candidates"] - summary["Dominant_changed"]
    summary.to_csv(output_dir / "summary_by_tissue.csv", index=False)

    sensitivity_rows = []
    for threshold in thresholds:
        sensitivity_rows.append({"Scope": "Overall", "Tissue": "All",
                                 "JSD_cutoff": threshold, "Candidates": int((results["JSD"] >= threshold).sum())})
        for tissue, group in results.groupby("Tissue"):
            sensitivity_rows.append({"Scope": "Tissue", "Tissue": tissue,
                                     "JSD_cutoff": threshold, "Candidates": int((group["JSD"] >= threshold).sum())})
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    return results
