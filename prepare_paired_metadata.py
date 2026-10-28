"""Validate and split a non-identifying metadata table by tissue and state."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = ["Tissue", "State", "Protein Name", "Protein Id"]
VALID_STATES = {"normal", "tumor"}


def prepare_paired_metadata(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    if df.empty or not set(REQUIRED).issubset(df.columns):
        raise ValueError(f"Input metadata must be nonempty and contain {REQUIRED}")

    for col in REQUIRED:
        df[col] = df[col].str.strip()
        if df[col].eq("").any():
            raise ValueError(f"Blank values found in required column: {col}")

    if not set(df["State"]).issubset(VALID_STATES):
        raise ValueError("State must be 'normal' or 'tumor'")

    name_counts = df.groupby("Protein Id")["Protein Name"].nunique()
    if (name_counts > 1).any():
        raise ValueError("A Protein Id maps to multiple Protein Name values")

    output_dir.mkdir(parents=True, exist_ok=False)
    exclusions = []
    summary = []

    for tissue, group in df.groupby("Tissue", sort=False):
        normal_ids = set(group.loc[group["State"].eq("normal"), "Protein Id"])
        tumor_ids = set(group.loc[group["State"].eq("tumor"), "Protein Id"])
        shared = normal_ids & tumor_ids
        if not shared:
            raise ValueError(f"No paired normal/tumor proteins found for {tissue}")

        for state in ("normal", "tumor"):
            state_df = group[group["State"].eq(state)].copy()
            missing = state_df[~state_df["Protein Id"].isin(shared)]
            for pid in missing["Protein Id"].drop_duplicates():
                exclusions.append({"Tissue": tissue, "State": state, "Protein Id": pid,
                                   "Reason": "missing_corresponding_state"})
            paired = state_df[state_df["Protein Id"].isin(shared)].copy()
            folder = output_dir / tissue / state
            folder.mkdir(parents=True, exist_ok=True)
            paired.drop(columns=["Tissue", "State"]).to_csv(folder / "metadata.csv", index=False)
            summary.append({"Tissue": tissue, "State": state,
                            "Images": len(paired), "Paired proteins": len(shared)})

    pd.DataFrame(exclusions, columns=["Tissue", "State", "Protein Id", "Reason"]).to_csv(
        output_dir / "pairing_exclusions.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / "metadata_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_paired_metadata(args.input, args.output)
