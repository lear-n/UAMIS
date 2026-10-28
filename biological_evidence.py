"""Utilities for organizing literature, DAVID, and TCGA evidence for UAMIS candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["Tissue", "Protein Id"]


def read_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Protein Id": str})
    required = KEY + ["Protein Name"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Candidate table must contain {required}")
    if df[required].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError("Candidate identifiers must be complete and unique within tissue")
    return df


def export_candidates(candidates: Path, output: Path) -> None:
    df = read_candidates(candidates)
    output.mkdir(parents=True, exist_ok=False)
    for tissue, group in df.groupby("Tissue"):
        folder = output / tissue
        folder.mkdir()
        folder.joinpath("gene_symbols.txt").write_text(
            "\n".join(sorted(set(group["Protein Name"]))) + "\n", encoding="utf-8")
        folder.joinpath("protein_ids.txt").write_text(
            "\n".join(sorted(set(group["Protein Id"]))) + "\n", encoding="utf-8")
    template = df[["Tissue", "Protein Id", "Protein Name"]].copy()
    template["Evidence category"] = ""
    template["Reference DOI or PMID"] = ""
    template["Evidence note"] = ""
    template.to_csv(output / "literature_review_template.csv", index=False)


def summarize_literature(candidates: Path, evidence: Path, output: Path) -> None:
    df = read_candidates(candidates)
    ev = pd.read_csv(evidence, dtype={"Protein Id": str}, keep_default_na=False)
    required = KEY + ["Evidence category", "Reference DOI or PMID", "Evidence note"]
    if not set(required).issubset(ev.columns):
        raise ValueError(f"Literature evidence must contain {required}")
    if ev.duplicated(KEY).any():
        raise ValueError("Use one consolidated literature-evidence row per tissue/protein")
    allowed = {"same_cancer", "broader_cancer_related", "no_suitable_support"}
    if not set(ev["Evidence category"]).issubset(allowed):
        raise ValueError(f"Evidence category must be one of {sorted(allowed)}")
    supported = ev["Evidence category"].ne("no_suitable_support")
    if ev.loc[supported, "Reference DOI or PMID"].str.strip().eq("").any():
        raise ValueError("Supported evidence rows require a DOI or PMID")

    merged = df.merge(ev[required], on=KEY, how="left", validate="one_to_one")
    merged["Evidence category"] = merged["Evidence category"].fillna("unreviewed")
    output.mkdir(parents=True, exist_ok=False)
    merged.to_csv(output / "candidate_literature.csv", index=False)
    merged.groupby(["Tissue", "Evidence category"]).size().rename("Count").reset_index().to_csv(
        output / "literature_summary.csv", index=False)


def import_david(input_path: Path, tissue: str, output: Path) -> None:
    df = pd.read_csv(input_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    required = ["Category", "Term", "Count", "PValue", "Genes", "Fold Enrichment", "Bonferroni", "Benjamini", "FDR"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"DAVID Functional Annotation Chart must contain {required}")
    for col in ["PValue", "Bonferroni", "Benjamini", "FDR"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or not df[col].between(0, 1).all():
            raise ValueError(f"Invalid probability column: {col}")
    df.insert(0, "Tissue", tissue)
    df["Nominal_P_lt_0.05"] = df["PValue"] < 0.05
    output.mkdir(parents=True, exist_ok=False)
    df.to_csv(output / "david_complete_output.csv", index=False)
    df[df["Nominal_P_lt_0.05"]].to_csv(output / "david_main_text_terms.csv", index=False)


def summarize_tcga(candidates: Path, matrix: Path, tissue: str, output: Path, threshold: float = 1.0) -> None:
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("z-score threshold must be positive and finite")
    candidate_df = read_candidates(candidates)
    genes = sorted(set(candidate_df.loc[candidate_df["Tissue"].eq(tissue), "Protein Name"]))
    df = pd.read_csv(matrix, sep=None, engine="python")
    if "Sample Id" not in df.columns or df["Sample Id"].isna().any() or df["Sample Id"].duplicated().any():
        raise ValueError("TCGA z-score matrix must contain a unique nonmissing Sample Id column")

    rows = []
    for gene in genes:
        if gene not in df.columns:
            rows.append({"Tissue": tissue, "Gene": gene, "Status": "not_in_matrix", "Valid_samples": 0})
            continue
        values = pd.to_numeric(df[gene], errors="raise").dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite z-score values for {gene}")
        n = len(values)
        rows.append({
            "Tissue": tissue,
            "Gene": gene,
            "Status": "ok" if n else "all_missing",
            "Valid_samples": n,
            "Missing_samples": len(df) - n,
            "z_gt_threshold": int((values > threshold).sum()),
            "z_lt_negative_threshold": int((values < -threshold).sum()),
            "Percent_high": 100 * (values > threshold).sum() / n if n else np.nan,
            "Percent_low": 100 * (values < -threshold).sum() / n if n else np.nan,
            "z_threshold": threshold,
        })

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "tcga_expression_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("literature")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("david")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tissue", required=True)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("tcga")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--tissue", required=True)
    p.add_argument("--z-threshold", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "export":
        export_candidates(args.candidates, args.output)
    elif args.command == "literature":
        summarize_literature(args.candidates, args.evidence, args.output)
    elif args.command == "david":
        import_david(args.input, args.tissue, args.output)
    else:
        summarize_tcga(args.candidates, args.input, args.tissue, args.output, args.z_threshold)
