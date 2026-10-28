"""Run the public UAMIS downstream analysis from MC-Dropout prediction tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsd_screening import run_screening
from summarize_mc_predictions import summarize_folder


def run(config_path: Path) -> None:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    input_root = (base / config.get("input_root", "../inputs")).resolve()
    output_root = (base / config.get("output_root", "../results")).resolve()
    tissues = list(config["tissues"])
    passes = int(config.get("passes", 30))
    cutoff = float(config.get("jsd_cutoff", 0.1))
    delta = float(config.get("delta", 1e-12))

    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    image_root = output_root / "image_level"
    for tissue in tissues:
        for state in ("normal", "tumor"):
            source = input_root / tissue / state
            target = image_root / tissue / state / "image_level_summary.csv"
            summarize_folder(source, target, expected_passes=passes)

    run_screening(
        image_root=image_root,
        output_dir=output_root / "screening",
        tissues=tissues,
        cutoff=cutoff,
        delta=delta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.json"))
    args = parser.parse_args()
    run(args.config)
