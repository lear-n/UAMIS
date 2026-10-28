# Input format

The public UAMIS workflow begins from stochastic localization predictions produced by a compatible pretrained multimodal localization model. 

## 1. Directory layout

For each tissue, provide one `normal` and one `tumor` directory:

```text
inputs/<Tissue>/normal/
  metadata.csv
  mc_predictions.npy

inputs/<Tissue>/tumor/
  metadata.csv
  mc_predictions.npy
```

The tissue names used in `configs/pipeline.json` must match the directory names exactly.

## 2. metadata.csv

Required columns:

```text
Protein Name,Protein Id
```

Each row represents one IHC image sample. Additional non-identifying columns such as `Antibody Id` or `Sample Id` may be included and will be retained in the image-level summary.

Rows must remain in the exact order used when generating `mc_predictions.npy`.

For a given `Protein Id`, `Protein Name` must be consistent across all associated samples and tissue states.

## 3. mc_predictions.npy

Required shape:

```text
[R, N, 5]
```

where:

- `R` is the number of stochastic forward passes (`R = 30` in the study);
- `N` is the number of rows in `metadata.csv`;
- the final dimension contains five independent Sigmoid localization probabilities.

Category order:

```text
0  Cytoplasm
1  Endoplasmic reticulum
2  Mitochondria
3  Nucleus
4  Plasma membrane
```

All values must be finite and in `[0, 1]`. Because the upstream predictor uses independent Sigmoid outputs, the five values for a single prediction do not need to sum to one.

## 4. Upstream ProLoc-IHS interface

A compatible upstream implementation should perform the following conceptual operation for each paired IHC image and protein sequence:

```text
paired IHC image + matched protein sequence
        ↓
pretrained ProLoc-IHS
        ↓
keep dropout active during inference
        ↓
repeat R times
        ↓
store five Sigmoid probabilities per pass
```

The public UAMIS repository does not prescribe a particular local file layout for ProLoc-IHS, its feature extractors, or its pretrained weights. Only the resulting prediction tensor and aligned metadata are required by the downstream analysis.

## 5. Optional master metadata table

`prepare_paired_metadata.py` can be used to validate a combined metadata table and split it into paired tissue/state folders before externally generated prediction tensors are added.

Required columns:

```text
Tissue,State,Protein Name,Protein Id
```

`State` must be `normal` or `tumor`.

Example:

```bash
python prepare_paired_metadata.py --input master_metadata.csv --output inputs
```

This utility does not download, preprocess, or redistribute IHC images or protein sequences.
