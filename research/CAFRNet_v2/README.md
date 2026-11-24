# CAFR-Net v2 Research Package

**CAFR-Net:** Cartoon-Aware Feature Refinement Network

## Overview

This package contains the implementation, experimental results, evaluation outputs, figures, and documentation for CAFR-Net v2.

## Research experiments

### 1. Imagenette
- 10-class image classification benchmark.
- Three independent training seeds: 42, 123, and 2026.
- Evaluation across Original, OpenCV cartoonized, AnimeGANv2, and Stable Diffusion domains.
- Component ablation experiments covering edge refinement, semantic fusion, and attention.

### 2. Oxford-IIIT Pet
- 37-class external dataset evaluation.
- Training and validation split with seed 42.
- Evaluation on original and OpenCV-cartoonized official test images.

## Package structure

- `source/` — Model, training and evaluation source code.
- `results/imagenette/` — Predictions, metrics, training histories and ablations.
- `results/oxford_pet/` — External dataset evaluation outputs.
- `results/figures/` — Research plots in PNG and PDF formats.
- `results/tables/` — Consolidated result tables.
- `manifests/` — Dataset split manifest and metadata.
- `RESULTS_SUMMARY.md` — Consolidated numerical findings.
- `package_inventory.json` — File integrity inventory.

## Important research notes

- Imagenette and Oxford-IIIT Pet results are reported separately.
- The Oxford-IIIT Pet experiment is a single-seed external-dataset evaluation.
- Ablation results are single-seed results and should not be interpreted as multi-seed statistical evidence.
- Cartoon-domain performance differences indicate that domain shift remains a challenge.

## Reproducibility

See `source/requirements.txt` for the Python dependencies. Run metadata, training histories, predictions, and dataset manifests are included where available.

## Citation

If you use this work, please cite the corresponding research manuscript once published.
