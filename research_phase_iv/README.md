# Phase IV — CAFR-Net

**Status: implementation prototype committed; empirical validation pending.**

CAFR-Net (Cartoon-Aware Feature Refinement Network) is a proposed model for
semantic image classification under cartoonization and JPEG compression shifts.

## Components
- ImageNet-1K pretrained ConvNeXt-Tiny hierarchical encoder.
- Three multi-scale semantic feature projections.
- Sobel edge/contour stream.
- Gradient-energy-conditioned four-way soft fusion.
- Residual refinement and classification head.
- Symmetric KL consistency loss for matched clean/shifted training views.

## Files
- `models/cafr_net.py`: model and consistency loss.
- `training/train_cafr_net.py`: paired-view training starter.
- `evaluation/evaluate_cafr_net.py`: per-image evaluation CSV exporter.
- `config.yaml`: initial configuration.
- `PROTOCOL.md`: hypotheses, controls, ablations, metrics, and success criteria.

## Quick start
Install compatible PyTorch and torchvision builds, then from repository root:

```bash
python training/train_cafr_net.py --data /path/to/imagenette --epochs 20 --batch-size 32 --seed 42
python evaluation/evaluate_cafr_net.py \
  --checkpoint outputs/cafr_net/best_cafr_net.pt \
  --conditions original=/path/to/test/original \
               opencv=/path/to/test/opencv \
               animegan=/path/to/test/animegan \
               stable_diffusion=/path/to/test/stable_diffusion \
  --out outputs/cafr_predictions.csv
```

Expected training layout is `DATA_ROOT/train/<class>/...` and
`DATA_ROOT/val/<class>/...`. Each evaluation condition uses
`CONDITION_ROOT/<class>/...`.

## Important limitations before publication
1. The training starter currently uses JPEG, blur, brightness, and noise
   perturbations. It does **not** yet generate OpenCV, AnimeGANv2, or Stable
   Diffusion cartoon views. Add the verified train-split cartoonization
   pipelines before describing training as cartoonization-aware.
2. The starter expects folder-based train/validation data and does not itself
   parse the source-identity manifest. Build the folders from the verified
   identity manifest and audit for identity leakage before training.
3. Evaluation exports relative paths; join them to the exact source-identity
   manifest to obtain stable matched source IDs. Do not assume filenames alone
   are unique identifiers.
4. The scripts are starter implementations and require a smoke test on the
   selected GPU/runtime before a full run.
5. No accuracy, superiority, statistical significance, or successful
   generalization claim has been established yet.

## Research references
- Liu et al. (2022), *A ConvNet for the 2020s*, CVPR.
  https://doi.org/10.1109/CVPR52688.2022.01167
- Yang & Soatto (2020), *FDA: Fourier Domain Adaptation for Semantic
  Segmentation*, CVPR. https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html
- Ganin et al. (2016), *Domain-Adversarial Training of Neural Networks*,
  JMLR 17(59). https://jmlr.csail.mit.edu/papers/v17/15-239.html

These references establish relevant background, not the novelty of CAFR-Net.
A full related-work search and comparison against closely related edge-aware,
frequency-aware, and domain-generalization methods is still required.
