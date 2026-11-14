
# BICR-Net

Blind Interaction-aware Compositional Robustness Network

## Research objective

Investigate blind inference of latent degradation factors,
pairwise degradation interactions, and feature-level correction
for image classification under held-out combinations of
synthetic visual corruptions.

## Architecture

1. Shared ResNet-18 encoder.
2. Blind degradation factorizer.
3. Pairwise latent interaction module.
4. Residual feature correction.
5. Layer normalization and classification head.

## Training protocol

Dataset: Imagenette
Input: 3-channel 160 x 160 images
Backbone: ResNet-18 trained from scratch
Optimizer: AdamW
Seed: 42
Training combinations:
- Clean
- Noise
- Blur
- Brightness
- Noise + brightness
- Noise + blur

Held-out combinations:
- Blur + brightness
- Noise + blur + brightness

## Important implementation details

The factorizer predicts three synthetic corruption indicators
and continuous strength values. Training uses synthetic
corruption metadata as supervision. Inference uses only the image.

The current model uses a simplified average-pooling blur operator
and synthetic additive Gaussian noise and brightness changes.

The current implementation is a prototype and should not be
described as a validated solution to general real-world
distribution shifts.

## Current limitations

- One training seed only.
- Factor identification is imperfect, especially on combined shifts.
- Interaction ablation is an inference-time ablation, not a
  separately retrained ablation.
- Validation results were used for checkpoint selection.
- Independent test results must be reported separately.
- Baseline comparisons require matched training and evaluation
  protocols.
- Novelty requires further literature and citation verification.

## Reproducibility

The Kaggle notebook containing the implementation should be
saved separately using Kaggle Save Version. This folder stores
the model checkpoints, experiment configuration, result tables,
and generated figures.
