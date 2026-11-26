
# CAFR-Net v2 — Final Research Experiment Package

## Dataset
Imagenette2-320, 10 classes.

The main model and baselines were trained using the reconstructed
seed-42 training/validation split. Final comparative evaluation used
the official validation directory containing 3,925 images.

## Model experiments
1. CAFR-Net v2 full configuration
2. CAFR-Net without edge refinement
3. CAFR-Net without semantic fusion
4. CAFR-Net without attention
5. ResNet-18 pretrained baseline
6. ConvNeXt-Tiny pretrained baseline

## Clean official validation results

| Model | Accuracy |
|---|---:|
| ConvNeXt-Tiny | 99.8981% |
| CAFR-Net v2 Full | 99.7962% |
| CAFR-Net No Semantic Fusion | 99.7962% |
| CAFR-Net No Edge | 99.7707% |
| CAFR-Net No Attention | 99.6943% |
| ResNet-18 | 97.9873% |

## Average corruption accuracy

| Model | Average accuracy |
|---|---:|
| ConvNeXt-Tiny | 99.3987% |
| CAFR-Net v2 Full | 99.2611% |
| CAFR-Net No Semantic Fusion | 99.1745% |
| CAFR-Net No Edge | 99.0675% |
| CAFR-Net No Attention | 98.7771% |
| ResNet-18 | 95.7911% |

Average corruption accuracy is the arithmetic mean across five
conditions, excluding clean images.

## Robustness conditions
- Gaussian noise, sigma 20
- Gaussian blur, radius 2
- JPEG compression, quality 20
- Brightness factor 0.5
- Brightness factor 1.5

Six models x six conditions x 3,925 images = 141,300
image-condition predictions.

## Ablation protocol qualification
The three ablation variants were initialized from the full-model
epoch-30 checkpoint and fine-tuned for 10 epochs. The full model
was trained for 30 epochs. Therefore, the ablation comparison is
a warm-started component-removal study with a different training
budget, not an equal-budget from-scratch comparison.

## Interpretation and limitations
- ConvNeXt-Tiny achieved the highest clean accuracy and highest
  average corruption accuracy in this evaluation.
- CAFR-Net v2 outperformed ResNet-18 across all six conditions.
- The full CAFR-Net showed better blur and noise accuracy than
  its component-removal variants.
- These controlled corruption tests use the same official
  validation images. They are not external-dataset evaluations.
- Results do not establish generalization to arbitrary unseen
  datasets, domains, or real-world distribution shifts.
- The official validation directory is used for comparative
  evaluation; it should not be described as an independent
  external test dataset.

## Reproducibility
The package includes model checkpoints, source code, predictions,
per-class metrics, confusion matrices, robustness results,
comparison tables, figures, and evaluation metadata.

Generated: 2026-11-26T08:13:52

