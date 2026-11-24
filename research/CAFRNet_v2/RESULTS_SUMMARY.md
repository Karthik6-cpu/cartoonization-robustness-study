# CAFR-Net v2 — Consolidated Research Results

## Project scope

CAFR-Net (Cartoon-Aware Feature Refinement Network) combines ConvNeXt-Tiny hierarchical features, multi-scale semantic fusion, Sobel-based edge refinement, channel-spatial attention and a classification head.

The repository contains three methodologically distinct experiment groups. Their results must not be merged as if they shared the same split or training protocol.

## 1. Earlier three-seed Imagenette evaluation

- Dataset: Imagenette, 10 classes.
- Three full-model seeds: 42, 123 and 2026.
- Evaluation domains: original, OpenCV, AnimeGANv2 and Stable Diffusion.

| Domain | Mean accuracy | SD (percentage points) |
|---|---:|---:|
| Original | 99.771% | 0.044 |
| OpenCV | 90.403% | 4.180 |
| AnimeGANv2 | 98.735% | 0.550 |
| Stable Diffusion | 94.726% | 0.602 |

The OpenCV condition exhibits greater seed variability than the other reported domains. These are the earlier three-seed results and are not the matched seed-314 ResNet-18 comparison below.

## 2. Oxford-IIIT Pet external-dataset evaluation

- 37 classes; 7,349 images.
- Train: 2,944; validation: 736; official test: 3,669.
- CAFR-Net seed 42; 10 epochs.
- Original and OpenCV-cartoonized test conditions.

| Metric | Original | OpenCV |
|---|---:|---:|
| Accuracy | 94.603% | 86.781% |
| Macro precision | 94.620% | 87.193% |
| Macro recall | 94.554% | 86.719% |
| Macro-F1 | 94.552% | 86.660% |
| ECE (15 bins) | 0.019410 | 0.051689 |
| Mean confidence | 0.96435 | 0.91695 |

Paired correctness: both correct 3,129; original-only correct 342; OpenCV-only correct 55; both wrong 143. The reported exact two-sided paired p-value is 9.8420827e-52.

Paired bootstrap (5,000 resamples):
- Accuracy difference (original minus OpenCV): 0.07812; 95% CI [0.06759, 0.08831].
- Macro-F1 difference: 0.07909; 95% CI [0.06858, 0.08948].

This is a single-seed external-dataset evaluation. A matched ResNet-18 Oxford-IIIT Pet comparison is not available.

## 3. Matched Imagenette comparison against ResNet-18 (seed 314)

### Protocol and provenance

- Both models evaluated on the same 3,925 test identities.
- Six conditions: original, JPEG Q25, JPEG Q50, blur, noise and brightness.
- Total paired records: 23,550 per model (3,925 identities × 6 conditions).
- Prediction alignment checked one-to-one using `condition` and `relative_path`.
- The split was reconstructed using stratified `train_test_split(test_size=1421, random_state=42)`.
- This is a separate seed-314 experiment; it must not be described as sharing the historical split of the earlier three-seed CAFR-Net experiments.

### Accuracy comparison

| Condition | CAFR-Net v2 | ResNet-18 | Advantage (pp) |
|---|---:|---:|---:|
| Original | 99.796% | 98.140% | 1.6561 |
| JPEG Q25 | 99.720% | 97.783% | 1.9363 |
| JPEG Q50 | 99.771% | 97.860% | 1.9108 |
| Blur | 99.465% | 96.153% | 3.3121 |
| Noise | 99.694% | 97.299% | 2.3949 |
| Brightness | 99.796% | 97.401% | 2.3949 |

### Paired bootstrap confidence intervals

Paired image-level bootstrap used 2,000 resamples. The reported percentile 95% intervals for the accuracy difference (CAFR-Net minus ResNet-18) were:

| Condition | Advantage (pp) | 95% CI (pp) |
|---|---:|---:|
| Original | 1.6561 | [1.2739, 2.0892] |
| JPEG Q25 | 1.9363 | [1.5032, 2.3949] |
| JPEG Q50 | 1.9108 | [1.4777, 2.3701] |
| Blur | 3.3121 | [2.7006, 3.9236] |
| Noise | 2.3949 | [1.8854, 2.9045] |
| Brightness | 2.3949 | [1.9108, 2.8790] |

All six intervals exclude zero. These intervals quantify image-sampling uncertainty conditional on the trained checkpoints; they do not quantify variation across training seeds.

### Exact McNemar tests with Holm correction

All six paired exact McNemar tests remained significant after Holm correction. See `results/matched_resnet18_seed314/mcnemar_holm_exact.csv` for discordant-pair counts and exact adjusted p-values.

### Efficiency benchmark

NVIDIA Tesla T4; 224×224 input; 20 warm-up iterations and 100 measured repetitions.

| Model | Parameters (M) | Batch 1 (ms) | Batch 32 (ms) |
|---|---:|---:|---:|
| CAFR-Net v2 | 30.281 | 15.975 | 154.168 |
| ResNet-18 | 11.182 | 3.609 | 33.494 |

CAFR-Net v2 has approximately 2.71× the parameter count and is slower at both reported batch sizes. The accuracy advantage is accompanied by a compute and latency cost. Report memory figures as incremental peak memory above baseline, not total GPU footprint.

### Interpretation and limits

The matched seed-314 experiment supports higher observed CAFR-Net accuracy than ResNet-18 on these six conditions and this reconstructed test split. It is single-seed evidence. Multi-seed matched comparisons and broader external-dataset baseline comparisons remain future work. Do not infer universal robustness or state-of-the-art performance.

## 4. Ablations

The no-edge, no-semantic-fusion and no-attention ablations were run with seed 42. Treat these as single-seed component diagnostics; do not make multi-seed inferential claims from them.

## 5. Artifact map

- `results/imagenette/` — earlier three-seed results, ablations and predictions.
- `results/oxford_pet/` — Oxford-IIIT Pet evaluation.
- `results/matched_resnet18_seed314/` — matched baseline aggregate tables and statistical results.
- `results/figures/` — existing figures.
- `source/` — model, training and evaluation scripts.
- `manifests/` — dataset split and run metadata.

The matched-baseline prediction CSVs and the full Colab-generated result archive must be copied from the Colab runtime into this repository to complete the raw-artifact upload.

## 6. Final Colab artifacts and checkpoints

The large seed-314 checkpoints and complete CAFR-Net Colab experiment archives are available in the [final Colab artifact release](https://github.com/Karthik6-cpu/cartoonization-robustness-study/releases/tag/cafrnet-v2-colab-artifacts-2026-11-24). The release contains original and edge-corrected CAFR-Net archives and best/latest checkpoints, plus the matched ResNet-18 latest checkpoint.

The dated [Colab session backup](colab_session_backup/20261124_161028/) contains the notebook copy, prediction files, run histories, metadata, statistical tables, plots and packaged research outputs. The seven large release assets were uploaded separately to avoid storing large binary checkpoints in ordinary Git history.
