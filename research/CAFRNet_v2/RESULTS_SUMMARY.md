# CAFR-Net v2 — Consolidated Research Results

## Project
CAFR-Net: Cartoon-Aware Feature Refinement Network.

This folder consolidates the active CAFR-Net v2 implementation and the
Imagenette and Oxford-IIIT Pet experiments. Older BICR-Net research files
in the parent repository are separate work and are not part of these results.

## Imagenette experiments

- Dataset: 13,394 verified images across 10 classes.
- Official split: 8,048 train, 1,421 validation, 3,925 test.
- Three independent full-model seeds: 42, 123, 2026.
- Three ablations: no edge, no semantic fusion, no attention.
- Test domains: original, OpenCV, AnimeGANv2, Stable Diffusion.
- The full model's OpenCV performance is seed-sensitive.

Three-seed mean accuracy:
- Original: 99.771% (SD 0.044 percentage points)
- OpenCV: 90.403% (SD 4.180 percentage points)
- AnimeGANv2: 98.735% (SD 0.550 percentage points)
- Stable Diffusion: 94.726% (SD 0.602 percentage points)

These are three-seed results on Imagenette, not evidence of universal
robustness across all datasets or cartoonization methods.

## Oxford-IIIT Pet independent-dataset experiment

- 37 classes; 7,349 images.
- Train: 2,944; validation: 736; official test: 3,669.
- One CAFR-Net v2 training seed: 42; 10 epochs.
- Best validation accuracy: 95.9239%.
- Test conditions: original and matched OpenCV cartoonized images.

| Metric | Original | OpenCV |
|---|---:|---:|
| Accuracy | 94.603% | 86.781% |
| Macro precision | 94.620% | 87.193% |
| Macro recall | 94.554% | 86.719% |
| Macro-F1 | 94.552% | 86.660% |
| ECE (15 bins) | 0.019410 | 0.051689 |
| Mean confidence | 0.96435 | 0.91695 |

Paired correctness counts:
- Both correct: 3,129
- Original only correct: 342
- OpenCV only correct: 55
- Both wrong: 143
- Exact two-sided paired binomial/McNemar-equivalent p-value:
  9.8420827e-52

Paired bootstrap (5,000 resamples, seed 2026):
- Accuracy difference (original minus OpenCV): 0.07812
  95% CI: [0.06759, 0.08831]
- Macro-F1 difference: 0.07909
  95% CI: [0.06858, 0.08948]

The Oxford experiment uses one seed. Its results must be reported as
single-seed external-dataset validation, not multi-seed validation.

## Important interpretation

The results show that cartoonization remains a meaningful domain shift.
CAFR-Net v2 does not eliminate the performance gap. Imagenette results
also show seed sensitivity for OpenCV cartoonization. These limitations
must be stated in the manuscript.

## Reproducibility

- Active model, trainer and evaluator are in source/.
- Experiment CSVs, predictions and logs are in results/.
- Oxford split manifest and metadata are in manifests/.
- Large checkpoints, datasets and generated archives are intentionally
  excluded from this consolidated source/results folder.
