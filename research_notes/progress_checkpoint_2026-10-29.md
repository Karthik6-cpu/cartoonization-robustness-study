# Research Progress Checkpoint — 29 October 2026

Project: Cartoonization / JPEG robustness study  
Repository: https://github.com/Karthik6-cpu/cartoonization-robustness-study  
Google Drive project root: `CartoonizationResearchBackup/Advanced_Project`

## Current task status

We were calculating **original-image vs JPEG-compressed-image PSNR and SSIM** for the primary 3,925-image Imagenette test set at JPEG qualities 25, 50, 75, and 95.

Expected output:
`metrics/original_vs_jpeg_quality_metrics.csv`

Expected rows when complete: 3,925 images × 4 qualities = **15,700 rows**.

The first attempt ran slowly and did not write a completed CSV because saving was only at the end. At the time we stopped, the in-memory `records` variable contained only 5 records, all for quality 25. No completed results CSV was found. Therefore, do not treat the original-vs-JPEG PSNR/SSIM calculation as completed.

The replacement cell indexes original images once, resumes from any existing CSV, and saves checkpoints every 100 images and after each quality. On the next session:
1. Mount Google Drive in Colab.
2. Check whether `metrics/original_vs_jpeg_quality_metrics.csv` exists and inspect row counts by quality.
3. Run the resumable calculation only for missing image-quality pairs.
4. Verify 3,925 rows per quality and 15,700 rows total, with no duplicate `image_id, quality` pairs.
5. Report mean and standard deviation of PSNR and SSIM by quality.

## Established main experiment results

Primary test set: 3,925 Imagenette images. Two ResNet18 models:
- `resnet18_standard`
- `resnet18_jpeg_augmented`

### Clean accuracy
- Standard: 86.6497%
- JPEG-augmented: 88.5605%
- Difference: +1.9108 percentage points

### JPEG accuracy

| JPEG quality | Standard accuracy | Augmented accuracy | Difference |
|---:|---:|---:|---:|
| 95 | 86.7771% | 88.4841% | +1.7070 pp |
| 75 | 86.7771% | 88.5350% | +1.7580 pp |
| 50 | 86.5223% | 88.1529% | +1.6306 pp |
| 25 | 86.1656% | 87.7197% | +1.5541 pp |

### Paired exact McNemar tests

Augmented-only correct vs standard-only correct:
- Clean: 307 vs 232; p = 0.00141316
- Q95: 299 vs 232; p = 0.00413844
- Q75: 304 vs 235; p = 0.00336333
- Q50: 305 vs 241; p = 0.00696019
- Q25: 310 vs 249; p = 0.0110917

Holm-adjusted p-values over the family of five tests (clean + four JPEG conditions):
- Clean: 0.00706579
- Q95: 0.0134533
- Q75: 0.0134533
- Q50: 0.0139204
- Q25: 0.0139204

All five remain statistically significant at 0.05 under this correction. Clearly specify this five-test family in the paper.

## Class-wise findings

Class-wise accuracy difference (augmented minus standard, percentage points):

| Class | Q25 | Q50 | Q75 | Q95 |
|---|---:|---:|---:|---:|
| Tench (`n01440764`) | -0.52 | -1.03 | -0.52 | -0.52 |
| English Springer (`n02102040`) | +2.78 | +0.51 | +1.27 | +1.77 |
| Cassette player (`n02979186`) | -6.16 | -7.56 | -6.16 | -6.44 |
| Chainsaw (`n03000684`) | +3.89 | +5.44 | +4.92 | +5.44 |
| Church (`n03028079`) | +3.91 | +4.40 | +3.67 | +3.18 |
| French horn (`n03394916`) | -5.84 | -4.31 | -2.54 | -2.79 |
| Garbage truck (`n03417042`) | -14.14 | -13.11 | -13.37 | -13.11 |
| Gas pump (`n03425413`) | +26.49 | +28.16 | +26.49 | +26.25 |
| Golf ball (`n03445777`) | -3.01 | -3.76 | -3.76 | -3.76 |
| Parachute (`n03888257`) | +5.64 | +4.87 | +5.13 | +4.62 |

Across 40 class × quality paired tests, 10 were significant after Holm correction. Gas pump improved significantly at all four qualities; garbage truck worsened significantly at all four qualities. Cassette player was significantly worse at Q50 (adjusted p = 0.0063) and Q95 (adjusted p = 0.0255). Values displayed as 0.0000 are rounded values, not literal zero.

These results support class-dependent performance effects, not a proven texture/shape mechanism or a causal effect of cartoonization.

## Q50 confusion observations

For true gas-pump images, the standard model often predicted cassette player (88) or garbage truck (49); the augmented model predicted cassette player 12 times. For true garbage-truck images, the augmented model frequently predicted gas pump (38). These are descriptive confusion patterns and need cautious interpretation.

## Separate 500-image OpenCV visual-quality pilot — keep separate from the main test

File: `metrics/opencv_500_visual_quality_metrics.csv` contains 2,000 rows (500 images × 4 qualities). Its PSNR/SSIM compare OpenCV-cartoonized images with their JPEG versions, **not original images with cartoonized images**.

Known summary values:
- Q25: mean PSNR 28.490940 dB; mean SSIM 0.871769; mean size reduction 81.112915%
- Q50: mean PSNR 31.155241 dB; mean SSIM 0.918965; mean size reduction 70.718144%
- Q75: mean PSNR 34.461362 dB; mean SSIM 0.953256; mean size reduction 56.430641%
- Q95: summary not yet recorded in this checkpoint.

Exploratory pooled correlations across repeated image-quality observations were weak; they may be confounded by quality and repeated measures, so do not interpret them causally.

## Important files already reported in the Drive project

- `predictions/imagenette_clean_test_predictions.csv`
- `predictions/jpeg_robustness_predictions.csv` (31,400 rows: 3,925 × 2 models × 4 qualities)
- `manifests/split_manifest.csv`
- `manifests/imagenette_split_manifest.csv`
- `manifests/jpeg_test_manifest.csv`
- `metrics/clean_test_summary.csv`
- `metrics/jpeg_robustness_summary.csv`
- `metrics/jpeg_paired_statistical_tests.csv`
- `metrics/clean_test_per_class_metrics.csv`
- `metrics/jpeg_per_class_metrics.csv`

JPEG output folders `outputs/imagenette_jpeg/quality_25`, `quality_50`, `quality_75`, and `quality_95` each contain 3,925 files. Prediction IDs were checked against output filename stems: no missing or extra IDs for the four qualities.

## Next session checklist

- [ ] Mount Drive and verify existing CSV/checkpoints.
- [ ] Finish original-vs-JPEG PSNR/SSIM with incremental saving.
- [ ] Verify row counts: 3,925 per quality, 15,700 total, no duplicate pairs.
- [ ] Read Q95 summary from the separate OpenCV 500-image pilot.
- [ ] Keep the 500-image visual-quality pilot separate from the 3,925-image classification robustness experiment.
- [ ] Save the resulting CSV, summary table, and code in Drive and commit appropriate research artifacts to GitHub.
