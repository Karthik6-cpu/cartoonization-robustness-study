# Step 3 — Statistical analysis of ResNet18 prediction files

## Scope and caveat
Calculated from the two uploaded prediction CSVs. This verifies prediction-table structure and analyzes stored predictions; it does **not** independently verify model training, the train/test split, preprocessing, or how the CSVs were generated.

## Data checks
- Standard ResNet18 rows: 78,500
- JPEG-augmented ResNet18 rows: 78,500
- Unique source IDs: 3,925
- Four image conditions and five quality settings per model.
- Each source ID has one ground-truth label and 20 condition/quality rows per model.
- Prediction tables use matching source/condition/quality keys.

## Uncompressed Top-1 accuracy (95% Wilson confidence intervals)
| Condition | Standard ResNet18 | JPEG-augmented ResNet18 |
|---|---:|---:|
| original | 86.65% (85.55–87.68) | 88.56% (87.53–89.52) |
| opencv | 58.78% (57.23–60.31) | 48.94% (47.38–50.51) |
| animegan2 | 76.92% (75.57–78.21) | 74.14% (72.75–75.49) |
| stable_diffusion | 79.77% (78.49–81.00) | 77.66% (76.33–78.93) |

## Paired robustness results at uncompressed quality
Paired bootstrap confidence intervals estimate original-minus-transformed accuracy differences by resampling source images. Exact McNemar tests assess paired correctness; Holm-adjusted p-values account for 30 original-versus-cartoon comparisons.

### Standard
- original - opencv: drop 27.87 percentage points (paired bootstrap 95% CI 26.22 to 29.43); Holm-adjusted McNemar p=7.38e-235.
- original - animegan2: drop 9.73 percentage points (paired bootstrap 95% CI 8.56 to 10.93); Holm-adjusted McNemar p=6.18e-57.
- original - stable_diffusion: drop 6.88 percentage points (paired bootstrap 95% CI 5.73 to 8.03); Holm-adjusted McNemar p=8.75e-31.
### Jpeg Augmented
- original - opencv: drop 39.62 percentage points (paired bootstrap 95% CI 38.01 to 41.27); Holm-adjusted McNemar p=0.
- original - animegan2: drop 14.42 percentage points (paired bootstrap 95% CI 13.17 to 15.72); Holm-adjusted McNemar p=3.31e-111.
- original - stable_diffusion: drop 10.90 percentage points (paired bootstrap 95% CI 9.68 to 12.13); Holm-adjusted McNemar p=5.74e-69.

## Interpretation
- OpenCV cartoonization has the largest observed accuracy reduction in the uncompressed condition for both models.
- JPEG-augmented ResNet18 is more accurate on original images in these files but less accurate on all three cartoonized conditions than standard ResNet18. This is an observed association, not proof of causation.
- The workbook contains all 20 condition/quality summaries, paired tests, model comparisons, per-class results, and robustness summaries.

## Required before publication
1. Verify the train/validation/test split and ensure test images were not used for training or model selection.
2. Verify checkpoints, preprocessing, and exact JPEG encoding settings.
3. Report statistical methods and multiple-testing correction.
4. Describe this as analysis of stored predictions, not a fresh model evaluation.
