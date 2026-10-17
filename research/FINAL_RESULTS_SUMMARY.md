# Final Results and Validation Report

## Study overview

This report consolidates the existing results for a matched evaluation of image cartoonization and JPEG-compression robustness.

- Dataset: Imagenette 320, 10 classes.
- Matched test images: 3,925 unique source images.
- Image conditions: Original, OpenCV cartoonization, AnimeGANv2, and Stable Diffusion v1.5 image-to-image.
- JPEG settings: uncompressed, Q95, Q75, Q50, and Q25.
- Classifiers: standard ResNet18 and JPEG-augmentation-trained ResNet18.
- Prediction rows: 78,500 per classifier; 157,000 combined.
- Visual-quality rows: 62,800 across four conditions and four compressed quality settings.

## Recognition results

The following table shows Top-1 accuracy for uncompressed images and JPEG quality 25. Refer to `paper/tables/final_recognition_results_all_conditions.csv` for all 40 classifier-condition-quality combinations.

| Classifier | Image condition | JPEG quality | N | Top-1 accuracy | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| JPEG-Augmented ResNet18 | AnimeGANv2 | 25 | 3925 | 0.7457 | 0.7506 | 0.7434 |
| JPEG-Augmented ResNet18 | OpenCV cartoonization | 25 | 3925 | 0.5192 | 0.4902 | 0.5141 |
| JPEG-Augmented ResNet18 | Original | 25 | 3925 | 0.8787 | 0.8787 | 0.8780 |
| JPEG-Augmented ResNet18 | Stable Diffusion img2img | 25 | 3925 | 0.7727 | 0.7708 | 0.7694 |
| JPEG-Augmented ResNet18 | AnimeGANv2 | uncompressed | 3925 | 0.7414 | 0.7462 | 0.7394 |
| JPEG-Augmented ResNet18 | OpenCV cartoonization | uncompressed | 3925 | 0.4894 | 0.4528 | 0.4841 |
| JPEG-Augmented ResNet18 | Original | uncompressed | 3925 | 0.8856 | 0.8855 | 0.8850 |
| JPEG-Augmented ResNet18 | Stable Diffusion img2img | uncompressed | 3925 | 0.7766 | 0.7749 | 0.7734 |
| Standard ResNet18 | AnimeGANv2 | 25 | 3925 | 0.7707 | 0.7786 | 0.7713 |
| Standard ResNet18 | OpenCV cartoonization | 25 | 3925 | 0.6092 | 0.6037 | 0.6111 |
| Standard ResNet18 | Original | 25 | 3925 | 0.8619 | 0.8609 | 0.8637 |
| Standard ResNet18 | Stable Diffusion img2img | 25 | 3925 | 0.7952 | 0.7944 | 0.7968 |
| Standard ResNet18 | AnimeGANv2 | uncompressed | 3925 | 0.7692 | 0.7771 | 0.7701 |
| Standard ResNet18 | OpenCV cartoonization | uncompressed | 3925 | 0.5878 | 0.5836 | 0.5894 |
| Standard ResNet18 | Original | uncompressed | 3925 | 0.8665 | 0.8651 | 0.8684 |
| Standard ResNet18 | Stable Diffusion img2img | uncompressed | 3925 | 0.7977 | 0.7969 | 0.7993 |

## Visual-quality measurements

PSNR and SSIM measure distortion introduced by JPEG compression relative to the corresponding decoded JPEG image, not fidelity of a cartoonized image to the original photograph. Infinite PSNR values for identical images are reported separately where applicable. PNG-versus-JPEG file sizes should not be treated as a like-for-like compression comparison.

Full per-condition visual-quality summaries are in `metrics/visual_quality_all_four_summary.csv` and the consolidated table is `paper/tables/final_visual_quality_results.csv`.

## Statistical analyses

- Wilson confidence intervals for recognition accuracy.
- Paired McNemar tests with Holm correction for multiple comparisons.
- Paired bootstrap confidence intervals for accuracy differences.
- Per-class precision, recall, F1, and confusion matrices.
- Prediction-flip summaries across JPEG settings.
- Exploratory class-adjusted logistic associations between image descriptors and correctness.

Descriptor associations are observational and do not establish causal shape or texture bias. Prediction flips are changes in predicted class and do not necessarily mean a prediction became incorrect.

## Main interpretation

Recognition performance differs substantially by image condition. Cartoonization changes visual information relevant to the selected classifier, while JPEG compression effects vary by transformation and classifier. Conclusions should be based on the complete matched tables, confidence intervals, and paired tests rather than visual-quality scores alone.

## Reproducibility and limitations

- The original test images are physically stored under the dataset's `val/` directory; their evaluation role is determined by the saved split manifest. The split provenance must be described accurately.
- Stable Diffusion v1.5 image-to-image used 512×512 output, FP16, strength 0.45, guidance scale 7.5, 20 inference steps, and an empty prompt. The safety checker was disabled for this experiment and this must be disclosed in the methods.
- The current results do not include a three-seed replication, a second dataset, a second classifier architecture, LPIPS, or calibration metrics.
- The descriptor analysis is associative rather than causal. Novelty claims require a complete literature review.
- Verify source/model licenses, dataset attribution, software versions, checkpoint provenance, and manuscript references before submission.

## Validation audit

The recorded artifact audit contains 33 checks: 33 passed and 0 marked for review.

## Key artifacts

- `metrics/full_recognition_summary.csv`
- `metrics/visual_quality_all_four_summary.csv`
- `metrics/accuracy_wilson_summary.csv`
- `metrics/paired_mcnemar_holm_results.csv`
- `metrics/paired_bootstrap_accuracy_ci.csv`
- `metrics/prediction_flip_summary.csv`
- `metrics/per_class_metrics.csv`
- `metrics/class_adjusted_descriptor_logistic.csv`
- `FINAL_VALIDATION_AUDIT.csv`

Report generated from saved experiment outputs on 2026-10-17 07:29 UTC.