# Beyond Visual Quality: Cartoonization and Recognition Robustness Under JPEG Compression

## Overview

This repository evaluates how three image cartoonization methods and JPEG
compression affect visual measurements and image-classification performance.

### Experimental conditions

- Original images
- OpenCV cartoonization
- Pretrained AnimeGANv2
- Stable Diffusion v1.5 image-to-image transformation

### Dataset and classifiers

- Imagenette 320, ten classes
- 3,925 matched test images
- Standard ResNet18
- ResNet18 trained with random JPEG augmentation

### JPEG settings

Uncompressed, Q95, Q75, Q50, and Q25.

## Main evaluation outputs

- 157,000 predictions across two classifiers
- Top-1 accuracy, macro-F1, balanced accuracy, and mean confidence
- Per-class precision, recall, F1, and confusion matrices
- PSNR, SSIM, and file-size measurements
- Wilson confidence intervals
- Paired McNemar tests with Holm correction
- Paired bootstrap confidence intervals
- Prediction-flip analysis
- Exploratory class-adjusted descriptor/correctness associations

## Start here

- [Final results report](research/FINAL_RESULTS_SUMMARY.md)
- [Research figures](research/figures/)
- [All recognition results](research/paper/tables/final_recognition_results_all_conditions.csv)
- [Final visual-quality table](research/paper/tables/final_visual_quality_results.csv)
- [Final headline table](research/paper/tables/final_headline_results.csv)
- [Validation audit](research/manifests/FINAL_VALIDATION_AUDIT.csv)
- [Paper package](release/cartoonization_paper_package.zip)
- [Critical backup](release/cartoonization_critical_backup.zip)

## Important interpretation notes

PSNR and SSIM compare the image to its corresponding JPEG-decoded version;
they do not measure cartoonization fidelity to the original photograph.
Descriptor associations are observational and do not prove causal shape or
texture bias. Prediction changes are not necessarily errors.

Stable Diffusion settings included 512×512 output, FP16, strength 0.45,
guidance scale 7.5, 20 inference steps, and an empty prompt. The safety checker
was disabled and this should be disclosed in the methods.

## Limitations and remaining work

The reported experiment does not include three-seed replication, a second
dataset, a second classifier architecture, LPIPS, or calibration metrics.
Novelty claims require a thorough literature review. Check dataset attribution,
model licenses, checkpoint provenance, software versions, and citations
before submission.

## Reproducibility

The repository includes research tables, figures, manifests, prediction
files, and analysis outputs. Dataset images and model weights may need to be
downloaded separately. See the saved manifests and notebook for split and
processing details.
