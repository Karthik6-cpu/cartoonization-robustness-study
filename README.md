# Beyond Visual Quality: Cartoonization Robustness Study

## Overview
A comparative study of OpenCV cartoonization, pretrained AnimeGANv2,
and pretrained Stable Diffusion image-to-image transformation.

## Research evaluation
- Balanced 300-image subset of ten Imagenette classes
- ResNet18 recognition performance
- SSIM and PSNR image similarity metrics
- Gaussian blur, Gaussian noise, low illumination, and JPEG compression
- Per-class results, confusion matrices, and failure-case analysis

## Limitations
The Gaussian-noise results require matched-sample verification.
Failed Stable Diffusion outputs and differences in valid sample counts
must be reported transparently. SSIM and PSNR do not measure subjective
visual appeal. Novelty has not been established as unprecedented.

## Results
See the results directory for available CSV files, charts, and reports.

## Reproducibility
The experimental notebook, source code, model requirements, and dataset
instructions should accompany this repository. Dataset and model files
may need to be obtained separately.
