# Beyond Visual Quality: Evaluating the Robustness and Recognition Performance of Image Cartoonization Methods

## Overview
This study compares three image cartoonization methods:
- OpenCV cartoonization
- Pretrained AnimeGANv2
- Stable Diffusion v1.5 image-to-image transformation

A pretrained ResNet18 classifier evaluates recognition performance on original and transformed images.

## Dataset and evaluation
- Balanced subset of 300 images across ten Imagenette classes
- Clean-image matched analysis: 294 common images
- Corruptions: Gaussian noise, Gaussian blur, low illumination, and JPEG compression
- Recognition accuracy, Wilson 95% confidence intervals, paired McNemar tests with Holm correction
- SSIM/PSNR metrics and failure-case analysis

## Verified clean-image results
On the common matched subset (n=294):
| Method | Correct | Accuracy |
|---|---:|---:|
| Original images | 237/294 | 80.61% |
| AnimeGANv2 | 174/294 | 59.18% |
| Stable Diffusion v1.5 | 98/294 | 33.33% |
| OpenCV | 53/294 | 18.03% |

Six of 300 Stable Diffusion clean outputs were entirely black (2.0% output-failure rate). The 33.33% Stable Diffusion accuracy is conditional on the 294 valid outputs; the end-to-end correct rate over all 300 inputs is 32.67%.

## Stable Diffusion configuration
- Checkpoint: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`
- Pipeline: `StableDiffusionImg2ImgPipeline`
- Scheduler: PNDM (`PNDMScheduler`)
- Input resolution: 512 x 512
- FP16 weights, Safetensors, CUDA GPU, attention slicing
- Strength: 0.45; guidance scale: 7.5; inference steps: 20
- Positive and negative prompts are documented in the experimental notebook.
- Random-seed handling differs between clean and corruption experiments.

## Statistical analysis
Matched accuracy results and paired McNemar tests were independently checked against the saved prediction CSVs. Holm-adjusted p-values were verified for the saved comparisons. Refer to the results CSV files for condition-specific denominators and confidence intervals.

## Limitations
- Results apply to this dataset subset, the selected methods, and the chosen classifier.
- SSIM and PSNR do not fully represent perceptual quality.
- Black Stable Diffusion outputs are reported as failures rather than silently discarded.
- The exact cause of the black outputs has not been established.
- Research novelty has not been established as unprecedented.
- Dataset provenance/licensing, citations, and the final manuscript still require final review.

## Reproducibility
The notebook, `requirements.txt`, prediction files, and result tables are included where applicable. Dataset images and model weights may need to be downloaded separately. Exact library versions and checkpoint revision should be consulted alongside the notebook.

## Repository structure
- `01_opencv_cartoonization_pilot (1).ipynb`: experimental notebook
- `docs/`: research manuscript drafts
- `results/`: prediction tables and statistical audit outputs
