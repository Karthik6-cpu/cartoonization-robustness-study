# CAFR-Net v2 — Additional Experimental Results

## Package update

Updated: 2026-11-26T09:07:47

Dataset: Imagenette2-320 official validation set, 3,925 images.

## 1. OpenCV cartoonization

All six models were evaluated on OpenCV-stylized images using
sigma_s=60 and sigma_r=0.45.

| model                      |   correct |   total |   accuracy_percent |
|:---------------------------|----------:|--------:|-------------------:|
| ConvNeXt_Tiny              |      3896 |    3925 |            99.2611 |
| CAFRNet_Full               |      3889 |    3925 |            99.0828 |
| CAFRNet_No_Semantic_Fusion |      3881 |    3925 |            98.879  |
| CAFRNet_No_Edge            |      3870 |    3925 |            98.5987 |
| CAFRNet_No_Attention       |      3850 |    3925 |            98.0892 |
| ResNet18                   |      3735 |    3925 |            95.1592 |

ConvNeXt-Tiny achieved the highest accuracy in this experiment.
CAFR-Net Full outperformed the three tested CAFR-Net ablations and ResNet18,
but did not outperform ConvNeXt-Tiny.

## 2. Paired statistical comparison

CAFR-Net Full versus ConvNeXt-Tiny was compared over seven conditions.
The analysis used exact McNemar tests, paired bootstrap confidence intervals
with 3,000 resamples, and Holm correction.

| condition                           |   images |   CAFR_accuracy_% |   ConvNeXt_accuracy_% |   difference_CAFR_minus_ConvNeXt_pp |   CAFR_only_correct |   ConvNeXt_only_correct |   discordant_pairs |   McNemar_p |   bootstrap_95CI_low_pp |   bootstrap_95CI_high_pp |   Holm_adjusted_p | significant_0.05   |
|:------------------------------------|---------:|------------------:|----------------------:|------------------------------------:|--------------------:|------------------------:|-------------------:|------------:|------------------------:|-------------------------:|------------------:|:-------------------|
| clean                               |     3925 |           99.7962 |               99.8981 |                           -0.101911 |                   1 |                       5 |                  6 |  0.21875    |               -0.229299 |                0.0254777 |         1         | False              |
| noise_sigma20                       |     3925 |           99.414  |               99.3121 |                            0.101911 |                  18 |                      14 |                 32 |  0.596615   |               -0.178344 |                0.407643  |         1         | False              |
| blur_radius2                        |     3925 |           98.2675 |               98.5732 |                           -0.305732 |                  24 |                      36 |                 60 |  0.155002   |               -0.687898 |                0.0764331 |         0.930011  | False              |
| jpeg_quality20                      |     3925 |           99.2611 |               99.3631 |                           -0.101911 |                   8 |                      12 |                 20 |  0.503445   |               -0.33121  |                0.101911  |         1         | False              |
| brightness_050                      |     3925 |           99.7452 |               99.8471 |                           -0.101911 |                   2 |                       6 |                  8 |  0.289062   |               -0.254777 |                0.0254777 |         1         | False              |
| brightness_150                      |     3925 |           99.6178 |               99.8981 |                           -0.280255 |                   2 |                      13 |                 15 |  0.00738525 |               -0.484076 |               -0.101911  |         0.0516968 | False              |
| opencv_cartoon_sigma_s60_sigma_r045 |     3925 |           99.0828 |               99.2611 |                           -0.178344 |                  15 |                      22 |                 37 |  0.324009   |               -0.484076 |                0.127389  |         1         | False              |

Number of conditions significant after Holm correction:
**0 of 7**.

No condition crossed the 0.05 adjusted significance threshold.

## 3. High-severity corruption tests

| model         | condition      |   correct |   total |   accuracy_percent |   seconds_approx |
|:--------------|:---------------|----------:|--------:|-------------------:|-----------------:|
| CAFRNet_Full  | noise_sigma40  |      3889 |    3925 |            99.0828 |             30   |
| CAFRNet_Full  | blur_severe    |      3914 |    3925 |            99.7197 |             12   |
| CAFRNet_Full  | jpeg_quality10 |      3903 |    3925 |            99.4395 |             12   |
| ConvNeXt_Tiny | noise_sigma40  |      3886 |    3925 |            99.0064 |             29.4 |
| ConvNeXt_Tiny | blur_severe    |      3919 |    3925 |            99.8471 |             13.8 |
| ConvNeXt_Tiny | jpeg_quality10 |      3908 |    3925 |            99.5669 |             15   |

These are controlled severe stress tests:
Gaussian noise sigma=40, Gaussian blur kernel 9x9, and JPEG quality 10.
They are not a full ImageNet-C evaluation.

## 4. Inference efficiency

| model         |   parameters |   trainable_parameters |   batch1_latency_ms |   throughput_images_sec |   peak_gpu_memory_MB |
|:--------------|-------------:|-----------------------:|--------------------:|------------------------:|---------------------:|
| CAFRNet_Full  |     30281441 |               30281441 |            10.7209  |                 93.2756 |             149.122  |
| ConvNeXt_Tiny |     27827818 |               27827818 |             7.18373 |                139.203  |             139.742  |
| ResNet18      |     11181642 |               11181642 |             3.15837 |                316.619  |              76.8037 |

Measurements were obtained on the current Kaggle GPU environment using
batch size 1, 20 warm-up iterations and 100 timed iterations.
Peak GPU memory is the maximum allocated memory observed during the benchmark.
These measurements are hardware- and software-environment-specific.

## 5. Existing research results

The original research package and its existing standard-corruption,
clean-validation, baseline and ablation results have been preserved.
This update adds the new experiments; it does not replace historical results.

## 6. Outstanding work

- Investigate the historical OpenCV discrepancy: 78.115% versus 90.403%.
- Evaluate AnimeGANv2 and Stable Diffusion cartoonization domains.
- Run CAFR-Net Full and ConvNeXt-Tiny with seeds 123 and 314.
- Evaluate a matched ConvNeXt-Tiny baseline on Oxford-IIIT Pet.
- Consider scratch-trained multi-seed ablations.
- Consider ConvNeXt+CBAM and ConvNeXt+AugMix.
- Consider full ImageNet-C and PACS evaluations.

## 7. Research interpretation

The current evidence demonstrates strong performance by CAFR-Net Full on
the tested Imagenette conditions. ConvNeXt-Tiny remains slightly stronger
on the standard corruption mean and OpenCV cartoonization evaluation.
Paired statistical testing does not establish a significant difference
after multiple-comparison correction. CAFR-Net Full has higher measured
inference cost than ConvNeXt-Tiny and ResNet18 in the current benchmark.

These results should not be described as proof of general superiority
or as a completed cross-dataset generalization study.
