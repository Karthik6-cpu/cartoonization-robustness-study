# CAFR-Net Experimental Results Summary

## Experimental setup
- Dataset: Imagenette, 10 classes.
- Training images: 8,048.
- Validation images: 1,421.
- Test images per condition: 3,925.
- Test conditions: original, JPEG25, JPEG50, blur, noise, brightness.
- Matched baseline: pretrained ResNet18.
- Matched comparison seed: 314.
- Hardware: NVIDIA Tesla T4.
- Input resolution: 224 x 224.

## Classification performance
| condition   | model    |   accuracy |   macro_f1 |   samples |   accuracy_drop_pp |
|:------------|:---------|-----------:|-----------:|----------:|-------------------:|
| original    | CAFR-Net |   0.997962 |   0.997962 |      3925 |           0        |
| original    | ResNet18 |   0.981401 |   0.981374 |      3925 |           0        |
| jpeg25      | CAFR-Net |   0.997197 |   0.997211 |      3925 |           0.076433 |
| jpeg25      | ResNet18 |   0.977834 |   0.977842 |      3925 |           0.356688 |
| jpeg50      | CAFR-Net |   0.997707 |   0.997722 |      3925 |           0.025478 |
| jpeg50      | ResNet18 |   0.978599 |   0.978598 |      3925 |           0.280255 |
| blur        | CAFR-Net |   0.99465  |   0.994637 |      3925 |           0.33121  |
| blur        | ResNet18 |   0.961529 |   0.961618 |      3925 |           1.98726  |
| noise       | CAFR-Net |   0.996943 |   0.996926 |      3925 |           0.101911 |
| noise       | ResNet18 |   0.972994 |   0.97302  |      3925 |           0.840764 |
| brightness  | CAFR-Net |   0.997962 |   0.997957 |      3925 |           0        |
| brightness  | ResNet18 |   0.974013 |   0.973976 |      3925 |           0.738854 |

## Paired statistical testing
| condition   |   CAFR_only_correct |   ResNet_only_correct |   discordant_pairs |   exact_p_value |   holm_adjusted_p | significant_after_holm   |
|:------------|--------------------:|----------------------:|-------------------:|----------------:|------------------:|:-------------------------|
| original    |                  67 |                     2 |                 69 |               0 |                 0 | True                     |
| jpeg25      |                  78 |                     2 |                 80 |               0 |                 0 | True                     |
| jpeg50      |                  79 |                     4 |                 83 |               0 |                 0 | True                     |
| blur        |                 138 |                     8 |                146 |               0 |                 0 | True                     |
| noise       |                  97 |                     3 |                100 |               0 |                 0 | True                     |
| brightness  |                  97 |                     3 |                100 |               0 |                 0 | True                     |

## Computational efficiency
| model    |   parameters_M |   trainable_parameters_M |   batch1_latency_ms |   batch1_peak_extra_memory_MB |   batch32_latency_ms |   batch32_peak_extra_memory_MB |
|:---------|---------------:|-------------------------:|--------------------:|------------------------------:|---------------------:|-------------------------------:|
| CAFR-Net |        30.2814 |                  30.2814 |             15.9754 |                       11.4844 |             154.168  |                          369.5 |
| ResNet18 |        11.1816 |                  11.1816 |              3.6087 |                       26.4785 |              33.4944 |                          196   |

## Interpretation
- CAFR-Net achieved higher accuracy and macro-F1 than ResNet18 in all six conditions.
- All six paired comparisons were significant after Holm correction.
- Blur produced the largest accuracy advantage for CAFR-Net.
- CAFR-Net has more parameters and higher inference latency.
- Results indicate a robustness-computation trade-off.

## Limitations and reporting cautions
- The matched CAFR-Net versus ResNet18 comparison uses one seed (314).
- The training/validation split was reconstructed and must be identified as such.
- Corruption tests are synthetic; they do not alone establish generalization to all real-world cartoon domains.
- The efficiency measurements are hardware- and implementation-dependent.
- The edge-correction comparison showed no change in predicted classes; it is a negative result, not an improvement claim.
- Further independent-seed and external-domain validation would strengthen the conclusions.