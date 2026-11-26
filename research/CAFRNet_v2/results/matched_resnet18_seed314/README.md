# Matched ResNet-18 comparison — seed 314

## Scope

This directory contains aggregate results for a matched comparison between CAFR-Net v2 and a ResNet-18 baseline on Imagenette under six evaluation conditions.

- Test identities: 3,925.
- Conditions: original, JPEG Q25, JPEG Q50, blur, noise and brightness.
- Paired image-condition records per model: 23,550.
- Matching keys: `condition` and `relative_path`.
- Training seed: 314.
- Split provenance: reconstructed stratified split using `train_test_split(test_size=1421, random_state=42)`.

This experiment is separate from the earlier three-seed CAFR-Net evaluation (seeds 42, 123 and 2026). It must not be presented as a multi-seed matched comparison.

## Files

- `model_performance.csv` — accuracy and macro-F1 comparison.
- `paired_bootstrap_accuracy_CI.csv` — paired image-level percentile bootstrap intervals, 2,000 resamples.
- `mcnemar_holm_exact.csv` — exact paired McNemar p-values and Holm-adjusted p-values.
- `efficiency_benchmark.csv` — Tesla T4 latency, parameter count and incremental peak memory.

## Statistical interpretation

All six paired bootstrap 95% confidence intervals for the accuracy advantage (CAFR-Net minus ResNet-18) exclude zero. The bootstrap resamples matched test images and therefore quantifies test-image sampling uncertainty conditional on the trained checkpoints. It does not quantify training-seed variability.

All six exact two-sided McNemar tests remain significant after Holm correction. McNemar testing and bootstrap intervals answer related but distinct questions and should both be described accurately.

## Computational trade-off

CAFR-Net v2 has 30.281 million parameters compared with 11.182 million for ResNet-18. Its measured latency is higher at both batch sizes. Do not describe CAFR-Net v2 as lightweight or faster than ResNet-18.

Memory values are incremental peak memory above the measured baseline, not total GPU memory consumption.

## Reproduction artifacts still to be copied from Colab

The aggregate tables in this directory were written from the verified outputs reported for this experiment. The per-image prediction CSVs and generated figures are not included in this directory yet. Copy the corresponding files from the Colab run directory to the repository to complete the raw-artifact archive.

## Claim limits

The results support a higher observed accuracy for CAFR-Net v2 on this reconstructed Imagenette test split and these six conditions. They do not establish multi-seed superiority, universal corruption robustness, state-of-the-art performance or matched external-dataset generalization.
