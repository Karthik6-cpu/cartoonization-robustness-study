# CAFR-Net v2 Research Package

**CAFR-Net:** Cartoon-Aware Feature Refinement Network

This directory contains the CAFR-Net v2 architecture, training/evaluation source, dataset manifests, experiment outputs and research documentation.

## Research scope

CAFR-Net v2 combines a ConvNeXt-Tiny hierarchical encoder with multi-scale feature projection and semantic fusion, Sobel-based edge refinement, channel-spatial attention and a classification head. The training procedure uses paired views and a consistency objective. Consult the source scripts and run metadata for the exact implementation and settings.

### Experiments kept distinct

1. **Three-seed Imagenette cartoon-domain evaluation:** seeds 42, 123 and 2026; original, OpenCV, AnimeGANv2 and Stable Diffusion domains.
2. **Oxford-IIIT Pet external-dataset evaluation:** one CAFR-Net seed (42), original and OpenCV-cartoonized test conditions.
3. **Matched Imagenette ResNet-18 comparison:** a separate reconstructed split, seed 314, six matched conditions, paired statistical tests and a T4 efficiency benchmark.

Do not combine these as if they shared an identical split, seed protocol or baseline experiment.

## Main three-seed Imagenette results

| Evaluation domain | Mean accuracy |
|---|---:|
| Original | 99.771% |
| OpenCV | 90.403% |
| AnimeGANv2 | 98.735% |
| Stable Diffusion | 94.726% |

These results are the earlier three-seed CAFR-Net evaluation. The OpenCV result has notable seed variability (SD 4.180 percentage points).

## Matched ResNet-18 comparison (seed 314)

The matched comparison contains 3,925 test identities across six conditions (23,550 image-condition records per model). The prediction files were checked for one-to-one matching using `condition` and `relative_path`.

| Condition | CAFR-Net v2 accuracy | ResNet-18 accuracy | Difference (pp) |
|---|---:|---:|---:|
| Original | 99.796% | 98.140% | +1.656 |
| JPEG Q25 | 99.720% | 97.783% | +1.936 |
| JPEG Q50 | 99.771% | 97.860% | +1.911 |
| Blur | 99.465% | 96.153% | +3.312 |
| Noise | 99.694% | 97.299% | +2.395 |
| Brightness | 99.796% | 97.401% | +2.395 |

Paired bootstrap confidence intervals (2,000 image-level resamples) were positive in all six conditions. Exact McNemar tests with Holm correction were also significant in all six conditions. The bootstrap quantifies test-image sampling uncertainty; it does not quantify training-seed variability.

### Efficiency trade-off

On an NVIDIA Tesla T4 with 224×224 inputs, CAFR-Net v2 measured 15.975 ms per batch-1 inference and 154.168 ms per batch of 32. ResNet-18 measured 3.609 ms and 33.494 ms, respectively. CAFR-Net v2 has 30.281 M parameters versus 11.182 M for ResNet-18. The accuracy gain therefore comes with a compute and parameter-count cost.

**Split provenance:** the seed-314 matched comparison uses a reconstructed stratified split with `test_size=1421` and `random_state=42`. It is distinct from the earlier three-seed experiment. Report this clearly in any manuscript.

## Oxford-IIIT Pet

- 37 classes; 7,349 images.
- Train: 2,944; validation: 736; official test: 3,669.
- Single CAFR-Net seed: 42.
- Original accuracy: 94.603%; OpenCV accuracy: 86.781%.
- Original macro-F1: 94.552%; OpenCV macro-F1: 86.660%.
- This is single-seed external-dataset evidence. A matched ResNet-18 comparison on Oxford-IIIT Pet has not been completed.

## Ablation study

The no-edge, no-semantic-fusion and no-attention ablations were evaluated using seed 42. Treat these as single-seed component diagnostics, not multi-seed inferential evidence.

## Directory structure

- `source/` — model, training and evaluation scripts.
- `manifests/` — dataset split manifests and metadata.
- `results/imagenette/` — earlier three-seed and ablation results.
- `results/oxford_pet/` — Oxford-IIIT Pet outputs.
- `results/matched_resnet18_seed314/` — matched baseline aggregate metrics and statistical analysis.
- `results/figures/` and `results/tables/` — existing plots and tables.
- `RESULTS_SUMMARY.md` — consolidated findings and caveats.
- `package_inventory.json` — inventory for the earlier package; regenerate it when files change.

## Reproducibility and claim discipline

- Preserve model checkpoints, per-image predictions, split manifests, run metadata, software versions and source commit identifiers.
- Keep the three-seed results separate from the seed-314 matched baseline.
- Do not claim multi-seed superiority from the matched comparison.
- Do not describe CAFR-Net v2 as lightweight or faster than ResNet-18.
- Do not claim universal robustness, state-of-the-art performance or first-ever novelty.
- Report Oxford-IIIT Pet separately and disclose its single-seed protocol.

See [the consolidated results summary](RESULTS_SUMMARY.md) and [matched comparison protocol](results/matched_resnet18_seed314/README.md).

## Final Colab artifacts and checkpoints

Large model files are distributed through the [CAFR-Net v2 final Colab artifact release](https://github.com/Karthik6-cpu/cartoonization-robustness-study/releases/tag/cafrnet-v2-colab-artifacts-2026-11-24) rather than committed as ordinary Git blobs.

Release contents include:
- Original seed-314 CAFR-Net artifact archive and best/latest checkpoints.
- Edge-corrected seed-314 CAFR-Net artifact archive and best/latest checkpoints.
- Matched ResNet-18 seed-314 latest checkpoint.

The dated runtime backup contains the notebook, predictions, run histories, metadata, result tables, figures and ZIP packages: [colab_session_backup](colab_session_backup/20261124_161028/).

The checkpoint names explicitly distinguish original versus edge-corrected CAFR-Net. Use the matching archive and checkpoint together when reproducing a run.
