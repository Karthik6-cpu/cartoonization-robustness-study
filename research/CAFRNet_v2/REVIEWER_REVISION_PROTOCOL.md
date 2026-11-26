# CAFR-Net v2 — Reviewer Revision and Reproducibility Protocol

**Status:** experiment harness committed; new GPU runs are not yet executed. Do not report proposed experiments as results.

## 1. Audit of the OpenCV accuracy conflict

The current repository contains two distinct records:

| Artifact | OpenCV CAFR-Net accuracy | What it represents |
|---|---:|---|
| `CAFRNet_Three_Seed_Results.csv` | 95.159% (seed 42), 88.739% (seed 123), 87.312% (seed 2026) | Three reported CAFR-Net runs |
| `CAFRNet_Three_Seed_Summary.csv` | 90.403% mean; SD 4.180 pp | Arithmetic summary of the three rows above |
| `model_comparison_metrics.csv` | 78.115% CAFR-Net; 83.236% ConvNeXt | A separate comparison record; exact run/checkpoint/split provenance is not established by the CSV |

The three-seed mean is internally reproducible from its three seed rows. The 78.115% value is not an arithmetic alternative to that mean; it is a separate record. Both CSVs list 3,925 images, but equal row counts alone do not establish identical image identities, preprocessing, model checkpoint, edge implementation, or run configuration.

The committed validation notebook `Phase2_Colab_Conference_Validation_Pack (1).ipynb` also prints all four values and explicitly warns that the comparison CSV must not be treated as a definitive head-to-head until provenance is reconciled.

**Required action:** preserve both historical records. Recover the original comparison prediction CSV, checkpoint, run metadata, data manifest and evaluation code. Join by stable relative image path and label. If any of those artifacts cannot be recovered, mark the 78.115% comparison as *unverified historical result* and exclude it from headline claims. Do not overwrite it or relabel it as the three-seed result.

## 2. Existing ConvNeXt baseline status

The repository already has a `convnext.ipynb` and a `convnext_next_steps.ipynb`, and `model_comparison_metrics.csv` contains a ConvNeXt accuracy column. This means a ConvNeXt experiment was attempted previously. However, the current aggregate table does not provide enough traceable metadata to certify that it was trained with the exact CAFR-Net split, paired-view objective, augmentation probabilities, freeze schedule, epoch budget, and checkpoint selection.

The new plain ConvNeXt-Tiny control must therefore be rerun in a controlled protocol rather than assuming the historical row is a valid matched baseline.

## 3. Source-level reproducibility correction

The historical `source/cafr_net.py` applied Sobel filters directly to ImageNet-normalized input. A later Colab experiment notebook explicitly denormalized the input before computing luminance and Sobel edges. Those are not identical implementations.

To preserve provenance:
- The original implementation is archived at `source/legacy/cafr_net_pre_denormalization.py`.
- The active `source/cafr_net.py` now converts normalized input back to RGB [0,1] before grayscale/Sobel computation.
- New CAFR-Net v2 experiments must use the corrected active source and record its Git commit.
- Historical results trained with the earlier implementation remain historical and must not be silently relabeled as corrected-source results.

This correction does not change the tensor shapes or parameter names, but it changes the edge input values and therefore requires fresh training for a clean comparison.

## 4. Controlled Imagenette experiment matrix

### Models
1. CAFR-Net v2 (corrected edge input).
2. Plain pretrained ConvNeXt-Tiny (same backbone family, standard classifier head; no CAFR fusion, edge, or attention modules).
3. Pretrained ResNet-18 (secondary lightweight baseline).

### Seeds and training budget
- Seeds: 42, 123, 2026.
- Epochs: 20 for every model and seed.
- Input size: 224×224.
- Batch size: 32 where GPU memory permits; if reduced, use the same batch size for all three models and document it.
- Optimizer: AdamW, weight decay 0.05.
- Encoder learning rate: 2e-5.
- New-head learning rate: 2e-4.
- Freeze encoder for first 2 epochs; unfreeze from epoch 3.
- Cosine annealing over 20 epochs.
- Paired clean/shifted views, supervised cross-entropy on both views, plus 0.25 × symmetric KL consistency loss at temperature 2.
- Identical train/validation split and class order for all models within each seed.
- Select the best checkpoint using validation accuracy only. Do not use test results for model selection.

### Evaluation conditions
Evaluate every saved best checkpoint on exactly the same test identities and labels:
- Original.
- OpenCV cartoonization.
- AnimeGANv2.
- Stable Diffusion.
- JPEG Q25 and Q50.
- Gaussian blur.
- Gaussian noise.
- Brightness shift.

Store per-image predictions with architecture, seed, condition, relative path, true label, predicted label, confidence and correctness.

### Ablations
Run the full CAFR-Net and three variants (no edge, no semantic fusion, no attention) using all three seeds, same split, 20 epochs, augmentations and optimizer schedule. Report seed-wise results and mean ± SD. Do not infer component effects from the existing single-seed ablations alone.

## 5. Oxford-IIIT Pet external-dataset experiment

Use the same official test set and a fixed train/validation split for all models. Train:
- CAFR-Net v2 (corrected edge input).
- Plain ConvNeXt-Tiny.
- ResNet-18.

Use seeds 42, 123 and 2026 and the same 20-epoch training budget and paired-view objective. Evaluate original and OpenCV-cartoonized test views. Report accuracy, macro precision, macro recall, macro-F1, ECE, confidence and per-class metrics. Keep this experiment separate from Imagenette.

## 6. Commands

Run these from `research/CAFRNet_v2/source` in the checked-out repository. The data directory must contain `train/` and `val/` folders in ImageFolder format.

### Imagenette main comparison
For each seed in 42, 123, 2026, run all three commands (nine training runs total):

```bash
for SEED in 42 123 2026; do
  python train_architecture.py --arch cafrnet_v2 --data /content/cafr_data --out /content/cafr_runs/cafr_seed$SEED --seed $SEED --epochs 20 --batch-size 32
  python train_architecture.py --arch convnext_tiny --data /content/cafr_data --out /content/cafr_runs/convnext_seed$SEED --seed $SEED --epochs 20 --batch-size 32
  python train_architecture.py --arch resnet18 --data /content/cafr_data --out /content/cafr_runs/resnet_seed$SEED --seed $SEED --epochs 20 --batch-size 32
done
```

### Evaluation example
Use the same condition folders for every architecture and seed:

```bash
python evaluate_architecture.py \
  --checkpoint /content/cafr_runs/convnext_seed42/best_model.pt \
  --conditions original=/content/cafr_test opencv=/content/cafr_opencv \
              animegan2=/content/cafr_animegan2 stable_diffusion=/content/cafr_stable_diffusion \
              jpeg25=/content/cafr_stress_test/jpeg25 jpeg50=/content/cafr_stress_test/jpeg50 \
              blur=/content/cafr_stress_test/blur noise=/content/cafr_stress_test/noise \
              brightness=/content/cafr_stress_test/brightness \
  --out /content/cafr_runs/convnext_seed42/predictions.csv
```

Concatenate all prediction CSVs and analyze:

```bash
python analyze_architecture_comparison.py \
  --predictions /content/cafr_runs/all_predictions.csv \
  --outdir /content/cafr_runs/analysis
```

### Ablations
For each seed, train the three ablation variants using `--arch cafrnet_v2` and `--ablation no_edge`, `no_semantic_fusion`, and `no_attention`. Also retain the full CAFR-Net runs from the main matrix.

### Oxford-IIIT Pet
Repeat the three-model, three-seed matrix with the Oxford train/validation folders and then evaluate on the exact same original and OpenCV test folders for all models.


### One-command matrix launcher

The repository also includes `source/run_reviewer_matrix.sh`. From Colab, after cloning the branch and mounting the datasets, change into the source directory and run:

```bash
cd /content/cartoonization-robustness-study/research/CAFRNet_v2/source
bash run_reviewer_matrix.sh imagenette /content/cafr_data /content/cafr_runs
bash run_reviewer_matrix.sh oxford /content/oxford_pet_data /content/oxford_runs
```

The Imagenette mode runs 18 jobs (three main architectures plus three CAFR ablations, each across three seeds). Oxford mode runs the three main architectures across three seeds (nine jobs). If a Colab session is interrupted, rerun with `RESUME=1` to resume any job with a saved `latest_model.pt`:

```bash
RESUME=1 bash run_reviewer_matrix.sh imagenette /content/cafr_data /content/cafr_runs
```

## 7. Analysis and reporting rules

- Primary architecture question: does CAFR-Net v2 outperform *plain ConvNeXt-Tiny* under the same backbone, split, augmentation, objective, epoch budget and evaluation conditions?
- Secondary comparison: CAFR-Net v2 versus ResNet-18.
- Report per-seed scores, mean ± SD across seeds, and seed-level confidence intervals. Image-level bootstrap intervals are additional conditional uncertainty, not a replacement for independent training seeds.
- Use paired exact McNemar tests on aligned per-image predictions and apply Holm correction across the declared family of tests.
- Include macro-F1 and per-class results, not accuracy alone.
- Report parameter count, inference latency and peak memory using the same device, input size, warm-up, batch sizes and timing repetitions.
- Do not claim CAFR-Net's added modules caused gains unless the matched ConvNeXt control and multi-seed ablations support that conclusion.
- Do not call it state of the art or universally robust based on these experiments.

## 8. Current status

| Item | Status |
|---|---|
| Preserve and document conflicting historical metrics | Done in this protocol; historical run provenance still unresolved |
| Archive earlier Sobel implementation | Done |
| Correct active Sobel input normalization | Done |
| Unified matched training harness | Added; not yet GPU-executed |
| Unified evaluator with per-image IDs | Added; not yet GPU-executed |
| Multi-seed paired analysis script | Added; not yet GPU-executed |
| Re-run three-model Imagenette matrix | Pending Colab GPU execution |
| Re-run three-seed ablations | Pending Colab GPU execution |
| Run Oxford-IIIT Pet matched baselines | Pending Colab GPU execution |
| Update final results and paper tables | Only after the new runs finish and artifacts are checked |

**Important:** These scripts and the protocol are experimental infrastructure, not new empirical results. The revised manuscript must continue to label the old 90.403%, 78.115%, and 83.236% values according to their separate provenance status until the raw artifacts are recovered or new controlled runs replace them.
