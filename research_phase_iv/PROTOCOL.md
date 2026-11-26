# Phase IV — CAFR-Net research protocol

## Working name
**CAFR-Net: Cartoon-Aware Feature Refinement Network**

The name and architecture are provisional until the literature review, ablations,
and empirical evaluation establish a defensible contribution.

## Research objective
Improve semantic classification under appearance shifts caused by image
cartoonization and subsequent JPEG compression, while preserving performance
on original images.

## Architecture hypothesis
CAFR-Net uses a pretrained ConvNeXt-Tiny encoder and adds:
1. Multi-scale semantic projections from the 192-, 384-, and 768-channel stages.
2. A Sobel-based edge/contour stream intended to retain structural cues when
   texture and local appearance statistics change.
3. A global gradient-energy-conditioned soft gate over the three semantic
   streams and the edge stream.
4. Residual feature refinement and a classification head.

The model is a testable hypothesis, not a claim of novelty or improved accuracy.
Related work already studies domain generalization, feature disentanglement,
frequency augmentation, and multi-scale fusion. The exact combination and
its value for cartoonization/JPEG robustness must be established empirically.

## Controlled baseline
Train ResNet-18 and ConvNeXt-Tiny from ImageNet-1K pretrained weights under
the same training identities, input size, class mapping, augmentations,
optimizer family, epoch budget, checkpoint-selection rule, and random seeds.
Record any architecture-specific differences. Keep the historical Phase-I
ResNet-18 checkpoints as a separate continuity comparison, not as the only
controlled baseline.

## Split integrity
- Use the existing source-identity split manifest.
- No source identity may cross train/validation/test.
- Fit transforms and select checkpoints using training/validation only.
- The test split is evaluated once after protocol freeze.
- Generate cartoonized/compressed training views on-the-fly from training
  identities only, or use a separately versioned train-only transform cache.
- Do not use test-domain results to tune architecture or hyperparameters.

## Training strategy
Stage A: initialize the ConvNeXt-Tiny backbone with ImageNet-1K weights.
Stage B: train the classifier and new refinement modules with the backbone
  frozen for a short warm-up (e.g. 2 epochs).
Stage C: unfreeze the backbone and fine-tune end-to-end with discriminative
  learning rates (backbone 1e-5 to 3e-5; new modules/head 1e-4 to 3e-4).
Use AdamW, cosine decay, weight decay 0.05, mixed precision when supported,
and early stopping / best validation checkpoint. These are starting values
to tune on validation, not fixed empirical findings.

## Paired-view objective
For each training identity, sample a clean view and one training-only
appearance/compression view. Optimize:
L = L_CE(clean) + L_CE(shifted) + lambda_cons * L_sym-KL
where symmetric KL encourages matched predictions to remain consistent.
Start lambda_cons at 0.25 and compare 0.1, 0.25, 0.5 on validation only.
Never use held-out test identities for this training objective.

## Required experiments
1. ResNet-18 matched-training baseline.
2. ConvNeXt-Tiny matched-training baseline.
3. CAFR-Net full model.
4. CAFR-Net without edge stream.
5. CAFR-Net without frequency gate (uniform fusion).
6. CAFR-Net without paired consistency loss.
7. At least three seeds for the final model and key baselines if compute permits.

## Evaluation
Use identical held-out source identities and the existing appearance/JPEG
conditions. Report top-1 accuracy, macro-F1, per-class recall, calibration
(ECE), confidence, paired prediction flips, and confusion matrices.
Use paired bootstrap confidence intervals over source identities and
McNemar tests with Holm correction for paired model comparisons.
Report clean accuracy alongside every robustness metric.

## Primary outcomes
- Mean accuracy across cartoonized conditions.
- Cartoonization gap: original accuracy minus cartoonized accuracy.
- JPEG degradation within each appearance domain relative to its uncompressed
  counterpart.
- Clean-to-shifted consistency and worst-condition accuracy.

## Success criteria (pre-registered targets, not promises)
A useful candidate should improve mean cartoonized accuracy by at least
2 percentage points over matched ConvNeXt-Tiny, with a paired 95% CI for
the improvement above zero, while clean accuracy drops by no more than
1 percentage point. It should also show a repeatable direction across seeds.
If these criteria are not met, report the negative result and simplify or
redesign based on ablations rather than claiming success.

## Reproducibility artifacts
Save configuration, seed, split-manifest hash, package versions, checkpoint,
per-image logits/predictions, epoch history, and evaluation script outputs.
Do not merge BICR-Net's synthetic-corruption results with this experiment.

## Known limitations to resolve
- Verify exact transform implementations and parameters.
- Verify all Phase-I split and checkpoint provenance.
- Confirm the ConvNeXt notebook's completed outputs before reusing them.
- Obtain a second independent dataset before making broad generalization claims.
