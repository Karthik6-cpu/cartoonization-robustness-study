# Image Cartoonization, JPEG Compression and Compositional Robustness

**Research repository and experimental record**  
**Researcher:** Karthikeya Satya Sai Pavan Challapalli  
MCA (Artificial Intelligence & Machine Learning), Department of Computer Applications  
Vignan's Foundation for Science and Technology Research (VFSTR), Guntur, Andhra Pradesh, India

This repository documents two related but methodologically distinct research tracks:

1. **Phase I — Cartoonization and JPEG robustness:** a source-matched study of three cartoonization pipelines followed by JPEG recompression, evaluated using two saved ResNet-18 checkpoints.
2. **Phase III — BICR-Net:** a separate prototype investigating blind degradation-factor estimation and interactions under combinations of synthetic image corruptions.

The tracks are discussed together as a research-development programme, but their datasets, objectives, corruption protocols and evaluation provenance must not be conflated.

---

## 1. Phase I: Cartoonization and JPEG robustness

### Research question

How do different image cartoonization methods affect semantic image classification, and how does subsequent JPEG compression interact with those appearance changes when the same source images are compared?

### Experimental design

| Component | Description |
|---|---|
| Dataset | Imagenette-320, 10 classes |
| Matched evaluation identities | 3,925 |
| Appearance domains | Original, OpenCV cartoonization, AnimeGANv2, Stable Diffusion v1.5 image-to-image |
| JPEG states | Uncompressed, Q95, Q75, Q50, Q25 |
| Classifiers | Standard ResNet-18 and saved JPEG-augmented ResNet-18 checkpoint |
| Prediction records | 157,000 (= 3,925 × 4 × 5 × 2) |

The saved split manifest is authoritative for source roles: 8,048 training, 1,421 validation and 3,925 testing identities. The 157,000 prediction records are repeated observations of 3,925 matched sources, not 157,000 independent images.

### Main reported results

Standard ResNet-18 uncompressed Top-1 accuracy:

| Appearance | Accuracy |
|---|---:|
| Original | 86.65% |
| OpenCV | 58.78% |
| AnimeGANv2 | 76.92% |
| Stable Diffusion | 79.77% |

At JPEG Q25, Standard ResNet-18 accuracy was 86.19% (Original), 60.92% (OpenCV), 77.07% (AnimeGANv2) and 79.52% (Stable Diffusion). The OpenCV result increased by 2.14 percentage points relative to its uncompressed condition; this is a condition-specific observation, not evidence that JPEG generally improves images.

The saved analysis reports 62 paired McNemar tests with Holm correction, paired bootstrap intervals, prediction-flip analysis, per-class metrics, PSNR and SSIM. OpenCV had the largest reported Q25 prediction-flip rate (306/3,925; 7.80%) for the Standard checkpoint.

### Interpretation

- Cartoonization caused larger recognition differences than JPEG compression in this matched benchmark.
- Stable Diffusion had the highest cartoon-domain Top-1 accuracy among the three cartoonization methods for Standard ResNet-18.
- OpenCV produced the largest recognition gap from original photographs.
- JPEG effects depended on appearance domain and checkpoint; they were not uniformly monotonic.
- PSNR and SSIM describe JPEG reconstruction relative to each appearance's own uncompressed image. They do not measure semantic preservation or human preference.

### Start here: Phase I artifacts

- [Final results summary](research/FINAL_RESULTS_SUMMARY.md)
- [Research figures](research/figures/)
- [All recognition results](research/paper/tables/final_recognition_results_all_conditions.csv)
- [Visual-quality results](research/paper/tables/final_visual_quality_results.csv)
- [Headline results](research/paper/tables/final_headline_results.csv)
- [Validation audit](research/manifests/FINAL_VALIDATION_AUDIT.csv)
- [Paper package](release/cartoonization_paper_package.zip)
- [Critical backup](release/cartoonization_critical_backup.zip)

---

## 2. Phase III: BICR-Net prototype

**BICR-Net** means *Blind Interaction-aware Compositional Robustness Network*. The documented architecture contains:

1. ResNet-18 feature encoder;
2. blind degradation factorizer for noise, blur and brightness;
3. pairwise latent interaction module;
4. residual feature correction;
5. LayerNorm and classification head.

### Documented corruption protocol

| Condition | Training status |
|---|---|
| Clean | Seen |
| Noise | Seen |
| Blur | Seen |
| Brightness | Seen |
| Noise + brightness | Seen |
| Noise + blur | Seen |
| Blur + brightness | Held out |
| Noise + blur + brightness | Held out |

The prototype used 160×160 RGB inputs, ResNet-18 trained from scratch, AdamW (learning rate 0.0003), six epochs and a recorded seed of 42. The reported parameter count is 11,726,353.

### Current findings and limitations

The saved three-seed evaluation summaries show BICR-Net below the matched ResNet-18 summary in every listed condition, by approximately 2.4–3.0 percentage points. Factor diagnostics are uneven, particularly brightness recall on held-out combinations. The documented inference-time interaction ablation is not a retrained module ablation. The reported latency and parameter counts do not establish an efficiency advantage.

**Critical provenance warning:** the experiment configuration records validation evaluation and seed 42, while the limitations record says an independent test evaluation was not performed because a separate test loader was unavailable. Some later CSV files imply multiple seeds and 3,925 examples per condition, and older independent-test CSVs differ. Until checkpoint, seed, split, dataloader and result-file provenance are reconciled, these values must be called *saved evaluation summaries*, not verified independent-test results.

BICR-Net is a prototype, not a validated solution to general real-world distribution shifts. No claim of state-of-the-art performance or superiority over ResNet-18 is supported by the current records.

### Start here: BICR-Net artifacts

- [Architecture and method](BICR_Net_Research/ARCHITECTURE_AND_METHOD.md)
- [Results and limitations](BICR_Net_Research/RESULTS_AND_LIMITATIONS.md)

---


## 3. Phase IV: CAFR-Net v2 — Cartoon-Aware Feature Refinement Network

CAFR-Net v2 investigates feature refinement for image classification under
cartoonization-induced domain shifts.

### Research objectives

- Improve semantic feature representations across visual domains.
- Investigate the contribution of edge information to cartoonized image recognition.
- Evaluate multi-scale semantic feature fusion.
- Study channel-spatial attention for feature refinement.
- Examine generalization on an external image dataset.

### Architecture

1. **ConvNeXt-Tiny backbone:** extracts hierarchical visual representations.
2. **Multi-scale feature projections:** align features from different backbone stages.
3. **Semantic scale fusion:** combines multi-scale semantic information.
4. **Edge refinement:** incorporates Sobel-based edge information.
5. **Channel-spatial attention:** refines informative channels and spatial regions.
6. **Classification head:** generates final class predictions.

### Experimental evaluation

**Imagenette**

- 10-class image classification benchmark.
- 13,394 verified images.
- Training: 8,048 images.
- Validation: 1,421 images.
- Testing: 3,925 images.
- Three independent seeds: 42, 123 and 2026.
- Evaluation domains: Original, OpenCV, AnimeGANv2 and Stable Diffusion.
- Ablations: without edge refinement, semantic fusion and attention.

**Oxford-IIIT Pet**

- 37-class external dataset.
- Single-seed evaluation using seed 42.
- Original and OpenCV-cartoonized test images.

### Three-seed Imagenette results

| Evaluation domain | Mean accuracy |
|---|---:|
| Original | 99.771% |
| OpenCV | 90.403% |
| AnimeGANv2 | 98.735% |
| Stable Diffusion | 94.726% |

These results describe the evaluated Imagenette experiments. They should not
be interpreted as evidence of universal robustness. OpenCV performance also
shows seed sensitivity.

### Research package

- [CAFR-Net v2 documentation](research/CAFRNet_v2/README.md)
- [Consolidated results](research/CAFRNet_v2/RESULTS_SUMMARY.md)
- [Source code](research/CAFRNet_v2/source/)
- [Experimental results](research/CAFRNet_v2/results/)
- [Dataset manifests](research/CAFRNet_v2/manifests/)

### Limitations

- Oxford-IIIT Pet evaluation uses a single seed.
- Component ablations use single-seed experiments.
- Results on Imagenette and Oxford-IIIT Pet must be reported separately.
- Further independent-dataset experiments are needed before making broader
  generalization claims.

## 4. Quick research extension: matched compound-corruption pilot

A deterministic image-generation utility and protocol have been added under [research_extension](research_extension/README.md).

- [Compound corruption generator](research_extension/compound_corruptions.py)
- Pilot conditions: clean, JPEG Q50, Gaussian noise, Gaussian blur, brightness increase, noise+blur, JPEG Q50+noise and JPEG Q50+blur.
- The generator writes a source-matched manifest so the same source identities can be evaluated across conditions.

Example:

```bash
python -m pip install pillow
python research_extension/compound_corruptions.py --input PATH_TO_TEST_SPLIT --output PATH_TO_PILOT_OUTPUT --max-images 200
```

Use only an already-held-out test split. Do not generate variants before splitting the original dataset. Run the model's existing inference pipeline on the generated images to obtain predictions; the generator itself does **not** train a model or produce accuracy results. See the extension README for condition definitions and the time-boxed protocol.

---

## 5. Research journal structure

The integrated research journal consolidates the experimental record and includes:

- background, research questions and dataset/split accounting;
- transformation and JPEG protocols;
- Phase I accuracy trajectories, PSNR/SSIM, paired statistics and prediction flips;
- exploratory ViT/ImageWoof work and conceptual CARP-Net/TCA-Net directions, clearly separated from validated results;
- BICR-Net architecture, synthetic corruption protocol, training configuration, saved evaluation summaries and factor diagnostics;
- expanded literature review, research-gap mapping and defensible contribution statements;
- mathematical formulations for classification metrics, JPEG/DCT, PSNR, SSIM, McNemar testing, bootstrap intervals and the conceptual BICR-Net pipeline;
- diagrams, figure interpretation, limitations, validity threats and a publication-readiness checklist.

The journal distinguishes measured results from conceptual proposals and flags unresolved provenance rather than filling gaps with invented results.

---

## 6. Reproducibility and publication checklist

Before submitting a paper or claiming generalization:

1. Preserve the exact source-ID and split manifests; verify no source overlap.
2. Archive checkpoints, commit hashes, environment versions and per-image predictions.
3. Verify Stable Diffusion checkpoint hash, sampler, seed and preprocessing.
4. Record JPEG library/version, quality, chroma subsampling and colour conversion.
5. Reconcile every BICR-Net seed, checkpoint, evaluation script, dataloader and split.
6. Re-run BICR-Net on a genuinely independent test split not used for checkpoint selection.
7. Conduct retrained ablations with matched training budgets.
8. Report confidence intervals, per-class metrics and paired source-level tests.
9. Verify bibliographic metadata, licenses, dataset attribution and exact pretrained model citations.

## 7. Scope and claim discipline

- Phase I is a matched evaluation of appearance transformations and JPEG compression; it is not a new cartoonization generator.
- Phase III is a separate synthetic compositional-corruption prototype; it is not a direct continuation of the Phase I benchmark.
- CARP-Net and TCA-Net remain conceptual directions unless separately implemented and validated.
- Exploratory ViT, ImageWoof and small LPIPS pilots must not be presented as directly comparable to the main matched ResNet-18 experiment.
- Do not claim “first ever,” state-of-the-art performance, BICR-Net superiority or verified independent-test performance without supporting evidence.

---

**Repository:** [Karthik6-cpu/cartoonization-robustness-study](https://github.com/Karthik6-cpu/cartoonization-robustness-study)
