# Saiswaroop's Research Contribution: Cross-Architectural Cartoonization Robustness

This folder contains the complete empirical evaluation, high-accuracy benchmarks, and architectural proposals developed by **Saiswaroop** as part of the collaborative Cartoonization Robustness Study.

---

## 🎯 Executive Summary & Core Results

While prior work evaluated cartoonization solely on convolutional backbones (ResNet18) and reported severe recognition drops (58.78%), this work demonstrates that **downstream recognition accuracy is fundamentally governed by classifier inductive bias**:
* **On Shape-Biased Vision Transformers (ViT-B/16):** OpenCV Clean achieves **93.93%** Top-1 accuracy, breaking the literature benchmark.
* **On Modern Hybrid Convolutions (ConvNeXt-Tiny):** OpenCV achieves **88.60%** and Stable Diffusion achieves **86.40%**.
* **LPIPS Perceptual Feature Drift:** Empirically measured on GPU:
  * OpenCV Mean LPIPS: `0.4186` (Closer low-level patch alignment)
  * AnimeGANv2 Mean LPIPS: `0.4518` (Generative artistic style shift)

---

## 📊 The Tri-Architectural Benchmark Matrix

| Condition | ResNet18 (Standard CNN) | ConvNeXt-Tiny (Modern Hybrid) | ViT-B/16 (Pure Transformer) | Primary Governing Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Original Photos (Baseline)** | 86.65% | 98.40% | **99.93%** | Global self-attention removes class ambiguity |
| **OpenCV Clean** | 58.78% | 88.60% | **93.93%** 🚀 | Preserves 1:1 macro-shape without generative artifacts |
| **AnimeGANv2 (Feed-Forward)** | 76.92% | 83.20% | **85.13%** 🚀 | Edge-loss outline preservation |
| **Stable Diffusion v1.5 (img2img)**| 79.77% | 86.40% | **89.20%** 🚀 | Pretrained latent VAE retains continuous textures |

---

## 🚀 The Proposed Architecture: CARP-Net

We propose **CARP-Net** (*Content-Aware, Robustness-Preserving Cartoonization Network*), a novel feed-forward generator designed to bridge the CNN texture gap:
* **Dual-Path Feature Encoder:** Decouples high-frequency contours from semantic tokens.
* **Gated Contour Attention:** Eliminates spurious edge noise ($C=7$, block size 15).
* **In-Loop DCT Quantization Regularization:** Trains with random JPEG perturbations to guarantee compression robustness.
* **Compound Loss Function:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{adv}} + \lambda_{\text{sem}} \mathcal{L}_{\text{sem}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}} + \lambda_{\text{dct}} \mathcal{L}_{\text{dct}}$$

---

## 📂 Folder Structure

```
Saiswaroop/
├── figures/
│   ├── fig1_tri_architectural_benchmark.png    # 3-architecture comparison chart
│   ├── fig2_jpeg_trajectory.png                # Uncompressed to Q25 accuracy curves
│   ├── fig3_per_class_f1_heatmap.png           # 10 official synset F1-scores
│   └── fig_carp_net_benchmark.png              # CARP-Net multi-classifier performance
├── scripts/
│   ├── models_carp_net.py                      # CARP-Net PyTorch architecture
│   ├── evaluate_benchmark.py                   # 10-class logit subsetting evaluator
│   └── evaluate_lpips.py                       # VGG perceptual loss evaluator
└── reports/
    ├── Global_Monograph_Cartoonization_Robustness.pdf
    └── Cartoonization_Proposed_Architecture_Action_Plan_Execution.pdf
```

---

## 🎓 Publication Deliverables
All reports are formatted with academic running headers, numbered pagination, formal statistical tables, and high-resolution figures in the `reports/` directory.
