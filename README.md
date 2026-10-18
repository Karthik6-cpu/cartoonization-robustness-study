# Cartoonization Robustness Study: Multi-Tier Compression and Cross-Architectural Benchmarking

This repository contains the collaborative empirical evaluation of cartoonization models (OpenCV, AnimeGANv2, Stable Diffusion v1.5, and the proposed CARP-Net) across progressive JPEG quantization and diverse classifier architectures.

## 👥 Contributors & Workspaces
* **`Saiswaroop/`**: Cross-architectural Vision Transformer (ViT-B/16) robustness evaluation, empirical LPIPS perceptual drift benchmarks, executable Kaggle notebooks, publication figures, and the novel CARP-Net proposed architecture.
* **Karthik's Research**: ResNet18 baseline pipelines, initial 300-image pilot & 3,925-image evaluations under composite corruptions, and JPEG augmentation analysis.

## 📄 Core Monograph & Reports
* Complete Four-Stage Monograph: `Saiswaroop/reports/Global_Monograph_Cartoonization_Robustness.pdf`
* Architecture Action Plan Execution: `Saiswaroop/reports/Cartoonization_Proposed_Architecture_Action_Plan_Execution.pdf`
* Accuracy Comparison Report: `Saiswaroop/reports/Cartoonization_Accuracy_Comparative_Report.pdf`
* Comprehensive Evaluation: `Saiswaroop/reports/Comprehensive_Cartoonization_Evaluation_Report.pdf`

## 📓 Notebooks & Scripts
* `Saiswaroop/notebooks/cartoon-gan.ipynb`: Original Kaggle GPU benchmark session.
* `Saiswaroop/notebooks/Kaggle_Cartoonization_Benchmark.ipynb`: Standalone executable notebook.
* `Saiswaroop/scripts/models_carp_net.py`: Complete PyTorch implementation of CARP-Net.
* `Saiswaroop/scripts/evaluate_benchmark.py`: ViT-B/16 10-class logit subsetting evaluator.
* `Saiswaroop/scripts/evaluate_lpips.py`: VGG perceptual feature drift calculator.
