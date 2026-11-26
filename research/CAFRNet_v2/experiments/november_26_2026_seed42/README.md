# CAFR-Net v2 — Seed 42 metric record (26 November 2026)

This folder preserves the epoch metrics manually transcribed from the Kaggle training output for the November 26, 2026 run.

- Architecture: `cafrnet_v2`
- Seed: `42`
- Recorded epochs: 1–12
- Next intended epoch number: 13
- Hardware: NVIDIA Tesla T4
- Framework: PyTorch 2.11.0+cu130; torchvision 0.26.0+cu130

## Important checkpoint provenance

The November 26 run's `latest_model.pt` was not found in the replacement Kaggle session. Therefore, these files preserve metrics only; they do **not** restore model weights, optimizer/scheduler state, RNG state, or enable an exact resume at epoch 13.

A separately uploaded Kaggle dataset named `seed42` contains an earlier CAFR-Net v2 checkpoint whose metadata reports epoch 20. That is a distinct run and must not be treated as the missing November 26 epoch-12 checkpoint.

See `seed42_epochs_1_to_12.csv` for the tabular metrics and `seed42_epochs_1_to_12.json` for structured metadata.
