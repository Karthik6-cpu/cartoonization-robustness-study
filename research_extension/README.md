# Compound-corruption pilot: instructions

## Purpose
This extension creates deterministic, source-matched image variants for a small robustness pilot. It does not train a model and does not calculate classification accuracy. Accuracy must be obtained by running an existing classifier checkpoint on the generated images.

## Conditions generated
- clean
- JPEG Q50
- additive Gaussian noise
- Gaussian blur
- brightness increase (1.25 multiplier)
- noise + blur
- JPEG Q50 + noise
- JPEG Q50 + blur

The script writes a `matched_corruption_manifest.csv` containing source identity, label, condition, output path and seed.

## Run

Install Pillow:

```bash
python -m pip install pillow
```

Run on a small subset of an already-held-out test folder:

```bash
python research_extension/compound_corruptions.py --input PATH_TO_TEST_SPLIT --output PATH_TO_PILOT_OUTPUT --max-images 20
```

After checking the generated images and manifest, increase to 100–300 images if runtime permits.

## Important protocol rules
1. Use only the existing held-out test split. Split the original source images before creating any transformed variants.
2. Keep the same source IDs in every condition. Do not treat transformed copies as independent samples.
3. Run the exact same classifier checkpoint and preprocessing pipeline for every condition.
4. Save per-image predictions with source ID, label, predicted class, confidence, condition and checkpoint identifier.
5. Report accuracy, macro-F1, balanced accuracy and paired prediction transitions. Bootstrap by source identity.
6. If inference cannot be completed, report only that the matched corruption dataset was generated. Do not invent model metrics.

## Pilot implementation details
Noise is additive per-pixel Gaussian noise with sigma 12 on 8-bit RGB values. Blur uses PIL GaussianBlur radius 1.2. Brightness uses a 1.25 multiplier. JPEG quality is 50. For JPEG+noise the order is JPEG then noise; for JPEG+blur it is blur then JPEG. These are simple pilot settings, not a standardized benchmark.

This initial pilot does not itself include cartoonization. To examine cartoonization+JPEG, provide an already-generated cartoonized test set as input and preserve original source IDs in the manifest. Ensure that the same source identity is paired across appearance domains.

## Reporting
Call this a preliminary matched compound-corruption pilot. Do not claim a new trained architecture, general real-world robustness, statistical significance without paired tests, or superiority over a baseline based on generated images alone.
