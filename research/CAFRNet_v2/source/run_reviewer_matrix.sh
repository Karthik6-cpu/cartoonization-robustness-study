#!/usr/bin/env bash
# Run the controlled three-model, three-seed experiment matrix.
# Usage:
#   bash run_reviewer_matrix.sh imagenette /content/cafr_data /content/cafr_runs
#   bash run_reviewer_matrix.sh oxford /content/oxford_pet_data /content/oxford_runs
# Optional environment:
#   EPOCHS=20 BATCH_SIZE=32 WORKERS=2 RESUME=1
set -euo pipefail

MODE="${1:?mode must be imagenette or oxford}"
DATA_ROOT="${2:?provide DATA_ROOT with train/ and val/}"
OUT_ROOT="${3:?provide output directory}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-2}"
RESUME="${RESUME:-0}"

if [[ "$MODE" != "imagenette" && "$MODE" != "oxford" ]]; then
  echo "Unsupported mode: $MODE (use imagenette or oxford)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
test -d "$DATA_ROOT/train" && test -d "$DATA_ROOT/val"

run_one () {
  local arch="$1" seed="$2" ablation="$3" name="$4"
  local out="$OUT_ROOT/$MODE/$name"
  mkdir -p "$out"
  local resume_arg=()
  if [[ "$RESUME" == "1" && -f "$out/latest_model.pt" ]]; then
    resume_arg=(--resume)
  fi
  python -u train_architecture.py \
    --arch "$arch" --ablation "$ablation" \
    --data "$DATA_ROOT" --out "$out" \
    --seed "$seed" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
    "${resume_arg[@]}"
}

for seed in 42 123 2026; do
  run_one cafrnet_v2 "$seed" full "cafrnet_seed$seed"
  run_one convnext_tiny "$seed" full "convnext_tiny_seed$seed"
  run_one resnet18 "$seed" full "resnet18_seed$seed"

  if [[ "$MODE" == "imagenette" ]]; then
    run_one cafrnet_v2 "$seed" no_edge "cafrnet_no_edge_seed$seed"
    run_one cafrnet_v2 "$seed" no_semantic_fusion "cafrnet_no_fusion_seed$seed"
    run_one cafrnet_v2 "$seed" no_attention "cafrnet_no_attention_seed$seed"
  fi
done

echo "Training matrix complete for $MODE. Evaluate all best_model.pt checkpoints on identical test folders next."
