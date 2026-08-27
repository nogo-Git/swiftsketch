#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
PROJECT="ControlSketch"
OUTPUT_DIR="./output_sketches/ControlSketch_normal"
EXAMPLES_DIR="../SwiftSketch/examples"

STROKES=(16 24 32)

shopt -s nullglob
IMAGES=("${EXAMPLES_DIR}"/*.png)

for image in "${IMAGES[@]}"; do
  name="$(basename "${image%.*}")"

  for strokes in "${STROKES[@]}"; do
    run_name="${name}_${strokes}_strokes"
    final_path="${OUTPUT_DIR}/${name}/${run_name}/final_sketch.png"

    if [[ -f "${final_path}" ]]; then
      echo "=== Skip ${name}, ${strokes} strokes: already done ==="
      continue
    fi

    echo "=== Running ${name}, ${strokes} strokes ==="

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python object_sketching.py \
      --target "${image}" \
      --output_dir "${OUTPUT_DIR}" \
      --use_wandb 1 \
      --wandb_project_name "${PROJECT}" \
      --wandb_name "${run_name}" \
      --num_strokes "${strokes}"
  done
done