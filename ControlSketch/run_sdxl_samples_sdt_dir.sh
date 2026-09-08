#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
samples_dir="${1:-${SAMPLES_DIR:-$script_dir/../SDXL_samples}}"
output_dir="${OUTPUT_DIR:-$script_dir/output_sketches/20260820/sdtx100_dirx0.1}"
gpu_id="${GPU_ID:-0}"
control_sketch_python="${CONTROL_SKETCH_PYTHON:-python}"
sam3_python="${SAM3_PYTHON:-python}"
sdt_weight="${SDT_WEIGHT:-100}"
dir_weight="${DIR_WEIGHT:-0.1}"
loss_ramp_iters="${LOSS_RAMP_ITERS:-300}"
dry_run="${DRY_RUN:-0}"
require_cuda="${REQUIRE_CUDA:-1}"
stroke_counts=(16 24 32)

if [[ ! -d "$samples_dir" ]]; then
    echo "SDXL sample directory does not exist: $samples_dir" >&2
    exit 1
fi

if ! command -v "$control_sketch_python" >/dev/null 2>&1; then
    echo "ControlSketch Python is not available: $control_sketch_python" >&2
    echo "Set CONTROL_SKETCH_PYTHON=/path/to/python" >&2
    exit 1
fi

if ! command -v "$sam3_python" >/dev/null 2>&1; then
    echo "SAM3 Python is not available: $sam3_python" >&2
    echo "Set SAM3_PYTHON=/path/to/sam3_env/bin/python" >&2
    exit 1
fi

if [[ "$dry_run" != "1" && "$require_cuda" == "1" ]]; then
    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$control_sketch_python" -c \
        'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
    then
        echo "CUDA GPU $gpu_id is not available from: $control_sketch_python" >&2
        echo "Run this script on a GPU host, or set REQUIRE_CUDA=0 to allow CPU execution." >&2
        exit 1
    fi
fi

samples_dir=$(cd "$samples_dir" && pwd)
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
cd "$script_dir"

mapfile -d '' targets < <(
    find "$samples_dir" -type f \( \
        -iname '*.npz' -o \
        -iname '*.npy' -o \
        -iname '*.png' -o \
        -iname '*.jpg' -o \
        -iname '*.jpeg' -o \
        -iname '*.bmp' -o \
        -iname '*.gif' -o \
        -iname '*.webp' -o \
        -iname '*.tif' -o \
        -iname '*.tiff' \
    \) -print0 | sort -z
)

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "No supported images or arrays found under: $samples_dir" >&2
    exit 1
fi

total_runs=$((${#targets[@]} * ${#stroke_counts[@]}))
echo "Method: semantic initialization + SDT loss + direction loss"
echo "GPU: $gpu_id"
echo "Inputs: ${#targets[@]}"
echo "Stroke counts: ${stroke_counts[*]}"
echo "Planned runs: $total_runs"
echo "SDT weight: $sdt_weight"
echo "Direction weight: $dir_weight"
echo "Loss ramp iterations: $loss_ramp_iters"
echo "Output: $output_dir"

completed=0
skipped=0
planned=0

for target in "${targets[@]}"; do
    relative_path=${target#"$samples_dir"/}
    relative_dir=$(dirname "$relative_path")
    sample=$(basename "${target%.*}")

    target_output_dir="$output_dir"
    if [[ "$relative_dir" != "." ]]; then
        target_output_dir="$output_dir/$relative_dir"
    fi

    for strokes in "${stroke_counts[@]}"; do
        run_name="${sample}_${strokes}_strokes"
        result="$target_output_dir/$sample/$run_name/final_svg.svg"

        if [[ -s "$result" ]]; then
            echo "SKIP: $relative_path ($strokes strokes)"
            skipped=$((skipped + 1))
            continue
        fi

        planned=$((planned + 1))
        if [[ "$dry_run" == "1" ]]; then
            echo "PLAN: $relative_path ($strokes strokes)"
            continue
        fi

        echo "RUN: $relative_path ($strokes strokes) on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
            "$control_sketch_python" object_sketching.py \
            --target "$target" \
            --output_dir "$target_output_dir" \
            --wandb_name "$run_name" \
            --num_strokes "$strokes" \
            --init_placement semantic \
            --semantic_parts auto \
            --semantic_segmenter sam3 \
            --sam3_python "$sam3_python" \
            --semantic_fallback error \
            --semantic_outline_overlap_tolerance 4 \
            --semantic_sdt_loss_weight "$sdt_weight" \
            --semantic_sdt_loss_margin 2 \
            --semantic_sdt_loss_ramp_iters "$loss_ramp_iters" \
            --semantic_dir_loss_weight "$dir_weight" \
            --semantic_dir_loss_ramp_iters "$loss_ramp_iters" \
            --save_interval 100 \
            --save_svg_in_dict 0

        completed=$((completed + 1))
    done
done

if [[ "$dry_run" == "1" ]]; then
    echo "Dry run finished: $planned pending, $skipped skipped."
else
    echo "Generation finished: $completed completed, $skipped skipped."
fi
