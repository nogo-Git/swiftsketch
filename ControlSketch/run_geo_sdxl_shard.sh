#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shard_index="${1:?Usage: $0 SHARD_INDEX GPU_ID [SDXL_SAMPLES_DIR]}"
gpu_id="${2:?Usage: $0 SHARD_INDEX GPU_ID [SDXL_SAMPLES_DIR]}"
shard_count=4
samples_dir="${3:-${SAMPLES_DIR:-$script_dir/../SDXL_samples}}"
output_dir="${OUTPUT_DIR:-$script_dir/output_sketches/outputs_cov}"
control_sketch_python="${CONTROL_SKETCH_PYTHON:-python}"
sam3_python="${SAM3_PYTHON:-python}"
dry_run="${DRY_RUN:-0}"
stroke_counts=(16 24 32)

if ! [[ "$shard_index" =~ ^[0-3]$ ]]; then
    echo "SHARD_INDEX must be 0, 1, 2, or 3" >&2
    exit 1
fi
if [[ ! -d "$samples_dir" ]]; then
    echo "SDXL sample directory does not exist: $samples_dir" >&2
    exit 1
fi
if ! command -v "$control_sketch_python" >/dev/null 2>&1; then
    echo "ControlSketch Python is not available: $control_sketch_python" >&2
    exit 1
fi
if ! command -v "$sam3_python" >/dev/null 2>&1; then
    echo "SAM3 Python is not available: $sam3_python" >&2
    exit 1
fi

samples_dir=$(cd "$samples_dir" && pwd)
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
cache_dir="$output_dir/cache"
mkdir -p "$cache_dir"

mapfile -d '' targets < <(
    find "$samples_dir" -type f \( \
        -iname '*.npz' -o -iname '*.npy' -o \
        -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o \
        -iname '*.bmp' -o -iname '*.gif' -o -iname '*.webp' -o \
        -iname '*.tif' -o -iname '*.tiff' \
    \) -print0 | sort -z
)

selected=0
for ((target_index = shard_index; target_index < ${#targets[@]}; target_index += shard_count)); do
    target=${targets[target_index]}
    relative_path=${target#"$samples_dir"/}
    relative_dir=$(dirname "$relative_path")
    sample=$(basename "${target%.*}")
    object_name="${OBJECT_NAME:-$(basename "$(dirname "$target")")}"
    if [[ "$relative_dir" == "." ]]; then
        object_name="${OBJECT_NAME:-$sample}"
        target_output_dir="$output_dir"
    else
        target_output_dir="$output_dir/$relative_dir"
    fi
    mkdir -p "$target_output_dir"

    for strokes in "${stroke_counts[@]}"; do
        run_name="${sample}_${strokes}_strokes"
        result="$target_output_dir/$sample/$run_name/final_svg.svg"
        if [[ -s "$result" ]]; then
            echo "SKIP shard=$shard_index image=$relative_path strokes=$strokes"
            continue
        fi
        echo "RUN shard=$shard_index gpu=$gpu_id image=$relative_path strokes=$strokes object=$object_name"
        selected=$((selected + 1))
        if [[ "$dry_run" == "1" ]]; then
            continue
        fi

        (
            cd "$script_dir"
            CUDA_VISIBLE_DEVICES="$gpu_id" "$control_sketch_python" object_sketching.py \
                --target "$target" \
                --output_dir "$target_output_dir" \
                --cache_dir "$cache_dir" \
                --wandb_name "$run_name" \
                --num_iter 2000 \
                --num_strokes "$strokes" \
                --seed 0 \
                --object_name "$object_name" \
                --init_placement semantic \
                --semantic_parts auto \
                --semantic_segmenter sam3 \
                --sam3_python "$sam3_python" \
                --semantic_fallback error \
                --save_svg_in_dict 0 \
                --annotate_clip_score 0 \
                --geo_anchor_weight 0 \
                --geo_hold 0.3 \
                --geo_pos_weight 0.1 \
                --geo_cov_weight 0.1 \
                --geo_decay 0.3 \
                --grad_log false
        )
    done
done

echo "DONE shard=$shard_index scheduled_runs=$selected"
