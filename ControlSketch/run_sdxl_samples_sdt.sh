#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
samples_dir="${1:-$script_dir/../SDXL_samples}"
output_dir="${OUTPUT_DIR:-$script_dir/output_sketches/20260727/sdtx100}"
gpu_id="${GPU_ID:-0}"
control_sketch_python="${CONTROL_SKETCH_PYTHON:-python}"
sam3_python="${SAM3_PYTHON:-python}"
stroke_counts=(16 24 32)

if [[ ! -d "$samples_dir" ]]; then
    echo "SDXL sample directory does not exist: $samples_dir" >&2
    echo "Pass it as the first argument: $0 /path/to/SDXL_samples" >&2
    exit 1
fi

if ! command -v "$sam3_python" >/dev/null 2>&1; then
    echo "SAM3 Python is not available: $sam3_python" >&2
    echo "Set SAM3_PYTHON=/path/to/sam3_env/bin/python" >&2
    exit 1
fi

samples_dir=$(cd "$samples_dir" && pwd)
mkdir -p "$output_dir"
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
    echo "No supported images or npy/npz files found in: $samples_dir" >&2
    exit 1
fi

echo "GPU: $gpu_id"
echo "Inputs: ${#targets[@]}"
echo "Stroke counts: ${stroke_counts[*]}"
echo "Output: $output_dir"

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
            continue
        fi

        echo "RUN: $relative_path ($strokes strokes) on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
            "$control_sketch_python" object_sketching.py \
            --target "$target" \
            --init_placement semantic \
            --semantic_parts auto \
            --semantic_segmenter sam3 \
            --sam3_python "$sam3_python" \
            --semantic_outline_overlap_tolerance 4 \
            --semantic_sdt_loss_weight 100 \
            --semantic_sdt_loss_margin 2 \
            --semantic_sdt_loss_ramp_iters 300 \
            --save_interval 100 \
            --output_dir "$target_output_dir" \
            --num_strokes "$strokes" \
            --save_svg_in_dict 0
    done
done

echo "All SDXL sample sketches finished."
