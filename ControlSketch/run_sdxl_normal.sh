#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
samples_dir="${SAMPLES_DIR:-$script_dir/../SDXL_samples}"
output_dir="${OUTPUT_DIR:-$script_dir/output_sketches/SDXL_normal}"
gpu_id="${GPU_ID:-0}"
control_sketch_python="${CONTROL_SKETCH_PYTHON:-python}"
stroke_counts=(16 24 32)

if [[ ! -d "$samples_dir" ]]; then
    echo "SDXL sample directory does not exist: $samples_dir" >&2
    exit 1
fi

mkdir -p "$output_dir"
cd "$script_dir"

shopt -s nullglob
targets=("$samples_dir"/*/*.npz)

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "No .npz files found under: $samples_dir" >&2
    exit 1
fi

echo "Method: normal ControlSketch (k-means initialization)"
echo "GPU: $gpu_id"
echo "Inputs: ${#targets[@]}"
echo "Output: $output_dir"

for target in "${targets[@]}"; do
    class_name=$(basename "$(dirname "$target")")
    sample=$(basename "$target" .npz)

    for strokes in "${stroke_counts[@]}"; do
        run_name="normal_${strokes}_strokes"
        result="$output_dir/$sample/$run_name/final_svg.svg"

        if [[ -s "$result" ]]; then
            echo "SKIP: $sample ($strokes strokes)"
            continue
        fi

        echo "RUN: $sample ($strokes strokes) on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES="$gpu_id" "$control_sketch_python" object_sketching.py \
            --target "$target" \
            --output_dir "$output_dir" \
            --wandb_name "$run_name" \
            --num_strokes "$strokes" \
            --object_name "$class_name" \
            --init_placement kmeans \
            --save_svg_in_dict 0
    done
done

echo "Normal ControlSketch generation finished."
