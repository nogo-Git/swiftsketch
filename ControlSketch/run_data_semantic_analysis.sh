#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_dir="${DATA_DIR:-$script_dir/data}"
output_dir="${OUTPUT_DIR:-$script_dir/output_sketches/semantic_analysis2}"
gpu_id="${GPU_ID:-1}"
control_sketch_python="${CONTROL_SKETCH_PYTHON:-python}"
sam3_python="${SAM3_PYTHON:-python}"
dry_run="${DRY_RUN:-0}"

samples=(cat elephant ice_cream lion rabbit)
stroke_counts=(16 24 32)

if ! command -v "$control_sketch_python" >/dev/null 2>&1; then
    echo "ControlSketch Python is not available: $control_sketch_python" >&2
    echo "Set CONTROL_SKETCH_PYTHON=/path/to/python if needed." >&2
    exit 1
fi

if ! command -v "$sam3_python" >/dev/null 2>&1; then
    echo "SAM3 Python is not available: $sam3_python" >&2
    echo "Set SAM3_PYTHON=/path/to/sam3_env/bin/python if needed." >&2
    exit 1
fi

mkdir -p "$output_dir"
data_dir=$(cd "$data_dir" && pwd)
output_dir=$(cd "$output_dir" && pwd)
cd "$script_dir"

echo "Method: semantic-part contour initialization"
echo "GPU: $gpu_id"
echo "Inputs: ${samples[*]}"
echo "Stroke counts: ${stroke_counts[*]}"
echo "Planned runs: $((${#samples[@]} * ${#stroke_counts[@]}))"
echo "Output: $output_dir"

completed=0
skipped=0
planned=0

for sample in "${samples[@]}"; do
    if [[ "$sample" == "cat" ]]; then
        target="$data_dir/cat.npz"
    else
        target="$data_dir/$sample.png"
    fi

    if [[ ! -f "$target" ]]; then
        echo "Input does not exist: $target" >&2
        exit 1
    fi

    object_name=${sample//_/ }

    for strokes in "${stroke_counts[@]}"; do
        run_name="${sample}_${strokes}_strokes"
        result="$output_dir/$sample/$run_name/final_svg.svg"

        if [[ -s "$result" ]]; then
            echo "SKIP: $sample ($strokes strokes)"
            skipped=$((skipped + 1))
            continue
        fi

        planned=$((planned + 1))
        if [[ "$dry_run" == "1" ]]; then
            echo "PLAN: $sample ($strokes strokes)"
            continue
        fi

        echo "RUN: $sample ($strokes strokes) on GPU $gpu_id"

        CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$control_sketch_python" object_sketching.py \
            --target "$target" \
            --output_dir "$output_dir" \
            --num_strokes "$strokes" \
            --object_name "$object_name" \
            --init_placement semantic \
            --semantic_parts auto \
            --semantic_segmenter sam3 \
            --sam3_python "$sam3_python"

        completed=$((completed + 1))
    done
done

if [[ "$dry_run" == "1" ]]; then
    echo "Dry run finished: $planned pending, $skipped skipped."
else
    echo "Generation finished: $completed completed, $skipped skipped."
fi
