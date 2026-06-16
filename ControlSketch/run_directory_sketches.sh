#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_directory_sketches.sh IMAGE_DIR [options passed to object_sketching.py]

Examples:
  ./run_directory_sketches.sh ./data

  ./run_directory_sketches.sh ./data \
    --output_dir ./output_sketches_batch \
    --num_iter 1000 \
    --num_strokes 32

  ./run_directory_sketches.sh ./data \
    --init_placement semantic \
    --semantic_parts auto \
    --semantic_segmenter sam3 \
    --sam3_python /mnt/cggfs01disk/takei/miniconda3/envs/sam3_env/bin/python

Notes:
  Run this from the original ControlSketch/SwiftSketch conda environment.
  The first argument is the image directory. All remaining arguments are passed
  unchanged to object_sketching.py.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

image_dir=$1
shift

if [[ ! -d "$image_dir" ]]; then
  echo "Image directory does not exist: $image_dir" >&2
  exit 1
fi

image_dir=$(cd "$image_dir" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

mapfile -d '' images < <(
  find "$image_dir" -maxdepth 1 -type f \( \
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

if [[ ${#images[@]} -eq 0 ]]; then
  echo "No image files found in: $image_dir" >&2
  exit 1
fi

echo "Found ${#images[@]} image(s) in $image_dir"

for image_path in "${images[@]}"; do
  echo
  echo "=== Sketching: $image_path ==="
  "${CONTROL_SKETCH_PYTHON:-python}" object_sketching.py --target "$image_path" "$@"
done

echo
echo "All sketches finished."
