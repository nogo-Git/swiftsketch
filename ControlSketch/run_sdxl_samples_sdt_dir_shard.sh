#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shard_index="${1:-${SHARD_INDEX:-}}"
shard_count="${SHARD_COUNT:-3}"
samples_dir="${2:-${SAMPLES_DIR:-$script_dir/../SDXL_samples}}"

if ! [[ "$shard_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "SHARD_COUNT must be a positive integer: $shard_count" >&2
    exit 1
fi

if ! [[ "$shard_index" =~ ^[0-9]+$ ]] || ((shard_index >= shard_count)); then
    echo "Usage: $0 SHARD_INDEX [SDXL_SAMPLES_DIR]" >&2
    echo "SHARD_INDEX must satisfy 0 <= index < SHARD_COUNT." >&2
    exit 1
fi

if [[ ! -d "$samples_dir" ]]; then
    echo "SDXL sample directory does not exist: $samples_dir" >&2
    exit 1
fi

samples_dir=$(cd "$samples_dir" && pwd)
shard_dir=$(mktemp -d "/tmp/sdxl-dir-shard-${shard_index}.XXXXXX")

cleanup() {
    rm -rf "$shard_dir"
}
trap cleanup EXIT INT TERM

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

selected=0
for ((target_index = shard_index; target_index < ${#targets[@]}; target_index += shard_count)); do
    target=${targets[target_index]}
    relative_path=${target#"$samples_dir"/}
    mkdir -p "$shard_dir/$(dirname "$relative_path")"
    cp --reflink=auto "$target" "$shard_dir/$relative_path"
    selected=$((selected + 1))
done

echo "Shard: $shard_index/$shard_count"
echo "GPU: ${GPU_ID:-0}"
echo "Selected images: $selected of ${#targets[@]}"
echo "Planned sketches: $((selected * 3))"

"$script_dir/run_sdxl_samples_sdt_dir.sh" "$shard_dir"
