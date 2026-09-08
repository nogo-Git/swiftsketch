# DreamSIM and MS-SSIM evaluation

`compute_controlsketch_perceptual_scores.py` evaluates ControlSketch output pairs
with the perceptual metrics reported by the SwiftSketch paper.

- `dreamsim_distance` compares the RGB reference and RGB sketch with DreamSIM.
  It is a cosine distance, so lower is better.
- `ms_ssim` compares an XDoG edge map of the reference with the grayscale sketch.
  Higher is better.

The default sketch layout is:

```text
<sketch_root>/<category>/*_strokes/final_sketch.png
```

Each sketch is paired first with `input.png` in the same directory as `final_sketch.png`. If that file is absent, the evaluator falls back to `<input_root>/<category>.<extension>`.

## Usage

Install the project requirements first. DreamSIM downloads its pretrained weights
the first time it is used.

```bash
python compute_controlsketch_perceptual_scores.py \
  path/to/input_images \
  path/to/output_sketches \
  --device auto \
  --batch-size 16 \
  --output perceptual_scores.json
```

Run only one metric with `--metrics dreamsim` or `--metrics ms-ssim`.

Progress bars for DreamSIM embedding batches and MS-SSIM pair batches are shown on stderr by default. Use `--no-progress` to disable them.

The MS-SSIM evaluation defaults to 256x256 images, five scales, grayscale input,
and these XDoG parameters:

```text
sigma=0.5, k=10.0, gamma=0.98, epsilon=-0.1, phi=200.0
```

All preprocessing settings are included in the output JSON. They can be changed
with `--image-size`, `--resize-mode`, and the `--xdog-*` arguments. Keep these
settings fixed when comparing methods or runs.

For non-square images, `--resize-mode letterbox` preserves the aspect ratio and
adds a white border. The default `stretch` matches the square-image protocol used
by the paper's dataset.

## Output

The JSON contains per-pair results, global means, means grouped by category and
stroke count, discovery errors, and complete preprocessing metadata.

DreamSIM's default ensemble is relatively large. A supported single-branch model
can be selected with, for example, `--dreamsim-type dino_vitb16`.
