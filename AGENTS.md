# Repository Guidelines

## Project Structure & Module Organization

This repository contains two related Python pipelines. `SwiftSketch/` holds diffusion-based vector sketch generation, training code, model definitions, refinement modules, and shared utilities. `ControlSketch/` contains the optimization-based data-generation pipeline, configuration, and sample inputs under `ControlSketch/data/`. Root-level scripts prepare datasets, convert NPZ files, run batches, and compute CLIP scores. Documentation assets live in `docs/`; example inputs are in `SwiftSketch/examples/`. Treat `output_sketches/`, `wandb/`, `tmp/`, checkpoints, and generated dataset directories as artifacts rather than source.

## Setup, Development, and Execution

Use Python 3.9.19 in a dedicated Conda environment. Install diffvg separately, then install the pinned stack:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

Run ControlSketch from its directory with `python object_sketching.py --target ./data/lion.png`. Run SwiftSketch from `SwiftSketch/` with `python -m generate --model_path PATH --refine_model_path PATH --input_data ./examples --output_dir ./output_sketches`. Prepare features with `python -m utils.get_features --dir_name PATH`; training entry points are `python -m train.train_SwiftSketch ...` and `python -m refine_model.train_refine.train_refine_model ...`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions, variables, modules, and CLI flags, and `PascalCase` for classes. Keep imports grouped at the top and expose scripts through a `if __name__ == "__main__":` guard. No repository-wide formatter or linter is configured, so keep changes PEP 8-compatible and avoid unrelated reformatting.

## Testing Guidelines

The only automated tests currently cover the bundled CLIP implementations. Run them with `pytest SwiftSketch/CLIP_/tests ControlSketch/CLIP_/tests`. Name new files `test_*.py` and test functions `test_*`. For GPU-heavy pipeline changes, also run the smallest relevant example and report the device, command, checkpoint, and resulting artifact in the pull request.

## Commits & Pull Requests

Recent history uses short, imperative summaries, sometimes in Japanese (for example, `初期点配置の問題を修正`). Keep each commit focused and describe the observable change. Pull requests should explain motivation, list validation commands, link relevant issues, and include before/after images for sketch-quality or rendering changes. Do not commit model weights, generated outputs, W&B runs, or local paths and credentials.
