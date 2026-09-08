# Repository Guidelines

## Project Structure & Module Organization

`SwiftSketch/` contains the diffusion model, generation entry point, training loops, refinement network, and shared utilities. `ControlSketch/` contains the optimization-based data-generation pipeline, configuration, shell launchers, and sample assets under `ControlSketch/data/`. Root-level Python scripts prepare datasets and compute CLIP or perceptual metrics. Keep focused unit tests in `tests/`; vendored CLIP compatibility tests live in each `CLIP_/tests/` directory. Documentation and figures belong in `docs/`, while checked-in evaluation summaries are under `scores/`.

## Build, Test, and Development Commands

Use Python 3.9.19, as documented in the README. Install diffvg separately, then install dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

There is no separate build step. Run the lightweight test suite from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Generate a SwiftSketch result from `SwiftSketch/` with `python -m generate --model_path ... --refine_model_path ... --input_data ... --output_dir ...`. Run ControlSketch from `ControlSketch/` with `python object_sketching.py --target ./data/lion.png`. Both full pipelines require downloaded model weights; GPU execution is strongly preferred.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation. Use `snake_case` for modules, functions, variables, and CLI flags; use `PascalCase` for classes and `UPPER_CASE` for constants. Prefer `pathlib.Path`, type hints, and small testable helpers in new root utilities. Preserve established command-line interfaces and dictionary keys such as `image`, `mask`, `attn_map`, and `caption`. No repository-wide formatter or linter is configured, so keep imports organized and changes narrowly scoped.

## Testing Guidelines

Tests use Python's `unittest` framework and names beginning with `test_`. Add regression tests for input discovery, image transforms, metric preprocessing, and error handling. Use temporary directories and generated tiny images instead of committing large fixtures. Run the root suite before submitting; run CLIP compatibility tests only when changing the vendored CLIP code.

## Commit & Pull Request Guidelines

History favors short, single-purpose subjects, often in Japanese (for example, `重複輪郭を排除`). Use a concise imperative subject in Japanese or English and separate unrelated changes. Pull requests should explain the motivation, list validation commands, link relevant issues, and include before/after images for visual-output changes. Do not commit checkpoints, generated sketches, caches, logs, or local configuration; the existing `.gitignore` covers common output and model paths.
