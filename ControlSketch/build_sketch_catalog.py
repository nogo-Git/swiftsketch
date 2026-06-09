#!/usr/bin/env python3
"""Build a static HTML catalog for ControlSketch output directories."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote


RUN_MARKER_FILES = {
    "config.npy",
    "final_sketch.png",
    "final_svg.svg",
    "input.png",
    "initial_points.jpg",
    "mask.png",
    "sketch.mp4",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "jpg_logs",
    "semantic_debug",
    "svg_logs",
    "svg_to_png",
}

CONFIG_FIELDS = (
    "target",
    "caption",
    "object_name",
    "num_strokes",
    "width",
    "num_segments",
    "num_iter",
    "save_interval",
    "condition",
    "conditioning_scale",
    "diffusion_guidance_scale",
    "init_placement",
    "semantic_parts",
    "semantic_segmenter",
    "semantic_curvature_sampling",
    "seed",
    "wandb_name",
    "experiment_name",
)


@dataclass
class Run:
    path: Path
    rel_path: str
    title: str
    status: str
    modified_at: float
    input_image: Path | None
    final_image: Path | None
    final_svg: Path | None
    video: Path | None
    initial_points: Path | None
    mask: Path | None
    config_path: Path | None
    condition_images: list[Path]
    iterations: list[Path]
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create output_sketches/index.html for browsing generated sketch runs."
    )
    parser.add_argument(
        "--root",
        default="output_sketches",
        help="Directory containing generated ControlSketch outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="HTML file to write. Defaults to <root>/index.html.",
    )
    parser.add_argument(
        "--title",
        default="ControlSketch Catalog",
        help="Page title shown in the generated catalog.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum number of per-run iteration thumbnails to embed.",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Only include runs that have final_sketch.png or final_svg.svg.",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Do not read config.npy metadata.",
    )
    return parser.parse_args()


def posix_rel(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def asset_url(path: Path | None, html_dir: Path) -> str:
    if path is None:
        return ""
    rel = path.relative_to(html_dir).as_posix()
    return quote(rel, safe="/._-~:@")


def text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def json_attr(value: Any) -> str:
    return html.escape(json.dumps(value), quote=True)


def simplify_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return simplify_value(value.item())
        except Exception:
            pass
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(simplify_value(item)) for item in value)
    if isinstance(value, dict):
        return {str(key): simplify_value(val) for key, val in value.items()}
    return str(value)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import numpy as np

        loaded = np.load(path, allow_pickle=True)
        if hasattr(loaded, "item"):
            loaded = loaded.item()
        if not isinstance(loaded, dict):
            return {}
        return {str(key): simplify_value(val) for key, val in loaded.items()}
    except Exception:
        return {}


def interesting_config(config: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in CONFIG_FIELDS:
        if key in config and config[key] not in ("", None):
            value = config[key]
            if key == "target":
                value = Path(str(value)).name
            values[key] = value
    return values


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    return int(match.group(1)) if match else -1


def sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[-1]]

    selected: list[Path] = []
    for i in range(limit):
        index = round(i * (len(paths) - 1) / (limit - 1))
        selected.append(paths[index])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def has_run_marker(path: Path) -> bool:
    try:
        names = {item.name for item in path.iterdir()}
    except OSError:
        return False

    if RUN_MARKER_FILES & names:
        return True
    if any(name.endswith("_condition.png") for name in names):
        return True
    svg_to_png = path / "svg_to_png"
    return svg_to_png.is_dir() and any(svg_to_png.glob("iter_*.png"))


def find_runs(root: Path, *, complete_only: bool, read_config: bool) -> list[Run]:
    runs: list[Run] = []

    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if name not in SKIP_DIRS]
        current_path = Path(current)
        if not has_run_marker(current_path):
            continue

        files = set(file_names)
        final_image = current_path / "final_sketch.png" if "final_sketch.png" in files else None
        final_svg = current_path / "final_svg.svg" if "final_svg.svg" in files else None
        if complete_only and final_image is None and final_svg is None:
            continue

        input_image = current_path / "input.png" if "input.png" in files else None
        video = current_path / "sketch.mp4" if "sketch.mp4" in files else None
        initial_points = (
            current_path / "initial_points.jpg" if "initial_points.jpg" in files else None
        )
        mask = current_path / "mask.png" if "mask.png" in files else None
        config_path = current_path / "config.npy" if "config.npy" in files else None
        condition_images = sorted(current_path.glob("*_condition.png"))
        iterations = sorted(
            (current_path / "svg_to_png").glob("iter_*.png"),
            key=lambda path: (numeric_suffix(path), path.name),
        )

        display_final = final_image or final_svg or (iterations[-1] if iterations else None)
        modified_candidates = [
            path
            for path in (final_image, final_svg, video, config_path, input_image, display_final)
            if path is not None and path.exists()
        ]
        modified_at = max(
            (path.stat().st_mtime for path in modified_candidates),
            default=current_path.stat().st_mtime,
        )
        config = load_config(config_path) if read_config and config_path else {}
        status = "complete" if final_image or final_svg else "incomplete"

        runs.append(
            Run(
                path=current_path,
                rel_path=posix_rel(current_path, root),
                title=current_path.name,
                status=status,
                modified_at=modified_at,
                input_image=input_image,
                final_image=display_final,
                final_svg=final_svg,
                video=video,
                initial_points=initial_points,
                mask=mask,
                config_path=config_path,
                condition_images=condition_images,
                iterations=iterations,
                config=config,
            )
        )

    runs.sort(key=lambda run: (-run.modified_at, run.rel_path.lower()))
    return runs


def file_size(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    size = path.stat().st_size
    units = ("B", "KB", "MB", "GB")
    if size == 0:
        return "0 B"
    exponent = min(int(math.log(size, 1024)), len(units) - 1)
    value = size / (1024**exponent)
    if exponent == 0:
        return f"{int(value)} {units[exponent]}"
    return f"{value:.1f} {units[exponent]}"


def run_search_text(run: Run, meta: dict[str, Any]) -> str:
    parts = [run.rel_path, run.title, run.status]
    parts.extend(str(value) for value in meta.values())
    return " ".join(parts).lower()


def image_block(label: str, path: Path | None, html_dir: Path) -> str:
    if path is None:
        return f"""
          <figure class="asset asset-missing">
            <div class="missing">Missing</div>
            <figcaption>{text(label)}</figcaption>
          </figure>
        """

    url = asset_url(path, html_dir)
    return f"""
      <figure class="asset">
        <a href="{url}" title="Open {text(path.name)}">
          <img src="{url}" alt="{text(label)}" loading="lazy">
        </a>
        <figcaption>{text(label)}</figcaption>
      </figure>
    """


def meta_pills(meta: dict[str, Any], run: Run) -> str:
    pills: list[str] = []
    for key in ("target", "num_strokes", "condition", "init_placement", "seed"):
        if key in meta:
            label = key.replace("_", " ")
            pills.append(f"<span>{text(label)}: {text(meta[key])}</span>")
    pills.append(f"<span>{len(run.iterations)} frames</span>")
    if run.video:
        pills.append(f"<span>video</span>")
    return "\n".join(pills)


def link_buttons(run: Run, html_dir: Path) -> str:
    links: list[tuple[str, Path | None]] = [
        ("Run", run.path),
        ("Final", run.final_image),
        ("SVG", run.final_svg),
        ("Video", run.video),
        ("Config", run.config_path),
    ]
    buttons: list[str] = []
    for label, path in links:
        if path is None:
            continue
        buttons.append(f'<a class="link-button" href="{asset_url(path, html_dir)}">{label}</a>')
    return "\n".join(buttons)


def iteration_details(run: Run, html_dir: Path, max_iterations: int) -> str:
    if not run.iterations:
        return ""

    shown = sample_paths(run.iterations, max_iterations)
    thumbs = []
    for path in shown:
        url = asset_url(path, html_dir)
        iteration = numeric_suffix(path)
        label = f"iter {iteration}" if iteration >= 0 else path.stem
        thumbs.append(
            f"""
            <a class="iter-thumb" href="{url}" title="{text(label)}">
              <img src="{url}" alt="{text(label)}" loading="lazy">
              <span>{text(label)}</span>
            </a>
            """
        )
    folder = run.path / "svg_to_png"
    folder_link = (
        f'<a href="{asset_url(folder, html_dir)}">{len(run.iterations)} frame files</a>'
        if folder.exists()
        else f"{len(run.iterations)} frame files"
    )
    return f"""
      <details class="panel">
        <summary>Progress <span>{len(shown)} shown of {folder_link}</span></summary>
        <div class="iteration-grid">
          {"".join(thumbs)}
        </div>
      </details>
    """


def supporting_assets(run: Run, html_dir: Path) -> str:
    assets: list[tuple[str, Path]] = []
    if run.initial_points:
        assets.append(("Initial points", run.initial_points))
    if run.mask:
        assets.append(("Mask", run.mask))
    assets.extend((path.stem.replace("_", " "), path) for path in run.condition_images)
    if not assets:
        return ""

    figures = "\n".join(image_block(label, path, html_dir) for label, path in assets)
    return f"""
      <details class="panel">
        <summary>Supporting Images <span>{len(assets)}</span></summary>
        <div class="support-grid">
          {figures}
        </div>
      </details>
    """


def config_details(meta: dict[str, Any], run: Run) -> str:
    if not meta:
        return ""

    rows = "\n".join(
        f"<tr><th>{text(key.replace('_', ' '))}</th><td>{text(value)}</td></tr>"
        for key, value in meta.items()
    )
    if run.final_image:
        rows += (
            f"<tr><th>final file size</th><td>{text(file_size(run.final_image))}</td></tr>"
        )
    return f"""
      <details class="panel">
        <summary>Run Metadata <span>{len(meta)} fields</span></summary>
        <table>
          {rows}
        </table>
      </details>
    """


def render_run(run: Run, html_dir: Path, max_iterations: int) -> str:
    meta = interesting_config(run.config)
    modified = datetime.fromtimestamp(run.modified_at).strftime("%Y-%m-%d %H:%M")
    strokes = meta.get("num_strokes", "")
    search = run_search_text(run, meta)
    status_label = "Complete" if run.status == "complete" else "Incomplete"

    return f"""
    <article class="run-card"
      data-search="{text(search)}"
      data-status="{text(run.status)}"
      data-name="{text(run.rel_path.lower())}"
      data-strokes="{text(strokes)}"
      data-mtime="{run.modified_at}">
      <div class="run-media">
        {image_block("Input", run.input_image, html_dir)}
        {image_block("Final", run.final_image, html_dir)}
      </div>
      <div class="run-body">
        <div class="run-heading">
          <div>
            <p class="path">{text(run.rel_path)}</p>
            <h2>{text(run.title)}</h2>
          </div>
          <span class="status {text(run.status)}">{status_label}</span>
        </div>
        <div class="pills">
          {meta_pills(meta, run)}
        </div>
        <div class="links">
          {link_buttons(run, html_dir)}
        </div>
        <p class="modified">Updated {text(modified)}</p>
        {iteration_details(run, html_dir, max_iterations)}
        {supporting_assets(run, html_dir)}
        {config_details(meta, run)}
      </div>
    </article>
    """


def render_html(title: str, root: Path, output: Path, runs: list[Run], max_iterations: int) -> str:
    complete_count = sum(1 for run in runs if run.status == "complete")
    incomplete_count = len(runs) - complete_count
    html_dir = output.parent
    cards = "\n".join(render_run(run, html_dir, max_iterations) for run in runs)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    root_label = output.parent.relative_to(Path.cwd()).as_posix() if output.parent.is_relative_to(Path.cwd()) else output.parent.as_posix()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{text(title)}</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #16202a;
      --muted: #657282;
      --line: #d9e0e8;
      --blue: #1f6feb;
      --green: #227a4d;
      --amber: #9a6700;
      --shadow: 0 12px 30px rgba(22, 32, 42, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}

    a {{
      color: inherit;
      text-decoration: none;
    }}

    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, 0.95);
      backdrop-filter: blur(10px);
    }}

    .header-inner {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 18px 22px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: center;
    }}

    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 740;
      line-height: 1.15;
    }}

    .subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}

    .stats {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      font-size: 13px;
    }}

    .stats span,
    .pills span {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 10px;
      background: #fff;
      color: var(--muted);
      white-space: nowrap;
    }}

    .toolbar {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 0 22px 18px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 160px 160px;
      gap: 10px;
    }}

    input,
    select {{
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      padding: 0 11px;
      outline: none;
    }}

    input:focus,
    select:focus {{
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.12);
    }}

    main {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 22px;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 18px;
      align-items: start;
    }}

    .run-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .run-media {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      border-bottom: 1px solid var(--line);
      background: #edf1f5;
    }}

    .asset {{
      margin: 0;
      min-width: 0;
      border-right: 1px solid var(--line);
      background: #fff;
    }}

    .asset:last-child {{
      border-right: 0;
    }}

    .asset a,
    .missing {{
      display: grid;
      place-items: center;
      aspect-ratio: 1 / 1;
      background:
        linear-gradient(45deg, #f9fafb 25%, transparent 25%),
        linear-gradient(-45deg, #f9fafb 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #f9fafb 75%),
        linear-gradient(-45deg, transparent 75%, #f9fafb 75%);
      background-size: 22px 22px;
      background-position: 0 0, 0 11px, 11px -11px, -11px 0;
    }}

    .asset img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fff;
    }}

    figcaption {{
      height: 30px;
      padding: 7px 10px 0;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      line-height: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .missing {{
      color: #8a96a3;
      font-size: 13px;
    }}

    .run-body {{
      padding: 14px;
    }}

    .run-heading {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
    }}

    .path,
    .modified {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}

    h2 {{
      margin: 3px 0 0;
      font-size: 17px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }}

    .status {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}

    .status.complete {{
      background: #e9f6ef;
      color: var(--green);
    }}

    .status.incomplete {{
      background: #fff3d6;
      color: var(--amber);
    }}

    .pills,
    .links {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}

    .pills {{
      font-size: 12px;
    }}

    .link-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 11px;
      background: #f8fafc;
      color: #223143;
      font-size: 13px;
      font-weight: 650;
    }}

    .link-button:hover {{
      border-color: #b6c2d0;
      background: #eef4fb;
    }}

    .modified {{
      margin-top: 12px;
    }}

    .panel {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}

    summary {{
      cursor: pointer;
      color: #223143;
      font-size: 13px;
      font-weight: 700;
      list-style-position: outside;
    }}

    summary span {{
      color: var(--muted);
      font-weight: 500;
    }}

    summary a {{
      color: var(--blue);
      text-decoration: underline;
      text-underline-offset: 2px;
    }}

    .iteration-grid,
    .support-grid {{
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }}

    .iteration-grid {{
      grid-template-columns: repeat(auto-fill, minmax(86px, 1fr));
    }}

    .support-grid {{
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    }}

    .iter-thumb {{
      display: block;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }}

    .iter-thumb img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #fff;
    }}

    .iter-thumb span {{
      display: block;
      height: 24px;
      border-top: 1px solid var(--line);
      padding: 5px 6px 0;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    table {{
      width: 100%;
      margin-top: 10px;
      border-collapse: collapse;
      font-size: 12px;
    }}

    th,
    td {{
      border-top: 1px solid var(--line);
      padding: 7px 0;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}

    th {{
      width: 34%;
      padding-right: 14px;
      color: var(--muted);
      font-weight: 650;
    }}

    .empty {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 28px;
      color: var(--muted);
      text-align: center;
    }}

    @media (max-width: 760px) {{
      .header-inner,
      .toolbar {{
        grid-template-columns: 1fr;
      }}

      .stats {{
        justify-content: flex-start;
      }}

      main {{
        padding: 14px;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 470px) {{
      .run-media {{
        grid-template-columns: 1fr;
      }}

      .asset {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}

      .asset:last-child {{
        border-bottom: 0;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>{text(title)}</h1>
        <p class="subtitle">{text(root_label)} - generated {text(generated)}</p>
      </div>
      <div class="stats">
        <span id="visibleCount">{len(runs)} visible</span>
        <span>{complete_count} complete</span>
        <span>{incomplete_count} incomplete</span>
      </div>
    </div>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search target, caption, path, settings">
      <select id="status">
        <option value="all">All statuses</option>
        <option value="complete">Complete</option>
        <option value="incomplete">Incomplete</option>
      </select>
      <select id="sort">
        <option value="newest">Newest first</option>
        <option value="name">Name</option>
        <option value="strokes">Strokes</option>
      </select>
    </div>
  </header>
  <main>
    <div class="grid" id="grid">
      {cards if cards else '<div class="empty">No sketch runs found.</div>'}
    </div>
  </main>
  <script>
    const grid = document.getElementById("grid");
    const search = document.getElementById("search");
    const status = document.getElementById("status");
    const sort = document.getElementById("sort");
    const visibleCount = document.getElementById("visibleCount");

    function numeric(value) {{
      const parsed = Number.parseFloat(value);
      return Number.isFinite(parsed) ? parsed : -1;
    }}

    function applyControls() {{
      const query = search.value.trim().toLowerCase();
      const statusValue = status.value;
      const cards = Array.from(grid.querySelectorAll(".run-card"));
      cards.sort((a, b) => {{
        if (sort.value === "name") {{
          return a.dataset.name.localeCompare(b.dataset.name);
        }}
        if (sort.value === "strokes") {{
          return numeric(a.dataset.strokes) - numeric(b.dataset.strokes)
            || a.dataset.name.localeCompare(b.dataset.name);
        }}
        return numeric(b.dataset.mtime) - numeric(a.dataset.mtime);
      }});
      cards.forEach(card => grid.appendChild(card));

      let visible = 0;
      cards.forEach(card => {{
        const statusMatches = statusValue === "all" || card.dataset.status === statusValue;
        const queryMatches = !query || card.dataset.search.includes(query);
        const show = statusMatches && queryMatches;
        card.hidden = !show;
        if (show) visible += 1;
      }});
      visibleCount.textContent = `${{visible}} visible`;
    }}

    search.addEventListener("input", applyControls);
    status.addEventListener("change", applyControls);
    sort.addEventListener("change", applyControls);
    applyControls();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "index.html"

    if not root.exists():
        raise SystemExit(f"Output root does not exist: {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    runs = find_runs(root, complete_only=args.complete_only, read_config=not args.skip_config)
    output.write_text(
        render_html(args.title, root, output, runs, args.max_iterations),
        encoding="utf-8",
    )
    print(f"Wrote {output} with {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
