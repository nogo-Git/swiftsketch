import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os
import random
import sys
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from control_sds_loss_file import ControlSDSLoss
import config
import sketch_utils as utils
from painter_params import Painter, PainterOptimizer
from semantic_stroke_analysis import (
    SemanticStrokeAnalyzer,
    analysis_checkpoint_steps,
)
from semantic_colors import add_semantic_legend
from pathlib import Path
import csv
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.geo_loss import (
    GeoLoss,
    build_reference,
    contour_prf,
    grad_norm_ratio,
)
from semantic_init import parse_semantic_weights


GRAD_LOG_FIELDS = [
    "step",
    "g_sds",
    "g_geo",
    "ratio",
    "cos",
    "g_sds_ema",
    "g_geo_ema",
    "tau",
    "w_geo",
    "w_anc",
    "pos",
    "cov",
    "tan",
    "anc",
    "total",
    "precision",
    "recall",
    "f",
]


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_strokes, args=args,
                       num_segments=args.num_segments,
                       device=args.device,
                       target_im=target_im,
                       mask=mask,
                       )
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    if args.target_is_dict:  # in case a dictionary is provided
        if os.path.splitext(os.path.basename(args.target))[-1] == ".npy":
            input_dict = np.load(args.target, allow_pickle='TRUE').item()
        else:
            input_dict= utils.load_compressed_npz(args.target)
        target = input_dict["image"]
        target = target.resize((args.render_size, args.render_size))
        if "mask" in input_dict:
            mask = input_dict["mask"] 
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
        else: 
            mask = utils.get_mask(target, args.device)
        if not args.caption and "caption" in input_dict:
            args.caption = input_dict["caption"]
        target= utils.create_masked_image(target, mask)
        if "attn_map" in input_dict:
            attn = input_dict["attn_map"]
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
            args.attn_from_dict= attn

    else:
        target = Image.open(args.target)
        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")
        mask = utils.get_mask(target, args.device)
        target = utils.create_masked_image(target, mask)
        if args.fix_scale:
            target, mask = utils.fix_image_mask_scale(target, mask)
        target = target.resize((args.render_size, args.render_size))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (args.render_size, args.render_size))[0][0]
    
    # Keep an output copy in the original coordinate frame. The target used for
    # optimization may be cropped, resized, and centered below.
    output_input = target.copy()
    target_was_reframed = False

    # Reduces the size of the object on the canvas if needed
    im_np = np.array(target)
    test_mask = mask.clone()
    test_mask[test_mask < 0.5] = 0
    test_mask[test_mask >= 0.5] = 1
    w, h = target.size[0], target.size[1]
    x0, x1, y0, y1 = utils.get_obj_bb(test_mask)
    im_width, im_height = x1 - x0, y1 - y0
    max_size = max(im_width, im_height)
    target_size = int(args.render_size * args.object_size_ratio)
    if max_size > target_size: 
        target_was_reframed = True
        if im_width > im_height:
            new_width, new_height = target_size, int((target_size / im_width) * im_height)
        else:
            new_width, new_height = int((target_size / im_height) * im_width), target_size
        mask_np3 = np.stack([test_mask] * 3, axis=-1)
        mask = utils.cut_and_resize(mask_np3, x0, x1, y0, y1, new_height, new_width, "mask")
        mask = torch.from_numpy(mask[:, :, 0])  
        target_np = im_np / im_np.max()
        im_np = utils.cut_and_resize(target_np, x0, x1, y0, y1, new_height, new_width, "image")
        im_np_final = (im_np / im_np.max() * 255).astype(np.uint8)
        target= Image.fromarray(im_np_final)
        

        args.new_width= new_width
        args.new_height= new_height
        args.obj_bb= (x0, x1, y0, y1)
        args.original_center_y = (y0 + (y1 - y0) / 2) / h
        args.original_center_x = (x0 + (x1 - x0) / 2) / w
        args.scale_w = new_width / im_width
        args.scale_h = new_height / im_height

    
    args.input_image = target
    args.mask = mask
    output_size = (args.output_svg_size, args.output_svg_size)
    if output_input.size != output_size:
        output_input = output_input.resize(output_size, Image.Resampling.LANCZOS)
    output_input.save(f"{args.output_dir}/input.png")
    if target_was_reframed:
        target.save(f"{args.output_dir}/optimization_input.png")
    if args.use_wandb:
        wandb.log({"input": wandb.Image(output_input)})
    utils.save_mask(mask, args.output_dir, args.use_wandb)
    data_transform = transforms.ToTensor()
    target = data_transform(target).unsqueeze(0).to(args.device)
    return target, mask


def _draw_clip_score(image, clip_score):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    text = f"CLIP score: {clip_score:.2f}"
    font_size = max(18, min(46, min(image.size) // 24))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    margin = max(8, min(image.size) // 45)
    pad_x = max(8, font_size // 3)
    pad_y = max(5, font_size // 5)

    x0 = image.width - text_w - pad_x * 2 - margin
    y0 = image.height - text_h - pad_y * 2 - margin
    x1 = image.width - margin
    y1 = image.height - margin

    draw.rounded_rectangle(
        [x0, y0, x1, y1],
        radius=max(4, font_size // 5),
        fill=(255, 255, 255, 220),
        outline=(0, 0, 0, 90),
    )
    draw.text(
        (x0 + pad_x, y0 + pad_y),
        text,
        font=font,
        fill=(20, 20, 20, 255),
    )

    return Image.alpha_composite(image, overlay).convert("RGB")


def _compute_and_annotate_clip_score(args, sketch_path, final_sketch):
    if not args.annotate_clip_score:
        return final_sketch

    try:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from compute_clip_score import compute_clip_score

        result = compute_clip_score(
            image_path=Path(args.output_dir) / "input.png",
            sketch_path=Path(sketch_path),
            model_name=args.clip_score_model,
            device=args.clip_score_device,
            jit=bool(args.clip_score_jit),
        )

        clip_score = float(result["clip_score_x100"])
        annotated = _draw_clip_score(final_sketch, clip_score)
        annotated.save(sketch_path)

        with open(Path(args.output_dir) / "clip_score.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"CLIP score: {clip_score:.4f}")
        return annotated

    except Exception as err:
        print(f"Warning: failed to compute or annotate CLIP score: {err}")
        return final_sketch


def save_loss_history(loss_history, output_dir):
    if not loss_history:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "loss_history.csv"
    graph_path = output_dir / "loss_curve.png"

    fieldnames = [
        "epoch",
        "sds_loss",
        "sdt_loss_raw",
        "sdt_weight",
        "sdt_loss_weighted",
        "dir_loss_raw",
        "dir_weight",
        "dir_loss_weighted",
        "geo_tau",
        "geo_weight",
        "anchor_weight",
        "pos_loss",
        "cov_loss",
        "tan_loss",
        "anchor_loss",
        "geo_loss_total",
        "total_loss",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(loss_history)

    epochs = [row["epoch"] for row in loss_history]
    sds_losses = [row["sds_loss"] for row in loss_history]
    sdt_losses = [
        row["sdt_loss_weighted"] for row in loss_history
    ]
    dir_losses = [
        row["dir_loss_weighted"] for row in loss_history
    ]
    pos_losses = [row["pos_loss"] for row in loss_history]
    cov_losses = [row["cov_loss"] for row in loss_history]
    geo_losses = [row["geo_loss_total"] for row in loss_history]
    total_losses = [row["total_loss"] for row in loss_history]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        sds_losses,
        label="SDS loss",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        sdt_losses,
        label="SDT loss",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        dir_losses,
        label="Direction loss",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        pos_losses,
        label="Position loss (raw)",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        cov_losses,
        label="Coverage loss (raw)",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        geo_losses,
        label="Geometry loss (weighted)",
        linewidth=1.5,
    )
    ax.plot(
        epochs,
        total_losses,
        label="Total loss",
        linewidth=2.0,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(graph_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_grad_meta(args, renderer, geo_crit, reference):
    scale = float(renderer.canvas_width) / 224.0
    tau0 = geo_crit.tau0 if geo_crit is not None else args.geo_tau0 * scale
    tau1 = geo_crit.tau1 if geo_crit is not None else args.geo_tau1 * scale
    reference_count = 0 if reference is None else int(
        reference["edge_pts"].shape[0]
    )
    semantic_metadata = {
        "vlm_model": getattr(args, "semantic_vlm_model", ""),
        "max_parts": int(getattr(args, "max_parts", 8)),
        "vlm_seed": int(getattr(args, "vlm_seed", 0)),
        "vlm_temperature": float(getattr(args, "vlm_temperature", 0.3)),
        "n_parts_returned": 0,
        "cache_hash": "",
        "parts": [],
        "sam_dropped_parts": [],
        "vlm_fallback": False,
    }
    semantic_metadata.update(getattr(renderer, "semantic_cache_info", {}))
    semantic_metadata["Q"] = reference_count
    metadata = {
        "canvas_width": int(renderer.canvas_width),
        "n": len(renderer.shapes),
        "num_iter": int(args.num_iter),
        "tau0": float(tau0),
        "tau1": float(tau1),
        "w_pos": float(args.geo_pos_weight),
        "w_cov": float(args.geo_cov_weight),
        "w_tan": float(args.geo_tan_weight),
        "w_anchor": float(args.geo_anchor_weight),
        "geo_hold": float(args.geo_hold),
        "geo_decay": float(args.geo_decay),
        "k": 1.0,
        "LAMBDA": 1.0,
        "Q": reference_count,
        "seed": int(args.seed),
        "input_image_path": str(Path(args.target).resolve()),
    }
    metadata.update(semantic_metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2, ensure_ascii=False)


def save_grad_history(grad_history, output_dir):
    if not grad_history:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grad_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GRAD_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(grad_history)


def main(args):
    print("run object sketching", flush=True)
    inputs, mask = get_target(args)  
    renderer = load_renderer(args, inputs, mask) 
    sds_loss = ControlSDSLoss(args, args.device)

    optimizer = PainterOptimizer(args, renderer)
    img = renderer.init_image()  
    optimizer.init_optimizers()

    geo_weights = (
        args.geo_pos_weight,
        args.geo_cov_weight,
        args.geo_tan_weight,
    )
    geo_crit = None
    reference = None
    initial_control_points = None
    if any(weight != 0.0 for weight in geo_weights):
        if args.num_segments != 1 or args.control_points_per_seg != 4:
            raise ValueError(
                "GeoLoss requires one cubic segment (four control points) "
                "per stroke."
            )
        reference_cache = getattr(renderer, "semantic_reference_cache", None)
        use_cached_reference = (
            reference_cache is not None
            and reference_cache.reference_ready
            and not args.rebuild_reference
        )
        if use_cached_reference:
            reference = reference_cache.load_reference(args.device)
            expected_size = (renderer.canvas_height, renderer.canvas_width)
            use_cached_reference = tuple(reference["size"]) == expected_size
            if not use_cached_reference:
                print(
                    "[semantic_cache] reference size changed; rebuilding",
                    flush=True,
                )
        if use_cached_reference:
            print("[semantic_cache] loaded reference.npz", flush=True)
        else:
            reference_layers = []
            if renderer.semantic_part_masks:
                parts = list(renderer.semantic_part_masks)
                weights_text = getattr(
                    renderer, "semantic_weights_text", args.semantic_weights
                )
                importance = parse_semantic_weights(weights_text, parts)
                reference_layers.extend(
                    (renderer.semantic_part_masks[part], 3.0 * importance[part])
                    for part in parts
                )
            else:
                reference_layers.append(
                    (renderer.mask.detach().cpu().numpy(), 3.0)
                )
            reference_layers.sort(key=lambda item: item[1])
            reference = build_reference(
                reference_layers,
                size=(renderer.canvas_height, renderer.canvas_width),
                min_len=args.geo_min_component_len,
                blur=args.geo_blur,
                rho=args.geo_structure_rho,
                device=args.device,
            )
            if reference_cache is not None:
                reference_cache.save_reference(reference)
        print(
            f"GeoLoss reference points: {reference['edge_pts'].shape[0]}",
            flush=True,
        )
        geo_crit = GeoLoss(
            reference,
            T=args.num_iter,
            M=args.geo_polyline_segments,
            w_pos=args.geo_pos_weight,
            w_cov=args.geo_cov_weight,
            w_tan=args.geo_tan_weight,
            w_anchor=args.geo_anchor_weight,
            tau0=args.geo_tau0,
            tau1=args.geo_tau1,
            geo_hold=args.geo_hold,
            geo_decay=args.geo_decay,
            cov_subsample=args.geo_cov_subsample,
            chunk=args.geo_chunk,
        )
        initial_control_points = torch.stack(
            [path.points.detach().clone() for path in renderer.shapes]
        )
    print("Starting the optimization process", flush=True)

    semantic_analyzer = None
    semantic_checkpoints = set()
    if (
        getattr(args, "init_placement", "kmeans") == "semantic"
        and renderer.semantic_stroke_parts
        and renderer.semantic_part_contours
    ):
        semantic_analyzer = SemanticStrokeAnalyzer(
            shapes=renderer.shapes,
            assigned_parts=renderer.semantic_stroke_parts,
            part_contours=renderer.semantic_part_contours,
            canvas_width=renderer.canvas_width,
            canvas_height=renderer.canvas_height,
            num_samples=100,
        )
        semantic_checkpoints = set(analysis_checkpoint_steps(args.num_iter))
        semantic_analyzer.capture(0, renderer.shapes)
    
    epoch_range = tqdm(range(args.num_iter + 1))
    loss_history = []
    grad_history = []
    g_sds_ema = None
    g_geo_ema = None
    save_grad_meta(args, renderer, geo_crit, reference)

    inputs = inputs.detach()
    for epoch in epoch_range:
        if (
            semantic_analyzer is not None
            and epoch in semantic_checkpoints
            and epoch != 0
        ):
            semantic_analyzer.capture(epoch, renderer.shapes)
        optimizer.zero_grad_()
        sketches = renderer.get_image().to(args.device)
        sds_loss_value = sds_loss(sketches)

        geo_loss_value = sds_loss_value.new_zeros(())
        geo_logs = None
        if geo_crit is not None:
            control_points = torch.stack(
                [path.points for path in renderer.shapes]
            )
            geo_loss_value, geo_logs = geo_crit(
                control_points,
                step=epoch,
                P_init=initial_control_points,
            )

        configured_sdt_weight = getattr(
            args,
            "semantic_sdt_loss_weight",
            0.0,
        )
        if geo_crit is not None:
            configured_sdt_weight = 0.0

        effective_sdt_weight = configured_sdt_weight
        sdt_loss_raw = torch.zeros(
            (),
            device=args.device,
            dtype=sds_loss_value.dtype,
        )

        if configured_sdt_weight > 0:
            ramp_iters = getattr(
                args,
                "semantic_sdt_loss_ramp_iters",
                0,
            )

            if ramp_iters > 0:
                ramp_ratio = min(
                    1.0,
                    epoch / float(ramp_iters),
                )
                effective_sdt_weight *= ramp_ratio

        configured_dir_weight = getattr(
            args,
            "semantic_dir_loss_weight",
            0.0,
        )
        if geo_crit is not None:
            configured_dir_weight = 0.0
        effective_dir_weight = configured_dir_weight
        dir_loss_raw = torch.zeros(
            (),
            device=args.device,
            dtype=sds_loss_value.dtype,
        )

        if configured_dir_weight > 0:
            ramp_iters = getattr(
                args,
                "semantic_dir_loss_ramp_iters",
                0,
            )

            if ramp_iters > 0:
                ramp_ratio = min(
                    1.0,
                    epoch / float(ramp_iters),
                )
                effective_dir_weight *= ramp_ratio

        if configured_sdt_weight > 0 or configured_dir_weight > 0:
            sdt_loss_raw, dir_loss_raw = (
                renderer.semantic_geometry_losses(
                    compute_sdt=configured_sdt_weight > 0,
                    compute_dir=configured_dir_weight > 0,
                )
            )

        sdt_loss_weighted = effective_sdt_weight * sdt_loss_raw
        dir_loss_weighted = effective_dir_weight * dir_loss_raw
        loss = (
            sds_loss_value
            + sdt_loss_weighted
            + dir_loss_weighted
            + geo_loss_value
        )

        (
            sds_loss_log,
            sdt_loss_raw_log,
            sdt_loss_weighted_log,
            dir_loss_raw_log,
            dir_loss_weighted_log,
            total_loss_log,
        ) = torch.stack(
            [
                sds_loss_value,
                sdt_loss_raw,
                sdt_loss_weighted,
                dir_loss_raw,
                dir_loss_weighted,
                loss,
            ]
        ).detach().cpu().tolist()

        if geo_logs is None:
            geo_logs = {
                "tau": 0.0,
                "w_geo": 0.0,
                "w_anc": 0.0,
                "pos": 0.0,
                "cov": 0.0,
                "tan": 0.0,
                "anc": 0.0,
                "total": 0.0,
            }

        loss_history.append({
            "epoch": epoch,
            "sds_loss": sds_loss_log,
            "sdt_loss_raw": sdt_loss_raw_log,
            "sdt_weight": float(effective_sdt_weight),
            "sdt_loss_weighted": sdt_loss_weighted_log,
            "dir_loss_raw": dir_loss_raw_log,
            "dir_weight": float(effective_dir_weight),
            "dir_loss_weighted": dir_loss_weighted_log,
            "geo_tau": geo_logs["tau"],
            "geo_weight": geo_logs["w_geo"],
            "anchor_weight": geo_logs["w_anc"],
            "pos_loss": geo_logs["pos"],
            "cov_loss": geo_logs["cov"],
            "tan_loss": geo_logs["tan"],
            "anchor_loss": geo_logs["anc"],
            "geo_loss_total": geo_logs["total"],
            "total_loss": total_loss_log,
        })

        if args.use_wandb:
            wandb.log(
                {
                    "loss/total": loss_history[-1]["total_loss"],
                    "loss/sds": loss_history[-1]["sds_loss"],
                    "loss/sdt_raw": loss_history[-1]["sdt_loss_raw"],
                    "loss/sdt_weighted": (
                        loss_history[-1]["sdt_loss_weighted"]
                    ),
                    "loss/sdt_weight": (
                        loss_history[-1]["sdt_weight"]
                    ),
                    "loss/dir_raw": loss_history[-1]["dir_loss_raw"],
                    "loss/dir_weighted": (
                        loss_history[-1]["dir_loss_weighted"]
                    ),
                    "loss/dir_weight": (
                        loss_history[-1]["dir_weight"]
                    ),
                    "loss/geo_tau": loss_history[-1]["geo_tau"],
                    "loss/geo_weight": loss_history[-1]["geo_weight"],
                    "loss/anchor_weight": (
                        loss_history[-1]["anchor_weight"]
                    ),
                    "loss/pos_raw": loss_history[-1]["pos_loss"],
                    "loss/cov_raw": loss_history[-1]["cov_loss"],
                    "loss/tan_raw": loss_history[-1]["tan_loss"],
                    "loss/anchor_raw": loss_history[-1]["anchor_loss"],
                    "loss/geo_total": (
                        loss_history[-1]["geo_loss_total"]
                    ),
                },
                step=epoch,
            )


        should_record_grad = (
            args.grad_log
            and epoch < args.num_iter
            and epoch % args.grad_log_every == 0
        )
        should_summarize_grad = (
            args.grad_log
            and epoch < args.num_iter
            and epoch % 50 == 0
        )
        if should_record_grad or should_summarize_grad:
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(args.device)
                if args.device.type == "cuda"
                else None
            )
            numpy_rng_state = np.random.get_state()
            python_rng_state = random.getstate()
            grad_logs = grad_norm_ratio(
                renderer.get_points_parans(), sds_loss_value, geo_loss_value
            )
            if g_sds_ema is None:
                g_sds_ema = grad_logs["g_sds"]
                g_geo_ema = grad_logs["g_geo"]
            else:
                g_sds_ema = 0.9 * g_sds_ema + 0.1 * grad_logs["g_sds"]
                g_geo_ema = 0.9 * g_geo_ema + 0.1 * grad_logs["g_geo"]

            if should_record_grad:
                prf = {"precision": "", "recall": "", "f": ""}
                if epoch % 100 == 0 and geo_crit is not None:
                    prf = contour_prf(control_points, reference, sigma=4.0)
                grad_history.append({
                    "step": epoch,
                    **grad_logs,
                    "g_sds_ema": g_sds_ema,
                    "g_geo_ema": g_geo_ema,
                    "tau": geo_logs["tau"],
                    "w_geo": geo_logs["w_geo"],
                    "w_anc": geo_logs["w_anc"],
                    "pos": geo_logs["pos"],
                    "cov": geo_logs["cov"],
                    "tan": geo_logs["tan"],
                    "anc": geo_logs["anc"],
                    "total": geo_logs["total"],
                    **prf,
                })

            if should_summarize_grad:
                print(
                    f"[grad] step={epoch} "
                    f"g_sds={grad_logs['g_sds']:.6g} "
                    f"g_geo={grad_logs['g_geo']:.6g} "
                    f"ratio={grad_logs['ratio']:.6g} "
                    f"cos={grad_logs['cos']:.6g} "
                    f"tau={geo_logs['tau']:.6g} "
                    f"w_geo={geo_logs['w_geo']:.6g} "
                    f"w_anc={geo_logs['w_anc']:.6g}",
                    flush=True,
                )

            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, args.device)
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)

        loss.backward()
        optimizer.step_()


        if epoch % args.save_interval == 0:
            renderer.save_svg(
                f"{args.output_dir}/svg_logs",
                f"svg_iter{epoch}",
            )
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
            renderer.save_semantic_part_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}_semantic_parts")
            if not os.path.exists(f"{args.output_dir}/svg_to_png"):
                os.mkdir(f"{args.output_dir}/svg_to_png")
            path_svg = f"{args.output_dir}/svg_logs/svg_iter{epoch}.svg"
            sketch_iter = utils.read_svg(path_svg, args.device, multiply=True, args=args).cpu().numpy()
            sketch_iter = Image.fromarray((sketch_iter * 255).astype('uint8'), 'RGB')
            sketch_iter.save("{0}/{1}/iter_{2:04}.png".format(args.output_dir, "svg_to_png", int(epoch)))
            if args.use_wandb:
                sketch_array = np.array(sketch_iter)
                wandb.log({"cur_sketch": wandb.Image(sketch_array)}, step=epoch)


        if epoch == 0 and args.use_init_method:
            utils.plot_initial_points(renderer.get_attn_map_to_plot(), renderer.get_clustered_mask(), inputs, renderer.get_inds(),
                            args.use_wandb, "{}/{}.jpg".format(
                args.output_dir, "initial_points"))

    if args.grad_log:
        save_grad_history(grad_history, args.output_dir)

    save_loss_history(
        loss_history,
        args.output_dir,
    )

    if semantic_analyzer is not None:
        # The existing loop is inclusive (num_iter + 1 updates). Replace the
        # 100% row with the actual post-loop stroke used for final output.
        semantic_analyzer.capture(args.num_iter, renderer.shapes)
        semantic_analyzer.save(args.output_dir)

    # save final sketch
    if args.sort_final_sketch:
        utils.sort_by_contour_and_attn(renderer, args.mask, renderer.get_attn())
    if (hasattr(args, 'scale_w') or hasattr(args, 'scale_h')): 
        #Increases the size of the object on the canvas to its original size if it has been reduced
        utils.increase_object_size(renderer, args.scale_w, args.scale_h, args.original_center_x, args.original_center_y)
    if args.output_svg_size!=512:
        utils.resize_svg(renderer, args.output_svg_size, args.output_svg_size)
    renderer.save_svg(args.output_dir, "final_svg")
    semantic_svg_saved = renderer.save_semantic_part_svg(
        args.output_dir,
        "final_svg_semantic_parts",
    )
    if semantic_svg_saved:
        semantic_sketch_num = utils.read_svg(
            f"{args.output_dir}/final_svg_semantic_parts.svg",
            args.device,
            multiply=False,
            args=None,
        ).cpu().numpy()
        semantic_sketch = Image.fromarray(
            (semantic_sketch_num * 255).astype("uint8"),
            "RGB",
        )
        semantic_sketch = add_semantic_legend(
            semantic_sketch,
            renderer.semantic_stroke_parts,
        )
        semantic_sketch.save(
            Path(args.output_dir) / "final_sketch_semantic_parts.png"
        )
    final_sketch_num = utils.read_svg(f"{args.output_dir}/final_svg.svg", args.device, multiply=False,
                                      args=None).cpu().numpy()
    final_sketch = Image.fromarray((final_sketch_num * 255).astype('uint8'), 'RGB')
    final_sketch_path = Path(args.output_dir) / "final_sketch.png"
    final_sketch.save(final_sketch_path)
    final_sketch = _compute_and_annotate_clip_score(
        args,
        final_sketch_path,
        final_sketch,
    )

    print(f"You can download the result sketch from {final_sketch_path}")
    if args.use_wandb:
        final_sketch = np.array(final_sketch)
        wandb.log({f"final sketch": wandb.Image(final_sketch)})


    # save video
    utils.make_video(args.output_dir)
    if args.use_wandb:
        video_path = f"{args.output_dir}/sketch.mp4"
        wandb.log({"sketch_video": wandb.Video(video_path, format="mp4")})


    if args.target_is_dict  and args.save_svg_in_dict:
        # saves the result svg into the dict
        final_key = f'svg_{args.num_strokes}s'
    
        with open(f"{args.output_dir}/final_svg.svg", 'r') as svg_file:
            svg_content = svg_file.read()

        if os.path.splitext(os.path.basename(args.target))[-1] == ".npy":
            data = np.load(args.target, allow_pickle=True).item()
            data[final_key] = svg_content
            np.save(args.target, data)
        else:
            data = dict(np.load(args.target, allow_pickle=True))
            data[final_key] = svg_content
            np.savez_compressed(args.target, **data)



            
        print("The final SVG was saved to the input dictionary")


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
