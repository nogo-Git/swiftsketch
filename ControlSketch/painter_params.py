import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.cluster import KMeans
import sketch_utils as utils
import inversion
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import matplotlib.pyplot as plt
from attn_utils import (
    upscale,
    resize_net_attn_map,
    return_net_attn_map,
)
import semantic_init
from semantic_colors import semantic_part_rgba
import semantic_segmenter
import os
from scipy.ndimage import distance_transform_edt


class Painter(torch.nn.Module):
    def __init__(self, args,
                 num_strokes=4,
                 num_segments=4,
                 device=None,
                 target_im=None,
                 mask=None,
                 ):
        super(Painter, self).__init__()

        self.args = args
        self.render_size = args.render_size
        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.mask = mask
        self.strokes_counter = 0  
        self.shapes = []
        self.shape_groups = []
        self.device = device
        self.canvas_width, self.canvas_height = args.render_size, args.render_size
        self.points_vars = []
        self.optimize_flag = []
        self.initial_points = []
        self.semantic_stroke_parts = None
        self.semantic_part_contours = None
        self.semantic_part_masks = {}
        self.semantic_sdt_maps = {}
        self.semantic_sdt_gradient_maps = {}
        self.semantic_sdt_penalty_maps = {}
        self.semantic_geometry_maps = {}

        # attention related for strokes initialisation
        self.use_init_method = args.use_init_method
        self.target_im= target_im
        self.target_path = args.target
        self.attn_model = args.attn_model
        self.attention_map = self.set_attention_map() if self.use_init_method else None
        self.attn_map_to_plot = self.set_attention_threshold_map() if self.use_init_method else None
        

    def init_image(self):
        for i in range(self.num_paths):  
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path() 
            self.shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                fill_color=None,
                                                stroke_color=stroke_color)
            self.shape_groups.append(path_group)
        self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp() 
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
                1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # HWC -> NCHW
        return img
        

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # HWC -> NCHW
        return img

    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.use_init_method else (random.random(), random.random())
        self.initial_points.append(p0)
        points.append(p0) 
        for j in range(self.num_segments):  # here is 1 by defult
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                self.initial_points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)  

        self.strokes_counter += 1
        return path

    def render_warp(self):  
        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = _render(self.canvas_width,  
                      self.canvas_height, 
                      2,  
                      2,  
                      0,  
                      None,
                      *scene_args)
        return img

    def parameters(self):
        self.points_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.points_vars.append(path.points)
        return self.points_vars

    def get_points_parans(self):
        return self.points_vars
    

    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes,
                          self.shape_groups)

    def save_semantic_part_svg(self, output_dir, name):
        if not self.semantic_stroke_parts:
            return False

        semantic_shape_groups = []

        for i, part in enumerate(self.semantic_stroke_parts):
            semantic_shape_groups.append(
                pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([i]),
                    fill_color=None,
                    stroke_color=torch.tensor(semantic_part_rgba(part)),
                )
            )

        pydiffvg.save_svg(
            "{}/{}.svg".format(output_dir, name),
            self.canvas_width,
            self.canvas_height,
            self.shapes,
            semantic_shape_groups,
        )
        return True

    def get_initial_points(self):
        return torch.tensor(self.initial_points)

 

    def define_clip_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        image_input_attn_clip= data_transforms(target_im).to(self.device)
        image_input_attn_clip = F.interpolate(image_input_attn_clip, size=(224, 224), mode='bicubic', align_corners=False)
        self.image_input_attn_clip = image_input_attn_clip
    
    def interpret(self, image, model, device):
        images = image.repeat(1, 1, 1, 1)
        res = model.encode_image(images)
        model.zero_grad()
        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
        cams = []  # there are 12 attention blocks
        for i, blk in enumerate(image_attn_blocks):
            cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
            # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
            cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0)
            cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
            cams.append(cam)
            R = R + torch.bmm(cam, R)

        cams_avg = torch.cat(cams)  # 12, 50, 50
        cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
        image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
        image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
        return image_relevance



    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        attn_map = self.interpret(self.image_input_attn_clip, model, device=self.device)
        attn_map = torch.from_numpy(attn_map)
        del model
        torch.cuda.empty_cache()
        return attn_map
    
 
    

    def diffusion_attn(self):
        # DDIM inversion
        num_inference_steps = 50
        # orig_image= Image.open(self.args.target).resize((1024, 1024))
        orig_image= self.args.input_image.resize((1024, 1024))
        
        x0 = np.array(orig_image)
        caption = f"a portrait of a {self.args.object_name}"

        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False)

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True,
            scheduler=scheduler
        ).to(self.device)


        zts = inversion.ddim_inversion(pipeline, x0, caption, num_inference_steps, 2)

        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)

        # Register the custom attention processor and get the attn_maps list
        pipeline.unet, attn_maps = register_attention_store(pipeline.unet)
        pipeline = pipeline.to(self.device)

        g_cpu = torch.Generator(device='cpu')
        g_cpu.manual_seed(10)

        latents = torch.randn(1, 4, 128, 128, device='cpu', generator=g_cpu,
                            dtype=pipeline.unet.dtype, ).to(self.device)
        
        latents[0] = zT

        image = pipeline(caption, latents=latents,
                        callback_on_step_end=inversion_callback,
                        num_inference_steps=num_inference_steps, guidance_scale=10.0).images[0]

        attn_map = inference_and_extract_attn(attn_maps, caption, pipeline, image, self.args.object_name)
        attn_map= torch.pow(attn_map, 2)
        
        del latents, zts, zT, image, attn_maps, inversion_callback
        del pipeline
        torch.cuda.empty_cache()
        return attn_map


    def set_attention_map(self):
        if hasattr(self.args, 'attn_from_dict'): 
            attn = self.args.attn_from_dict
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
            if hasattr(self.args, 'obj_bb'): 
                attn = np.stack([attn] * 3, axis=-1)
                x0, x1, y0, y1= self.args.obj_bb
                attn = utils.cut_and_resize(attn, x0, x1, y0, y1, self.args.new_height, self.args.new_width, "mask")
                attn = torch.from_numpy(attn[:, :, 0])
        elif self.attn_model == "clip" or self.args.object_name == "" :
            self.saliency_clip_model = "ViT-B/32"
            self.define_clip_attention_input(self.target_im)
            attn= self.clip_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        else: # self.attn_model == "diffusion":
            attn= self.diffusion_attn()
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), (self.render_size, self.render_size))[0][0]
        return attn
    
    def weighted_kmeans_segmentation(self, mask, weights, num_regions, spatial_weight=1.0, weight_scale=1.0, max_iter=300):
        """
        Perform K-means clustering with spatial and single-channel weight features.
        
        Parameters:
        - mask: Binary mask where 1 indicates the region of interest.
        - weights: A single-channel image (grayscale or weights) defining pixel-level features.
        - num_regions: Number of regions (clusters) to create.
        - spatial_weight: Weight factor for spatial coordinates.
        - weight_scale: Scale factor for the weights.
        - max_iter: Maximum iterations for K-means.

        Returns:
        - segmented_image: Image with regions visualized as distinct colors.
        - labels: Cluster labels for each pixel in the region.
        """
        # Get coordinates of the valid region
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        # Get weight values for valid region
        pixel_weights = weights[y_coords, x_coords]

        # Scale spatial and weight features
        spatial_features = valid_coords * spatial_weight
        weight_features = pixel_weights[:, np.newaxis] * weight_scale

        # Combine spatial and weight features
        features = np.hstack((spatial_features, weight_features))

        # Run K-means clustering
        kmeans = KMeans(n_clusters=num_regions, max_iter=max_iter, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Create an output image with unique colors for each cluster
        segmented_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        unique_colors = plt.cm.tab10(np.linspace(0, 1, num_regions))[:, :3] * 255  # Use Matplotlib's Tab10 colormap


        for i, (x, y) in enumerate(valid_coords):
            segmented_image[y, x] = unique_colors[labels[i]]

        return segmented_image, labels

    def distribute_points(self, labels, weights, mask, total_points):
        """
        Distribute points among regions proportionally based on region scores.

        Parameters:
        - labels: 1D array of region labels for valid pixels.
        - weights: Full 2D array of pixel weights (grayscale or other).
        - mask: Binary mask where 1 indicates the region of interest.
        - total_points: Total number of points to distribute.

        Returns:
        - points_per_region: List of number of points for each region.
        """
        # Extract valid pixel coordinates from the mask
        y_coords, x_coords = np.where(mask > 0)

        # Initialize scores for each region
        num_regions = labels.max() + 1  # Number of unique regions
        region_scores = np.zeros(num_regions)

        # Calculate scores for each region based on the weight values
        for region_id in range(num_regions):
            region_mask = (labels == region_id)  # Mask for current region
            region_weights = weights[y_coords[region_mask], x_coords[region_mask]]
            region_scores[region_id] = region_weights.sum()  # Sum of weights in the region

        # Normalize scores to distribute points proportionally
        total_score = region_scores.sum()
        points_per_region = (region_scores / total_score * total_points).round().astype(int)

        # Adjust to ensure the total matches exactly
        while points_per_region.sum() < total_points:
            points_per_region[np.argmax(region_scores)] += 1
        while points_per_region.sum() > total_points:
            points_per_region[np.argmax(points_per_region)] -= 1

        return points_per_region
    

    def generate_kmeans_points(self, mask, num_points, max_iter=300):
        """
        Generate equidistributed points on an arbitrary shape using K-means clustering.
        
        Parameters:
        - mask: A 2D numpy array (binary mask) where 1 indicates the shape and 0 is the background.
        - num_points: Number of points to generate.
        - max_iter: Maximum iterations for K-means clustering.
        
        Returns:
        - points: Array of shape (num_points, 2) with the coordinates of the points.
        """
        # Get coordinates of the valid region
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        # Run K-means clustering on the valid coordinates
        kmeans = KMeans(n_clusters=num_points, max_iter=max_iter, random_state=42, n_init=10)
        kmeans.fit(valid_coords)

        # Cluster centers are the resulting equidistributed points
        points = kmeans.cluster_centers_

        # Clip points to the canvas first. A K-means centroid can still fall
        # outside a non-convex region even when all assigned samples are inside.
        points = np.clip(points, [0, 0], [mask.shape[1] - 1, mask.shape[0] - 1])

        # Keep exactly num_points entries. Discarding an invalid centroid here
        # leaves fewer initial points than strokes and later makes get_path()
        # index past the end of inds_normalised.
        pixel_points = np.rint(points).astype(np.int64)
        pixel_points[:, 0] = np.clip(pixel_points[:, 0], 0, mask.shape[1] - 1)
        pixel_points[:, 1] = np.clip(pixel_points[:, 1], 0, mask.shape[0] - 1)
        inside_mask = mask[pixel_points[:, 1], pixel_points[:, 0]] > 0

        for point_index in np.flatnonzero(~inside_mask):
            offsets = valid_coords - points[point_index]
            nearest_index = np.argmin(np.einsum("ij,ij->i", offsets, offsets))
            points[point_index] = valid_coords[nearest_index]

        return points


    def distribute_and_visualize_points(self, segmented_image, labels, points_per_region, mask):
        """
        Apply K-means to distribute points in each region and visualize them on the segmented image.

        Parameters:
        - segmented_image: Image with regions visualized as distinct colors.
        - labels: 1D array of region labels for valid pixels.
        - points_per_region: Number of points to distribute in each region.
        - mask: Binary mask (H x W) defining the valid region of interest.

        Returns:
        - combined_points: List of all (x, y) coordinates for distributed points.
        """
        num_regions = len(points_per_region)
        combined_points = []

        # Get valid pixel coordinates
        y_coords, x_coords = np.where(mask > 0)
        valid_coords = np.column_stack((x_coords, y_coords))

        for region_id, num_points in enumerate(points_per_region):
            if num_points > 0:
                # Create a binary mask for the current region
                region_mask = np.zeros_like(mask, dtype=np.uint8)
                region_pixel_indices = np.where(labels == region_id)
                region_pixels = valid_coords[region_pixel_indices]
                region_mask[region_pixels[:, 1], region_pixels[:, 0]] = 1

                # Generate points using K-means
                points = self.generate_kmeans_points(region_mask, num_points)
                combined_points.extend(points)

        segmented_image_ = segmented_image.copy()
        segmented_image_[mask == 0] = 255
        combined_points = np.array(combined_points)
        return combined_points, segmented_image_
    

    def get_points_smart_clustering(self, mask, weights):
        all_points = self.num_paths
        num_regions = 6  # Number of regions to divide

        # Keep a small guaranteed allocation per region, but never exceed all_points.
        base_points_per_region = min(3, all_points // num_regions)
        remain_points = all_points - (base_points_per_region * num_regions)

        spatial_weight = 1.0  # Weight for spatial coordinates
        weight_scale = 0.5  # Scale for pixel weights (e.g., intensity)

        segmented_image, labels = self.weighted_kmeans_segmentation(
            mask, weights, num_regions, spatial_weight, weight_scale
        )

        extra_points_per_region = self.distribute_points(
            labels, weights, mask, remain_points
        )

        final_points_per_region = extra_points_per_region + base_points_per_region

        combined_points, segmented_image_ = self.distribute_and_visualize_points(
            segmented_image, labels, final_points_per_region, mask
        )
        return combined_points, segmented_image_


    def set_attention_threshold_map(self):
        attn_map = torch.pow(self.attention_map, 2)
        attn_map_to_plot = (attn_map * self.mask)

        weights = attn_map.detach().cpu().numpy().astype(np.float32)

        mask = self.mask
        mask = (mask / mask.max()) * 255
        mask = mask.detach().cpu().numpy().astype(np.uint8)

        if getattr(self.args, "init_placement", "kmeans") == "semantic":
            part_masks = None

            parts_text = getattr(self.args, "semantic_parts", "outline")
            weights_text = getattr(self.args, "semantic_weights", "")
            part_queries = {}

            self.semantic_reference_cache = None
            self.semantic_cache_info = {}
            semantic_prepared = False

            if parts_text.strip().lower() == "auto":
                from semantic_pipeline import prepare_auto_semantic_data

                (
                    enumeration,
                    part_masks,
                    self.semantic_reference_cache,
                    self.semantic_cache_info,
                ) = prepare_auto_semantic_data(
                    args=self.args,
                    image=self.args.input_image,
                    foreground_mask=self.mask,
                    device=self.device,
                )
                semantic_prepared = True
                parts_text = enumeration.parts_text
                weights_text = enumeration.weights_text
                part_queries = enumeration.part_queries
                print(f"[semantic_vlm] parts={parts_text}", flush=True)
                print(f"[semantic_vlm] weights={weights_text}", flush=True)

            self.semantic_parts_text = parts_text
            self.semantic_weights_text = weights_text
            parts = semantic_init.parse_semantic_parts(parts_text)
            non_outline_parts = [part for part in parts if part != "outline"]

            segmenter_name = getattr(self.args, "semantic_segmenter", "sam3")

            if (
                non_outline_parts
                and segmenter_name != "none"
                and not semantic_prepared
            ):
                try:
                    if segmenter_name == "sam3":
                        sam3_python = getattr(self.args, "sam3_python", "")
                        if not sam3_python:
                            raise ValueError("--sam3_python must point to the Python executable in sam3_env")

                        segmenter = semantic_segmenter.SAM3SubprocessSegmenter(
                            sam3_python=sam3_python,
                            checkpoint_path=getattr(self.args, "sam3_checkpoint_path", ""),
                            confidence_threshold=getattr(self.args, "sam3_confidence_threshold", 0.5),
                            min_area_ratio=getattr(self.args, "semantic_min_area_ratio", 0.0002),
                            max_masks_per_part=getattr(self.args, "semantic_max_masks_per_part", 4),
                            debug_dir=os.path.join(self.args.output_dir, "semantic_debug"),
                            max_part_area_ratio=getattr(self.args, "semantic_max_part_area_ratio", 0.25),
                        )

                    elif segmenter_name == "grounded_sam":
                        segmenter = semantic_segmenter.GroundedSAMSegmenter(
                            device=self.device,
                            grounding_model_id=getattr(
                                self.args,
                                "grounding_dino_model",
                                "IDEA-Research/grounding-dino-base",
                            ),
                            sam_model_id=getattr(
                                self.args,
                                "sam_model",
                                "facebook/sam-vit-base",
                            ),
                            box_threshold=getattr(self.args, "grounding_box_threshold", 0.25),
                            text_threshold=getattr(self.args, "grounding_text_threshold", 0.20),
                            min_area_ratio=getattr(self.args, "semantic_min_area_ratio", 0.0002),
                            max_masks_per_part=getattr(self.args, "semantic_max_masks_per_part", 4),
                            debug_dir=os.path.join(self.args.output_dir, "semantic_debug"),
                            max_box_object_area_ratio=getattr(self.args, "grounding_max_box_object_area_ratio", 0.35),
                            max_part_area_ratio=getattr(self.args, "semantic_max_part_area_ratio", 0.25),
                            prefer_small_boxes=getattr(self.args, "grounding_prefer_small_boxes", 1) == 1,
                        )

                    else:
                        raise ValueError(f"Unknown semantic segmenter: {segmenter_name}")

                    part_masks = segmenter.segment_parts(
                        image=self.args.input_image,
                        parts=non_outline_parts,
                        foreground_mask=self.mask,
                        object_name=getattr(self.args, "object_name", ""),
                        part_queries=part_queries,
                    )

                except Exception as err:
                    print(f"{segmenter_name} segmentation failed: {err}", flush=True)
                    part_masks = None

            result = semantic_init.build_semantic_initial_points(
                mask=self.mask,
                total_points=self.num_paths,
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
                parts_text=parts_text,
                weights_text=weights_text,
                part_masks=part_masks,
                min_perimeter=getattr(self.args, "semantic_min_perimeter", 8.0,),
                outline_overlap_tolerance=getattr(self.args, "semantic_outline_overlap_tolerance", 4.0),
                curvature_sampling=getattr(self.args, "semantic_curvature_sampling", 0) == 1,
                curvature_weight=getattr(self.args, "semantic_curvature_weight", 2.0),
                curvature_window=getattr(self.args, "semantic_curvature_window", 6),
                min_sampling_density=getattr(self.args, "semantic_min_sampling_density", 0.20),
                return_metadata=True,
                return_analysis_contours=True,
            )

            if result is not None:
                (
                    self.inds,
                    self.clustered_mask_to_plot,
                    self.semantic_stroke_parts,
                    part_masks_for_loss,
                    self.semantic_part_contours,
                ) = result
                self.semantic_part_masks = part_masks_for_loss
                self._prepare_semantic_sdt_maps(part_masks_for_loss)
            elif getattr(self.args, "semantic_fallback", "kmeans") == "kmeans":
                self.inds, self.clustered_mask_to_plot = self.get_points_smart_clustering(mask, weights)
            else:
                raise RuntimeError("Semantic initialization failed and fallback is disabled.")
        else:
            self.inds, self.clustered_mask_to_plot = self.get_points_smart_clustering(mask, weights)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 0] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 1] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()

        return attn_map_to_plot
    
    def _prepare_semantic_sdt_maps(self, part_masks):
        self.semantic_sdt_maps = {}
        self.semantic_sdt_gradient_maps = {}
        self.semantic_sdt_penalty_maps = {}
        self.semantic_geometry_maps = {}

        for part, mask in part_masks.items():
            binary = np.asarray(mask).astype(np.uint8) > 0

            outside = distance_transform_edt(~binary)
            inside = distance_transform_edt(binary)
            sdt = outside - inside
            grad_y, grad_x = np.gradient(sdt)
            sdt_tensor = torch.from_numpy(sdt).float().to(self.device)
            gradient_tensor = torch.from_numpy(
                np.stack([grad_x, grad_y], axis=0)
            ).float().to(self.device)

            outside_margin = getattr(
                self.args,
                "semantic_sdt_outside_margin",
                getattr(self.args, "semantic_sdt_loss_margin", 2.0),
            )
            inside_margin = getattr(
                self.args,
                "semantic_sdt_inside_margin",
                4.0,
            )
            inside_weight = getattr(
                self.args,
                "semantic_sdt_inside_weight",
                0.5,
            )
            stroke_radius = float(self.width) * 0.5
            penalty = (
                torch.relu(sdt_tensor + stroke_radius - outside_margin)
                + inside_weight * torch.relu(-sdt_tensor - inside_margin)
            ) / max(self.canvas_width, self.canvas_height)

            self.semantic_sdt_maps[part] = sdt_tensor
            self.semantic_sdt_gradient_maps[part] = gradient_tensor
            self.semantic_sdt_penalty_maps[part] = penalty
            self.semantic_geometry_maps[part] = torch.cat(
                [penalty.unsqueeze(0), gradient_tensor],
                dim=0,
            )
            
        self.save_semantic_sdt_debug()
    
    def save_semantic_sdt_debug(self):
        if not self.semantic_sdt_maps:
            return

        debug_dir = os.path.join(self.args.output_dir, "semantic_sdt_debug")
        os.makedirs(debug_dir, exist_ok=True)

        outside_margin = getattr(
            self.args,
            "semantic_sdt_outside_margin",
            getattr(self.args, "semantic_sdt_loss_margin", 2.0),
        )
        inside_margin = getattr(self.args, "semantic_sdt_inside_margin", 4.0)
        inside_weight = getattr(self.args, "semantic_sdt_inside_weight", 0.5)

        stroke_radius = float(self.width) * 0.5

        for part, sdt in self.semantic_sdt_maps.items():
            sdt_np = sdt.detach().cpu().numpy()

            max_abs = np.percentile(np.abs(sdt_np), 95)
            max_abs = max(max_abs, 1e-6)

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(sdt_np, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)

            if sdt_np.min() <= 0 <= sdt_np.max():
                ax.contour(sdt_np, levels=[0], colors="black", linewidths=1)

            fig.colorbar(im, ax=ax, label="signed distance")
            ax.set_title(f"{part} SDT")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, f"{part}_sdt.png"), dpi=160)
            plt.close(fig)

            outside_penalty = np.maximum(sdt_np + stroke_radius - outside_margin, 0.0)
            inside_penalty = np.maximum(-sdt_np - inside_margin, 0.0)
            penalty = outside_penalty + inside_weight * inside_penalty

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(penalty, cmap="magma")

            if sdt_np.min() <= 0 <= sdt_np.max():
                ax.contour(sdt_np, levels=[0], colors="white", linewidths=1)

            fig.colorbar(im, ax=ax, label="penalty")
            ax.set_title(f"{part} SDT penalty")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(debug_dir, f"{part}_penalty.png"), dpi=160)
            plt.close(fig)

            grad_y, grad_x = np.gradient(sdt_np)
            norm = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6
            inward_x = -grad_x / norm
            inward_y = -grad_y / norm

            step = max(8, sdt_np.shape[0] // 32)
            yy, xx = np.mgrid[0:sdt_np.shape[0]:step, 0:sdt_np.shape[1]:step]

            plt.figure(figsize=(6, 6))
            plt.imshow(sdt_np, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
            plt.contour(sdt_np, levels=[0], colors="black", linewidths=1)
            plt.quiver(
                xx,
                yy,
                inward_x[::step, ::step],
                inward_y[::step, ::step],
                color="yellow",
                angles="xy",
                scale_units="xy",
                scale=0.25,
                width=0.003,
            )
            plt.title(f"{part} inward SDT gradient")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, f"{part}_gradient.png"), dpi=160)
            plt.close()


    def _render_shape_subset_alpha(self, indices):
        shapes = [self.shapes[i] for i in indices]
        shape_groups = [
            pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([j]),
                fill_color=None,
                stroke_color=self.shape_groups[i].stroke_color,
            )
            for j, i in enumerate(indices)
        ]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_width,
            self.canvas_height,
            shapes,
            shape_groups,
        )

        img = pydiffvg.RenderFunction.apply(
            self.canvas_width,
            self.canvas_height,
            2,
            2,
            0,
            None,
            *scene_args,
        )

        return img[:, :, 3]


    def _sample_path_points(self, path, samples_per_segment=16):
        points = path.points
        num_control_points = path.num_control_points.detach().cpu().tolist()

        t = torch.linspace(
            0.0,
            1.0,
            samples_per_segment,
            device=points.device,
            dtype=points.dtype,
        )

        sampled = []
        point_offset = 0

        for n_ctrl in num_control_points:
            segment_points = points[point_offset: point_offset + n_ctrl + 2]

            curve = segment_points.unsqueeze(0).expand(samples_per_segment, -1, -1)
            for _ in range(segment_points.shape[0] - 1):
                curve = (1.0 - t[:, None, None]) * curve[:, :-1] + t[:, None, None] * curve[:, 1:]

            sampled.append(curve[:, 0])
            point_offset += n_ctrl + 1

        return torch.cat(sampled, dim=0)

    def _sample_path_points_and_tangents(self, path, samples_per_segment=16):
        points = path.points
        num_control_points = path.num_control_points.detach().cpu().tolist()

        t = torch.linspace(
            0.0,
            1.0,
            samples_per_segment,
            device=points.device,
            dtype=points.dtype,
        )

        sampled_points = []
        sampled_tangents = []
        point_offset = 0

        for n_ctrl in num_control_points:
            segment_points = points[point_offset: point_offset + n_ctrl + 2]

            curve = segment_points.unsqueeze(0).expand(
                samples_per_segment, -1, -1
            )
            for _ in range(segment_points.shape[0] - 1):
                curve = (
                    (1.0 - t[:, None, None]) * curve[:, :-1]
                    + t[:, None, None] * curve[:, 1:]
                )
            sampled_points.append(curve[:, 0])

            degree = segment_points.shape[0] - 1
            derivative_points = degree * (
                segment_points[1:] - segment_points[:-1]
            )
            derivative_curve = derivative_points.unsqueeze(0).expand(
                samples_per_segment, -1, -1
            )
            for _ in range(derivative_points.shape[0] - 1):
                derivative_curve = (
                    (1.0 - t[:, None, None]) * derivative_curve[:, :-1]
                    + t[:, None, None] * derivative_curve[:, 1:]
                )
            sampled_tangents.append(derivative_curve[:, 0])

            point_offset += n_ctrl + 1

        return (
            torch.cat(sampled_points, dim=0),
            torch.cat(sampled_tangents, dim=0),
        )

    def _sample_paths_points_and_tangents(
        self,
        paths,
        samples_per_segment,
        include_tangents=True,
    ):
        points = torch.stack([path.points for path in paths], dim=0)
        num_control_points = (
            paths[0].num_control_points.detach().cpu().tolist()
        )

        t = torch.linspace(
            0.0,
            1.0,
            samples_per_segment,
            device=points.device,
            dtype=points.dtype,
        ).view(1, samples_per_segment, 1, 1)

        sampled_points = []
        sampled_tangents = []
        point_offset = 0

        for n_ctrl in num_control_points:
            segment_points = points[
                :,
                point_offset: point_offset + n_ctrl + 2,
            ]
            curve = segment_points.unsqueeze(1).expand(
                -1,
                samples_per_segment,
                -1,
                -1,
            )
            for _ in range(segment_points.shape[1] - 1):
                curve = (
                    (1.0 - t) * curve[:, :, :-1]
                    + t * curve[:, :, 1:]
                )
            sampled_points.append(curve[:, :, 0])

            if include_tangents:
                degree = segment_points.shape[1] - 1
                derivative_points = degree * (
                    segment_points[:, 1:] - segment_points[:, :-1]
                )
                derivative_curve = derivative_points.unsqueeze(1).expand(
                    -1,
                    samples_per_segment,
                    -1,
                    -1,
                )
                for _ in range(derivative_points.shape[1] - 1):
                    derivative_curve = (
                        (1.0 - t) * derivative_curve[:, :, :-1]
                        + t * derivative_curve[:, :, 1:]
                    )
                sampled_tangents.append(derivative_curve[:, :, 0])

            point_offset += n_ctrl + 1

        points_result = torch.cat(sampled_points, dim=1)
        if not include_tangents:
            return points_result, None

        return points_result, torch.cat(sampled_tangents, dim=1)

    def _semantic_path_groups(self):
        groups = {}

        for stroke_idx, part in enumerate(self.semantic_stroke_parts or []):
            if stroke_idx >= len(self.shapes):
                continue

            path = self.shapes[stroke_idx]
            topology = tuple(
                path.num_control_points.detach().cpu().tolist()
            )
            groups.setdefault((part, topology), []).append(path)

        return groups

    @staticmethod
    def _sample_field_at_points(field_map, sampled_points):
        batch_size = sampled_points.shape[0]
        _, h, w = field_map.shape

        grid_x = 2.0 * sampled_points[:, :, 0] / max(w - 1, 1) - 1.0
        grid_y = 2.0 * sampled_points[:, :, 1] / max(h - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        sampled = F.grid_sample(
            field_map.unsqueeze(0).expand(batch_size, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def semantic_geometry_losses(
        self,
        compute_sdt=True,
        compute_dir=True,
    ):
        zero = torch.zeros((), device=self.device)
        if not self.semantic_stroke_parts:
            return zero, zero

        sdt_samples = getattr(
            self.args,
            "semantic_sdt_samples_per_segment",
            8,
        )
        dir_samples = getattr(
            self.args,
            "semantic_dir_samples_per_segment",
            8,
        )
        gradient_min_norm = getattr(
            self.args,
            "semantic_dir_gradient_min_norm",
            1e-3,
        )
        eps = 1e-6
        sdt_losses = []
        dir_losses = []

        for (part, _), paths in self._semantic_path_groups().items():
            has_sdt = (
                compute_sdt
                and part in self.semantic_sdt_penalty_maps
            )
            has_dir = (
                compute_dir
                and part in self.semantic_sdt_gradient_maps
            )
            if not has_sdt and not has_dir:
                continue

            dir_points = None
            curve_tangents = None
            if has_dir:
                dir_points, curve_tangents = (
                    self._sample_paths_points_and_tangents(
                        paths,
                        samples_per_segment=dir_samples,
                        include_tangents=True,
                    )
                )

            if has_sdt and has_dir and sdt_samples == dir_samples:
                sampled_geometry = self._sample_field_at_points(
                    self.semantic_geometry_maps[part],
                    dir_points,
                )
                sampled_penalty = sampled_geometry[:, :, 0]
                sampled_gradients = sampled_geometry[:, :, 1:]
            else:
                if has_sdt:
                    sdt_points, _ = (
                        self._sample_paths_points_and_tangents(
                            paths,
                            samples_per_segment=sdt_samples,
                            include_tangents=False,
                        )
                    )
                    sampled_penalty = self._sample_field_at_points(
                        self.semantic_sdt_penalty_maps[
                            part
                        ].unsqueeze(0),
                        sdt_points,
                    )[:, :, 0]

                if has_dir:
                    sampled_gradients = self._sample_field_at_points(
                        self.semantic_sdt_gradient_maps[part],
                        dir_points,
                    )

            if has_sdt:
                sdt_losses.append(sampled_penalty.mean(dim=1))

            if has_dir:
                field_norm = torch.linalg.vector_norm(
                    sampled_gradients,
                    dim=-1,
                )
                tangent_norm = torch.linalg.vector_norm(
                    curve_tangents,
                    dim=-1,
                )
                rotated_gradients = torch.stack(
                    [
                        -sampled_gradients[:, :, 1],
                        sampled_gradients[:, :, 0],
                    ],
                    dim=-1,
                )
                field_directions = (
                    rotated_gradients
                    / field_norm.clamp_min(eps).unsqueeze(-1)
                )
                curve_directions = (
                    curve_tangents
                    / tangent_norm.clamp_min(eps).unsqueeze(-1)
                )
                cosine = (
                    (field_directions * curve_directions).sum(dim=-1)
                ).clamp(min=-1.0, max=1.0)
                direction_penalty = 1.0 - cosine.square()
                valid = (
                    (field_norm >= gradient_min_norm)
                    & (tangent_norm >= eps)
                ).to(direction_penalty.dtype)
                dir_losses.append(
                    (direction_penalty * valid).sum(dim=1)
                    / valid.sum(dim=1).clamp_min(1.0)
                )

        sdt_loss = (
            torch.cat(sdt_losses).mean()
            if sdt_losses
            else zero
        )
        dir_loss = (
            torch.cat(dir_losses).mean()
            if dir_losses
            else zero
        )
        return sdt_loss, dir_loss


    def _semantic_sdt_loss_unbatched(self):
        if not self.semantic_stroke_parts or not self.semantic_sdt_maps:
            return torch.zeros((), device=self.device)

        outside_margin = getattr(
            self.args,
            "semantic_sdt_outside_margin",
            getattr(self.args, "semantic_sdt_loss_margin", 2.0),
        )
        inside_margin = getattr(self.args, "semantic_sdt_inside_margin", 4.0)
        inside_weight = getattr(self.args, "semantic_sdt_inside_weight", 0.5)
        samples_per_segment = getattr(self.args, "semantic_sdt_samples_per_segment", 8)

        losses = []

        for stroke_idx, part in enumerate(self.semantic_stroke_parts):
            if part not in self.semantic_sdt_maps:
                continue

            path = self.shapes[stroke_idx]
            sampled_points = self._sample_path_points(
                path,
                samples_per_segment=samples_per_segment,
            )

            sdt = self.semantic_sdt_maps[part]
            h, w = sdt.shape

            grid_x = 2.0 * sampled_points[:, 0] / max(w - 1, 1) - 1.0
            grid_y = 2.0 * sampled_points[:, 1] / max(h - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

            stroke_radius = torch.as_tensor(
                path.stroke_width,
                device=self.device,
                dtype=sdt.dtype,
            ) * 0.5

            outside_penalty = torch.relu(sdt + stroke_radius - outside_margin)
            inside_penalty = torch.relu(-sdt - inside_margin)

            penalty_map = outside_penalty + inside_weight * inside_penalty
            penalty_map = penalty_map / max(self.canvas_width, self.canvas_height)

            values = F.grid_sample(
                penalty_map.view(1, 1, h, w),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(-1)

            losses.append(values.mean())

        if not losses:
            return torch.zeros((), device=self.device)

        return torch.stack(losses).mean()

    def _semantic_dir_loss_unbatched(self):
        if (
            not self.semantic_stroke_parts
            or not self.semantic_sdt_gradient_maps
        ):
            return torch.zeros((), device=self.device)

        samples_per_segment = getattr(
            self.args,
            "semantic_dir_samples_per_segment",
            8,
        )
        gradient_min_norm = getattr(
            self.args,
            "semantic_dir_gradient_min_norm",
            1e-3,
        )
        eps = 1e-6
        losses = []

        for stroke_idx, part in enumerate(self.semantic_stroke_parts):
            if part not in self.semantic_sdt_gradient_maps:
                continue

            path = self.shapes[stroke_idx]
            sampled_points, curve_tangents = (
                self._sample_path_points_and_tangents(
                    path,
                    samples_per_segment=samples_per_segment,
                )
            )

            gradient_map = self.semantic_sdt_gradient_maps[part]
            _, h, w = gradient_map.shape

            grid_x = 2.0 * sampled_points[:, 0] / max(w - 1, 1) - 1.0
            grid_y = 2.0 * sampled_points[:, 1] / max(h - 1, 1) - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)

            sampled_gradients = F.grid_sample(
                gradient_map.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(2, -1).transpose(0, 1)

            field_norm = torch.linalg.vector_norm(
                sampled_gradients,
                dim=-1,
            )
            tangent_norm = torch.linalg.vector_norm(
                curve_tangents,
                dim=-1,
            )

            rotated_gradients = torch.stack(
                [
                    -sampled_gradients[:, 1],
                    sampled_gradients[:, 0],
                ],
                dim=-1,
            )
            field_directions = (
                rotated_gradients / field_norm.clamp_min(eps).unsqueeze(-1)
            )
            curve_directions = (
                curve_tangents / tangent_norm.clamp_min(eps).unsqueeze(-1)
            )

            cosine = (
                (field_directions * curve_directions).sum(dim=-1)
            ).clamp(min=-1.0, max=1.0)
            direction_penalty = 1.0 - cosine.square()

            valid = (
                (field_norm >= gradient_min_norm)
                & (tangent_norm >= eps)
            ).to(direction_penalty.dtype)
            losses.append(
                (direction_penalty * valid).sum()
                / valid.sum().clamp_min(1.0)
            )

        if not losses:
            return torch.zeros((), device=self.device)

        return torch.stack(losses).mean()

    def semantic_sdt_loss(self):
        sdt_loss, _ = self.semantic_geometry_losses(
            compute_sdt=True,
            compute_dir=False,
        )
        return sdt_loss

    def semantic_dir_loss(self):
        _, dir_loss = self.semantic_geometry_losses(
            compute_sdt=False,
            compute_dir=True,
        )
        return dir_loss
        
    def get_attn(self):
        return self.attention_map
    
    def get_clustered_mask(self):
        return self.clustered_mask_to_plot

    def get_attn_map_to_plot(self):
        return self.attn_map_to_plot

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask

   
class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.args = args

    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr, betas=(0.9, 0.9), eps=1e-6)

      
    def zero_grad_(self):
        self.points_optim.zero_grad()
        
    def step_(self):
        self.points_optim.step()
     
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']
    


class AttnStoreProcessor(AttnProcessor2_0):
    def __init__(self, attn_maps):
        super().__init__()
        self.attn_maps = attn_maps

    def __call__(
            self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        # Ensure encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        # Standard attention computation
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # **Store attention maps**
        if encoder_hidden_states is not hidden_states:
            # This is cross-attention
            self.attn_maps.append(attention_probs.detach().cpu())

        # Apply attention to values
        hidden_states = torch.bmm(attention_probs, value)

        # Reshape back to original dimensions
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def get_net_attn_map(attn_maps, image_size, batch_size=2, instance_or_negative=False, detach=True):
    target_size = (image_size[0]//16, image_size[1]//16)
    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for attn_map in attn_maps:
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx] # (20, 32*32, 77) -> (10, 32*32, 77) # negative & positive CFG
        if len(attn_map.shape) == 4:
            attn_map = attn_map.squeeze()

        attn_map = upscale(attn_map, target_size) # (10,32*32,77) -> (77,64*64)
        net_attn_maps.append(attn_map) # (10,32*32,77) -> (77,64*64)

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)
    net_attn_maps = net_attn_maps.reshape(net_attn_maps.shape[0], 64,64) # (77,64*64) -> (77,64,64)

    return net_attn_maps
    

def inference_and_extract_attn(attn_maps, prompt, pipe, image, obj):
    net_attn_maps = get_net_attn_map(attn_maps, image.size)
    net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
    net_attn_maps = return_net_attn_map(net_attn_maps, pipe.tokenizer, prompt)

    # remove sos and eos
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|startoftext|>>"]
    net_attn_maps = [attn_map for attn_map in net_attn_maps if attn_map[1].split('_')[-1] != "<<|endoftext|>>"]
    ind = 4
    # for i, at_ in enumerate(net_attn_maps):
    #     if obj in at_[-1]:
    #         ind = i
    #         break
    attn = net_attn_maps[ind][0]
    attn = torch.tensor(np.array(attn))
    attn = (attn - attn.min()) / (attn.max() - attn.min())
    return attn

def register_attention_store(unet):
    attn_maps = []
    for name, module in unet.named_modules():
        if not name.split('.')[-1].startswith('attn2'):
            continue
        module.processor = AttnStoreProcessor(attn_maps)
    return unet, attn_maps

