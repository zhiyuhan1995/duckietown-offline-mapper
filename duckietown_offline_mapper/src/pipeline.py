from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .alignment import estimate_sim2, sim2_to_sim3
from .bev import metadata_from_bounds, rasterize_point_cloud
from .export import export_all
from .ground_texture import render_ground_texture_bev
from .io_utils import ensure_dir, load_yaml, save_yaml
from .occupancy import fuse_occupancy, inflate_obstacles, obstacle_grid_from_non_ground, remove_isolated_occupied_cells
from .plane import fit_ground_plane, ground_alignment_transform
from .pointcloud import PointCloud, save_ply, transform_point_cloud
from .reconstruction import backend_from_config
from .segmentation import SemanticClass, segment_bev_rgb


def load_default_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return load_yaml(root / "configs" / "default.yaml")


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config["export"]["output_dir"])

    recon_config = dict(config.get("reconstruction", {}))
    recon_config["keyframe_interval"] = config.get("input", {}).get("keyframe_interval", recon_config.get("keyframe_interval", 30))
    recon_config["max_keyframes"] = config.get("input", {}).get("max_keyframes", recon_config.get("max_keyframes", 40))
    backend = backend_from_config(recon_config)
    reconstruction = backend.run(config["input"]["path"], output_dir / "work")
    raw_cloud = reconstruction.point_cloud

    plane_config = config.get("ground_plane", {})
    plane_fit = fit_ground_plane(
        raw_cloud,
        distance_threshold=float(plane_config.get("distance_threshold", 0.025)),
        max_iterations=int(plane_config.get("max_iterations", 800)),
    )
    ground_transform = ground_alignment_transform(plane_fit.coefficients)
    ground_aligned_cloud = transform_point_cloud(raw_cloud, ground_transform)
    save_ply(ground_aligned_cloud, output_dir / "ground_aligned_point_cloud.ply")

    alignment_transform = np.eye(4)
    alignment_metadata: dict[str, Any] = {
        "mode": "identity",
        "control_points": [],
        "rms_error": 0.0,
        "transform": alignment_transform.tolist(),
    }
    control_points = config.get("alignment", {}).get("control_points", [])
    if len(control_points) >= 3:
        source = np.array([p["source"][:2] for p in control_points], dtype=np.float64)
        target = np.array([p["target"][:2] for p in control_points], dtype=np.float64)
        alignment_config = config.get("alignment", {})
        estimate_scale = alignment_config.get("mode", "sim2") != "se2"
        allow_reflection = bool(alignment_config.get("allow_reflection", False))
        sim2 = estimate_sim2(source, target, estimate_scale=estimate_scale, allow_reflection=allow_reflection)
        alignment_transform = sim2_to_sim3(sim2)
        alignment_metadata = {
            "mode": "sim2_reflection" if allow_reflection and sim2.reflection else ("sim2" if estimate_scale else "se2"),
            "allow_reflection": allow_reflection,
            "control_points": control_points,
            "scale": float(sim2.scale),
            "rotation": sim2.rotation.tolist(),
            "determinant": float(sim2.determinant),
            "reflection": bool(sim2.reflection),
            "translation": sim2.translation.tolist(),
            "rms_error": float(sim2.rms_error),
            "residuals": sim2.residuals.tolist(),
            "transform": alignment_transform.tolist(),
        }

    aligned_cloud = transform_point_cloud(ground_aligned_cloud, alignment_transform)

    roi = config.get("roi", {})
    if bool(roi.get("auto", False)):
        padding = float(roi.get("padding", 0.15))
        q_low, q_high = float(roi.get("percentile_low", 1.0)), float(roi.get("percentile_high", 99.0))
        x_min, x_max = np.percentile(aligned_cloud.points[:, 0], [q_low, q_high])
        y_min, y_max = np.percentile(aligned_cloud.points[:, 1], [q_low, q_high])
        x_min, x_max = float(x_min - padding), float(x_max + padding)
        y_min, y_max = float(y_min - padding), float(y_max + padding)
    else:
        x_min = float(roi.get("x_min", np.min(aligned_cloud.points[:, 0])))
        x_max = float(roi.get("x_max", np.max(aligned_cloud.points[:, 0])))
        y_min = float(roi.get("y_min", np.min(aligned_cloud.points[:, 1])))
        y_max = float(roi.get("y_max", np.max(aligned_cloud.points[:, 1])))
    p = aligned_cloud.points
    crop_mask = (p[:, 0] >= x_min) & (p[:, 0] <= x_max) & (p[:, 1] >= y_min) & (p[:, 1] <= y_max)
    if not np.any(crop_mask):
        raise RuntimeError("ROI cropped out all reconstructed points. Enable roi.auto or adjust ROI bounds.")
    cropped_cloud = PointCloud(
        p[crop_mask],
        aligned_cloud.colors[crop_mask],
        None if aligned_cloud.confidence is None else aligned_cloud.confidence[crop_mask],
    )
    cropped_ground_mask = plane_fit.inlier_mask[crop_mask]

    bev_config = config.get("bev", {})
    resolution = float(bev_config.get("resolution", 0.02))
    metadata = metadata_from_bounds(x_min, x_max, y_min, y_max, resolution, config.get("export", {}).get("map_frame", "map"))
    unknown_rgb = tuple(int(x) for x in bev_config.get("unknown_rgb", [80, 80, 80]))

    ground_plane_metadata = {
        "coefficients": plane_fit.coefficients.tolist(),
        "ground_count": plane_fit.ground_count,
        "non_ground_count": plane_fit.non_ground_count,
        "ground_alignment_transform": ground_transform.tolist(),
    }
    texture_project_metadata = {
        "reconstruction_to_map_transform": alignment_metadata,
        "ground_plane": ground_plane_metadata,
        "reconstruction": reconstruction.metadata,
    }

    texture_result = None
    ground_texture_config = config.get("ground_texture", {})
    if bool(ground_texture_config.get("enabled", True)):
        texture_summary_path = output_dir / "ground_texture_input_summary.yaml"
        save_yaml(
            {
                "config": config,
                "result": {
                    "metadata": metadata.to_dict(),
                    "project_metadata": texture_project_metadata,
                },
            },
            texture_summary_path,
        )
        texture_result = render_ground_texture_bev(
            run_summary_path=texture_summary_path,
            output_dir=output_dir / "ground_texture",
            resolution=resolution,
            fusion_mode=str(ground_texture_config.get("fusion_mode", "weighted_mean")),
            padding=0.0,
            confidence_scale=ground_texture_config.get("confidence_scale"),
            min_weight=float(ground_texture_config.get("min_weight", 1e-5)),
            view_angle_power=float(ground_texture_config.get("view_angle_power", 1.5)),
            distance_power=float(ground_texture_config.get("distance_power", 1.0)),
            border_margin_px=float(ground_texture_config.get("border_margin_px", 4.0)),
            inpaint_radius=int(ground_texture_config.get("inpaint_radius", 3)),
            unknown_rgb=unknown_rgb,
        )
        bev_rgb = texture_result.texture
        observed_cell_count = int(np.count_nonzero(texture_result.observed_mask))
        bev_generation_metadata = {
            "source": "vggt_camera_ground_texture",
            "fusion_mode": str(ground_texture_config.get("fusion_mode", "weighted_mean")),
            "unknown_rgb": list(unknown_rgb),
            "observed_cell_count": observed_cell_count,
            "observed_fraction": float(texture_result.stats.get("observed_fraction", 0.0)),
            "mean_observations_on_observed": float(texture_result.stats.get("mean_observations_on_observed", 0.0)),
            "texture_paths": {k: str(v) for k, v in texture_result.paths.items()},
        }
    else:
        bev_rgb, point_counts = rasterize_point_cloud(cropped_cloud, metadata, unknown_rgb=unknown_rgb)
        bev_generation_metadata = {
            "source": "point_cloud_raster",
            "unknown_rgb": list(unknown_rgb),
            "observed_cell_count": int(np.count_nonzero(point_counts)),
        }

    segmentation_config = dict(config.get("segmentation", {}))
    segmentation_config["unknown_rgb"] = list(unknown_rgb)
    semantic_grid = segment_bev_rgb(bev_rgb, segmentation_config)
    obstacle_config = config.get("occupancy", {})
    obstacle_grid = obstacle_grid_from_non_ground(
        cropped_cloud,
        cropped_ground_mask,
        metadata,
        height_threshold=float(obstacle_config.get("non_ground_height_threshold", 0.06)),
    )
    inflation_radius = float(obstacle_config.get("robot_radius", 0.085)) + float(obstacle_config.get("safety_margin", 0.025))
    inflated_obstacle = inflate_obstacles(obstacle_grid, inflation_radius, resolution)
    occupancy_grid = fuse_occupancy(
        semantic_grid,
        inflated_obstacle,
        unknown_as_occupied=bool(obstacle_config.get("unknown_as_occupied", False)),
    )
    isolated_removed_cells = 0
    if bool(obstacle_config.get("remove_isolated_occupied", True)):
        occupancy_grid, isolated_removed_cells = remove_isolated_occupied_cells(occupancy_grid)
    gradient_margin_m = float(obstacle_config.get("gradient_margin_m", 0.0))

    project_metadata = {
        "map_frame_name": config.get("export", {}).get("map_frame", "map"),
        "grid_width": metadata.width,
        "grid_height": metadata.height,
        "reconstruction_to_map_transform": alignment_metadata,
        "robot_radius": float(obstacle_config.get("robot_radius", 0.085)),
        "safety_margin": float(obstacle_config.get("safety_margin", 0.025)),
        "obstacle_inflation_radius": inflation_radius,
        "gradient_margin_m": gradient_margin_m,
        "remove_isolated_occupied": bool(obstacle_config.get("remove_isolated_occupied", True)),
        "isolated_occupied_removed_cells": isolated_removed_cells,
        "semantic_classes": {cls.name: int(cls.value) for cls in SemanticClass},
        "bev_generation": bev_generation_metadata,
        "ground_plane": ground_plane_metadata,
        "reconstruction": reconstruction.metadata,
    }
    paths = export_all(
        output_dir,
        cropped_cloud,
        bev_rgb,
        semantic_grid,
        inflated_obstacle,
        occupancy_grid,
        metadata,
        project_metadata,
        gradient_margin_m=gradient_margin_m,
    )
    if texture_result is not None:
        paths.update({f"ground_texture_{k}": v for k, v in texture_result.paths.items()})
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "metadata": metadata.to_dict(),
        "project_metadata": project_metadata,
        "stats": {
            "raw_points": raw_cloud.size,
            "cropped_points": cropped_cloud.size,
            "free_cells": int(np.count_nonzero(occupancy_grid == 0)),
            "occupied_cells": int(np.count_nonzero(occupancy_grid == 100)),
            "unknown_cells": int(np.count_nonzero(occupancy_grid == -1)),
            "gradient_margin_m": gradient_margin_m,
            "isolated_occupied_removed_cells": isolated_removed_cells,
        },
    }
