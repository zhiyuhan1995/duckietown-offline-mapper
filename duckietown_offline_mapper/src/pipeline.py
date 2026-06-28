from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .alignment import estimate_sim2, sim2_to_sim3
from .bev import metadata_from_bounds, rasterize_point_cloud
from .export import export_all
from .io_utils import ensure_dir, load_yaml
from .occupancy import fuse_occupancy, inflate_obstacles, obstacle_grid_from_non_ground
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
        estimate_scale = config.get("alignment", {}).get("mode", "sim2") != "se2"
        sim2 = estimate_sim2(source, target, estimate_scale=estimate_scale)
        alignment_transform = sim2_to_sim3(sim2)
        alignment_metadata = {
            "mode": "sim2" if estimate_scale else "se2",
            "control_points": control_points,
            "scale": float(sim2.scale),
            "rotation": sim2.rotation.tolist(),
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
    bev_rgb, point_counts = rasterize_point_cloud(cropped_cloud, metadata, unknown_rgb=unknown_rgb)

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

    project_metadata = {
        "map_frame_name": config.get("export", {}).get("map_frame", "map"),
        "grid_width": metadata.width,
        "grid_height": metadata.height,
        "reconstruction_to_map_transform": alignment_metadata,
        "robot_radius": float(obstacle_config.get("robot_radius", 0.085)),
        "safety_margin": float(obstacle_config.get("safety_margin", 0.025)),
        "obstacle_inflation_radius": inflation_radius,
        "semantic_classes": {cls.name: int(cls.value) for cls in SemanticClass},
        "bev_generation": {
            "unknown_rgb": list(unknown_rgb),
            "observed_cell_count": int(np.count_nonzero(point_counts)),
        },
        "ground_plane": {
            "coefficients": plane_fit.coefficients.tolist(),
            "ground_count": plane_fit.ground_count,
            "non_ground_count": plane_fit.non_ground_count,
            "ground_alignment_transform": ground_transform.tolist(),
        },
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
    )
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
        },
    }
