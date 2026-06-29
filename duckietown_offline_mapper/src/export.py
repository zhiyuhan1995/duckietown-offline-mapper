from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bev import BevMetadata
from .io_utils import ensure_dir, save_yaml
from .occupancy import gradient_margin_from_occupancy
from .pointcloud import PointCloud, save_ply
from .segmentation import colorize_semantic


def _save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        arr = image
        if arr.ndim == 3:
            arr = arr[..., ::-1]
        cv2.imwrite(str(path), arr)
        return
    except Exception:
        pass

    from PIL import Image

    Image.fromarray(image).save(path)


def ros_image_from_occupancy(occupancy_grid: np.ndarray) -> np.ndarray:
    image = np.full(occupancy_grid.shape, 205, dtype=np.uint8)
    image[occupancy_grid == 0] = 254
    image[occupancy_grid == 100] = 0
    return image


def ros_image_from_occupancy_margin(occupancy_grid: np.ndarray, margin_layer: np.ndarray) -> np.ndarray:
    margin = np.clip(np.asarray(margin_layer, dtype=np.float32), 0.0, 1.0)
    image = np.rint(254.0 * (1.0 - margin)).astype(np.uint8)
    image[np.asarray(occupancy_grid) == -1] = 205
    return image


def occupancy_cost_grid_from_margin(occupancy_grid: np.ndarray, margin_layer: np.ndarray) -> np.ndarray:
    cost = np.rint(np.clip(np.asarray(margin_layer, dtype=np.float32), 0.0, 1.0) * 100.0).astype(np.int8)
    cost[np.asarray(occupancy_grid) == -1] = -1
    return cost


def _save_map_yaml(path: str | Path, image_name: str, metadata: BevMetadata, mode: str) -> None:
    save_yaml(
        {
            "image": image_name,
            "mode": mode,
            "resolution": float(metadata.resolution),
            "origin": [float(metadata.x_min), float(metadata.y_min), 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        },
        path,
    )


def export_all(
    output_dir: str | Path,
    aligned_cloud: PointCloud,
    bev_rgb: np.ndarray,
    semantic_grid: np.ndarray,
    obstacle_grid: np.ndarray,
    occupancy_grid: np.ndarray,
    metadata: BevMetadata,
    project_metadata: dict[str, Any],
    gradient_margin_m: float = 0.0,
) -> dict[str, Path]:
    out = ensure_dir(output_dir)
    paths = {
        "aligned_point_cloud": out / "aligned_point_cloud.ply",
        "bev_rgb": out / "bev_rgb.png",
        "semantic_mask": out / "semantic_mask.png",
        "obstacle_occupancy": out / "obstacle_occupancy.png",
        "final_occupancy_grid": out / "final_occupancy_grid.png",
        "semantic_grid": out / "semantic_grid.npy",
        "obstacle_grid": out / "obstacle_grid.npy",
        "occupancy_grid": out / "occupancy_grid.npy",
        "occupancy_margin": out / "occupancy_margin.npy",
        "occupancy_cost_grid": out / "occupancy_cost_grid.npy",
        "map_png": out / "map.png",
        "map_yaml": out / "map.yaml",
        "map_hard_png": out / "map_hard.png",
        "map_hard_yaml": out / "map_hard.yaml",
        "map_with_margin_png": out / "map_with_margin.png",
        "map_with_margin_yaml": out / "map_with_margin.yaml",
        "map_metadata": out / "map_metadata.yaml",
    }

    margin_layer = gradient_margin_from_occupancy(occupancy_grid, float(gradient_margin_m), float(metadata.resolution))
    margin_image = ros_image_from_occupancy_margin(occupancy_grid, margin_layer)
    hard_image = ros_image_from_occupancy(occupancy_grid)
    cost_grid = occupancy_cost_grid_from_margin(occupancy_grid, margin_layer)
    map_mode = "scale" if float(gradient_margin_m) > 0.0 else "trinary"

    save_ply(aligned_cloud, paths["aligned_point_cloud"])
    _save_image(paths["bev_rgb"], bev_rgb)
    _save_image(paths["semantic_mask"], colorize_semantic(semantic_grid))
    _save_image(paths["obstacle_occupancy"], (obstacle_grid.astype(np.uint8) * 255))
    _save_image(paths["final_occupancy_grid"], hard_image)
    _save_image(paths["map_png"], margin_image)
    _save_image(paths["map_hard_png"], hard_image)
    _save_image(paths["map_with_margin_png"], margin_image)
    np.save(paths["semantic_grid"], semantic_grid.astype(np.uint8))
    np.save(paths["obstacle_grid"], obstacle_grid.astype(bool))
    np.save(paths["occupancy_grid"], occupancy_grid.astype(np.int8))
    np.save(paths["occupancy_margin"], margin_layer.astype(np.float32))
    np.save(paths["occupancy_cost_grid"], cost_grid.astype(np.int8))

    ros_origin = [float(metadata.x_min), float(metadata.y_min), 0.0]
    _save_map_yaml(paths["map_yaml"], "map.png", metadata, map_mode)
    _save_map_yaml(paths["map_hard_yaml"], "map_hard.png", metadata, "trinary")
    _save_map_yaml(paths["map_with_margin_yaml"], "map_with_margin.png", metadata, "scale")
    full_metadata = {
        **metadata.to_dict(),
        **project_metadata,
        "ros_map": {
            "image": "map.png",
            "mode": map_mode,
            "resolution": float(metadata.resolution),
            "origin": ros_origin,
            "occupied_values": {"free": 0, "occupied": 100, "unknown": -1},
            "pixel_values": {"free": 254, "occupied": 0, "unknown": 205},
            "hard_map": {"image": "map_hard.png", "yaml": "map_hard.yaml", "mode": "trinary"},
            "margin_map": {"image": "map_with_margin.png", "yaml": "map_with_margin.yaml", "mode": "scale"},
            "gradient_margin": {
                "radius_m": float(gradient_margin_m),
                "radius_cells": float(gradient_margin_m) / float(metadata.resolution),
                "layer_npy": "occupancy_margin.npy",
                "cost_grid_npy": "occupancy_cost_grid.npy",
                "value_convention": "1.0 at occupied cells, linearly falling to 0.0 at radius_m",
            },
        },
    }
    save_yaml(full_metadata, paths["map_metadata"])
    return paths
