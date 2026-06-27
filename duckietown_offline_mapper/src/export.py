from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bev import BevMetadata
from .io_utils import ensure_dir, save_yaml
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


def export_all(
    output_dir: str | Path,
    aligned_cloud: PointCloud,
    bev_rgb: np.ndarray,
    semantic_grid: np.ndarray,
    obstacle_grid: np.ndarray,
    occupancy_grid: np.ndarray,
    metadata: BevMetadata,
    project_metadata: dict[str, Any],
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
        "map_png": out / "map.png",
        "map_yaml": out / "map.yaml",
        "map_metadata": out / "map_metadata.yaml",
    }

    save_ply(aligned_cloud, paths["aligned_point_cloud"])
    _save_image(paths["bev_rgb"], bev_rgb)
    _save_image(paths["semantic_mask"], colorize_semantic(semantic_grid))
    _save_image(paths["obstacle_occupancy"], (obstacle_grid.astype(np.uint8) * 255))
    _save_image(paths["final_occupancy_grid"], ros_image_from_occupancy(occupancy_grid))
    _save_image(paths["map_png"], ros_image_from_occupancy(occupancy_grid))
    np.save(paths["semantic_grid"], semantic_grid.astype(np.uint8))
    np.save(paths["obstacle_grid"], obstacle_grid.astype(bool))
    np.save(paths["occupancy_grid"], occupancy_grid.astype(np.int8))

    ros_origin = [float(metadata.x_min), float(metadata.y_min), 0.0]
    save_yaml(
        {
            "image": "map.png",
            "mode": "trinary",
            "resolution": float(metadata.resolution),
            "origin": ros_origin,
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        },
        paths["map_yaml"],
    )
    full_metadata = {
        **metadata.to_dict(),
        **project_metadata,
        "ros_map": {
            "image": "map.png",
            "resolution": float(metadata.resolution),
            "origin": ros_origin,
            "occupied_values": {"free": 0, "occupied": 100, "unknown": -1},
            "pixel_values": {"free": 254, "occupied": 0, "unknown": 205},
        },
    }
    save_yaml(full_metadata, paths["map_metadata"])
    return paths

