from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pointcloud import PointCloud


@dataclass
class BevMetadata:
    resolution: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int
    frame_id: str = "map"

    def to_dict(self) -> dict:
        return {
            "resolution": float(self.resolution),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "width": int(self.width),
            "height": int(self.height),
            "frame_id": self.frame_id,
            "world_to_grid": "u=floor((x-x_min)/resolution), v=floor((y_max-y)/resolution)",
            "grid_to_world": "x=x_min+(u+0.5)*resolution, y=y_max-(v+0.5)*resolution",
        }


def metadata_from_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution: float,
    frame_id: str = "map",
) -> BevMetadata:
    width = int(np.ceil((x_max - x_min) / resolution))
    height = int(np.ceil((y_max - y_min) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid BEV bounds")
    return BevMetadata(resolution, x_min, x_max, y_min, y_max, width, height, frame_id)


def world_to_grid(x: float | np.ndarray, y: float | np.ndarray, metadata: BevMetadata | dict) -> tuple[np.ndarray, np.ndarray]:
    m = metadata if isinstance(metadata, dict) else metadata.to_dict()
    u = np.floor((np.asarray(x) - m["x_min"]) / m["resolution"]).astype(int)
    v = np.floor((m["y_max"] - np.asarray(y)) / m["resolution"]).astype(int)
    return u, v


def grid_to_world(u: int | np.ndarray, v: int | np.ndarray, metadata: BevMetadata | dict) -> tuple[np.ndarray, np.ndarray]:
    m = metadata if isinstance(metadata, dict) else metadata.to_dict()
    x = m["x_min"] + (np.asarray(u) + 0.5) * m["resolution"]
    y = m["y_max"] - (np.asarray(v) + 0.5) * m["resolution"]
    return x, y


def rasterize_point_cloud(
    cloud: PointCloud,
    metadata: BevMetadata,
    unknown_rgb: tuple[int, int, int] = (80, 80, 80),
) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((metadata.height, metadata.width, 3), unknown_rgb, dtype=np.uint8)
    count = np.zeros((metadata.height, metadata.width), dtype=np.int32)
    if cloud.size == 0:
        return image, count

    u, v = world_to_grid(cloud.points[:, 0], cloud.points[:, 1], metadata)
    valid = (u >= 0) & (u < metadata.width) & (v >= 0) & (v < metadata.height)
    if not np.any(valid):
        return image, count
    u = u[valid]
    v = v[valid]
    colors = cloud.colors[valid].astype(np.float64)
    acc = np.zeros((metadata.height, metadata.width, 3), dtype=np.float64)
    np.add.at(acc, (v, u), colors)
    np.add.at(count, (v, u), 1)
    filled = count > 0
    image[filled] = np.clip(acc[filled] / count[filled, None], 0, 255).astype(np.uint8)
    return image, count

