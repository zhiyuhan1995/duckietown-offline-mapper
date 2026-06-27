from __future__ import annotations

import numpy as np

from .bev import BevMetadata


def bev_pixel_to_world_xy(
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    metadata: BevMetadata | dict,
) -> tuple[float, float]:
    """Convert a BEV image pixel to continuous world/map xy on z=0."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    m = metadata if isinstance(metadata, dict) else metadata.to_dict()
    x = m["x_min"] + (float(pixel_x) + 0.5) / image_width * (m["x_max"] - m["x_min"])
    y = m["y_max"] - (float(pixel_y) + 0.5) / image_height * (m["y_max"] - m["y_min"])
    return float(x), float(y)


def invert_planar_transform_point(x: float, y: float, transform: np.ndarray) -> tuple[float, float]:
    """Map an aligned/map-frame xy point back to its pre-alignment source xy."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}")
    point = np.array([float(x), float(y), 0.0, 1.0], dtype=np.float64)
    source = np.linalg.inv(matrix) @ point
    return float(source[0] / source[3]), float(source[1] / source[3])
