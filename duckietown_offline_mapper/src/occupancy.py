from __future__ import annotations

import numpy as np
from scipy import ndimage

from .bev import BevMetadata, world_to_grid
from .pointcloud import PointCloud
from .segmentation import SemanticClass


def obstacle_grid_from_non_ground(
    cloud: PointCloud,
    ground_mask: np.ndarray,
    metadata: BevMetadata,
    height_threshold: float = 0.06,
) -> np.ndarray:
    obstacle = np.zeros((metadata.height, metadata.width), dtype=bool)
    if cloud.size == 0:
        return obstacle
    non_ground = ~np.asarray(ground_mask, dtype=bool)
    high = cloud.points[:, 2] >= height_threshold
    mask = non_ground & high
    if not np.any(mask):
        return obstacle
    u, v = world_to_grid(cloud.points[mask, 0], cloud.points[mask, 1], metadata)
    valid = (u >= 0) & (u < metadata.width) & (v >= 0) & (v < metadata.height)
    obstacle[v[valid], u[valid]] = True
    return obstacle


def inflate_obstacles(obstacle_grid: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    obstacle = obstacle_grid.astype(bool)
    radius_px = int(np.ceil(radius_m / resolution))
    if radius_px <= 0 or not np.any(obstacle):
        return obstacle
    if radius_px > 24:
        return ndimage.distance_transform_edt(~obstacle) <= radius_px
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    struct = (x * x + y * y) <= radius_px * radius_px
    return ndimage.binary_dilation(obstacle, structure=struct)


def fuse_occupancy(
    semantic_grid: np.ndarray,
    obstacle_grid: np.ndarray,
    unknown_as_occupied: bool = False,
) -> np.ndarray:
    semantic = np.asarray(semantic_grid)
    obstacle = np.asarray(obstacle_grid, dtype=bool)
    occupancy = np.full(semantic.shape, -1, dtype=np.int8)

    free = (
        (semantic == SemanticClass.DRIVABLE_ROAD)
        | (semantic == SemanticClass.LANE_MARKING)
        | (semantic == SemanticClass.STOP_LINE)
    )
    occupied = (semantic == SemanticClass.NON_DRIVABLE) | obstacle
    occupancy[free] = 0
    occupancy[occupied] = 100
    if unknown_as_occupied:
        occupancy[semantic == SemanticClass.UNKNOWN] = 100
    return occupancy
