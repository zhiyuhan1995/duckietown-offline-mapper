from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pointcloud import PointCloud


@dataclass
class PlaneFit:
    coefficients: np.ndarray
    inlier_mask: np.ndarray

    @property
    def ground_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def non_ground_count(self) -> int:
        return int(len(self.inlier_mask) - self.ground_count)


def fit_ground_plane(
    cloud: PointCloud,
    distance_threshold: float = 0.025,
    max_iterations: int = 800,
    seed: int = 7,
) -> PlaneFit:
    if cloud.size < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        coeffs, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=max_iterations,
        )
        mask = np.zeros(cloud.size, dtype=bool)
        mask[np.asarray(inliers, dtype=int)] = True
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        if coeffs_arr[2] < 0:
            coeffs_arr *= -1.0
        return PlaneFit(coeffs_arr, mask)
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    pts = cloud.points
    best_mask = np.zeros(cloud.size, dtype=bool)
    best_coeffs = np.array([0.0, 0.0, 1.0, 0.0])
    iterations = min(max_iterations, max(1, cloud.size * 5))
    for _ in range(iterations):
        sample = pts[rng.choice(cloud.size, size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ sample[0])
        distances = np.abs(pts @ normal + d)
        mask = distances <= distance_threshold
        if np.count_nonzero(mask) > np.count_nonzero(best_mask):
            best_mask = mask
            best_coeffs = np.r_[normal, d]
    if best_coeffs[2] < 0:
        best_coeffs *= -1.0
    return PlaneFit(best_coeffs, best_mask)


def ground_alignment_transform(coefficients: np.ndarray) -> np.ndarray:
    """Return a transform that maps the plane to z=0 with normal +z."""
    coeffs = np.asarray(coefficients, dtype=np.float64)
    normal = coeffs[:3]
    d = float(coeffs[3])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("Invalid plane normal")
    normal = normal / norm
    d = d / norm
    if normal[2] < 0:
        normal = -normal
        d = -d

    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, target)
    sin_theta = np.linalg.norm(cross)
    cos_theta = float(normal @ target)
    if sin_theta < 1e-12:
        rotation = np.eye(3)
    else:
        axis = cross / sin_theta
        k = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = np.eye(3) + k * sin_theta + (k @ k) * (1.0 - cos_theta)

    point_on_plane = -d * normal
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -(rotation @ point_on_plane)
    return transform
