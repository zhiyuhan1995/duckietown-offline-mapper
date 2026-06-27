from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
        if len(self.points) != len(self.colors):
            raise ValueError("points and colors must have the same length")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float64).reshape(-1)
            if len(self.confidence) != len(self.points):
                raise ValueError("confidence length must match points")

    @property
    def size(self) -> int:
        return int(self.points.shape[0])

    def copy(self) -> "PointCloud":
        return PointCloud(
            self.points.copy(),
            self.colors.copy(),
            None if self.confidence is None else self.confidence.copy(),
        )


def transform_point_cloud(cloud: PointCloud, transform: np.ndarray) -> PointCloud:
    pts_h = np.c_[cloud.points, np.ones(cloud.size)]
    pts = (np.asarray(transform, dtype=np.float64) @ pts_h.T).T[:, :3]
    return PointCloud(pts, cloud.colors.copy(), None if cloud.confidence is None else cloud.confidence.copy())


def crop_xy(cloud: PointCloud, x_min: float, x_max: float, y_min: float, y_max: float) -> PointCloud:
    p = cloud.points
    mask = (p[:, 0] >= x_min) & (p[:, 0] <= x_max) & (p[:, 1] >= y_min) & (p[:, 1] <= y_max)
    return PointCloud(p[mask], cloud.colors[mask], None if cloud.confidence is None else cloud.confidence[mask])


def save_ply(cloud: PointCloud, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.points)
        pcd.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(path), pcd)
        return
    except Exception:
        pass

    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {cloud.size}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(cloud.points, cloud.colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def load_ply(path: str | Path) -> PointCloud:
    path = Path(path)
    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        colors = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)
        if colors.size == 0:
            colors = np.full((len(points), 3), 128, dtype=np.uint8)
        return PointCloud(points, colors)
    except Exception:
        pass

    with path.open("r", encoding="ascii", errors="ignore") as f:
        lines = f.readlines()
    header_end = next(i for i, line in enumerate(lines) if line.strip() == "end_header")
    data = []
    for line in lines[header_end + 1 :]:
        parts = line.split()
        if len(parts) >= 6:
            data.append([float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8))
    return PointCloud(arr[:, :3], arr[:, 3:6].astype(np.uint8))

