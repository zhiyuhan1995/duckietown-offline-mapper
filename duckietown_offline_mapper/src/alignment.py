from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignmentResult:
    transform: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rms_error: float
    residuals: np.ndarray
    reflection: bool = False
    determinant: float = 1.0


def umeyama_sim3(
    source: np.ndarray,
    target: np.ndarray,
    estimate_scale: bool = True,
    allow_reflection: bool = False,
) -> AlignmentResult:
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("source and target must have shape (N, D)")
    n, dim = src.shape
    if n < dim:
        raise ValueError(f"Need at least {dim} correspondences for {dim}D alignment")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    covariance = (dst_c.T @ src_c) / n
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(dim)
    if not allow_reflection and np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rotation = u @ np.diag(d) @ vt
    if estimate_scale:
        variance = np.sum(src_c**2) / n
        scale = float(np.sum(singular_values * d) / variance)
    else:
        scale = 1.0
    translation = dst_mean - scale * rotation @ src_mean

    transform = np.eye(dim + 1)
    transform[:dim, :dim] = scale * rotation
    transform[:dim, dim] = translation
    predicted = (scale * (rotation @ src.T)).T + translation
    residuals = np.linalg.norm(predicted - dst, axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    determinant = float(np.linalg.det(rotation))
    return AlignmentResult(transform, scale, rotation, translation, rms, residuals, determinant < 0.0, determinant)


def estimate_sim2(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    estimate_scale: bool = True,
    allow_reflection: bool = False,
) -> AlignmentResult:
    src = np.asarray(source_xy, dtype=np.float64)
    dst = np.asarray(target_xy, dtype=np.float64)
    if src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("Sim(2) alignment expects Nx2 point arrays")
    if len(src) < 3:
        raise ValueError("Need at least 3 control points for planar metric alignment")
    a = src[1] - src[0]
    b = src[2] - src[0]
    area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
    if area < 1e-9:
        raise ValueError("The first three source control points are collinear")
    return umeyama_sim3(src, dst, estimate_scale=estimate_scale, allow_reflection=allow_reflection)


def sim2_to_sim3(sim2: AlignmentResult) -> np.ndarray:
    transform = np.eye(4)
    transform[:2, :2] = sim2.transform[:2, :2]
    transform[:2, 3] = sim2.transform[:2, 2]
    return transform
