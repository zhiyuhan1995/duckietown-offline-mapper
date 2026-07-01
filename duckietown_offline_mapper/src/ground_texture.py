from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .bev import BevMetadata, metadata_from_bounds, grid_to_world
from .io_utils import ensure_dir, save_yaml


@dataclass
class GroundTextureResult:
    texture: np.ndarray
    raw_texture: np.ndarray
    observed_mask: np.ndarray
    weight_map: np.ndarray
    observation_count: np.ndarray
    metadata: BevMetadata
    stats: dict[str, Any]
    paths: dict[str, Path]


def render_ground_texture_bev(
    run_summary_path: str | Path,
    output_dir: str | Path,
    resolution: float = 0.005,
    fusion_mode: str = "best_view",
    padding: float = 0.0,
    confidence_scale: float | None = None,
    min_weight: float = 1e-5,
    view_angle_power: float = 1.5,
    distance_power: float = 1.0,
    border_margin_px: float = 4.0,
    inpaint_radius: int = 3,
    unknown_rgb: tuple[int, int, int] = (80, 80, 80),
) -> GroundTextureResult:
    """Render a ground-plane texture by inverse-projecting BEV cells into VGGT views."""

    run_summary_path = Path(run_summary_path)
    output_dir = ensure_dir(output_dir)
    summary = _load_summary(run_summary_path)
    result = summary.get("result", {})
    project = result.get("project_metadata", {})
    work_dir = _work_dir_from_summary(run_summary_path, summary)

    extrinsics = np.load(work_dir / "camera_extrinsics.npy").astype(np.float64)
    intrinsics = np.load(work_dir / "camera_intrinsics.npy").astype(np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Expected camera_extrinsics.npy shape (N,3,4), got {extrinsics.shape}")
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Expected camera_intrinsics.npy shape (N,3,3), got {intrinsics.shape}")
    if len(extrinsics) != len(intrinsics):
        raise ValueError("Camera extrinsic/intrinsic counts do not match")

    preprocess_mode = (
        summary.get("config", {})
        .get("reconstruction", {})
        .get("vggt", {})
        .get("preprocess_mode", "crop")
    )
    image_paths = _image_paths_from_summary(run_summary_path, summary, work_dir)
    if len(image_paths) < len(extrinsics):
        raise ValueError(f"Only found {len(image_paths)} images for {len(extrinsics)} cameras")
    image_paths = image_paths[: len(extrinsics)]
    images = _load_preprocessed_vggt_images(image_paths, str(preprocess_mode))

    confidence = _load_confidence(work_dir, images.shape[1:3])
    if confidence is not None and len(confidence) != len(extrinsics):
        confidence = confidence[: len(extrinsics)]
    if confidence_scale is None:
        confidence_scale = float(
            summary.get("config", {})
            .get("reconstruction", {})
            .get("vggt", {})
            .get("confidence_threshold", 5.0)
        )
    confidence_scale = max(float(confidence_scale), 1e-6)

    metadata = _metadata_from_summary(result, float(resolution), float(padding), unknown_rgb)
    map_to_raw = np.linalg.inv(_raw_to_map_transform(project))
    plane_coeffs = np.asarray(project.get("ground_plane", {}).get("coefficients"), dtype=np.float64)
    if plane_coeffs.shape != (4,):
        raise ValueError("run_summary.yaml does not contain a valid ground plane")
    plane_normal_raw = _normalize(plane_coeffs[:3])

    raw_points, grid_shape = _raw_ground_points(metadata, map_to_raw)
    pixel_count = raw_points.shape[0]
    raw_texture = np.full((pixel_count, 3), unknown_rgb, dtype=np.float64)
    weight_map = np.zeros(pixel_count, dtype=np.float64)
    weight_sum = np.zeros(pixel_count, dtype=np.float64)
    color_sum = np.zeros((pixel_count, 3), dtype=np.float64)
    observation_count = np.zeros(pixel_count, dtype=np.uint16)

    if fusion_mode not in {"best_view", "weighted_mean"}:
        raise ValueError("fusion_mode must be 'best_view' or 'weighted_mean'")

    for frame_idx, (extrinsic, intrinsic, image) in enumerate(zip(extrinsics, intrinsics, images)):
        projected = _project_raw_points(raw_points, extrinsic, intrinsic)
        u, v, depth, in_front = projected
        h, w = image.shape[:2]
        valid = in_front & (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)
        if not np.any(valid):
            continue

        weight = np.zeros(pixel_count, dtype=np.float64)
        rgb = _bilinear_sample(image, u, v, valid)

        angle = _view_angle_weight(raw_points, extrinsic, plane_normal_raw)
        distance = np.power(np.maximum(depth, 1e-6), -float(distance_power))
        border = _border_weight(u, v, w, h, float(border_margin_px))
        weight[valid] = angle[valid] * distance[valid] * border[valid]

        if confidence is not None:
            conf_sample = _bilinear_sample_scalar(confidence[frame_idx], u, v, valid)
            weight[valid] *= np.clip(conf_sample[valid] / confidence_scale, 0.0, 1.0)

        if view_angle_power != 1.0:
            weight[valid] = np.power(np.clip(weight[valid], 0.0, None), 1.0)
            angle_only = _view_angle_weight(raw_points, extrinsic, plane_normal_raw)
            weight[valid] *= np.power(np.clip(angle_only[valid], 0.0, 1.0), max(float(view_angle_power) - 1.0, 0.0))

        used = valid & np.isfinite(weight) & (weight > float(min_weight))
        if not np.any(used):
            continue
        observation_count[used] += 1

        if fusion_mode == "best_view":
            replace = used & (weight > weight_map)
            raw_texture[replace] = rgb[replace]
            weight_map[replace] = weight[replace]
        else:
            color_sum[used] += rgb[used] * weight[used, None]
            weight_sum[used] += weight[used]

    if fusion_mode == "weighted_mean":
        filled = weight_sum > float(min_weight)
        raw_texture[filled] = color_sum[filled] / weight_sum[filled, None]
        weight_map = weight_sum
    else:
        filled = weight_map > float(min_weight)

    raw_image = np.clip(raw_texture.reshape((*grid_shape, 3)), 0, 255).astype(np.uint8)
    observed_mask = filled.reshape(grid_shape)
    observation_image = observation_count.reshape(grid_shape)
    weight_image = weight_map.reshape(grid_shape)
    texture = _inpaint_small_holes(raw_image, observed_mask, int(inpaint_radius))

    paths = _write_outputs(output_dir, texture, raw_image, observed_mask, weight_image, observation_image)
    run_summary_sha256 = hashlib.sha256(run_summary_path.read_bytes()).hexdigest() if run_summary_path.exists() else None
    stats = {
        "run_summary": str(run_summary_path),
        "run_summary_mtime": float(run_summary_path.stat().st_mtime) if run_summary_path.exists() else None,
        "run_summary_sha256": run_summary_sha256,
        "work_dir": str(work_dir),
        "image_count": int(len(image_paths)),
        "resolution": float(resolution),
        "width": int(metadata.width),
        "height": int(metadata.height),
        "fusion_mode": fusion_mode,
        "settings": {
            "resolution": float(resolution),
            "fusion_mode": str(fusion_mode),
            "padding": float(padding),
            "confidence_scale": float(confidence_scale),
            "min_weight": float(min_weight),
            "view_angle_power": float(view_angle_power),
            "distance_power": float(distance_power),
            "border_margin_px": float(border_margin_px),
            "inpaint_radius": int(inpaint_radius),
            "unknown_rgb": [int(c) for c in unknown_rgb],
        },
        "observed_pixels": int(np.count_nonzero(observed_mask)),
        "total_pixels": int(observed_mask.size),
        "observed_fraction": float(np.count_nonzero(observed_mask) / max(1, observed_mask.size)),
        "max_observations_per_pixel": int(observation_image.max()) if observation_image.size else 0,
        "mean_observations_on_observed": float(observation_image[observed_mask].mean()) if np.any(observed_mask) else 0.0,
        "max_weight": float(np.nanmax(weight_image)) if weight_image.size else 0.0,
        "metadata": metadata.to_dict(),
        "paths": {k: str(v) for k, v in paths.items()},
    }
    save_yaml(stats, output_dir / "ground_texture_metadata.yaml")
    paths["metadata"] = output_dir / "ground_texture_metadata.yaml"
    return GroundTextureResult(
        texture=texture,
        raw_texture=raw_image,
        observed_mask=observed_mask,
        weight_map=weight_image,
        observation_count=observation_image,
        metadata=metadata,
        stats=stats,
        paths=paths,
    )


def _load_summary(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _work_dir_from_summary(summary_path: Path, summary: dict[str, Any]) -> Path:
    result_paths = summary.get("result", {}).get("paths", {})
    cloud_path = result_paths.get("aligned_point_cloud")
    if cloud_path:
        output_dir = _resolve_path(summary_path, cloud_path).parent
        work_dir = output_dir / "work"
        if work_dir.exists():
            return work_dir
    return summary_path.parent / "work"


def _image_paths_from_summary(summary_path: Path, summary: dict[str, Any], work_dir: Path) -> list[Path]:
    project = summary.get("result", {}).get("project_metadata", {})
    reconstruction = project.get("reconstruction", {})
    image_paths = [_resolve_path(summary_path, p) for p in reconstruction.get("image_paths", [])]
    image_paths = [p for p in image_paths if p.exists()]
    if image_paths:
        return image_paths
    return sorted((work_dir / "vggt_images").glob("*.png"))


def _resolve_path(summary_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    candidate = summary_path.parents[1] / path if len(summary_path.parents) > 1 else path
    if candidate.exists():
        return candidate
    return path


def _load_preprocessed_vggt_images(image_paths: list[Path], preprocess_mode: str) -> np.ndarray:
    try:
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ground texture fusion requires the official VGGT preprocessing loader.") from exc

    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode)
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Unexpected VGGT preprocessed image tensor shape {images.shape}")
    images = np.transpose(images, (0, 2, 3, 1))
    return np.clip(images * 255.0, 0, 255).astype(np.uint8)


def _load_confidence(work_dir: Path, image_hw: tuple[int, int]) -> np.ndarray | None:
    path = work_dir / "point_confidence.npy"
    if not path.exists():
        return None
    confidence = np.load(path).astype(np.float32)
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 3:
        return None
    target_h, target_w = image_hw
    if confidence.shape[1:3] == (target_h, target_w):
        return confidence
    try:
        import cv2  # type: ignore

        resized = [
            cv2.resize(conf, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            for conf in confidence
        ]
        return np.asarray(resized, dtype=np.float32)
    except Exception:
        return None


def _metadata_from_summary(
    result: dict[str, Any],
    resolution: float,
    padding: float,
    unknown_rgb: tuple[int, int, int],
) -> BevMetadata:
    existing = result.get("metadata", {})
    if not existing:
        raise ValueError("run_summary.yaml does not contain BEV metadata")
    return metadata_from_bounds(
        float(existing["x_min"]) - padding,
        float(existing["x_max"]) + padding,
        float(existing["y_min"]) - padding,
        float(existing["y_max"]) + padding,
        resolution,
        str(existing.get("frame_id", "map")),
    )


def _raw_to_map_transform(project: dict[str, Any]) -> np.ndarray:
    ground = np.asarray(project.get("ground_plane", {}).get("ground_alignment_transform"), dtype=np.float64)
    if ground.shape != (4, 4):
        raise ValueError("run_summary.yaml does not contain ground_alignment_transform")
    alignment = np.asarray(project.get("reconstruction_to_map_transform", {}).get("transform", np.eye(4)), dtype=np.float64)
    if alignment.shape != (4, 4):
        raise ValueError("run_summary.yaml contains invalid reconstruction_to_map_transform")
    return alignment @ ground


def _raw_ground_points(metadata: BevMetadata, map_to_raw: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    vv, uu = np.indices((metadata.height, metadata.width))
    x, y = grid_to_world(uu, vv, metadata)
    map_points = np.stack([x, y, np.zeros_like(x), np.ones_like(x)], axis=-1).reshape(-1, 4)
    raw = (map_to_raw @ map_points.T).T[:, :3]
    return raw.astype(np.float64), (metadata.height, metadata.width)


def _project_raw_points(
    raw_points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    camera = raw_points @ rotation.T + translation
    depth = camera[:, 2]
    in_front = depth > 1e-6
    x = camera[:, 0] / np.maximum(depth, 1e-6)
    y = camera[:, 1] / np.maximum(depth, 1e-6)
    u = intrinsic[0, 0] * x + intrinsic[0, 2]
    v = intrinsic[1, 1] * y + intrinsic[1, 2]
    return u, v, depth, in_front


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    out = np.zeros((len(u), image.shape[2]), dtype=np.float64)
    if not np.any(valid):
        return out
    uu = np.clip(u[valid], 0.0, w - 1.0)
    vv = np.clip(v[valid], 0.0, h - 1.0)
    x0 = np.floor(uu).astype(np.int64)
    y0 = np.floor(vv).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = uu - x0
    wy = vv - y0
    top = image[y0, x0].astype(np.float64) * (1.0 - wx[:, None]) + image[y0, x1].astype(np.float64) * wx[:, None]
    bottom = image[y1, x0].astype(np.float64) * (1.0 - wx[:, None]) + image[y1, x1].astype(np.float64) * wx[:, None]
    out[valid] = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
    return out


def _bilinear_sample_scalar(image: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    sampled = _bilinear_sample(image[..., None], u, v, valid)
    return sampled[:, 0]


def _view_angle_weight(raw_points: np.ndarray, extrinsic: np.ndarray, plane_normal_raw: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    camera_center = -(rotation.T @ translation)
    ray = camera_center[None, :] - raw_points
    ray_norm = np.linalg.norm(ray, axis=1)
    good = ray_norm > 1e-9
    weight = np.zeros(len(raw_points), dtype=np.float64)
    weight[good] = np.abs(ray[good] @ plane_normal_raw) / ray_norm[good]
    return np.clip(weight, 0.0, 1.0)


def _border_weight(u: np.ndarray, v: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    if margin <= 0:
        return np.ones_like(u, dtype=np.float64)
    dist = np.minimum.reduce([u, v, width - 1.0 - u, height - 1.0 - v])
    return np.clip(dist / margin, 0.0, 1.0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Invalid zero-length vector")
    return vector / norm


def _inpaint_small_holes(image: np.ndarray, observed_mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or np.all(observed_mask):
        return image.copy()
    try:
        import cv2  # type: ignore

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        closed = cv2.morphologyEx((observed_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel) > 0
        fill_mask = (closed & ~observed_mask).astype(np.uint8) * 255
        if np.count_nonzero(fill_mask) == 0:
            return image.copy()
        return cv2.inpaint(image, fill_mask, float(radius), cv2.INPAINT_TELEA)
    except Exception:
        return image.copy()


def _write_outputs(
    output_dir: Path,
    texture: np.ndarray,
    raw_texture: np.ndarray,
    observed_mask: np.ndarray,
    weight_map: np.ndarray,
    observation_count: np.ndarray,
) -> dict[str, Path]:
    paths = {
        "texture": output_dir / "ground_texture_bev.png",
        "raw_texture": output_dir / "ground_texture_bev_raw.png",
        "observed_mask": output_dir / "ground_texture_observed_mask.png",
        "weight_map": output_dir / "ground_texture_weight.png",
        "observation_count": output_dir / "ground_texture_observation_count.png",
        "weight_array": output_dir / "ground_texture_weight.npy",
        "observation_count_array": output_dir / "ground_texture_observation_count.npy",
    }
    _save_image(paths["texture"], texture)
    _save_image(paths["raw_texture"], raw_texture)
    _save_image(paths["observed_mask"], (observed_mask.astype(np.uint8) * 255))
    _save_image(paths["weight_map"], _normalize_to_u8(weight_map))
    _save_image(paths["observation_count"], _normalize_to_u8(observation_count.astype(np.float64)))
    np.save(paths["weight_array"], weight_map.astype(np.float32))
    np.save(paths["observation_count_array"], observation_count.astype(np.uint16))
    return paths


def _normalize_to_u8(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    hi = float(np.percentile(values[finite], 99.0))
    if hi <= 1e-12:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip(values / hi * 255.0, 0, 255).astype(np.uint8)


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
