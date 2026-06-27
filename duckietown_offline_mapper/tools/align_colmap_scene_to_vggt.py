#!/usr/bin/env python3
"""Align a dense COLMAP video scene to a VGGT seed sparse model.

The dense scene supplies camera poses for many 3DGS training frames. The VGGT
seed remains the map-frame geometry anchor. Alignment uses shared original
video frame indices encoded in image filenames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _require_pycolmap():
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pycolmap is required to align COLMAP scenes.") from exc
    return pycolmap


def _find_sparse_dir(path: Path) -> Path:
    candidates = [
        path,
        path / "sparse" / "0",
        path / "sparse",
        path / "work" / "gsplat_scene" / "sparse" / "0",
        path / "work" / "gsplat_scene" / "sparse",
        path / "work" / "colmap_sparse_ba",
        path / "work" / "colmap_sparse",
    ]
    for candidate in candidates:
        if (candidate / "images.bin").exists() and (candidate / "cameras.bin").exists():
            return candidate
    raise FileNotFoundError(f"Could not find a COLMAP sparse model under {path}")


def _frame_index_from_name(name: str) -> int | None:
    numbers = re.findall(r"\d+", Path(name).stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _camera_centers_by_frame(reconstruction: Any) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for _, image in reconstruction.images.items():
        frame_index = _frame_index_from_name(image.name)
        if frame_index is None:
            continue
        centers[frame_index] = np.asarray(image.projection_center(), dtype=np.float64)
    return centers


def _umeyama_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be Nx3 arrays")
    if source.shape[0] < 3:
        raise ValueError("At least 3 shared camera centers are required")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance_source = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular_values * np.diag(correction)) / max(variance_source, 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation, translation


def _residuals(source: np.ndarray, target: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = (scale * (rotation @ source.T)).T + translation
    return np.linalg.norm(aligned - target, axis=1)


def _prepare_output_scene(output_scene: Path, image_source: Path, image_mode: str) -> None:
    if output_scene.exists():
        shutil.rmtree(output_scene)
    (output_scene / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    if image_mode == "none":
        return

    output_images = output_scene / "images"
    if image_mode == "copy":
        shutil.copytree(image_source, output_images)
    elif image_mode == "symlink":
        os.symlink(image_source.resolve(), output_images, target_is_directory=True)
    else:
        raise ValueError(f"Unsupported image mode: {image_mode}")


def align(args: argparse.Namespace) -> dict[str, Any]:
    pycolmap = _require_pycolmap()

    source_scene = Path(args.source_scene)
    source_sparse = _find_sparse_dir(Path(args.source_sparse) if args.source_sparse else source_scene)
    target_sparse = _find_sparse_dir(Path(args.target_sparse))
    output_scene = Path(args.output_scene)
    image_source = Path(args.image_source) if args.image_source else source_scene / "images"
    if args.image_mode != "none" and not image_source.exists():
        raise FileNotFoundError(f"Image source does not exist: {image_source}")

    source_reconstruction = pycolmap.Reconstruction(str(source_sparse))
    target_reconstruction = pycolmap.Reconstruction(str(target_sparse))

    source_centers = _camera_centers_by_frame(source_reconstruction)
    target_centers = _camera_centers_by_frame(target_reconstruction)
    common_frames = sorted(set(source_centers) & set(target_centers))
    if len(common_frames) < args.min_common:
        raise RuntimeError(
            f"Only {len(common_frames)} shared frames found; need at least {args.min_common}."
        )

    source = np.stack([source_centers[index] for index in common_frames], axis=0)
    target = np.stack([target_centers[index] for index in common_frames], axis=0)
    scale, rotation, translation = _umeyama_similarity(source, target)
    errors = _residuals(source, target, scale, rotation, translation)

    sim3d = pycolmap.Sim3d(scale=scale, rotation=rotation, translation=translation)
    source_reconstruction.transform(sim3d)

    _prepare_output_scene(output_scene, image_source, args.image_mode)
    aligned_sparse = output_scene / "sparse" / "0"
    source_reconstruction.write(str(aligned_sparse))

    summary = {
        "source_sparse": str(source_sparse),
        "target_sparse": str(target_sparse),
        "output_scene": str(output_scene),
        "image_source": str(image_source),
        "image_mode": args.image_mode,
        "source_registered_images": len(source_reconstruction.images),
        "source_points3D": len(source_reconstruction.points3D),
        "target_registered_images": len(target_reconstruction.images),
        "target_points3D": len(target_reconstruction.points3D),
        "common_frames": common_frames,
        "num_common_frames": len(common_frames),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "alignment_error_mean": float(np.mean(errors)),
        "alignment_error_median": float(np.median(errors)),
        "alignment_error_max": float(np.max(errors)),
        "alignment_errors": {
            str(frame): float(error) for frame, error in zip(common_frames, errors)
        },
    }
    (output_scene / "alignment_to_vggt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, help="Dense COLMAP/3DGS scene with images/ and sparse/")
    parser.add_argument("--source-sparse", default=None, help="Optional explicit sparse model path for the source scene")
    parser.add_argument("--target-sparse", required=True, help="VGGT seed sparse model path or parent output directory")
    parser.add_argument("--output-scene", required=True)
    parser.add_argument("--image-source", default=None, help="Optional image directory to link/copy into output scene")
    parser.add_argument("--image-mode", choices=["symlink", "copy", "none"], default="symlink")
    parser.add_argument("--min-common", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    summary = align(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
