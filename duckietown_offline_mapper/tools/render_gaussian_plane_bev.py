#!/usr/bin/env python3
"""Render a ground-plane BEV texture by projecting trained 3DGS centers.

This is a standalone QA renderer. It avoids the extreme novel-view artifacts
that appear when an orthographic top-down camera looks through oblique 3D
Gaussians trained from ground-level video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


SH_C0 = 0.28209479177387814


def _points_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.c_[points, np.ones(len(points), dtype=np.float64)]
    out = (matrix @ points_h.T).T
    return out[:, :3] / out[:, 3:4]


def _load_colmap_normalization(scene_dir: str) -> np.ndarray:
    try:
        from datasets.colmap import Parser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "This renderer requires gsplat examples on PYTHONPATH so datasets.colmap.Parser can be imported."
        ) from exc
    parser = Parser(scene_dir, factor=1, normalize=True, test_every=8)
    return np.asarray(parser.transform, dtype=np.float64)


def _load_run_summary(path: Path) -> tuple[str, np.ndarray]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("--run-summary requires PyYAML.") from exc
    summary = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project = summary["result"]["project_metadata"]
    scene_dir = project["reconstruction"]["colmap"]["gsplat_scene_dir"]
    ground_transform = np.asarray(project["ground_plane"]["ground_alignment_transform"], dtype=np.float64)
    alignment_transform = np.asarray(
        project["reconstruction_to_map_transform"]["transform"], dtype=np.float64
    )
    return scene_dir, alignment_transform @ ground_transform


def _load_splats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = torch.load(checkpoint, map_location="cpu")
    raw = data["splats"]
    means = raw["means"].detach().cpu().numpy().astype(np.float64)
    opacities = torch.sigmoid(raw["opacities"]).detach().cpu().numpy().astype(np.float32)
    if "colors" in raw:
        colors = torch.sigmoid(raw["colors"]).detach().cpu().numpy().astype(np.float32)
    else:
        colors = (raw["sh0"][:, 0, :].detach().cpu().numpy().astype(np.float32) * SH_C0) + 0.5
        colors = np.clip(colors, 0.0, 1.0)
    return means, colors, opacities


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    try:
        import cv2  # type: ignore

        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    except Exception:
        from scipy.ndimage import gaussian_filter  # type: ignore

        if image.ndim == 2:
            return gaussian_filter(image, sigma=sigma, mode="nearest")
        return gaussian_filter(image, sigma=(sigma, sigma, 0), mode="nearest")


def _auto_bounds(points_map: np.ndarray, args: argparse.Namespace) -> tuple[float, float, float, float]:
    lo = np.percentile(points_map[:, :2], args.bounds_percentile, axis=0)
    hi = np.percentile(points_map[:, :2], 100.0 - args.bounds_percentile, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = np.maximum(span * args.padding_fraction, args.min_padding)
    lo -= pad
    hi += pad
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def render(args: argparse.Namespace) -> dict[str, object]:
    scene_dir, raw_to_map = _load_run_summary(Path(args.run_summary))
    raw_to_norm = _load_colmap_normalization(scene_dir)
    norm_to_map = raw_to_map @ np.linalg.inv(raw_to_norm)

    means, colors, opacities = _load_splats(Path(args.checkpoint))
    points_map = _points_transform(norm_to_map, means)
    z = points_map[:, 2]
    if args.z_center == "median":
        z_center = float(np.median(z))
    else:
        z_center = float(args.z_center)

    z_mask = (z >= float(args.z_min)) & (z <= float(args.z_max))
    opacity_mask = opacities >= float(args.min_opacity)
    mask = z_mask & opacity_mask & np.isfinite(points_map).all(axis=1)
    if not np.any(mask):
        raise RuntimeError("No Gaussians survived the z/opacity filters.")

    points_sel = points_map[mask]
    colors_sel = colors[mask]
    opacity_sel = opacities[mask]
    z_sel = z[mask]

    if args.map_bounds:
        x_min, x_max, y_min, y_max = [float(v) for v in args.map_bounds]
    else:
        x_min, x_max, y_min, y_max = _auto_bounds(points_sel, args)
    resolution = (x_max - x_min) / float(args.width)
    height = int(args.height) if args.height else int(np.ceil((y_max - y_min) / resolution))
    width = int(args.width)
    height = max(height, 1)

    u = np.floor((points_sel[:, 0] - x_min) / resolution).astype(np.int64)
    v = np.floor((y_max - points_sel[:, 1]) / resolution).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[inside], v[inside]
    colors_sel = colors_sel[inside]
    opacity_sel = opacity_sel[inside]
    z_sel = z_sel[inside]

    z_weight = np.exp(-0.5 * ((z_sel - z_center) / max(float(args.z_sigma), 1e-6)) ** 2).astype(np.float32)
    weights = (opacity_sel * z_weight).astype(np.float32)

    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    np.add.at(weight, (v, u), weights)
    for channel in range(3):
        np.add.at(accum[..., channel], (v, u), colors_sel[:, channel] * weights)

    accum = _blur(accum, float(args.splat_sigma_px))
    weight = _blur(weight, float(args.splat_sigma_px))

    rgb = np.full((height, width, 3), np.asarray(args.background, dtype=np.float32), dtype=np.float32)
    observed = weight > float(args.min_weight)
    rgb[observed] = accum[observed] / np.maximum(weight[observed, None], 1e-8)
    alpha = np.clip(weight / max(float(args.alpha_scale), 1e-6), 0.0, 1.0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / "gaussian_plane_bev_rgb.png"
    alpha_path = output_dir / "gaussian_plane_bev_alpha.png"
    meta_path = output_dir / "gaussian_plane_bev_metadata.json"
    imageio.imwrite(rgb_path, (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    imageio.imwrite(alpha_path, (alpha * 255.0 + 0.5).astype(np.uint8))

    meta = {
        "checkpoint": str(args.checkpoint),
        "run_summary": str(args.run_summary),
        "scene_dir": scene_dir,
        "source_gaussians": int(len(means)),
        "selected_gaussians": int(np.count_nonzero(mask)),
        "rasterized_gaussians": int(len(u)),
        "z_center": z_center,
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "z_sigma": float(args.z_sigma),
        "min_opacity": float(args.min_opacity),
        "width": width,
        "height": height,
        "resolution": float(resolution),
        "map_bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "splat_sigma_px": float(args.splat_sigma_px),
        "observed_pixels": int(np.count_nonzero(observed)),
        "rgb_path": str(rgb_path),
        "alpha_path": str(alpha_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--map-bounds", type=float, nargs=4, default=None)
    parser.add_argument("--bounds-percentile", type=float, default=0.5)
    parser.add_argument("--padding-fraction", type=float, default=0.05)
    parser.add_argument("--min-padding", type=float, default=0.05)
    parser.add_argument("--z-min", type=float, default=-0.06)
    parser.add_argument("--z-max", type=float, default=0.04)
    parser.add_argument("--z-center", default="median")
    parser.add_argument("--z-sigma", type=float, default=0.035)
    parser.add_argument("--min-opacity", type=float, default=0.01)
    parser.add_argument("--splat-sigma-px", type=float, default=3.0)
    parser.add_argument("--min-weight", type=float, default=1e-5)
    parser.add_argument("--alpha-scale", type=float, default=0.08)
    parser.add_argument("--background", type=float, nargs=3, default=(0.32, 0.32, 0.32))
    return parser.parse_args()


def main() -> None:
    meta = render(parse_args())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
