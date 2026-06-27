#!/usr/bin/env python3
"""Render a trained gsplat checkpoint into an orthographic top-down BEV image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
import torch
from gsplat.rendering import rasterization


AxisName = Literal["x", "y", "z"]


def _axis_vector(axis: AxisName) -> np.ndarray:
    if axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _choose_vertical_axis(points: np.ndarray) -> AxisName:
    extents = np.percentile(points, 99.0, axis=0) - np.percentile(points, 1.0, axis=0)
    return ("x", "y", "z")[int(np.argmin(extents))]


def _camera_basis(vertical_axis: AxisName) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertical = _axis_vector(vertical_axis)
    forward = -vertical
    if vertical_axis == "x":
        x_axis = _axis_vector("y")
    else:
        x_axis = _axis_vector("x")
    y_axis = np.cross(forward, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return x_axis, y_axis.astype(np.float32), vertical


def _load_splats(
    checkpoint: Path, device: torch.device, sh_degree: int | None
) -> tuple[dict[str, torch.Tensor], int | None, int]:
    data = torch.load(checkpoint, map_location="cpu")
    step = int(data.get("step", -1))
    raw_splats = data["splats"]

    splats = {
        "means": raw_splats["means"].to(device),
        "quats": raw_splats["quats"].to(device),
        "scales": torch.exp(raw_splats["scales"].to(device)),
        "opacities": torch.sigmoid(raw_splats["opacities"].to(device)),
    }

    if "colors" in raw_splats:
        splats["colors"] = torch.sigmoid(raw_splats["colors"].to(device))
        render_sh_degree = None
    else:
        splats["colors"] = torch.cat(
            [raw_splats["sh0"], raw_splats["shN"]], dim=1
        ).to(device)
        render_sh_degree = sh_degree

    return splats, render_sh_degree, step


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def _load_transform(path: str | None, matrix: np.ndarray | None = None) -> np.ndarray:
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float64)
    if not path:
        raise ValueError("--raw-to-map-transform is required for map-aligned BEV rendering")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = data["matrix"] if isinstance(data, dict) and "matrix" in data else data
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 raw-to-map transform, got {arr.shape}")
    return arr


def _point_transform(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    hom = np.ones(4, dtype=np.float64)
    hom[:3] = point
    out = matrix @ hom
    return out[:3] / out[3]


def _points_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.c_[points, np.ones(len(points), dtype=np.float64)]
    out = (matrix @ points_h.T).T
    return out[:, :3] / out[:, 3:4]


def _load_colmap_normalization(scene_dir: str | None) -> np.ndarray:
    if not scene_dir:
        raise ValueError("--scene-dir is required for map-aligned BEV rendering")
    try:
        from datasets.colmap import Parser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Map-aligned Gaussian BEV rendering requires gsplat examples on PYTHONPATH "
            "so datasets.colmap.Parser can be imported."
        ) from exc
    parser = Parser(scene_dir, factor=1, normalize=True, test_every=8)
    return np.asarray(parser.transform, dtype=np.float64)


def _apply_run_summary_defaults(args: argparse.Namespace, means_np: np.ndarray) -> None:
    if not args.run_summary:
        return
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("--run-summary requires PyYAML.") from exc

    summary = yaml.safe_load(Path(args.run_summary).read_text(encoding="utf-8")) or {}
    project = summary["result"]["project_metadata"]
    if args.scene_dir is None:
        args.scene_dir = project["reconstruction"]["colmap"]["gsplat_scene_dir"]

    ground_transform = np.asarray(
        project["ground_plane"]["ground_alignment_transform"], dtype=np.float64
    )
    alignment_transform = np.asarray(
        project["reconstruction_to_map_transform"]["transform"], dtype=np.float64
    )
    raw_to_map = alignment_transform @ ground_transform
    args.raw_to_map_matrix = raw_to_map

    raw_to_norm = _load_colmap_normalization(args.scene_dir)
    norm_to_map = raw_to_map @ np.linalg.inv(raw_to_norm)
    means_map = _points_transform(norm_to_map, means_np)

    lo = np.percentile(means_map, args.bounds_percentile, axis=0)
    hi = np.percentile(means_map, 100.0 - args.bounds_percentile, axis=0)
    xy_span = np.maximum(hi[:2] - lo[:2], 1e-6)
    pad = np.maximum(xy_span * args.padding_fraction, args.min_padding)
    lo[:2] -= pad
    hi[:2] += pad

    if args.map_bounds is None:
        args.map_bounds = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]
    if args.map_z_bounds is None:
        args.map_z_bounds = [float(lo[2]), float(hi[2])]
    if args.map_resolution is None:
        args.map_resolution = (float(args.map_bounds[1]) - float(args.map_bounds[0])) / float(args.width)
    if args.height <= 0:
        y_span = float(args.map_bounds[3]) - float(args.map_bounds[2])
        args.height = max(1, int(np.ceil(y_span / float(args.map_resolution))))


def _map_aligned_camera(
    args: argparse.Namespace,
    width: int,
    height: int,
) -> tuple[np.ndarray, torch.Tensor, dict[str, object]]:
    if not args.map_bounds or len(args.map_bounds) != 4:
        raise ValueError("--map-bounds X_MIN X_MAX Y_MIN Y_MAX is required")
    if not args.map_resolution or args.map_resolution <= 0:
        raise ValueError("--map-resolution must be positive")
    if not args.map_z_bounds or len(args.map_z_bounds) != 2:
        raise ValueError("--map-z-bounds Z_MIN Z_MAX is required")

    x_min, x_max, y_min, y_max = [float(v) for v in args.map_bounds]
    z_min, z_max = [float(v) for v in args.map_z_bounds]
    raw_to_map = _load_transform(args.raw_to_map_transform, getattr(args, "raw_to_map_matrix", None))
    map_to_norm = _load_colmap_normalization(args.scene_dir) @ np.linalg.inv(raw_to_map)

    x_center = (x_min + x_max) * 0.5
    y_center = (y_min + y_max) * 0.5
    z_span = max(z_max - z_min, 1e-6)
    camera_z = z_max + max(args.map_camera_z_margin, z_span * args.vertical_margin_fraction)
    origin_norm = _point_transform(map_to_norm, np.array([x_center, y_center, camera_z], dtype=np.float64))

    linear = map_to_norm[:3, :3]
    x_vec = linear @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_vec = linear @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    forward_vec = linear @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_scale = float(np.linalg.norm(x_vec))
    y_scale = float(np.linalg.norm(y_vec))
    forward_scale = float(np.linalg.norm(forward_vec))
    if min(x_scale, y_scale, forward_scale) <= 0:
        raise ValueError("Map-to-normalized transform has a degenerate camera axis")

    x_axis = x_vec / x_scale
    y_axis = y_vec / y_scale
    forward = forward_vec / forward_scale
    camtoworld = np.eye(4, dtype=np.float32)
    camtoworld[:3, 0] = x_axis.astype(np.float32)
    camtoworld[:3, 1] = y_axis.astype(np.float32)
    camtoworld[:3, 2] = forward.astype(np.float32)
    camtoworld[:3, 3] = origin_norm.astype(np.float32)

    fx = 1.0 / (float(args.map_resolution) * x_scale)
    fy = 1.0 / (float(args.map_resolution) * y_scale)
    cx = (x_center - x_min) / float(args.map_resolution)
    cy = (y_max - y_center) / float(args.map_resolution)
    K = torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    meta = {
        "render_mode": "map_aligned",
        "scene_dir": str(args.scene_dir),
        "run_summary": args.run_summary,
        "map_bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        },
        "map_resolution": float(args.map_resolution),
        "raw_to_map_transform": raw_to_map.tolist(),
        "map_to_normalized_transform": map_to_norm.tolist(),
        "normalized_units_per_map_unit": {
            "x": x_scale,
            "y": y_scale,
            "forward": forward_scale,
        },
    }
    return camtoworld, K, meta


def _auto_camera(
    args: argparse.Namespace,
    means_np: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, int, int, dict[str, object]]:
    vertical_axis = args.vertical_axis
    if vertical_axis == "auto":
        vertical_axis = _choose_vertical_axis(means_np)

    x_axis, y_axis, vertical = _camera_basis(vertical_axis)
    coords = np.stack(
        [means_np @ x_axis, means_np @ y_axis, means_np @ vertical], axis=1
    )
    lo = np.percentile(coords, args.bounds_percentile, axis=0)
    hi = np.percentile(coords, 100.0 - args.bounds_percentile, axis=0)

    xy_span = hi[:2] - lo[:2]
    pad = np.maximum(xy_span * args.padding_fraction, args.min_padding)
    lo[:2] -= pad
    hi[:2] += pad
    xy_span = np.maximum(hi[:2] - lo[:2], 1e-6)
    xy_center = (lo[:2] + hi[:2]) * 0.5

    scale = float(args.width) / float(xy_span[0])
    width = int(args.width)
    height = int(args.height) if args.height else int(np.ceil(xy_span[1] * scale))
    height = max(height, 1)
    if args.height:
        scale = min(float(width) / float(xy_span[0]), float(height) / float(xy_span[1]))

    vertical_span = max(float(hi[2] - lo[2]), 1e-6)
    camera_height = float(hi[2] + vertical_span * args.vertical_margin_fraction)
    origin = x_axis * xy_center[0] + y_axis * xy_center[1] + vertical * camera_height

    camtoworld = np.eye(4, dtype=np.float32)
    camtoworld[:3, 0] = x_axis
    camtoworld[:3, 1] = y_axis
    camtoworld[:3, 2] = -vertical
    camtoworld[:3, 3] = origin
    K = torch.tensor(
        [[[scale, 0.0, width * 0.5], [0.0, scale, height * 0.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    meta = {
        "render_mode": "auto_bounds",
        "vertical_axis": vertical_axis,
        "pixels_per_unit": scale,
        "xy_bounds": {
            "x_axis_min": float(lo[0]),
            "x_axis_max": float(hi[0]),
            "y_axis_min": float(lo[1]),
            "y_axis_max": float(hi[1]),
        },
        "vertical_bounds": {
            "min": float(lo[2]),
            "max": float(hi[2]),
            "camera_height": camera_height,
        },
    }
    return camtoworld, K, width, height, meta


def render_bev(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    splats, render_sh_degree, step = _load_splats(
        Path(args.checkpoint), device, args.sh_degree
    )

    means_np = splats["means"].detach().cpu().numpy()
    _apply_run_summary_defaults(args, means_np)
    if args.scene_dir or args.raw_to_map_transform or args.map_bounds:
        width = int(args.width)
        height = int(args.height)
        if width <= 0 or height <= 0:
            raise ValueError("Map-aligned BEV rendering requires explicit --width and --height")
        camtoworld, K_cpu, camera_meta = _map_aligned_camera(args, width, height)
    else:
        camtoworld, K_cpu, width, height, camera_meta = _auto_camera(args, means_np)

    viewmat = torch.linalg.inv(torch.from_numpy(camtoworld)).to(device)[None]
    K = K_cpu.to(device)
    background = torch.tensor([args.background], dtype=torch.float32, device=device)

    with torch.no_grad():
        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            render_mode="RGB+ED",
            sh_degree=render_sh_degree,
            packed=True,
            backgrounds=background,
            camera_model="ortho",
        )

    rgb = torch.clamp(renders[0, ..., :3], 0.0, 1.0).detach().cpu().numpy()
    depth = renders[0, ..., 3].detach().cpu().numpy()
    alpha = torch.clamp(alphas[0, ..., 0], 0.0, 1.0).detach().cpu().numpy()

    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
    alpha_u8 = (alpha * 255.0 + 0.5).astype(np.uint8)
    valid_depth = depth[alpha > args.alpha_depth_threshold]
    if valid_depth.size:
        d0, d1 = np.percentile(valid_depth, [1.0, 99.0])
        depth_norm = np.clip((depth - d0) / max(d1 - d0, 1e-6), 0.0, 1.0)
    else:
        depth_norm = np.zeros_like(depth)
    depth_u8 = (depth_norm * 255.0 + 0.5).astype(np.uint8)

    output_dir = Path(args.output_dir)
    rgb_path = output_dir / "gaussian_bev_rgb.png"
    alpha_path = output_dir / "gaussian_bev_alpha.png"
    depth_path = output_dir / "gaussian_bev_expected_depth.png"
    meta_path = output_dir / "gaussian_bev_metadata.json"

    _write_png(rgb_path, rgb_u8)
    _write_png(alpha_path, alpha_u8)
    _write_png(depth_path, depth_u8)

    meta: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "num_gaussians": int(means_np.shape[0]),
        "image_width": width,
        "image_height": height,
        "camera_to_world": camtoworld.tolist(),
        "alpha_nonzero_pixels": int(np.count_nonzero(alpha_u8)),
        "rgb_path": str(rgb_path),
        "alpha_path": str(alpha_path),
        "depth_path": str(depth_path),
        "raster_info_keys": sorted(info.keys()),
    }
    meta.update(camera_meta)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to gsplat ckpt_*.pt")
    parser.add_argument("--output-dir", required=True, help="Directory for BEV images")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--width", type=int, default=1600, help="Output image width")
    parser.add_argument("--height", type=int, default=0, help="Optional output height")
    parser.add_argument(
        "--run-summary",
        default=None,
        help="Pipeline run_summary.yaml; derives scene, RANSAC ground transform, and BEV bounds.",
    )
    parser.add_argument("--scene-dir", default=None, help="gsplat scene dir with images/ and sparse/")
    parser.add_argument(
        "--raw-to-map-transform",
        default=None,
        help="JSON file containing the mapper raw-world to map-world 4x4 transform.",
    )
    parser.add_argument(
        "--map-bounds",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help="Map-frame BEV bounds for map-aligned rendering.",
    )
    parser.add_argument("--map-resolution", type=float, default=None)
    parser.add_argument(
        "--map-z-bounds",
        type=float,
        nargs=2,
        metavar=("Z_MIN", "Z_MAX"),
        default=None,
        help="Map-frame z bounds for placing the orthographic camera.",
    )
    parser.add_argument(
        "--map-camera-z-margin",
        type=float,
        default=0.5,
        help="Minimum camera z margin above the selected map z max.",
    )
    parser.add_argument(
        "--vertical-axis",
        choices=["auto", "x", "y", "z"],
        default="auto",
        help="Scene axis treated as vertical; auto chooses the thinnest axis.",
    )
    parser.add_argument(
        "--bounds-percentile",
        type=float,
        default=0.5,
        help="Percentile trim for BEV bounds on each side.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.05,
        help="Extra BEV margin as a fraction of each axis span.",
    )
    parser.add_argument(
        "--min-padding",
        type=float,
        default=0.05,
        help="Minimum BEV margin in scene units.",
    )
    parser.add_argument(
        "--vertical-margin-fraction",
        type=float,
        default=0.5,
        help="Camera height margin above the highest selected point.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=0,
        help="Spherical harmonic degree to render; 0 uses the stable DC color.",
    )
    parser.add_argument(
        "--background",
        type=float,
        nargs=3,
        default=(0.32, 0.32, 0.32),
        help="RGB background for transparent areas, in 0..1.",
    )
    parser.add_argument(
        "--alpha-depth-threshold",
        type=float,
        default=0.02,
        help="Alpha threshold used only for depth preview normalization.",
    )
    parser.add_argument("--near-plane", type=float, default=0.001)
    parser.add_argument("--far-plane", type=float, default=1.0e10)
    return parser.parse_args()


def main() -> None:
    meta = render_bev(parse_args())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
