#!/usr/bin/env python3
"""Render a gsplat checkpoint from the COLMAP training camera views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from gsplat.rendering import rasterization


def _load_splats(checkpoint: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    data = torch.load(checkpoint, map_location="cpu")
    raw = data["splats"]
    splats = {
        "means": raw["means"].to(device),
        "quats": raw["quats"].to(device),
        "scales": torch.exp(raw["scales"].to(device)),
        "opacities": torch.sigmoid(raw["opacities"].to(device)),
    }
    if "colors" in raw:
        splats["colors"] = torch.sigmoid(raw["colors"].to(device))
    else:
        splats["colors"] = torch.cat([raw["sh0"], raw["shN"]], dim=1).to(device)
    return splats, int(data.get("step", -1))


def _resize(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    try:
        import cv2  # type: ignore

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height)))


def _pad_to_height(image: np.ndarray, height: int, value: int = 32) -> np.ndarray:
    if image.shape[0] >= height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], image.shape[2]), value, dtype=image.dtype)
    return np.concatenate([image, pad], axis=0)


def render(args: argparse.Namespace) -> dict[str, object]:
    try:
        from datasets.colmap import Parser  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Camera-view rendering requires gsplat examples on PYTHONPATH so datasets.colmap.Parser can be imported."
        ) from exc

    output_dir = Path(args.output_dir)
    renders_dir = output_dir / "renders"
    compares_dir = output_dir / "compares"
    renders_dir.mkdir(parents=True, exist_ok=True)
    compares_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    splats, step = _load_splats(Path(args.checkpoint), device)
    parser = Parser(args.scene_dir, factor=args.factor, normalize=True, test_every=args.test_every)

    selected = list(range(len(parser.image_names)))
    if args.indices:
        selected = [int(i) for i in args.indices]

    rows: list[np.ndarray] = []
    metrics = []
    background = torch.tensor([args.background], dtype=torch.float32, device=device)
    with torch.no_grad():
        for index in selected:
            image = imageio.imread(parser.image_paths[index])[..., :3]
            camera_id = parser.camera_ids[index]
            K = torch.from_numpy(parser.Ks_dict[camera_id]).float().to(device)[None]
            c2w = torch.from_numpy(parser.camtoworlds[index]).float().to(device)[None]
            height, width = image.shape[:2]
            colors, alphas, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=splats["scales"],
                opacities=splats["opacities"],
                colors=splats["colors"],
                viewmats=torch.linalg.inv(c2w),
                Ks=K,
                width=width,
                height=height,
                packed=args.packed,
                sh_degree=args.sh_degree,
                near_plane=args.near_plane,
                far_plane=args.far_plane,
                backgrounds=background,
            )
            render_rgb = torch.clamp(colors[0], 0.0, 1.0).detach().cpu().numpy()
            render_u8 = (render_rgb * 255.0 + 0.5).astype(np.uint8)
            image_u8 = image.astype(np.uint8)
            diff_u8 = np.abs(image_u8.astype(np.int16) - render_u8.astype(np.int16)).astype(np.uint8)
            compare = np.concatenate([image_u8, render_u8, diff_u8], axis=1)

            image_name = Path(parser.image_names[index]).stem
            render_path = renders_dir / f"{index:04d}_{image_name}_render.png"
            compare_path = compares_dir / f"{index:04d}_{image_name}_compare.png"
            imageio.imwrite(render_path, render_u8)
            imageio.imwrite(compare_path, compare)

            mse = float(np.mean((image_u8.astype(np.float32) / 255.0 - render_rgb) ** 2))
            psnr = float(-10.0 * np.log10(max(mse, 1e-12)))
            alpha_nonzero = int(np.count_nonzero((alphas[0, ..., 0] > 1e-4).detach().cpu().numpy()))
            metrics.append(
                {
                    "index": int(index),
                    "image_name": parser.image_names[index],
                    "render_path": str(render_path),
                    "compare_path": str(compare_path),
                    "mse": mse,
                    "psnr": psnr,
                    "alpha_nonzero_pixels": alpha_nonzero,
                }
            )

            row = _resize(compare, int(args.sheet_panel_width) * 3)
            rows.append(row)

    if rows:
        row_width = max(row.shape[1] for row in rows)
        padded_rows = []
        for row in rows:
            if row.shape[1] < row_width:
                pad = np.full((row.shape[0], row_width - row.shape[1], 3), 32, dtype=row.dtype)
                row = np.concatenate([row, pad], axis=1)
            padded_rows.append(row)
        sheet = np.concatenate(padded_rows, axis=0)
        sheet_path = output_dir / "camera_view_contact_sheet.png"
        imageio.imwrite(sheet_path, sheet)
    else:
        sheet_path = None

    meta = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "scene_dir": str(args.scene_dir),
        "num_gaussians": int(splats["means"].shape[0]),
        "sh_degree": int(args.sh_degree),
        "indices": selected,
        "mean_psnr": float(np.mean([m["psnr"] for m in metrics])) if metrics else None,
        "contact_sheet": str(sheet_path) if sheet_path else None,
        "metrics": metrics,
    }
    (output_dir / "camera_view_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--indices", nargs="*", default=None)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--sheet-panel-width", type=int, default=360)
    parser.add_argument("--background", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return parser.parse_args()


def main() -> None:
    meta = render(parse_args())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
