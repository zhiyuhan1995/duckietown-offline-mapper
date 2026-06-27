#!/usr/bin/env python3
"""Prepare an every-N-frame COLMAP scene for 3DGS training.

This is a 3DGS QA utility: it uses dense video sampling for Gaussian training
without asking VGGT to process hundreds of frames at once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to extract video frames.") from exc
    return cv2


def _require_pycolmap():
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pycolmap is required to build the COLMAP scene.") from exc
    return pycolmap


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _video_info(video_path: Path) -> dict[str, Any]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}


def _extract_every_n_frames(
    video_path: Path,
    image_dir: Path,
    interval: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    cv2 = _require_cv2()
    if overwrite:
        _reset_dir(image_dir)
    else:
        image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    selected: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % interval == 0:
            name = f"frame_{frame_index:06d}.png"
            path = image_dir / name
            if overwrite or not path.exists():
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to write frame image: {path}")
            selected.append(
                {
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "name": name,
                }
            )
        frame_index += 1
    cap.release()

    if not selected:
        raise RuntimeError(f"No frames selected from {video_path} with interval {interval}")
    return selected


def _database_stats(database_path: Path) -> dict[str, int]:
    if not database_path.exists():
        return {}
    con = sqlite3.connect(database_path)
    try:
        stats = {}
        for table in ["cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries"]:
            try:
                stats[table] = int(con.execute(f"select count(*) from {table}").fetchone()[0])
            except sqlite3.Error:
                pass
        return stats
    finally:
        con.close()


def _clear_pair_tables(database_path: Path) -> None:
    con = sqlite3.connect(database_path, timeout=60)
    try:
        for table in ["matches", "two_view_geometries"]:
            con.execute(f"delete from {table}")
        con.commit()
    finally:
        con.close()


def _largest_reconstruction_summary(sparse_dir: Path) -> dict[str, Any]:
    pycolmap = _require_pycolmap()
    candidates = []
    if (sparse_dir / "images.bin").exists():
        candidates.append(sparse_dir)
    candidates.extend(sorted(p for p in sparse_dir.iterdir() if p.is_dir()))

    best: tuple[int, Path, Any] | None = None
    for path in candidates:
        try:
            reconstruction = pycolmap.Reconstruction(str(path))
        except Exception:
            continue
        num_images = len(reconstruction.images)
        if best is None or num_images > best[0]:
            best = (num_images, path, reconstruction)

    if best is None:
        return {"model_path": None, "registered_images": 0, "points3D": 0}

    _, path, reconstruction = best
    return {
        "model_path": str(path),
        "registered_images": len(reconstruction.images),
        "cameras": len(reconstruction.cameras),
        "points3D": len(reconstruction.points3D),
        "image_names": [image.name for _, image in sorted(reconstruction.images.items())[:20]],
    }


def _run_colmap_scene(args: argparse.Namespace, image_names: list[str]) -> dict[str, Any]:
    pycolmap = _require_pycolmap()

    database_path = Path(args.output_dir) / "database.db"
    sparse_dir = Path(args.output_dir) / "sparse"
    if args.overwrite_colmap:
        if database_path.exists():
            database_path.unlink()
        _reset_dir(sparse_dir)
    else:
        sparse_dir.mkdir(parents=True, exist_ok=True)

    device = getattr(pycolmap.Device, args.device)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = args.camera_model
    if args.camera_params:
        reader_options.camera_params = args.camera_params

    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_num_features = int(args.max_num_features)
    sift_options.max_image_size = int(args.max_image_size)
    sift_options.gpu_index = str(args.gpu_index)
    sift_options.num_threads = int(args.num_threads)

    feature_seconds = None
    if args.skip_features:
        print("[prepare] skipping SIFT extraction", flush=True)
    else:
        print(f"[prepare] extracting SIFT for {len(image_names)} images", flush=True)
        t0 = time.time()
        pycolmap.extract_features(
            str(database_path),
            str(Path(args.output_dir) / "images"),
            image_list=image_names,
            camera_mode=pycolmap.CameraMode.SINGLE if args.single_camera else pycolmap.CameraMode.AUTO,
            camera_model=args.camera_model,
            reader_options=reader_options,
            sift_options=sift_options,
            device=device,
        )
        feature_seconds = time.time() - t0

    if args.clear_matches:
        print("[prepare] clearing existing matches and two-view geometries", flush=True)
        _clear_pair_tables(database_path)

    match_options = pycolmap.SiftMatchingOptions()
    match_options.gpu_index = str(args.gpu_index)
    match_options.num_threads = int(args.num_threads)
    match_options.max_num_matches = int(args.max_num_matches)
    match_options.guided_matching = bool(args.guided_matching)

    seq_options = pycolmap.SequentialMatchingOptions()
    seq_options.overlap = int(args.match_overlap)
    seq_options.quadratic_overlap = bool(args.quadratic_overlap)
    seq_options.loop_detection = bool(args.loop_detection)

    matching_seconds = None
    if args.skip_matching:
        print("[prepare] skipping sequential matching", flush=True)
    else:
        print(f"[prepare] sequential matching with overlap={seq_options.overlap}", flush=True)
        t0 = time.time()
        pycolmap.match_sequential(
            str(database_path),
            sift_options=match_options,
            matching_options=seq_options,
            device=device,
        )
        matching_seconds = time.time() - t0

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.multiple_models = False
    pipeline_options.max_num_models = 1
    pipeline_options.min_num_matches = int(args.min_num_matches)
    pipeline_options.num_threads = int(args.num_threads)
    pipeline_options.ba_refine_focal_length = bool(args.ba_refine_focal_length)
    pipeline_options.ba_refine_principal_point = bool(args.ba_refine_principal_point)
    pipeline_options.ba_refine_extra_params = bool(args.ba_refine_extra_params)
    pipeline_options.mapper.abs_pose_refine_focal_length = bool(args.abs_pose_refine_focal_length)
    pipeline_options.mapper.abs_pose_refine_extra_params = bool(args.abs_pose_refine_extra_params)
    pipeline_options.mapper.abs_pose_min_num_inliers = int(args.abs_pose_min_num_inliers)
    pipeline_options.mapper.abs_pose_min_inlier_ratio = float(args.abs_pose_min_inlier_ratio)
    pipeline_options.mapper.abs_pose_max_error = float(args.abs_pose_max_error)
    pipeline_options.mapper.min_focal_length_ratio = float(args.min_focal_length_ratio)
    pipeline_options.mapper.max_focal_length_ratio = float(args.max_focal_length_ratio)

    mapping_seconds = None
    reconstructions = {}
    if args.skip_mapping:
        print("[prepare] skipping incremental SfM", flush=True)
    else:
        print("[prepare] incremental SfM", flush=True)
        t0 = time.time()
        reconstructions = pycolmap.incremental_mapping(
            str(database_path),
            str(Path(args.output_dir) / "images"),
            str(sparse_dir),
            options=pipeline_options,
        )
        mapping_seconds = time.time() - t0

    summary = _largest_reconstruction_summary(sparse_dir)
    summary.update(
        {
            "num_reconstructions_returned": len(reconstructions),
            "database": str(database_path),
            "database_stats": _database_stats(database_path),
            "feature_seconds": feature_seconds,
            "matching_seconds": matching_seconds,
            "mapping_seconds": mapping_seconds,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="track.mp4")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--overwrite-colmap", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-matching", action="store_true")
    parser.add_argument("--skip-mapping", action="store_true")
    parser.add_argument("--clear-matches", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-index", default="-1")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--camera-params", default="")
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-image-size", type=int, default=1280)
    parser.add_argument("--max-num-features", type=int, default=8192)
    parser.add_argument("--max-num-matches", type=int, default=32768)
    parser.add_argument("--match-overlap", type=int, default=40)
    parser.add_argument("--quadratic-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop-detection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guided-matching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-num-matches", type=int, default=15)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--ba-refine-focal-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ba-refine-principal-point", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ba-refine-extra-params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-pose-refine-focal-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-pose-refine-extra-params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-pose-min-num-inliers", type=int, default=30)
    parser.add_argument("--abs-pose-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--abs-pose-max-error", type=float, default=12.0)
    parser.add_argument("--min-focal-length-ratio", type=float, default=0.1)
    parser.add_argument("--max-focal-length-ratio", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval < 1:
        raise ValueError("--interval must be >= 1")

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    info = _video_info(video_path)
    selected = _extract_every_n_frames(video_path, image_dir, args.interval, args.overwrite_images)
    image_names = [item["name"] for item in selected]

    (output_dir / "image_list.txt").write_text("\n".join(image_names) + "\n", encoding="utf-8")
    manifest = {
        "video": str(video_path),
        "video_info": info,
        "interval": args.interval,
        "selected_count": len(selected),
        "selected_first": selected[:5],
        "selected_last": selected[-5:],
        "images_dir": str(image_dir),
        "image_list": str(output_dir / "image_list.txt"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    colmap_summary = _run_colmap_scene(args, image_names)
    (output_dir / "colmap_summary.json").write_text(json.dumps(colmap_summary, indent=2), encoding="utf-8")
    print(json.dumps(colmap_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
