from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import streamlit as st
import yaml

from src.alignment import estimate_sim2
from src.bev import metadata_from_bounds, world_to_grid
from src.export import ros_image_from_occupancy, ros_image_from_occupancy_margin
from src.ground_texture import render_ground_texture_bev
from src.io_utils import deep_update, load_yaml
from src.keyframes import extract_keyframes, load_image_folder
from src.occupancy import fuse_occupancy, gradient_margin_from_occupancy, inflate_obstacles, remove_isolated_occupied_cells
from src.plane import fit_ground_plane
from src.pointcloud import PointCloud, load_ply
from src.segmentation import SemanticClass, colorize_semantic, segment_bev_rgb


st.set_page_config(page_title="Duckietown Offline Mapper", layout="wide")
st.title("Duckietown Offline Semantic-Occupancy BEV Mapper")


def _load_default_config() -> dict:
    return load_yaml(Path(__file__).resolve().parent / "configs" / "default.yaml")


if "config" not in st.session_state:
    st.session_state.config = _load_default_config()

config = st.session_state.config

METRIC_PIXELS_PER_METER = 1000
METRIC_RENDER_LIMIT_PIXELS = 32_000_000
METRIC_RENDER_VERSION = "world-origin-axes-v2"
METRIC_RENDER_FILENAME = "metric_aligned_map_world_origin_1000pxpm_v2.png"
PREVIEW_DISPLAY_WIDTH = 1400


def _image_rgb_from_bgr(frame):
    return frame[..., ::-1]


def _clear_cached_app_state() -> None:
    import gc

    for key in [
        "reconstruction",
        "reconstruction_cloud",
        "last_run",
        "last_run_input_signature",
        "frames",
        "frames_input_signature",
        "alignment_ground_texture_preview",
        "bev_metric_render",
        "semantic_preview",
        "occupancy_preview",
        "point_cloud_viewer_path",
        "alignment_source_path",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    _load_cloud_for_viewer.clear()
    gc.collect()


def _stored_uploaded_video_path(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    digest = hashlib.sha256(data).hexdigest()[:16]
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in uploaded_file.name)
    upload_dir = Path("outputs") / "streamlit_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{digest}_{safe_name}"
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return str(path)


def _input_signature(config: dict) -> dict:
    path = Path(str(config.get("input", {}).get("path", "")))
    signature: dict = {
        "path": str(path),
        "keyframe_interval": int(config.get("input", {}).get("keyframe_interval", 0) or 0),
        "max_keyframes": int(config.get("input", {}).get("max_keyframes", 0) or 0),
    }
    if path.exists():
        stat = path.stat()
        signature.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return signature


def _pipeline_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    child_cuda_devices = env.get("DUCKIETOWN_MAPPER_CUDA_VISIBLE_DEVICES")
    if child_cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = child_cuda_devices
    return env


def _run_pipeline_subprocess(config: dict) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / str(config["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = output_dir / "streamlit_runtime_config.yaml"
    with runtime_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    command = [
        sys.executable,
        str(project_root / "duckietown_offline_mapper" / "run_pipeline.py"),
        "--config",
        str(runtime_config_path),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=_pipeline_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        details = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
        raise RuntimeError(details or f"Pipeline subprocess failed with exit code {completed.returncode}")

    summary_path = output_dir / "run_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) or {}
    return summary.get("result", {})


@st.cache_data(show_spinner=False)
def _load_cloud_for_viewer(path: str, mtime: float | None = None) -> PointCloud:
    del mtime
    return load_ply(path)


def _default_viewer_path(export_dir: str) -> str:
    candidates = [
        Path(export_dir) / "aligned_point_cloud.ply",
        Path(export_dir) / "work" / "vggt_point_cloud.ply",
        Path(export_dir) / "ground_aligned_point_cloud.ply",
        Path("outputs/track_map_cluster_gpu01_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_cluster_gpu01_edge_complete/work/vggt_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/work/vggt_point_cloud.ply"),
        Path("outputs/track_map_local/aligned_point_cloud.ply"),
        Path("outputs/track_map/aligned_point_cloud.ply"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _default_alignment_source_path(export_dir: str) -> str:
    candidates = [
        Path(export_dir) / "ground_aligned_point_cloud.ply",
        Path(export_dir) / "aligned_point_cloud.ply",
        Path("outputs/track_map_cluster_gpu01_edge_complete/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map_cluster_gpu01_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map/aligned_point_cloud.ply"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _default_run_summary_path(export_dir: str) -> str:
    candidates = [
        Path(export_dir) / "run_summary.yaml",
        Path("outputs/track_map_cluster_gpu01_edge_complete") / "run_summary.yaml",
        Path("outputs/track_map_edge_complete") / "run_summary.yaml",
        Path("outputs/track_map") / "run_summary.yaml",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        current = Path(export_dir) / "run_summary.yaml"
        return str(current if current.exists() else max(existing, key=lambda path: path.stat().st_mtime))
    return str(candidates[0])


def _default_alignment_ground_texture_output_dir(run_summary_path: str) -> str:
    path = Path(run_summary_path)
    if path.name == "run_summary.yaml":
        return str(path.parent / "alignment_ground_texture")
    return "outputs/alignment_ground_texture"


def _resize_rgb_to_width(image: np.ndarray, width: int) -> np.ndarray:
    width = int(width)
    if width <= 0 or image.shape[1] == width:
        return image
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    try:
        import cv2  # type: ignore

        interpolation = cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(image, (width, height), interpolation=interpolation)
    except Exception:
        return image


@st.cache_data(show_spinner=False)
def _load_rgb_image(path: str, mtime: float) -> np.ndarray:
    del mtime
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        return bgr[..., ::-1]
    except Exception:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_rgb_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore

        cv2.imwrite(str(path), np.asarray(image, dtype=np.uint8)[..., ::-1])
        return
    except Exception:
        pass

    from PIL import Image

    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def _latest_bev_rgb_path(config: dict, last_run: dict | None) -> str:
    alignment_texture = st.session_state.get("alignment_ground_texture_preview")
    if alignment_texture:
        path = alignment_texture.get("paths", {}).get("texture")
        if path and Path(path).exists():
            return str(path)
    output_dir = Path(config["export"].get("output_dir", "outputs/track_map"))
    ground_texture = output_dir / "ground_texture" / "ground_texture_bev.png"
    if ground_texture.exists():
        return str(ground_texture)
    if last_run:
        path = last_run.get("paths", {}).get("bev_rgb")
        if path:
            return str(path)
    return str(output_dir / "bev_rgb.png")


def _is_file_fresh_enough(path: Path, reference_paths: list[Path]) -> bool:
    if not path.exists():
        return False
    existing_refs = [ref for ref in reference_paths if ref.exists()]
    if not existing_refs:
        return True
    return path.stat().st_mtime >= max(ref.stat().st_mtime for ref in existing_refs)


def _latest_metric_render_source_paths(config: dict, last_run: dict | None) -> tuple[str, str]:
    alignment_texture = st.session_state.get("alignment_ground_texture_preview")
    if alignment_texture:
        texture_path = alignment_texture.get("paths", {}).get("texture")
        metadata_path = alignment_texture.get("paths", {}).get("metadata")
        if texture_path and metadata_path and Path(texture_path).exists() and Path(metadata_path).exists():
            return str(texture_path), str(metadata_path)

    output_dir = Path(config["export"].get("output_dir", "outputs/track_map"))
    run_summary = output_dir / "run_summary.yaml"
    paired_candidates = [
        (
            output_dir / "ground_texture" / "ground_texture_bev.png",
            output_dir / "ground_texture" / "ground_texture_metadata.yaml",
        ),
        (
            output_dir / "alignment_ground_texture" / "ground_texture_bev.png",
            output_dir / "alignment_ground_texture" / "ground_texture_metadata.yaml",
        ),
    ]
    for texture_path, metadata_path in paired_candidates:
        references = [run_summary] if "alignment_ground_texture" in texture_path.parts else []
        if texture_path.exists() and metadata_path.exists() and _is_file_fresh_enough(metadata_path, references):
            return str(texture_path), str(metadata_path)

    if last_run:
        path = last_run.get("paths", {}).get("bev_rgb")
        if path:
            return str(path), _latest_output_path(config, last_run, "map_metadata", "map_metadata.yaml")
    return str(output_dir / "bev_rgb.png"), _latest_output_path(config, last_run, "map_metadata", "map_metadata.yaml")


def _looks_like_metric_render_output_path(path: str | Path) -> bool:
    return Path(path).name.startswith("metric_aligned_map")


def _is_metric_aligned_bev_path(path: str | Path, config: dict) -> bool:
    candidate = Path(path)
    metric_render = st.session_state.get("bev_metric_render")
    if metric_render and metric_render.get("render_version") == METRIC_RENDER_VERSION:
        rendered_path = metric_render.get("path")
        if rendered_path and Path(rendered_path).exists() and candidate.resolve() == Path(rendered_path).resolve():
            return True
    default_metric = Path(config["export"].get("output_dir", "outputs/track_map")) / METRIC_RENDER_FILENAME
    return default_metric.exists() and candidate.resolve() == default_metric.resolve()


def _latest_output_path(config: dict, last_run: dict | None, key: str, filename: str) -> str:
    if key == "map_metadata":
        alignment_texture = st.session_state.get("alignment_ground_texture_preview")
        if alignment_texture:
            path = alignment_texture.get("paths", {}).get("metadata")
            if path and Path(path).exists():
                return str(path)
    if last_run:
        path = last_run.get("paths", {}).get(key)
        if path:
            return str(path)
    return str(Path(config["export"].get("output_dir", "outputs/track_map")) / filename)


def _metadata_from_map_yaml(path: str | Path):
    data = load_yaml(path)
    meta = data.get("metadata", data)
    return metadata_from_bounds(
        float(meta["x_min"]),
        float(meta["x_max"]),
        float(meta["y_min"]),
        float(meta["y_max"]),
        float(meta["resolution"]),
        str(meta.get("frame_id", "map")),
    )


def _metric_view_shape(metadata, pixels_per_meter: int = 1000) -> tuple[int, int]:
    ppm = int(pixels_per_meter)
    x_span = max(float(metadata.x_max) - float(metadata.x_min), float(metadata.resolution))
    y_span = max(float(metadata.y_max) - float(metadata.y_min), float(metadata.resolution))
    return max(1, int(np.ceil(x_span * ppm))), max(1, int(np.ceil(y_span * ppm)))


def _world_to_metric_view_pixel(x: float, y: float, metadata, pixels_per_meter: int = 1000) -> tuple[float, float]:
    width, height = _metric_view_shape(metadata, pixels_per_meter)
    ppm = float(pixels_per_meter)
    u = width - 0.5 - (float(x) - float(metadata.x_min)) * ppm
    v = height - 0.5 - (float(y) - float(metadata.y_min)) * ppm
    return float(u), float(v)


def _render_metric_aligned_map(
    bev_rgb: np.ndarray,
    metadata,
    pixels_per_meter: int = METRIC_PIXELS_PER_METER,
    unknown_rgb: tuple[int, int, int] = (80, 80, 80),
    draw_axes: bool = True,
) -> np.ndarray:
    ppm = int(pixels_per_meter)
    if ppm <= 0:
        raise ValueError("pixels_per_meter must be positive")

    x_min = float(metadata.x_min)
    x_max = float(metadata.x_max)
    y_min = float(metadata.y_min)
    y_max = float(metadata.y_max)
    width, height = _metric_view_shape(metadata, ppm)
    if width * height > METRIC_RENDER_LIMIT_PIXELS:
        raise ValueError(
            f"Metric view would be too large: {width} x {height} pixels "
            f"({width * height / 1_000_000:.2f} MP, limit {METRIC_RENDER_LIMIT_PIXELS / 1_000_000:.0f} MP). "
            "Crop the ROI or use tighter alignment bounds before rendering at 1000 px/m."
        )

    xs = x_min + (width - 1 - np.arange(width, dtype=np.float32) + 0.5) / float(ppm)
    ys = y_min + (height - 1 - np.arange(height, dtype=np.float32) + 0.5) / float(ppm)
    src_x = ((xs[None, :] - x_min) / float(metadata.resolution) - 0.5).astype(np.float32)
    src_y = ((y_max - ys[:, None]) / float(metadata.resolution) - 0.5).astype(np.float32)
    map_x = np.broadcast_to(src_x, (height, width)).copy()
    map_y = np.broadcast_to(src_y, (height, width)).copy()

    try:
        import cv2  # type: ignore

        rendered = cv2.remap(
            np.asarray(bev_rgb, dtype=np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(int(c) for c in unknown_rgb),
        )
    except Exception:
        uu = np.rint(map_x).astype(int)
        vv = np.rint(map_y).astype(int)
        rendered = np.full((height, width, 3), unknown_rgb, dtype=np.uint8)
        valid = (uu >= 0) & (uu < bev_rgb.shape[1]) & (vv >= 0) & (vv < bev_rgb.shape[0])
        rendered[valid] = bev_rgb[vv[valid], uu[valid]]

    if draw_axes:
        rendered = rendered.copy()
        try:
            import cv2  # type: ignore

            origin_u, origin_v = _world_to_metric_view_pixel(0.0, 0.0, metadata, ppm)
            if not (0.0 <= origin_u < width and 0.0 <= origin_v < height):
                return rendered
            origin = (int(round(origin_u)), int(round(origin_v)))
            x_end = (max(0, origin[0] - min(ppm, origin[0])), origin[1])
            y_end = (origin[0], max(0, origin[1] - min(ppm, origin[1])))
            cv2.arrowedLine(rendered, origin, x_end, (255, 64, 64), 3, cv2.LINE_AA, tipLength=0.04)
            cv2.arrowedLine(rendered, origin, y_end, (64, 220, 64), 3, cv2.LINE_AA, tipLength=0.04)
            cv2.circle(rendered, origin, 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(rendered, origin, 6, (0, 180, 255), -1, cv2.LINE_AA)
            cv2.drawMarker(rendered, origin, (0, 0, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            if width > 180 and height > 80:
                label_x = min(width - 170, max(4, origin[0] + 12))
                label_y = min(height - 10, max(22, origin[1] - 12))
                cv2.putText(rendered, "world (0,0)", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 180, 255), 2, cv2.LINE_AA)
                cv2.putText(rendered, "+x 1m", (max(4, x_end[0] + 8), min(height - 12, max(18, x_end[1] + 22))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 64, 64), 2, cv2.LINE_AA)
                cv2.putText(rendered, "+y 1m", (min(width - 90, max(4, y_end[0] + 10)), max(22, y_end[1] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (64, 220, 64), 2, cv2.LINE_AA)
        except Exception:
            pass
    return rendered


def _obstacle_grid_from_aligned_height(cloud: PointCloud, metadata, height_threshold: float) -> np.ndarray:
    obstacle = np.zeros((metadata.height, metadata.width), dtype=bool)
    if cloud.size == 0:
        return obstacle
    points = cloud.points
    mask = points[:, 2] >= float(height_threshold)
    if not np.any(mask):
        return obstacle
    u, v = world_to_grid(points[mask, 0], points[mask, 1], metadata)
    valid = (u >= 0) & (u < metadata.width) & (v >= 0) & (v < metadata.height)
    obstacle[v[valid], u[valid]] = True
    return obstacle


def _obstacle_grid_from_metric_view_height(
    cloud: PointCloud,
    metadata,
    image_shape: tuple[int, int],
    height_threshold: float,
    pixels_per_meter: float = METRIC_PIXELS_PER_METER,
) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    obstacle = np.zeros((height, width), dtype=bool)
    if cloud.size == 0:
        return obstacle
    points = cloud.points
    mask = points[:, 2] >= float(height_threshold)
    if not np.any(mask):
        return obstacle
    ppm = float(pixels_per_meter)
    u = (width - 1 - np.floor((points[mask, 0] - float(metadata.x_min)) * ppm)).astype(int)
    v = (height - 1 - np.floor((points[mask, 1] - float(metadata.y_min)) * ppm)).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    obstacle[v[valid], u[valid]] = True
    return obstacle


def _resample_bev_rgb_to_resolution(bev_rgb: np.ndarray, metadata, resolution: float) -> tuple[np.ndarray, object]:
    target_metadata = metadata_from_bounds(
        float(metadata.x_min),
        float(metadata.x_max),
        float(metadata.y_min),
        float(metadata.y_max),
        float(resolution),
        str(metadata.frame_id),
    )
    target_size = (int(target_metadata.width), int(target_metadata.height))
    if bev_rgb.shape[:2] == (target_metadata.height, target_metadata.width):
        return bev_rgb, target_metadata
    try:
        import cv2  # type: ignore

        interpolation = cv2.INTER_AREA if target_size[0] < bev_rgb.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(bev_rgb, target_size, interpolation=interpolation), target_metadata
    except Exception:
        from PIL import Image

        resized = Image.fromarray(np.asarray(bev_rgb, dtype=np.uint8)).resize(target_size, Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.uint8), target_metadata


def _auto_metadata_for_cloud(cloud: PointCloud, resolution: float):
    points = cloud.points
    x_min, x_max = np.percentile(points[:, 0], [0.25, 99.75])
    y_min, y_max = np.percentile(points[:, 1], [0.25, 99.75])
    padding = 0.08
    return metadata_from_bounds(
        float(x_min - padding),
        float(x_max + padding),
        float(y_min - padding),
        float(y_max + padding),
        resolution,
    )


def _rasterize_clickable_bev(
    cloud: PointCloud,
    metadata,
    max_points: int,
    point_radius: int,
    unknown_rgb: tuple[int, int, int],
    seed: int = 23,
) -> np.ndarray:
    from src.bev import world_to_grid

    points = cloud.points
    colors = cloud.colors
    if cloud.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(cloud.size, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    image = np.full((metadata.height, metadata.width, 3), unknown_rgb, dtype=np.uint8)
    u, v = world_to_grid(points[:, 0], points[:, 1], metadata)
    valid = (u >= 0) & (u < metadata.width) & (v >= 0) & (v < metadata.height)
    if not np.any(valid):
        return image

    try:
        import cv2  # type: ignore

        radius = max(0, int(point_radius))
        for uu, vv, color in zip(u[valid], v[valid], colors[valid]):
            if radius <= 0:
                image[vv, uu] = color
            else:
                cv2.circle(image, (int(uu), int(vv)), radius, tuple(int(c) for c in color.tolist()), -1)
    except Exception:
        image[v[valid], u[valid]] = colors[valid]
    return image


def _render_clickable_bev_hd(
    cloud: PointCloud,
    metadata,
    max_points: int,
    target_width: int,
    point_radius: int,
    unknown_rgb: tuple[int, int, int],
    seed: int = 23,
) -> np.ndarray:
    points = cloud.points
    colors = cloud.colors
    if cloud.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(cloud.size, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    x_span = max(float(metadata.x_max - metadata.x_min), 1e-9)
    y_span = max(float(metadata.y_max - metadata.y_min), 1e-9)
    width = max(320, int(target_width))
    height = max(240, int(round(width * y_span / x_span)))
    image = np.full((height, width, 3), unknown_rgb, dtype=np.uint8)

    u = np.round((points[:, 0] - metadata.x_min) / x_span * (width - 1)).astype(int)
    v = np.round((metadata.y_max - points[:, 1]) / y_span * (height - 1)).astype(int)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return image

    u = u[valid]
    v = v[valid]
    colors = colors[valid]
    radius = max(0, int(point_radius))
    if radius <= 0:
        image[v, u] = colors
        return image

    try:
        import cv2  # type: ignore

        line_type = cv2.LINE_AA if radius <= 3 else cv2.LINE_8
        for uu, vv, color in zip(u, v, colors):
            cv2.circle(image, (int(uu), int(vv)), radius, tuple(int(c) for c in color.tolist()), -1, line_type)
    except Exception:
        image[v, u] = colors
    return image


def _bev_display_pixel_to_world(click_x: float, click_y: float, image: np.ndarray, metadata) -> tuple[float, float]:
    height, width = image.shape[:2]
    x = metadata.x_min + (float(click_x) + 0.5) / max(width, 1) * (metadata.x_max - metadata.x_min)
    y = metadata.y_max - (float(click_y) + 0.5) / max(height, 1) * (metadata.y_max - metadata.y_min)
    return float(x), float(y)


def _map_xy_to_alignment_source_xy(run_summary_path: str | Path, map_x: float, map_y: float) -> tuple[float, float]:
    summary = load_yaml(run_summary_path)
    transform = np.asarray(
        summary.get("result", {})
        .get("project_metadata", {})
        .get("reconstruction_to_map_transform", {})
        .get("transform", np.eye(4)),
        dtype=np.float64,
    )
    if transform.shape != (4, 4):
        raise ValueError("run_summary.yaml has an invalid reconstruction_to_map_transform")
    source = np.linalg.inv(transform) @ np.array([float(map_x), float(map_y), 0.0, 1.0], dtype=np.float64)
    return float(source[0]), float(source[1])


def _point_cloud_figure(
    cloud: PointCloud,
    max_points: int,
    point_size: float,
    color_mode: str,
    viewer_theme: str,
    seed: int = 13,
):
    import plotly.graph_objects as go

    points = cloud.points
    colors = cloud.colors
    confidence = cloud.confidence
    if cloud.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(cloud.size, size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
        confidence = None if confidence is None else confidence[idx]

    chunk_size = 80_000 if color_mode == "RGB" else 120_000
    hover_enabled = len(points) <= 200_000
    z_min, z_max = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    conf_min = conf_max = None
    if confidence is not None:
        conf_min, conf_max = float(np.min(confidence)), float(np.max(confidence))

    traces = []
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        marker: dict = {"size": point_size, "opacity": 0.92, "line": {"width": 0}}
        if color_mode == "RGB":
            marker["color"] = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors[start:end]]
        elif color_mode == "Height":
            marker.update(
                {
                    "color": points[start:end, 2],
                    "colorscale": "Viridis",
                    "showscale": start == 0,
                    "cmin": z_min,
                    "cmax": z_max,
                }
            )
            if start == 0:
                marker["colorbar"] = {"title": "z"}
        elif color_mode == "Confidence" and confidence is not None:
            marker.update(
                {
                    "color": confidence[start:end],
                    "colorscale": "Turbo",
                    "showscale": start == 0,
                    "cmin": conf_min,
                    "cmax": conf_max,
                }
            )
            if start == 0:
                marker["colorbar"] = {"title": "conf"}
        else:
            marker["color"] = "#5fb3ff"

        trace_kwargs = {
            "x": points[start:end, 0],
            "y": points[start:end, 1],
            "z": points[start:end, 2],
            "mode": "markers",
            "marker": marker,
            "showlegend": False,
            "name": f"points {start:,}-{end:,}",
        }
        if hover_enabled:
            trace_kwargs["hovertemplate"] = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
        else:
            trace_kwargs["hoverinfo"] = "skip"
        traces.append(go.Scatter3d(**trace_kwargs))

    fig = go.Figure(data=traces)
    template = "plotly_white" if viewer_theme == "Light" else "plotly_dark"
    fig.update_layout(
        template=template,
        height=660,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        scene={
            "aspectmode": "data",
            "dragmode": "orbit",
            "xaxis": {"title": "x", "showbackground": False},
            "yaxis": {"title": "y", "showbackground": False},
            "zaxis": {"title": "z", "showbackground": False},
        },
        uirevision="point-cloud-viewer",
    )
    return fig


def _load_ros_map_for_world_view(map_yaml_path: str | Path) -> dict:
    yaml_path = Path(map_yaml_path)
    map_data = load_yaml(yaml_path)
    image_path = Path(str(map_data.get("image", "")))
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    resolution = float(map_data["resolution"])
    origin = map_data.get("origin", [0.0, 0.0, 0.0])
    x_min = float(origin[0])
    y_min = float(origin[1])
    yaw = float(origin[2]) if len(origin) > 2 else 0.0
    image = _load_rgb_image(str(image_path), image_path.stat().st_mtime)
    height, width = image.shape[:2]
    x_max = x_min + float(width) * resolution
    y_max = y_min + float(height) * resolution
    return {
        "map_yaml": str(yaml_path),
        "image_path": str(image_path),
        "image": image,
        "resolution": resolution,
        "origin": [x_min, y_min, yaw],
        "bounds": [x_min, x_max, y_min, y_max],
        "size": [int(width), int(height)],
        "metadata": map_data,
    }


@st.cache_data(show_spinner=False)
def _load_numpy_array(path: str, mtime: float) -> np.ndarray:
    del mtime
    return np.load(path)


def _control_points_for_world_view(metadata_path: str | Path, config: dict) -> list[dict]:
    path = Path(metadata_path)
    if path.exists():
        data = load_yaml(path)
        control_points = data.get("reconstruction_to_map_transform", {}).get("control_points", [])
        if control_points:
            return control_points
    return list(config.get("alignment", {}).get("control_points", []))


def _world_map_figure(
    map_info: dict,
    control_points: list[dict],
    height_px: int,
    grid_spacing_m: float,
    padding_m: float,
    show_axes: bool,
    show_control_points: bool,
):
    import plotly.graph_objects as go

    image = map_info["image"]
    resolution = float(map_info["resolution"])
    x_min, x_max, y_min, y_max = [float(v) for v in map_info["bounds"]]
    view_x_min = x_min - padding_m
    view_x_max = x_max + padding_m
    view_y_min = y_min - padding_m
    view_y_max = y_max + padding_m

    fig = go.Figure()
    fig.add_trace(
        go.Image(
            z=image,
            x0=x_min,
            y0=y_max,
            dx=resolution,
            dy=-resolution,
            name="map.png",
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<extra>map.png</extra>",
        )
    )

    shapes = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "y",
            "x0": x_min,
            "x1": x_max,
            "y0": y_min,
            "y1": y_max,
            "line": {"color": "#38bdf8", "width": 2},
            "fillcolor": "rgba(0,0,0,0)",
        }
    ]
    if show_axes:
        if view_y_min <= 0.0 <= view_y_max:
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": view_x_min,
                    "x1": view_x_max,
                    "y0": 0.0,
                    "y1": 0.0,
                    "line": {"color": "#ef4444", "width": 3},
                }
            )
        if view_x_min <= 0.0 <= view_x_max:
            shapes.append(
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y",
                    "x0": 0.0,
                    "x1": 0.0,
                    "y0": view_y_min,
                    "y1": view_y_max,
                    "line": {"color": "#22c55e", "width": 3},
                }
            )

    fig.update_layout(shapes=shapes)

    if show_axes:
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers+text",
                marker={"size": 12, "color": "#f59e0b", "line": {"color": "#111827", "width": 1}},
                text=["world (0,0)"],
                textposition="top right",
                name="world origin",
                hovertemplate="world origin<br>x=0.000 m<br>y=0.000 m<extra></extra>",
            )
        )
        fig.add_annotation(
            x=min(view_x_max, 1.0),
            y=0.0,
            ax=0.0,
            ay=0.0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="+x",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="#ef4444",
            font={"color": "#ef4444"},
        )
        fig.add_annotation(
            x=0.0,
            y=min(view_y_max, 1.0),
            ax=0.0,
            ay=0.0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="+y",
            showarrow=True,
            arrowhead=3,
            arrowwidth=2,
            arrowcolor="#22c55e",
            font={"color": "#22c55e"},
        )

    if show_control_points and control_points:
        target_xy = np.array([p["target"][:2] for p in control_points if len(p.get("target", [])) >= 2], dtype=np.float64)
        if target_xy.size:
            labels = [f"target {i}" for i in range(len(target_xy))]
            fig.add_trace(
                go.Scatter(
                    x=target_xy[:, 0],
                    y=target_xy[:, 1],
                    mode="markers+text",
                    marker={"size": 12, "color": "#a855f7", "symbol": "x", "line": {"width": 2}},
                    text=labels,
                    textposition="top center",
                    name="alignment targets",
                    hovertemplate="%{text}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<extra></extra>",
                )
            )

    fig.update_xaxes(
        title="x (m)",
        range=[view_x_min, view_x_max],
        constrain="domain",
        gridcolor="rgba(148, 163, 184, 0.35)",
        zeroline=False,
        dtick=float(grid_spacing_m),
    )
    fig.update_yaxes(
        title="y (m)",
        range=[view_y_min, view_y_max],
        scaleanchor="x",
        scaleratio=1,
        gridcolor="rgba(148, 163, 184, 0.35)",
        zeroline=False,
        dtick=float(grid_spacing_m),
    )
    fig.update_layout(
        template="plotly_dark",
        height=int(height_px),
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="closest",
        dragmode="pan",
        uirevision="world-map-viewer",
    )
    return fig


tabs = st.tabs(
    [
        "Input",
        "Reconstruction",
        "Ground Plane",
        "Alignment",
        "Crop / ROI",
        "BEV",
        "Semantic",
        "Occupancy",
        "World Map",
        "Export",
    ]
)

with tabs[0]:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    folder = st.text_input("Or image folder / video path", value=config["input"].get("path", "track.mp4"))
    config["input"]["keyframe_interval"] = st.number_input(
        "Keyframe interval", min_value=1, value=int(config["input"].get("keyframe_interval", 30))
    )
    config["input"]["max_keyframes"] = st.number_input(
        "Max keyframes", min_value=1, value=int(config["input"].get("max_keyframes", 40))
    )
    if uploaded:
        config["input"]["path"] = _stored_uploaded_video_path(uploaded)
    else:
        config["input"]["path"] = folder

    current_input_signature = _input_signature(config)
    previous_input_signature = st.session_state.get("active_input_signature")
    if previous_input_signature is None:
        st.session_state.active_input_signature = current_input_signature
    elif previous_input_signature != current_input_signature:
        _clear_cached_app_state()
        st.session_state.active_input_signature = current_input_signature
        st.session_state.input_changed_notice = True

    st.write(
        {
            "active_input": config["input"]["path"],
            "keyframe_interval": int(config["input"]["keyframe_interval"]),
            "max_keyframes": int(config["input"]["max_keyframes"]),
        }
    )
    if st.session_state.get("input_changed_notice", False):
        del st.session_state.input_changed_notice
        st.warning("Input changed. Cleared stale keyframe previews, reconstruction state, and downstream previews.")

    if st.button("Preview keyframes"):
        path = Path(config["input"]["path"])
        frames = load_image_folder(path, int(config["input"]["max_keyframes"])) if path.is_dir() else extract_keyframes(
            path,
            int(config["input"]["keyframe_interval"]),
            int(config["input"]["max_keyframes"]),
        )
        st.session_state.frames = frames
        st.session_state.frames_input_signature = current_input_signature
    frames = st.session_state.get("frames", [])
    if frames:
        if st.session_state.get("frames_input_signature") != current_input_signature:
            st.warning("Displayed keyframes are stale. Press Preview keyframes to refresh them.")
        cols = st.columns(min(4, len(frames)))
        for i, frame in enumerate(frames[:8]):
            cols[i % len(cols)].image(_image_rgb_from_bgr(frame.image_bgr), caption=f"frame {frame.index}", use_container_width=True)

with tabs[1]:
    config["reconstruction"]["backend"] = "vggt_sfm"
    st.info("Reconstruction backend: VGGT-SfM only")
    current_input_signature = _input_signature(config)
    st.write(
        {
            "current_input": config["input"]["path"],
            "keyframe_interval": int(config["input"]["keyframe_interval"]),
            "max_keyframes": int(config["input"]["max_keyframes"]),
        }
    )
    if st.session_state.get("last_run_input_signature") and st.session_state.get("last_run_input_signature") != current_input_signature:
        st.warning("The last reconstruction was generated from a different input. Press Run reconstruction to rebuild.")
    vggt = config["reconstruction"].setdefault("vggt", {})
    sfm = config["reconstruction"].setdefault("sfm", {})
    c1, c2, c3 = st.columns(3)
    vggt["model_id"] = c1.text_input("VGGT model id", value=str(vggt.get("model_id", "facebook/VGGT-1B")))
    vggt["device"] = c2.selectbox("Device", ["cuda", "auto", "cpu"], index=["cuda", "auto", "cpu"].index(str(vggt.get("device", "cuda"))))
    vggt["dtype"] = c3.selectbox("Dtype", ["auto", "bfloat16", "float16", "float32"], index=["auto", "bfloat16", "float16", "float32"].index(str(vggt.get("dtype", "auto"))))
    vggt["checkpoint_url"] = st.text_input("Optional checkpoint URL", value=str(vggt.get("checkpoint_url") or "")) or None
    c1, c2, c3, c4 = st.columns(4)
    vggt["preprocess_mode"] = c1.selectbox("Preprocess", ["crop", "pad"], index=["crop", "pad"].index(str(vggt.get("preprocess_mode", "crop"))))
    vggt["confidence_threshold"] = c2.number_input("Confidence threshold", min_value=0.0, value=float(vggt.get("confidence_threshold", 5.0)))
    vggt["sample_stride"] = c3.number_input("Sample stride", min_value=1, value=int(vggt.get("sample_stride", 2)))
    vggt["max_points"] = c4.number_input("Max fused points", min_value=1000, value=int(vggt.get("max_points", 250000)), step=1000)
    c1, c2 = st.columns(2)
    vggt["relax_ground_confidence"] = c1.checkbox(
        "Relax confidence for black road / lane colors",
        value=bool(vggt.get("relax_ground_confidence", True)),
    )
    vggt["ground_confidence_threshold"] = c2.number_input(
        "Ground-color confidence threshold",
        min_value=0.0,
        value=float(vggt.get("ground_confidence_threshold", 1.2)),
        step=0.1,
    )
    vggt["use_point_map"] = st.checkbox(
        "Use VGGT point map instead of depth unprojection",
        value=bool(vggt.get("use_point_map", False)),
        help="For dense Duckietown road surfaces, leave this off. Depth unprojection usually keeps black road areas more complete.",
    )
    vggt["save_depth"] = st.checkbox("Save depth arrays", value=bool(vggt.get("save_depth", False)))
    sfm["save_colmap"] = st.checkbox("Export COLMAP sparse model", value=bool(sfm.get("save_colmap", False)))
    sfm["bundle_adjustment"] = st.checkbox("Run VGGT-SfM bundle adjustment", value=bool(sfm.get("bundle_adjustment", False)))
    if st.button("Clear app cache"):
        _clear_cached_app_state()
        st.success("Cleared cached Streamlit point clouds and in-memory reconstruction state.")
    if st.button("Run reconstruction"):
        try:
            run_input_signature = _input_signature(config)
            _clear_cached_app_state()
            with st.spinner("Running VGGT-SfM in an isolated subprocess..."):
                res = _run_pipeline_subprocess(config)
            st.session_state.last_run = res
            st.session_state.last_run_input_signature = run_input_signature
            st.success(f"Exported to {config['export']['output_dir']}")
            st.write(res.get("paths", {}))
        except Exception as exc:
            st.error(str(exc))
    recon = st.session_state.get("reconstruction")
    cloud = recon.point_cloud if recon else None
    if cloud is not None:
        st.metric("Point count", cloud.size)
        st.write({"xyz_min": cloud.points.min(axis=0).tolist(), "xyz_max": cloud.points.max(axis=0).tolist()})

    st.subheader("Point Cloud Viewer")
    viewer_source = st.radio(
        "Point source",
        ["Exported PLY", "Latest reconstruction in memory"],
        horizontal=True,
        key="point_cloud_viewer_source",
    )
    viewer_path = st.text_input(
        "PLY path",
        value=_default_viewer_path(config["export"].get("output_dir", "outputs/track_map")),
        disabled=viewer_source != "Exported PLY",
        key="point_cloud_viewer_path",
    )
    viewer_cloud = None
    if viewer_source == "Latest reconstruction in memory":
        viewer_cloud = cloud
        if viewer_cloud is None:
            st.info("Run reconstruction first, or switch to Exported PLY.")
    else:
        path = Path(viewer_path)
        if path.exists():
            viewer_cloud = _load_cloud_for_viewer(str(path))
        else:
            st.warning(f"Point cloud file not found: {path}")

    if viewer_cloud is not None and viewer_cloud.size > 0:
        c1, c2, c3, c4 = st.columns(4)
        max_displayable_points = max(1000, int(viewer_cloud.size))
        default_display_points = min(60000, max_displayable_points)
        requested_display_points = c1.number_input(
            "Displayed points",
            min_value=1000,
            max_value=2_000_000,
            value=default_display_points,
            step=1000,
            key="point_cloud_viewer_displayed_points_v2",
        )
        viewer_max_points = min(int(requested_display_points), max_displayable_points)
        viewer_point_size = c2.slider("Point size", 0.5, 5.0, 1.6, 0.1)
        viewer_color_mode = c3.selectbox("Color mode", ["RGB", "Height", "Confidence", "Solid"])
        viewer_theme = c4.selectbox("Viewer background", ["Dark", "Light"])
        viewer_chunk_size = 80_000 if viewer_color_mode == "RGB" else 120_000
        viewer_trace_count = max(1, (int(viewer_max_points) + viewer_chunk_size - 1) // viewer_chunk_size)
        if requested_display_points > max_displayable_points:
            st.warning(
                f"Requested {int(requested_display_points)} points, but this cloud only has "
                f"{max_displayable_points}. Showing all available points."
            )
        st.caption(
            f"Actually plotting {viewer_max_points:,} / {viewer_cloud.size:,} points as "
            f"{viewer_trace_count} WebGL traces of up to {viewer_chunk_size:,} points each. "
            "Hover is disabled above 200,000 points."
        )
        st.write(
            {
                "points": viewer_cloud.size,
                "plotted_points": viewer_max_points,
                "plotly_traces": viewer_trace_count,
                "trace_chunk_size": viewer_chunk_size,
                "xyz_min": np.round(viewer_cloud.points.min(axis=0), 4).tolist(),
                "xyz_max": np.round(viewer_cloud.points.max(axis=0), 4).tolist(),
            }
        )
        st.plotly_chart(
            _point_cloud_figure(
                viewer_cloud,
                int(viewer_max_points),
                float(viewer_point_size),
                viewer_color_mode,
                viewer_theme,
            ),
            use_container_width=True,
            config={"scrollZoom": True, "displaylogo": False, "responsive": True},
        )

with tabs[2]:
    config["ground_plane"]["distance_threshold"] = st.slider(
        "Distance threshold (m)", 0.005, 0.100, float(config["ground_plane"].get("distance_threshold", 0.025)), 0.005
    )
    config["ground_plane"]["max_iterations"] = st.slider(
        "Max iterations", 100, 3000, int(config["ground_plane"].get("max_iterations", 800)), 100
    )
    cloud = None
    recon = st.session_state.get("reconstruction")
    if recon:
        cloud = recon.point_cloud
    if cloud is not None and st.button("Fit ground plane"):
        st.session_state.plane = fit_ground_plane(
            cloud,
            float(config["ground_plane"]["distance_threshold"]),
            int(config["ground_plane"]["max_iterations"]),
        )
    plane = st.session_state.get("plane")
    if plane:
        st.write({"plane": plane.coefficients.tolist(), "ground": plane.ground_count, "non_ground": plane.non_ground_count})

with tabs[3]:
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except Exception:
        streamlit_image_coordinates = None

    if "alignment_control_points" not in st.session_state:
        st.session_state.alignment_control_points = list(config["alignment"].get("control_points", []))

    st.subheader("Planar Control Points")
    alignment_preview_source = st.radio(
        "Alignment preview source",
        ["Point Cloud", "Ground Texture"],
        horizontal=True,
        key="alignment_preview_source",
    )

    if alignment_preview_source == "Ground Texture":
        st.info(
            "Click the IPM texture to stage a correspondence. The clicked pixel is a map-frame target point; "
            "the source point is recovered by inverting the current run_summary source-to-map transform."
        )
        ground_texture_config = config.setdefault("ground_texture", {})
        alignment_run_summary_path = st.text_input(
            "Run summary for IPM",
            value=_default_run_summary_path(config["export"].get("output_dir", "outputs/track_map")),
            key="alignment_ground_texture_run_summary",
        )
        ground_texture_config["enabled"] = st.checkbox(
            "Use alignment IPM texture as downstream BEV source",
            value=bool(ground_texture_config.get("enabled", True)),
            help="The full export pipeline uses this same VGGT camera-guided IPM texture instead of the point-cloud raster.",
        )
        with st.expander("IPM texture settings", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            fusion_options = ["weighted_mean", "best_view"]
            current_fusion_mode = str(ground_texture_config.get("fusion_mode", "weighted_mean"))
            if current_fusion_mode not in fusion_options:
                current_fusion_mode = "weighted_mean"
            ground_texture_config["fusion_mode"] = c1.selectbox(
                "Fusion mode",
                fusion_options,
                index=fusion_options.index(current_fusion_mode),
                key="alignment_texture_fusion_mode",
            )
            ground_texture_config["confidence_scale"] = c2.number_input(
                "Confidence scale",
                min_value=0.001,
                value=float(
                    ground_texture_config.get(
                        "confidence_scale",
                        config["reconstruction"].get("vggt", {}).get("confidence_threshold", 1.0),
                    )
                    or config["reconstruction"].get("vggt", {}).get("confidence_threshold", 1.0)
                ),
                step=0.1,
                key="alignment_texture_confidence_scale",
            )
            ground_texture_config["view_angle_power"] = c3.slider(
                "View-angle power",
                0.0,
                4.0,
                float(ground_texture_config.get("view_angle_power", 1.5)),
                0.1,
                key="alignment_texture_view_angle_power",
            )
            ground_texture_config["distance_power"] = c4.slider(
                "Distance power",
                0.0,
                4.0,
                float(ground_texture_config.get("distance_power", 1.0)),
                0.1,
                key="alignment_texture_distance_power",
            )
            c1, c2 = st.columns(2)
            ground_texture_config["border_margin_px"] = c1.slider(
                "Image border margin (px)",
                0.0,
                32.0,
                float(ground_texture_config.get("border_margin_px", 4.0)),
                1.0,
                key="alignment_texture_border_margin_px",
            )
            ground_texture_config["inpaint_radius"] = c2.slider(
                "Small-hole inpaint radius",
                0,
                12,
                int(ground_texture_config.get("inpaint_radius", 3)),
                1,
                key="alignment_texture_inpaint_radius",
            )
        c1, c2 = st.columns(2)
        alignment_texture_resolution = c1.slider(
            "Alignment IPM resolution (m/pixel)",
            0.001,
            0.050,
            float(ground_texture_config.get("alignment_preview_resolution", 0.001)),
            0.001,
        )
        texture_display_width = c2.slider("Texture display width", 800, 2400, 1600, 100)
        ground_texture_config["alignment_preview_resolution"] = float(alignment_texture_resolution)
        alignment_texture_output_dir = _default_alignment_ground_texture_output_dir(alignment_run_summary_path)
        if Path(alignment_run_summary_path).exists():
            unknown_rgb = tuple(int(x) for x in config["bev"].get("unknown_rgb", [80, 80, 80]))
            signature = {
                "run_summary_path": str(alignment_run_summary_path),
                "run_summary_mtime": Path(alignment_run_summary_path).stat().st_mtime,
                "output_dir": str(alignment_texture_output_dir),
                "resolution": float(alignment_texture_resolution),
                "fusion_mode": str(ground_texture_config.get("fusion_mode", "weighted_mean")),
                "confidence_scale": ground_texture_config.get("confidence_scale"),
                "min_weight": float(ground_texture_config.get("min_weight", 1e-5)),
                "view_angle_power": float(ground_texture_config.get("view_angle_power", 1.5)),
                "distance_power": float(ground_texture_config.get("distance_power", 1.0)),
                "border_margin_px": float(ground_texture_config.get("border_margin_px", 4.0)),
                "inpaint_radius": int(ground_texture_config.get("inpaint_radius", 3)),
                "unknown_rgb": unknown_rgb,
            }
            force_regenerate = st.button("Regenerate alignment IPM texture")
            preview_state = st.session_state.get("alignment_ground_texture_preview")
            if force_regenerate or preview_state is None or preview_state.get("signature") != signature:
                with st.spinner("Regenerating IPM ground texture for alignment preview..."):
                    texture_result = render_ground_texture_bev(
                        run_summary_path=alignment_run_summary_path,
                        output_dir=alignment_texture_output_dir,
                        resolution=float(alignment_texture_resolution),
                        fusion_mode=signature["fusion_mode"],
                        padding=0.0,
                        confidence_scale=signature["confidence_scale"],
                        min_weight=signature["min_weight"],
                        view_angle_power=signature["view_angle_power"],
                        distance_power=signature["distance_power"],
                        border_margin_px=signature["border_margin_px"],
                        inpaint_radius=signature["inpaint_radius"],
                        unknown_rgb=unknown_rgb,
                    )
                st.session_state.alignment_ground_texture_preview = {
                    "signature": signature,
                    "texture": texture_result.texture,
                    "metadata": texture_result.metadata,
                    "paths": texture_result.paths,
                }
                preview_state = st.session_state.alignment_ground_texture_preview

            alignment_metadata = preview_state["metadata"]
            texture_rgb = preview_state["texture"]
            display_rgb = _resize_rgb_to_width(texture_rgb, int(texture_display_width))
            st.caption(
                f"Regenerated IPM texture {texture_rgb.shape[1]} x {texture_rgb.shape[0]} "
                f"from {alignment_run_summary_path}. Output: {preview_state['paths']['texture']}"
            )
            if streamlit_image_coordinates:
                click = streamlit_image_coordinates(display_rgb, key="alignment_texture_click")
                if click:
                    map_x, map_y = _bev_display_pixel_to_world(float(click["x"]), float(click["y"]), display_rgb, alignment_metadata)
                    try:
                        source_x, source_y = _map_xy_to_alignment_source_xy(alignment_run_summary_path, map_x, map_y)
                        st.session_state.alignment_pending_source = [float(source_x), float(source_y), 0.0]
                        st.session_state.alignment_pending_tx = float(map_x)
                        st.session_state.alignment_pending_ty = float(map_y)
                        st.success(
                            "Staged texture click: "
                            f"source=({source_x:.4f}, {source_y:.4f}, 0), "
                            f"target=({map_x:.4f}, {map_y:.4f}, 0)."
                        )
                    except Exception as exc:
                        st.warning(f"Could not invert current alignment transform for this click: {exc}")
            else:
                st.image(display_rgb, caption="Ground texture in the current map frame", use_container_width=False)
                st.warning("Install streamlit-image-coordinates to click the IPM texture directly.")
        else:
            st.warning(f"Run summary not found: {alignment_run_summary_path}")
    else:
        alignment_path = st.text_input(
            "Ground-aligned source point cloud",
            value=_default_alignment_source_path(config["export"].get("output_dir", "outputs/track_map")),
            key="alignment_source_path",
        )
        c1, c2, c3, c4 = st.columns(4)
        alignment_resolution = c1.slider("Alignment bounds resolution", 0.005, 0.050, 0.015, 0.005)
        alignment_max_points = c2.number_input(
            "BEV displayed points",
            min_value=1000,
            max_value=2_000_000,
            value=300_000,
            step=1000,
        )
        alignment_point_radius = c3.slider("BEV point radius", 0, 8, 2, 1)
        alignment_display_width = c4.slider("BEV display width", 800, 2400, 1600, 100)

        alignment_cloud = None
        alignment_metadata = None
        if Path(alignment_path).exists():
            alignment_cloud = _load_cloud_for_viewer(alignment_path)
            alignment_metadata = _auto_metadata_for_cloud(alignment_cloud, float(alignment_resolution))
            max_bev_points = min(int(alignment_max_points), alignment_cloud.size)
            display_rgb = _render_clickable_bev_hd(
                alignment_cloud,
                alignment_metadata,
                max_bev_points,
                int(alignment_display_width),
                int(alignment_point_radius),
                tuple(config["bev"].get("unknown_rgb", [80, 80, 80])),
            )
            st.caption(
                f"Showing {max_bev_points:,} / {alignment_cloud.size:,} source points. "
                f"HD display {display_rgb.shape[1]} x {display_rgb.shape[0]}."
            )
            if streamlit_image_coordinates:
                click = streamlit_image_coordinates(display_rgb, key="alignment_bev_click")
                if click:
                    x, y = _bev_display_pixel_to_world(float(click["x"]), float(click["y"]), display_rgb, alignment_metadata)
                    st.session_state.alignment_pending_source = [float(x), float(y), 0.0]
            else:
                st.image(display_rgb, caption="Ground-aligned BEV source plane")
                st.warning("Install streamlit-image-coordinates to click the BEV image directly.")
        else:
            st.warning(f"Point cloud file not found: {alignment_path}")

    pending_source = st.session_state.get("alignment_pending_source")
    st.subheader("Add Correspondence")
    if pending_source:
        st.write(
            {
                "clicked_source_x": round(float(pending_source[0]), 4),
                "clicked_source_y": round(float(pending_source[1]), 4),
                "clicked_source_z": 0.0,
            }
        )
        sx, sy = float(pending_source[0]), float(pending_source[1])
        c3, c4 = st.columns(2)
    else:
        c1, c2, c3, c4 = st.columns(4)
        sx = c1.number_input("source x", value=0.0, key="alignment_pending_sx")
        sy = c2.number_input("source y", value=0.0, key="alignment_pending_sy")
    tx = c3.number_input("target x (map)", value=0.0, key="alignment_pending_tx")
    ty = c4.number_input("target y (map)", value=0.0, key="alignment_pending_ty")
    if st.button("Add control point"):
        source = pending_source if pending_source else [float(sx), float(sy), 0.0]
        st.session_state.alignment_control_points.append(
            {"source": [float(source[0]), float(source[1]), 0.0], "target": [float(tx), float(ty), 0.0]}
        )
        st.session_state.alignment_pending_source = None
        st.rerun()

    st.subheader("Control Correspondences")
    st.caption("Source points are in the ground-aligned pre-map coordinate frame; target points are the real-world map coordinates.")
    if st.button("Reset to default raw-source correspondences"):
        st.session_state.alignment_control_points = list(_load_default_config()["alignment"].get("control_points", []))
        config["alignment"]["control_points"] = list(st.session_state.alignment_control_points)
        st.session_state.alignment_pending_source = None
        st.rerun()
    edited_points = []
    for i, point in enumerate(st.session_state.alignment_control_points):
        cols = st.columns([1.2, 1.2, 1.2, 1.2, 0.5])
        source = point["source"]
        target = point["target"]
        cols[0].number_input(f"source x {i}", value=float(source[0]), key=f"align_sx_{i}", disabled=True)
        cols[1].number_input(f"source y {i}", value=float(source[1]), key=f"align_sy_{i}", disabled=True)
        target_x = cols[2].number_input(f"target x {i}", value=float(target[0]), key=f"align_tx_{i}")
        target_y = cols[3].number_input(f"target y {i}", value=float(target[1]), key=f"align_ty_{i}")
        if cols[4].button("Delete", key=f"align_del_{i}"):
            continue
        edited_points.append({"source": [float(source[0]), float(source[1]), 0.0], "target": [target_x, target_y, 0.0]})
    st.session_state.alignment_control_points = edited_points
    config["alignment"]["control_points"] = edited_points

    if st.button("Estimate planar transform", disabled=len(edited_points) < 3):
        res = estimate_sim2(
            np.array([p["source"][:2] for p in edited_points]),
            np.array([p["target"][:2] for p in edited_points]),
            estimate_scale=config.get("alignment", {}).get("mode", "sim2") != "se2",
        )
        st.write({"scale": res.scale, "rms_error": res.rms_error, "transform": res.transform.tolist()})

with tabs[4]:
    roi = config["roi"]
    roi["auto"] = st.checkbox("Auto ROI from reconstructed cloud percentiles", value=bool(roi.get("auto", True)))
    c1, c2, c3 = st.columns(3)
    roi["percentile_low"] = c1.number_input("Percentile low", value=float(roi.get("percentile_low", 1.0)))
    roi["percentile_high"] = c2.number_input("Percentile high", value=float(roi.get("percentile_high", 99.0)))
    roi["padding"] = c3.number_input("Auto padding", value=float(roi.get("padding", 0.15)))
    c1, c2, c3, c4 = st.columns(4)
    roi["x_min"] = c1.number_input("x_min", value=float(roi.get("x_min", 0.0)))
    roi["x_max"] = c2.number_input("x_max", value=float(roi.get("x_max", 5.0)))
    roi["y_min"] = c3.number_input("y_min", value=float(roi.get("y_min", -1.0)))
    roi["y_max"] = c4.number_input("y_max", value=float(roi.get("y_max", 1.0)))
    st.caption("Polygon ROI hook is reserved; rectangular ROI is exported in this implementation.")

with tabs[5]:
    config["bev"]["resolution"] = st.slider("Resolution (m/cell)", 0.005, 0.100, float(config["bev"].get("resolution", 0.02)), 0.005)
    if st.button("Run BEV rasterization preview"):
        try:
            _clear_cached_app_state()
            with st.spinner("Running pipeline in an isolated subprocess..."):
                res = _run_pipeline_subprocess(config)
            st.session_state.last_run = res
            st.image(res["paths"]["bev_rgb"], use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
    last = st.session_state.get("last_run")
    st.subheader("Metric Aligned Map")
    st.caption("Fixed display convention: +x points left, +y points up, 1000 pixels = 1 m. The yellow marker is the world origin (0,0).")
    default_metric_bev_path, default_metric_metadata_path = _latest_metric_render_source_paths(config, last)
    if _looks_like_metric_render_output_path(st.session_state.get("bev_metric_rgb_path", "")):
        st.session_state.bev_metric_rgb_path = default_metric_bev_path
    if not Path(st.session_state.get("bev_metric_rgb_path", default_metric_bev_path)).exists():
        st.session_state.bev_metric_rgb_path = default_metric_bev_path
    if not Path(st.session_state.get("bev_metric_metadata_path", default_metric_metadata_path)).exists():
        st.session_state.bev_metric_metadata_path = default_metric_metadata_path
    metric_bev_path = st.text_input(
        "Aligned BEV source image",
        value=default_metric_bev_path,
        key="bev_metric_rgb_path",
    )
    metric_metadata_path = st.text_input(
        "Aligned source metadata",
        value=default_metric_metadata_path,
        key="bev_metric_metadata_path",
    )
    draw_metric_axes = st.checkbox("Show world origin / 1m axes", value=True, key="bev_metric_axes")
    bev_path = Path(metric_bev_path)
    metadata_path = Path(metric_metadata_path)
    missing = [str(path) for path in [bev_path, metadata_path] if not path.exists()]
    if missing:
        st.warning(f"Missing metric map source files: {missing}. Run Alignment IPM, BEV preview, or full export first.")
    else:
        try:
            metadata = _metadata_from_map_yaml(metadata_path)
            expected_width, expected_height = _metric_view_shape(metadata, METRIC_PIXELS_PER_METER)
            origin_u, origin_v = _world_to_metric_view_pixel(0.0, 0.0, metadata, METRIC_PIXELS_PER_METER)
            origin_inside = 0.0 <= origin_u < expected_width and 0.0 <= origin_v < expected_height
            metric_output_path = Path(config["export"].get("output_dir", "outputs/track_map")) / METRIC_RENDER_FILENAME
            st.write(
                {
                    "pixels_per_meter": METRIC_PIXELS_PER_METER,
                    "metric_image_size": [int(expected_width), int(expected_height)],
                    "metric_megapixels": round(float(expected_width * expected_height) / 1_000_000.0, 2),
                    "metric_render_limit_megapixels": int(METRIC_RENDER_LIMIT_PIXELS / 1_000_000),
                    "world_origin_pixel_xy": [round(float(origin_u), 1), round(float(origin_v), 1)],
                    "world_origin_inside_image": bool(origin_inside),
                    "world_origin_offset_from_lower_right_m": [
                        round(float(0.0 - metadata.x_min), 4),
                        round(float(0.0 - metadata.y_min), 4),
                    ],
                    "lower_right_pixel_xy": [int(expected_width - 1), int(expected_height - 1)],
                    "lower_right_world_xy": [float(metadata.x_min), float(metadata.y_min)],
                    "upper_left_world_xy": [float(metadata.x_max), float(metadata.y_max)],
                    "x_axis": "+x is horizontal left",
                    "y_axis": "+y is vertical up",
                    "source": {"bev_rgb": str(bev_path), "metadata": str(metadata_path)},
                    "rendered_png": str(metric_output_path),
                }
            )
            if st.button("Render metric aligned map", key="render_metric_aligned_map"):
                bev_rgb = _load_rgb_image(str(bev_path), bev_path.stat().st_mtime)
                metric_rgb = _render_metric_aligned_map(
                    bev_rgb,
                    metadata,
                    pixels_per_meter=METRIC_PIXELS_PER_METER,
                    unknown_rgb=tuple(config["bev"].get("unknown_rgb", [80, 80, 80])),
                    draw_axes=draw_metric_axes,
                )
                _save_rgb_image(metric_output_path, metric_rgb)
                st.session_state.bev_metric_render = {
                    "path": str(metric_output_path),
                    "bev_path": str(bev_path),
                    "bev_mtime": float(bev_path.stat().st_mtime),
                    "metadata_path": str(metadata_path),
                    "metadata_mtime": float(metadata_path.stat().st_mtime),
                    "draw_axes": bool(draw_metric_axes),
                    "render_version": METRIC_RENDER_VERSION,
                }
                st.success(f"Rendered metric map: {metric_output_path}")

            render_info = st.session_state.get("bev_metric_render")
            if render_info and Path(render_info.get("path", "")).exists():
                stale = (
                    render_info.get("bev_path") != str(bev_path)
                    or render_info.get("metadata_path") != str(metadata_path)
                    or float(render_info.get("bev_mtime", -1.0)) != float(bev_path.stat().st_mtime)
                    or float(render_info.get("metadata_mtime", -1.0)) != float(metadata_path.stat().st_mtime)
                    or bool(render_info.get("draw_axes", True)) != bool(draw_metric_axes)
                    or render_info.get("render_version") != METRIC_RENDER_VERSION
                )
                if stale:
                    st.warning("The displayed metric map was rendered from older inputs. Press Render metric aligned map to refresh it.")
                else:
                    st.image(render_info["path"], caption="Metric aligned map view", use_container_width=False)
            elif metric_output_path.exists():
                if _is_file_fresh_enough(metric_output_path, [bev_path, metadata_path]):
                    st.image(str(metric_output_path), caption="Metric aligned map view", use_container_width=False)
                else:
                    st.warning("Existing metric map PNG is older than the selected source. Press Render metric aligned map to refresh it.")
            else:
                st.info("Press Render metric aligned map to generate the fixed-scale view.")
        except Exception as exc:
            st.error(str(exc))

with tabs[6]:
    seg = config["segmentation"]
    mode_options = ["white_lane_occupied", "duckietown_color_classes"]
    seg_mode = str(seg.get("mode", "white_lane_occupied"))
    if seg_mode not in mode_options:
        seg_mode = "white_lane_occupied"
    seg["mode"] = st.selectbox("Semantic mode", mode_options, index=mode_options.index(seg_mode))
    seg["white_occupied_unknown_as_free"] = st.checkbox(
        "Treat non-white / unknown as non-occupied",
        value=bool(seg.get("white_occupied_unknown_as_free", True)),
    )
    seg["road_v_max"] = st.slider("Road V max", 0, 255, int(seg.get("road_v_max", 95)))
    seg["road_s_max"] = st.slider("Road S max", 0, 255, int(seg.get("road_s_max", 115)))
    seg["white_v_min"] = st.slider("White V min", 0, 255, int(seg.get("white_v_min", 150)))
    seg["white_s_max"] = st.slider("White S max", 0, 255, int(seg.get("white_s_max", 85)))
    seg["yellow_s_min"] = st.slider("Yellow S min", 0, 255, int(seg.get("yellow_s_min", 60)))
    seg["morphology_open"] = st.slider("Opening radius", 0, 10, int(seg.get("morphology_open", 1)))
    seg["morphology_close"] = st.slider("Closing radius", 0, 15, int(seg.get("morphology_close", 3)))
    last = st.session_state.get("last_run")
    bev_rgb_path = st.text_input("BEV image for live semantic preview", value=_latest_bev_rgb_path(config, last))
    bev_path = Path(bev_rgb_path)
    semantic_signature = None
    if not bev_path.exists():
        st.warning(f"BEV image not found: {bev_rgb_path}. Run BEV preview or full export first.")
    else:
        preview_config = dict(seg)
        preview_config["unknown_rgb"] = list(config["bev"].get("unknown_rgb", [80, 80, 80]))
        semantic_signature = {
            "source": str(bev_path),
            "mtime": float(bev_path.stat().st_mtime),
            "coordinate_convention": "metric_aligned_x_left_y_up" if _is_metric_aligned_bev_path(bev_path, config) else "standard_bev_x_right_y_up",
            "config": preview_config,
        }
        if st.button("Run semantic preview", key="run_semantic_preview"):
            with st.spinner("Computing semantic preview..."):
                bev_rgb = _load_rgb_image(str(bev_path), bev_path.stat().st_mtime)
                semantic = segment_bev_rgb(bev_rgb, preview_config)
                semantic_rgb = colorize_semantic(semantic)
                unique, counts = np.unique(semantic, return_counts=True)
                class_counts = {str(int(cls)): int(count) for cls, count in zip(unique, counts)}
                display_width = min(PREVIEW_DISPLAY_WIDTH, int(bev_rgb.shape[1]))
                st.session_state.semantic_preview = {
                    "signature": semantic_signature,
                    "stats": {
                        "source": str(bev_path),
                        "coordinate_convention": semantic_signature["coordinate_convention"],
                        "semantic_mode": str(seg.get("mode", "white_lane_occupied")),
                        "computed_image_size": [int(bev_rgb.shape[1]), int(bev_rgb.shape[0])],
                        "display_width_px": display_width,
                        "white_lane_occupied_cells": class_counts.get(str(int(SemanticClass.NON_DRIVABLE)), 0),
                        "non_occupied_cells": class_counts.get(str(int(SemanticClass.DRIVABLE_ROAD)), 0),
                        "class_pixel_counts": class_counts,
                    },
                    "display_bev_rgb": _resize_rgb_to_width(bev_rgb, display_width),
                    "display_semantic_rgb": _resize_rgb_to_width(semantic_rgb, display_width),
                }

    semantic_preview = st.session_state.get("semantic_preview")
    if semantic_preview:
        if semantic_signature is not None and semantic_preview.get("signature") != semantic_signature:
            st.warning("Displayed semantic preview is stale. Press Run semantic preview to refresh it.")
        st.write(semantic_preview["stats"])
        cols = st.columns(2)
        cols[0].image(semantic_preview["display_bev_rgb"], caption="Current BEV source", use_container_width=True)
        cols[1].image(semantic_preview["display_semantic_rgb"], caption="Semantic preview", use_container_width=True)
    elif bev_path.exists():
        st.info("Press Run semantic preview to compute the color-filtered semantic map.")

with tabs[7]:
    occ = config["occupancy"]
    occ["non_ground_height_threshold"] = st.slider(
        "Non-ground height threshold (m)", 0.01, 0.30, float(occ.get("non_ground_height_threshold", 0.06)), 0.01
    )
    occ["preview_resolution"] = st.slider(
        "Preview resolution (m/cell)",
        0.001,
        0.050,
        float(occ.get("preview_resolution", 0.01)),
        0.001,
        format="%.3f",
    )
    st.caption("Occupancy preview defaults to 0.01 m/cell for speed. 0.001 m/cell creates ~9M cells and makes obstacle inflation slow.")
    occ["robot_radius"] = st.slider("Robot radius (m)", 0.02, 0.25, float(occ.get("robot_radius", 0.085)), 0.005)
    occ["safety_margin"] = st.slider("Safety margin (m)", 0.0, 0.20, float(occ.get("safety_margin", 0.025)), 0.005)
    occ["gradient_margin_m"] = st.slider(
        "Gradient margin radius (m)",
        0.0,
        1.0,
        float(occ.get("gradient_margin_m", 0.15)),
        0.01,
    )
    occ["remove_isolated_occupied"] = st.checkbox(
        "Remove isolated occupied cells",
        value=bool(occ.get("remove_isolated_occupied", True)),
        help="Remove only single occupied cells whose 8-neighborhood contains no other occupied cell.",
    )
    occ["unknown_as_occupied"] = st.checkbox("Unknown as occupied", value=bool(occ.get("unknown_as_occupied", False)))
    last = st.session_state.get("last_run")
    bev_rgb_path = st.text_input(
        "BEV image for live occupancy semantic source",
        value=_latest_bev_rgb_path(config, last),
        key="occupancy_bev_rgb_path",
    )
    aligned_cloud_path = st.text_input(
        "Aligned point cloud for live obstacle preview",
        value=_latest_output_path(config, last, "aligned_point_cloud", "aligned_point_cloud.ply"),
        key="occupancy_aligned_cloud_path",
    )
    map_metadata_path = st.text_input(
        "Map metadata for live occupancy preview",
        value=_latest_output_path(config, last, "map_metadata", "map_metadata.yaml"),
        key="occupancy_map_metadata_path",
    )
    bev_path = Path(bev_rgb_path)
    cloud_path = Path(aligned_cloud_path)
    metadata_path = Path(map_metadata_path)
    missing = [str(path) for path in [bev_path, cloud_path, metadata_path] if not path.exists()]
    if missing:
        st.warning(f"Missing live occupancy source files: {missing}. Run BEV preview or full export first.")
    else:
        metadata = _metadata_from_map_yaml(metadata_path)
        metric_source = _is_metric_aligned_bev_path(bev_path, config)
        semantic_config = dict(config["segmentation"])
        semantic_config["unknown_rgb"] = list(config["bev"].get("unknown_rgb", [80, 80, 80]))
        occupancy_signature = {
            "bev": str(bev_path),
            "bev_mtime": float(bev_path.stat().st_mtime),
            "cloud": str(cloud_path),
            "cloud_mtime": float(cloud_path.stat().st_mtime),
            "metadata": str(metadata_path),
            "metadata_mtime": float(metadata_path.stat().st_mtime),
            "semantic": semantic_config,
            "occupancy": {
                "non_ground_height_threshold": float(occ.get("non_ground_height_threshold", 0.06)),
                "preview_resolution": float(occ.get("preview_resolution", 0.01)),
                "robot_radius": float(occ.get("robot_radius", 0.085)),
                "safety_margin": float(occ.get("safety_margin", 0.025)),
                "gradient_margin_m": float(occ.get("gradient_margin_m", 0.15)),
                "remove_isolated_occupied": bool(occ.get("remove_isolated_occupied", True)),
                "unknown_as_occupied": bool(occ.get("unknown_as_occupied", False)),
            },
            "metric_source": bool(metric_source),
        }
        if metric_source:
            expected_width, expected_height = _metric_view_shape(metadata, METRIC_PIXELS_PER_METER)
            expected_shape = (expected_height, expected_width)
        else:
            expected_shape = (metadata.height, metadata.width)
        preview_metadata = metadata_from_bounds(
            float(metadata.x_min),
            float(metadata.x_max),
            float(metadata.y_min),
            float(metadata.y_max),
            float(occ.get("preview_resolution", 0.01)),
            str(metadata.frame_id),
        )
        st.write(
            {
                "source_image_size": [int(expected_shape[1]), int(expected_shape[0])],
                "preview_compute_size": [int(preview_metadata.width), int(preview_metadata.height)],
                "coordinate_convention": "metric_aligned_x_left_y_up" if metric_source else "standard_bev_x_right_y_up",
            }
        )
        if st.button("Run occupancy preview", key="run_occupancy_preview"):
            with st.spinner("Computing occupancy preview..."):
                timings: dict[str, float] = {}
                step_t = time.perf_counter()
                bev_rgb = _load_rgb_image(str(bev_path), bev_path.stat().st_mtime)
                timings["load_bev_s"] = round(time.perf_counter() - step_t, 4)
                if bev_rgb.shape[:2] != expected_shape:
                    st.warning(
                        "BEV image size does not match map metadata: "
                        f"image={bev_rgb.shape[1]}x{bev_rgb.shape[0]}, expected={expected_shape[1]}x{expected_shape[0]}."
                    )
                else:
                    preview_resolution = float(occ.get("preview_resolution", 0.01))
                    step_t = time.perf_counter()
                    bev_rgb, preview_metadata = _resample_bev_rgb_to_resolution(bev_rgb, metadata, preview_resolution)
                    timings["resample_bev_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    semantic = segment_bev_rgb(bev_rgb, semantic_config)
                    timings["semantic_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    cloud = _load_cloud_for_viewer(str(cloud_path), cloud_path.stat().st_mtime)
                    timings["load_cloud_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    if metric_source:
                        raw_obstacle = _obstacle_grid_from_metric_view_height(
                            cloud,
                            metadata,
                            bev_rgb.shape[:2],
                            float(occ.get("non_ground_height_threshold", 0.06)),
                            1.0 / preview_resolution,
                        )
                        occupancy_resolution = preview_resolution
                        coordinate_convention = "metric_aligned_x_left_y_up"
                    else:
                        raw_obstacle = _obstacle_grid_from_aligned_height(
                            cloud,
                            preview_metadata,
                            float(occ.get("non_ground_height_threshold", 0.06)),
                        )
                        occupancy_resolution = float(preview_metadata.resolution)
                        coordinate_convention = "standard_bev_x_right_y_up"
                    timings["obstacle_projection_s"] = round(time.perf_counter() - step_t, 4)
                    inflation_radius = float(occ.get("robot_radius", 0.085)) + float(occ.get("safety_margin", 0.025))
                    step_t = time.perf_counter()
                    inflated_obstacle = inflate_obstacles(raw_obstacle, inflation_radius, occupancy_resolution)
                    timings["inflate_obstacles_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    occupancy_grid = fuse_occupancy(
                        semantic,
                        inflated_obstacle,
                        unknown_as_occupied=bool(occ.get("unknown_as_occupied", False)),
                    )
                    timings["fuse_occupancy_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    isolated_removed_cells = 0
                    if bool(occ.get("remove_isolated_occupied", True)):
                        occupancy_grid, isolated_removed_cells = remove_isolated_occupied_cells(occupancy_grid)
                    timings["remove_isolated_occupied_s"] = round(time.perf_counter() - step_t, 4)
                    step_t = time.perf_counter()
                    gradient_margin_m = float(occ.get("gradient_margin_m", 0.15))
                    margin_layer = gradient_margin_from_occupancy(occupancy_grid, gradient_margin_m, occupancy_resolution)
                    timings["gradient_margin_s"] = round(time.perf_counter() - step_t, 4)
                    display_width = min(PREVIEW_DISPLAY_WIDTH, int(bev_rgb.shape[1]))
                    step_t = time.perf_counter()
                    raw_obstacle_preview = _resize_rgb_to_width(raw_obstacle.astype(np.uint8) * 255, display_width)
                    inflated_obstacle_preview = _resize_rgb_to_width(inflated_obstacle.astype(np.uint8) * 255, display_width)
                    final_occupancy_preview = _resize_rgb_to_width(ros_image_from_occupancy(occupancy_grid), display_width)
                    margin_preview = _resize_rgb_to_width(ros_image_from_occupancy_margin(occupancy_grid, margin_layer), display_width)
                    timings["preview_images_s"] = round(time.perf_counter() - step_t, 4)
                    timings["total_s"] = round(sum(timings.values()), 4)
                    st.session_state.occupancy_preview = {
                        "signature": occupancy_signature,
                        "stats": {
                            "source": {
                                "bev_rgb": str(bev_path),
                                "aligned_cloud": str(cloud_path),
                                "map_metadata": str(metadata_path),
                                "coordinate_convention": coordinate_convention,
                            },
                            "computed_image_size": [int(bev_rgb.shape[1]), int(bev_rgb.shape[0])],
                            "display_width_px": display_width,
                            "occupancy_resolution_m_per_cell": float(occupancy_resolution),
                            "timings_s": timings,
                            "raw_obstacle_cells": int(np.count_nonzero(raw_obstacle)),
                            "inflated_obstacle_cells": int(np.count_nonzero(inflated_obstacle)),
                            "inflation_radius_m": float(inflation_radius),
                            "remove_isolated_occupied": bool(occ.get("remove_isolated_occupied", True)),
                            "isolated_occupied_removed_cells": int(isolated_removed_cells),
                            "gradient_margin_m": float(gradient_margin_m),
                            "gradient_margin_nonzero_cells": int(np.count_nonzero(margin_layer > 0.0)),
                            "free_cells": int(np.count_nonzero(occupancy_grid == 0)),
                            "occupied_cells": int(np.count_nonzero(occupancy_grid == 100)),
                            "unknown_cells": int(np.count_nonzero(occupancy_grid == -1)),
                        },
                        "raw_obstacle_rgb": raw_obstacle_preview,
                        "inflated_obstacle_rgb": inflated_obstacle_preview,
                        "final_occupancy_rgb": final_occupancy_preview,
                        "gradient_margin_rgb": margin_preview,
                    }

        occupancy_preview = st.session_state.get("occupancy_preview")
        if occupancy_preview:
            if occupancy_preview.get("signature") != occupancy_signature:
                st.warning("Displayed occupancy preview is stale. Press Run occupancy preview to refresh it.")
            st.write(occupancy_preview["stats"])
            cols = st.columns(4)
            cols[0].image(occupancy_preview["raw_obstacle_rgb"], caption="Raw height obstacles", use_container_width=True)
            cols[1].image(
                occupancy_preview["inflated_obstacle_rgb"],
                caption="Inflated obstacles",
                use_container_width=True,
            )
            cols[2].image(
                occupancy_preview["final_occupancy_rgb"],
                caption="Hard occupancy",
                use_container_width=True,
            )
            cols[3].image(
                occupancy_preview["gradient_margin_rgb"],
                caption="Gradient margin cost",
                use_container_width=True,
            )
        else:
            st.info("Press Run occupancy preview to compute the occupancy map.")

with tabs[8]:
    st.header("Output Map In World Coordinates")
    output_dir = Path(config["export"].get("output_dir", "outputs/track_map"))
    default_map_yaml = output_dir / "map.yaml"
    default_map_metadata = output_dir / "map_metadata.yaml"
    default_occupancy_grid = output_dir / "occupancy_grid.npy"
    map_yaml_path = st.text_input("ROS map YAML", value=str(default_map_yaml), key="world_map_yaml_path")
    map_metadata_path = st.text_input("Map metadata", value=str(default_map_metadata), key="world_map_metadata_path")
    occupancy_grid_path = st.text_input("Occupancy grid for margin preview", value=str(default_occupancy_grid), key="world_map_occupancy_grid_path")

    c1, c2, c3, c4 = st.columns(4)
    world_view_height = c1.slider("Viewer height", 520, 1100, 760, 20)
    world_grid_spacing = c2.slider("Grid spacing (m)", 0.05, 1.0, 0.25, 0.05)
    world_padding = c3.slider("Padding (m)", 0.0, 2.0, 0.20, 0.05)
    show_world_axes = c4.checkbox("Show world axes", value=True)
    c1, c2 = st.columns(2)
    config["occupancy"]["gradient_margin_m"] = c1.slider(
        "Gradient margin radius (m)",
        0.0,
        1.0,
        float(config["occupancy"].get("gradient_margin_m", 0.15)),
        0.01,
        key="world_gradient_margin_m",
    )
    show_margin_preview = c2.checkbox("Preview gradient margin from occupancy grid", value=True)
    config["occupancy"]["remove_isolated_occupied"] = st.checkbox(
        "Remove isolated occupied cells before margin",
        value=bool(config["occupancy"].get("remove_isolated_occupied", True)),
        key="world_remove_isolated_occupied",
        help="Remove only single occupied cells whose 8-neighborhood contains no other occupied cell.",
    )
    show_world_control = st.checkbox("Show alignment target control points", value=True)

    map_yaml = Path(map_yaml_path)
    if not map_yaml.exists():
        st.warning(f"Map YAML not found: {map_yaml_path}. Run full export first.")
    else:
        try:
            map_info = _load_ros_map_for_world_view(map_yaml)
            margin_preview_stats = None
            occupancy_grid_path_obj = Path(occupancy_grid_path)
            if show_margin_preview:
                if occupancy_grid_path_obj.exists():
                    occupancy_grid = _load_numpy_array(str(occupancy_grid_path_obj), occupancy_grid_path_obj.stat().st_mtime)
                    expected_shape = (int(map_info["size"][1]), int(map_info["size"][0]))
                    if occupancy_grid.shape[:2] == expected_shape:
                        isolated_removed_cells = 0
                        if bool(config["occupancy"].get("remove_isolated_occupied", True)):
                            occupancy_grid, isolated_removed_cells = remove_isolated_occupied_cells(occupancy_grid)
                        margin_layer = gradient_margin_from_occupancy(
                            occupancy_grid,
                            float(config["occupancy"].get("gradient_margin_m", 0.15)),
                            float(map_info["resolution"]),
                        )
                        map_info = dict(map_info)
                        map_info["image"] = np.stack(
                            [ros_image_from_occupancy_margin(occupancy_grid, margin_layer)] * 3,
                            axis=-1,
                        )
                        map_info["image_path"] = f"{occupancy_grid_path_obj} + gradient margin preview"
                        margin_preview_stats = {
                            "margin_radius_m": float(config["occupancy"].get("gradient_margin_m", 0.15)),
                            "remove_isolated_occupied": bool(config["occupancy"].get("remove_isolated_occupied", True)),
                            "isolated_occupied_removed_cells": int(isolated_removed_cells),
                            "margin_nonzero_cells": int(np.count_nonzero(margin_layer > 0.0)),
                            "margin_max": float(np.max(margin_layer)) if margin_layer.size else 0.0,
                            "value_convention": "1 at occupied cells, linearly falling to 0 at margin radius",
                        }
                    else:
                        st.warning(
                            "Occupancy grid shape does not match map image size: "
                            f"grid={occupancy_grid.shape[:2]}, map={expected_shape}."
                        )
                else:
                    st.warning(f"Occupancy grid not found: {occupancy_grid_path}")
            control_points = _control_points_for_world_view(map_metadata_path, config)
            x_min, x_max, y_min, y_max = [float(v) for v in map_info["bounds"]]
            target_checks = []
            for i, point in enumerate(control_points):
                target = point.get("target", [])
                if len(target) >= 2:
                    tx, ty = float(target[0]), float(target[1])
                    target_checks.append(
                        {
                            "point": i,
                            "target_xy": [round(tx, 4), round(ty, 4)],
                            "inside_map_bounds": bool(x_min <= tx <= x_max and y_min <= ty <= y_max),
                        }
                    )
            st.write(
                {
                    "map_yaml": str(map_yaml),
                    "image": map_info["image_path"],
                    "resolution_m_per_pixel": float(map_info["resolution"]),
                    "image_size_px": map_info["size"],
                    "world_bounds_m": {
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                    },
                    "origin_lower_left_xy_yaw": map_info["origin"],
                    "map_yaml_mode": map_info["metadata"].get("mode", "trinary"),
                    "gradient_margin_preview": margin_preview_stats,
                    "display_convention": "ROS/map view: +x right, +y up",
                    "alignment_targets": target_checks,
                }
            )
            fig = _world_map_figure(
                map_info,
                control_points,
                int(world_view_height),
                float(world_grid_spacing),
                float(world_padding),
                bool(show_world_axes),
                bool(show_world_control),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

with tabs[9]:
    config["export"]["output_dir"] = st.text_input("Output directory", value=config["export"].get("output_dir", "outputs/track_map"))
    config["export"]["map_frame"] = st.text_input("Map frame", value=config["export"].get("map_frame", "map"))
    config["occupancy"]["gradient_margin_m"] = st.slider(
        "Export gradient margin radius (m)",
        0.0,
        1.0,
        float(config["occupancy"].get("gradient_margin_m", 0.15)),
        0.01,
        key="export_gradient_margin_m",
    )
    config["occupancy"]["remove_isolated_occupied"] = st.checkbox(
        "Remove isolated occupied cells before export",
        value=bool(config["occupancy"].get("remove_isolated_occupied", True)),
        key="export_remove_isolated_occupied",
        help="Remove only single occupied cells whose 8-neighborhood contains no other occupied cell.",
    )
    st.caption("Full export writes map.yaml/map.png with this gradient margin, plus map_hard.yaml/map_hard.png for the hard occupancy map.")
    config_file = st.file_uploader("Optional YAML override", type=["yaml", "yml"])
    if config_file:
        override = yaml.safe_load(config_file.getvalue().decode("utf-8")) or {}
        st.session_state.config = deep_update(config, override)
        st.rerun()
    if st.button("Run full export"):
        try:
            _clear_cached_app_state()
            with st.spinner("Running pipeline in an isolated subprocess..."):
                res = _run_pipeline_subprocess(config)
            st.session_state.last_run = res
            st.success(f"Exported to {config['export']['output_dir']}")
            st.write(res["paths"])
        except Exception as exc:
            st.error(str(exc))
