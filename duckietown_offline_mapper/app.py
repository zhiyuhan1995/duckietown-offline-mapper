from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import streamlit as st
import yaml

from src.alignment import estimate_sim2
from src.bev import metadata_from_bounds
from src.io_utils import deep_update, load_yaml
from src.keyframes import extract_keyframes, load_image_folder
from src.plane import fit_ground_plane
from src.pointcloud import PointCloud, load_ply


st.set_page_config(page_title="Duckietown Offline Mapper", layout="wide")
st.title("Duckietown Offline Semantic-Occupancy BEV Mapper")


def _load_default_config() -> dict:
    return load_yaml(Path(__file__).resolve().parent / "configs" / "default.yaml")


if "config" not in st.session_state:
    st.session_state.config = _load_default_config()

config = st.session_state.config


def _image_rgb_from_bgr(frame):
    return frame[..., ::-1]


def _clear_cached_app_state() -> None:
    import gc

    for key in ["reconstruction", "reconstruction_cloud"]:
        if key in st.session_state:
            del st.session_state[key]
    _load_cloud_for_viewer.clear()
    gc.collect()


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
def _load_cloud_for_viewer(path: str) -> PointCloud:
    return load_ply(path)


def _default_viewer_path(export_dir: str) -> str:
    candidates = [
        Path("outputs/track_map_cluster_gpu01_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_cluster_gpu01_edge_complete/work/vggt_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/work/vggt_point_cloud.ply"),
        Path(export_dir) / "aligned_point_cloud.ply",
        Path(export_dir) / "work" / "vggt_point_cloud.ply",
        Path("outputs/track_map_local/aligned_point_cloud.ply"),
        Path("outputs/track_map/aligned_point_cloud.ply"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _default_alignment_source_path(export_dir: str) -> str:
    candidates = [
        Path("outputs/track_map_cluster_gpu01_edge_complete/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map_cluster_gpu01_edge_complete/aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map_edge_complete/aligned_point_cloud.ply"),
        Path(export_dir) / "ground_aligned_point_cloud.ply",
        Path(export_dir) / "aligned_point_cloud.ply",
        Path("outputs/track_map/ground_aligned_point_cloud.ply"),
        Path("outputs/track_map/aligned_point_cloud.ply"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


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
        tmp = NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()
        config["input"]["path"] = tmp.name
    else:
        config["input"]["path"] = folder

    if st.button("Preview keyframes"):
        path = Path(config["input"]["path"])
        frames = load_image_folder(path, int(config["input"]["max_keyframes"])) if path.is_dir() else extract_keyframes(
            path,
            int(config["input"]["keyframe_interval"]),
            int(config["input"]["max_keyframes"]),
        )
        st.session_state.frames = frames
    frames = st.session_state.get("frames", [])
    if frames:
        cols = st.columns(min(4, len(frames)))
        for i, frame in enumerate(frames[:8]):
            cols[i % len(cols)].image(_image_rgb_from_bgr(frame.image_bgr), caption=f"frame {frame.index}", use_container_width=True)

with tabs[1]:
    config["reconstruction"]["backend"] = "vggt_sfm"
    st.info("Reconstruction backend: VGGT-SfM only")
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
            _clear_cached_app_state()
            with st.spinner("Running VGGT-SfM in an isolated subprocess..."):
                res = _run_pipeline_subprocess(config)
            st.session_state.last_run = res
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
    alignment_path = st.text_input(
        "Ground-aligned source point cloud",
        value=_default_alignment_source_path(config["export"].get("output_dir", "outputs/track_map")),
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

with tabs[6]:
    seg = config["segmentation"]
    seg["road_v_max"] = st.slider("Road V max", 0, 255, int(seg.get("road_v_max", 95)))
    seg["road_s_max"] = st.slider("Road S max", 0, 255, int(seg.get("road_s_max", 115)))
    seg["white_v_min"] = st.slider("White V min", 0, 255, int(seg.get("white_v_min", 150)))
    seg["yellow_s_min"] = st.slider("Yellow S min", 0, 255, int(seg.get("yellow_s_min", 60)))
    seg["morphology_open"] = st.slider("Opening radius", 0, 10, int(seg.get("morphology_open", 1)))
    seg["morphology_close"] = st.slider("Closing radius", 0, 15, int(seg.get("morphology_close", 3)))
    last = st.session_state.get("last_run")
    if last:
        st.image(last["paths"]["semantic_mask"], use_container_width=True)

with tabs[7]:
    occ = config["occupancy"]
    occ["non_ground_height_threshold"] = st.slider(
        "Non-ground height threshold (m)", 0.01, 0.30, float(occ.get("non_ground_height_threshold", 0.06)), 0.01
    )
    occ["robot_radius"] = st.slider("Robot radius (m)", 0.02, 0.25, float(occ.get("robot_radius", 0.085)), 0.005)
    occ["safety_margin"] = st.slider("Safety margin (m)", 0.0, 0.20, float(occ.get("safety_margin", 0.025)), 0.005)
    occ["unknown_as_occupied"] = st.checkbox("Unknown as occupied", value=bool(occ.get("unknown_as_occupied", False)))
    last = st.session_state.get("last_run")
    if last:
        st.write(last["stats"])
        st.image(last["paths"]["final_occupancy_grid"], use_container_width=True)

with tabs[8]:
    config["export"]["output_dir"] = st.text_input("Output directory", value=config["export"].get("output_dir", "outputs/track_map"))
    config["export"]["map_frame"] = st.text_input("Map frame", value=config["export"].get("map_frame", "map"))
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
