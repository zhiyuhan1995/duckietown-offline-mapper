from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .keyframes import Keyframe, extract_keyframes, load_image_folder, save_keyframe_previews
from .pointcloud import PointCloud, save_ply


@dataclass
class ReconstructionResult:
    point_cloud: PointCloud
    camera_poses: list[np.ndarray]
    intrinsics: np.ndarray
    metadata: dict[str, Any]


class ReconstructionBackend(ABC):
    @abstractmethod
    def run(self, input_path: str | Path, output_dir: str | Path) -> ReconstructionResult:
        """Run reconstruction and return global geometry."""


class VGGTDependencyError(RuntimeError):
    """Raised when the required VGGT-SfM runtime is not installed."""


class VGGT_SfMReconstructionBackend(ReconstructionBackend):
    """VGGT-based global geometry backend with optional COLMAP/SfM refinement.

    This is the only reconstruction route in the project. It follows the
    official VGGT API: preprocess images, run camera/depth heads, unproject
    depth into world coordinates, and fuse confident pixels into a global
    colored point cloud. Optional COLMAP export / bundle adjustment stays inside
    this backend and requires VGGT demo dependencies.
    """

    def __init__(
        self,
        keyframe_interval: int = 30,
        max_keyframes: int | None = 40,
        model_id: str = "facebook/VGGT-1B",
        checkpoint_url: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        preprocess_mode: str = "crop",
        use_point_map: bool = False,
        confidence_threshold: float = 5.0,
        relax_ground_confidence: bool = True,
        ground_confidence_threshold: float = 1.2,
        max_points: int = 250_000,
        sample_stride: int = 2,
        seed: int = 7,
        save_depth: bool = False,
        save_colmap: bool = True,
        bundle_adjustment: bool = False,
        ba_max_query_points: int = 4096,
        ba_query_frame_num: int = 8,
        ba_vis_threshold: float = 0.2,
        ba_max_reproj_error: float = 8.0,
    ):
        self.keyframe_interval = keyframe_interval
        self.max_keyframes = max_keyframes
        self.model_id = model_id
        self.checkpoint_url = checkpoint_url
        self.device = device
        self.dtype = dtype
        self.preprocess_mode = preprocess_mode
        self.use_point_map = use_point_map
        self.confidence_threshold = confidence_threshold
        self.relax_ground_confidence = relax_ground_confidence
        self.ground_confidence_threshold = ground_confidence_threshold
        self.max_points = max_points
        self.sample_stride = max(1, sample_stride)
        self.seed = seed
        self.save_depth = save_depth
        self.save_colmap = save_colmap
        self.bundle_adjustment = bundle_adjustment
        self.ba_max_query_points = ba_max_query_points
        self.ba_query_frame_num = ba_query_frame_num
        self.ba_vis_threshold = ba_vis_threshold
        self.ba_max_reproj_error = ba_max_reproj_error

    def run(self, input_path: str | Path, output_dir: str | Path) -> ReconstructionResult:
        deps = _load_vggt_dependencies(require_ba=self.bundle_adjustment)
        torch = deps["torch"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self._load_frames(Path(input_path))
        image_paths = self._write_vggt_images(frames, output_dir / "vggt_images")
        save_keyframe_previews(frames, output_dir / "keyframes", limit=12)

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)
        model = None
        images = None
        predictions = None
        try:
            model = self._load_model(deps["VGGT"], torch, device)
            effective_preprocess_mode = "pad" if self.bundle_adjustment else self.preprocess_mode
            images = deps["load_and_preprocess_images"](image_paths, mode=effective_preprocess_mode).to(device)
            predictions = self._run_vggt(model, images, deps, torch, dtype)

            cloud = self._fuse_predictions_to_cloud(predictions, images, torch)
            save_ply(cloud, output_dir / "vggt_point_cloud.ply")
            self._save_prediction_arrays(predictions, output_dir)

            colmap_metadata: dict[str, Any] = {}
            if self.save_colmap:
                colmap_metadata = self._export_colmap_if_available(
                    predictions, images, image_paths, deps, torch, dtype, output_dir
                )

            metadata = {
                "backend": "vggt_sfm",
                "model_id": self.model_id,
                "checkpoint_url": self.checkpoint_url,
                "device": str(device),
                "dtype": str(dtype).replace("torch.", ""),
                "keyframe_count": len(frames),
                "keyframe_interval": self.keyframe_interval,
                "max_keyframes": self.max_keyframes,
                "preprocess_mode": self.preprocess_mode,
                "effective_preprocess_mode": effective_preprocess_mode,
                "point_source": "point_map" if self.use_point_map else "depth_unprojection",
                "confidence_threshold": self.confidence_threshold,
                "relax_ground_confidence": self.relax_ground_confidence,
                "ground_confidence_threshold": self.ground_confidence_threshold,
                "max_points": self.max_points,
                "sample_stride": self.sample_stride,
                "save_depth": self.save_depth,
                "save_colmap": self.save_colmap,
                "bundle_adjustment": self.bundle_adjustment,
                "image_paths": [str(p) for p in image_paths],
                "prediction_files": {
                    "camera_extrinsics": str(output_dir / "camera_extrinsics.npy"),
                    "camera_intrinsics": str(output_dir / "camera_intrinsics.npy"),
                    "point_cloud": str(output_dir / "vggt_point_cloud.ply"),
                },
                "colmap": colmap_metadata,
            }
            camera_poses = _extrinsics_to_camera_poses(predictions["extrinsic"])
            intrinsics = predictions["intrinsic"].copy()
        finally:
            del model, images, predictions
            _cleanup_cuda(torch, device)
        return ReconstructionResult(
            point_cloud=cloud,
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            metadata=metadata,
        )

    def _load_frames(self, input_path: Path) -> list[Keyframe]:
        if input_path.is_dir():
            return load_image_folder(input_path, max_images=self.max_keyframes)
        return extract_keyframes(input_path, interval=self.keyframe_interval, max_keyframes=self.max_keyframes)

    @staticmethod
    def _write_vggt_images(frames: list[Keyframe], output_dir: Path) -> list[Path]:
        cv2 = _require_cv2()
        _reset_generated_dir(output_dir)
        paths: list[Path] = []
        for i, frame in enumerate(frames):
            path = output_dir / f"frame_{i:04d}_{frame.index:06d}.png"
            cv2.imwrite(str(path), frame.image_bgr)
            paths.append(path)
        return paths

    def _resolve_device(self, torch: Any) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        if self.dtype == "float32":
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        if device == "cuda":
            return torch.float16
        return torch.float32

    def _load_model(self, model_cls: Any, torch: Any, device: str) -> Any:
        try:
            if self.checkpoint_url:
                model = model_cls()
                state = torch.hub.load_state_dict_from_url(self.checkpoint_url, map_location="cpu")
                model.load_state_dict(state)
            else:
                model = model_cls.from_pretrained(self.model_id)
        except Exception as exc:
            raise VGGTDependencyError(
                "Could not load VGGT weights. Check Hugging Face access, network, or set "
                "`reconstruction.vggt.checkpoint_url` to a local/accessible checkpoint URL."
            ) from exc
        model.eval()
        return model.to(device)

    def _run_vggt(self, model: Any, images: Any, deps: dict[str, Any], torch: Any, dtype: Any) -> dict[str, np.ndarray]:
        autocast = _autocast_context(torch, images.device.type, dtype)
        with torch.no_grad():
            with autocast:
                batched = images[None]
                aggregated_tokens_list, ps_idx = model.aggregator(batched)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = deps["pose_encoding_to_extri_intri"](pose_enc, batched.shape[-2:])
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched, ps_idx)
                point_map = None
                point_conf = None
                if self.use_point_map:
                    point_map, point_conf = model.point_head(aggregated_tokens_list, batched, ps_idx)

        extrinsic_np = _as_numpy(extrinsic.squeeze(0))
        intrinsic_np = _as_numpy(intrinsic.squeeze(0))
        depth_np = _as_numpy(depth_map.squeeze(0))
        depth_conf_np = _as_numpy(depth_conf.squeeze(0))
        if self.use_point_map and point_map is not None:
            points_np = _as_numpy(point_map.squeeze(0))
            point_conf_np = _as_numpy(point_conf.squeeze(0)) if point_conf is not None else depth_conf_np
        else:
            points_np = deps["unproject_depth_map_to_point_map"](depth_np, extrinsic_np, intrinsic_np)
            point_conf_np = depth_conf_np

        return {
            "extrinsic": extrinsic_np,
            "intrinsic": intrinsic_np,
            "depth": depth_np,
            "depth_conf": depth_conf_np,
            "points_3d": points_np,
            "point_conf": point_conf_np,
        }

    def _fuse_predictions_to_cloud(self, predictions: dict[str, np.ndarray], images: Any, torch: Any) -> PointCloud:
        points_3d = predictions["points_3d"]
        point_conf = _squeeze_confidence(predictions["point_conf"])
        colors = _resized_image_colors(images, points_3d.shape[1:3], torch)
        mask = _vggt_keep_mask(
            points_3d,
            point_conf,
            colors,
            confidence_threshold=self.confidence_threshold,
            relax_ground_confidence=self.relax_ground_confidence,
            ground_confidence_threshold=self.ground_confidence_threshold,
            sample_stride=self.sample_stride,
        )
        if not np.any(mask):
            raise RuntimeError(
                "VGGT produced no points after confidence filtering. "
                "Lower `reconstruction.vggt.confidence_threshold` or check input keyframes."
            )

        flat_points = points_3d[mask].astype(np.float64)
        flat_colors = colors[mask].astype(np.uint8)
        flat_conf = point_conf[mask].astype(np.float64)
        if self.max_points and len(flat_points) > self.max_points:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(flat_points), size=self.max_points, replace=False)
            flat_points = flat_points[idx]
            flat_colors = flat_colors[idx]
            flat_conf = flat_conf[idx]
        return PointCloud(flat_points, flat_colors, flat_conf)

    def _save_prediction_arrays(self, predictions: dict[str, np.ndarray], output_dir: Path) -> None:
        np.save(output_dir / "camera_extrinsics.npy", predictions["extrinsic"])
        np.save(output_dir / "camera_intrinsics.npy", predictions["intrinsic"])
        np.save(output_dir / "point_confidence.npy", predictions["point_conf"])
        if self.save_depth:
            np.save(output_dir / "depth_maps.npy", predictions["depth"])
            np.save(output_dir / "depth_confidence.npy", predictions["depth_conf"])

    def _export_colmap_if_available(
        self,
        predictions: dict[str, np.ndarray],
        images: Any,
        image_paths: list[Path],
        deps: dict[str, Any],
        torch: Any,
        dtype: Any,
        output_dir: Path,
    ) -> dict[str, Any]:
        if self.bundle_adjustment:
            return self._export_colmap_with_ba(predictions, images, image_paths, deps, torch, dtype, output_dir)
        return self._export_colmap_without_ba(predictions, images, image_paths, deps, torch, output_dir)

    def _export_colmap_without_ba(
        self,
        predictions: dict[str, np.ndarray],
        images: Any,
        image_paths: list[Path],
        deps: dict[str, Any],
        torch: Any,
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            import torch.nn.functional as F  # type: ignore
            from vggt.utils.helper import create_pixel_coordinate_grid  # type: ignore
            from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track  # type: ignore
        except Exception as exc:
            raise VGGTDependencyError(
                "COLMAP export requires VGGT demo dependencies including pycolmap. "
                "Install the official VGGT demo requirements or set `save_colmap: false`."
            ) from exc

        points_3d = predictions["points_3d"]
        point_conf = _squeeze_confidence(predictions["point_conf"])
        num_frames, height, width, _ = points_3d.shape
        points_rgb = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        points_rgb = (_as_numpy(points_rgb) * 255.0).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
        keep_mask = _vggt_keep_mask(
            points_3d,
            point_conf,
            points_rgb,
            confidence_threshold=self.confidence_threshold,
            relax_ground_confidence=self.relax_ground_confidence,
            ground_confidence_threshold=self.ground_confidence_threshold,
            sample_stride=1,
        )
        selected_mask = _limit_mask(keep_mask, self.max_points, self.seed)
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d[selected_mask],
            points_xyf[selected_mask],
            points_rgb[selected_mask],
            predictions["extrinsic"],
            predictions["intrinsic"],
            np.array([height, width]),
            shared_camera=False,
            camera_type="PINHOLE",
        )
        self._assign_colmap_image_names(reconstruction, image_paths)
        sparse_dir = output_dir / "colmap_sparse"
        _reset_generated_dir(sparse_dir)
        reconstruction.write(str(sparse_dir))
        gsplat_scene_dir = self._prepare_gsplat_scene(output_dir, sparse_dir, image_paths)
        return {
            "mode": "direct_vggt_depth",
            "bundle_adjustment": False,
            "sparse_dir": str(sparse_dir),
            "gsplat_scene_dir": str(gsplat_scene_dir),
            "image_names": [p.name for p in image_paths],
            "selected_points": int(np.count_nonzero(selected_mask)),
            "candidate_points": int(np.count_nonzero(keep_mask)),
            "uses_relaxed_ground_mask": bool(self.relax_ground_confidence),
            "ground_confidence_threshold": float(self.ground_confidence_threshold),
        }

    def _export_colmap_with_ba(
        self,
        predictions: dict[str, np.ndarray],
        images: Any,
        image_paths: list[Path],
        deps: dict[str, Any],
        torch: Any,
        dtype: Any,
        output_dir: Path,
    ) -> dict[str, Any]:
        try:
            import pycolmap  # type: ignore
            from vggt.dependency.track_predict import predict_tracks  # type: ignore
            from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap  # type: ignore
        except Exception as exc:
            raise VGGTDependencyError(
                "Bundle adjustment requires VGGT demo dependencies: pycolmap, pyceres, and LightGlue."
            ) from exc

        autocast = _autocast_context(torch, images.device.type, dtype)
        with torch.no_grad():
            with autocast:
                pred_tracks, pred_vis_scores, _pred_confs, points_3d, points_rgb = predict_tracks(
                    images,
                    conf=predictions["depth_conf"],
                    points_3d=predictions["points_3d"],
                    masks=None,
                    max_query_pts=self.ba_max_query_points,
                    query_frame_num=self.ba_query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=True,
                )
        track_mask = pred_vis_scores > self.ba_vis_threshold
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            predictions["extrinsic"],
            predictions["intrinsic"],
            pred_tracks,
            np.array(images.shape[-2:]),
            masks=track_mask,
            max_reproj_error=self.ba_max_reproj_error,
            shared_camera=False,
            camera_type="PINHOLE",
            points_rgb=points_rgb,
        )
        if reconstruction is None:
            raise RuntimeError("VGGT-SfM bundle adjustment could not build a reconstruction")
        self._assign_colmap_image_names(reconstruction, image_paths)
        pycolmap.bundle_adjustment(reconstruction, pycolmap.BundleAdjustmentOptions())
        sparse_dir = output_dir / "colmap_sparse_ba"
        _reset_generated_dir(sparse_dir)
        reconstruction.write(str(sparse_dir))
        gsplat_scene_dir = self._prepare_gsplat_scene(output_dir, sparse_dir, image_paths)
        return {
            "mode": "tracked_bundle_adjustment",
            "bundle_adjustment": True,
            "sparse_dir": str(sparse_dir),
            "gsplat_scene_dir": str(gsplat_scene_dir),
            "valid_track_count": int(np.count_nonzero(valid_track_mask)),
            "image_names": [p.name for p in image_paths],
        }

    @staticmethod
    def _assign_colmap_image_names(reconstruction: Any, image_paths: list[Path]) -> None:
        image_ids = sorted(reconstruction.images)
        if len(image_ids) != len(image_paths):
            raise RuntimeError(
                "VGGT COLMAP export produced a different image count than the extracted keyframes: "
                f"{len(image_ids)} sparse images vs {len(image_paths)} keyframes."
            )
        for image_id, image_path in zip(image_ids, image_paths, strict=True):
            reconstruction.images[image_id].name = image_path.name

    @staticmethod
    def _prepare_gsplat_scene(output_dir: Path, sparse_dir: Path, image_paths: list[Path]) -> Path:
        scene_dir = output_dir / "gsplat_scene"
        images_dir = scene_dir / "images"
        scene_sparse_dir = scene_dir / "sparse"
        _reset_generated_dir(images_dir)
        _reset_generated_dir(scene_sparse_dir)
        for image_path in image_paths:
            shutil.copy2(image_path, images_dir / image_path.name)
        for sparse_file in sorted(sparse_dir.iterdir()):
            if sparse_file.is_file():
                shutil.copy2(sparse_file, scene_sparse_dir / sparse_file.name)
        return scene_dir


def backend_from_config(config: dict[str, Any]) -> ReconstructionBackend:
    backend_name = config.get("backend", "vggt_sfm")
    if backend_name != "vggt_sfm":
        raise ValueError("Only `vggt_sfm` reconstruction is supported. Remove fallback backends from the config.")
    vggt = config.get("vggt", {})
    sfm = config.get("sfm", {})
    return VGGT_SfMReconstructionBackend(
        keyframe_interval=int(config.get("keyframe_interval", 30)),
        max_keyframes=config.get("max_keyframes", 40),
        model_id=str(vggt.get("model_id", "facebook/VGGT-1B")),
        checkpoint_url=vggt.get("checkpoint_url"),
        device=str(vggt.get("device", "auto")),
        dtype=str(vggt.get("dtype", "auto")),
        preprocess_mode=str(vggt.get("preprocess_mode", "crop")),
        use_point_map=bool(vggt.get("use_point_map", False)),
        confidence_threshold=float(vggt.get("confidence_threshold", 5.0)),
        relax_ground_confidence=bool(vggt.get("relax_ground_confidence", True)),
        ground_confidence_threshold=float(vggt.get("ground_confidence_threshold", 1.2)),
        max_points=int(vggt.get("max_points", 250_000)),
        sample_stride=int(vggt.get("sample_stride", 2)),
        seed=int(vggt.get("seed", 7)),
        save_depth=bool(vggt.get("save_depth", False)),
        save_colmap=bool(sfm.get("save_colmap", True)),
        bundle_adjustment=bool(sfm.get("bundle_adjustment", False)),
        ba_max_query_points=int(sfm.get("ba_max_query_points", 4096)),
        ba_query_frame_num=int(sfm.get("ba_query_frame_num", 8)),
        ba_vis_threshold=float(sfm.get("ba_vis_threshold", 0.2)),
        ba_max_reproj_error=float(sfm.get("ba_max_reproj_error", 8.0)),
    )


def _load_vggt_dependencies(require_ba: bool = False) -> dict[str, Any]:
    try:
        import torch  # type: ignore
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
        from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
    except Exception as exc:
        raise VGGTDependencyError(
            "VGGT-SfM backend requires PyTorch, torchvision, and the official VGGT package. "
            "Install them with the commands documented in README.md."
        ) from exc
    if require_ba:
        try:
            import pycolmap  # noqa: F401  # type: ignore
            import pyceres  # noqa: F401  # type: ignore
            import lightglue  # noqa: F401  # type: ignore
        except Exception as exc:
            raise VGGTDependencyError(
                "VGGT-SfM bundle adjustment requires pycolmap, pyceres, and LightGlue."
            ) from exc
    return {
        "torch": torch,
        "VGGT": VGGT,
        "load_and_preprocess_images": load_and_preprocess_images,
        "pose_encoding_to_extri_intri": pose_encoding_to_extri_intri,
        "unproject_depth_map_to_point_map": unproject_depth_map_to_point_map,
    }


def _autocast_context(torch: Any, device_type: str, dtype: Any) -> Any:
    if dtype == torch.float32:
        return torch.autocast(device_type=device_type, enabled=False)
    return torch.autocast(device_type=device_type, dtype=dtype)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _cleanup_cuda(torch: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _reset_generated_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resized_image_colors(images: Any, size_hw: tuple[int, int], torch: Any) -> np.ndarray:
    import torch.nn.functional as F  # type: ignore

    resized = F.interpolate(images, size=size_hw, mode="bilinear", align_corners=False)
    colors = (_as_numpy(resized) * 255.0).clip(0, 255).astype(np.uint8)
    return colors.transpose(0, 2, 3, 1)


def _squeeze_confidence(confidence: np.ndarray) -> np.ndarray:
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        return confidence[..., 0]
    return confidence


def _vggt_keep_mask(
    points_3d: np.ndarray,
    point_conf: np.ndarray,
    colors: np.ndarray,
    *,
    confidence_threshold: float,
    relax_ground_confidence: bool,
    ground_confidence_threshold: float,
    sample_stride: int,
) -> np.ndarray:
    finite = np.isfinite(points_3d).all(axis=-1) & np.isfinite(point_conf)
    mask = finite & (point_conf >= confidence_threshold)
    if relax_ground_confidence:
        mask |= finite & _duckietown_ground_color_mask(colors) & (point_conf >= ground_confidence_threshold)
    if sample_stride > 1:
        stride_mask = np.zeros(mask.shape, dtype=bool)
        stride_mask[:, ::sample_stride, ::sample_stride] = True
        mask &= stride_mask
    return mask


def _limit_mask(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if not max_points or np.count_nonzero(mask) <= max_points:
        return mask
    selected_indices = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    keep_indices = rng.choice(selected_indices, size=max_points, replace=False)
    limited = np.zeros(mask.size, dtype=bool)
    limited[keep_indices] = True
    return limited.reshape(mask.shape)


def _duckietown_ground_color_mask(colors: np.ndarray) -> np.ndarray:
    arr = colors.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    saturation = maxc - minc
    black_road = (maxc <= 100) & (saturation <= 90)
    white_lines = (maxc >= 145) & (saturation <= 85)
    yellow_lines = (r >= 95) & (g >= 75) & (b <= 125) & (saturation >= 30)
    red_stop = (r >= 100) & (r >= 1.25 * np.maximum(g, b)) & (saturation >= 35)
    return black_road | white_lines | yellow_lines | red_stop


def _extrinsics_to_camera_poses(extrinsics: np.ndarray) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    for extrinsic in extrinsics:
        transform = np.eye(4)
        transform[:3, :4] = extrinsic
        poses.append(np.linalg.inv(transform))
    return poses


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for keyframe image export.") from exc
    return cv2
