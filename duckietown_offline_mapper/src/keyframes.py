from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Keyframe:
    index: int
    timestamp_s: float
    image_bgr: np.ndarray


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video/image loading. Install requirements.txt."
        ) from exc
    return cv2


def extract_keyframes(
    video_path: str | Path,
    interval: int = 30,
    max_keyframes: int | None = None,
) -> list[Keyframe]:
    cv2 = _require_cv2()
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[Keyframe] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, interval) == 0:
            frames.append(Keyframe(frame_index, frame_index / fps, frame))
            if max_keyframes and len(frames) >= max_keyframes:
                break
        frame_index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frames


def load_image_folder(folder: str | Path, max_images: int | None = None) -> list[Keyframe]:
    cv2 = _require_cv2()
    folder = Path(folder)
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)
    if max_images:
        paths = paths[:max_images]
    frames: list[Keyframe] = []
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(Keyframe(i, float(i), image))
    if not frames:
        raise RuntimeError(f"No readable images found in {folder}")
    return frames


def save_keyframe_previews(frames: list[Keyframe], output_dir: str | Path, limit: int = 12) -> list[Path]:
    cv2 = _require_cv2()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("keyframe_*.jpg"):
        old_path.unlink()
    paths: list[Path] = []
    for frame in frames[:limit]:
        path = output_dir / f"keyframe_{frame.index:06d}.jpg"
        cv2.imwrite(str(path), frame.image_bgr)
        paths.append(path)
    return paths
