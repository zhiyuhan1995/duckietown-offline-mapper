from __future__ import annotations

from enum import IntEnum

import numpy as np


class SemanticClass(IntEnum):
    UNKNOWN = 0
    DRIVABLE_ROAD = 1
    NON_DRIVABLE = 2
    LANE_MARKING = 3
    STOP_LINE = 4


SEMANTIC_COLORS = {
    SemanticClass.UNKNOWN: (127, 127, 127),
    SemanticClass.DRIVABLE_ROAD: (255, 255, 255),
    SemanticClass.NON_DRIVABLE: (0, 0, 0),
    SemanticClass.LANE_MARKING: (255, 220, 0),
    SemanticClass.STOP_LINE: (255, 0, 0),
}


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for BEV segmentation.") from exc
    return cv2


def _morph(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    cv2 = _require_cv2()
    out = mask.astype(np.uint8)
    if open_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_size + 1, 2 * open_size + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_size > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_size + 1, 2 * close_size + 1))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out.astype(bool)


def segment_bev_rgb(bev_rgb: np.ndarray, config: dict) -> np.ndarray:
    cv2 = _require_cv2()
    rgb = np.asarray(bev_rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    unknown = np.all(rgb == np.array(config.get("unknown_rgb", [80, 80, 80]), dtype=np.uint8), axis=-1)

    road = (v <= config.get("road_v_max", 95)) & (s <= config.get("road_s_max", 115)) & ~unknown
    white = (v >= config.get("white_v_min", 150)) & (s <= config.get("white_s_max", 85)) & ~unknown
    yellow = (
        (h >= config.get("yellow_h_min", 15))
        & (h <= config.get("yellow_h_max", 40))
        & (s >= config.get("yellow_s_min", 60))
        & (v >= config.get("yellow_v_min", 80))
        & ~unknown
    )
    red_low = h <= 8
    red_high = h >= 170
    red = (red_low | red_high) & (s >= config.get("red_s_min", 70)) & (v >= config.get("red_v_min", 70)) & ~unknown

    open_size = int(config.get("morphology_open", 1))
    close_size = int(config.get("morphology_close", 3))
    road = _morph(road, open_size, close_size)
    white = _morph(white, 0, max(1, close_size // 2))
    yellow = _morph(yellow, 0, max(1, close_size // 2))
    red = _morph(red, 0, max(1, close_size // 2))

    semantic = np.full(rgb.shape[:2], SemanticClass.UNKNOWN, dtype=np.uint8)
    observed = ~unknown
    semantic[observed] = SemanticClass.NON_DRIVABLE
    semantic[road] = SemanticClass.DRIVABLE_ROAD
    semantic[white | yellow] = SemanticClass.LANE_MARKING
    semantic[red] = SemanticClass.STOP_LINE
    return semantic


def colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    out = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for cls, color in SEMANTIC_COLORS.items():
        out[semantic == int(cls)] = color
    return out

