from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - runtime fallback
    YOLO = None


@dataclass
class SceneConfig:
    trail_length: int = 28
    max_players_per_team: int = 6
    player_track_ttl: int = 26
    player_active_ttl: int = 18
    player_zone_ttl: int = 10
    player_vote_gain: float = 1.0
    player_vote_decay: float = 0.05
    player_stable_threshold: float = 2.8
    team_color_floor_ratio: float = 0.012
    team_color_margin: float = 0.004
    torso_x_margin_ratio: float = 0.32
    torso_y_start_ratio: float = 0.16
    torso_y_end_ratio: float = 0.42
    top_player_area_min: int = 700
    top_player_area_max: int = 42000
    bottom_player_area_min: int = 1500
    bottom_player_area_max: int = 65000
    player_min_height_top: int = 30
    player_min_height_bottom: int = 54
    player_min_width: int = 14
    player_max_width: int = 170
    player_max_aspect_ratio: float = 1.25
    player_min_fill_ratio: float = 0.18
    player_match_distance_top: float = 72.0
    player_match_distance_bottom: float = 92.0
    player_iou_match_threshold: float = 0.06
    player_bg_history: int = 240
    player_bg_var_threshold: int = 24
    player_bg_detect_shadows: bool = False
    player_top_open_kernel: tuple[int, int] = (3, 3)
    player_top_close_kernel: tuple[int, int] = (9, 9)
    player_top_dilate_kernel: tuple[int, int] = (5, 7)
    player_bottom_open_kernel: tuple[int, int] = (3, 3)
    player_bottom_close_kernel: tuple[int, int] = (9, 11)
    player_bottom_dilate_kernel: tuple[int, int] = (7, 11)
    player_morph_iterations: int = 1
    player_warmup_frames: int = 45
    player_color_open_kernel: tuple[int, int] = (3, 3)
    player_color_close_kernel: tuple[int, int] = (9, 9)
    player_zone_cols: int = 3
    player_zone_rows: int = 2
    player_min_track_age_for_count: int = 3
    player_track_confidence_gain: float = 1.0
    player_track_confidence_decay: float = 0.4
    player_track_confidence_floor: float = 1.6
    player_camshift_max_misses: int = 3
    player_hist_bins_h: int = 24
    player_hist_bins_s: int = 16
    player_hist_min_saturation: int = 40
    player_camshift_min_area: int = 700
    player_split_width_top: int = 84
    player_split_width_bottom: int = 120
    player_split_height_top: int = 120
    player_split_height_bottom: int = 180
    player_split_min_width: int = 26
    top_seed_area_min: int = 90
    top_seed_area_max: int = 2600
    bottom_seed_area_min: int = 120
    bottom_seed_area_max: int = 4200
    top_seed_min_height: int = 10
    bottom_seed_min_height: int = 12
    top_seed_expand_x: float = 0.7
    top_seed_expand_y: float = 0.35
    top_seed_expand_h: float = 2.5
    bottom_seed_expand_x: float = 0.6
    bottom_seed_expand_y: float = 0.38
    bottom_seed_expand_h: float = 2.8
    yellow_player_lower: np.ndarray = field(
        default_factory=lambda: np.array((14, 85, 75), dtype=np.uint8)
    )
    yellow_player_upper: np.ndarray = field(
        default_factory=lambda: np.array((40, 255, 255), dtype=np.uint8)
    )
    blue_player_lower: np.ndarray = field(
        default_factory=lambda: np.array((95, 45, 35), dtype=np.uint8)
    )
    blue_player_upper: np.ndarray = field(
        default_factory=lambda: np.array((140, 255, 255), dtype=np.uint8)
    )
    red_lower_1: np.ndarray = field(
        default_factory=lambda: np.array((0, 50, 70), dtype=np.uint8)
    )
    red_upper_1: np.ndarray = field(
        default_factory=lambda: np.array((12, 255, 255), dtype=np.uint8)
    )
    red_lower_2: np.ndarray = field(
        default_factory=lambda: np.array((165, 50, 70), dtype=np.uint8)
    )
    red_upper_2: np.ndarray = field(
        default_factory=lambda: np.array((180, 255, 255), dtype=np.uint8)
    )
    white_player_lower: np.ndarray = field(
        default_factory=lambda: np.array((0, 0, 145), dtype=np.uint8)
    )
    white_player_upper: np.ndarray = field(
        default_factory=lambda: np.array((180, 70, 255), dtype=np.uint8)
    )
    ball_motion_threshold: int = 20
    ball_motion_ratio_min: float = 0.018
    ball_color_ratio_min: float = 0.012
    ball_dual_color_bonus: float = 1.6
    ball_motion_bonus: float = 1.2
    ball_inside_player_penalty: float = 2.0
    ball_inside_player_gate: float = 42.0
    ball_min_area: float = 10.0
    ball_max_area: float = 420.0
    ball_min_radius: float = 2.5
    ball_max_radius: float = 15.5
    ball_min_circularity: float = 0.45
    ball_max_aspect_ratio: float = 1.5
    ball_distance_gate: float = 210.0
    ball_predicted_distance_gate: float = 92.0
    ball_reacquire_distance_gate: float = 145.0
    ball_max_misses: int = 8
    ball_bridge_limit: int = 3
    ball_predicted_draw_misses: int = 3
    ball_max_uncertainty: float = 2200.0
    ball_base_roi: int = 160
    ball_velocity_roi_gain: float = 2.8
    ball_miss_roi_gain: float = 36.0
    ball_max_roi: int = 420
    ball_optical_flow_max_frames: int = 4
    ball_optical_flow_win_size: tuple[int, int] = (21, 21)
    ball_optical_flow_max_level: int = 2
    ball_optical_flow_quality: float = 0.01
    ball_optical_flow_min_distance: int = 2
    ball_optical_flow_block_size: int = 5
    ball_color_open_kernel: tuple[int, int] = (3, 3)
    ball_color_close_kernel: tuple[int, int] = (5, 5)
    ball_min_combined_color_ratio: float = 0.024
    ball_min_single_color_ratio: float = 0.018
    ball_global_motion_relaxed_ratio: float = 0.008
    ball_reacquire_after_misses: int = 2
    ball_local_track_misses: int = 1
    ball_confirm_frames: int = 2
    ball_candidate_match_distance: float = 34.0
    ball_banner_guard_y: int = 150
    ball_banner_motion_ratio: float = 0.03
    ball_local_color_density_max: float = 0.42
    ball_context_window_scale: float = 3.2
    ball_context_window_min: int = 16
    ball_predicted_player_gate: float = 22.0
    ball_motion_seed_threshold: int = 26
    ball_min_progress_pixels: float = 6.0
    ball_stale_lock_frames: int = 2
    ball_detector_interval: int = 3
    ball_detector_confidence: float = 0.18
    ball_detector_imgsz_full: int = 1280
    ball_detector_imgsz_roi: int = 640
    ball_yellow_lower: np.ndarray = field(
        default_factory=lambda: np.array((12, 85, 75), dtype=np.uint8)
    )
    ball_yellow_upper: np.ndarray = field(
        default_factory=lambda: np.array((42, 255, 255), dtype=np.uint8)
    )
    ball_blue_lower: np.ndarray = field(
        default_factory=lambda: np.array((90, 45, 35), dtype=np.uint8)
    )
    ball_blue_upper: np.ndarray = field(
        default_factory=lambda: np.array((140, 255, 255), dtype=np.uint8)
    )
    ball_search_polygon: np.ndarray = field(
        default_factory=lambda: np.array(
            [(110, 0), (1170, 0), (1230, 470), (50, 470)],
            dtype=np.int32,
        )
    )
    top_court_polygon: np.ndarray = field(
        default_factory=lambda: np.array(
            [(250, 252), (1030, 252), (1110, 448), (170, 448)],
            dtype=np.int32,
        )
    )
    bottom_court_polygon: np.ndarray = field(
        default_factory=lambda: np.array(
            [(170, 338), (1110, 338), (1190, 698), (92, 698)],
            dtype=np.int32,
        )
    )
    player_exclusion_rects: tuple[tuple[int, int, int, int], ...] = (
        (0, 0, 155, 720),
        (1125, 0, 1280, 720),
        (214, 120, 322, 335),
        (905, 130, 1035, 315),
    )
    ball_exclusion_rects: tuple[tuple[int, int, int, int], ...] = (
        (0, 0, 135, 720),
        (1145, 0, 1280, 720),
        (228, 90, 304, 535),
        (978, 90, 1048, 535),
        (492, 106, 575, 162),
    )


@dataclass
class PlayerBlob:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    footpoint: tuple[int, int]
    side: str
    zone_hints: int = 1
    source: str = "motion"


@dataclass
class PlayerTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    footpoint: tuple[int, int]
    side: str
    kalman: cv2.KalmanFilter
    votes: dict[str, float] = field(default_factory=lambda: {"team_a": 0.0, "team_b": 0.0})
    stable_team: Optional[str] = None
    last_seen: int = 0
    misses: int = 0
    age: int = 0
    confidence: float = 0.0
    zone_index: Optional[int] = None
    hist: Optional[np.ndarray] = None
    track_window: Optional[tuple[int, int, int, int]] = None


@dataclass
class BallCandidate:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    radius: float
    circularity: float
    yellow_ratio: float
    blue_ratio: float
    motion_ratio: float
    inside_player: bool
    source: str
    motion_backed: bool
    circle_support: bool
    local_color_density: float = 0.0
    score_hint: float = 0.0


@dataclass
class BallState:
    center: Optional[tuple[int, int]]
    radius: Optional[float]
    status: str


@dataclass
class PendingBallLock:
    center: tuple[int, int]
    radius: float
    frames: int = 1


def build_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def apply_exclusions(mask: np.ndarray, exclusions: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    for x1, y1, x2, y2 in exclusions:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    return mask


def point_in_mask(point: tuple[int, int], mask: np.ndarray) -> bool:
    x, y = point
    if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
        return False
    return bool(mask[y, x])


def clip_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x, y, w, h = bbox
    x = int(np.clip(x, 0, max(width - 1, 0)))
    y = int(np.clip(y, 0, max(height - 1, 0)))
    w = int(np.clip(w, 0, width - x))
    h = int(np.clip(h, 0, height - y))
    return (x, y, w, h)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return clip_bbox((x - pad_x, y - pad_y, w + (pad_x * 2), h + (pad_y * 2)), frame_shape)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = bbox
    return (x + (w // 2), y + (h // 2))


def bbox_footpoint(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, w, h = bbox
    return (x + (w // 2), y + h - 1)


def torso_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    torso_x = x + int(round(w * 0.22))
    torso_w = max(8, w - int(round(w * 0.44)))
    torso_y = y + int(round(h * 0.12))
    torso_h = max(10, int(round(h * 0.42)))
    return (torso_x, torso_y, torso_w, torso_h)


def distance_between(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def crop_roi(frame: np.ndarray, center: tuple[int, int], roi_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    half = roi_size // 2
    x1 = max(center[0] - half, 0)
    y1 = max(center[1] - half, 0)
    x2 = min(center[0] + half, frame.shape[1])
    y2 = min(center[1] + half, frame.shape[0])
    return frame[y1:y2, x1:x2], (x1, y1)


def build_track_kalman(center: tuple[int, int]) -> cv2.KalmanFilter:
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 30.0
    kalman.statePost = np.array(
        [[np.float32(center[0])], [np.float32(center[1])], [0.0], [0.0]],
        dtype=np.float32,
    )
    return kalman


def extract_torso_histogram(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    config: SceneConfig,
) -> Optional[np.ndarray]:
    x, y, w, h = torso_bbox(bbox)
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_mask = cv2.inRange(
        hsv,
        np.array((0, config.player_hist_min_saturation, 0), dtype=np.uint8),
        np.array((180, 255, 255), dtype=np.uint8),
    )
    if cv2.countNonZero(sat_mask) == 0:
        return None
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        sat_mask,
        [config.player_hist_bins_h, config.player_hist_bins_s],
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def classify_team(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    expected_team: str,
    config: SceneConfig,
) -> Optional[str]:
    x, y, w, h = bbox
    x1 = x + int(round(w * config.torso_x_margin_ratio))
    x2 = x + w - int(round(w * config.torso_x_margin_ratio))
    y1 = y + int(round(h * config.torso_y_start_ratio))
    y2 = y + int(round(h * config.torso_y_end_ratio))
    x1 = max(x, x1)
    x2 = min(x + w, x2)
    y1 = max(y, y1)
    y2 = min(y + h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return expected_team

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixel_count = float(max(crop.shape[0] * crop.shape[1], 1))
    yellow_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.yellow_player_lower, config.yellow_player_upper)
    ) / pixel_count
    blue_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.blue_player_lower, config.blue_player_upper)
    ) / pixel_count
    red_ratio = cv2.countNonZero(
        cv2.bitwise_or(
            cv2.inRange(hsv_crop, config.red_lower_1, config.red_upper_1),
            cv2.inRange(hsv_crop, config.red_lower_2, config.red_upper_2),
        )
    ) / pixel_count
    white_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.white_player_lower, config.white_player_upper)
    ) / pixel_count

    if red_ratio > max(yellow_ratio, blue_ratio) and red_ratio >= 0.02:
        return None
    if yellow_ratio >= config.team_color_floor_ratio and yellow_ratio > blue_ratio + config.team_color_margin:
        return "team_a"
    if blue_ratio >= config.team_color_floor_ratio and blue_ratio > yellow_ratio + config.team_color_margin:
        return "team_b"
    if white_ratio >= 0.08:
        return expected_team
    return expected_team


class PlayerTrackerOpenCVV2:
    def __init__(self, config: SceneConfig, top_mask: np.ndarray, bottom_mask: np.ndarray) -> None:
        self.config = config
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.bg_top = cv2.createBackgroundSubtractorMOG2(
            history=config.player_bg_history,
            varThreshold=config.player_bg_var_threshold,
            detectShadows=config.player_bg_detect_shadows,
        )
        self.bg_bottom = cv2.createBackgroundSubtractorMOG2(
            history=config.player_bg_history,
            varThreshold=config.player_bg_var_threshold,
            detectShadows=config.player_bg_detect_shadows,
        )
        self.top_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_open_kernel)
        self.top_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_close_kernel)
        self.top_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_top_dilate_kernel)
        self.bottom_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_open_kernel)
        self.bottom_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_close_kernel)
        self.bottom_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_bottom_dilate_kernel)
        self.color_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_color_open_kernel)
        self.color_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_color_close_kernel)
        self.tracks: dict[int, PlayerTrack] = {}
        self.next_track_id = 1
        self.last_output_counts: Optional[tuple[int, int]] = None
        self.side_zone_memory: dict[str, dict[int, int]] = {"top": {}, "bottom": {}}
        self.top_zone_boxes = self._build_zone_boxes(config.top_court_polygon)
        self.bottom_zone_boxes = self._build_zone_boxes(config.bottom_court_polygon)
        self.ball_suppression_boxes: list[tuple[int, int, int, int]] = []

    def update(self, frame: np.ndarray, frame_index: int) -> tuple[list[PlayerTrack], int, int]:
        top_motion = self._extract_motion_blobs(frame, frame_index, "top")
        bottom_motion = self._extract_motion_blobs(frame, frame_index, "bottom")
        color_blobs = self._extract_color_blobs(frame)
        blobs = self._merge_blobs(top_motion + bottom_motion + color_blobs)
        self._associate(blobs, frame, frame_index)
        active = self._active_tracks(frame_index)
        team_a_count, team_b_count = self._count_from_tracks(active, blobs, frame_index)
        self.ball_suppression_boxes = self._build_ball_suppression_boxes(active, blobs, frame.shape)
        return active, team_a_count, team_b_count

    def _build_zone_boxes(self, polygon: np.ndarray) -> list[tuple[int, int, int, int]]:
        x, y, w, h = cv2.boundingRect(polygon)
        boxes: list[tuple[int, int, int, int]] = []
        for row in range(self.config.player_zone_rows):
            for col in range(self.config.player_zone_cols):
                x1 = x + int(round((col / self.config.player_zone_cols) * w))
                x2 = x + int(round(((col + 1) / self.config.player_zone_cols) * w))
                y1 = y + int(round((row / self.config.player_zone_rows) * h))
                y2 = y + int(round(((row + 1) / self.config.player_zone_rows) * h))
                boxes.append((x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
        return boxes

    def _extract_motion_blobs(self, frame: np.ndarray, frame_index: int, side: str) -> list[PlayerBlob]:
        mask = self.top_mask if side == "top" else self.bottom_mask
        subtractor = self.bg_top if side == "top" else self.bg_bottom
        open_kernel = self.top_open_kernel if side == "top" else self.bottom_open_kernel
        close_kernel = self.top_close_kernel if side == "top" else self.bottom_close_kernel
        dilate_kernel = self.top_dilate_kernel if side == "top" else self.bottom_dilate_kernel

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        fg_mask = subtractor.apply(masked_frame)
        if frame_index < self.config.player_warmup_frames:
            return []

        _, fg_mask = cv2.threshold(fg_mask, 210, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, mask)
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=self.config.player_morph_iterations,
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=self.config.player_morph_iterations + 1,
        )
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)
        return self._extract_player_blobs(fg_mask, frame.shape, side)

    def _extract_player_blobs(
        self,
        fg_mask: np.ndarray,
        frame_shape: tuple[int, int, int],
        side: str,
    ) -> list[PlayerBlob]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[PlayerBlob] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)

            x, y, w, h = cv2.boundingRect(contour)
            bbox = clip_bbox((x, y, w, h), frame_shape)
            x, y, w, h = bbox
            if w < self.config.player_min_width or w > self.config.player_max_width:
                continue
            if h <= 0 or w / max(float(h), 1.0) > self.config.player_max_aspect_ratio:
                continue
            if side == "top":
                min_height = self.config.player_min_height_top
                area_min = self.config.top_player_area_min
                area_max = self.config.top_player_area_max
            else:
                min_height = self.config.player_min_height_bottom
                area_min = self.config.bottom_player_area_min
                area_max = self.config.bottom_player_area_max
            if h < min_height or area < area_min or area > area_max:
                continue
            fill_ratio = area / max(float(w * h), 1.0)
            if fill_ratio < self.config.player_min_fill_ratio or solidity < 0.18:
                continue

            footpoint = bbox_footpoint(bbox)
            side_mask = self.top_mask if side == "top" else self.bottom_mask
            if not point_in_mask(footpoint, side_mask):
                continue
            zone_hints = self._split_count_hint(bbox, side)
            blobs.append(
                PlayerBlob(
                    bbox=bbox,
                    center=bbox_center(bbox),
                    footpoint=footpoint,
                    side=side,
                    zone_hints=zone_hints,
                    source="motion",
                )
            )
        return blobs

    def _split_count_hint(
        self,
        bbox: tuple[int, int, int, int],
        side: str,
    ) -> int:
        x, y, w, h = bbox
        split_width = self.config.player_split_width_top if side == "top" else self.config.player_split_width_bottom
        split_height = self.config.player_split_height_top if side == "top" else self.config.player_split_height_bottom
        if w < split_width and h < split_height:
            return 1

        split_count = 1
        if w >= split_width * 1.8:
            split_count = 3
        elif w >= split_width:
            split_count = 2
        elif h >= split_height:
            split_count = 2
        return split_count

    def _extract_color_blobs(self, frame: np.ndarray) -> list[PlayerBlob]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        top_seed = cv2.bitwise_and(
            cv2.inRange(hsv, self.config.yellow_player_lower, self.config.yellow_player_upper),
            self.top_mask,
        )
        bottom_seed = cv2.bitwise_and(
            cv2.inRange(hsv, self.config.blue_player_lower, self.config.blue_player_upper),
            self.bottom_mask,
        )
        top_seed = cv2.morphologyEx(top_seed, cv2.MORPH_OPEN, self.color_open_kernel, iterations=1)
        top_seed = cv2.morphologyEx(top_seed, cv2.MORPH_CLOSE, self.color_close_kernel, iterations=1)
        bottom_seed = cv2.morphologyEx(bottom_seed, cv2.MORPH_OPEN, self.color_open_kernel, iterations=1)
        bottom_seed = cv2.morphologyEx(bottom_seed, cv2.MORPH_CLOSE, self.color_close_kernel, iterations=1)
        return self._extract_seed_blobs(top_seed, frame.shape, "top") + self._extract_seed_blobs(
            bottom_seed,
            frame.shape,
            "bottom",
        )

    def _extract_seed_blobs(
        self,
        seed_mask: np.ndarray,
        frame_shape: tuple[int, int, int],
        side: str,
    ) -> list[PlayerBlob]:
        contours, _ = cv2.findContours(seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[PlayerBlob] = []
        area_min = self.config.top_seed_area_min if side == "top" else self.config.bottom_seed_area_min
        area_max = self.config.top_seed_area_max if side == "top" else self.config.bottom_seed_area_max
        min_height = self.config.top_seed_min_height if side == "top" else self.config.bottom_seed_min_height
        expand_x = self.config.top_seed_expand_x if side == "top" else self.config.bottom_seed_expand_x
        expand_y = self.config.top_seed_expand_y if side == "top" else self.config.bottom_seed_expand_y
        expand_h = self.config.top_seed_expand_h if side == "top" else self.config.bottom_seed_expand_h
        side_mask = self.top_mask if side == "top" else self.bottom_mask
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < area_min or area > area_max:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < min_height or w < 8 or w > 80:
                continue
            expanded = clip_bbox(
                (
                    int(round(x - (w * expand_x))),
                    int(round(y - (h * expand_y))),
                    int(round(w * (1.0 + (expand_x * 2.0)))),
                    int(round(h * expand_h)),
                ),
                frame_shape,
            )
            footpoint = bbox_footpoint(expanded)
            if not point_in_mask(footpoint, side_mask):
                continue
            blobs.append(
                PlayerBlob(
                    bbox=expanded,
                    center=bbox_center(expanded),
                    footpoint=footpoint,
                    side=side,
                    zone_hints=1,
                    source="seed",
                )
            )
        return blobs

    def _merge_blobs(self, blobs: list[PlayerBlob]) -> list[PlayerBlob]:
        merged: list[PlayerBlob] = []
        for blob in blobs:
            replaced = False
            for index, existing in enumerate(merged):
                if existing.side != blob.side:
                    continue
                if bbox_iou(existing.bbox, blob.bbox) > 0.38 or distance_between(existing.center, blob.center) < 24.0:
                    existing_area = existing.bbox[2] * existing.bbox[3]
                    blob_area = blob.bbox[2] * blob.bbox[3]
                    if existing.source == blob.source:
                        if blob_area > existing_area:
                            merged[index] = blob
                    elif blob.source == "seed" and existing.source == "motion":
                        if existing_area > blob_area * 1.65:
                            merged[index] = blob
                    elif blob.source == "motion" and existing.source == "seed":
                        if blob_area <= existing_area * 1.65:
                            merged[index] = blob
                    replaced = True
                    break
            if not replaced:
                merged.append(blob)
        return merged

    def _associate(self, blobs: list[PlayerBlob], frame: np.ndarray, frame_index: int) -> None:
        predictions: dict[int, tuple[int, int]] = {}
        unmatched_tracks = set(self.tracks.keys())
        used_blobs: set[int] = set()
        for track_id, track in self.tracks.items():
            prediction = track.kalman.predict()
            predictions[track_id] = (int(round(prediction[0][0])), int(round(prediction[1][0])))

        while True:
            best_pair: Optional[tuple[int, int]] = None
            best_score = float("inf")
            for blob_index, blob in enumerate(blobs):
                if blob_index in used_blobs:
                    continue
                for track_id in list(unmatched_tracks):
                    track = self.tracks[track_id]
                    if track.side != blob.side:
                        continue
                    match_distance = (
                        self.config.player_match_distance_top if blob.side == "top" else self.config.player_match_distance_bottom
                    )
                    pred_center = predictions.get(track_id, track.center)
                    foot_distance = distance_between(blob.footpoint, track.footpoint)
                    center_distance = distance_between(blob.center, pred_center)
                    iou = bbox_iou(blob.bbox, track.bbox)
                    hist_score = self._hist_similarity(track, frame, blob.bbox)
                    if center_distance > match_distance and iou < self.config.player_iou_match_threshold:
                        continue
                    score = foot_distance + (center_distance * 0.6) - (iou * 140.0) + ((1.0 - hist_score) * 24.0)
                    if score < best_score:
                        best_score = score
                        best_pair = (blob_index, track_id)
            if best_pair is None:
                break
            blob_index, track_id = best_pair
            used_blobs.add(blob_index)
            unmatched_tracks.discard(track_id)
            self._update_track(self.tracks[track_id], blobs[blob_index], frame, frame_index)

        for blob_index, blob in enumerate(blobs):
            if blob_index in used_blobs:
                continue
            track = PlayerTrack(
                track_id=self.next_track_id,
                bbox=blob.bbox,
                center=blob.center,
                footpoint=blob.footpoint,
                side=blob.side,
                kalman=build_track_kalman(blob.center),
                stable_team="team_a" if blob.side == "top" else "team_b",
                last_seen=frame_index,
                misses=0,
                age=1,
                confidence=self.config.player_track_confidence_gain,
            )
            track.zone_index = self._zone_for_track(track)
            track.track_window = track.bbox
            track.hist = extract_torso_histogram(frame, track.bbox, self.config)
            self._update_votes(track, frame)
            self.tracks[track.track_id] = track
            self.next_track_id += 1

        self._maintain_unmatched_tracks(frame, frame_index, list(unmatched_tracks))

    def _update_track(self, track: PlayerTrack, blob: PlayerBlob, frame: np.ndarray, frame_index: int) -> None:
        measurement = np.array([[np.float32(blob.center[0])], [np.float32(blob.center[1])]], dtype=np.float32)
        track.kalman.correct(measurement)
        track.bbox = blob.bbox
        track.center = blob.center
        track.footpoint = blob.footpoint
        track.side = blob.side
        track.last_seen = frame_index
        track.misses = 0
        track.age += 1
        track.confidence = min(track.confidence + self.config.player_track_confidence_gain, 10.0)
        track.zone_index = self._zone_for_track(track)
        track.track_window = track.bbox
        refreshed_hist = extract_torso_histogram(frame, track.bbox, self.config)
        if refreshed_hist is not None:
            track.hist = refreshed_hist
        self._update_votes(track, frame)

    def _zone_for_track(self, track: PlayerTrack) -> Optional[int]:
        boxes = self.top_zone_boxes if track.side == "top" else self.bottom_zone_boxes
        for index, box in enumerate(boxes):
            x, y, w, h = box
            if x <= track.footpoint[0] <= x + w and y <= track.footpoint[1] <= y + h:
                return index
        if not boxes:
            return None
        distances = [distance_between(track.footpoint, (box[0] + box[2] // 2, box[1] + box[3] // 2)) for box in boxes]
        return int(np.argmin(distances))

    def _update_votes(self, track: PlayerTrack, frame: np.ndarray) -> None:
        expected_team = "team_a" if track.side == "top" else "team_b"
        observed_team = classify_team(frame, track.bbox, expected_team, self.config)
        if observed_team is not None:
            track.votes[observed_team] += self.config.player_vote_gain
            other_team = "team_b" if observed_team == "team_a" else "team_a"
            track.votes[other_team] = max(0.0, track.votes[other_team] - self.config.player_vote_decay)
        else:
            for team_name in track.votes:
                track.votes[team_name] = max(0.0, track.votes[team_name] - self.config.player_vote_decay)

        if track.votes["team_a"] >= self.config.player_stable_threshold and track.votes["team_a"] > track.votes["team_b"]:
            track.stable_team = "team_a"
        elif track.votes["team_b"] >= self.config.player_stable_threshold and track.votes["team_b"] > track.votes["team_a"]:
            track.stable_team = "team_b"
        elif track.stable_team is None:
            track.stable_team = expected_team

    def _maintain_unmatched_tracks(self, frame: np.ndarray, frame_index: int, unmatched_tracks: list[int]) -> None:
        for track_id in unmatched_tracks:
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            if not self._camshift_update(track, frame, frame_index):
                track.misses += 1
                track.confidence = max(0.0, track.confidence - self.config.player_track_confidence_decay)

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_index - track.last_seen > self.config.player_track_ttl
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def _camshift_update(self, track: PlayerTrack, frame: np.ndarray, frame_index: int) -> bool:
        if track.hist is None or track.track_window is None or track.misses >= self.config.player_camshift_max_misses:
            return False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject(
            [hsv],
            [0, 1],
            track.hist,
            [0, 180, 0, 256],
            1,
        )
        side_mask = self.top_mask if track.side == "top" else self.bottom_mask
        back_proj = cv2.bitwise_and(back_proj, side_mask)
        x, y, w, h = clip_bbox(track.track_window, frame.shape)
        if w <= 0 or h <= 0:
            return False
        window = (x, y, w, h)
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, window = cv2.CamShift(back_proj, window, term)
        x, y, w, h = clip_bbox(window, frame.shape)
        if w * h < self.config.player_camshift_min_area:
            return False
        pred = track.kalman.predict()
        pred_center = (int(round(pred[0][0])), int(round(pred[1][0])))
        found_center = bbox_center((x, y, w, h))
        if distance_between(found_center, pred_center) > (
            self.config.player_match_distance_top if track.side == "top" else self.config.player_match_distance_bottom
        ):
            return False

        prev_w = max(track.bbox[2], 1)
        prev_h = max(track.bbox[3], 1)
        clamped_w = int(np.clip(w, int(prev_w * 0.75), int(prev_w * 1.35)))
        clamped_h = int(np.clip(h, int(prev_h * 0.75), int(prev_h * 1.35)))
        bbox = clip_bbox(
            (
                found_center[0] - clamped_w // 2,
                found_center[1] - clamped_h // 2,
                clamped_w,
                clamped_h,
            ),
            frame.shape,
        )
        footpoint = bbox_footpoint(bbox)
        side_mask = self.top_mask if track.side == "top" else self.bottom_mask
        if not point_in_mask(footpoint, side_mask):
            return False
        measurement = np.array([[np.float32(bbox_center(bbox)[0])], [np.float32(bbox_center(bbox)[1])]], dtype=np.float32)
        track.kalman.correct(measurement)
        track.bbox = bbox
        track.center = bbox_center(bbox)
        track.footpoint = footpoint
        track.track_window = bbox
        track.zone_index = self._zone_for_track(track)
        track.last_seen = frame_index
        track.misses += 1
        track.confidence = max(0.0, track.confidence - (self.config.player_track_confidence_decay * 0.35))
        return True

    def _hist_similarity(
        self,
        track: PlayerTrack,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        if track.hist is None:
            return 0.5
        candidate_hist = extract_torso_histogram(frame, bbox, self.config)
        if candidate_hist is None:
            return 0.5
        score = cv2.compareHist(track.hist, candidate_hist, cv2.HISTCMP_CORREL)
        return float(np.clip((score + 1.0) * 0.5, 0.0, 1.0))

    def _active_tracks(self, frame_index: int) -> list[PlayerTrack]:
        active: list[PlayerTrack] = []
        for track in self.tracks.values():
            if frame_index - track.last_seen > self.config.player_active_ttl:
                continue
            if track.age < self.config.player_min_track_age_for_count:
                continue
            if track.confidence < self.config.player_track_confidence_floor:
                continue
            active.append(track)
        return active

    def _count_from_tracks(
        self,
        active: list[PlayerTrack],
        blobs: list[PlayerBlob],
        frame_index: int,
    ) -> tuple[int, int]:
        top_live = self.side_zone_memory["top"]
        bottom_live = self.side_zone_memory["bottom"]
        top_live.clear()
        bottom_live.clear()
        top_blob_zones: set[int] = set()
        bottom_blob_zones: set[int] = set()
        top_seed_zones: set[int] = set()
        bottom_seed_zones: set[int] = set()

        for track in active:
            if track.zone_index is None:
                continue
            side_memory = top_live if track.side == "top" else bottom_live
            side_memory[track.zone_index] = frame_index

        for blob in blobs:
            zone_indices = self._zone_indices_for_blob(blob)
            if not zone_indices:
                continue
            for zone_index in zone_indices:
                if blob.side == "top":
                    top_blob_zones.add(zone_index)
                    if blob.source == "seed":
                        top_seed_zones.add(zone_index)
                else:
                    bottom_blob_zones.add(zone_index)
                    if blob.source == "seed":
                        bottom_seed_zones.add(zone_index)

        for track in self.tracks.values():
            if frame_index - track.last_seen > self.config.player_zone_ttl:
                continue
            if track.zone_index is None:
                continue
            if track.side == "top":
                top_live.setdefault(track.zone_index, track.last_seen)
            else:
                bottom_live.setdefault(track.zone_index, track.last_seen)

        team_a_candidate = max(
            len(top_live),
            len(top_seed_zones),
            min(len(top_blob_zones) + 1, self.config.max_players_per_team),
        )
        team_b_candidate = max(
            len(bottom_live),
            len(bottom_seed_zones),
            min(len(bottom_blob_zones) + 1, self.config.max_players_per_team),
        )
        team_a_candidate = int(np.clip(team_a_candidate, 0, self.config.max_players_per_team))
        team_b_candidate = int(np.clip(team_b_candidate, 0, self.config.max_players_per_team))

        merged_top = any(blob.side == "top" and blob.zone_hints > 1 for blob in blobs if blob.source == "motion")
        merged_bottom = any(blob.side == "bottom" and blob.zone_hints > 1 for blob in blobs if blob.source == "motion")

        if self.last_output_counts is None:
            self.last_output_counts = (max(team_a_candidate, 5), max(team_b_candidate, 5))
        else:
            last_a, last_b = self.last_output_counts
            if len(top_seed_zones) >= max(last_a - 1, 4):
                team_a_candidate = max(team_a_candidate, len(top_seed_zones))
            if len(bottom_seed_zones) >= max(last_b - 1, 4):
                team_b_candidate = max(team_b_candidate, len(bottom_seed_zones))
            if merged_top and team_a_candidate < last_a:
                team_a_candidate = last_a
            if merged_bottom and team_b_candidate < last_b:
                team_b_candidate = last_b
            if last_a >= 5 and team_a_candidate < 4:
                team_a_candidate = max(team_a_candidate, last_a - 1)
            if last_b >= 5 and team_b_candidate < 4:
                team_b_candidate = max(team_b_candidate, last_b - 1)
            if team_a_candidate < last_a - 2:
                team_a_candidate = max(team_a_candidate, last_a - 1)
            if team_b_candidate < last_b - 2:
                team_b_candidate = max(team_b_candidate, last_b - 1)
            if team_a_candidate == 0 and last_a >= 4:
                team_a_candidate = last_a
            if team_b_candidate == 0 and last_b >= 4:
                team_b_candidate = last_b
            if last_a >= 5 and (merged_top or len(top_seed_zones) >= 3 or len(top_blob_zones) >= 3) and team_a_candidate < 5:
                team_a_candidate = 5
            if last_b >= 5 and (merged_bottom or len(bottom_seed_zones) >= 3 or len(bottom_blob_zones) >= 3) and team_b_candidate < 5:
                team_b_candidate = 5
            self.last_output_counts = (
                int(np.clip(team_a_candidate, 0, self.config.max_players_per_team)),
                int(np.clip(team_b_candidate, 0, self.config.max_players_per_team)),
            )
        return self.last_output_counts

    def _build_ball_suppression_boxes(
        self,
        active: list[PlayerTrack],
        blobs: list[PlayerBlob],
        frame_shape: tuple[int, int, int],
    ) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for track in active:
            boxes.append(expand_bbox(torso_bbox(track.bbox), frame_shape, 8, 8))
        for blob in blobs:
            suppression = blob.bbox if blob.source == "seed" else torso_bbox(blob.bbox)
            boxes.append(expand_bbox(suppression, frame_shape, 6, 6))
        return boxes

    def _zone_for_blob(self, blob: PlayerBlob) -> Optional[int]:
        boxes = self.top_zone_boxes if blob.side == "top" else self.bottom_zone_boxes
        for index, box in enumerate(boxes):
            x, y, w, h = box
            if x <= blob.footpoint[0] <= x + w and y <= blob.footpoint[1] <= y + h:
                return index
        return None

    def _zone_indices_for_blob(self, blob: PlayerBlob) -> list[int]:
        boxes = self.top_zone_boxes if blob.side == "top" else self.bottom_zone_boxes
        if not boxes:
            return []
        base_index = self._zone_for_blob(blob)
        if base_index is None:
            return []
        if blob.zone_hints <= 1:
            return [base_index]

        x, y, w, h = blob.bbox
        candidate_indices = []
        for index, box in enumerate(boxes):
            if bbox_iou(blob.bbox, box) > 0.08:
                candidate_indices.append(index)
        if base_index not in candidate_indices:
            candidate_indices.append(base_index)
        candidate_indices = sorted(set(candidate_indices), key=lambda idx: abs(idx - base_index))
        return candidate_indices[: blob.zone_hints]


class BallTrackerOpenCVV2:
    def __init__(self, config: SceneConfig, search_mask: np.ndarray, detector_model: Optional["YOLO"] = None) -> None:
        self.config = config
        self.search_mask = search_mask
        self.detector_model = detector_model
        self.kalman = self._build_kalman()
        self.initialized = False
        self.mode = "SEARCH_INIT"
        self.last_confirmed_center: Optional[tuple[int, int]] = None
        self.last_confirmed_radius: float = 0.0
        self.miss_count = 0
        self.bridge_count = 0
        self.trail: deque[tuple[int, int]] = deque(maxlen=config.trail_length)
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_ball_gray: Optional[np.ndarray] = None
        self.previous_ball_points: Optional[np.ndarray] = None
        self.color_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_color_open_kernel)
        self.color_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_color_close_kernel)
        self.pending_lock: Optional[PendingBallLock] = None
        self.current_player_boxes: list[tuple[int, int, int, int]] = []
        self.stale_lock_frames = 0

    def _build_kalman(self) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 40.0
        return kalman

    def update(self, frame: np.ndarray, frame_index: int, player_boxes: list[tuple[int, int, int, int]]) -> BallState:
        self.current_player_boxes = player_boxes
        prediction = self._predict()
        predicted_center = None if prediction is None else (int(round(prediction[0])), int(round(prediction[1])))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        motion_mask, gray = self._build_motion_mask(frame)

        candidates = self._find_candidates(frame, hsv, motion_mask, player_boxes, predicted_center)
        if self.detector_model is not None and (
            frame_index == 0
            or self.mode in {"SEARCH_INIT", "SEARCH_REACQUIRE"}
            or (frame_index % self.config.ball_detector_interval == 0)
            or self.miss_count > 0
        ):
            candidates.extend(self._detect_with_model(frame, hsv, motion_mask, predicted_center))
        if self.mode in {"TRACK_CONTOUR", "TRACK_BRIDGE"} and predicted_center is not None:
            candidate = self._choose_candidate(candidates, predicted_center, prefer_roi=True)
            escape_candidate = self._choose_candidate(candidates, None, prefer_roi=False, global_only=True)
            if candidate is not None and self._is_suspicious_candidate(candidate):
                candidate = escape_candidate if self._is_strong_global_candidate(escape_candidate) else None
            elif candidate is None and self._is_strong_global_candidate(escape_candidate):
                candidate = escape_candidate
        elif self.mode == "SEARCH_REACQUIRE":
            candidate = self._choose_candidate(candidates, predicted_center, prefer_roi=False, global_only=True)
        else:
            candidate = self._choose_candidate(candidates, predicted_center, prefer_roi=False)
        if candidate is not None:
            if self.mode in {"SEARCH_INIT", "SEARCH_REACQUIRE"}:
                return self._confirm_pending_candidate(gray, candidate)
            return self._confirm_candidate(gray, candidate)

        if self.mode in {"TRACK_CONTOUR", "TRACK_BRIDGE"}:
            flow_state = self._optical_flow_bridge(gray, predicted_center)
            if flow_state is not None:
                return flow_state
        else:
            self.pending_lock = None

        return self._handle_missing(prediction)

    def _predict(self) -> Optional[np.ndarray]:
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return prediction[:2].reshape(-1)

    def _build_motion_mask(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros_like(gray), gray

        delta = cv2.absdiff(gray, self.previous_gray)
        _, motion_mask = cv2.threshold(delta, self.config.ball_motion_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        self.previous_gray = gray
        return motion_mask, gray

    def _find_candidates(
        self,
        frame: np.ndarray,
        hsv: np.ndarray,
        motion_mask: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
        predicted_center: Optional[tuple[int, int]],
    ) -> list[BallCandidate]:
        blurred_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        yellow_mask = cv2.inRange(blurred_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
        blue_mask = cv2.inRange(blurred_hsv, self.config.ball_blue_lower, self.config.ball_blue_upper)
        combined_raw = cv2.bitwise_or(yellow_mask, blue_mask)
        combined_raw = cv2.bitwise_and(combined_raw, self.search_mask)
        suppression_mask = np.zeros_like(self.search_mask)
        for x, y, w, h in self.current_player_boxes:
            cv2.rectangle(suppression_mask, (x, y), (x + w, y + h), 255, -1)
        if cv2.countNonZero(suppression_mask) > 0:
            combined_raw = cv2.bitwise_and(combined_raw, cv2.bitwise_not(suppression_mask))
        candidates: list[BallCandidate] = []

        combined_mask = combined_raw.copy()
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.color_open_kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.color_close_kernel, iterations=1)
        strong_mask = cv2.bitwise_and(combined_mask, motion_mask)
        motion_seed_mask = cv2.bitwise_and(motion_mask, self.search_mask)
        if cv2.countNonZero(suppression_mask) > 0:
            motion_seed_mask = cv2.bitwise_and(motion_seed_mask, cv2.bitwise_not(suppression_mask))
        motion_seed_mask = cv2.morphologyEx(motion_seed_mask, cv2.MORPH_OPEN, self.color_open_kernel, iterations=1)

        if self.mode in {"SEARCH_INIT", "SEARCH_REACQUIRE"} or predicted_center is None or self.miss_count >= self.config.ball_reacquire_after_misses:
            candidates.extend(
                self._candidates_from_mask(
                    combined_mask,
                    combined_raw,
                    hsv,
                    motion_mask,
                    player_boxes,
                    "global_color",
                )
            )
            candidates.extend(
                self._candidates_from_mask(
                    strong_mask,
                    combined_raw,
                    hsv,
                    motion_mask,
                    player_boxes,
                    "global_motion",
                )
            )
            candidates.extend(
                self._motion_seed_candidates(
                    motion_seed_mask,
                    combined_raw,
                    hsv,
                    player_boxes,
                    "global_motion_seed",
                )
            )

        if predicted_center is not None and point_in_mask(predicted_center, self.search_mask):
            roi_size = self._roi_size()
            roi_frame, offset = crop_roi(frame, predicted_center, roi_size)
            if roi_frame.size > 0:
                roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
                roi_motion = motion_mask[offset[1] : offset[1] + roi_frame.shape[0], offset[0] : offset[0] + roi_frame.shape[1]]
                roi_search = self.search_mask[offset[1] : offset[1] + roi_frame.shape[0], offset[0] : offset[0] + roi_frame.shape[1]]
                roi_suppression = suppression_mask[offset[1] : offset[1] + roi_frame.shape[0], offset[0] : offset[0] + roi_frame.shape[1]]
                roi_yellow = cv2.inRange(roi_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
                roi_blue = cv2.inRange(roi_hsv, self.config.ball_blue_lower, self.config.ball_blue_upper)
                roi_raw = cv2.bitwise_and(cv2.bitwise_or(roi_yellow, roi_blue), roi_search)
                if cv2.countNonZero(roi_suppression) > 0:
                    roi_raw = cv2.bitwise_and(roi_raw, cv2.bitwise_not(roi_suppression))
                roi_mask = roi_raw.copy()
                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.color_open_kernel, iterations=1)
                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.color_close_kernel, iterations=1)
                candidates.extend(
                    self._candidates_from_mask(
                        roi_mask,
                        roi_raw,
                        roi_hsv,
                        roi_motion,
                        player_boxes,
                        "roi_color",
                        offset,
                    )
                )
                candidates.extend(
                    self._candidates_from_mask(
                        cv2.bitwise_and(roi_mask, roi_motion),
                        roi_raw,
                        roi_hsv,
                        roi_motion,
                        player_boxes,
                        "roi_motion",
                        offset,
                    )
                )
                candidates.extend(
                    self._motion_seed_candidates(
                        cv2.bitwise_and(cv2.bitwise_and(roi_motion, roi_search), cv2.bitwise_not(roi_suppression)),
                        roi_raw,
                        roi_hsv,
                        player_boxes,
                        "roi_motion_seed",
                        offset,
                    )
                )
        return candidates

    def _candidates_from_mask(
        self,
        mask: np.ndarray,
        context_mask: np.ndarray,
        hsv: np.ndarray,
        motion_mask: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
        source: str,
        offset: tuple[int, int] = (0, 0),
    ) -> list[BallCandidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallCandidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.ball_min_area or area > self.config.ball_max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.config.ball_min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            if (max(w, h) / max(float(min(w, h)), 1.0)) > self.config.ball_max_aspect_ratio:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.config.ball_min_radius or radius > self.config.ball_max_radius:
                continue

            bbox = (
                int(round(x + offset[0])),
                int(round(y + offset[1])),
                int(round(w)),
                int(round(h)),
            )
            center = (int(round(cx + offset[0])), int(round(cy + offset[1])))
            if not point_in_mask(center, self.search_mask):
                continue

            roi = hsv[y : y + h, x : x + w]
            roi_motion = motion_mask[y : y + h, x : x + w]
            pixel_count = float(max(w * h, 1))
            yellow_ratio = cv2.countNonZero(
                cv2.inRange(roi, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / pixel_count
            blue_ratio = cv2.countNonZero(
                cv2.inRange(roi, self.config.ball_blue_lower, self.config.ball_blue_upper)
            ) / pixel_count
            motion_ratio = cv2.countNonZero(roi_motion) / pixel_count
            inside_player = self._inside_player_torso(center, bbox, player_boxes)
            dual_color = (
                yellow_ratio >= self.config.ball_color_ratio_min
                and blue_ratio >= self.config.ball_color_ratio_min
            )
            local_density = self._local_color_density(context_mask, (x, y), radius)
            circle_support = self._has_circle_support(roi, radius)

            combined_color_ratio = yellow_ratio + blue_ratio
            if combined_color_ratio < self.config.ball_min_combined_color_ratio and max(yellow_ratio, blue_ratio) < self.config.ball_min_single_color_ratio:
                continue
            if center[1] < self.config.ball_banner_guard_y and motion_ratio < self.config.ball_banner_motion_ratio:
                continue
            if motion_ratio < self.config.ball_motion_ratio_min and source == "global_motion":
                continue
            if motion_ratio < self.config.ball_global_motion_relaxed_ratio and source == "global_color":
                continue
            if local_density > self.config.ball_local_color_density_max and not dual_color and motion_ratio < (self.config.ball_banner_motion_ratio * 1.2):
                continue

            candidates.append(
                BallCandidate(
                    center=center,
                    bbox=bbox,
                    radius=float(radius),
                    circularity=circularity,
                    yellow_ratio=yellow_ratio,
                    blue_ratio=blue_ratio,
                    motion_ratio=motion_ratio,
                    inside_player=inside_player,
                    source=source,
                    motion_backed="motion" in source,
                    circle_support=circle_support,
                    local_color_density=local_density,
                )
            )
        return candidates

    def _motion_seed_candidates(
        self,
        motion_mask: np.ndarray,
        context_mask: np.ndarray,
        hsv: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
        source: str,
        offset: tuple[int, int] = (0, 0),
    ) -> list[BallCandidate]:
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallCandidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (self.config.ball_min_area * 0.5) or area > (self.config.ball_max_area * 1.45):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if h <= 0 or w <= 0:
                continue
            aspect_ratio = max(w, h) / max(float(min(w, h)), 1.0)
            if aspect_ratio > 2.35:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < (self.config.ball_min_radius * 0.8) or radius > (self.config.ball_max_radius * 1.35):
                continue

            bbox = (
                int(round(x + offset[0])),
                int(round(y + offset[1])),
                int(round(w)),
                int(round(h)),
            )
            center = (int(round(cx + offset[0])), int(round(cy + offset[1])))
            if not point_in_mask(center, self.search_mask):
                continue

            roi = hsv[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            pixel_count = float(max(w * h, 1))
            yellow_ratio = cv2.countNonZero(
                cv2.inRange(roi, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / pixel_count
            blue_ratio = cv2.countNonZero(
                cv2.inRange(roi, self.config.ball_blue_lower, self.config.ball_blue_upper)
            ) / pixel_count
            combined_color_ratio = yellow_ratio + blue_ratio
            if combined_color_ratio < (self.config.ball_min_combined_color_ratio * 0.55) and max(yellow_ratio, blue_ratio) < (self.config.ball_min_single_color_ratio * 0.7):
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else float((4.0 * np.pi * area) / (perimeter * perimeter))
            motion_ratio = 1.0
            inside_player = self._inside_player_torso(center, bbox, player_boxes)
            local_density = self._local_color_density(context_mask, (x, y), radius)
            dual_color = (
                yellow_ratio >= self.config.ball_color_ratio_min
                and blue_ratio >= self.config.ball_color_ratio_min
            )
            circle_support = self._has_circle_support(roi, radius)
            if center[1] < self.config.ball_banner_guard_y and local_density > self.config.ball_local_color_density_max:
                continue
            if local_density > (self.config.ball_local_color_density_max * 1.05) and not dual_color:
                continue

            candidates.append(
                BallCandidate(
                    center=center,
                    bbox=bbox,
                    radius=float(radius),
                    circularity=circularity,
                    yellow_ratio=yellow_ratio,
                    blue_ratio=blue_ratio,
                    motion_ratio=motion_ratio,
                    inside_player=inside_player,
                    source=source,
                    motion_backed=True,
                    circle_support=circle_support,
                    local_color_density=local_density,
                )
            )
        return candidates

    def _detect_with_model(
        self,
        frame: np.ndarray,
        hsv: np.ndarray,
        motion_mask: np.ndarray,
        predicted_center: Optional[tuple[int, int]],
    ) -> list[BallCandidate]:
        if self.detector_model is None:
            return []

        inference_frame = frame
        offset = (0, 0)
        source = "detector_full"
        imgsz = self.config.ball_detector_imgsz_full
        if predicted_center is not None and point_in_mask(predicted_center, self.search_mask):
            inference_frame, offset = crop_roi(frame, predicted_center, int(self._roi_size() * 1.15))
            if inference_frame.size > 0:
                source = "detector_roi"
                imgsz = self.config.ball_detector_imgsz_roi
            else:
                inference_frame = frame
                offset = (0, 0)
        result = self.detector_model.predict(
            source=inference_frame,
            conf=self.config.ball_detector_confidence,
            imgsz=imgsz,
            iou=0.45,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        candidates: list[BallCandidate] = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = clip_bbox(
                (
                    int(round(x1 + offset[0])),
                    int(round(y1 + offset[1])),
                    int(round(x2 - x1)),
                    int(round(y2 - y1)),
                ),
                frame.shape,
            )
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            center = bbox_center(bbox)
            if not point_in_mask(center, self.search_mask):
                continue
            if self._inside_player_torso(center, bbox, self.current_player_boxes):
                continue

            patch_hsv = hsv[y : y + h, x : x + w]
            if patch_hsv.size == 0:
                continue
            pixel_count = float(max(w * h, 1))
            yellow_ratio = cv2.countNonZero(
                cv2.inRange(patch_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / pixel_count
            blue_ratio = cv2.countNonZero(
                cv2.inRange(patch_hsv, self.config.ball_blue_lower, self.config.ball_blue_upper)
            ) / pixel_count
            motion_ratio = cv2.countNonZero(motion_mask[y : y + h, x : x + w]) / pixel_count
            if yellow_ratio + blue_ratio < (self.config.ball_min_combined_color_ratio * 0.45) and motion_ratio < self.config.ball_global_motion_relaxed_ratio:
                continue

            candidates.append(
                BallCandidate(
                    center=center,
                    bbox=bbox,
                    radius=max(float(max(w, h)) / 2.0, 4.0),
                    circularity=1.0,
                    yellow_ratio=yellow_ratio,
                    blue_ratio=blue_ratio,
                    motion_ratio=motion_ratio,
                    inside_player=False,
                    source=source,
                    motion_backed=(motion_ratio >= self.config.ball_global_motion_relaxed_ratio),
                    circle_support=True,
                    local_color_density=yellow_ratio + blue_ratio,
                )
            )
        return candidates

    def _local_color_density(
        self,
        context_mask: np.ndarray,
        local_center: tuple[int, int],
        radius: float,
    ) -> float:
        half = max(self.config.ball_context_window_min // 2, int(round(radius * self.config.ball_context_window_scale)))
        x1 = max(local_center[0] - half, 0)
        y1 = max(local_center[1] - half, 0)
        x2 = min(local_center[0] + half, context_mask.shape[1])
        y2 = min(local_center[1] + half, context_mask.shape[0])
        patch = context_mask[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        return float(cv2.countNonZero(patch) / max(float(patch.shape[0] * patch.shape[1]), 1.0))

    def _inside_player_torso(
        self,
        center: tuple[int, int],
        bbox: tuple[int, int, int, int],
        player_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        for player_box in player_boxes:
            if bbox_iou(bbox, player_box) > 0.14:
                return True
            if player_box[0] <= center[0] <= player_box[0] + player_box[2] and player_box[1] <= center[1] <= player_box[1] + player_box[3]:
                return True
        return False

    def _has_circle_support(self, roi: np.ndarray, radius: float) -> bool:
        if roi.size == 0 or min(roi.shape[:2]) < 6:
            return False
        gray = cv2.cvtColor(roi, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=6,
            param1=60,
            param2=7,
            minRadius=max(2, int(round(radius * 0.55))),
            maxRadius=max(4, int(round(radius * 1.45))),
        )
        return circles is not None and len(circles) > 0

    def _choose_candidate(
        self,
        candidates: list[BallCandidate],
        predicted_center: Optional[tuple[int, int]],
        prefer_roi: bool,
        global_only: bool = False,
    ) -> Optional[BallCandidate]:
        best_candidate: Optional[BallCandidate] = None
        best_score = float("-inf")

        for candidate in candidates:
            if global_only and not candidate.source.startswith("global"):
                continue
            dual_color = (
                candidate.yellow_ratio >= self.config.ball_color_ratio_min
                and candidate.blue_ratio >= self.config.ball_color_ratio_min
            )
            score = candidate.circularity * 2.8
            score += min(candidate.motion_ratio / max(self.config.ball_motion_ratio_min * 2.0, 1e-6), 1.0) * self.config.ball_motion_bonus
            score += max(0.0, 1.0 - (candidate.local_color_density / max(self.config.ball_local_color_density_max, 1e-6))) * 0.9
            if candidate.yellow_ratio >= self.config.ball_min_single_color_ratio:
                score += 0.4
            if candidate.blue_ratio >= self.config.ball_min_single_color_ratio:
                score += 0.4
            if dual_color:
                score += self.config.ball_dual_color_bonus
            if candidate.circle_support:
                score += 1.15
            if candidate.source.startswith("detector"):
                score += 3.2
            if candidate.source.startswith("roi"):
                score += 0.75 if prefer_roi else 0.3
            if candidate.motion_backed:
                score += 0.5

            if predicted_center is not None:
                distance = distance_between(candidate.center, predicted_center)
                if candidate.source == "detector_full":
                    gate = max(self.config.ball_distance_gate, self.config.ball_reacquire_distance_gate * 2.2)
                elif candidate.source == "detector_roi":
                    gate = self.config.ball_reacquire_distance_gate * 1.4
                else:
                    gate = self.config.ball_predicted_distance_gate if candidate.source.startswith("roi") else self.config.ball_reacquire_distance_gate
                if distance > gate:
                    continue
                score += max(0.0, 1.0 - (distance / max(gate, 1.0))) * 2.6
                if candidate.inside_player and distance > self.config.ball_inside_player_gate:
                    continue
            elif candidate.source.startswith("global") and (
                candidate.yellow_ratio < self.config.ball_min_single_color_ratio and candidate.blue_ratio < self.config.ball_min_single_color_ratio
            ):
                continue
            if candidate.inside_player:
                if predicted_center is None and not candidate.motion_backed:
                    continue
                if candidate.local_color_density > (self.config.ball_local_color_density_max * 0.92) and not dual_color:
                    continue
                if candidate.motion_ratio < (self.config.ball_motion_ratio_min * 1.15) and not candidate.motion_backed:
                    continue
                if predicted_center is not None and distance_between(candidate.center, predicted_center) > self.config.ball_predicted_player_gate:
                    continue
                if not candidate.circle_support and not dual_color:
                    continue
                score -= (self.config.ball_inside_player_penalty + 0.8)
            if candidate.local_color_density > self.config.ball_local_color_density_max and not dual_color and not candidate.circle_support:
                continue
            candidate.score_hint = score

            if best_candidate is None or score > best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def _confirm_pending_candidate(self, gray: np.ndarray, candidate: BallCandidate) -> BallState:
        if not candidate.motion_backed and candidate.motion_ratio < self.config.ball_motion_ratio_min:
            return BallState(None, None, "searching")
        if self.pending_lock is None:
            self.pending_lock = PendingBallLock(candidate.center, candidate.radius, frames=1)
            return BallState(None, None, "searching")
        if distance_between(candidate.center, self.pending_lock.center) <= self.config.ball_candidate_match_distance:
            self.pending_lock.center = candidate.center
            self.pending_lock.radius = candidate.radius
            self.pending_lock.frames += 1
        else:
            self.pending_lock = PendingBallLock(candidate.center, candidate.radius, frames=1)
            return BallState(None, None, "searching")
        if self.pending_lock.frames < self.config.ball_confirm_frames:
            return BallState(None, None, "searching")
        self.pending_lock = None
        return self._confirm_candidate(gray, candidate)

    def _is_suspicious_candidate(self, candidate: BallCandidate) -> bool:
        dual_color = (
            candidate.yellow_ratio >= self.config.ball_color_ratio_min
            and candidate.blue_ratio >= self.config.ball_color_ratio_min
        )
        if candidate.inside_player:
            return True
        if candidate.local_color_density > self.config.ball_local_color_density_max and not candidate.circle_support:
            return True
        if not candidate.motion_backed and not candidate.circle_support and not dual_color:
            return True
        return False

    def _is_strong_global_candidate(self, candidate: Optional[BallCandidate]) -> bool:
        if candidate is None or not (candidate.source.startswith("global") or candidate.source == "detector_full"):
            return False
        dual_color = (
            candidate.yellow_ratio >= self.config.ball_color_ratio_min
            and candidate.blue_ratio >= self.config.ball_color_ratio_min
        )
        if candidate.inside_player:
            return False
        if candidate.center[1] < self.config.ball_banner_guard_y and candidate.motion_ratio < self.config.ball_banner_motion_ratio:
            return False
        if candidate.local_color_density > self.config.ball_local_color_density_max and not candidate.circle_support:
            return False
        return candidate.motion_backed and (candidate.circle_support or dual_color or candidate.motion_ratio >= 0.3)

    def _confirm_candidate(self, gray: np.ndarray, candidate: BallCandidate) -> BallState:
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        if not self.initialized:
            self.kalman.statePost = np.array(
                [[np.float32(candidate.center[0])], [np.float32(candidate.center[1])], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.initialized = True
        self.kalman.correct(measurement)

        if self.last_confirmed_center is not None:
            jump_distance = distance_between(candidate.center, self.last_confirmed_center)
            if jump_distance < self.config.ball_min_progress_pixels and (
                candidate.inside_player or candidate.local_color_density > self.config.ball_local_color_density_max or not candidate.motion_backed
            ):
                self.stale_lock_frames += 1
                if self.stale_lock_frames >= self.config.ball_stale_lock_frames:
                    self.mode = "SEARCH_REACQUIRE"
                    self.miss_count = self.config.ball_reacquire_after_misses
                    self.pending_lock = None
                    self.previous_ball_points = None
                    self.previous_ball_gray = None
                    return BallState(None, None, "missing")
            else:
                self.stale_lock_frames = 0
            if self.miss_count > 2 or jump_distance > self.config.ball_reacquire_distance_gate:
                self.trail.clear()
        else:
            self.stale_lock_frames = 0

        self.last_confirmed_center = candidate.center
        self.last_confirmed_radius = candidate.radius
        self.miss_count = 0
        self.bridge_count = 0
        self.mode = "TRACK_CONTOUR"
        self.trail.append(candidate.center)
        self._refresh_optical_flow(gray, candidate)
        return BallState(candidate.center, max(4.0, candidate.radius), "confirmed")

    def _refresh_optical_flow(self, gray: np.ndarray, candidate: BallCandidate) -> None:
        x, y, w, h = candidate.bbox
        patch = gray[y : y + h, x : x + w]
        if patch.size == 0:
            self.previous_ball_gray = None
            self.previous_ball_points = None
            return
        points = cv2.goodFeaturesToTrack(
            patch,
            maxCorners=8,
            qualityLevel=self.config.ball_optical_flow_quality,
            minDistance=self.config.ball_optical_flow_min_distance,
            blockSize=self.config.ball_optical_flow_block_size,
        )
        if points is None:
            self.previous_ball_gray = gray.copy()
            self.previous_ball_points = None
            return
        points[:, 0, 0] += x
        points[:, 0, 1] += y
        self.previous_ball_gray = gray.copy()
        self.previous_ball_points = points

    def _optical_flow_bridge(
        self,
        gray: np.ndarray,
        predicted_center: Optional[tuple[int, int]],
    ) -> Optional[BallState]:
        if (
            self.previous_ball_gray is None
            or self.previous_ball_points is None
            or self.last_confirmed_center is None
            or self.bridge_count >= self.config.ball_optical_flow_max_frames
        ):
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_ball_gray,
            gray,
            self.previous_ball_points,
            None,
            winSize=self.config.ball_optical_flow_win_size,
            maxLevel=self.config.ball_optical_flow_max_level,
        )
        if next_points is None or status is None:
            return None

        valid = next_points[status.flatten() == 1]
        if len(valid) == 0:
            return None
        valid = valid.reshape(-1, 2)

        center = (int(round(float(np.mean(valid[:, 0])))), int(round(float(np.mean(valid[:, 1])))))
        if not point_in_mask(center, self.search_mask):
            return None
        if self._inside_player_torso(center, (center[0] - 6, center[1] - 6, 12, 12), self.current_player_boxes):
            return None
        if predicted_center is not None and distance_between(center, predicted_center) > self.config.ball_predicted_distance_gate:
            return None
        if self.last_confirmed_center is not None and distance_between(center, self.last_confirmed_center) > self.config.ball_reacquire_distance_gate:
            return None

        self.previous_ball_gray = gray.copy()
        self.previous_ball_points = valid.reshape(-1, 1, 2).astype(np.float32)
        self.bridge_count += 1
        self.miss_count += 1
        self.mode = "TRACK_BRIDGE"
        return BallState(center, max(4.0, self.last_confirmed_radius), "predicted")

    def _handle_missing(self, prediction: Optional[np.ndarray]) -> BallState:
        self.miss_count += 1
        if prediction is None:
            self.mode = "SEARCH_REACQUIRE"
            self.bridge_count = 0
            self.pending_lock = None
            self.previous_ball_points = None
            self.previous_ball_gray = None
            self.stale_lock_frames = 0
            return BallState(None, None, "missing")

        center = (int(round(prediction[0])), int(round(prediction[1])))
        uncertainty = float(np.trace(self.kalman.errorCovPre[:2, :2]))
        if (
            self.miss_count <= self.config.ball_predicted_draw_misses
            and uncertainty <= self.config.ball_max_uncertainty
            and point_in_mask(center, self.search_mask)
            and center[1] >= self.config.ball_banner_guard_y
            and not self._inside_player_torso(center, (center[0] - 8, center[1] - 8, 16, 16), self.current_player_boxes)
            and self.mode != "SEARCH_REACQUIRE"
        ):
            self.mode = "TRACK_BRIDGE"
            return BallState(center, max(4.0, self.last_confirmed_radius), "predicted")

        if self.miss_count > self.config.ball_max_misses or uncertainty > self.config.ball_max_uncertainty:
            self.initialized = False
            self.last_confirmed_center = None
            self.previous_ball_points = None
            self.previous_ball_gray = None
            self.pending_lock = None
            self.stale_lock_frames = 0
            self.trail.clear()
        self.mode = "SEARCH_REACQUIRE"
        self.bridge_count = 0
        return BallState(None, None, "missing")

    def _roi_size(self) -> int:
        velocity = 0.0
        if self.initialized:
            state = self.kalman.statePost.reshape(-1)
            velocity = float(np.hypot(state[2], state[3]))
        roi_size = int(
            self.config.ball_base_roi
            + velocity * self.config.ball_velocity_roi_gain
            + self.miss_count * self.config.ball_miss_roi_gain
        )
        return int(np.clip(roi_size, self.config.ball_base_roi, self.config.ball_max_roi))


def draw_players(frame: np.ndarray, players: list[PlayerTrack]) -> None:
    for player in players:
        x, y, w, h = player.bbox
        if w < 18 or h < 28:
            continue
        color = (0, 220, 255) if player.stable_team == "team_a" else (255, 180, 60)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, state: BallState) -> None:
    if state.center is None or state.radius is None:
        return
    color = (0, 235, 0) if state.status == "confirmed" else (0, 170, 255)
    cv2.circle(frame, state.center, max(5, int(round(state.radius))), color, 2, cv2.LINE_AA)
    cv2.circle(frame, state.center, 2, color, -1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: deque[tuple[int, int]]) -> None:
    if len(trail) < 2:
        return
    points = list(trail)
    for index in range(1, len(points)):
        thickness = max(1, int(np.interp(index, [1, len(points) - 1], [5, 1])))
        cv2.line(frame, points[index - 1], points[index], (0, 220, 255), thickness, cv2.LINE_AA)


def overlay_status(frame: np.ndarray, team_a_count: int, team_b_count: int, fps_value: float) -> None:
    cv2.rectangle(frame, (20, 20), (250, 110), (24, 24, 24), -1)
    cv2.putText(frame, f"Team A: {team_a_count}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Team B: {team_b_count}", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 60), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_value:4.1f}", (35, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    phase_root = Path(__file__).resolve().parents[1]
    phase6_dir = phase_root / "task phase 6"
    parser = argparse.ArgumentParser(description="OpenCV-first volleyball tracker v2 for Task 6.")
    parser.add_argument("--input", type=Path, default=phase6_dir / "Volleyball.mp4", help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Volleyball_annotated_opencv_v2.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--ball-model",
        type=Path,
        default=phase6_dir / "models" / "ball_best.pt",
        help="Optional sparse ball-detector weights used only for correction when OpenCV tracking is lost.",
    )
    parser.add_argument("--display", action="store_true", help="Display frames during processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SceneConfig()

    capture = cv2.VideoCapture(str(args.input), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {args.output}")

    ball_search_mask = apply_exclusions(
        build_mask((height, width), config.ball_search_polygon),
        config.ball_exclusion_rects,
    )
    top_court_mask = apply_exclusions(
        build_mask((height, width), config.top_court_polygon),
        config.player_exclusion_rects,
    )
    bottom_court_mask = apply_exclusions(
        build_mask((height, width), config.bottom_court_polygon),
        config.player_exclusion_rects,
    )

    player_tracker = PlayerTrackerOpenCVV2(config, top_court_mask, bottom_court_mask)
    detector_model = None
    if YOLO is not None and args.ball_model.exists():
        detector_model = YOLO(str(args.ball_model))
    ball_tracker = BallTrackerOpenCVV2(config, ball_search_mask, detector_model)

    fps_estimate = fps
    last_tick = time.perf_counter()
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            active_players, team_a_count, team_b_count = player_tracker.update(frame, frame_index)
            player_boxes = player_tracker.ball_suppression_boxes
            ball_state = ball_tracker.update(frame, frame_index, player_boxes)

            annotated = frame.copy()
            draw_players(annotated, active_players)
            draw_trail(annotated, ball_tracker.trail)
            draw_ball(annotated, ball_state)

            now = time.perf_counter()
            frame_elapsed = now - last_tick
            if frame_elapsed > 0:
                fps_estimate = (fps_estimate * 0.9) + ((1.0 / frame_elapsed) * 0.1)
            last_tick = now

            overlay_status(annotated, team_a_count, team_b_count, fps_estimate)
            writer.write(annotated)

            if args.display:
                cv2.imshow("Volleyball Tracking OpenCV V2", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
