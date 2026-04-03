from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SceneConfig:
    trail_length: int = 22
    max_players_per_team: int = 6
    player_track_ttl: int = 24
    player_active_ttl: int = 12
    player_zone_ttl: int = 10
    player_vote_gain: float = 1.0
    player_vote_decay: float = 0.05
    player_stable_threshold: float = 2.8
    player_track_confidence_gain: float = 1.0
    player_track_confidence_decay: float = 0.35
    player_track_confidence_floor: float = 1.2
    player_min_track_age_for_count: int = 3
    team_color_floor_ratio: float = 0.012
    team_color_margin: float = 0.004
    torso_x_margin_ratio: float = 0.32
    torso_y_start_ratio: float = 0.16
    torso_y_end_ratio: float = 0.42
    bg_warmup_frames: int = 20
    bg_alpha_warmup: float = 0.08
    bg_alpha_top: float = 0.012
    bg_alpha_bottom: float = 0.01
    player_fg_threshold_top: int = 18
    player_fg_threshold_bottom: int = 22
    top_player_area_min: int = 650
    top_player_area_max: int = 42000
    bottom_player_area_min: int = 1350
    bottom_player_area_max: int = 72000
    player_min_height_top: int = 30
    player_min_height_bottom: int = 56
    player_min_width: int = 14
    player_max_width: int = 175
    player_max_aspect_ratio: float = 1.22
    player_min_fill_ratio: float = 0.18
    player_match_distance_top: float = 70.0
    player_match_distance_bottom: float = 95.0
    player_iou_match_threshold: float = 0.05
    player_zone_cols: int = 3
    player_zone_rows: int = 2
    player_split_width_top: int = 88
    player_split_width_bottom: int = 124
    player_split_height_top: int = 116
    player_split_height_bottom: int = 184
    player_split_min_width: int = 24
    player_top_open_kernel: tuple[int, int] = (3, 3)
    player_top_close_kernel: tuple[int, int] = (7, 9)
    player_top_erode_kernel: tuple[int, int] = (3, 3)
    player_top_dilate_kernel: tuple[int, int] = (5, 9)
    player_bottom_open_kernel: tuple[int, int] = (3, 3)
    player_bottom_close_kernel: tuple[int, int] = (9, 11)
    player_bottom_erode_kernel: tuple[int, int] = (3, 3)
    player_bottom_dilate_kernel: tuple[int, int] = (7, 11)
    player_seed_open_kernel: tuple[int, int] = (3, 3)
    player_seed_close_kernel: tuple[int, int] = (9, 9)
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
    ball_recover_motion_threshold: int = 16
    ball_min_area: float = 8.0
    ball_max_area: float = 250.0
    ball_min_radius: float = 2.0
    ball_max_radius: float = 10.5
    ball_min_circularity: float = 0.28
    ball_max_aspect_ratio: float = 1.8
    ball_min_motion_ratio: float = 0.025
    ball_banner_guard_y: int = 150
    ball_banner_motion_ratio: float = 0.04
    ball_confirm_distance: float = 92.0
    ball_min_progress_pixels: float = 5.0
    ball_track_gate: float = 96.0
    ball_recover_gate: float = 180.0
    ball_coast_frames: int = 4
    ball_recover_after_misses: int = 4
    ball_lost_after_misses: int = 10
    ball_stationary_pixels: float = 4.0
    ball_stationary_frames: int = 3
    ball_player_penalty_gate: float = 28.0
    ball_player_penalty_scale: float = 4.5
    ball_roi_base: int = 88
    ball_roi_miss_gain: int = 24
    ball_roi_max: int = 220
    ball_max_uncertainty: float = 2400.0
    ball_flow_max_frames: int = 2
    ball_flow_win_size: tuple[int, int] = (21, 21)
    ball_flow_quality: float = 0.01
    ball_flow_min_distance: int = 2
    ball_flow_block_size: int = 5
    ball_yellow_lower: np.ndarray = field(
        default_factory=lambda: np.array((12, 85, 75), dtype=np.uint8)
    )
    ball_yellow_upper: np.ndarray = field(
        default_factory=lambda: np.array((42, 255, 255), dtype=np.uint8)
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
class BallObservation:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    radius: float
    area: float
    circularity: float
    compactness: float
    motion_ratio: float
    weak_yellow_ratio: float
    source: str


@dataclass
class PlayerBlob:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    footpoint: tuple[int, int]
    side: str
    area: float
    fill_ratio: float
    solidity: float
    zone_hints: int = 1


@dataclass
class FrameObservation:
    ball_candidates: list[BallObservation]
    recovery_candidates: list[BallObservation]
    player_blobs: list[PlayerBlob]


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


@dataclass
class BallState:
    center: Optional[tuple[int, int]]
    radius: Optional[float]
    status: str


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


def player_ball_suppression_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (x, y, w, max(14, int(round(h * 0.72))))


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


class ObservationCollector:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        mask_shape = frame_shape[:2]
        self.ball_mask = apply_exclusions(build_mask(mask_shape, config.ball_search_polygon), config.ball_exclusion_rects)
        self.top_mask = apply_exclusions(build_mask(mask_shape, config.top_court_polygon), config.player_exclusion_rects)
        self.bottom_mask = apply_exclusions(build_mask(mask_shape, config.bottom_court_polygon), config.player_exclusion_rects)
        self.prev_gray: Optional[np.ndarray] = None
        self.bg_top: Optional[np.ndarray] = None
        self.bg_bottom: Optional[np.ndarray] = None
        self.ball_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.ball_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.top_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_open_kernel)
        self.top_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_close_kernel)
        self.top_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_erode_kernel)
        self.top_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_top_dilate_kernel)
        self.bottom_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_open_kernel)
        self.bottom_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_close_kernel)
        self.bottom_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_erode_kernel)
        self.bottom_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_bottom_dilate_kernel)
        self.seed_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_seed_open_kernel)
        self.seed_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_seed_close_kernel)

    def collect(self, input_path: Path) -> tuple[list[FrameObservation], float, int, int]:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        observations: list[FrameObservation] = []
        frame_index = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            motion_mask = self._build_ball_motion_mask(gray)
            ball_candidates = self._extract_ball_candidates(motion_mask, hsv)
            recovery_candidates = self._extract_recovery_candidates(hsv, motion_mask)
            top_fg = self._extract_side_foreground(gray, self.top_mask, "top", frame_index)
            bottom_fg = self._extract_side_foreground(gray, self.bottom_mask, "bottom", frame_index)
            motion_blobs = self._extract_player_blobs(top_fg, "top") + self._extract_player_blobs(bottom_fg, "bottom")
            seed_blobs = self._extract_color_seed_blobs(hsv)
            player_blobs = self._merge_player_blobs(motion_blobs, seed_blobs)
            observations.append(
                FrameObservation(
                    ball_candidates=ball_candidates,
                    recovery_candidates=recovery_candidates,
                    player_blobs=player_blobs,
                )
            )
            self.prev_gray = gray
            frame_index += 1

        capture.release()
        return observations, fps, width, height

    def _build_ball_motion_mask(self, gray: np.ndarray) -> np.ndarray:
        if self.prev_gray is None:
            return np.zeros_like(gray)
        diff = cv2.absdiff(gray, self.prev_gray)
        _, motion_mask = cv2.threshold(diff, self.config.ball_motion_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.ball_open_kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.ball_close_kernel, iterations=1)
        return cv2.bitwise_and(motion_mask, self.ball_mask)

    def _extract_ball_candidates(self, motion_mask: np.ndarray, hsv: np.ndarray) -> list[BallObservation]:
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallObservation] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.ball_min_area or area > self.config.ball_max_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0.0 else float((4.0 * np.pi * area) / (perimeter * perimeter))
            x, y, w, h = clip_bbox(cv2.boundingRect(contour), self.frame_shape)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = w / max(float(h), 1.0)
            if aspect_ratio > self.config.ball_max_aspect_ratio or (1.0 / max(aspect_ratio, 1e-6)) > self.config.ball_max_aspect_ratio:
                continue
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.config.ball_min_radius or radius > self.config.ball_max_radius:
                continue
            compactness = area / max(float(w * h), 1.0)
            center = bbox_center((x, y, w, h))
            if not point_in_mask(center, self.ball_mask):
                continue
            bbox = (x, y, w, h)
            patch_motion = motion_mask[y : y + h, x : x + w]
            motion_ratio = cv2.countNonZero(patch_motion) / max(float(w * h), 1.0)
            patch_hsv = hsv[y : y + h, x : x + w]
            weak_yellow_ratio = cv2.countNonZero(
                cv2.inRange(patch_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / max(float(w * h), 1.0)
            candidates.append(
                BallObservation(
                    center=center,
                    bbox=bbox,
                    radius=float(radius),
                    area=float(area),
                    circularity=circularity,
                    compactness=compactness,
                    motion_ratio=motion_ratio,
                    weak_yellow_ratio=weak_yellow_ratio,
                    source="motion",
                )
            )
        return candidates

    def _extract_recovery_candidates(self, hsv: np.ndarray, motion_mask: np.ndarray) -> list[BallObservation]:
        yellow_mask = cv2.inRange(hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
        yellow_mask = cv2.bitwise_and(yellow_mask, self.ball_mask)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.ball_open_kernel, iterations=1)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, self.ball_close_kernel, iterations=1)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallObservation] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.ball_min_area or area > self.config.ball_max_area:
                continue
            x, y, w, h = clip_bbox(cv2.boundingRect(contour), self.frame_shape)
            if w <= 0 or h <= 0:
                continue
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.config.ball_min_radius or radius > self.config.ball_max_radius:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0.0 else float((4.0 * np.pi * area) / (perimeter * perimeter))
            center = (int(round(circle_x)), int(round(circle_y)))
            if not point_in_mask(center, self.ball_mask):
                continue
            patch_motion = motion_mask[y : y + h, x : x + w]
            motion_ratio = cv2.countNonZero(patch_motion) / max(float(w * h), 1.0)
            weak_yellow_ratio = cv2.countNonZero(yellow_mask[y : y + h, x : x + w]) / max(float(w * h), 1.0)
            if weak_yellow_ratio < 0.04:
                continue
            candidates.append(
                BallObservation(
                    center=center,
                    bbox=(x, y, w, h),
                    radius=float(radius),
                    area=float(area),
                    circularity=circularity,
                    compactness=area / max(float(w * h), 1.0),
                    motion_ratio=motion_ratio,
                    weak_yellow_ratio=weak_yellow_ratio,
                    source="recovery",
                )
            )
        return candidates

    def _extract_side_foreground(
        self,
        gray: np.ndarray,
        side_mask: np.ndarray,
        side: str,
        frame_index: int,
    ) -> np.ndarray:
        masked_gray = cv2.bitwise_and(gray, gray, mask=side_mask)
        if side == "top":
            if self.bg_top is None:
                self.bg_top = masked_gray.astype(np.float32)
            bg_model = self.bg_top
            threshold = self.config.player_fg_threshold_top
            open_kernel = self.top_open_kernel
            close_kernel = self.top_close_kernel
            erode_kernel = self.top_erode_kernel
            dilate_kernel = self.top_dilate_kernel
            alpha = self.config.bg_alpha_top
        else:
            if self.bg_bottom is None:
                self.bg_bottom = masked_gray.astype(np.float32)
            bg_model = self.bg_bottom
            threshold = self.config.player_fg_threshold_bottom
            open_kernel = self.bottom_open_kernel
            close_kernel = self.bottom_close_kernel
            erode_kernel = self.bottom_erode_kernel
            dilate_kernel = self.bottom_dilate_kernel
            alpha = self.config.bg_alpha_bottom

        alpha_to_use = self.config.bg_alpha_warmup if frame_index < self.config.bg_warmup_frames else alpha
        bg_uint8 = cv2.convertScaleAbs(bg_model)
        diff = cv2.absdiff(masked_gray, bg_uint8)
        _, fg_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, side_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        fg_mask = cv2.erode(fg_mask, erode_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)
        if frame_index < self.config.bg_warmup_frames:
            cv2.accumulateWeighted(masked_gray, bg_model, alpha_to_use, mask=side_mask)
            return np.zeros_like(fg_mask)

        update_mask = cv2.bitwise_and(side_mask, cv2.bitwise_not(fg_mask))
        cv2.accumulateWeighted(masked_gray, bg_model, alpha_to_use, mask=update_mask)
        return fg_mask

    def _extract_player_blobs(self, fg_mask: np.ndarray, side: str) -> list[PlayerBlob]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[PlayerBlob] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            bbox = clip_bbox(cv2.boundingRect(contour), self.frame_shape)
            x, y, w, h = bbox
            if w < self.config.player_min_width or w > self.config.player_max_width:
                continue
            if h <= 0 or w / max(float(h), 1.0) > self.config.player_max_aspect_ratio:
                continue
            if side == "top":
                min_height = self.config.player_min_height_top
                area_min = self.config.top_player_area_min
                area_max = self.config.top_player_area_max
                side_mask = self.top_mask
            else:
                min_height = self.config.player_min_height_bottom
                area_min = self.config.bottom_player_area_min
                area_max = self.config.bottom_player_area_max
                side_mask = self.bottom_mask
            if h < min_height or area < area_min or area > area_max:
                continue
            fill_ratio = area / max(float(w * h), 1.0)
            if fill_ratio < self.config.player_min_fill_ratio or solidity < 0.18:
                continue
            footpoint = bbox_footpoint(bbox)
            if not point_in_mask(footpoint, side_mask):
                continue
            blobs.append(
                PlayerBlob(
                    bbox=bbox,
                    center=bbox_center(bbox),
                    footpoint=footpoint,
                    side=side,
                    area=float(area),
                    fill_ratio=fill_ratio,
                    solidity=solidity,
                    zone_hints=self._split_count_hint(bbox, side),
                )
            )
        return blobs

    def _extract_color_seed_blobs(self, hsv: np.ndarray) -> list[PlayerBlob]:
        top_seed = cv2.bitwise_and(
            cv2.inRange(hsv, self.config.yellow_player_lower, self.config.yellow_player_upper),
            self.top_mask,
        )
        bottom_seed = cv2.bitwise_and(
            cv2.inRange(hsv, self.config.blue_player_lower, self.config.blue_player_upper),
            self.bottom_mask,
        )
        top_seed = cv2.morphologyEx(top_seed, cv2.MORPH_OPEN, self.seed_open_kernel, iterations=1)
        top_seed = cv2.morphologyEx(top_seed, cv2.MORPH_CLOSE, self.seed_close_kernel, iterations=1)
        bottom_seed = cv2.morphologyEx(bottom_seed, cv2.MORPH_OPEN, self.seed_open_kernel, iterations=1)
        bottom_seed = cv2.morphologyEx(bottom_seed, cv2.MORPH_CLOSE, self.seed_close_kernel, iterations=1)
        return self._extract_seed_blobs(top_seed, "top") + self._extract_seed_blobs(bottom_seed, "bottom")

    def _extract_seed_blobs(self, seed_mask: np.ndarray, side: str) -> list[PlayerBlob]:
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
                self.frame_shape,
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
                    area=float(area),
                    fill_ratio=0.3,
                    solidity=0.3,
                    zone_hints=1,
                )
            )
        return blobs

    def _merge_player_blobs(
        self,
        motion_blobs: list[PlayerBlob],
        seed_blobs: list[PlayerBlob],
    ) -> list[PlayerBlob]:
        merged: list[PlayerBlob] = list(motion_blobs)
        for blob in seed_blobs:
            if any(
                existing.side == blob.side
                and (bbox_iou(existing.bbox, blob.bbox) > 0.12 or distance_between(existing.center, blob.center) < 44.0)
                for existing in motion_blobs
            ):
                continue
            replaced = False
            for index, existing in enumerate(merged):
                if existing.side != blob.side:
                    continue
                if bbox_iou(existing.bbox, blob.bbox) > 0.38 or distance_between(existing.center, blob.center) < 24.0:
                    existing_area = existing.bbox[2] * existing.bbox[3]
                    blob_area = blob.bbox[2] * blob.bbox[3]
                    if blob_area > existing_area:
                        merged[index] = blob
                    replaced = True
                    break
            if not replaced:
                merged.append(blob)
        return merged

    def _split_count_hint(self, bbox: tuple[int, int, int, int], side: str) -> int:
        x, y, w, h = bbox
        split_width = self.config.player_split_width_top if side == "top" else self.config.player_split_width_bottom
        split_height = self.config.player_split_height_top if side == "top" else self.config.player_split_height_bottom
        if w < split_width and h < split_height:
            return 1
        if w >= split_width * 1.8:
            return 3
        if w >= split_width or h >= split_height:
            return 2
        return 1


class PlayerTrackerOpenCVV3:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        mask_shape = frame_shape[:2]
        self.frame_shape = frame_shape
        self.top_mask = apply_exclusions(build_mask(mask_shape, config.top_court_polygon), config.player_exclusion_rects)
        self.bottom_mask = apply_exclusions(build_mask(mask_shape, config.bottom_court_polygon), config.player_exclusion_rects)
        self.top_zone_boxes = self._build_zone_boxes(config.top_court_polygon)
        self.bottom_zone_boxes = self._build_zone_boxes(config.bottom_court_polygon)
        self.tracks: dict[int, PlayerTrack] = {}
        self.next_track_id = 1
        self.side_zone_memory: dict[str, dict[int, int]] = {"top": {}, "bottom": {}}
        self.last_counts: Optional[tuple[int, int]] = None

    def update(
        self,
        frame: np.ndarray,
        raw_blobs: list[PlayerBlob],
        frame_index: int,
    ) -> tuple[list[PlayerTrack], int, int, list[tuple[int, int, int, int]]]:
        predictions = self._predict_tracks()
        blobs = self._expand_merged_blobs(raw_blobs, predictions)
        matches, unmatched_blob_ids, unmatched_track_ids = self._associate(blobs, predictions)

        for blob_index, track_id in matches:
            self._update_track(self.tracks[track_id], blobs[blob_index], frame, frame_index)

        for blob_index in unmatched_blob_ids:
            self._create_track(blobs[blob_index], frame, frame_index)

        for track_id in unmatched_track_ids:
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            pred = predictions[track_id]
            track.center = pred
            track.footpoint = (pred[0], track.footpoint[1])
            track.bbox = self._bbox_from_center(track.bbox, pred)
            track.misses += 1
            track.confidence = max(0.0, track.confidence - self.config.player_track_confidence_decay)
            track.zone_index = self._zone_for_point(track.side, track.footpoint)

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.misses > self.config.player_track_ttl or frame_index - track.last_seen > self.config.player_track_ttl
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        active_tracks = [
            track
            for track in self.tracks.values()
            if track.misses <= self.config.player_active_ttl and track.confidence >= self.config.player_track_confidence_floor
        ]
        active_tracks = self._select_visible_tracks(active_tracks)
        team_a_count, team_b_count = self._count_from_tracks(active_tracks, blobs)
        suppression_boxes = self._build_ball_suppression_boxes(active_tracks, blobs)
        return active_tracks, team_a_count, team_b_count, suppression_boxes

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

    def _predict_tracks(self) -> dict[int, tuple[int, int]]:
        predictions: dict[int, tuple[int, int]] = {}
        for track_id, track in self.tracks.items():
            prediction = track.kalman.predict()
            predictions[track_id] = (int(round(prediction[0][0])), int(round(prediction[1][0])))
        return predictions

    def _expand_merged_blobs(
        self,
        raw_blobs: list[PlayerBlob],
        predictions: dict[int, tuple[int, int]],
    ) -> list[PlayerBlob]:
        expanded: list[PlayerBlob] = []
        for blob in raw_blobs:
            if blob.zone_hints <= 1:
                expanded.append(blob)
                continue
            anchors = self._anchors_for_blob(blob, predictions)
            split_count = min(blob.zone_hints, len(anchors)) if anchors else blob.zone_hints
            split_count = max(1, min(split_count, 3))
            if split_count == 1:
                expanded.append(blob)
                continue
            centers_x = sorted(anchor[0] for anchor in anchors[:split_count])
            if not centers_x:
                step = blob.bbox[2] / split_count
                centers_x = [int(round(blob.bbox[0] + ((index + 0.5) * step))) for index in range(split_count)]
            boundaries = [blob.bbox[0]]
            for left, right in zip(centers_x, centers_x[1:]):
                boundaries.append(int(round((left + right) * 0.5)))
            boundaries.append(blob.bbox[0] + blob.bbox[2])
            for index in range(split_count):
                x1 = boundaries[index]
                x2 = boundaries[index + 1]
                if x2 - x1 < self.config.player_split_min_width:
                    continue
                split_bbox = clip_bbox((x1, blob.bbox[1], x2 - x1, blob.bbox[3]), self.frame_shape)
                expanded.append(
                    PlayerBlob(
                        bbox=split_bbox,
                        center=bbox_center(split_bbox),
                        footpoint=bbox_footpoint(split_bbox),
                        side=blob.side,
                        area=blob.area / split_count,
                        fill_ratio=blob.fill_ratio,
                        solidity=blob.solidity,
                        zone_hints=1,
                    )
                )
        return expanded

    def _anchors_for_blob(
        self,
        blob: PlayerBlob,
        predictions: dict[int, tuple[int, int]],
    ) -> list[tuple[int, int]]:
        anchors: list[tuple[int, int]] = []
        x, y, w, h = blob.bbox
        for track_id, track in self.tracks.items():
            if track.side != blob.side:
                continue
            pred = predictions.get(track_id, track.center)
            if x <= pred[0] <= x + w and y <= pred[1] <= y + h + 30:
                anchors.append(pred)
        if len(anchors) >= blob.zone_hints:
            return sorted(anchors, key=lambda point: point[0])
        zone_boxes = self.top_zone_boxes if blob.side == "top" else self.bottom_zone_boxes
        for zone_box in zone_boxes:
            zx, zy, zw, zh = zone_box
            zone_center = (zx + (zw // 2), zy + (zh // 2))
            if x <= zone_center[0] <= x + w and y <= zone_center[1] <= y + h:
                anchors.append(zone_center)
        if anchors:
            deduped: list[tuple[int, int]] = []
            for point in sorted(anchors, key=lambda item: item[0]):
                if not deduped or abs(point[0] - deduped[-1][0]) > 12:
                    deduped.append(point)
            return deduped
        return []

    def _associate(
        self,
        blobs: list[PlayerBlob],
        predictions: dict[int, tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        matches: list[tuple[int, int]] = []
        unmatched_blob_ids = set(range(len(blobs)))
        unmatched_track_ids = set(self.tracks.keys())
        while True:
            best_pair: Optional[tuple[int, int]] = None
            best_score = float("inf")
            for blob_index in list(unmatched_blob_ids):
                blob = blobs[blob_index]
                for track_id in list(unmatched_track_ids):
                    track = self.tracks[track_id]
                    if track.side != blob.side:
                        continue
                    match_distance = (
                        self.config.player_match_distance_top if blob.side == "top" else self.config.player_match_distance_bottom
                    )
                    predicted_center = predictions.get(track_id, track.center)
                    center_distance = distance_between(blob.center, predicted_center)
                    foot_distance = distance_between(blob.footpoint, track.footpoint)
                    iou = bbox_iou(blob.bbox, track.bbox)
                    if center_distance > match_distance and iou < self.config.player_iou_match_threshold:
                        continue
                    score = foot_distance + (center_distance * 0.6) - (iou * 140.0)
                    if score < best_score:
                        best_score = score
                        best_pair = (blob_index, track_id)
            if best_pair is None:
                break
            blob_index, track_id = best_pair
            matches.append((blob_index, track_id))
            unmatched_blob_ids.discard(blob_index)
            unmatched_track_ids.discard(track_id)
        return matches, unmatched_blob_ids, unmatched_track_ids

    def _create_track(self, blob: PlayerBlob, frame: np.ndarray, frame_index: int) -> None:
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
        track.zone_index = self._zone_for_point(track.side, track.footpoint)
        self._update_votes(track, frame)
        self.tracks[track.track_id] = track
        self.next_track_id += 1

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
        track.zone_index = self._zone_for_point(track.side, track.footpoint)
        self._update_votes(track, frame)

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

    def _bbox_from_center(self, bbox: tuple[int, int, int, int], center: tuple[int, int]) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        new_x = center[0] - (w // 2)
        new_y = center[1] - (h // 2)
        return clip_bbox((new_x, new_y, w, h), self.frame_shape)

    def _zone_for_point(self, side: str, point: tuple[int, int]) -> Optional[int]:
        boxes = self.top_zone_boxes if side == "top" else self.bottom_zone_boxes
        for index, box in enumerate(boxes):
            x, y, w, h = box
            if x <= point[0] <= x + w and y <= point[1] <= y + h:
                return index
        if not boxes:
            return None
        distances = [distance_between(point, (box[0] + box[2] // 2, box[1] + box[3] // 2)) for box in boxes]
        return int(np.argmin(distances))

    def _zones_for_blob(self, blob: PlayerBlob) -> set[int]:
        zones: set[int] = set()
        zone_boxes = self.top_zone_boxes if blob.side == "top" else self.bottom_zone_boxes
        bx, by, bw, bh = blob.bbox
        for index, box in enumerate(zone_boxes):
            if bbox_iou(blob.bbox, box) > 0.08:
                zones.add(index)
        if not zones:
            zone = self._zone_for_point(blob.side, blob.footpoint)
            if zone is not None:
                zones.add(zone)
        if blob.zone_hints > 1 and len(zones) < blob.zone_hints:
            zone = self._zone_for_point(blob.side, blob.footpoint)
            if zone is not None:
                zones.add(zone)
                if zone - 1 >= 0:
                    zones.add(zone - 1)
                if zone + 1 < self.config.player_zone_cols * self.config.player_zone_rows:
                    zones.add(zone + 1)
        return zones

    def _select_visible_tracks(self, active_tracks: list[PlayerTrack]) -> list[PlayerTrack]:
        visible: list[PlayerTrack] = []
        for side in ("top", "bottom"):
            side_tracks = [track for track in active_tracks if track.side == side]
            side_tracks.sort(
                key=lambda track: (
                    track.confidence,
                    track.age,
                    -track.misses,
                    track.last_seen,
                ),
                reverse=True,
            )
            used_zones: set[int] = set()
            side_visible: list[PlayerTrack] = []
            for track in side_tracks:
                if track.zone_index is not None and track.zone_index not in used_zones:
                    used_zones.add(track.zone_index)
                    side_visible.append(track)
                elif len(side_visible) < self.config.max_players_per_team:
                    side_visible.append(track)
                if len(side_visible) >= self.config.max_players_per_team:
                    break
            visible.extend(side_visible)
        return visible

    def _count_from_tracks(self, active_tracks: list[PlayerTrack], blobs: list[PlayerBlob]) -> tuple[int, int]:
        counts: dict[str, int] = {}
        for side in ("top", "bottom"):
            memory = self.side_zone_memory[side]
            for zone_index in list(memory):
                memory[zone_index] -= 1
                if memory[zone_index] <= 0:
                    del memory[zone_index]

            stable_tracks = [track for track in active_tracks if track.side == side and track.age >= self.config.player_min_track_age_for_count]
            for track in stable_tracks:
                if track.zone_index is not None:
                    memory[track.zone_index] = self.config.player_zone_ttl

            raw_zones: set[int] = set()
            for blob in blobs:
                if blob.side != side:
                    continue
                raw_zones.update(self._zones_for_blob(blob))
            for zone_index in raw_zones:
                memory[zone_index] = max(memory.get(zone_index, 0), self.config.player_zone_ttl // 2)

            track_count = len(stable_tracks)
            zone_count = len(memory)
            raw_count = len(raw_zones)
            count = max(track_count, zone_count, raw_count)
            counts[side] = int(np.clip(count, 0, self.config.max_players_per_team))

        if self.last_counts is not None:
            top_last, bottom_last = self.last_counts
            if counts["top"] < top_last - 1:
                counts["top"] = top_last - 1
            if counts["bottom"] < bottom_last - 1:
                counts["bottom"] = bottom_last - 1
            if counts["top"] > top_last + 1:
                counts["top"] = top_last + 1
            if counts["bottom"] > bottom_last + 1:
                counts["bottom"] = bottom_last + 1

        counts["top"] = int(np.clip(counts["top"], 0, self.config.max_players_per_team))
        counts["bottom"] = int(np.clip(counts["bottom"], 0, self.config.max_players_per_team))
        self.last_counts = (counts["top"], counts["bottom"])
        return counts["top"], counts["bottom"]

    def _build_ball_suppression_boxes(
        self,
        active_tracks: list[PlayerTrack],
        blobs: list[PlayerBlob],
    ) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for track in active_tracks:
            boxes.append(expand_bbox(player_ball_suppression_bbox(track.bbox), self.frame_shape, 8, 6))
        for blob in blobs:
            boxes.append(expand_bbox(player_ball_suppression_bbox(blob.bbox), self.frame_shape, 6, 4))
        return boxes


class BallTrackerOpenCVV3:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        self.ball_mask = apply_exclusions(build_mask(frame_shape[:2], config.ball_search_polygon), config.ball_exclusion_rects)
        self.mode = "SEARCH_INIT"
        self.kalman: Optional[cv2.KalmanFilter] = None
        self.last_confirmed_center: Optional[tuple[int, int]] = None
        self.last_confirmed_radius: Optional[float] = None
        self.last_confirmed_frame = -1
        self.miss_count = 0
        self.stationary_frames = 0
        self.trail: deque[Optional[tuple[int, int]]] = deque(maxlen=config.trail_length * 3)
        self.prev_gray: Optional[np.ndarray] = None
        self.flow_points: Optional[np.ndarray] = None
        self.flow_frames_left = 0

    def update(
        self,
        gray: np.ndarray,
        observation: FrameObservation,
        suppression_boxes: list[tuple[int, int, int, int]],
        frame_index: int,
        observations: list[FrameObservation],
    ) -> BallState:
        predicted_center: Optional[tuple[int, int]] = None
        uncertainty = 0.0
        if self.kalman is not None:
            prediction = self.kalman.predict()
            predicted_center = (int(round(prediction[0][0])), int(round(prediction[1][0])))
            uncertainty = float(np.trace(self.kalman.errorCovPre[:2, :2]))

        if self.mode in {"SEARCH_INIT", "LOST"}:
            init_candidates = observation.ball_candidates if self.mode == "SEARCH_INIT" else (
                observation.ball_candidates + observation.recovery_candidates
            )
            if self.mode == "SEARCH_INIT":
                init_candidates = [
                    candidate
                    for candidate in init_candidates
                    if not self._center_in_boxes(candidate.center, suppression_boxes)
                    and 160 <= candidate.center[0] <= self.frame_shape[1] - 160
                ]
            candidate = self._select_candidate(
                init_candidates,
                None,
                suppression_boxes,
                allow_recovery=self.mode == "LOST",
            )
            if candidate and self._candidate_confirmed(candidate, frame_index, observations, allow_recovery=self.mode == "LOST"):
                state = self._confirm_candidate(gray, candidate, frame_index)
            else:
                state = BallState(center=None, radius=None, status="missing")
        elif self.mode == "TRACK":
            candidate = self._select_local_candidate(observation.ball_candidates, predicted_center, suppression_boxes)
            if candidate is not None:
                state = self._confirm_candidate(gray, candidate, frame_index)
            else:
                state = self._coast_or_recover(gray, predicted_center, suppression_boxes, frame_index, observation, observations)
        elif self.mode == "COAST":
            candidate = self._select_local_candidate(observation.ball_candidates, predicted_center, suppression_boxes)
            if candidate is not None:
                state = self._confirm_candidate(gray, candidate, frame_index)
            else:
                state = self._coast_or_recover(gray, predicted_center, suppression_boxes, frame_index, observation, observations)
        else:  # RECOVER
            combined_candidates = observation.ball_candidates + observation.recovery_candidates
            recover_center = predicted_center if self.miss_count <= self.config.ball_recover_after_misses + 1 else None
            candidate = self._select_candidate(
                combined_candidates,
                recover_center,
                suppression_boxes,
                allow_recovery=True,
            )
            if candidate and self._candidate_confirmed(candidate, frame_index, observations, allow_recovery=True):
                state = self._confirm_candidate(gray, candidate, frame_index)
            else:
                state = self._coast_or_recover(gray, predicted_center, suppression_boxes, frame_index, observation, observations)

        self.prev_gray = gray
        return state

    def _select_local_candidate(
        self,
        candidates: list[BallObservation],
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> Optional[BallObservation]:
        if predicted_center is None:
            return self._select_candidate(candidates, predicted_center, suppression_boxes, allow_recovery=False)
        roi = min(
            self.config.ball_roi_base + (self.miss_count * self.config.ball_roi_miss_gain),
            self.config.ball_roi_max,
        )
        filtered = [candidate for candidate in candidates if distance_between(candidate.center, predicted_center) <= roi]
        return self._select_candidate(filtered, predicted_center, suppression_boxes, allow_recovery=False)

    def _select_candidate(
        self,
        candidates: list[BallObservation],
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
        allow_recovery: bool,
    ) -> Optional[BallObservation]:
        best_candidate: Optional[BallObservation] = None
        best_score = -1e9
        for candidate in candidates:
            score = self._score_candidate(candidate, predicted_center, suppression_boxes, allow_recovery)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate if best_score > 0.0 else None

    def _score_candidate(
        self,
        candidate: BallObservation,
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
        allow_recovery: bool,
    ) -> float:
        if candidate.area < self.config.ball_min_area or candidate.area > self.config.ball_max_area:
            return -1e9
        if candidate.radius < self.config.ball_min_radius or candidate.radius > self.config.ball_max_radius:
            return -1e9
        if candidate.circularity < self.config.ball_min_circularity:
            return -6.0
        x, y, w, h = candidate.bbox
        aspect_ratio = w / max(float(h), 1.0)
        if aspect_ratio > self.config.ball_max_aspect_ratio or (1.0 / max(aspect_ratio, 1e-6)) > self.config.ball_max_aspect_ratio:
            return -1e9
        if not point_in_mask(candidate.center, self.ball_mask):
            return -1e9

        score = 0.0
        score += candidate.motion_ratio * 18.0
        score += candidate.circularity * 6.0
        score += candidate.compactness * 4.0
        if candidate.source == "motion":
            score += 1.0
        elif allow_recovery:
            score += candidate.weak_yellow_ratio * 8.0

        if predicted_center is not None:
            distance = distance_between(candidate.center, predicted_center)
            gate = self.config.ball_recover_gate if allow_recovery else self.config.ball_track_gate
            if distance > gate:
                return -1e9
            score += max(0.0, 6.0 - ((distance / max(gate, 1.0)) * 6.0))
        elif candidate.motion_ratio < self.config.ball_min_motion_ratio and not allow_recovery:
            return -1e9

        overlap_penalty = 0.0
        for box in suppression_boxes:
            overlap = bbox_iou(candidate.bbox, box)
            if overlap > 0.0:
                overlap_penalty = max(overlap_penalty, overlap)
            if box[0] <= candidate.center[0] <= box[0] + box[2] and box[1] <= candidate.center[1] <= box[1] + box[3]:
                if predicted_center is None:
                    return -1e9
                else:
                    if (
                        distance_between(candidate.center, predicted_center) > self.config.ball_player_penalty_gate
                        or candidate.radius > 5.5
                        or candidate.motion_ratio < max(0.08, self.config.ball_min_motion_ratio * 2.0)
                    ):
                        return -1e9
                    score -= self.config.ball_player_penalty_scale + 1.5
        score -= overlap_penalty * self.config.ball_player_penalty_scale

        if candidate.center[1] < self.config.ball_banner_guard_y and candidate.motion_ratio < self.config.ball_banner_motion_ratio:
            score -= 6.0

        if self.last_confirmed_center is not None:
            jump = distance_between(candidate.center, self.last_confirmed_center)
            if jump < self.config.ball_stationary_pixels:
                score -= 1.5
        return score

    def _candidate_confirmed(
        self,
        candidate: BallObservation,
        frame_index: int,
        observations: list[FrameObservation],
        allow_recovery: bool,
    ) -> bool:
        if frame_index >= len(observations) - 3:
            return candidate.motion_ratio >= self.config.ball_min_motion_ratio * 0.8

        supports = 0
        best_progress = 0.0
        for delta in (1, 2, 3):
            future_observation = observations[frame_index + delta]
            future_candidates = list(future_observation.ball_candidates)
            if allow_recovery:
                future_candidates.extend(future_observation.recovery_candidates)
            gate = self.config.ball_confirm_distance + ((delta - 1) * 24.0)
            best_distance: Optional[float] = None
            best_local_progress = 0.0
            for future_candidate in future_candidates:
                distance = distance_between(candidate.center, future_candidate.center)
                if distance > gate:
                    continue
                progress = distance
                if future_candidate.motion_ratio >= self.config.ball_min_motion_ratio * 0.7:
                    progress = max(progress, self.config.ball_min_progress_pixels)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_local_progress = progress
            if best_distance is not None:
                supports += 1
                best_progress = max(best_progress, best_local_progress)

        if supports == 0:
            return False
        if candidate.center[1] < self.config.ball_banner_guard_y:
            return (
                supports >= 2
                or best_progress >= self.config.ball_min_progress_pixels * 1.5
                or candidate.motion_ratio >= self.config.ball_banner_motion_ratio
            )
        return (
            supports >= 2
            or best_progress >= self.config.ball_min_progress_pixels * 1.5
            or candidate.motion_ratio >= self.config.ball_min_motion_ratio
        )

    def _confirm_candidate(self, gray: np.ndarray, candidate: BallObservation, frame_index: int) -> BallState:
        if self.kalman is None:
            self.kalman = build_track_kalman(candidate.center)
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        self.kalman.correct(measurement)
        if self.last_confirmed_center is not None:
            jump = distance_between(candidate.center, self.last_confirmed_center)
            if jump > self.config.ball_recover_gate:
                self._break_trail()
        if self.last_confirmed_center is not None and distance_between(candidate.center, self.last_confirmed_center) <= self.config.ball_stationary_pixels:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0

        self.last_confirmed_center = candidate.center
        self.last_confirmed_radius = candidate.radius
        self.last_confirmed_frame = frame_index
        self.mode = "TRACK"
        self.miss_count = 0
        self.trail.append(candidate.center)
        self._refresh_flow_points(gray, candidate.bbox)
        return BallState(center=candidate.center, radius=candidate.radius, status="confirmed")

    def _coast_or_recover(
        self,
        gray: np.ndarray,
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
        frame_index: int,
        observation: FrameObservation,
        observations: list[FrameObservation],
    ) -> BallState:
        self.miss_count += 1
        flow_state = self._bridge_with_flow(gray, predicted_center, suppression_boxes)
        if flow_state is not None:
            self.mode = "COAST"
            return flow_state

        if predicted_center is not None and self._predicted_point_valid(predicted_center, suppression_boxes):
            uncertainty = 0.0
            if self.kalman is not None:
                uncertainty = float(np.trace(self.kalman.errorCovPre[:2, :2]))
            if self.miss_count <= self.config.ball_coast_frames and uncertainty <= self.config.ball_max_uncertainty:
                self.mode = "COAST"
                return BallState(center=predicted_center, radius=self.last_confirmed_radius, status="coast")

        if self.miss_count <= self.config.ball_recover_after_misses:
            self.mode = "COAST"
            return BallState(center=None, radius=None, status="missing")

        if self.miss_count >= self.config.ball_lost_after_misses:
            self.mode = "LOST"
            self._break_trail()
            self.kalman = None
            self.last_confirmed_center = None
            self.last_confirmed_radius = None
            self.flow_points = None
            self.flow_frames_left = 0
            return BallState(center=None, radius=None, status="missing")

        self.mode = "RECOVER"
        combined_candidates = observation.ball_candidates + observation.recovery_candidates
        candidate = self._select_candidate(combined_candidates, predicted_center, suppression_boxes, allow_recovery=True)
        if candidate and self._candidate_confirmed(candidate, frame_index, observations, allow_recovery=True):
            return self._confirm_candidate(gray, candidate, frame_index)

        return BallState(center=None, radius=None, status="missing")

    def _refresh_flow_points(self, gray: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        mask = np.zeros_like(gray)
        x, y, w, h = expand_bbox(bbox, self.frame_shape, 5, 5)
        mask[y : y + h, x : x + w] = 255
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=6,
            qualityLevel=self.config.ball_flow_quality,
            minDistance=self.config.ball_flow_min_distance,
            blockSize=self.config.ball_flow_block_size,
            mask=mask,
        )
        self.flow_points = points
        self.flow_frames_left = self.config.ball_flow_max_frames if points is not None else 0

    def _bridge_with_flow(
        self,
        gray: np.ndarray,
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> Optional[BallState]:
        if (
            self.prev_gray is None
            or self.flow_points is None
            or self.flow_frames_left <= 0
            or predicted_center is None
        ):
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.flow_points,
            None,
            winSize=self.config.ball_flow_win_size,
            maxLevel=2,
        )
        if next_points is None or status is None:
            return None
        valid_points = next_points[status.reshape(-1) == 1]
        if len(valid_points) == 0:
            return None

        bridged_center = tuple(np.round(valid_points.reshape(-1, 2).mean(axis=0)).astype(int))
        if not self._predicted_point_valid(bridged_center, suppression_boxes):
            return None
        if distance_between(bridged_center, predicted_center) > self.config.ball_track_gate:
            return None

        self.flow_points = valid_points.reshape(-1, 1, 2)
        self.flow_frames_left -= 1
        return BallState(center=bridged_center, radius=self.last_confirmed_radius, status="bridge")

    def _predicted_point_valid(
        self,
        point: tuple[int, int],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        if not point_in_mask(point, self.ball_mask):
            return False
        for box in suppression_boxes:
            if box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]:
                return False
        return True

    def _center_in_boxes(
        self,
        center: tuple[int, int],
        boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        for box in boxes:
            if box[0] <= center[0] <= box[0] + box[2] and box[1] <= center[1] <= box[1] + box[3]:
                return True
        return False

    def _break_trail(self) -> None:
        if not self.trail or self.trail[-1] is not None:
            self.trail.append(None)


def draw_players(frame: np.ndarray, tracks: list[PlayerTrack]) -> None:
    for track in tracks:
        x, y, w, h = track.bbox
        color = (0, 215, 255) if track.stable_team == "team_a" else (255, 170, 70)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ball(frame: np.ndarray, tracker: BallTrackerOpenCVV3, state: BallState) -> None:
    trail_points = list(tracker.trail)
    for previous, current in zip(trail_points, trail_points[1:]):
        if previous is None or current is None:
            continue
        cv2.line(frame, previous, current, (0, 210, 255), 3)
    if state.center is not None and state.radius is not None:
        center = state.center
        radius = max(3, int(round(state.radius)))
        color = (0, 230, 0) if state.status == "confirmed" else (0, 170, 255)
        cv2.circle(frame, center, radius, color, 2)
        cv2.circle(frame, center, 2, color, -1)


def draw_overlay(frame: np.ndarray, team_a_count: int, team_b_count: int, fps: float) -> None:
    cv2.rectangle(frame, (18, 18), (210, 108), (30, 30, 30), -1)
    cv2.putText(frame, f"Team A: {team_a_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
    cv2.putText(frame, f"Team B: {team_b_count}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 170, 70), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(description="Strict OpenCV-only volleyball tracking").parse_args()


def build_argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Strict OpenCV-only volleyball tracking")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "task phase 6" / "Volleyball.mp4",
        help="Path to the input volleyball video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "tp6" / "Volleyball_annotated_opencv_v3.mp4",
        help="Path to the rendered output video.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated video while rendering.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = SceneConfig()

    probe = cv2.VideoCapture(str(args.input))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    frame_shape = (
        int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
        3,
    )
    probe.release()

    collector = ObservationCollector(config, frame_shape)
    observations, fps, width, height = collector.collect(args.input)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen input video: {args.input}")

    player_tracker = PlayerTrackerOpenCVV3(config, frame_shape)
    ball_tracker = BallTrackerOpenCVV3(config, frame_shape)
    render_start = time.time()
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        observation = observations[frame_index]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        active_tracks, team_a_count, team_b_count, suppression_boxes = player_tracker.update(
            frame,
            observation.player_blobs,
            frame_index,
        )
        ball_state = ball_tracker.update(
            gray,
            observation,
            suppression_boxes,
            frame_index,
            observations,
        )

        annotated = frame.copy()
        draw_players(annotated, active_tracks)
        draw_ball(annotated, ball_tracker, ball_state)
        elapsed = max(time.time() - render_start, 1e-6)
        current_fps = (frame_index + 1) / elapsed
        draw_overlay(annotated, team_a_count, team_b_count, current_fps)

        writer.write(annotated)
        if args.display:
            cv2.imshow("Volleyball Tracker OpenCV V3", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    capture.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
