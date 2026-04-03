from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class SceneConfig:
    trail_length: int = 26
    max_players_per_team: int = 6
    bg_warmup_frames: int = 28
    top_player_area_min: int = 700
    top_player_area_max: int = 42000
    bottom_player_area_min: int = 1500
    bottom_player_area_max: int = 72000
    player_min_height_top: int = 30
    player_min_height_bottom: int = 54
    player_min_width: int = 14
    player_max_width: int = 175
    player_max_aspect_ratio: float = 1.25
    player_min_fill_ratio: float = 0.18
    player_match_distance_top: float = 72.0
    player_match_distance_bottom: float = 96.0
    player_iou_match_threshold: float = 0.05
    player_track_ttl: int = 18
    player_active_ttl: int = 9
    player_confidence_gain: float = 1.0
    player_confidence_decay: float = 0.35
    player_confidence_floor: float = 1.1
    player_zone_ttl: int = 8
    player_zone_cols: int = 3
    player_zone_rows: int = 2
    player_split_width_top: int = 84
    player_split_width_bottom: int = 122
    player_split_height_top: int = 118
    player_split_height_bottom: int = 180
    player_split_min_width: int = 24
    player_count_history: int = 5
    player_bg_history: int = 240
    player_bg_var_threshold: int = 20
    player_top_open_kernel: tuple[int, int] = (3, 3)
    player_top_close_kernel: tuple[int, int] = (7, 9)
    player_top_dilate_kernel: tuple[int, int] = (5, 9)
    player_bottom_open_kernel: tuple[int, int] = (3, 3)
    player_bottom_close_kernel: tuple[int, int] = (9, 11)
    player_bottom_dilate_kernel: tuple[int, int] = (7, 11)
    team_color_floor_ratio: float = 0.012
    team_color_margin: float = 0.004
    torso_x_margin_ratio: float = 0.32
    torso_y_start_ratio: float = 0.16
    torso_y_end_ratio: float = 0.42
    player_team_history: int = 7
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
    ball_bg_history: int = 220
    ball_bg_var_threshold: int = 16
    ball_open_kernel: tuple[int, int] = (3, 3)
    ball_close_kernel: tuple[int, int] = (5, 5)
    ball_dilate_kernel: tuple[int, int] = (3, 3)
    ball_min_area: float = 6.0
    ball_max_area: float = 240.0
    ball_min_radius: float = 2.0
    ball_max_radius: float = 10.5
    ball_min_circularity: float = 0.24
    ball_max_aspect_ratio: float = 1.85
    ball_typical_radius: float = 4.8
    ball_search_init_gate: float = 9999.0
    ball_track_gate: float = 86.0
    ball_recover_gate: float = 170.0
    ball_confirm_distance: float = 88.0
    ball_min_progress_pixels: float = 7.0
    ball_coast_frames: int = 3
    ball_recover_after_misses: int = 3
    ball_lost_after_misses: int = 9
    ball_max_uncertainty: float = 2600.0
    ball_banner_guard_y: int = 152
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
class BallCandidate:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    radius: float
    area: float
    circularity: float
    compactness: float
    weak_yellow_ratio: float


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
    ball_candidates: list[BallCandidate]
    player_blobs: list[PlayerBlob]


@dataclass
class PlayerTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    footpoint: tuple[int, int]
    side: str
    last_seen: int
    age: int = 1
    misses: int = 0
    confidence: float = 1.0
    zone_index: Optional[int] = None
    label_history: deque[str] = field(default_factory=lambda: deque(maxlen=7))
    stable_team: Optional[str] = None


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


def ball_suppression_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
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


def build_kalman(center: tuple[int, int]) -> cv2.KalmanFilter:
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


class ObservationCollectorV4:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        mask_shape = frame_shape[:2]
        self.ball_mask = apply_exclusions(build_mask(mask_shape, config.ball_search_polygon), config.ball_exclusion_rects)
        self.top_mask = apply_exclusions(build_mask(mask_shape, config.top_court_polygon), config.player_exclusion_rects)
        self.bottom_mask = apply_exclusions(build_mask(mask_shape, config.bottom_court_polygon), config.player_exclusion_rects)
        self.ball_bg = cv2.createBackgroundSubtractorMOG2(
            history=config.ball_bg_history,
            varThreshold=config.ball_bg_var_threshold,
            detectShadows=False,
        )
        self.top_bg = cv2.createBackgroundSubtractorMOG2(
            history=config.player_bg_history,
            varThreshold=config.player_bg_var_threshold,
            detectShadows=False,
        )
        self.bottom_bg = cv2.createBackgroundSubtractorMOG2(
            history=config.player_bg_history,
            varThreshold=config.player_bg_var_threshold,
            detectShadows=False,
        )
        self.ball_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_open_kernel)
        self.ball_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_close_kernel)
        self.ball_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_dilate_kernel)
        self.top_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_open_kernel)
        self.top_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_close_kernel)
        self.top_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_top_dilate_kernel)
        self.bottom_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_open_kernel)
        self.bottom_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_close_kernel)
        self.bottom_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_bottom_dilate_kernel)

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

            ball_mask = self._foreground_mask(
                gray,
                self.ball_mask,
                self.ball_bg,
                self.ball_open_kernel,
                self.ball_close_kernel,
                self.ball_dilate_kernel,
            )
            top_mask = self._foreground_mask(
                gray,
                self.top_mask,
                self.top_bg,
                self.top_open_kernel,
                self.top_close_kernel,
                self.top_dilate_kernel,
            )
            bottom_mask = self._foreground_mask(
                gray,
                self.bottom_mask,
                self.bottom_bg,
                self.bottom_open_kernel,
                self.bottom_close_kernel,
                self.bottom_dilate_kernel,
            )

            if frame_index < self.config.bg_warmup_frames:
                observations.append(FrameObservation(ball_candidates=[], player_blobs=[]))
            else:
                ball_candidates = self._extract_ball_candidates(ball_mask, hsv)
                player_blobs = self._extract_player_blobs(top_mask, "top") + self._extract_player_blobs(bottom_mask, "bottom")
                observations.append(
                    FrameObservation(
                        ball_candidates=ball_candidates,
                        player_blobs=self._merge_player_blobs(player_blobs),
                    )
                )
            frame_index += 1

        capture.release()
        return observations, fps, width, height

    def _foreground_mask(
        self,
        gray: np.ndarray,
        region_mask: np.ndarray,
        subtractor: cv2.BackgroundSubtractorMOG2,
        open_kernel: np.ndarray,
        close_kernel: np.ndarray,
        dilate_kernel: np.ndarray,
    ) -> np.ndarray:
        masked = cv2.bitwise_and(gray, gray, mask=region_mask)
        fg_mask = subtractor.apply(masked)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, region_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)
        return fg_mask

    def _extract_ball_candidates(self, fg_mask: np.ndarray, hsv: np.ndarray) -> list[BallCandidate]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[BallCandidate] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.ball_min_area or area > self.config.ball_max_area:
                continue
            x, y, w, h = clip_bbox(cv2.boundingRect(contour), self.frame_shape)
            if w <= 0 or h <= 0:
                continue
            aspect_ratio = w / max(float(h), 1.0)
            if aspect_ratio > self.config.ball_max_aspect_ratio or (1.0 / max(aspect_ratio, 1e-6)) > self.config.ball_max_aspect_ratio:
                continue
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.config.ball_min_radius or radius > self.config.ball_max_radius:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0.0 else float((4.0 * np.pi * area) / (perimeter * perimeter))
            if circularity < self.config.ball_min_circularity:
                continue
            center = (int(round(circle_x)), int(round(circle_y)))
            if not point_in_mask(center, self.ball_mask):
                continue
            patch_hsv = hsv[y : y + h, x : x + w]
            weak_yellow_ratio = cv2.countNonZero(
                cv2.inRange(patch_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / max(float(w * h), 1.0)
            candidates.append(
                BallCandidate(
                    center=center,
                    bbox=(x, y, w, h),
                    radius=float(radius),
                    area=float(area),
                    circularity=circularity,
                    compactness=area / max(float(w * h), 1.0),
                    weak_yellow_ratio=weak_yellow_ratio,
                )
            )
        return candidates

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
                if h < self.config.player_min_height_top or area < self.config.top_player_area_min or area > self.config.top_player_area_max:
                    continue
                side_mask = self.top_mask
            else:
                if h < self.config.player_min_height_bottom or area < self.config.bottom_player_area_min or area > self.config.bottom_player_area_max:
                    continue
                side_mask = self.bottom_mask
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

    def _merge_player_blobs(self, blobs: list[PlayerBlob]) -> list[PlayerBlob]:
        merged: list[PlayerBlob] = []
        for blob in blobs:
            replaced = False
            for index, existing in enumerate(merged):
                if existing.side != blob.side:
                    continue
                if bbox_iou(existing.bbox, blob.bbox) > 0.35 or distance_between(existing.center, blob.center) < 24.0:
                    existing_area = existing.bbox[2] * existing.bbox[3]
                    blob_area = blob.bbox[2] * blob.bbox[3]
                    if blob_area > existing_area:
                        merged[index] = blob
                    replaced = True
                    break
            if not replaced:
                merged.append(blob)
        return merged


class PlayerTrackerV4:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        self.tracks: dict[int, PlayerTrack] = {}
        self.next_track_id = 1
        self.top_mask = apply_exclusions(build_mask(frame_shape[:2], config.top_court_polygon), config.player_exclusion_rects)
        self.bottom_mask = apply_exclusions(build_mask(frame_shape[:2], config.bottom_court_polygon), config.player_exclusion_rects)
        self.top_zone_boxes = self._build_zone_boxes(config.top_court_polygon)
        self.bottom_zone_boxes = self._build_zone_boxes(config.bottom_court_polygon)
        self.side_zone_memory: dict[str, dict[int, int]] = {"top": {}, "bottom": {}}
        self.count_history: dict[str, deque[int]] = {
            "top": deque(maxlen=config.player_count_history),
            "bottom": deque(maxlen=config.player_count_history),
        }
        self.last_counts: Optional[tuple[int, int]] = None

    def update(
        self,
        frame: np.ndarray,
        raw_blobs: list[PlayerBlob],
        frame_index: int,
    ) -> tuple[list[PlayerTrack], int, int, list[tuple[int, int, int, int]]]:
        blobs = self._split_blobs(raw_blobs)
        matches, unmatched_blob_ids, unmatched_track_ids = self._associate(blobs)

        for blob_index, track_id in matches:
            self._update_track(self.tracks[track_id], blobs[blob_index], frame, frame_index)

        for blob_index in unmatched_blob_ids:
            self._create_track(blobs[blob_index], frame, frame_index)

        for track_id in unmatched_track_ids:
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            track.misses += 1
            track.confidence = max(0.0, track.confidence - self.config.player_confidence_decay)

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
            if track.misses <= self.config.player_active_ttl and track.confidence >= self.config.player_confidence_floor
        ]
        visible_tracks = self._select_visible_tracks(active_tracks)
        team_a_count, team_b_count = self._count_tracks(visible_tracks, blobs)
        suppression_boxes = self._build_ball_suppression_boxes(visible_tracks, blobs)
        return visible_tracks, team_a_count, team_b_count, suppression_boxes

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

    def _split_blobs(self, raw_blobs: list[PlayerBlob]) -> list[PlayerBlob]:
        expanded: list[PlayerBlob] = []
        for blob in raw_blobs:
            if blob.zone_hints <= 1:
                expanded.append(blob)
                continue
            split_count = min(blob.zone_hints, 3)
            step = blob.bbox[2] / split_count
            for index in range(split_count):
                x1 = int(round(blob.bbox[0] + (index * step)))
                x2 = int(round(blob.bbox[0] + ((index + 1) * step)))
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

    def _associate(self, blobs: list[PlayerBlob]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
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
                    center_distance = distance_between(blob.center, track.center)
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
            last_seen=frame_index,
            confidence=self.config.player_confidence_gain,
        )
        track.zone_index = self._zone_for_point(track.side, track.footpoint)
        track.stable_team = "team_a" if track.side == "top" else "team_b"
        self._update_team(track, frame)
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    def _update_track(self, track: PlayerTrack, blob: PlayerBlob, frame: np.ndarray, frame_index: int) -> None:
        track.bbox = blob.bbox
        track.center = blob.center
        track.footpoint = blob.footpoint
        track.side = blob.side
        track.last_seen = frame_index
        track.age += 1
        track.misses = 0
        track.confidence = min(track.confidence + self.config.player_confidence_gain, 10.0)
        track.zone_index = self._zone_for_point(track.side, track.footpoint)
        self._update_team(track, frame)

    def _update_team(self, track: PlayerTrack, frame: np.ndarray) -> None:
        expected_team = "team_a" if track.side == "top" else "team_b"
        team = classify_team(frame, track.bbox, expected_team, self.config)
        if team is not None:
            track.label_history.append(team)
        if track.label_history:
            counts = Counter(track.label_history)
            track.stable_team = counts.most_common(1)[0][0]
        elif track.stable_team is None:
            track.stable_team = expected_team

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

    def _select_visible_tracks(self, active_tracks: list[PlayerTrack]) -> list[PlayerTrack]:
        visible: list[PlayerTrack] = []
        for side in ("top", "bottom"):
            side_tracks = [track for track in active_tracks if track.side == side]
            side_tracks.sort(
                key=lambda track: (track.confidence, track.age, -track.misses, track.last_seen),
                reverse=True,
            )
            used_zones: set[int] = set()
            chosen: list[PlayerTrack] = []
            for track in side_tracks:
                if track.zone_index is not None and track.zone_index not in used_zones:
                    used_zones.add(track.zone_index)
                    chosen.append(track)
                elif len(chosen) < self.config.max_players_per_team:
                    chosen.append(track)
                if len(chosen) >= self.config.max_players_per_team:
                    break
            visible.extend(chosen)
        return visible

    def _zones_for_blob(self, blob: PlayerBlob) -> set[int]:
        boxes = self.top_zone_boxes if blob.side == "top" else self.bottom_zone_boxes
        zones: set[int] = set()
        for index, box in enumerate(boxes):
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

    def _count_tracks(self, visible_tracks: list[PlayerTrack], raw_blobs: list[PlayerBlob]) -> tuple[int, int]:
        raw_counts: dict[str, int] = {}
        for side in ("top", "bottom"):
            memory = self.side_zone_memory[side]
            for zone_index in list(memory):
                memory[zone_index] -= 1
                if memory[zone_index] <= 0:
                    del memory[zone_index]

            side_tracks = [track for track in visible_tracks if track.side == side]
            for track in side_tracks:
                if track.zone_index is not None:
                    memory[track.zone_index] = self.config.player_zone_ttl

            raw_zones: set[int] = set()
            for blob in raw_blobs:
                if blob.side != side:
                    continue
                raw_zones.update(self._zones_for_blob(blob))
            for zone_index in raw_zones:
                memory[zone_index] = max(memory.get(zone_index, 0), self.config.player_zone_ttl // 2)

            raw_count = max(len(side_tracks), len(memory), len(raw_zones))
            raw_count = int(np.clip(raw_count, 0, self.config.max_players_per_team))
            self.count_history[side].append(raw_count)
            smooth_count = int(round(statistics.median(self.count_history[side])))
            raw_counts[side] = smooth_count

        if self.last_counts is not None:
            top_last, bottom_last = self.last_counts
            if raw_counts["top"] < top_last - 1:
                raw_counts["top"] = top_last - 1
            if raw_counts["bottom"] < bottom_last - 1:
                raw_counts["bottom"] = bottom_last - 1
            if raw_counts["top"] > top_last + 1:
                raw_counts["top"] = top_last + 1
            if raw_counts["bottom"] > bottom_last + 1:
                raw_counts["bottom"] = bottom_last + 1

        raw_counts["top"] = int(np.clip(raw_counts["top"], 0, self.config.max_players_per_team))
        raw_counts["bottom"] = int(np.clip(raw_counts["bottom"], 0, self.config.max_players_per_team))
        self.last_counts = (raw_counts["top"], raw_counts["bottom"])
        return raw_counts["top"], raw_counts["bottom"]

    def _build_ball_suppression_boxes(
        self,
        visible_tracks: list[PlayerTrack],
        raw_blobs: list[PlayerBlob],
    ) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for track in visible_tracks:
            boxes.append(clip_bbox(ball_suppression_bbox(track.bbox), self.frame_shape))
        for blob in raw_blobs:
            boxes.append(clip_bbox(ball_suppression_bbox(blob.bbox), self.frame_shape))
        return boxes


class BallTrackerV4:
    def __init__(self, config: SceneConfig, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        self.ball_mask = apply_exclusions(build_mask(frame_shape[:2], config.ball_search_polygon), config.ball_exclusion_rects)
        self.mode = "SEARCH_INIT"
        self.kalman: Optional[cv2.KalmanFilter] = None
        self.last_confirmed_center: Optional[tuple[int, int]] = None
        self.last_confirmed_radius: Optional[float] = None
        self.miss_count = 0
        self.trail: deque[Optional[tuple[int, int]]] = deque(maxlen=config.trail_length * 3)

    def update(
        self,
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
            candidate = self._select_global_candidate(observation.ball_candidates, frame_index, observations, suppression_boxes, allow_yellow=False)
            if candidate is not None:
                return self._confirm(candidate)
            return BallState(center=None, radius=None, status="missing")

        if self.mode == "TRACK":
            candidate = self._select_local_candidate(observation.ball_candidates, predicted_center, suppression_boxes)
            if candidate is not None:
                return self._confirm(candidate)
            return self._handle_miss(observation, frame_index, observations, suppression_boxes, predicted_center, uncertainty)

        if self.mode == "COAST":
            candidate = self._select_local_candidate(observation.ball_candidates, predicted_center, suppression_boxes)
            if candidate is not None:
                return self._confirm(candidate)
            return self._handle_miss(observation, frame_index, observations, suppression_boxes, predicted_center, uncertainty)

        if self.mode == "RECOVER":
            candidate = self._select_global_candidate(observation.ball_candidates, frame_index, observations, suppression_boxes, allow_yellow=True)
            if candidate is not None:
                return self._confirm(candidate)
            return self._handle_miss(observation, frame_index, observations, suppression_boxes, predicted_center, uncertainty)

        return BallState(center=None, radius=None, status="missing")

    def _select_local_candidate(
        self,
        candidates: list[BallCandidate],
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> Optional[BallCandidate]:
        if predicted_center is None:
            return None
        filtered = [
            candidate
            for candidate in candidates
            if distance_between(candidate.center, predicted_center) <= self.config.ball_track_gate
        ]
        best_candidate: Optional[BallCandidate] = None
        best_score = -1e9
        for candidate in filtered:
            score = self._score_candidate(candidate, predicted_center, suppression_boxes, allow_yellow=False)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate if best_score > 0.0 else None

    def _select_global_candidate(
        self,
        candidates: list[BallCandidate],
        frame_index: int,
        observations: list[FrameObservation],
        suppression_boxes: list[tuple[int, int, int, int]],
        allow_yellow: bool,
    ) -> Optional[BallCandidate]:
        best_candidate: Optional[BallCandidate] = None
        best_score = -1e9
        for candidate in candidates:
            support_count, support_progress = self._temporal_support(candidate, frame_index, observations, suppression_boxes)
            if support_count == 0:
                continue
            if candidate.center[1] < self.config.ball_banner_guard_y and support_count < 2:
                continue
            score = self._score_candidate(candidate, None, suppression_boxes, allow_yellow)
            score += support_count * 5.0
            score += support_progress * 0.08
            if score > best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate if best_score > 2.0 else None

    def _score_candidate(
        self,
        candidate: BallCandidate,
        predicted_center: Optional[tuple[int, int]],
        suppression_boxes: list[tuple[int, int, int, int]],
        allow_yellow: bool,
    ) -> float:
        if not point_in_mask(candidate.center, self.ball_mask):
            return -1e9
        if self._inside_boxes(candidate.center, candidate.bbox, suppression_boxes):
            return -1e9

        score = 0.0
        score += candidate.circularity * 6.0
        score += candidate.compactness * 4.0
        score += max(0.0, 3.0 - abs(candidate.radius - self.config.ball_typical_radius))

        if predicted_center is not None:
            distance = distance_between(candidate.center, predicted_center)
            if distance > self.config.ball_track_gate:
                return -1e9
            score += max(0.0, 8.0 - ((distance / self.config.ball_track_gate) * 8.0))

        if allow_yellow:
            score += candidate.weak_yellow_ratio * 10.0

        if candidate.center[1] < self.config.ball_banner_guard_y:
            score -= 1.5
        return score

    def _temporal_support(
        self,
        candidate: BallCandidate,
        frame_index: int,
        observations: list[FrameObservation],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, float]:
        supports = 0
        max_progress = 0.0
        for delta in (1, 2):
            if frame_index + delta >= len(observations):
                break
            future_candidates = observations[frame_index + delta].ball_candidates
            gate = self.config.ball_confirm_distance + ((delta - 1) * 28.0)
            best_progress: Optional[float] = None
            for future_candidate in future_candidates:
                if self._inside_boxes(future_candidate.center, future_candidate.bbox, suppression_boxes):
                    continue
                distance = distance_between(candidate.center, future_candidate.center)
                if distance > gate:
                    continue
                if distance < self.config.ball_min_progress_pixels:
                    continue
                if best_progress is None or distance < best_progress:
                    best_progress = distance
            if best_progress is not None:
                supports += 1
                max_progress = max(max_progress, best_progress)
        return supports, max_progress

    def _inside_boxes(
        self,
        center: tuple[int, int],
        bbox: tuple[int, int, int, int],
        boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        for box in boxes:
            if box[0] <= center[0] <= box[0] + box[2] and box[1] <= center[1] <= box[1] + box[3]:
                return True
            if bbox_iou(bbox, box) > 0.03:
                return True
        return False

    def _confirm(self, candidate: BallCandidate) -> BallState:
        if self.kalman is None:
            self.kalman = build_kalman(candidate.center)
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        self.kalman.correct(measurement)
        if self.last_confirmed_center is not None:
            jump = distance_between(candidate.center, self.last_confirmed_center)
            if jump > self.config.ball_recover_gate:
                self._break_trail()
        self.last_confirmed_center = candidate.center
        self.last_confirmed_radius = candidate.radius
        self.mode = "TRACK"
        self.miss_count = 0
        self.trail.append(candidate.center)
        return BallState(center=candidate.center, radius=candidate.radius, status="confirmed")

    def _handle_miss(
        self,
        observation: FrameObservation,
        frame_index: int,
        observations: list[FrameObservation],
        suppression_boxes: list[tuple[int, int, int, int]],
        predicted_center: Optional[tuple[int, int]],
        uncertainty: float,
    ) -> BallState:
        self.miss_count += 1

        if (
            predicted_center is not None
            and self._predicted_valid(predicted_center, suppression_boxes)
            and self.miss_count <= self.config.ball_coast_frames
            and uncertainty <= self.config.ball_max_uncertainty
        ):
            self.mode = "COAST"
            return BallState(center=predicted_center, radius=self.last_confirmed_radius, status="coast")

        if self.miss_count <= self.config.ball_recover_after_misses:
            self.mode = "COAST"
            return BallState(center=None, radius=None, status="missing")

        candidate = self._select_global_candidate(
            observation.ball_candidates,
            frame_index,
            observations,
            suppression_boxes,
            allow_yellow=True,
        )
        if candidate is not None:
            self.mode = "RECOVER"
            return self._confirm(candidate)

        if self.miss_count >= self.config.ball_lost_after_misses:
            self.mode = "LOST"
            self.kalman = None
            self.last_confirmed_center = None
            self.last_confirmed_radius = None
            self._break_trail()
            return BallState(center=None, radius=None, status="missing")

        self.mode = "RECOVER"
        return BallState(center=None, radius=None, status="missing")

    def _predicted_valid(
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

    def _break_trail(self) -> None:
        if not self.trail or self.trail[-1] is not None:
            self.trail.append(None)


def draw_players(frame: np.ndarray, tracks: list[PlayerTrack]) -> None:
    for track in tracks:
        x, y, w, h = track.bbox
        color = (0, 215, 255) if track.stable_team == "team_a" else (255, 170, 70)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ball(frame: np.ndarray, tracker: BallTrackerV4, state: BallState) -> None:
    points = list(tracker.trail)
    for previous, current in zip(points, points[1:]):
        if previous is None or current is None:
            continue
        cv2.line(frame, previous, current, (0, 210, 255), 3)
    if state.center is not None and state.radius is not None:
        color = (0, 235, 0) if state.status == "confirmed" else (0, 170, 255)
        radius = max(3, int(round(state.radius)))
        cv2.circle(frame, state.center, radius, color, 2)
        cv2.circle(frame, state.center, 2, color, -1)


def draw_overlay(frame: np.ndarray, team_a_count: int, team_b_count: int, fps: float) -> None:
    cv2.rectangle(frame, (18, 18), (210, 108), (30, 30, 30), -1)
    cv2.putText(frame, f"Team A: {team_a_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
    cv2.putText(frame, f"Team B: {team_b_count}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 170, 70), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)


def build_argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Strict OpenCV-only volleyball tracker v4")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "task phase 6" / "Volleyball.mp4",
        help="Path to the input volleyball video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "tp6" / "Volleyball_annotated_opencv_v4.mp4",
        help="Path to the output annotated video.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames during rendering.",
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

    collector = ObservationCollectorV4(config, frame_shape)
    observations, fps, width, height = collector.collect(args.input)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen input video: {args.input}")

    player_tracker = PlayerTrackerV4(config, frame_shape)
    ball_tracker = BallTrackerV4(config, frame_shape)
    render_start = time.time()
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        observation = observations[frame_index]
        visible_tracks, team_a_count, team_b_count, suppression_boxes = player_tracker.update(
            frame,
            observation.player_blobs,
            frame_index,
        )
        ball_state = ball_tracker.update(
            observation,
            suppression_boxes,
            frame_index,
            observations,
        )

        annotated = frame.copy()
        draw_players(annotated, visible_tracks)
        draw_ball(annotated, ball_tracker, ball_state)
        elapsed = max(time.time() - render_start, 1e-6)
        current_fps = (frame_index + 1) / elapsed
        draw_overlay(annotated, team_a_count, team_b_count, current_fps)
        writer.write(annotated)

        if args.display:
            cv2.imshow("Volleyball Tracker OpenCV V4", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    capture.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
