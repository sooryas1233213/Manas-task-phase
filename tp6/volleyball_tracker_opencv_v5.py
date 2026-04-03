from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=np.RankWarning)


@dataclass
class SceneConfigV5:
    trail_length: int = 32
    max_players_per_team: int = 6
    bg_warmup_frames: int = 30

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
    player_match_distance_top: float = 76.0
    player_match_distance_bottom: float = 100.0
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
    player_bg_history: int = 260
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
    player_team_history: int = 9
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

    stabilize_probe_frames: int = 120
    stabilize_feature_count: int = 1000
    stabilize_translation_threshold: float = 1.2
    stabilize_rotation_threshold_deg: float = 0.08
    stabilize_scale_threshold: float = 0.004

    ball_mog_history: int = 260
    ball_mog_var_threshold: int = 16
    ball_median_diff_threshold: int = 18
    ball_median_open_kernel: tuple[int, int] = (3, 3)
    ball_median_close_kernel: tuple[int, int] = (5, 5)
    ball_mog_open_kernel: tuple[int, int] = (3, 3)
    ball_mog_close_kernel: tuple[int, int] = (5, 5)
    ball_mog_dilate_kernel: tuple[int, int] = (3, 3)
    ball_candidate_keep_per_frame: int = 6
    ball_top_radius: float = 3.1
    ball_bottom_radius: float = 7.0
    ball_area_scale_min: float = 0.16
    ball_area_scale_max: float = 5.8
    ball_max_aspect_ratio: float = 2.35
    ball_min_compactness: float = 0.18
    ball_min_circularity: float = 0.06
    ball_seed_support_frames: int = 1
    ball_seed_search_gap: int = 2
    ball_seed_min_progress: float = 6.0
    ball_seed_max_progress: float = 130.0
    ball_track_search_gap: int = 4
    ball_lost_after_misses: int = 5
    ball_max_gate_per_step: float = 110.0
    ball_top_banner_guard_y: int = 152
    ball_segment_min_confirmed: int = 5
    ball_segment_merge_gap: int = 6
    ball_interp_short_gap: int = 3
    ball_interp_medium_gap: int = 6
    ball_interp_residual_limit: float = 18.0
    ball_poly_window: int = 7
    ball_poly_outlier_limit: float = 26.0
    ball_poly_mean_residual_limit: float = 18.0
    ball_ml_gap_trigger: int = 10
    ball_ml_sample_stride: int = 4
    ball_ml_roi_margin: int = 180
    ball_ml_tile_size: int = 640
    ball_ml_tile_overlap: int = 128
    ball_ml_conf_threshold: float = 0.08
    ball_ml_max_windows: int = 24
    ball_ml_imgsz: int = 960
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
class MotionProbeResult:
    enabled: bool
    mean_translation: float
    mean_rotation_deg: float
    mean_scale_delta: float


@dataclass
class BallCandidate:
    candidate_id: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area: float
    width: int
    height: int
    radius: float
    aspect_ratio: float
    compactness: float
    circularity: float
    weak_yellow_ratio: float
    source_median: bool
    source_mog: bool
    local_quality: float = 0.0
    support_count: int = 0
    support_progress: float = 0.0


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
    label_history: deque[str] = field(default_factory=lambda: deque(maxlen=9))
    stable_team: Optional[str] = None
    previous_center: Optional[tuple[int, int]] = None
    previous_footpoint: Optional[tuple[int, int]] = None


@dataclass
class VisiblePlayer:
    track_id: int
    bbox: tuple[int, int, int, int]
    side: str
    stable_team: Optional[str]
    confidence: float


@dataclass
class PlayerFrameState:
    visible_tracks: list[VisiblePlayer]
    team_a_count: int
    team_b_count: int
    suppression_boxes: list[tuple[int, int, int, int]]


@dataclass
class TrajectoryPoint:
    frame_index: int
    center: tuple[int, int]
    radius: float
    bbox: tuple[int, int, int, int]
    confidence: float
    status: str
    source: str


@dataclass
class BallSegment:
    confirmed_points: list[TrajectoryPoint]
    coast_points: list[TrajectoryPoint]
    score: float
    mean_residual: float
    support_score: float

    @property
    def start_frame(self) -> int:
        return self.confirmed_points[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.confirmed_points[-1].frame_index


@dataclass
class BallFrameState:
    center: Optional[tuple[int, int]] = None
    radius: Optional[float] = None
    status: str = "missing"
    confidence: float = 0.0
    source: str = ""


@dataclass
class RecoveryEvent:
    frame_index: int
    confidence: float
    source: str
    roi: tuple[int, int, int, int]


@dataclass
class BallDebugInfo:
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    overlap_frames: list[int] = field(default_factory=list)
    long_gaps: list[tuple[int, int]] = field(default_factory=list)
    flagged_frames: list[int] = field(default_factory=list)


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


def clip_bbox(bbox: tuple[int, int, int, int], frame_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    if len(frame_shape) == 3:
        height, width = frame_shape[:2]
    else:
        height, width = frame_shape
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


def compose_affine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.eye(3, dtype=np.float32)
    b = np.eye(3, dtype=np.float32)
    a[:2] = first
    b[:2] = second
    c = a @ b
    return c[:2].astype(np.float32)


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    mat = np.eye(3, dtype=np.float32)
    mat[:2] = matrix
    inv = np.linalg.inv(mat)
    return inv[:2].astype(np.float32)


def apply_transform(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def estimate_affine_between(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    orb = cv2.ORB_create(feature_count)
    kp_prev, desc_prev = orb.detectAndCompute(previous_gray, None)
    kp_curr, desc_curr = orb.detectAndCompute(gray, None)
    if desc_prev is None or desc_curr is None or len(kp_prev) < 12 or len(kp_curr) < 12:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_prev, desc_curr)
    if len(matches) < 12:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    matches = sorted(matches, key=lambda match: match.distance)[:220]
    points_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(
        points_prev,
        points_curr,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if matrix is None:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    return matrix.astype(np.float32)


def probe_camera_motion(input_path: Path, config: SceneConfigV5) -> MotionProbeResult:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video for motion probe: {input_path}")

    previous_gray: Optional[np.ndarray] = None
    translations: list[float] = []
    rotations: list[float] = []
    scales: list[float] = []
    frame_counter = 0

    while frame_counter < config.stabilize_probe_frames:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            matrix = estimate_affine_between(previous_gray, gray, config.stabilize_feature_count)
            translations.append(float(np.hypot(matrix[0, 2], matrix[1, 2])))
            rotations.append(float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))))
            scales.append(float(math.hypot(matrix[0, 0], matrix[1, 0]) - 1.0))
        previous_gray = gray
        frame_counter += 1

    capture.release()
    mean_translation = statistics.mean(translations) if translations else 0.0
    mean_rotation = statistics.mean(abs(value) for value in rotations) if rotations else 0.0
    mean_scale = statistics.mean(abs(value) for value in scales) if scales else 0.0
    enabled = bool(
        mean_translation > config.stabilize_translation_threshold
        or mean_rotation > config.stabilize_rotation_threshold_deg
        or mean_scale > config.stabilize_scale_threshold
    )
    return MotionProbeResult(
        enabled=enabled,
        mean_translation=mean_translation,
        mean_rotation_deg=mean_rotation,
        mean_scale_delta=mean_scale,
    )


def classify_team(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    expected_team: str,
    config: SceneConfigV5,
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


def expected_ball_radius(y: int, frame_height: int, config: SceneConfigV5) -> float:
    ratio = float(np.clip(y / max(frame_height - 1, 1), 0.0, 1.0))
    return config.ball_top_radius + ((config.ball_bottom_radius - config.ball_top_radius) * ratio)


def build_ball_kalman(
    start_center: tuple[int, int],
    next_center: tuple[int, int],
    delta_frames: int = 1,
) -> cv2.KalmanFilter:
    dt = float(max(delta_frames, 1))
    kalman = cv2.KalmanFilter(6, 2)
    kalman.transitionMatrix = np.array(
        [
            [1, 0, dt, 0, 0.5 * dt * dt, 0],
            [0, 1, 0, dt, 0, 0.5 * dt * dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.diag([0.15, 0.15, 0.35, 0.35, 0.02, 0.02]).astype(np.float32)
    kalman.measurementNoiseCov = np.diag([9.0, 9.0]).astype(np.float32)
    kalman.errorCovPost = np.eye(6, dtype=np.float32) * 24.0
    vx = (next_center[0] - start_center[0]) / dt
    vy = (next_center[1] - start_center[1]) / dt
    kalman.statePost = np.array(
        [
            [np.float32(next_center[0])],
            [np.float32(next_center[1])],
            [np.float32(vx)],
            [np.float32(vy)],
            [0.0],
            [0.0],
        ],
        dtype=np.float32,
    )
    return kalman


def advance_ball_kalman(kalman: cv2.KalmanFilter, steps: int) -> np.ndarray:
    prediction = kalman.statePost.copy()
    for _ in range(max(steps, 1)):
        prediction = kalman.predict()
    return prediction


def kalman_predicted_center(state: np.ndarray) -> tuple[int, int]:
    return (int(round(float(state[0][0]))), int(round(float(state[1][0]))))


def trajectory_fit_residual(points: list[TrajectoryPoint], candidate: Optional[TrajectoryPoint] = None) -> float:
    window = points[-6:]
    if candidate is not None:
        window = window + [candidate]
    if len(window) < 4:
        return 0.0
    frames = np.array([point.frame_index for point in window], dtype=np.float32)
    xs = np.array([point.center[0] for point in window], dtype=np.float32)
    ys = np.array([point.center[1] for point in window], dtype=np.float32)
    try:
        x_coeff = np.polyfit(frames, xs, 1)
        y_coeff = np.polyfit(frames, ys, 2)
    except np.linalg.LinAlgError:
        return 0.0
    x_fit = np.polyval(x_coeff, frames)
    y_fit = np.polyval(y_coeff, frames)
    residuals = np.sqrt(((x_fit - xs) ** 2) + ((y_fit - ys) ** 2))
    return float(residuals[-1] if candidate is not None else np.mean(residuals))


def candidate_corridor_roi(
    previous_state: Optional[BallFrameState],
    next_state: Optional[BallFrameState],
    frame_shape: tuple[int, int, int],
    margin: int,
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    points = [state.center for state in (previous_state, next_state) if state is not None and state.center is not None]
    if not points:
        return (0, 0, width, height)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(width, max(xs) + margin)
    y2 = min(height, max(ys) + margin)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


class ObservationCollectorV5:
    def __init__(self, config: SceneConfigV5, frame_shape: tuple[int, int, int]) -> None:
        self.config = config
        self.frame_shape = frame_shape
        mask_shape = frame_shape[:2]
        self.ball_mask = apply_exclusions(build_mask(mask_shape, config.ball_search_polygon), config.ball_exclusion_rects)
        self.top_mask = apply_exclusions(build_mask(mask_shape, config.top_court_polygon), config.player_exclusion_rects)
        self.bottom_mask = apply_exclusions(build_mask(mask_shape, config.bottom_court_polygon), config.player_exclusion_rects)

        self.ball_mog = cv2.createBackgroundSubtractorMOG2(
            history=config.ball_mog_history,
            varThreshold=config.ball_mog_var_threshold,
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
        self.ball_median_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_median_open_kernel)
        self.ball_median_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_median_close_kernel)
        self.ball_mog_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_mog_open_kernel)
        self.ball_mog_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_mog_close_kernel)
        self.ball_mog_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.ball_mog_dilate_kernel)
        self.top_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_open_kernel)
        self.top_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_top_close_kernel)
        self.top_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_top_dilate_kernel)
        self.bottom_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_open_kernel)
        self.bottom_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_bottom_close_kernel)
        self.bottom_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_bottom_dilate_kernel)

    def collect(
        self,
        input_path: Path,
        stabilize_mode: str,
    ) -> tuple[list[FrameObservation], list[np.ndarray], float, int, int, MotionProbeResult]:
        probe_result = probe_camera_motion(input_path, self.config)
        if stabilize_mode == "on":
            stabilization_enabled = True
        elif stabilize_mode == "off":
            stabilization_enabled = False
        else:
            stabilization_enabled = probe_result.enabled

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        observations: list[FrameObservation] = []
        transforms: list[np.ndarray] = []
        frame_index = 0
        previous_gray_unstabilized: Optional[np.ndarray] = None
        current_transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        ball_background: Optional[np.ndarray] = None

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if stabilization_enabled:
                if previous_gray_unstabilized is not None:
                    step = estimate_affine_between(
                        previous_gray_unstabilized,
                        raw_gray,
                        self.config.stabilize_feature_count,
                    )
                    current_transform = compose_affine(current_transform, invert_affine(step))
                stabilized_frame = apply_transform(frame, current_transform)
                transforms.append(current_transform.copy())
            else:
                stabilized_frame = frame
                transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
            previous_gray_unstabilized = raw_gray

            gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2HSV)

            if ball_background is None:
                ball_background = gray.astype(np.int16)

            median_mask = self._median_motion_mask(gray, ball_background)
            ball_background = self._update_approximate_median(ball_background, gray)
            ball_mog_mask = self._mog_motion_mask(gray)
            top_mask = self._player_foreground_mask(gray, self.top_mask, self.top_bg, "top")
            bottom_mask = self._player_foreground_mask(gray, self.bottom_mask, self.bottom_bg, "bottom")

            if frame_index < self.config.bg_warmup_frames:
                observations.append(FrameObservation(ball_candidates=[], player_blobs=[]))
            else:
                ball_candidates = self._extract_ball_candidates(median_mask, ball_mog_mask, hsv)
                player_blobs = self._extract_player_blobs(top_mask, "top") + self._extract_player_blobs(bottom_mask, "bottom")
                observations.append(
                    FrameObservation(
                        ball_candidates=ball_candidates,
                        player_blobs=self._merge_player_blobs(player_blobs),
                    )
                )
            frame_index += 1

        capture.release()
        return observations, transforms, fps, width, height, MotionProbeResult(
            enabled=stabilization_enabled,
            mean_translation=probe_result.mean_translation,
            mean_rotation_deg=probe_result.mean_rotation_deg,
            mean_scale_delta=probe_result.mean_scale_delta,
        )

    def _median_motion_mask(self, gray: np.ndarray, background: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gray, background.astype(np.uint8))
        _, mask = cv2.threshold(diff, self.config.ball_median_diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, self.ball_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.ball_median_open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.ball_median_close_kernel, iterations=1)
        return mask

    def _update_approximate_median(self, background: np.ndarray, gray: np.ndarray) -> np.ndarray:
        update_mask = self.ball_mask > 0
        increment = (gray.astype(np.int16) > background) & update_mask
        decrement = (gray.astype(np.int16) < background) & update_mask
        background[increment] += 1
        background[decrement] -= 1
        return background

    def _mog_motion_mask(self, gray: np.ndarray) -> np.ndarray:
        masked = cv2.bitwise_and(gray, gray, mask=self.ball_mask)
        fg_mask = self.ball_mog.apply(masked)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, self.ball_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.ball_mog_open_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.ball_mog_close_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self.ball_mog_dilate_kernel, iterations=1)
        return fg_mask

    def _player_foreground_mask(
        self,
        gray: np.ndarray,
        region_mask: np.ndarray,
        subtractor: cv2.BackgroundSubtractorMOG2,
        side: str,
    ) -> np.ndarray:
        masked = cv2.bitwise_and(gray, gray, mask=region_mask)
        fg_mask = subtractor.apply(masked)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.bitwise_and(fg_mask, region_mask)
        if side == "top":
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.top_open_kernel, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.top_close_kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, self.top_dilate_kernel, iterations=1)
        else:
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.bottom_open_kernel, iterations=1)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.bottom_close_kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, self.bottom_dilate_kernel, iterations=1)
        return fg_mask

    def _extract_ball_candidates(
        self,
        median_mask: np.ndarray,
        mog_mask: np.ndarray,
        hsv: np.ndarray,
    ) -> list[BallCandidate]:
        union_mask = cv2.bitwise_or(median_mask, mog_mask)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(union_mask, connectivity=8)
        candidates: list[BallCandidate] = []
        candidate_id = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox = clip_bbox((x, y, w, h), self.frame_shape)
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue
            component = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0 if perimeter <= 0 else float((4.0 * np.pi * area) / (perimeter * perimeter))
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
            center = (
                int(round(x + circle_x)),
                int(round(y + circle_y)),
            )
            if not point_in_mask(center, self.ball_mask):
                continue
            aspect_ratio = w / max(float(h), 1.0)
            compactness = area / max(float(w * h), 1.0)
            patch_hsv = hsv[y : y + h, x : x + w]
            weak_yellow_ratio = cv2.countNonZero(
                cv2.inRange(patch_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / max(float(w * h), 1.0)
            median_overlap = bool(np.any(cv2.bitwise_and(component, median_mask[y : y + h, x : x + w])))
            mog_overlap = bool(np.any(cv2.bitwise_and(component, mog_mask[y : y + h, x : x + w])))
            candidates.append(
                BallCandidate(
                    candidate_id=candidate_id,
                    center=center,
                    bbox=bbox,
                    area=float(area),
                    width=w,
                    height=h,
                    radius=float(radius),
                    aspect_ratio=aspect_ratio,
                    compactness=compactness,
                    circularity=circularity,
                    weak_yellow_ratio=weak_yellow_ratio,
                    source_median=median_overlap,
                    source_mog=mog_overlap,
                )
            )
            candidate_id += 1
        return candidates

    def _extract_player_blobs(self, fg_mask: np.ndarray, side: str) -> list[PlayerBlob]:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        blobs: list[PlayerBlob] = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox = clip_bbox((x, y, w, h), self.frame_shape)
            x, y, w, h = bbox
            component = (labels[y : y + h, x : x + w] == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if w < self.config.player_min_width or w > self.config.player_max_width:
                continue
            if h <= 0 or (w / max(float(h), 1.0)) > self.config.player_max_aspect_ratio:
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
                    area=area,
                    fill_ratio=fill_ratio,
                    solidity=solidity,
                    zone_hints=self._split_count_hint(bbox, side),
                )
            )
        return blobs

    def _split_count_hint(self, bbox: tuple[int, int, int, int], side: str) -> int:
        _, _, w, h = bbox
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


class PlayerTrackerV5:
    def __init__(self, config: SceneConfigV5, frame_shape: tuple[int, int, int]) -> None:
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
    ) -> PlayerFrameState:
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
        return PlayerFrameState(
            visible_tracks=[
                VisiblePlayer(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    side=track.side,
                    stable_team=track.stable_team,
                    confidence=track.confidence,
                )
                for track in visible_tracks
            ],
            team_a_count=team_a_count,
            team_b_count=team_b_count,
            suppression_boxes=suppression_boxes,
        )

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
                    predicted_center, predicted_foot = self._predicted_track_points(track)
                    center_distance = distance_between(blob.center, predicted_center)
                    foot_distance = distance_between(blob.footpoint, predicted_foot)
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

    def _predicted_track_points(self, track: PlayerTrack) -> tuple[tuple[int, int], tuple[int, int]]:
        if track.previous_center is None or track.previous_footpoint is None:
            return track.center, track.footpoint
        predicted_center = (
            int(round(track.center[0] + (track.center[0] - track.previous_center[0]))),
            int(round(track.center[1] + (track.center[1] - track.previous_center[1]))),
        )
        predicted_foot = (
            int(round(track.footpoint[0] + (track.footpoint[0] - track.previous_footpoint[0]))),
            int(round(track.footpoint[1] + (track.footpoint[1] - track.previous_footpoint[1]))),
        )
        return predicted_center, predicted_foot

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
        track.previous_center = track.center
        track.previous_footpoint = track.footpoint
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


def build_player_states(
    input_path: Path,
    observations: list[FrameObservation],
    transforms: list[np.ndarray],
    config: SceneConfigV5,
    frame_shape: tuple[int, int, int],
) -> list[PlayerFrameState]:
    tracker = PlayerTrackerV5(config, frame_shape)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen input video for player tracking: {input_path}")
    states: list[PlayerFrameState] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= len(observations):
            break
        transformed = apply_transform(frame, transforms[frame_index])
        states.append(tracker.update(transformed, observations[frame_index].player_blobs, frame_index))
        frame_index += 1
    capture.release()
    return states


class BallRecoveryModelV5:
    def __init__(
        self,
        model_path: Path,
        config: SceneConfigV5,
        disabled: bool,
    ) -> None:
        self.model_path = model_path
        self.config = config
        self.disabled = disabled
        self._model = None
        self._load_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return not self.disabled and self.model_path.exists()

    def _ensure_model(self):
        if self.disabled:
            return None
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            self._load_error = f"Missing recovery model: {self.model_path}"
            return None
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - import path depends on env
            self._load_error = f"Could not import ultralytics: {exc}"
            return None
        try:
            self._model = YOLO(str(self.model_path))
        except Exception as exc:  # pragma: no cover - model loading failure is env-specific
            self._load_error = f"Could not load recovery model: {exc}"
            self._model = None
        return self._model

    def detect(
        self,
        frame: np.ndarray,
        roi: Optional[tuple[int, int, int, int]],
        tile_full_frame: bool,
    ) -> Optional[tuple[tuple[int, int, int, int], float]]:
        model = self._ensure_model()
        if model is None:
            return None
        if roi is not None:
            bbox, confidence = self._predict_single(model, frame, roi)
            if bbox is not None:
                return bbox, confidence
        if tile_full_frame:
            return self._predict_tiled(model, frame)
        return None

    def _predict_single(
        self,
        model,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> tuple[Optional[tuple[int, int, int, int]], float]:
        x, y, w, h = clip_bbox(roi, frame.shape)
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return None, 0.0
        results = model.predict(
            source=crop,
            verbose=False,
            imgsz=self.config.ball_ml_imgsz,
            conf=self.config.ball_ml_conf_threshold,
            classes=[0],
        )
        best_bbox: Optional[tuple[int, int, int, int]] = None
        best_conf = 0.0
        for result in results:
            if getattr(result, "boxes", None) is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence <= best_conf:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
                bbox = clip_bbox((x + x1, y + y1, max(1, x2 - x1), max(1, y2 - y1)), frame.shape)
                best_bbox = bbox
                best_conf = confidence
        return best_bbox, best_conf

    def _predict_tiled(
        self,
        model,
        frame: np.ndarray,
    ) -> Optional[tuple[tuple[int, int, int, int], float]]:
        tile = self.config.ball_ml_tile_size
        step = max(64, tile - self.config.ball_ml_tile_overlap)
        best_bbox: Optional[tuple[int, int, int, int]] = None
        best_conf = 0.0
        for y in range(0, max(1, frame.shape[0] - tile + step), step):
            for x in range(0, max(1, frame.shape[1] - tile + step), step):
                roi = clip_bbox((x, y, tile, tile), frame.shape)
                bbox, confidence = self._predict_single(model, frame, roi)
                if bbox is not None and confidence > best_conf:
                    best_bbox = bbox
                    best_conf = confidence
        if best_bbox is None:
            return None
        return best_bbox, best_conf


class BallTrajectoryTrackerV5:
    def __init__(
        self,
        config: SceneConfigV5,
        frame_shape: tuple[int, int, int],
        fps: float,
        observations: list[FrameObservation],
        player_states: list[PlayerFrameState],
        input_path: Path,
        transforms: list[np.ndarray],
        recovery_model: BallRecoveryModelV5,
    ) -> None:
        self.config = config
        self.frame_shape = frame_shape
        self.fps = fps
        self.observations = observations
        self.player_states = player_states
        self.input_path = input_path
        self.transforms = transforms
        self.recovery_model = recovery_model
        self.ball_mask = apply_exclusions(build_mask(frame_shape[:2], config.ball_search_polygon), config.ball_exclusion_rects)
        self.candidates_by_frame: list[list[BallCandidate]] = []
        self.debug_info = BallDebugInfo()

    def build(self) -> tuple[list[BallFrameState], BallDebugInfo]:
        self._prepare_candidates()
        segments = self._build_segments()
        segments = self._merge_segments(segments)
        if self.recovery_model.enabled:
            segments = self._apply_recovery_ml(segments)
            segments = self._merge_segments(segments)
        frame_states = self._states_from_segments(segments)
        self._finalize_debug(frame_states)
        return frame_states, self.debug_info

    def _prepare_candidates(self) -> None:
        filtered: list[list[BallCandidate]] = []
        height = self.frame_shape[0]
        for frame_index, observation in enumerate(self.observations):
            candidates: list[BallCandidate] = []
            suppression_boxes = self.player_states[frame_index].suppression_boxes if frame_index < len(self.player_states) else []
            for candidate in observation.ball_candidates:
                if not self._passes_hard_sieves(candidate, suppression_boxes, height):
                    continue
                candidate.local_quality = self._local_quality(candidate, height)
                candidates.append(candidate)
            candidates.sort(key=lambda item: item.local_quality, reverse=True)
            filtered.append(candidates[: self.config.ball_candidate_keep_per_frame])
        self.candidates_by_frame = filtered

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            for candidate in frame_candidates:
                support_count, support_progress = self._temporal_support(frame_index, candidate)
                candidate.support_count = support_count
                candidate.support_progress = support_progress

    def _passes_hard_sieves(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        frame_height: int,
    ) -> bool:
        expected_radius = expected_ball_radius(candidate.center[1], frame_height, self.config)
        expected_area = math.pi * expected_radius * expected_radius
        if candidate.area < max(3.0, expected_area * self.config.ball_area_scale_min):
            return False
        if candidate.area > max(26.0, expected_area * self.config.ball_area_scale_max):
            return False
        ratio = candidate.aspect_ratio
        if ratio > self.config.ball_max_aspect_ratio or (1.0 / max(ratio, 1e-6)) > self.config.ball_max_aspect_ratio:
            return False
        if candidate.compactness < self.config.ball_min_compactness:
            return False
        if candidate.circularity < self.config.ball_min_circularity:
            return False
        if candidate.center[1] < self.config.ball_top_banner_guard_y and not candidate.source_median:
            return False
        if self._candidate_inside_suppression(candidate, suppression_boxes) and candidate.radius > expected_radius * 0.85:
            return False
        return True

    def _candidate_inside_suppression(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        for box in suppression_boxes:
            if box[0] <= candidate.center[0] <= box[0] + box[2] and box[1] <= candidate.center[1] <= box[1] + box[3]:
                return True
            if bbox_iou(candidate.bbox, box) > 0.08:
                return True
        return False

    def _local_quality(self, candidate: BallCandidate, frame_height: int) -> float:
        expected_radius = expected_ball_radius(candidate.center[1], frame_height, self.config)
        score = 0.0
        score += candidate.circularity * 3.8
        score += candidate.compactness * 2.8
        score += max(0.0, 2.8 - abs(candidate.radius - expected_radius))
        score += candidate.weak_yellow_ratio * 1.8
        if candidate.source_median:
            score += 1.6
        if candidate.source_mog:
            score += 1.1
        if candidate.source_median and candidate.source_mog:
            score += 1.4
        return score

    def _temporal_support(self, frame_index: int, candidate: BallCandidate) -> tuple[int, float]:
        supports = 0
        max_progress = 0.0
        for delta in range(1, self.config.ball_seed_search_gap + 1):
            future_index = frame_index + delta
            if future_index >= len(self.candidates_by_frame):
                break
            best_distance: Optional[float] = None
            for future_candidate in self.candidates_by_frame[future_index]:
                distance = distance_between(candidate.center, future_candidate.center)
                if distance < self.config.ball_seed_min_progress * delta:
                    continue
                if distance > self.config.ball_seed_max_progress * delta:
                    continue
                if best_distance is None or distance < best_distance:
                    best_distance = distance
            if best_distance is not None:
                supports += 1
                max_progress = max(max_progress, best_distance)
        return supports, max_progress

    def _build_segments(self) -> list[BallSegment]:
        segments: list[BallSegment] = []
        used_candidates: set[tuple[int, int]] = set()
        frame_index = self.config.bg_warmup_frames
        while frame_index < len(self.candidates_by_frame):
            seeds = [
                candidate
                for candidate in self.candidates_by_frame[frame_index]
                if candidate.support_count >= self.config.ball_seed_support_frames
                and (frame_index, candidate.candidate_id) not in used_candidates
            ]
            seeds.sort(key=lambda item: (item.support_count, item.local_quality, item.support_progress), reverse=True)
            best_segment: Optional[BallSegment] = None
            for seed in seeds[:3]:
                segment = self._grow_segment_from_seed(frame_index, seed, used_candidates)
                if segment is None:
                    continue
                if best_segment is None or segment.score > best_segment.score:
                    best_segment = segment
            if best_segment is not None:
                segments.append(best_segment)
                for point in best_segment.confirmed_points:
                    candidate_id = self._find_candidate_id(point.frame_index, point.center)
                    if candidate_id is not None:
                        used_candidates.add((point.frame_index, candidate_id))
                frame_index = best_segment.end_frame + 1
            else:
                frame_index += 1
        return segments

    def _find_candidate_id(self, frame_index: int, center: tuple[int, int]) -> Optional[int]:
        for candidate in self.candidates_by_frame[frame_index]:
            if candidate.center == center:
                return candidate.candidate_id
        return None

    def _grow_segment_from_seed(
        self,
        frame_index: int,
        seed: BallCandidate,
        used_candidates: set[tuple[int, int]],
    ) -> Optional[BallSegment]:
        partner = self._seed_partner(frame_index, seed, used_candidates)
        if partner is None:
            return None
        partner_frame, partner_candidate = partner
        kalman = build_ball_kalman(seed.center, partner_candidate.center, max(partner_frame - frame_index, 1))
        confirmed_points = [
            self._point_from_candidate(frame_index, seed, "seed"),
            self._point_from_candidate(partner_frame, partner_candidate, "confirmed"),
        ]
        coast_points: list[TrajectoryPoint] = []
        current_frame = partner_frame
        last_radius = partner_candidate.radius
        misses = 0

        while current_frame < len(self.candidates_by_frame) - 1 and misses <= self.config.ball_lost_after_misses:
            best = self._best_extension(current_frame, kalman, confirmed_points, used_candidates)
            if best is None:
                prediction = advance_ball_kalman(kalman, 1)
                current_frame += 1
                misses += 1
                if misses <= 2:
                    coast_points.append(
                        TrajectoryPoint(
                            frame_index=current_frame,
                            center=kalman_predicted_center(prediction),
                            radius=last_radius,
                            bbox=(kalman_predicted_center(prediction)[0] - 3, kalman_predicted_center(prediction)[1] - 3, 6, 6),
                            confidence=0.18,
                            status="coast",
                            source="kalman",
                        )
                    )
                continue

            target_frame, candidate, predicted_state = best
            delta = target_frame - current_frame
            predicted_center = kalman_predicted_center(predicted_state)
            innovation = distance_between(predicted_center, candidate.center)
            scale = min(10.0, 1.0 + (innovation / 10.0))
            kalman.processNoiseCov = np.diag(
                [0.15 * scale, 0.15 * scale, 0.35 * scale, 0.35 * scale, 0.02 * scale, 0.02 * scale]
            ).astype(np.float32)
            advance_ball_kalman(kalman, max(delta, 1))
            measurement = np.array(
                [[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]],
                dtype=np.float32,
            )
            kalman.correct(measurement)
            confirmed_points.append(self._point_from_candidate(target_frame, candidate, "confirmed"))
            current_frame = target_frame
            last_radius = candidate.radius
            misses = 0

        segment = self._score_segment(confirmed_points, coast_points)
        if segment is None:
            return None
        return segment

    def _seed_partner(
        self,
        frame_index: int,
        seed: BallCandidate,
        used_candidates: set[tuple[int, int]],
    ) -> Optional[tuple[int, BallCandidate]]:
        best_choice: Optional[tuple[int, BallCandidate]] = None
        best_score = float("inf")
        for delta in range(1, self.config.ball_seed_search_gap + 1):
            future_index = frame_index + delta
            if future_index >= len(self.candidates_by_frame):
                break
            for candidate in self.candidates_by_frame[future_index]:
                if (future_index, candidate.candidate_id) in used_candidates:
                    continue
                distance = distance_between(seed.center, candidate.center)
                if distance < self.config.ball_seed_min_progress * delta:
                    continue
                if distance > self.config.ball_seed_max_progress * delta:
                    continue
                score = distance - (candidate.local_quality * 6.0) - (candidate.support_count * 4.0)
                if score < best_score:
                    best_score = score
                    best_choice = (future_index, candidate)
        return best_choice

    def _best_extension(
        self,
        current_frame: int,
        kalman: cv2.KalmanFilter,
        confirmed_points: list[TrajectoryPoint],
        used_candidates: set[tuple[int, int]],
    ) -> Optional[tuple[int, BallCandidate, np.ndarray]]:
        best_choice: Optional[tuple[int, BallCandidate, np.ndarray]] = None
        best_cost = float("inf")
        state_post = kalman.statePost.copy()
        for delta in range(1, self.config.ball_track_search_gap + 1):
            future_frame = current_frame + delta
            if future_frame >= len(self.candidates_by_frame):
                break
            predicted_state = self._project_state(state_post, delta)
            predicted_center = kalman_predicted_center(predicted_state)
            gate = self.config.ball_max_gate_per_step * (1.0 + (0.55 * (delta - 1)))
            for candidate in self.candidates_by_frame[future_frame]:
                if (future_frame, candidate.candidate_id) in used_candidates:
                    continue
                distance = distance_between(predicted_center, candidate.center)
                if distance > gate:
                    continue
                trajectory_penalty = trajectory_fit_residual(
                    confirmed_points,
                    self._point_from_candidate(future_frame, candidate, "candidate"),
                )
                if trajectory_penalty > self.config.ball_poly_outlier_limit and candidate.local_quality < 6.0:
                    continue
                velocity_penalty = 0.0
                if len(confirmed_points) >= 2:
                    last = confirmed_points[-1]
                    previous = confirmed_points[-2]
                    last_dt = max(1, last.frame_index - previous.frame_index)
                    expected_vx = (last.center[0] - previous.center[0]) / last_dt
                    expected_vy = (last.center[1] - previous.center[1]) / last_dt
                    candidate_vx = (candidate.center[0] - last.center[0]) / max(delta, 1)
                    candidate_vy = (candidate.center[1] - last.center[1]) / max(delta, 1)
                    velocity_penalty = abs(candidate_vx - expected_vx) + abs(candidate_vy - expected_vy)
                expected_radius = expected_ball_radius(candidate.center[1], self.frame_shape[0], self.config)
                size_penalty = abs(candidate.radius - expected_radius) * 2.0
                source_bonus = 0.0
                if candidate.source_median:
                    source_bonus += 6.0
                if candidate.source_mog:
                    source_bonus += 3.0
                if candidate.source_median and candidate.source_mog:
                    source_bonus += 4.0
                torso_penalty = 20.0 if self._candidate_inside_suppression(candidate, self.player_states[future_frame].suppression_boxes) else 0.0
                cost = (
                    distance * 0.65
                    + (delta - 1) * 10.0
                    + velocity_penalty * 0.12
                    + size_penalty
                    + trajectory_penalty * 1.25
                    + torso_penalty
                    - (candidate.local_quality * 5.5)
                    - source_bonus
                    - (candidate.support_count * 4.0)
                )
                if cost < best_cost:
                    best_cost = cost
                    best_choice = (future_frame, candidate, predicted_state)
        if best_cost > 80.0:
            return None
        return best_choice

    def _project_state(self, state: np.ndarray, delta: int) -> np.ndarray:
        delta = max(delta, 1)
        x = float(state[0][0])
        y = float(state[1][0])
        vx = float(state[2][0])
        vy = float(state[3][0])
        ax = float(state[4][0])
        ay = float(state[5][0])
        projected = np.array(
            [
                [x + (vx * delta) + (0.5 * ax * delta * delta)],
                [y + (vy * delta) + (0.5 * ay * delta * delta)],
                [vx + (ax * delta)],
                [vy + (ay * delta)],
                [ax],
                [ay],
            ],
            dtype=np.float32,
        )
        return projected

    def _point_from_candidate(self, frame_index: int, candidate: BallCandidate, status: str) -> TrajectoryPoint:
        source_parts = []
        if candidate.source_median:
            source_parts.append("median")
        if candidate.source_mog:
            source_parts.append("mog")
        if not source_parts:
            source_parts.append("unknown")
        return TrajectoryPoint(
            frame_index=frame_index,
            center=candidate.center,
            radius=candidate.radius,
            bbox=candidate.bbox,
            confidence=max(0.1, candidate.local_quality + (candidate.support_count * 0.6)),
            status=status,
            source="+".join(source_parts),
        )

    def _score_segment(
        self,
        confirmed_points: list[TrajectoryPoint],
        coast_points: list[TrajectoryPoint],
    ) -> Optional[BallSegment]:
        if len(confirmed_points) < self.config.ball_segment_min_confirmed:
            return None
        confirmed_points = sorted(confirmed_points, key=lambda item: item.frame_index)
        pruned = confirmed_points[:]
        while len(pruned) > self.config.ball_segment_min_confirmed:
            residuals = self._point_residuals(pruned)
            worst_index = int(np.argmax(residuals))
            if residuals[worst_index] <= self.config.ball_poly_outlier_limit:
                break
            del pruned[worst_index]
        residuals = self._point_residuals(pruned)
        mean_residual = float(np.mean(residuals)) if residuals else 0.0
        if mean_residual > self.config.ball_poly_mean_residual_limit and len(pruned) < 9:
            return None
        gap_penalty = 0.0
        for previous, current in zip(pruned, pruned[1:]):
            gap_penalty += max(0, current.frame_index - previous.frame_index - 1) * 2.5
        support_score = sum(point.confidence for point in pruned)
        score = (len(pruned) * 12.0) + support_score - gap_penalty - (mean_residual * 1.3)
        return BallSegment(
            confirmed_points=pruned,
            coast_points=coast_points,
            score=score,
            mean_residual=mean_residual,
            support_score=support_score,
        )

    def _point_residuals(self, points: list[TrajectoryPoint]) -> list[float]:
        if len(points) < 4:
            return [0.0 for _ in points]
        frames = np.array([point.frame_index for point in points], dtype=np.float32)
        xs = np.array([point.center[0] for point in points], dtype=np.float32)
        ys = np.array([point.center[1] for point in points], dtype=np.float32)
        try:
            x_coeff = np.polyfit(frames, xs, 1)
            y_coeff = np.polyfit(frames, ys, 2)
        except np.linalg.LinAlgError:
            return [0.0 for _ in points]
        x_fit = np.polyval(x_coeff, frames)
        y_fit = np.polyval(y_coeff, frames)
        residuals = np.sqrt(((x_fit - xs) ** 2) + ((y_fit - ys) ** 2))
        return [float(value) for value in residuals]

    def _merge_segments(self, segments: list[BallSegment]) -> list[BallSegment]:
        if not segments:
            return []
        segments = sorted(segments, key=lambda item: (item.start_frame, -item.score))
        merged: list[BallSegment] = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            gap = segment.start_frame - previous.end_frame
            if gap <= self.config.ball_segment_merge_gap:
                end_point = previous.confirmed_points[-1]
                start_point = segment.confirmed_points[0]
                if distance_between(end_point.center, start_point.center) <= self.config.ball_max_gate_per_step * max(1, gap):
                    merged_points = previous.confirmed_points + segment.confirmed_points
                    merged_coast = previous.coast_points + segment.coast_points
                    rescored = self._score_segment(merged_points, merged_coast)
                    if rescored is not None:
                        merged[-1] = rescored
                        continue
            merged.append(segment)
        return merged

    def _apply_recovery_ml(self, segments: list[BallSegment]) -> list[BallSegment]:
        states = self._states_from_segments(segments)
        windows = self._long_gap_windows(states)
        if not windows:
            return segments
        recovered_segments: list[BallSegment] = list(segments)
        used_candidates: set[tuple[int, int]] = set()
        for segment in segments:
            for point in segment.confirmed_points:
                candidate_id = self._find_candidate_id(point.frame_index, point.center)
                if candidate_id is not None:
                    used_candidates.add((point.frame_index, candidate_id))
        for window_index, (start_frame, end_frame) in enumerate(windows[: self.config.ball_ml_max_windows]):
            previous_state = states[start_frame - 1] if start_frame > 0 else None
            next_state = states[end_frame + 1] if end_frame + 1 < len(states) else None
            roi = candidate_corridor_roi(previous_state, next_state, self.frame_shape, self.config.ball_ml_roi_margin)
            sample_frames = self._sample_gap_frames(start_frame, end_frame)
            for frame_index in sample_frames:
                frame = self._read_transformed_frame(frame_index)
                if frame is None:
                    continue
                result = self.recovery_model.detect(frame, roi, tile_full_frame=True)
                if result is None:
                    continue
                bbox, confidence = result
                center = bbox_center(bbox)
                if not point_in_mask(center, self.ball_mask):
                    continue
                if self._bbox_inside_suppression(bbox, self.player_states[frame_index].suppression_boxes):
                    continue
                anchor_frame, anchor_candidate = self._nearest_candidate_or_anchor(frame_index, bbox, confidence)
                if anchor_candidate is None:
                    continue
                recovered = self._grow_segment_from_seed(anchor_frame, anchor_candidate, used_candidates)
                if recovered is None:
                    continue
                recovered_segments.append(recovered)
                for point in recovered.confirmed_points:
                    candidate_id = self._find_candidate_id(point.frame_index, point.center)
                    if candidate_id is not None:
                        used_candidates.add((point.frame_index, candidate_id))
                self.debug_info.recovery_events.append(
                    RecoveryEvent(
                        frame_index=frame_index,
                        confidence=confidence,
                        source="recovery_ml",
                        roi=roi,
                    )
                )
                break
        return recovered_segments

    def _long_gap_windows(self, states: list[BallFrameState]) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        start: Optional[int] = None
        for index, state in enumerate(states):
            if state.center is None:
                if start is None:
                    start = index
            else:
                if start is not None and index - start >= self.config.ball_ml_gap_trigger:
                    windows.append((start, index - 1))
                start = None
        if start is not None and len(states) - start >= self.config.ball_ml_gap_trigger:
            windows.append((start, len(states) - 1))
        return windows

    def _sample_gap_frames(self, start_frame: int, end_frame: int) -> list[int]:
        if end_frame <= start_frame:
            return [start_frame]
        span = end_frame - start_frame + 1
        mid = start_frame + (span // 2)
        frames = {mid}
        if span > self.config.ball_ml_sample_stride * 2:
            frames.add(start_frame + (span // 3))
            frames.add(start_frame + ((2 * span) // 3))
        return sorted(frames)

    def _read_transformed_frame(self, frame_index: int) -> Optional[np.ndarray]:
        capture = cv2.VideoCapture(str(self.input_path))
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            return None
        return apply_transform(frame, self.transforms[frame_index])

    def _bbox_inside_suppression(
        self,
        bbox: tuple[int, int, int, int],
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        center = bbox_center(bbox)
        for box in suppression_boxes:
            if box[0] <= center[0] <= box[0] + box[2] and box[1] <= center[1] <= box[1] + box[3]:
                return True
        return False

    def _nearest_candidate_or_anchor(
        self,
        frame_index: int,
        bbox: tuple[int, int, int, int],
        confidence: float,
    ) -> tuple[int, Optional[BallCandidate]]:
        center = bbox_center(bbox)
        best_candidate: Optional[BallCandidate] = None
        best_frame_index = frame_index
        best_distance = 28.0
        for delta in (0, -1, 1):
            candidate_frame = frame_index + delta
            if candidate_frame < 0 or candidate_frame >= len(self.candidates_by_frame):
                continue
            for candidate in self.candidates_by_frame[candidate_frame]:
                distance = distance_between(center, candidate.center)
                if distance < best_distance:
                    best_distance = distance
                    best_frame_index = candidate_frame
                    best_candidate = candidate
        if best_candidate is not None:
            return best_frame_index, best_candidate
        x, y, w, h = bbox
        radius = max(2.0, min(w, h) / 2.0)
        return frame_index, BallCandidate(
            candidate_id=1000000 + frame_index,
            center=center,
            bbox=bbox,
            area=float(w * h),
            width=w,
            height=h,
            radius=radius,
            aspect_ratio=w / max(float(h), 1.0),
            compactness=0.5,
            circularity=0.45,
            weak_yellow_ratio=0.0,
            source_median=False,
            source_mog=False,
            local_quality=7.5 + confidence,
            support_count=1,
            support_progress=0.0,
        )

    def _states_from_segments(self, segments: list[BallSegment]) -> list[BallFrameState]:
        states = [BallFrameState() for _ in range(len(self.observations))]
        for segment in sorted(segments, key=lambda item: item.score, reverse=True):
            for point in segment.confirmed_points:
                if self._center_hits_torso(point.frame_index, point.center):
                    continue
                state = states[point.frame_index]
                if point.confidence > state.confidence:
                    states[point.frame_index] = BallFrameState(
                        center=point.center,
                        radius=point.radius,
                        status="confirmed",
                        confidence=point.confidence,
                        source=point.source,
                    )
            self._interpolate_segment(segment, states)
            for point in segment.coast_points:
                if self._center_hits_torso(point.frame_index, point.center):
                    continue
                state = states[point.frame_index]
                if state.center is None and point.confidence > state.confidence:
                    states[point.frame_index] = BallFrameState(
                        center=point.center,
                        radius=point.radius,
                        status="coast",
                        confidence=point.confidence,
                        source=point.source,
                    )
        return states

    def _interpolate_segment(self, segment: BallSegment, states: list[BallFrameState]) -> None:
        points = segment.confirmed_points
        for previous, current in zip(points, points[1:]):
            gap = current.frame_index - previous.frame_index - 1
            if gap <= 0:
                continue
            endpoint_distance = distance_between(previous.center, current.center)
            max_endpoint_distance = self.config.ball_max_gate_per_step * max(gap + 1, 1)
            if endpoint_distance > max_endpoint_distance:
                continue
            if gap <= self.config.ball_interp_short_gap:
                for step in range(1, gap + 1):
                    ratio = step / float(gap + 1)
                    frame_index = previous.frame_index + step
                    center = (
                        int(round((previous.center[0] * (1.0 - ratio)) + (current.center[0] * ratio))),
                        int(round((previous.center[1] * (1.0 - ratio)) + (current.center[1] * ratio))),
                    )
                    if self._center_hits_torso(frame_index, center):
                        continue
                    states[frame_index] = BallFrameState(
                        center=center,
                        radius=(previous.radius * (1.0 - ratio)) + (current.radius * ratio),
                        status="interpolated",
                        confidence=min(previous.confidence, current.confidence) * 0.7,
                        source="interp_short",
                    )
            elif gap <= self.config.ball_interp_medium_gap and segment.mean_residual <= self.config.ball_interp_residual_limit:
                window = points[max(0, points.index(previous) - 2) : min(len(points), points.index(current) + 3)]
                if len(window) < 4:
                    continue
                frames = np.array([point.frame_index for point in window], dtype=np.float32)
                xs = np.array([point.center[0] for point in window], dtype=np.float32)
                ys = np.array([point.center[1] for point in window], dtype=np.float32)
                try:
                    x_coeff = np.polyfit(frames, xs, 1)
                    y_coeff = np.polyfit(frames, ys, 2)
                except np.linalg.LinAlgError:
                    continue
                for frame_index in range(previous.frame_index + 1, current.frame_index):
                    center = (
                        int(round(float(np.polyval(x_coeff, frame_index)))),
                        int(round(float(np.polyval(y_coeff, frame_index)))),
                    )
                    if self._center_hits_torso(frame_index, center):
                        continue
                    states[frame_index] = BallFrameState(
                        center=center,
                        radius=(previous.radius + current.radius) / 2.0,
                        status="interpolated",
                        confidence=min(previous.confidence, current.confidence) * 0.55,
                        source="interp_poly",
                    )

    def _center_hits_torso(self, frame_index: int, center: tuple[int, int]) -> bool:
        for box in self.player_states[frame_index].suppression_boxes:
            if box[0] <= center[0] <= box[0] + box[2] and box[1] <= center[1] <= box[1] + box[3]:
                return True
        return False

    def _finalize_debug(self, states: list[BallFrameState]) -> None:
        gap_start: Optional[int] = None
        flagged: set[int] = set()
        for index, state in enumerate(states):
            if state.center is None:
                if gap_start is None:
                    gap_start = index
            else:
                if gap_start is not None and index - gap_start >= self.config.ball_ml_gap_trigger:
                    self.debug_info.long_gaps.append((gap_start, index - 1))
                    flagged.update({gap_start, gap_start + ((index - gap_start) // 2), index - 1})
                gap_start = None
            if state.center is not None and self._center_hits_torso(index, state.center):
                self.debug_info.overlap_frames.append(index)
                flagged.add(index)
        if gap_start is not None and len(states) - gap_start >= self.config.ball_ml_gap_trigger:
            self.debug_info.long_gaps.append((gap_start, len(states) - 1))
            flagged.update({gap_start, gap_start + ((len(states) - gap_start) // 2), len(states) - 1})
        for event in self.debug_info.recovery_events:
            flagged.add(event.frame_index)
        self.debug_info.flagged_frames = sorted(flagged)


def draw_players(frame: np.ndarray, tracks: list[VisiblePlayer]) -> None:
    for track in tracks:
        x, y, w, h = track.bbox
        color = (0, 215, 255) if track.stable_team == "team_a" else (255, 170, 70)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ball(frame: np.ndarray, states: list[BallFrameState], frame_index: int) -> None:
    start = max(0, frame_index - 80)
    max_trail_step = 90.0
    trail_points: list[tuple[int, int]] = []
    for state in states[start : frame_index + 1]:
        if state.center is None:
            trail_points.append((-1, -1))
        elif state.status in {"confirmed", "interpolated"}:
            trail_points.append(state.center)
        else:
            trail_points.append((-1, -1))
    for previous, current in zip(trail_points, trail_points[1:]):
        if previous[0] < 0 or current[0] < 0:
            continue
        if distance_between(previous, current) > max_trail_step:
            continue
        cv2.line(frame, previous, current, (0, 210, 255), 3)
    state = states[frame_index]
    if state.center is None or state.radius is None:
        return
    if state.status == "confirmed":
        color = (0, 235, 0)
    elif state.status == "interpolated":
        color = (0, 170, 255)
    else:
        color = (180, 180, 180)
    radius = max(3, int(round(state.radius)))
    cv2.circle(frame, state.center, radius, color, 2)
    cv2.circle(frame, state.center, 2, color, -1)


def draw_overlay(frame: np.ndarray, team_a_count: int, team_b_count: int, fps: float, stabilization_text: str) -> None:
    cv2.rectangle(frame, (18, 18), (248, 120), (30, 30, 30), -1)
    cv2.putText(frame, f"Team A: {team_a_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
    cv2.putText(frame, f"Team B: {team_b_count}", (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 170, 70), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (220, 220, 220), 2)
    cv2.putText(frame, stabilization_text, (132, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def save_debug_artifacts(
    debug_dir: Path,
    input_path: Path,
    transforms: list[np.ndarray],
    ball_states: list[BallFrameState],
    debug_info: BallDebugInfo,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    csv_path = debug_dir / "ball_report.csv"
    recovery_frames = {event.frame_index for event in debug_info.recovery_events}
    overlap_frames = set(debug_info.overlap_frames)
    gap_frames: set[int] = set()
    for start, end in debug_info.long_gaps:
        gap_frames.update(range(start, end + 1))
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "status", "confidence", "source", "recovery_ml", "torso_overlap", "long_gap"])
        for frame_index, state in enumerate(ball_states):
            writer.writerow(
                [
                    frame_index,
                    state.status,
                    f"{state.confidence:.3f}",
                    state.source,
                    int(frame_index in recovery_frames),
                    int(frame_index in overlap_frames),
                    int(frame_index in gap_frames),
                ]
            )

    selected_frames = debug_info.flagged_frames[:18]
    if not selected_frames:
        return
    capture = cv2.VideoCapture(str(input_path))
    images: list[np.ndarray] = []
    for frame_index in selected_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        transformed = apply_transform(frame, transforms[frame_index])
        state = ball_states[frame_index]
        label = f"F{frame_index} {state.status}"
        cv2.putText(transformed, label, (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if state.center is not None and state.radius is not None:
            cv2.circle(transformed, state.center, max(3, int(round(state.radius))), (0, 255, 0), 2)
        images.append(cv2.resize(transformed, (320, 180)))
    capture.release()
    if not images:
        return
    columns = 3
    rows = int(math.ceil(len(images) / columns))
    blank = np.zeros_like(images[0])
    while len(images) < rows * columns:
        images.append(blank.copy())
    strips = []
    for row in range(rows):
        strips.append(cv2.hconcat(images[row * columns : (row + 1) * columns]))
    sheet = cv2.vconcat(strips)
    cv2.imwrite(str(debug_dir / "contact_sheet.png"), sheet)


def build_argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Trajectory-first volleyball tracker v5")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "task phase 6" / "Volleyball.mp4",
        help="Path to the input volleyball video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "tp6" / "Volleyball_annotated_opencv_v5.mp4",
        help="Path to the output annotated video.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the annotated frames during rendering.",
    )
    parser.add_argument(
        "--stabilize",
        choices=("auto", "on", "off"),
        default="auto",
        help="Enable ORB-based global motion stabilization before tracking.",
    )
    parser.add_argument(
        "--ball-recovery-model",
        type=Path,
        default=root / "task phase 6" / "models" / "ball_best.pt",
        help="Path to the sparse recovery-only ball detector.",
    )
    parser.add_argument(
        "--disable-ball-recovery-ml",
        action="store_true",
        help="Disable the recovery-only ball detector and keep v5 classical end-to-end.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for v5 debug artifacts.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = SceneConfigV5()

    probe = cv2.VideoCapture(str(args.input))
    if not probe.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")
    frame_shape = (
        int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
        3,
    )
    probe.release()

    collector = ObservationCollectorV5(config, frame_shape)
    observations, transforms, fps, width, height, motion_probe = collector.collect(args.input, args.stabilize)
    player_states = build_player_states(args.input, observations, transforms, config, frame_shape)
    recovery_model = BallRecoveryModelV5(args.ball_recovery_model, config, args.disable_ball_recovery_ml)
    ball_tracker = BallTrajectoryTrackerV5(
        config,
        frame_shape,
        fps,
        observations,
        player_states,
        args.input,
        transforms,
        recovery_model,
    )
    ball_states, debug_info = ball_tracker.build()

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

    render_start = time.time()
    frame_index = 0
    stabilization_text = (
        "Stab: on"
        if motion_probe.enabled
        else f"Stab: off ({motion_probe.mean_translation:.2f}px)"
    )

    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= len(observations):
            break
        transformed = apply_transform(frame, transforms[frame_index])
        annotated = transformed.copy()
        draw_players(annotated, player_states[frame_index].visible_tracks)
        draw_ball(annotated, ball_states, frame_index)
        elapsed = max(time.time() - render_start, 1e-6)
        current_fps = (frame_index + 1) / elapsed
        draw_overlay(
            annotated,
            player_states[frame_index].team_a_count,
            player_states[frame_index].team_b_count,
            current_fps,
            stabilization_text,
        )
        writer.write(annotated)
        if args.display:
            cv2.imshow("Volleyball Tracker OpenCV V5", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    capture.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    if args.debug_dir is not None:
        save_debug_artifacts(args.debug_dir, args.input, transforms, ball_states, debug_info)


if __name__ == "__main__":
    main()
