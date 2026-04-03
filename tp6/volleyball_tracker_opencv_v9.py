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
    ball_trail_length_v8: int = 44
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

    ball_mog_history: int = 500
    ball_mog_var_threshold: int = 16
    ball_median_diff_threshold: int = 18
    ball_median_open_kernel: tuple[int, int] = (3, 3)
    ball_median_close_kernel: tuple[int, int] = (5, 5)
    ball_mog_open_kernel: tuple[int, int] = (3, 3)
    ball_mog_close_kernel: tuple[int, int] = (5, 5)
    ball_mog_dilate_kernel: tuple[int, int] = (3, 3)
    ball_candidate_keep_per_frame: int = 16
    ball_top_radius: float = 3.1
    ball_bottom_radius: float = 7.0
    ball_area_scale_min: float = 0.25
    ball_area_scale_max: float = 6.0
    ball_max_aspect_ratio: float = 2.0
    ball_min_compactness: float = 0.25
    ball_min_circularity: float = 0.04
    ball_seed_support_frames: int = 1
    ball_seed_search_gap: int = 2
    ball_seed_min_progress: float = 7.0
    ball_seed_max_progress: float = 140.0
    ball_track_search_gap: int = 2
    ball_lost_after_misses: int = 5
    ball_max_gate_per_step: float = 92.0
    ball_top_banner_guard_y: int = 152
    ball_offscreen_top_y: int = 128
    ball_safe_reentry_left_x: int = 190
    ball_safe_reentry_right_x: int = 1090
    ball_reentry_min_progress: float = 10.0
    ball_reentry_downward_bias: float = 3.0
    ball_reentry_inward_bias: float = 3.0
    ball_offscreen_mask_pending: int = 2
    ball_offscreen_grace_frames: int = 2
    ball_segment_min_confirmed: int = 4
    ball_segment_merge_gap: int = 5
    ball_interp_short_gap: int = 3
    ball_interp_medium_gap: int = 7
    ball_interp_residual_limit: float = 18.0
    ball_poly_window: int = 7
    ball_poly_outlier_limit: float = 24.0
    ball_poly_mean_residual_limit: float = 18.0
    ball_ml_gap_trigger: int = 9
    ball_ml_sample_stride: int = 4
    ball_ml_roi_margin: int = 180
    ball_coast_frames: int = 2
    ball_degraded_gate_scale: float = 1.7
    ball_recover_gate_scale: float = 2.5
    ball_touch_innovation_threshold: float = 34.0
    ball_rf_probability_weight: float = 12.0
    ball_rf_min_probability: float = 0.28
    ball_recover_fullframe_gate: float = 180.0
    ball_large_jump_threshold: float = 130.0
    ball_rf_negative_ratio: int = 6
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
    ball_high_risk_rects: tuple[tuple[int, int, int, int], ...] = (
        (132, 118, 222, 238),
        (1058, 118, 1148, 238),
        (110, 0, 1170, 150),
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
    mode: str = ""
    trail_generation: int = 0
    offscreen: bool = False
    offscreen_grace: bool = False
    risk_strip: bool = False
    risk_zone: str = ""
    reentry_reject: bool = False
    central_reacquire: bool = False


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
    large_jump_frames: list[int] = field(default_factory=list)
    risk_strip_frames: list[int] = field(default_factory=list)
    offscreen_frames: list[int] = field(default_factory=list)
    reentry_reject_frames: list[int] = field(default_factory=list)
    central_reacquire_frames: list[int] = field(default_factory=list)
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
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.float32,
    )
    kalman.processNoiseCov = np.diag([0.2, 0.2, 0.9, 0.9]).astype(np.float32)
    kalman.measurementNoiseCov = np.diag([16.0, 16.0]).astype(np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 24.0
    vx = (next_center[0] - start_center[0]) / dt
    vy = (next_center[1] - start_center[1]) / dt
    kalman.statePost = np.array(
        [
            [np.float32(next_center[0])],
            [np.float32(next_center[1])],
            [np.float32(vx)],
            [np.float32(vy)],
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


def kalman_measurement_distance(kalman: cv2.KalmanFilter, center: tuple[int, int]) -> float:
    measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]], dtype=np.float32)
    predicted = kalman.measurementMatrix @ kalman.statePre
    covariance = (
        kalman.measurementMatrix @ kalman.errorCovPre @ kalman.measurementMatrix.T
    ) + kalman.measurementNoiseCov
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        return distance_between(center, kalman_predicted_center(kalman.statePre))
    delta = measurement - predicted
    score = float((delta.T @ inverse @ delta)[0][0])
    return math.sqrt(max(score, 0.0))


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


def load_ball_label_map(
    dataset_root: Path,
    frame_shape: tuple[int, int, int],
) -> dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]]:
    label_map: dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]] = {}
    height, width = frame_shape[:2]
    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        return label_map
    for label_file in sorted(labels_dir.glob("*/*.txt")):
        try:
            frame_index = int(label_file.stem.split("_")[-1])
        except ValueError:
            continue
        rows = [row.strip().split() for row in label_file.read_text().splitlines() if row.strip()]
        if not rows:
            continue
        _, cx, cy, bw, bh = rows[0][:5]
        cx_f = float(cx) * width
        cy_f = float(cy) * height
        bw_f = max(1.0, float(bw) * width)
        bh_f = max(1.0, float(bh) * height)
        bbox = clip_bbox(
            (
                int(round(cx_f - (bw_f / 2.0))),
                int(round(cy_f - (bh_f / 2.0))),
                int(round(bw_f)),
                int(round(bh_f)),
            ),
            frame_shape,
        )
        radius = max(1.0, min(bw_f, bh_f) / 2.0)
        label_map[frame_index] = (bbox_center(bbox), bbox, radius)
    return label_map


def build_ball_radius_profile(
    label_map: dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]],
    frame_shape: tuple[int, int, int],
    config: SceneConfigV5,
    bins: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    height = frame_shape[0]
    bin_centers = np.linspace(0.0, max(height - 1, 1), bins, dtype=np.float32)
    samples: list[list[float]] = [[] for _ in range(bins)]
    for center, _, radius in label_map.values():
        index = int(np.clip((center[1] / max(height - 1, 1)) * bins, 0, bins - 1))
        samples[index].append(radius)
    known_x: list[float] = []
    known_y: list[float] = []
    for index, values in enumerate(samples):
        if values:
            known_x.append(float(bin_centers[index]))
            known_y.append(float(statistics.median(values)))
    if len(known_x) < 2:
        fallback = np.array(
            [expected_ball_radius(int(value), height, config) for value in bin_centers],
            dtype=np.float32,
        )
        return bin_centers, fallback
    profile = np.interp(bin_centers, np.array(known_x), np.array(known_y)).astype(np.float32)
    return bin_centers, profile


class BallRecoveryModelV5:
    FEATURE_VERSION = 1

    def __init__(
        self,
        artifact_path: Path,
        dataset_root: Path,
        config: SceneConfigV5,
        disabled: bool,
    ) -> None:
        self.artifact_path = artifact_path
        self.dataset_root = dataset_root
        self.config = config
        self.disabled = disabled
        self._model = None
        self._load_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return not self.disabled and self.dataset_root.exists()

    def ensure_ready(
        self,
        candidates_by_frame: list[list[BallCandidate]],
        player_states: list[PlayerFrameState],
        frame_shape: tuple[int, int, int],
        label_map: dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]],
        expected_radius_for_y,
    ) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        if self.artifact_path.exists():
            try:
                import joblib

                payload = joblib.load(self.artifact_path)
                if payload.get("feature_version") == self.FEATURE_VERSION:
                    self._model = payload.get("model")
                    if self._model is not None:
                        return True
            except Exception as exc:  # pragma: no cover - cache corruption is env specific
                self._load_error = f"Could not load RandomForest cache: {exc}"
        return self._train_model(candidates_by_frame, player_states, frame_shape, label_map, expected_radius_for_y)

    def score_candidates(
        self,
        frame_index: int,
        candidates: list[BallCandidate],
        suppression_boxes: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
        expected_radius_for_y,
    ) -> dict[int, float]:
        if self._model is None or not candidates:
            return {}
        features = np.array(
            [
                self._feature_vector(candidate, suppression_boxes, frame_shape, expected_radius_for_y)
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        try:
            probabilities = self._model.predict_proba(features)[:, 1]
        except Exception:
            return {}
        return {candidate.candidate_id: float(prob) for candidate, prob in zip(candidates, probabilities)}

    def _feature_vector(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
        expected_radius_for_y,
    ) -> list[float]:
        height, width = frame_shape[:2]
        expected_radius = max(1.0, float(expected_radius_for_y(candidate.center[1])))
        expected_area = math.pi * expected_radius * expected_radius
        torso_overlap = 1.0 if self._candidate_inside_suppression(candidate, suppression_boxes) else 0.0
        return [
            candidate.center[0] / max(width, 1),
            candidate.center[1] / max(height, 1),
            candidate.area / max(expected_area, 1.0),
            candidate.radius / expected_radius,
            candidate.aspect_ratio,
            candidate.compactness,
            candidate.circularity,
            candidate.weak_yellow_ratio,
            float(candidate.source_median),
            float(candidate.source_mog),
            float(candidate.source_median and candidate.source_mog),
            float(candidate.support_count),
            candidate.support_progress / max(self.config.ball_seed_max_progress, 1.0),
            torso_overlap,
        ]

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

    def _train_model(
        self,
        candidates_by_frame: list[list[BallCandidate]],
        player_states: list[PlayerFrameState],
        frame_shape: tuple[int, int, int],
        label_map: dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]],
        expected_radius_for_y,
    ) -> bool:
        if not label_map:
            self._load_error = "Ball recovery dataset labels were not found."
            return False
        positive_rows: list[list[float]] = []
        negative_rows: list[list[float]] = []
        for frame_index, (gt_center, gt_bbox, gt_radius) in label_map.items():
            if frame_index >= len(candidates_by_frame):
                continue
            candidates = candidates_by_frame[frame_index]
            if not candidates:
                continue
            suppression_boxes = player_states[frame_index].suppression_boxes if frame_index < len(player_states) else []
            positive_candidate: Optional[BallCandidate] = None
            best_distance = max(16.0, gt_radius * 3.2)
            for candidate in candidates:
                distance = distance_between(candidate.center, gt_center)
                if distance <= best_distance or bbox_iou(candidate.bbox, gt_bbox) >= 0.03:
                    if positive_candidate is None or distance < best_distance:
                        positive_candidate = candidate
                        best_distance = distance
            if positive_candidate is None:
                continue
            for candidate in candidates:
                row = self._feature_vector(candidate, suppression_boxes, frame_shape, expected_radius_for_y)
                if candidate.candidate_id == positive_candidate.candidate_id:
                    positive_rows.append(row)
                else:
                    negative_rows.append(row)
        if len(positive_rows) < 20 or len(negative_rows) < 20:
            self._load_error = "Not enough candidate rows to train RandomForest recovery."
            return False
        rng = np.random.default_rng(42)
        max_negatives = min(len(negative_rows), len(positive_rows) * self.config.ball_rf_negative_ratio)
        selected_negative_indices = rng.choice(len(negative_rows), size=max_negatives, replace=False)
        features = np.array(
            positive_rows + [negative_rows[int(index)] for index in selected_negative_indices],
            dtype=np.float32,
        )
        labels = np.array([1] * len(positive_rows) + [0] * max_negatives, dtype=np.int32)
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
        except Exception as exc:  # pragma: no cover - dependency presence depends on env
            self._load_error = f"Could not import RandomForest dependencies: {exc}"
            return False
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
        )
        model.fit(features, labels)
        self._model = model
        try:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"feature_version": self.FEATURE_VERSION, "model": model}, self.artifact_path)
        except Exception:
            pass
        return True


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


class BallTrajectoryTrackerV6(BallTrajectoryTrackerV5):
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
        super().__init__(config, frame_shape, fps, observations, player_states, input_path, transforms, recovery_model)
        self.label_map = load_ball_label_map(recovery_model.dataset_root, frame_shape)
        self.radius_profile_bins, self.radius_profile_values = build_ball_radius_profile(
            self.label_map,
            frame_shape,
            config,
        )

    def build(self) -> tuple[list[BallFrameState], BallDebugInfo]:
        self._prepare_candidates()
        self.recovery_model.ensure_ready(
            self.candidates_by_frame,
            self.player_states,
            self.frame_shape,
            self.label_map,
            self._expected_radius,
        )
        states = self._track_frames()
        self._interpolate_states(states)
        self._prune_jump_outliers(states)
        self._finalize_debug_v6(states)
        return states, self.debug_info

    def _expected_radius(self, y: int) -> float:
        if self.radius_profile_values.size == 0:
            return expected_ball_radius(y, self.frame_shape[0], self.config)
        return float(
            np.interp(
                float(np.clip(y, 0, self.frame_shape[0] - 1)),
                self.radius_profile_bins,
                self.radius_profile_values,
            )
        )

    def _prepare_candidates(self) -> None:
        filtered: list[list[BallCandidate]] = []
        for frame_index, observation in enumerate(self.observations):
            suppression_boxes = self.player_states[frame_index].suppression_boxes if frame_index < len(self.player_states) else []
            frame_candidates: list[BallCandidate] = []
            for candidate in observation.ball_candidates:
                if not self._passes_base_sieve(candidate, suppression_boxes):
                    continue
                candidate.local_quality = self._local_quality_v6(candidate)
                frame_candidates.append(candidate)
            frame_candidates.sort(key=lambda item: item.local_quality, reverse=True)
            filtered.append(frame_candidates[: self.config.ball_candidate_keep_per_frame])
        self.candidates_by_frame = filtered

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            for candidate in frame_candidates:
                support_count, support_progress = self._temporal_support_v6(frame_index, candidate)
                candidate.support_count = support_count
                candidate.support_progress = support_progress

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            pruned: list[BallCandidate] = []
            for candidate in frame_candidates:
                if (
                    candidate.center[1] < self.config.ball_top_banner_guard_y
                    and candidate.support_count <= 0
                    and not candidate.source_median
                ):
                    continue
                pruned.append(candidate)
            self.candidates_by_frame[frame_index] = pruned[: self.config.ball_candidate_keep_per_frame]

    def _passes_base_sieve(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        expected_radius = self._expected_radius(candidate.center[1])
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
        if self._candidate_inside_suppression(candidate, suppression_boxes) and candidate.radius > expected_radius * 0.85:
            return False
        return True

    def _local_quality_v6(self, candidate: BallCandidate) -> float:
        expected_radius = self._expected_radius(candidate.center[1])
        score = 0.0
        score += candidate.circularity * 4.2
        score += candidate.compactness * 3.0
        score += max(0.0, 2.4 - abs(candidate.radius - expected_radius))
        score += candidate.weak_yellow_ratio * 1.2
        if candidate.source_median:
            score += 1.6
        if candidate.source_mog:
            score += 1.0
        if candidate.source_median and candidate.source_mog:
            score += 1.8
        return score

    def _temporal_support_v6(self, frame_index: int, candidate: BallCandidate) -> tuple[int, float]:
        supports = 0
        max_progress = 0.0
        for direction in (-1, 1):
            for delta in range(1, self.config.ball_seed_search_gap + 1):
                neighbor_index = frame_index + (direction * delta)
                if neighbor_index < 0 or neighbor_index >= len(self.candidates_by_frame):
                    continue
                best_distance: Optional[float] = None
                for neighbor in self.candidates_by_frame[neighbor_index]:
                    distance = distance_between(candidate.center, neighbor.center)
                    if distance < self.config.ball_seed_min_progress * delta:
                        continue
                    if distance > self.config.ball_seed_max_progress * delta:
                        continue
                    if abs(neighbor.radius - candidate.radius) > max(4.0, self._expected_radius(candidate.center[1]) * 1.1):
                        continue
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                if best_distance is not None:
                    supports += 1
                    max_progress = max(max_progress, best_distance)
                    break
        return supports, max_progress

    def _track_frames(self) -> list[BallFrameState]:
        states = [BallFrameState() for _ in range(len(self.observations))]
        confirmed_points: list[TrajectoryPoint] = []
        kalman: Optional[cv2.KalmanFilter] = None
        mode = "SEARCH_INIT"
        misses = 0
        last_radius = self.config.ball_top_radius
        touch_boost_frames = 0

        for frame_index in range(self.config.bg_warmup_frames, len(self.observations)):
            predicted_center: Optional[tuple[int, int]] = None
            if kalman is not None:
                self._set_process_noise(kalman, touch_boost_frames > 0)
                predicted_center = kalman_predicted_center(kalman.predict())
                if touch_boost_frames > 0:
                    touch_boost_frames -= 1

            if mode in {"SEARCH_INIT", "LOST"}:
                anchor = self._find_anchor_v6(frame_index, predicted_center, recover=False)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    states[frame_index] = BallFrameState(
                        center=point.center,
                        radius=point.radius,
                        status="confirmed",
                        confidence=point.confidence,
                        source=source,
                    )
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                continue

            if mode == "TRACK":
                success, touch_event, candidate = self._apply_tracking_measurement_v6(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=1.0,
                    relaxed=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses = 1
                self._write_coast_state_v6(states, frame_index, predicted_center, last_radius)
                mode = "DEGRADED"
                continue

            if mode == "DEGRADED":
                success, touch_event, candidate = self._apply_tracking_measurement_v6(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=self.config.ball_degraded_gate_scale,
                    relaxed=True,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses += 1
                if misses <= self.config.ball_coast_frames:
                    self._write_coast_state_v6(states, frame_index, predicted_center, last_radius)
                else:
                    mode = "RECOVER"
                continue

            if mode == "RECOVER":
                anchor = self._find_anchor_v6(frame_index, predicted_center, recover=True)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    states[frame_index] = BallFrameState(
                        center=point.center,
                        radius=point.radius,
                        status="confirmed",
                        confidence=point.confidence,
                        source=source,
                    )
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    self.debug_info.recovery_events.append(
                        RecoveryEvent(
                            frame_index=frame_index,
                            confidence=point.confidence,
                            source=source,
                            roi=self._recovery_roi_v6(predicted_center),
                        )
                    )
                    continue
                misses += 1
                if misses <= self.config.ball_coast_frames:
                    self._write_coast_state_v6(states, frame_index, predicted_center, last_radius)
                elif misses > self.config.ball_lost_after_misses:
                    kalman = None
                    mode = "LOST"

        return states

    def _set_process_noise(self, kalman: cv2.KalmanFilter, boosted: bool) -> None:
        if boosted:
            kalman.processNoiseCov = np.diag([0.8, 0.8, 3.2, 3.2]).astype(np.float32)
        else:
            kalman.processNoiseCov = np.diag([0.2, 0.2, 0.9, 0.9]).astype(np.float32)

    def _apply_tracking_measurement_v6(
        self,
        frame_index: int,
        states: list[BallFrameState],
        kalman: Optional[cv2.KalmanFilter],
        predicted_center: Optional[tuple[int, int]],
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
    ) -> tuple[bool, bool, Optional[BallCandidate]]:
        if kalman is None or predicted_center is None:
            return False, False, None
        selected = self._select_candidate_v6(
            frame_index,
            predicted_center,
            kalman,
            confirmed_points,
            gate_scale,
            relaxed,
            recovery_scores=None,
        )
        if selected is None:
            return False, False, None
        candidate, score, innovation = selected
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        kalman.correct(measurement)
        point = super()._point_from_candidate(frame_index, candidate, "confirmed")
        point.confidence += score * 0.08
        states[frame_index] = BallFrameState(
            center=point.center,
            radius=point.radius,
            status="confirmed",
            confidence=point.confidence,
            source=point.source,
        )
        confirmed_points.append(point)
        return True, innovation >= self.config.ball_touch_innovation_threshold, candidate

    def _select_candidate_v6(
        self,
        frame_index: int,
        predicted_center: tuple[int, int],
        kalman: cv2.KalmanFilter,
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
        recovery_scores: Optional[dict[int, float]],
    ) -> Optional[tuple[BallCandidate, float, float]]:
        best_choice: Optional[tuple[BallCandidate, float, float]] = None
        gate = self.config.ball_max_gate_per_step * gate_scale
        suppression_boxes = self.player_states[frame_index].suppression_boxes
        for candidate in self.candidates_by_frame[frame_index]:
            if not self._candidate_mode_allowed_v6(candidate, suppression_boxes, relaxed):
                continue
            if distance_between(predicted_center, candidate.center) > gate:
                continue
            score, innovation = self._candidate_score_v6(
                frame_index,
                candidate,
                predicted_center,
                kalman,
                confirmed_points,
                recovery_scores.get(candidate.candidate_id, 0.0) if recovery_scores else 0.0,
            )
            if score is None:
                continue
            if best_choice is None or score > best_choice[1]:
                best_choice = (candidate, score, innovation)
        min_score = 6.0 if relaxed else 8.0
        if best_choice is None or best_choice[1] < min_score:
            return None
        return best_choice

    def _candidate_mode_allowed_v6(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        relaxed: bool,
    ) -> bool:
        if self._candidate_inside_suppression(candidate, suppression_boxes):
            expected_radius = self._expected_radius(candidate.center[1])
            if candidate.radius > expected_radius * 0.72:
                return False
        aspect_limit = self.config.ball_max_aspect_ratio * (1.18 if relaxed else 1.0)
        compactness_floor = self.config.ball_min_compactness * (0.7 if relaxed else 1.0)
        circularity_floor = self.config.ball_min_circularity * (0.45 if relaxed else 1.0)
        ratio = candidate.aspect_ratio
        if ratio > aspect_limit or (1.0 / max(ratio, 1e-6)) > aspect_limit:
            return False
        if candidate.compactness < compactness_floor:
            return False
        if candidate.circularity < circularity_floor:
            return False
        return True

    def _candidate_score_v6(
        self,
        frame_index: int,
        candidate: BallCandidate,
        predicted_center: tuple[int, int],
        kalman: cv2.KalmanFilter,
        confirmed_points: list[TrajectoryPoint],
        recovery_probability: float,
    ) -> tuple[Optional[float], float]:
        innovation = distance_between(predicted_center, candidate.center)
        mahalanobis = kalman_measurement_distance(kalman, candidate.center)
        candidate_point = super()._point_from_candidate(frame_index, candidate, "candidate")
        trajectory_penalty = trajectory_fit_residual(confirmed_points[-self.config.ball_poly_window :], candidate_point)
        if trajectory_penalty > self.config.ball_poly_outlier_limit and mahalanobis < self.config.ball_touch_innovation_threshold:
            return None, innovation
        velocity_penalty = 0.0
        if len(confirmed_points) >= 2:
            last = confirmed_points[-1]
            previous = confirmed_points[-2]
            delta_frames = max(1, last.frame_index - previous.frame_index)
            expected_vx = (last.center[0] - previous.center[0]) / delta_frames
            expected_vy = (last.center[1] - previous.center[1]) / delta_frames
            candidate_vx = candidate.center[0] - last.center[0]
            candidate_vy = candidate.center[1] - last.center[1]
            velocity_penalty = abs(candidate_vx - expected_vx) + abs(candidate_vy - expected_vy)
        expected_radius = self._expected_radius(candidate.center[1])
        size_penalty = abs(candidate.radius - expected_radius) * 1.8
        torso_penalty = 8.0 if self._candidate_inside_suppression(candidate, self.player_states[frame_index].suppression_boxes) else 0.0
        source_bonus = 0.0
        if candidate.source_median:
            source_bonus += 2.2
        if candidate.source_mog:
            source_bonus += 1.1
        if candidate.source_median and candidate.source_mog:
            source_bonus += 1.8
        score = (
            candidate.local_quality * 5.4
            + (candidate.support_count * 3.2)
            + source_bonus
            + (recovery_probability * self.config.ball_rf_probability_weight)
            - (mahalanobis * 2.4)
            - (velocity_penalty * 0.16)
            - size_penalty
            - (trajectory_penalty * 0.9)
            - torso_penalty
        )
        return score, innovation

    def _find_anchor_v6(
        self,
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        recover: bool,
    ) -> Optional[tuple[BallCandidate, int, BallCandidate, str]]:
        frame_candidates = self.candidates_by_frame[frame_index]
        if not frame_candidates:
            return None
        suppression_boxes = self.player_states[frame_index].suppression_boxes
        recovery_scores = (
            self.recovery_model.score_candidates(
                frame_index,
                frame_candidates,
                suppression_boxes,
                self.frame_shape,
                self._expected_radius,
            )
            if recover and self.recovery_model.enabled
            else {}
        )
        best_choice: Optional[tuple[BallCandidate, int, BallCandidate, str]] = None
        best_score = -1e9
        for candidate in frame_candidates:
            if predicted_center is not None and distance_between(predicted_center, candidate.center) > self.config.ball_recover_fullframe_gate:
                continue
            partner = self._find_confirmation_partner_v6(frame_index, candidate)
            if partner is None:
                continue
            partner_frame, partner_candidate = partner
            rf_probability = recovery_scores.get(candidate.candidate_id, 0.0)
            if recover and recovery_scores and rf_probability < self.config.ball_rf_min_probability and candidate.support_count <= 0:
                continue
            torso_penalty = 18.0 if self._candidate_inside_suppression(candidate, suppression_boxes) else 0.0
            distance_penalty = 0.0 if predicted_center is None else distance_between(predicted_center, candidate.center) * 0.25
            score = (
                candidate.local_quality * 6.2
                + partner_candidate.local_quality * 2.6
                + (candidate.support_count * 4.5)
                + (partner_candidate.support_count * 2.0)
                + (rf_probability * self.config.ball_rf_probability_weight)
                - torso_penalty
                - distance_penalty
            )
            if score > best_score:
                best_score = score
                source = "recover_rf" if recover and rf_probability >= self.config.ball_rf_min_probability else "motion_seed"
                best_choice = (candidate, partner_frame, partner_candidate, source)
        if best_choice is None or best_score < 18.0:
            return None
        return best_choice

    def _find_confirmation_partner_v6(
        self,
        frame_index: int,
        seed: BallCandidate,
    ) -> Optional[tuple[int, BallCandidate]]:
        best_choice: Optional[tuple[int, BallCandidate]] = None
        best_score = -1e9
        for delta in range(1, self.config.ball_seed_search_gap + 1):
            future_index = frame_index + delta
            if future_index >= len(self.candidates_by_frame):
                break
            for candidate in self.candidates_by_frame[future_index]:
                distance = distance_between(seed.center, candidate.center)
                if distance < self.config.ball_seed_min_progress * delta:
                    continue
                if distance > self.config.ball_seed_max_progress * delta:
                    continue
                score = (candidate.local_quality * 5.5) + (candidate.support_count * 3.0) - (distance * 0.08)
                if score > best_score:
                    best_score = score
                    best_choice = (future_index, candidate)
        return best_choice

    def _write_coast_state_v6(
        self,
        states: list[BallFrameState],
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        radius: float,
    ) -> None:
        if predicted_center is None:
            return
        if not point_in_mask(predicted_center, self.ball_mask):
            return
        if self._center_hits_torso(frame_index, predicted_center):
            return
        states[frame_index] = BallFrameState(
            center=predicted_center,
            radius=radius,
            status="coast",
            confidence=0.18,
            source="kalman",
        )

    def _recovery_roi_v6(self, predicted_center: Optional[tuple[int, int]]) -> tuple[int, int, int, int]:
        if predicted_center is None:
            return (0, 0, self.frame_shape[1], self.frame_shape[0])
        half = self.config.ball_ml_roi_margin
        return clip_bbox(
            (predicted_center[0] - half, predicted_center[1] - half, half * 2, half * 2),
            self.frame_shape,
        )

    def _interpolate_states(self, states: list[BallFrameState]) -> None:
        confirmed_frames = [index for index, state in enumerate(states) if state.status == "confirmed" and state.center is not None]
        if len(confirmed_frames) < 2:
            return
        for previous_index, current_index in zip(confirmed_frames, confirmed_frames[1:]):
            previous_state = states[previous_index]
            current_state = states[current_index]
            if previous_state.center is None or current_state.center is None or previous_state.radius is None or current_state.radius is None:
                continue
            gap = current_index - previous_index - 1
            if gap <= 0:
                continue
            endpoint_distance = distance_between(previous_state.center, current_state.center)
            if endpoint_distance > self.config.ball_large_jump_threshold * max(1, gap):
                continue
            if gap <= self.config.ball_interp_short_gap:
                for step in range(1, gap + 1):
                    ratio = step / float(gap + 1)
                    frame_index = previous_index + step
                    center = (
                        int(round((previous_state.center[0] * (1.0 - ratio)) + (current_state.center[0] * ratio))),
                        int(round((previous_state.center[1] * (1.0 - ratio)) + (current_state.center[1] * ratio))),
                    )
                    if self._center_hits_torso(frame_index, center):
                        continue
                    states[frame_index] = BallFrameState(
                        center=center,
                        radius=(previous_state.radius * (1.0 - ratio)) + (current_state.radius * ratio),
                        status="interpolated",
                        confidence=min(previous_state.confidence, current_state.confidence) * 0.7,
                        source="interp_short",
                    )
            elif gap <= self.config.ball_interp_medium_gap:
                window_indices = [
                    index
                    for index in range(max(0, previous_index - 2), min(len(states), current_index + 3))
                    if states[index].status == "confirmed" and states[index].center is not None
                ]
                if len(window_indices) < 4:
                    continue
                window_points = [
                    TrajectoryPoint(
                        frame_index=index,
                        center=states[index].center,
                        radius=states[index].radius or (previous_state.radius + current_state.radius) / 2.0,
                        bbox=(0, 0, 1, 1),
                        confidence=states[index].confidence,
                        status=states[index].status,
                        source=states[index].source,
                    )
                    for index in window_indices
                ]
                residuals = super()._point_residuals(window_points)
                if float(np.mean(residuals)) > self.config.ball_interp_residual_limit:
                    continue
                frames = np.array(window_indices, dtype=np.float32)
                xs = np.array([states[index].center[0] for index in window_indices], dtype=np.float32)
                ys = np.array([states[index].center[1] for index in window_indices], dtype=np.float32)
                try:
                    x_coeff = np.polyfit(frames, xs, 1)
                    y_coeff = np.polyfit(frames, ys, 2)
                except np.linalg.LinAlgError:
                    continue
                for frame_index in range(previous_index + 1, current_index):
                    center = (
                        int(round(float(np.polyval(x_coeff, frame_index)))),
                        int(round(float(np.polyval(y_coeff, frame_index)))),
                    )
                    if self._center_hits_torso(frame_index, center):
                        continue
                    states[frame_index] = BallFrameState(
                        center=center,
                        radius=(previous_state.radius + current_state.radius) / 2.0,
                        status="interpolated",
                        confidence=min(previous_state.confidence, current_state.confidence) * 0.55,
                        source="interp_poly",
                    )

    def _prune_jump_outliers(self, states: list[BallFrameState]) -> None:
        valid_frames = [
            index
            for index, state in enumerate(states)
            if state.center is not None and state.status in {"confirmed", "interpolated"}
        ]
        to_clear: set[int] = set()
        threshold = self.config.ball_large_jump_threshold
        for position, frame_index in enumerate(valid_frames):
            state = states[frame_index]
            if state.source == "recover_rf":
                continue
            previous_index = valid_frames[position - 1] if position > 0 else None
            next_index = valid_frames[position + 1] if position + 1 < len(valid_frames) else None
            previous_far = (
                previous_index is not None
                and distance_between(states[previous_index].center, state.center) > threshold
            )
            next_far = (
                next_index is not None
                and distance_between(state.center, states[next_index].center) > threshold
            )
            if previous_far and next_far and previous_index is not None and next_index is not None:
                bridge_distance = distance_between(states[previous_index].center, states[next_index].center)
                if bridge_distance <= threshold * 1.25:
                    to_clear.add(frame_index)
                    continue
            if previous_far and (next_index is None or next_far):
                to_clear.add(frame_index)
        for frame_index in to_clear:
            states[frame_index] = BallFrameState()

    def _finalize_debug_v6(self, states: list[BallFrameState]) -> None:
        gap_start: Optional[int] = None
        flagged: set[int] = set()
        previous_rendered: Optional[tuple[int, int]] = None
        for index, state in enumerate(states):
            if state.center is None:
                if gap_start is None:
                    gap_start = index
            else:
                if gap_start is not None and index - gap_start >= self.config.ball_ml_gap_trigger:
                    self.debug_info.long_gaps.append((gap_start, index - 1))
                    flagged.update({gap_start, gap_start + ((index - gap_start) // 2), index - 1})
                gap_start = None
                if state.status in {"confirmed", "interpolated"}:
                    if previous_rendered is not None and distance_between(previous_rendered, state.center) > self.config.ball_large_jump_threshold:
                        self.debug_info.large_jump_frames.append(index)
                        flagged.add(index)
                    previous_rendered = state.center
            if state.center is not None and self._center_hits_torso(index, state.center):
                self.debug_info.overlap_frames.append(index)
                flagged.add(index)
        if gap_start is not None and len(states) - gap_start >= self.config.ball_ml_gap_trigger:
            self.debug_info.long_gaps.append((gap_start, len(states) - 1))
            flagged.update({gap_start, gap_start + ((len(states) - gap_start) // 2), len(states) - 1})
        for event in self.debug_info.recovery_events:
            flagged.add(event.frame_index)
        flagged.update(self.debug_info.large_jump_frames)
        self.debug_info.flagged_frames = sorted(flagged)


class BallTrajectoryTrackerV8(BallTrajectoryTrackerV6):
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
        super().__init__(config, frame_shape, fps, observations, player_states, input_path, transforms, recovery_model)
        self.soft_edge_rects = {
            "left_edge": (0, 0, config.ball_safe_reentry_left_x, 260),
            "right_edge": (config.ball_safe_reentry_right_x, 0, frame_shape[1], 260),
        }
        self.hard_risk_rects = {
            "left_head": config.ball_high_risk_rects[0],
            "right_head": config.ball_high_risk_rects[1],
            "top_banner": config.ball_high_risk_rects[2],
        }
        self.safe_reentry_mask = self._build_safe_reentry_mask()
        self._risk_reject_frames: set[int] = set()
        self._offscreen_frames: set[int] = set()
        self._offscreen_grace_frames: set[int] = set()

    def build(self) -> tuple[list[BallFrameState], BallDebugInfo]:
        self._prepare_candidates()
        self.recovery_model.ensure_ready(
            self.candidates_by_frame,
            self.player_states,
            self.frame_shape,
            self.label_map,
            self._expected_radius,
        )
        states = self._track_frames()
        self._interpolate_states(states)
        self._prune_jump_outliers(states)
        self._finalize_debug_v8(states)
        return states, self.debug_info

    def _build_safe_reentry_mask(self) -> np.ndarray:
        mask = self.ball_mask.copy()
        return apply_exclusions(mask, self.config.ball_high_risk_rects)

    def _point_in_rect(self, point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _hard_risk_zone_name(self, point: tuple[int, int]) -> str:
        for name, rect in self.hard_risk_rects.items():
            if self._point_in_rect(point, rect):
                return name
        return ""

    def _soft_edge_zone_name(self, point: tuple[int, int]) -> str:
        for name, rect in self.soft_edge_rects.items():
            if self._point_in_rect(point, rect):
                return name
        return ""

    def _risk_zone_name(self, point: tuple[int, int]) -> str:
        hard = self._hard_risk_zone_name(point)
        if hard:
            return hard
        return self._soft_edge_zone_name(point)

    def _center_in_high_risk_rect(self, point: tuple[int, int]) -> bool:
        return bool(self._hard_risk_zone_name(point))

    def _center_safe_for_reentry(self, point: tuple[int, int]) -> bool:
        return point_in_mask(point, self.safe_reentry_mask)

    def _center_safe_for_coast(self, point: tuple[int, int]) -> bool:
        return point_in_mask(point, self.ball_mask) and not self._center_in_high_risk_rect(point)

    def _record_reentry_reject(self, frame_index: int) -> None:
        if frame_index >= 0:
            self._risk_reject_frames.add(frame_index)

    def _recovery_suppression_boxes(self, frame_index: int) -> list[tuple[int, int, int, int]]:
        boxes = list(self.player_states[frame_index].suppression_boxes)
        top_boxes = [box for box in boxes if box[1] < 360]
        if len(top_boxes) >= 2:
            x1 = min(box[0] for box in top_boxes)
            y1 = min(box[1] for box in top_boxes)
            x2 = max(box[0] + box[2] for box in top_boxes)
            y2 = max(box[1] + box[3] for box in top_boxes)
            boxes.append(clip_bbox((x1, y1, x2 - x1, y2 - y1), self.frame_shape))
        return boxes

    def _prepare_candidates(self) -> None:
        super()._prepare_candidates()
        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            pruned: list[BallCandidate] = []
            for candidate in frame_candidates:
                zone = self._hard_risk_zone_name(candidate.center)
                if zone in {"left_head", "right_head"} and candidate.support_count <= 0:
                    continue
                if zone == "top_banner" and (candidate.support_count <= 0 or not candidate.source_median):
                    continue
                pruned.append(candidate)
            self.candidates_by_frame[frame_index] = pruned[: self.config.ball_candidate_keep_per_frame]

    def _last_confirmed_velocity(self, confirmed_points: list[TrajectoryPoint]) -> tuple[float, float]:
        if len(confirmed_points) < 2:
            return (0.0, 0.0)
        last = confirmed_points[-1]
        previous = confirmed_points[-2]
        delta = max(1, last.frame_index - previous.frame_index)
        return (
            (last.center[0] - previous.center[0]) / float(delta),
            (last.center[1] - previous.center[1]) / float(delta),
        )

    def _offscreen_trigger_kind(
        self,
        predicted_center: Optional[tuple[int, int]],
        confirmed_points: list[TrajectoryPoint],
    ) -> str:
        if predicted_center is None or len(confirmed_points) < 2:
            return ""
        _, vy = self._last_confirmed_velocity(confirmed_points)
        if vy >= -2.0:
            return ""
        if self._center_in_high_risk_rect(predicted_center):
            return "hard"
        if predicted_center[1] <= self.config.ball_offscreen_top_y:
            return "top"
        if not point_in_mask(predicted_center, self.ball_mask):
            return "mask"
        return ""

    def _mark_frame_mode(
        self,
        states: list[BallFrameState],
        frame_index: int,
        mode: str,
        trail_generation: int,
    ) -> None:
        state = states[frame_index]
        if state.center is None:
            state.mode = mode
            state.trail_generation = trail_generation
            state.offscreen = mode == "OFFSCREEN"
            state.offscreen_grace = mode == "OFFSCREEN_GRACE"

    def _assign_state(
        self,
        states: list[BallFrameState],
        frame_index: int,
        center: tuple[int, int],
        radius: float,
        status: str,
        confidence: float,
        source: str,
        mode: str,
        trail_generation: int,
    ) -> None:
        risk_zone = self._risk_zone_name(center)
        states[frame_index] = BallFrameState(
            center=center,
            radius=radius,
            status=status,
            confidence=confidence,
            source=source,
            mode=mode,
            trail_generation=trail_generation,
            offscreen=mode == "OFFSCREEN",
            offscreen_grace=mode == "OFFSCREEN_GRACE",
            risk_strip=bool(risk_zone),
            risk_zone=risk_zone,
        )

    def _soft_edge_allowed(self, candidate: BallCandidate, from_offscreen: bool) -> bool:
        if not from_offscreen:
            return True
        if not self._soft_edge_zone_name(candidate.center):
            return True
        return candidate.source_median and candidate.support_count >= 1

    def _candidate_mode_allowed_v8(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        relaxed: bool,
        mode: str,
        predicted_center: Optional[tuple[int, int]],
        from_offscreen: bool,
    ) -> bool:
        if not self._candidate_mode_allowed_v6(candidate, suppression_boxes, relaxed):
            return False
        hard_zone = self._hard_risk_zone_name(candidate.center)
        if hard_zone:
            return False
        if mode in {"SEARCH_INIT", "RECOVER", "OFFSCREEN"} and not self._center_safe_for_reentry(candidate.center):
            return False
        if not self._soft_edge_allowed(candidate, from_offscreen):
            return False
        if (
            candidate.center[1] < self.config.ball_top_banner_guard_y
            and candidate.support_count <= 0
            and not candidate.source_median
            and self._soft_edge_zone_name(candidate.center)
        ):
            return False
        return True

    def _select_candidate_v8(
        self,
        frame_index: int,
        predicted_center: tuple[int, int],
        kalman: cv2.KalmanFilter,
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
        mode: str,
        from_offscreen: bool,
    ) -> Optional[tuple[BallCandidate, float, float]]:
        best_choice: Optional[tuple[BallCandidate, float, float]] = None
        gate = self.config.ball_max_gate_per_step * gate_scale
        suppression_boxes = self.player_states[frame_index].suppression_boxes
        for candidate in self.candidates_by_frame[frame_index]:
            if not self._candidate_mode_allowed_v8(candidate, suppression_boxes, relaxed, mode, predicted_center, from_offscreen):
                continue
            if distance_between(predicted_center, candidate.center) > gate:
                continue
            score, innovation = self._candidate_score_v6(
                frame_index,
                candidate,
                predicted_center,
                kalman,
                confirmed_points,
                0.0,
            )
            if score is None:
                continue
            zone = self._soft_edge_zone_name(candidate.center)
            if zone and from_offscreen:
                score -= 2.5
            elif zone:
                score -= 0.6
            if (
                candidate.center[1] < self.config.ball_top_banner_guard_y
                and candidate.support_count <= 0
                and not candidate.source_median
                and self._soft_edge_zone_name(candidate.center)
            ):
                continue
            if best_choice is None or score > best_choice[1]:
                best_choice = (candidate, score, innovation)
        min_score = 2.8 if relaxed else 5.6
        if best_choice is None or best_choice[1] < min_score:
            return None
        return best_choice

    def _apply_tracking_measurement_v8(
        self,
        frame_index: int,
        states: list[BallFrameState],
        kalman: Optional[cv2.KalmanFilter],
        predicted_center: Optional[tuple[int, int]],
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
        mode: str,
        trail_generation: int,
        from_offscreen: bool,
    ) -> tuple[bool, bool, Optional[BallCandidate]]:
        if kalman is None or predicted_center is None:
            return False, False, None
        selected = self._select_candidate_v8(
            frame_index,
            predicted_center,
            kalman,
            confirmed_points,
            gate_scale,
            relaxed,
            mode,
            from_offscreen,
        )
        if selected is None:
            return False, False, None
        candidate, score, innovation = selected
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        kalman.correct(measurement)
        point = super()._point_from_candidate(frame_index, candidate, "confirmed")
        point.confidence += score * 0.08
        self._assign_state(
            states,
            frame_index,
            point.center,
            point.radius,
            "confirmed",
            point.confidence,
            point.source,
            mode,
            trail_generation,
        )
        confirmed_points.append(point)
        return True, innovation >= self.config.ball_touch_innovation_threshold, candidate

    def _anchor_direction_ok_v8(
        self,
        seed: BallCandidate,
        partner: BallCandidate,
        from_offscreen: bool,
    ) -> bool:
        distance = distance_between(seed.center, partner.center)
        if distance < self.config.ball_reentry_min_progress:
            return False
        if not from_offscreen:
            return True
        dy = partner.center[1] - seed.center[1]
        dx = partner.center[0] - seed.center[0]
        seed_soft = self._soft_edge_zone_name(seed.center)
        partner_soft = self._soft_edge_zone_name(partner.center)
        if not seed_soft and not partner_soft:
            return True
        descending = dy >= self.config.ball_reentry_downward_bias
        inward = False
        if seed_soft == "left_edge" or partner_soft == "left_edge":
            inward = inward or (dx >= self.config.ball_reentry_inward_bias)
        if seed_soft == "right_edge" or partner_soft == "right_edge":
            inward = inward or (dx <= -self.config.ball_reentry_inward_bias)
        return descending or inward

    def _find_confirmation_partner_v8(
        self,
        frame_index: int,
        seed: BallCandidate,
        from_offscreen: bool,
    ) -> Optional[tuple[int, BallCandidate]]:
        best_choice: Optional[tuple[int, BallCandidate]] = None
        best_score = -1e9
        seed_zone = self._hard_risk_zone_name(seed.center)
        for delta in range(1, self.config.ball_seed_search_gap + 1):
            future_index = frame_index + delta
            if future_index >= len(self.candidates_by_frame):
                break
            for candidate in self.candidates_by_frame[future_index]:
                distance = distance_between(seed.center, candidate.center)
                if distance < self.config.ball_seed_min_progress * delta:
                    continue
                if distance > self.config.ball_seed_max_progress * delta:
                    continue
                candidate_zone = self._hard_risk_zone_name(candidate.center)
                if seed_zone and seed_zone == candidate_zone:
                    continue
                if from_offscreen and (not self._center_safe_for_reentry(candidate.center) or not self._soft_edge_allowed(candidate, True)):
                    continue
                if not self._anchor_direction_ok_v8(seed, candidate, from_offscreen):
                    continue
                score = (candidate.local_quality * 5.5) + (candidate.support_count * 3.0) - (distance * 0.08)
                if score > best_score:
                    best_score = score
                    best_choice = (future_index, candidate)
        return best_choice

    def _find_anchor_v8(
        self,
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        recover: bool,
        from_offscreen: bool,
    ) -> Optional[tuple[BallCandidate, int, BallCandidate, str]]:
        frame_candidates = self.candidates_by_frame[frame_index]
        if not frame_candidates:
            return None
        suppression_boxes = (
            self._recovery_suppression_boxes(frame_index)
            if recover or from_offscreen
            else self.player_states[frame_index].suppression_boxes
        )
        filtered_candidates = []
        for candidate in frame_candidates:
            if not self._candidate_mode_allowed_v8(
                candidate,
                suppression_boxes,
                True,
                "OFFSCREEN" if from_offscreen else ("RECOVER" if recover else "SEARCH_INIT"),
                predicted_center,
                from_offscreen,
            ):
                if recover or from_offscreen:
                    self._record_reentry_reject(frame_index)
                continue
            filtered_candidates.append(candidate)
        if not filtered_candidates:
            return None
        recovery_scores = (
            self.recovery_model.score_candidates(
                frame_index,
                filtered_candidates,
                suppression_boxes,
                self.frame_shape,
                self._expected_radius,
            )
            if recover and self.recovery_model.enabled
            else {}
        )
        best_choice: Optional[tuple[BallCandidate, int, BallCandidate, str]] = None
        best_score = -1e9
        for candidate in filtered_candidates:
            if predicted_center is not None and recover and distance_between(predicted_center, candidate.center) > self.config.ball_recover_fullframe_gate:
                continue
            partner = self._find_confirmation_partner_v8(frame_index, candidate, from_offscreen)
            if partner is None:
                if recover or from_offscreen:
                    self._record_reentry_reject(frame_index)
                continue
            partner_frame, partner_candidate = partner
            rf_probability = recovery_scores.get(candidate.candidate_id, 0.0)
            if recover and recovery_scores and rf_probability < self.config.ball_rf_min_probability and candidate.support_count <= 0:
                self._record_reentry_reject(frame_index)
                continue
            torso_penalty = 18.0 if self._candidate_inside_suppression(candidate, suppression_boxes) else 0.0
            distance_penalty = 0.0 if predicted_center is None else distance_between(predicted_center, candidate.center) * 0.14
            soft_penalty = 0.0
            if from_offscreen and (self._soft_edge_zone_name(candidate.center) or self._soft_edge_zone_name(partner_candidate.center)):
                soft_penalty += 3.0
            score = (
                candidate.local_quality * 6.2
                + partner_candidate.local_quality * 2.6
                + (candidate.support_count * 4.5)
                + (partner_candidate.support_count * 2.0)
                + (rf_probability * self.config.ball_rf_probability_weight)
                - torso_penalty
                - distance_penalty
                - soft_penalty
            )
            if from_offscreen:
                score += 3.0 if partner_candidate.center[1] >= candidate.center[1] else -2.5
            if score > best_score:
                best_score = score
                source = "recover_rf" if recover and rf_probability >= self.config.ball_rf_min_probability else "motion_seed"
                best_choice = (candidate, partner_frame, partner_candidate, source)
        minimum_score = 9.5 if from_offscreen else 12.5
        if best_choice is None or best_score < minimum_score:
            return None
        return best_choice

    def _write_coast_state_v8(
        self,
        states: list[BallFrameState],
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        radius: float,
        mode: str,
        trail_generation: int,
    ) -> bool:
        if predicted_center is None:
            return False
        if not self._center_safe_for_coast(predicted_center):
            return False
        if self._soft_edge_zone_name(predicted_center) and not point_in_mask(predicted_center, self.ball_mask):
            return False
        if self._center_hits_torso(frame_index, predicted_center):
            return False
        self._assign_state(
            states,
            frame_index,
            predicted_center,
            radius,
            "coast",
            0.18,
            "kalman",
            mode,
            trail_generation,
        )
        return True

    def _track_frames(self) -> list[BallFrameState]:
        states = [BallFrameState() for _ in range(len(self.observations))]
        confirmed_points: list[TrajectoryPoint] = []
        kalman: Optional[cv2.KalmanFilter] = None
        mode = "SEARCH_INIT"
        misses = 0
        last_radius = self.config.ball_top_radius
        touch_boost_frames = 0
        trail_generation = 0
        offscreen_pending = 0
        offscreen_grace_count = 0
        recovering_from_offscreen = False

        for frame_index in range(self.config.bg_warmup_frames, len(self.observations)):
            predicted_center: Optional[tuple[int, int]] = None
            if kalman is not None:
                self._set_process_noise(kalman, touch_boost_frames > 0)
                predicted_center = kalman_predicted_center(kalman.predict())
                if touch_boost_frames > 0:
                    touch_boost_frames -= 1

            self._mark_frame_mode(states, frame_index, mode, trail_generation)

            if mode in {"SEARCH_INIT", "LOST"}:
                anchor = self._find_anchor_v8(frame_index, predicted_center, recover=False, from_offscreen=False)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "SEARCH_INIT",
                        trail_generation,
                    )
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    offscreen_pending = 0
                    offscreen_grace_count = 0
                continue

            if mode == "TRACK":
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "mask":
                    offscreen_pending += 1
                else:
                    offscreen_pending = 0
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    recovering_from_offscreen = True
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                if trigger == "top" or offscreen_pending >= self.config.ball_offscreen_mask_pending:
                    mode = "OFFSCREEN_GRACE"
                    recovering_from_offscreen = True
                    offscreen_grace_count = 1
                    self._offscreen_grace_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "OFFSCREEN_GRACE", trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v8(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=1.0,
                    relaxed=False,
                    mode="TRACK",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses = 1
                self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "TRACK", trail_generation)
                mode = "DEGRADED"
                continue

            if mode == "DEGRADED":
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "mask":
                    offscreen_pending += 1
                else:
                    offscreen_pending = 0
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    recovering_from_offscreen = True
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                if trigger == "top" or offscreen_pending >= self.config.ball_offscreen_mask_pending:
                    mode = "OFFSCREEN_GRACE"
                    recovering_from_offscreen = True
                    offscreen_grace_count = 1
                    self._offscreen_grace_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "OFFSCREEN_GRACE", trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v8(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=self.config.ball_degraded_gate_scale,
                    relaxed=True,
                    mode="DEGRADED",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses += 1
                if misses <= self.config.ball_coast_frames and self._write_coast_state_v8(
                    states,
                    frame_index,
                    predicted_center,
                    last_radius,
                    "DEGRADED",
                    trail_generation,
                ):
                    continue
                mode = "RECOVER"
                continue

            if mode == "OFFSCREEN_GRACE":
                self._offscreen_grace_frames.add(frame_index)
                states[frame_index].offscreen_grace = True
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v8(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=self.config.ball_degraded_gate_scale,
                    relaxed=True,
                    mode="OFFSCREEN_GRACE",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    offscreen_grace_count = 0
                    offscreen_pending = 0
                    recovering_from_offscreen = False
                    mode = "TRACK"
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                if offscreen_grace_count < self.config.ball_offscreen_grace_frames and self._write_coast_state_v8(
                    states,
                    frame_index,
                    predicted_center,
                    last_radius,
                    "OFFSCREEN_GRACE",
                    trail_generation,
                ):
                    offscreen_grace_count += 1
                else:
                    offscreen_grace_count = self.config.ball_offscreen_grace_frames
                if offscreen_grace_count >= self.config.ball_offscreen_grace_frames:
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                continue

            if mode == "OFFSCREEN":
                self._offscreen_frames.add(frame_index)
                states[frame_index].offscreen = True
                anchor = self._find_anchor_v8(frame_index, predicted_center, recover=False, from_offscreen=True)
                if anchor is not None:
                    trail_generation += 1
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "OFFSCREEN",
                        trail_generation,
                    )
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    offscreen_grace_count = 0
                    offscreen_pending = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    continue
                misses += 1
                if misses > self.config.ball_lost_after_misses:
                    mode = "RECOVER"
                    misses = 0
                continue

            if mode == "RECOVER":
                anchor = self._find_anchor_v8(frame_index, predicted_center, recover=True, from_offscreen=recovering_from_offscreen)
                if anchor is not None:
                    trail_generation += 1
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "RECOVER",
                        trail_generation,
                    )
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    offscreen_grace_count = 0
                    self.debug_info.recovery_events.append(
                        RecoveryEvent(
                            frame_index=frame_index,
                            confidence=point.confidence,
                            source=source,
                            roi=self._recovery_roi_v6(predicted_center),
                        )
                    )
                    continue
                misses += 1
                if misses > self.config.ball_lost_after_misses:
                    kalman = None
                    trail_generation += 1
                    mode = "LOST"
                    recovering_from_offscreen = False

        return states

    def _interpolate_states(self, states: list[BallFrameState]) -> None:
        confirmed_frames = [index for index, state in enumerate(states) if state.status == "confirmed" and state.center is not None]
        if len(confirmed_frames) < 2:
            return
        for previous_index, current_index in zip(confirmed_frames, confirmed_frames[1:]):
            previous_state = states[previous_index]
            current_state = states[current_index]
            if previous_state.center is None or current_state.center is None or previous_state.radius is None or current_state.radius is None:
                continue
            if previous_state.trail_generation != current_state.trail_generation:
                continue
            gap = current_index - previous_index - 1
            if gap <= 0:
                continue
            if any(states[index].mode == "OFFSCREEN" for index in range(previous_index + 1, current_index)):
                continue
            if gap > 2 and any(states[index].mode == "OFFSCREEN_GRACE" for index in range(previous_index + 1, current_index)):
                continue
            endpoint_distance = distance_between(previous_state.center, current_state.center)
            if endpoint_distance > self.config.ball_large_jump_threshold * max(1, gap):
                continue
            if gap <= self.config.ball_interp_short_gap:
                for step in range(1, gap + 1):
                    ratio = step / float(gap + 1)
                    frame_index = previous_index + step
                    center = (
                        int(round((previous_state.center[0] * (1.0 - ratio)) + (current_state.center[0] * ratio))),
                        int(round((previous_state.center[1] * (1.0 - ratio)) + (current_state.center[1] * ratio))),
                    )
                    if not point_in_mask(center, self.ball_mask):
                        continue
                    if self._center_hits_torso(frame_index, center):
                        continue
                    if self._center_in_high_risk_rect(center):
                        continue
                    self._assign_state(
                        states,
                        frame_index,
                        center,
                        (previous_state.radius * (1.0 - ratio)) + (current_state.radius * ratio),
                        "interpolated",
                        min(previous_state.confidence, current_state.confidence) * 0.7,
                        "interp_short",
                        "TRACK",
                        previous_state.trail_generation,
                    )
            elif gap <= self.config.ball_interp_medium_gap:
                window_indices = [
                    index
                    for index in range(max(0, previous_index - 2), min(len(states), current_index + 3))
                    if states[index].status == "confirmed"
                    and states[index].center is not None
                    and states[index].trail_generation == previous_state.trail_generation
                ]
                if len(window_indices) < 4:
                    continue
                window_points = [
                    TrajectoryPoint(
                        frame_index=index,
                        center=states[index].center,
                        radius=states[index].radius or (previous_state.radius + current_state.radius) / 2.0,
                        bbox=(0, 0, 1, 1),
                        confidence=states[index].confidence,
                        status=states[index].status,
                        source=states[index].source,
                    )
                    for index in window_indices
                ]
                residuals = super()._point_residuals(window_points)
                if float(np.mean(residuals)) > self.config.ball_interp_residual_limit:
                    continue
                frames = np.array(window_indices, dtype=np.float32)
                xs = np.array([states[index].center[0] for index in window_indices], dtype=np.float32)
                ys = np.array([states[index].center[1] for index in window_indices], dtype=np.float32)
                try:
                    x_coeff = np.polyfit(frames, xs, 1)
                    y_coeff = np.polyfit(frames, ys, 2)
                except np.linalg.LinAlgError:
                    continue
                for frame_index in range(previous_index + 1, current_index):
                    center = (
                        int(round(float(np.polyval(x_coeff, frame_index)))),
                        int(round(float(np.polyval(y_coeff, frame_index)))),
                    )
                    if not point_in_mask(center, self.ball_mask):
                        continue
                    if self._center_hits_torso(frame_index, center):
                        continue
                    if self._center_in_high_risk_rect(center):
                        continue
                    self._assign_state(
                        states,
                        frame_index,
                        center,
                        (previous_state.radius + current_state.radius) / 2.0,
                        "interpolated",
                        min(previous_state.confidence, current_state.confidence) * 0.55,
                        "interp_poly",
                        "TRACK",
                        previous_state.trail_generation,
                    )

    def _prune_jump_outliers(self, states: list[BallFrameState]) -> None:
        super()._prune_jump_outliers(states)
        for frame_index, state in enumerate(states):
            if state.center is None:
                continue
            zone = self._hard_risk_zone_name(state.center)
            if zone in {"left_head", "right_head"}:
                states[frame_index] = BallFrameState(
                    status="missing",
                    mode=state.mode,
                    trail_generation=state.trail_generation,
                    offscreen=state.offscreen,
                    offscreen_grace=state.offscreen_grace,
                    risk_strip=True,
                    risk_zone=zone,
                )
                continue
            if zone == "top_banner":
                states[frame_index] = BallFrameState(
                    status="missing",
                    mode=state.mode,
                    trail_generation=state.trail_generation,
                    offscreen=state.offscreen,
                    offscreen_grace=state.offscreen_grace,
                    risk_strip=True,
                    risk_zone=zone,
                )

    def _finalize_debug_v8(self, states: list[BallFrameState]) -> None:
        gap_start: Optional[int] = None
        flagged: set[int] = set()
        previous_rendered: Optional[tuple[int, int]] = None
        previous_generation: Optional[int] = None
        self.debug_info.reentry_reject_frames = sorted(self._risk_reject_frames)
        self.debug_info.offscreen_frames = sorted(self._offscreen_frames)
        for index, state in enumerate(states):
            if state.center is None:
                if gap_start is None:
                    gap_start = index
            else:
                if gap_start is not None and index - gap_start >= self.config.ball_ml_gap_trigger:
                    self.debug_info.long_gaps.append((gap_start, index - 1))
                    flagged.update({gap_start, gap_start + ((index - gap_start) // 2), index - 1})
                gap_start = None
                if state.status in {"confirmed", "interpolated"}:
                    if (
                        previous_rendered is not None
                        and previous_generation == state.trail_generation
                        and distance_between(previous_rendered, state.center) > self.config.ball_large_jump_threshold
                    ):
                        self.debug_info.large_jump_frames.append(index)
                        flagged.add(index)
                    previous_rendered = state.center
                    previous_generation = state.trail_generation
                zone = self._risk_zone_name(state.center)
                if zone:
                    self.debug_info.risk_strip_frames.append(index)
                    flagged.add(index)
            if state.mode == "OFFSCREEN":
                flagged.add(index)
            if state.offscreen_grace:
                self._offscreen_grace_frames.add(index)
            if state.center is not None and self._center_hits_torso(index, state.center):
                self.debug_info.overlap_frames.append(index)
                flagged.add(index)
        if gap_start is not None and len(states) - gap_start >= self.config.ball_ml_gap_trigger:
            self.debug_info.long_gaps.append((gap_start, len(states) - 1))
            flagged.update({gap_start, gap_start + ((len(states) - gap_start) // 2), len(states) - 1})
        for event in self.debug_info.recovery_events:
            flagged.add(event.frame_index)
        flagged.update(self.debug_info.large_jump_frames)
        flagged.update(self.debug_info.reentry_reject_frames)
        self.debug_info.flagged_frames = sorted(flagged)


class BallTrajectoryTrackerV9(BallTrajectoryTrackerV8):
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
        super().__init__(config, frame_shape, fps, observations, player_states, input_path, transforms, recovery_model)
        self.candidate_keep_limit = max(config.ball_candidate_keep_per_frame, 24)
        self.rf_probability_floor = max(0.18, config.ball_rf_min_probability - 0.06)
        self.search_anchor_floor = 10.2
        self.recover_anchor_floor = 8.8
        self.offscreen_anchor_floor = 6.8

    def build(self) -> tuple[list[BallFrameState], BallDebugInfo]:
        self._prepare_candidates()
        self.recovery_model.ensure_ready(
            self.candidates_by_frame,
            self.player_states,
            self.frame_shape,
            self.label_map,
            self._expected_radius,
        )
        states = self._track_frames()
        self._interpolate_states(states)
        self._prune_jump_outliers(states)
        self._finalize_debug_v9(states)
        return states, self.debug_info

    def _prepare_candidates(self) -> None:
        filtered: list[list[BallCandidate]] = []
        for frame_index, observation in enumerate(self.observations):
            suppression_boxes = self.player_states[frame_index].suppression_boxes if frame_index < len(self.player_states) else []
            frame_candidates: list[BallCandidate] = []
            for candidate in observation.ball_candidates:
                if not self._passes_base_sieve(candidate, suppression_boxes):
                    continue
                candidate.local_quality = self._local_quality_v6(candidate)
                frame_candidates.append(candidate)
            frame_candidates.sort(key=lambda item: item.local_quality, reverse=True)
            filtered.append(frame_candidates[: self.candidate_keep_limit])
        self.candidates_by_frame = filtered

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            for candidate in frame_candidates:
                support_count, support_progress = self._temporal_support_v6(frame_index, candidate)
                candidate.support_count = support_count
                candidate.support_progress = support_progress

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            pruned: list[BallCandidate] = []
            for candidate in frame_candidates:
                zone = self._hard_risk_zone_name(candidate.center)
                if zone in {"left_head", "right_head"} and candidate.support_count <= 0:
                    continue
                if zone == "top_banner" and (candidate.support_count <= 0 or not candidate.source_median):
                    continue
                if (
                    candidate.center[1] < self.config.ball_top_banner_guard_y
                    and candidate.support_count <= 0
                    and not candidate.source_median
                    and self._soft_edge_zone_name(candidate.center)
                ):
                    continue
                pruned.append(candidate)
            self.candidates_by_frame[frame_index] = pruned[: self.candidate_keep_limit]

    def _is_central_point(self, center: tuple[int, int]) -> bool:
        return (
            not self._soft_edge_zone_name(center)
            and not self._hard_risk_zone_name(center)
            and center[1] >= self.config.ball_high_risk_rects[2][3]
        )

    def _is_central_candidate(self, candidate: BallCandidate) -> bool:
        return self._is_central_point(candidate.center)

    def _soft_edge_allowed(self, candidate: BallCandidate, from_offscreen: bool) -> bool:
        if not from_offscreen:
            return True
        if not self._soft_edge_zone_name(candidate.center):
            return True
        return candidate.source_median and (
            candidate.support_count >= 1 or candidate.support_progress >= (self.config.ball_seed_min_progress + 1.0)
        )

    def _candidate_mode_allowed_v9(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        relaxed: bool,
        mode: str,
        predicted_center: Optional[tuple[int, int]],
        from_offscreen: bool,
    ) -> bool:
        if not self._candidate_mode_allowed_v6(candidate, suppression_boxes, relaxed):
            return False
        if self._hard_risk_zone_name(candidate.center):
            return False
        if mode in {"SEARCH_INIT", "RECOVER", "OFFSCREEN"} and not self._center_safe_for_reentry(candidate.center):
            return False
        if not self._soft_edge_allowed(candidate, from_offscreen):
            return False
        if (
            self._soft_edge_zone_name(candidate.center)
            and candidate.center[1] < self.config.ball_top_banner_guard_y
            and candidate.support_count <= 0
            and not candidate.source_median
        ):
            return False
        if (
            from_offscreen
            and self._soft_edge_zone_name(candidate.center)
            and predicted_center is not None
            and not point_in_mask(predicted_center, self.ball_mask)
            and distance_between(predicted_center, candidate.center) > (self.config.ball_max_gate_per_step * 1.4)
        ):
            return False
        return True

    def _select_candidate_v9(
        self,
        frame_index: int,
        predicted_center: tuple[int, int],
        kalman: cv2.KalmanFilter,
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
        mode: str,
        from_offscreen: bool,
    ) -> Optional[tuple[BallCandidate, float, float]]:
        best_choice: Optional[tuple[BallCandidate, float, float]] = None
        gate = self.config.ball_max_gate_per_step * gate_scale
        suppression_boxes = self.player_states[frame_index].suppression_boxes
        for candidate in self.candidates_by_frame[frame_index]:
            if not self._candidate_mode_allowed_v9(candidate, suppression_boxes, relaxed, mode, predicted_center, from_offscreen):
                continue
            if distance_between(predicted_center, candidate.center) > gate:
                continue
            score, innovation = self._candidate_score_v6(
                frame_index,
                candidate,
                predicted_center,
                kalman,
                confirmed_points,
                0.0,
            )
            if score is None:
                continue
            zone = self._soft_edge_zone_name(candidate.center)
            if zone and from_offscreen:
                score -= 1.2
            elif zone:
                score -= 0.2
            if self._is_central_candidate(candidate) and not self._candidate_inside_suppression(candidate, suppression_boxes):
                score += 1.6
                if candidate.support_count > 0:
                    score += 1.4
                if candidate.center[1] < 335:
                    score += 0.8
            if candidate.source_median and candidate.support_count > 0:
                score += 0.9
            if best_choice is None or score > best_choice[1]:
                best_choice = (candidate, score, innovation)
        min_score = 2.1 if relaxed else 4.6
        if best_choice is None or best_choice[1] < min_score:
            return None
        return best_choice

    def _apply_tracking_measurement_v9(
        self,
        frame_index: int,
        states: list[BallFrameState],
        kalman: Optional[cv2.KalmanFilter],
        predicted_center: Optional[tuple[int, int]],
        confirmed_points: list[TrajectoryPoint],
        gate_scale: float,
        relaxed: bool,
        mode: str,
        trail_generation: int,
        from_offscreen: bool,
    ) -> tuple[bool, bool, Optional[BallCandidate]]:
        if kalman is None or predicted_center is None:
            return False, False, None
        selected = self._select_candidate_v9(
            frame_index,
            predicted_center,
            kalman,
            confirmed_points,
            gate_scale,
            relaxed,
            mode,
            from_offscreen,
        )
        if selected is None:
            return False, False, None
        candidate, score, innovation = selected
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]], dtype=np.float32)
        kalman.correct(measurement)
        point = super()._point_from_candidate(frame_index, candidate, "confirmed")
        point.confidence += score * 0.08
        self._assign_state(
            states,
            frame_index,
            point.center,
            point.radius,
            "confirmed",
            point.confidence,
            point.source,
            mode,
            trail_generation,
        )
        confirmed = states[frame_index]
        confirmed.central_reacquire = mode in {"SEARCH_INIT", "RECOVER", "OFFSCREEN", "OFFSCREEN_GRACE"} and self._is_central_candidate(candidate)
        confirmed_points.append(point)
        return True, innovation >= self.config.ball_touch_innovation_threshold, candidate

    def _anchor_direction_ok_v9(
        self,
        seed: BallCandidate,
        partner: BallCandidate,
        from_offscreen: bool,
    ) -> bool:
        seed_soft = self._soft_edge_zone_name(seed.center)
        partner_soft = self._soft_edge_zone_name(partner.center)
        central_pair = not seed_soft and not partner_soft
        distance = distance_between(seed.center, partner.center)
        if central_pair:
            if distance < 5.0:
                return False
        elif distance < self.config.ball_reentry_min_progress:
            return False
        if not from_offscreen:
            return True
        if central_pair:
            return True
        dy = partner.center[1] - seed.center[1]
        dx = partner.center[0] - seed.center[0]
        descending = dy >= self.config.ball_reentry_downward_bias
        inward = False
        if seed_soft == "left_edge" or partner_soft == "left_edge":
            inward = inward or (dx >= self.config.ball_reentry_inward_bias)
        if seed_soft == "right_edge" or partner_soft == "right_edge":
            inward = inward or (dx <= -self.config.ball_reentry_inward_bias)
        return descending or inward

    def _find_confirmation_partner_v9(
        self,
        frame_index: int,
        seed: BallCandidate,
        from_offscreen: bool,
    ) -> Optional[tuple[int, BallCandidate]]:
        best_choice: Optional[tuple[int, BallCandidate]] = None
        best_score = -1e9
        seed_zone = self._hard_risk_zone_name(seed.center)
        seed_soft = self._soft_edge_zone_name(seed.center)
        for delta in range(1, self.config.ball_seed_search_gap + 1):
            future_index = frame_index + delta
            if future_index >= len(self.candidates_by_frame):
                break
            for candidate in self.candidates_by_frame[future_index]:
                candidate_zone = self._hard_risk_zone_name(candidate.center)
                if seed_zone and seed_zone == candidate_zone:
                    continue
                candidate_soft = self._soft_edge_zone_name(candidate.center)
                central_pair = not seed_soft and not candidate_soft
                distance = distance_between(seed.center, candidate.center)
                min_progress = self.config.ball_seed_min_progress * delta
                if central_pair:
                    min_progress = max(4.0, (self.config.ball_seed_min_progress - 2.5) * delta)
                elif from_offscreen:
                    min_progress = self.config.ball_reentry_min_progress * delta
                if distance < min_progress:
                    continue
                if distance > self.config.ball_seed_max_progress * delta:
                    continue
                if from_offscreen and (not self._center_safe_for_reentry(candidate.center) or not self._soft_edge_allowed(candidate, True)):
                    continue
                if not self._anchor_direction_ok_v9(seed, candidate, from_offscreen):
                    continue
                score = (
                    (candidate.local_quality * 5.6)
                    + (candidate.support_count * 3.3)
                    + (candidate.support_progress * 0.025)
                    - (distance * 0.075)
                )
                if central_pair:
                    score += 1.2
                if score > best_score:
                    best_score = score
                    best_choice = (future_index, candidate)
        return best_choice

    def _find_anchor_v9(
        self,
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        recover: bool,
        from_offscreen: bool,
    ) -> Optional[tuple[BallCandidate, int, BallCandidate, str]]:
        frame_candidates = self.candidates_by_frame[frame_index]
        if not frame_candidates:
            return None
        suppression_boxes = (
            self._recovery_suppression_boxes(frame_index)
            if recover or from_offscreen
            else self.player_states[frame_index].suppression_boxes
        )
        filtered_candidates = []
        for candidate in frame_candidates:
            if not self._candidate_mode_allowed_v9(
                candidate,
                suppression_boxes,
                True,
                "OFFSCREEN" if from_offscreen else ("RECOVER" if recover else "SEARCH_INIT"),
                predicted_center,
                from_offscreen,
            ):
                if recover or from_offscreen:
                    self._record_reentry_reject(frame_index)
                continue
            filtered_candidates.append(candidate)
        if not filtered_candidates:
            return None
        recovery_scores = (
            self.recovery_model.score_candidates(
                frame_index,
                filtered_candidates,
                suppression_boxes,
                self.frame_shape,
                self._expected_radius,
            )
            if recover and self.recovery_model.enabled
            else {}
        )
        best_choice: Optional[tuple[BallCandidate, int, BallCandidate, str]] = None
        best_score = -1e9
        for candidate in filtered_candidates:
            if predicted_center is not None and recover and distance_between(predicted_center, candidate.center) > self.config.ball_recover_fullframe_gate:
                continue
            partner = self._find_confirmation_partner_v9(frame_index, candidate, from_offscreen)
            if partner is None:
                if recover or from_offscreen:
                    self._record_reentry_reject(frame_index)
                continue
            partner_frame, partner_candidate = partner
            rf_probability = recovery_scores.get(candidate.candidate_id, 0.0)
            if recover and recovery_scores and rf_probability < self.rf_probability_floor and candidate.support_count <= 0 and not candidate.source_median:
                self._record_reentry_reject(frame_index)
                continue
            torso_penalty = 18.0 if self._candidate_inside_suppression(candidate, suppression_boxes) else 0.0
            distance_penalty_factor = 0.10 if recover else 0.12
            distance_penalty = 0.0 if predicted_center is None else distance_between(predicted_center, candidate.center) * distance_penalty_factor
            soft_penalty = 0.0
            candidate_soft = self._soft_edge_zone_name(candidate.center)
            partner_soft = self._soft_edge_zone_name(partner_candidate.center)
            if from_offscreen and (candidate_soft or partner_soft):
                soft_penalty += 1.2
            central_bonus = 0.0
            if self._is_central_candidate(candidate) and not self._candidate_inside_suppression(candidate, suppression_boxes):
                central_bonus += 2.8
                if candidate.center[1] < 335:
                    central_bonus += 1.0
            if self._is_central_candidate(partner_candidate):
                central_bonus += 1.0
            support_bonus = 0.0
            if candidate.support_count > 0:
                support_bonus += 1.6
            if partner_candidate.support_count > 0:
                support_bonus += 1.0
            if candidate.source_median and partner_candidate.source_median:
                support_bonus += 0.8
            score = (
                candidate.local_quality * 6.2
                + partner_candidate.local_quality * 2.7
                + (candidate.support_count * 4.8)
                + (partner_candidate.support_count * 2.2)
                + (candidate.support_progress * 0.02)
                + support_bonus
                + central_bonus
                + (rf_probability * self.config.ball_rf_probability_weight)
                - torso_penalty
                - distance_penalty
                - soft_penalty
            )
            if from_offscreen:
                score += 2.2 if partner_candidate.center[1] >= candidate.center[1] else -1.4
            if score > best_score:
                best_score = score
                source = "recover_rf" if recover and rf_probability >= self.rf_probability_floor else "motion_seed"
                best_choice = (candidate, partner_frame, partner_candidate, source)
        minimum_score = self.offscreen_anchor_floor if from_offscreen else (self.recover_anchor_floor if recover else self.search_anchor_floor)
        if best_choice is None or best_score < minimum_score:
            return None
        return best_choice

    def _track_frames(self) -> list[BallFrameState]:
        states = [BallFrameState() for _ in range(len(self.observations))]
        confirmed_points: list[TrajectoryPoint] = []
        kalman: Optional[cv2.KalmanFilter] = None
        mode = "SEARCH_INIT"
        misses = 0
        last_radius = self.config.ball_top_radius
        touch_boost_frames = 0
        trail_generation = 0
        offscreen_pending = 0
        offscreen_grace_count = 0
        recovering_from_offscreen = False

        for frame_index in range(self.config.bg_warmup_frames, len(self.observations)):
            predicted_center: Optional[tuple[int, int]] = None
            if kalman is not None:
                self._set_process_noise(kalman, touch_boost_frames > 0)
                predicted_center = kalman_predicted_center(kalman.predict())
                if touch_boost_frames > 0:
                    touch_boost_frames -= 1

            self._mark_frame_mode(states, frame_index, mode, trail_generation)

            if mode in {"SEARCH_INIT", "LOST"}:
                anchor = self._find_anchor_v9(frame_index, predicted_center, recover=False, from_offscreen=False)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "SEARCH_INIT",
                        trail_generation,
                    )
                    states[frame_index].central_reacquire = self._is_central_candidate(candidate)
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    offscreen_pending = 0
                    offscreen_grace_count = 0
                continue

            if mode == "TRACK":
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "mask":
                    offscreen_pending += 1
                else:
                    offscreen_pending = 0
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    recovering_from_offscreen = True
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                if trigger == "top" or offscreen_pending >= self.config.ball_offscreen_mask_pending:
                    mode = "OFFSCREEN_GRACE"
                    recovering_from_offscreen = True
                    offscreen_grace_count = 1
                    self._offscreen_grace_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "OFFSCREEN_GRACE", trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v9(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=1.0,
                    relaxed=False,
                    mode="TRACK",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses = 1
                self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "TRACK", trail_generation)
                mode = "DEGRADED"
                continue

            if mode == "DEGRADED":
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "mask":
                    offscreen_pending += 1
                else:
                    offscreen_pending = 0
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    recovering_from_offscreen = True
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                if trigger == "top" or offscreen_pending >= self.config.ball_offscreen_mask_pending:
                    mode = "OFFSCREEN_GRACE"
                    recovering_from_offscreen = True
                    offscreen_grace_count = 1
                    self._offscreen_grace_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    self._write_coast_state_v8(states, frame_index, predicted_center, last_radius, "OFFSCREEN_GRACE", trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v9(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=self.config.ball_degraded_gate_scale,
                    relaxed=True,
                    mode="DEGRADED",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                misses += 1
                if misses <= self.config.ball_coast_frames and self._write_coast_state_v8(
                    states,
                    frame_index,
                    predicted_center,
                    last_radius,
                    "DEGRADED",
                    trail_generation,
                ):
                    continue
                mode = "RECOVER"
                continue

            if mode == "OFFSCREEN_GRACE":
                self._offscreen_grace_frames.add(frame_index)
                states[frame_index].offscreen_grace = True
                trigger = self._offscreen_trigger_kind(predicted_center, confirmed_points)
                if trigger == "hard":
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    self._offscreen_frames.add(frame_index)
                    self._mark_frame_mode(states, frame_index, mode, trail_generation)
                    continue
                success, touch_event, candidate = self._apply_tracking_measurement_v9(
                    frame_index,
                    states,
                    kalman,
                    predicted_center,
                    confirmed_points,
                    gate_scale=self.config.ball_degraded_gate_scale,
                    relaxed=True,
                    mode="OFFSCREEN_GRACE",
                    trail_generation=trail_generation,
                    from_offscreen=False,
                )
                if success and candidate is not None:
                    last_radius = candidate.radius
                    misses = 0
                    offscreen_grace_count = 0
                    offscreen_pending = 0
                    recovering_from_offscreen = False
                    mode = "TRACK"
                    if touch_event:
                        touch_boost_frames = 2
                    continue
                if offscreen_grace_count < self.config.ball_offscreen_grace_frames and self._write_coast_state_v8(
                    states,
                    frame_index,
                    predicted_center,
                    last_radius,
                    "OFFSCREEN_GRACE",
                    trail_generation,
                ):
                    offscreen_grace_count += 1
                    continue
                offscreen_grace_count = self.config.ball_offscreen_grace_frames
                mode = "RECOVER"
                misses = 0
                continue

            if mode == "OFFSCREEN":
                self._offscreen_frames.add(frame_index)
                states[frame_index].offscreen = True
                anchor = self._find_anchor_v9(frame_index, predicted_center, recover=False, from_offscreen=True)
                if anchor is not None:
                    trail_generation += 1
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "OFFSCREEN",
                        trail_generation,
                    )
                    states[frame_index].central_reacquire = self._is_central_candidate(candidate)
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    offscreen_grace_count = 0
                    offscreen_pending = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    continue
                misses += 1
                if misses >= 2:
                    mode = "RECOVER"
                    misses = 0
                continue

            if mode == "RECOVER":
                anchor = self._find_anchor_v9(frame_index, predicted_center, recover=True, from_offscreen=recovering_from_offscreen)
                if anchor is not None:
                    new_generation = trail_generation + 1 if mode == "RECOVER" and misses > 0 and predicted_center is not None and distance_between(predicted_center, anchor[0].center) > self.config.ball_large_jump_threshold else trail_generation
                    trail_generation = new_generation
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    self._assign_state(
                        states,
                        frame_index,
                        point.center,
                        point.radius,
                        "confirmed",
                        point.confidence,
                        source,
                        "RECOVER",
                        trail_generation,
                    )
                    states[frame_index].central_reacquire = self._is_central_candidate(candidate)
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                    recovering_from_offscreen = False
                    offscreen_pending = 0
                    offscreen_grace_count = 0
                    self.debug_info.recovery_events.append(
                        RecoveryEvent(
                            frame_index=frame_index,
                            confidence=point.confidence,
                            source=source,
                            roi=self._recovery_roi_v6(predicted_center),
                        )
                    )
                    continue
                misses += 1
                if recovering_from_offscreen and predicted_center is not None and self._center_in_high_risk_rect(predicted_center):
                    trail_generation += 1
                    mode = "OFFSCREEN"
                    misses = 0
                    self._offscreen_frames.add(frame_index)
                    continue
                if misses > self.config.ball_lost_after_misses:
                    kalman = None
                    trail_generation += 1
                    mode = "LOST"
                    recovering_from_offscreen = False

        return states

    def _finalize_debug_v9(self, states: list[BallFrameState]) -> None:
        self._finalize_debug_v8(states)
        self.debug_info.central_reacquire_frames = sorted(
            index for index, state in enumerate(states) if state.central_reacquire
        )


def draw_players(frame: np.ndarray, tracks: list[VisiblePlayer]) -> None:
    for track in tracks:
        x, y, w, h = track.bbox
        color = (0, 215, 255) if track.stable_team == "team_a" else (255, 170, 70)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ball(frame: np.ndarray, states: list[BallFrameState], frame_index: int) -> None:
    start = max(0, frame_index - 44)
    max_trail_step = 90.0
    current_generation = states[frame_index].trail_generation
    trail_points: list[tuple[int, int]] = []
    for state in states[start : frame_index + 1]:
        if state.center is None:
            trail_points.append((-1, -1))
        elif state.trail_generation != current_generation:
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
    large_jump_frames = set(debug_info.large_jump_frames)
    risk_strip_frames = set(debug_info.risk_strip_frames)
    offscreen_frames = set(debug_info.offscreen_frames)
    reentry_reject_frames = set(debug_info.reentry_reject_frames)
    central_reacquire_frames = set(debug_info.central_reacquire_frames)
    gap_frames: set[int] = set()
    for start, end in debug_info.long_gaps:
        gap_frames.update(range(start, end + 1))
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "status",
                "mode",
                "confidence",
                "source",
                "x",
                "y",
                "recovery_ml",
                "torso_overlap",
                "long_gap",
                "large_jump",
                "offscreen",
                "offscreen_grace",
                "risk_strip",
                "risk_zone",
                "reentry_reject",
                "central_reacquire",
            ]
        )
        for frame_index, state in enumerate(ball_states):
            writer.writerow(
                [
                    frame_index,
                    state.status,
                    state.mode,
                    f"{state.confidence:.3f}",
                    state.source,
                    "" if state.center is None else state.center[0],
                    "" if state.center is None else state.center[1],
                    int(frame_index in recovery_frames),
                    int(frame_index in overlap_frames),
                    int(frame_index in gap_frames),
                    int(frame_index in large_jump_frames),
                    int(frame_index in offscreen_frames or state.offscreen),
                    int(state.offscreen_grace),
                    int(frame_index in risk_strip_frames or state.risk_strip),
                    state.risk_zone,
                    int(frame_index in reentry_reject_frames or state.reentry_reject),
                    int(frame_index in central_reacquire_frames or state.central_reacquire),
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
        label = f"F{frame_index} {state.status} {state.mode} {state.risk_zone}".strip()
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
    parser = argparse.ArgumentParser(description="Liberal reacquisition volleyball tracker v9")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "task phase 6" / "Volleyball.mp4",
        help="Path to the input volleyball video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "tp6" / "Volleyball_annotated_opencv_v9.mp4",
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
        "--disable-ball-recovery-ml",
        action="store_true",
        help="Disable the recovery-only RandomForest reranker and keep v9 fully OpenCV-driven.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory for v9 debug artifacts.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = SceneConfigV5()
    root = Path(__file__).resolve().parents[1]

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
    artifact_path = root / "tp6" / "_artifacts" / "ball_recovery_rf_v9.joblib"
    dataset_root = root / "task phase 6" / "datasets" / "ball"
    recovery_model = BallRecoveryModelV5(artifact_path, dataset_root, config, args.disable_ball_recovery_ml)
    ball_tracker = BallTrajectoryTrackerV9(
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
            cv2.imshow("Volleyball Tracker OpenCV V9", annotated)
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
