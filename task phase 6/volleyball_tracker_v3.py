from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def pick_device() -> str:
    # Pick the best available device automatically.
    # On this machine, MPS was used for Apple Silicon acceleration.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


@dataclass
class TrackerConfig:
    # All important thresholds live here so the tracker logic stays readable.
    #
    # Think of this as the "control panel" for the whole system.
    # If behavior needs tuning, most changes happen here.
    blur_kernel: tuple[int, int] = (5, 5)
    trail_length: int = 28
    max_players_per_team: int = 6
    player_confidence: float = 0.25
    player_iou: float = 0.45
    player_imgsz: int = 960
    player_track_ttl: int = 15
    player_active_ttl: int = 10
    player_vote_gain: float = 1.0
    player_vote_decay: float = 0.05
    player_stable_threshold: float = 2.5
    torso_x_margin_ratio: float = 0.28
    torso_y_start_ratio: float = 0.14
    torso_y_end_ratio: float = 0.45
    team_color_floor_ratio: float = 0.012
    team_color_margin: float = 0.004
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
    ball_confidence_roi: float = 0.02
    ball_confidence_full: float = 0.05
    ball_iou: float = 0.45
    ball_imgsz_roi: int = 960
    ball_imgsz_full: int = 1280
    ball_full_redetect_interval: int = 6
    ball_max_misses: int = 6
    ball_predicted_draw_misses: int = 4
    ball_base_roi: int = 190
    ball_velocity_roi_gain: float = 2.4
    ball_miss_roi_gain: float = 34.0
    ball_max_roi: int = 420
    ball_min_box_size: int = 4
    ball_max_box_size: int = 34
    ball_distance_gate: float = 190.0
    ball_predicted_distance_gate: float = 110.0
    ball_inside_player_gate: float = 40.0
    ball_max_uncertainty: float = 1800.0
    ball_motion_threshold: int = 18
    ball_motion_ratio_min: float = 0.025
    ball_color_ratio_min: float = 0.015
    ball_inside_player_penalty: float = 2.2
    ball_dual_color_bonus: float = 1.25
    ball_motion_bonus: float = 1.1
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
class BallDetection:
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str
    yellow_ratio: float
    blue_ratio: float
    motion_ratio: float
    inside_player: bool


@dataclass
class BallState:
    center: Optional[tuple[int, int]]
    radius: Optional[float]
    status: str


@dataclass
class PlayerState:
    track_id: int
    bbox: tuple[int, int, int, int]
    footpoint: tuple[int, int]
    side: str
    votes: dict[str, float] = field(default_factory=lambda: {"team_a": 0.0, "team_b": 0.0})
    stable_team: Optional[str] = None
    last_seen: int = 0


def build_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def apply_exclusions(
    mask: np.ndarray,
    exclusions: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    for x1, y1, x2, y2 in exclusions:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    return mask


def point_in_mask(point: tuple[int, int], mask: np.ndarray) -> bool:
    x, y = point
    if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
        return False
    return bool(mask[y, x])


def clip_bbox(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
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


def point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def distance_between(point_a: tuple[int, int], point_b: tuple[int, int]) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def crop_roi(frame: np.ndarray, center: tuple[int, int], roi_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    half = roi_size // 2
    x1 = max(center[0] - half, 0)
    y1 = max(center[1] - half, 0)
    x2 = min(center[0] + half, frame.shape[1])
    y2 = min(center[1] + half, frame.shape[0])
    return frame[y1:y2, x1:x2], (x1, y1)


class TeamTracker:
    def __init__(self, config: TrackerConfig, top_mask: np.ndarray, bottom_mask: np.ndarray) -> None:
        self.config = config
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.players: dict[int, PlayerState] = {}

    def update(self, frame: np.ndarray, result, frame_index: int) -> tuple[list[PlayerState], int, int]:
        # This function updates the player state for ONE frame.
        #
        # Input:
        # - current video frame
        # - YOLO+ByteTrack result for people
        # - current frame number
        #
        # Output:
        # - active players that should be drawn
        # - Team A count
        # - Team B count
        # If tracking data is missing this frame, just age old tracks and keep recent ones alive briefly.
        if result.boxes is None or not result.boxes.is_track:
            self._age_tracks(frame_index)
            return self.active_players(frame_index)

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        seen_ids: set[int] = set()
        for xyxy, track_id in zip(boxes, ids):
            bbox = clip_bbox(
                (
                    int(round(xyxy[0])),
                    int(round(xyxy[1])),
                    int(round(xyxy[2] - xyxy[0])),
                    int(round(xyxy[3] - xyxy[1])),
                ),
                frame.shape,
            )
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            footpoint = bbox_footpoint(bbox)
            # Decide whether this player belongs to the top side or bottom side of the court.
            # We use the feet, not the box center, because feet tell us where the player is standing.
            if point_in_mask(footpoint, self.top_mask):
                expected_team = "team_a"
                side = "top"
            elif point_in_mask(footpoint, self.bottom_mask):
                expected_team = "team_b"
                side = "bottom"
            else:
                continue

            seen_ids.add(track_id)
            player = self.players.get(track_id)
            if player is None:
                # First time seeing this tracked player ID.
                player = PlayerState(
                    track_id=track_id,
                    bbox=bbox,
                    footpoint=footpoint,
                    side=side,
                    last_seen=frame_index,
                )
                self.players[track_id] = player

            player.bbox = bbox
            player.footpoint = footpoint
            player.side = side
            player.last_seen = frame_index

            observed_team = classify_team(frame, bbox, expected_team, self.config)
            if observed_team is not None:
                # Build up a stable team label over time instead of flipping on one bad frame.
                player.votes[observed_team] += self.config.player_vote_gain
                other_team = "team_b" if observed_team == "team_a" else "team_a"
                player.votes[other_team] = max(0.0, player.votes[other_team] - self.config.player_vote_decay)
            else:
                for team_name in player.votes:
                    player.votes[team_name] = max(0.0, player.votes[team_name] - self.config.player_vote_decay)

            if player.votes["team_a"] >= self.config.player_stable_threshold and player.votes["team_a"] > player.votes["team_b"]:
                player.stable_team = "team_a"
            elif player.votes["team_b"] >= self.config.player_stable_threshold and player.votes["team_b"] > player.votes["team_a"]:
                player.stable_team = "team_b"
            elif player.stable_team is None:
                player.stable_team = expected_team

        self._age_tracks(frame_index)
        return self.active_players(frame_index)

    def _age_tracks(self, frame_index: int) -> None:
        stale_ids = [
            track_id
            for track_id, player in self.players.items()
            if frame_index - player.last_seen > self.config.player_track_ttl
        ]
        for track_id in stale_ids:
            del self.players[track_id]

    def active_players(self, frame_index: int) -> tuple[list[PlayerState], int, int]:
        # Only count players seen recently.
        # This keeps counts stable when one frame has a weak detection.
        active = [
            player
            for player in self.players.values()
            if frame_index - player.last_seen <= self.config.player_active_ttl and player.stable_team is not None
        ]
        team_a_count = min(sum(1 for player in active if player.stable_team == "team_a"), self.config.max_players_per_team)
        team_b_count = min(sum(1 for player in active if player.stable_team == "team_b"), self.config.max_players_per_team)
        return active, team_a_count, team_b_count


def classify_team(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    expected_team: str,
    config: TrackerConfig,
) -> Optional[str]:
    # This is the team-classification function.
    #
    # It does NOT use deep learning.
    # It uses simple jersey-color analysis in HSV space.
    #
    # Why HSV?
    # Because HSV is usually easier than raw BGR/RGB for separating colors like yellow and blue.
    # Only look at the torso area because jersey color is clearest there.
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
    # Measure how much of the crop looks yellow or blue.
    yellow_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.yellow_player_lower, config.yellow_player_upper)
    ) / float(max(crop.shape[0] * crop.shape[1], 1))
    blue_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.blue_player_lower, config.blue_player_upper)
    ) / float(max(crop.shape[0] * crop.shape[1], 1))
    red_ratio = cv2.countNonZero(
        cv2.bitwise_or(
            cv2.inRange(hsv_crop, config.red_lower_1, config.red_upper_1),
            cv2.inRange(hsv_crop, config.red_lower_2, config.red_upper_2),
        )
    ) / float(max(crop.shape[0] * crop.shape[1], 1))
    white_ratio = cv2.countNonZero(
        cv2.inRange(hsv_crop, config.white_player_lower, config.white_player_upper)
    ) / float(max(crop.shape[0] * crop.shape[1], 1))

    if red_ratio > max(yellow_ratio, blue_ratio) and red_ratio >= 0.02:
        # Red usually means staff/libero/background in this clip, so do not force a team label here.
        return None
    if yellow_ratio >= config.team_color_floor_ratio and yellow_ratio > blue_ratio + config.team_color_margin:
        return "team_a"
    if blue_ratio >= config.team_color_floor_ratio and blue_ratio > yellow_ratio + config.team_color_margin:
        return "team_b"
    if white_ratio >= 0.08:
        return expected_team
    return expected_team


class BallTracker:
    def __init__(self, model: YOLO, config: TrackerConfig, search_mask: np.ndarray, device: str) -> None:
        # The ball tracker combines:
        # 1. a custom YOLO detector for the volleyball
        # 2. a Kalman filter for smooth prediction between detections
        # 3. validation rules so the tracker does not jump to wrong yellow objects
        self.model = model
        self.config = config
        self.search_mask = search_mask
        self.device = device
        self.kalman = self._build_kalman()
        self.initialized = False
        self.last_confirmed_center: Optional[tuple[int, int]] = None
        self.last_confirmed_frame = -1
        self.last_full_redetect_frame = -1
        self.miss_count = 0
        self.trail: deque[tuple[int, int]] = deque(maxlen=config.trail_length)
        self.previous_gray: Optional[np.ndarray] = None

    def _build_kalman(self) -> cv2.KalmanFilter:
        # State = [x, y, vx, vy]
        # The filter predicts where the ball should move next.
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

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        player_boxes: list[tuple[int, int, int, int]],
    ) -> BallState:
        # This is the main ball-tracking function for ONE frame.
        #
        # High-level flow:
        # 1. Predict where the ball should be
        # 2. Build a motion mask
        # 3. Search near the predicted area first
        # 4. If needed, search the full frame
        # 5. Score all ball candidates
        # 6. If a good detection exists, correct the Kalman filter
        # 7. Otherwise use a short predicted bridge or declare the ball missing
        # Step 1: predict where the ball should be now.
        prediction = self._predict()
        predicted_center = (int(round(prediction[0])), int(round(prediction[1]))) if prediction is not None else None

        # Step 2: build a simple motion mask so static yellow objects are less likely to be accepted.
        motion_mask = self._build_motion_mask(frame)

        roi_detection: list[BallDetection] = []
        roi_size = self._roi_size(prediction)
        if predicted_center is not None and point_in_mask(predicted_center, self.search_mask):
            # Step 3: when we already have a ball estimate, search near that area first.
            roi_frame, offset = crop_roi(frame, predicted_center, roi_size)
            roi_detection = self._detect_ball(
                roi_frame,
                offset=offset,
                conf=self.config.ball_confidence_roi,
                imgsz=self.config.ball_imgsz_roi,
                source="roi",
                motion_mask=motion_mask,
                player_boxes=player_boxes,
                prediction=prediction,
            )

        need_full = (
            not self.initialized
            or self.miss_count > 0
            or frame_index == 0
            or frame_index - self.last_full_redetect_frame >= self.config.ball_full_redetect_interval
            or predicted_center is None
            or not point_in_mask(predicted_center, self.search_mask)
        )
        full_detection: list[BallDetection] = []
        if need_full:
            # Step 4: if tracking is weak or stale, run a wider full-frame search.
            full_detection = self._detect_ball(
                frame,
                offset=(0, 0),
                conf=self.config.ball_confidence_full,
                imgsz=self.config.ball_imgsz_full,
                source="full",
                motion_mask=motion_mask,
                player_boxes=player_boxes,
                prediction=prediction,
            )
            self.last_full_redetect_frame = frame_index

        detection = self._choose_detection(roi_detection + full_detection, prediction)
        if detection is not None:
            # Step 5: if we found a good ball candidate, correct the Kalman filter with it.
            state = self._correct_with_detection(detection, frame_index)
            return state

        # Step 6: if no good detection exists, fall back to short-term prediction only.
        return self._handle_missing(prediction)

    def _predict(self) -> Optional[np.ndarray]:
        # Predict the next ball position from the current Kalman state.
        if not self.initialized:
            return None
        prediction = self.kalman.predict()
        return prediction[:2].reshape(-1)

    def _roi_size(self, prediction: Optional[np.ndarray]) -> int:
        # Bigger motion or more misses => search a larger ROI.
        velocity = 0.0
        if self.initialized:
            state = self.kalman.statePost.reshape(-1)
            velocity = float(np.hypot(state[2], state[3]))
        roi_size = int(
            self.config.ball_base_roi
            + (velocity * self.config.ball_velocity_roi_gain)
            + (self.miss_count * self.config.ball_miss_roi_gain)
        )
        return int(np.clip(roi_size, self.config.ball_base_roi, self.config.ball_max_roi))

    def _build_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        # Compare the current frame with the previous one.
        # This highlights moving regions, which helps reject static background objects.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros_like(gray)

        delta = cv2.absdiff(gray, self.previous_gray)
        _, motion_mask = cv2.threshold(delta, self.config.ball_motion_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 3)
        self.previous_gray = gray
        return motion_mask

    def _detect_ball(
        self,
        frame: np.ndarray,
        offset: tuple[int, int],
        conf: float,
        imgsz: int,
        source: str,
        motion_mask: np.ndarray,
        player_boxes: list[tuple[int, int, int, int]],
        prediction: Optional[np.ndarray],
    ) -> list[BallDetection]:
        # This function asks the trained YOLO ball model:
        # "Do you see the volleyball in this image?"
        #
        # The image can be:
        # - a small ROI around the predicted ball position
        # - the full frame if tracking is weak
        if frame.size == 0:
            return []

        # Run the trained ball detector on this image or ROI.
        result = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.config.ball_iou,
            imgsz=imgsz,
            verbose=False,
            device=self.device,
        )[0]

        detections: list[BallDetection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            # Convert YOLO's xyxy box into our internal (x, y, w, h) format.
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (
                int(round(x1 + offset[0])),
                int(round(y1 + offset[1])),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            )
            bbox = clip_bbox(bbox, (self.search_mask.shape[0], self.search_mask.shape[1], 3))
            if bbox[2] < self.config.ball_min_box_size or bbox[3] < self.config.ball_min_box_size:
                continue
            if bbox[2] > self.config.ball_max_box_size or bbox[3] > self.config.ball_max_box_size:
                continue

            center = bbox_center(bbox)
            if not point_in_mask(center, self.search_mask):
                # Ignore detections outside the allowed playable air region.
                continue

            x, y, w, h = bbox
            patch = frame[y - offset[1] : y - offset[1] + h, x - offset[0] : x - offset[0] + w]
            if patch.size == 0:
                continue
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            pixel_count = float(max(w * h, 1))

            # Check whether the patch has the colors we expect from the volleyball.
            yellow_ratio = cv2.countNonZero(
                cv2.inRange(hsv_patch, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
            ) / pixel_count
            blue_ratio = cv2.countNonZero(
                cv2.inRange(hsv_patch, self.config.ball_blue_lower, self.config.ball_blue_upper)
            ) / pixel_count
            motion_ratio = cv2.countNonZero(motion_mask[y : y + h, x : x + w]) / pixel_count
            inside_player = any(point_in_bbox(center, player_bbox) for player_bbox in player_boxes)

            if inside_player:
                # If a detection is inside a player box, be much stricter.
                # This prevents the tracker from snapping to yellow jerseys.
                if prediction is None:
                    continue
                predicted_center = (int(round(prediction[0])), int(round(prediction[1])))
                if distance_between(center, predicted_center) > self.config.ball_inside_player_gate:
                    continue
                if (
                    yellow_ratio < self.config.ball_color_ratio_min
                    or blue_ratio < self.config.ball_color_ratio_min
                    or motion_ratio < self.config.ball_motion_ratio_min
                ):
                    continue
            elif (
                yellow_ratio < self.config.ball_color_ratio_min
                and blue_ratio < self.config.ball_color_ratio_min
                and motion_ratio < self.config.ball_motion_ratio_min
            ):
                # If it does not look like the ball and it is barely moving, reject it.
                continue

            detections.append(
                BallDetection(
                    center=center,
                    bbox=bbox,
                    confidence=float(box.conf[0].item()),
                    source=source,
                    yellow_ratio=yellow_ratio,
                    blue_ratio=blue_ratio,
                    motion_ratio=motion_ratio,
                    inside_player=inside_player,
                )
            )
        return detections

    def _choose_detection(
        self,
        detections: list[BallDetection],
        prediction: Optional[np.ndarray],
    ) -> Optional[BallDetection]:
        # Multiple candidates may survive validation.
        # This function picks the best one.
        #
        # A candidate gets a better score when:
        # - YOLO confidence is high
        # - it is close to the predicted ball position
        # - it has useful yellow/blue ball colors
        # - it has visible motion
        #
        # A candidate gets penalized if it sits inside a player box.
        if not detections:
            return None

        # Score candidates and keep the one that best matches:
        # confidence + closeness to prediction + ball-like appearance + motion.
        predicted_center = (int(round(prediction[0])), int(round(prediction[1]))) if prediction is not None else None
        best_detection: Optional[BallDetection] = None
        best_score = float("-inf")

        for detection in detections:
            score = detection.confidence * 4.0
            if predicted_center is not None:
                distance = distance_between(detection.center, predicted_center)
                gate = self.config.ball_distance_gate if detection.source == "full" else self.config.ball_predicted_distance_gate
                if distance > gate:
                    continue
                score += max(0.0, 1.0 - (distance / max(gate, 1.0))) * 2.5
            elif detection.source == "roi":
                score += 0.4

            if detection.yellow_ratio >= self.config.ball_color_ratio_min:
                score += 0.35
            if detection.blue_ratio >= self.config.ball_color_ratio_min:
                score += 0.35
            if (
                detection.yellow_ratio >= self.config.ball_color_ratio_min
                and detection.blue_ratio >= self.config.ball_color_ratio_min
            ):
                score += self.config.ball_dual_color_bonus
            score += min(detection.motion_ratio / max(self.config.ball_motion_ratio_min * 4.0, 1e-6), 1.0) * self.config.ball_motion_bonus
            if detection.inside_player:
                score -= self.config.ball_inside_player_penalty

            if best_detection is None or score > best_score:
                best_detection = detection
                best_score = score

        return best_detection

    def _correct_with_detection(self, detection: BallDetection, frame_index: int) -> BallState:
        # Use the new measured ball position to correct the filter.
        measurement = np.array([[np.float32(detection.center[0])], [np.float32(detection.center[1])]])
        if not self.initialized:
            self.kalman.statePost = np.array(
                [[np.float32(detection.center[0])], [np.float32(detection.center[1])], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.initialized = True
        else:
            self.kalman.correct(measurement)

        if self.last_confirmed_center is not None:
            jump_distance = distance_between(detection.center, self.last_confirmed_center)
            if self.miss_count > 1 or jump_distance > (self.config.ball_predicted_distance_gate * 1.15):
                # If the ball was lost for a while, break the trail instead of drawing a fake line.
                self.trail.clear()

        self.last_confirmed_center = detection.center
        self.last_confirmed_frame = frame_index
        self.miss_count = 0

        # Only confirmed detections go into the visible trail.
        self.trail.append(detection.center)
        radius = max(float(max(detection.bbox[2], detection.bbox[3])) / 2.0, 4.0)
        return BallState(detection.center, radius, "confirmed")

    def _handle_missing(self, prediction: Optional[np.ndarray]) -> BallState:
        # Short misses are allowed: draw a predicted point briefly.
        # Long misses reset the tracker.
        self.miss_count += 1
        if prediction is None:
            self.trail.clear()
            return BallState(None, None, "missing")

        center = (int(round(prediction[0])), int(round(prediction[1])))
        uncertainty = float(np.trace(self.kalman.errorCovPre[:2, :2]))
        if (
            self.miss_count <= self.config.ball_predicted_draw_misses
            and uncertainty <= self.config.ball_max_uncertainty
            and point_in_mask(center, self.search_mask)
        ):
            return BallState(center, 6.0, "predicted")

        if self.miss_count > self.config.ball_max_misses or uncertainty > self.config.ball_max_uncertainty:
            self.initialized = False
            self.last_confirmed_center = None
            self.trail.clear()
        return BallState(None, None, "missing")


def draw_players(frame: np.ndarray, players: list[PlayerState]) -> None:
    # Draw one box per active player track.
    for player in players:
        x, y, w, h = player.bbox
        color = (0, 220, 255) if player.stable_team == "team_a" else (255, 180, 60)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)


def draw_ball(frame: np.ndarray, state: BallState) -> None:
    # Green = confirmed ball
    # Orange = short predicted bridge
    if state.center is None or state.radius is None:
        return

    color = (0, 235, 0) if state.status == "confirmed" else (0, 170, 255)
    cv2.circle(frame, state.center, max(5, int(round(state.radius))), color, 2, cv2.LINE_AA)
    cv2.circle(frame, state.center, 2, color, -1, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: deque[tuple[int, int]]) -> None:
    # Draw the recent path of confirmed ball positions.
    if len(trail) < 2:
        return
    points = list(trail)
    for index in range(1, len(points)):
        thickness = max(1, int(np.interp(index, [1, len(points) - 1], [5, 1])))
        cv2.line(frame, points[index - 1], points[index], (0, 220, 255), thickness, cv2.LINE_AA)


def overlay_status(frame: np.ndarray, team_a_count: int, team_b_count: int, fps_value: float) -> None:
    # Draw the scoreboard-like status box in the top-left corner.
    cv2.rectangle(frame, (20, 20), (250, 110), (24, 24, 24), -1)
    cv2.putText(frame, f"Team A: {team_a_count}", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Team B: {team_b_count}", (35, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 60), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_value:4.1f}", (35, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    phase_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Task 6 volleyball tracker with YOLO+ByteTrack players and YOLO+Kalman ball tracking.")
    parser.add_argument("--input", type=Path, default=phase_dir / "Volleyball.mp4", help="Input video path.")
    parser.add_argument("--output", type=Path, default=phase_dir / "Volleyball_annotated_v3.mp4", help="Output video path.")
    parser.add_argument("--display", action="store_true", help="Display frames during processing.")
    parser.add_argument("--player-model", type=str, default="yolov8s.pt", help="Pretrained player model.")
    parser.add_argument("--ball-model", type=Path, default=phase_dir / "models" / "ball_best.pt", help="Custom ball detector weights.")
    parser.add_argument("--device", type=str, default=pick_device(), help="Inference device.")
    return parser.parse_args()


def main() -> None:
    # This is the full end-to-end runtime pipeline.
    #
    # It does:
    # 1. open the video
    # 2. load the player model and trained ball model
    # 3. build masks for allowed regions
    # 4. process every frame
    # 5. write the annotated output video
    args = parse_args()
    config = TrackerConfig()

    if not args.ball_model.exists():
        raise FileNotFoundError(
            f"Custom ball model not found at {args.ball_model}. "
            "Run prepare_ball_dataset.py and train_ball_detector.py first."
        )

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
    # Build simple masks for:
    # 1. where the ball is allowed to be
    # 2. where top-side players can stand
    # 3. where bottom-side players can stand
    top_court_mask = apply_exclusions(
        build_mask((height, width), config.top_court_polygon),
        config.player_exclusion_rects,
    )
    bottom_court_mask = apply_exclusions(
        build_mask((height, width), config.bottom_court_polygon),
        config.player_exclusion_rects,
    )

    player_model = YOLO(args.player_model)
    ball_model = YOLO(str(args.ball_model))

    # Create the two main stateful systems:
    # - player/team tracker
    # - ball tracker
    team_tracker = TeamTracker(config, top_court_mask, bottom_court_mask)
    ball_tracker = BallTracker(ball_model, config, ball_search_mask, args.device)

    fps_estimate = fps
    last_tick = time.perf_counter()
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Player pipeline:
            # detect people -> track IDs -> classify team -> count active players
            player_result = player_model.track(
                source=frame,
                classes=[0],
                conf=config.player_confidence,
                iou=config.player_iou,
                imgsz=config.player_imgsz,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                device=args.device,
            )[0]

            active_players, team_a_count, team_b_count = team_tracker.update(frame, player_result, frame_index)
            player_boxes = [player.bbox for player in active_players]

            # Ball pipeline:
            # predict -> detect in ROI/full frame -> validate -> correct or mark missing
            ball_state = ball_tracker.update(frame, frame_index, player_boxes)

            # Draw everything onto the output frame.
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
                cv2.imshow("Volleyball Tracking V3", annotated)
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
