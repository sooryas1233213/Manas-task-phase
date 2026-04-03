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
    trail_length: int = 28
    max_players_per_team: int = 6
    player_track_ttl: int = 22
    player_active_ttl: int = 14
    player_vote_gain: float = 1.0
    player_vote_decay: float = 0.08
    player_stable_threshold: float = 2.2
    team_color_floor_ratio: float = 0.012
    team_color_margin: float = 0.004
    torso_x_margin_ratio: float = 0.28
    torso_y_start_ratio: float = 0.14
    torso_y_end_ratio: float = 0.45
    top_player_area_min: int = 900
    top_player_area_max: int = 42000
    bottom_player_area_min: int = 1800
    bottom_player_area_max: int = 65000
    player_min_height_top: int = 38
    player_min_height_bottom: int = 58
    player_min_width: int = 14
    player_max_width: int = 170
    player_max_aspect_ratio: float = 1.25
    player_min_fill_ratio: float = 0.22
    player_match_distance: float = 85.0
    player_iou_match_threshold: float = 0.06
    player_bg_history: int = 240
    player_bg_var_threshold: int = 28
    player_bg_detect_shadows: bool = False
    player_open_kernel: tuple[int, int] = (3, 3)
    player_close_kernel: tuple[int, int] = (7, 7)
    player_dilate_kernel: tuple[int, int] = (5, 9)
    player_morph_iterations: int = 2
    player_warmup_frames: int = 45
    player_color_open_kernel: tuple[int, int] = (3, 3)
    player_color_close_kernel: tuple[int, int] = (9, 9)
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
    ball_motion_ratio_min: float = 0.025
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
    ball_distance_gate: float = 195.0
    ball_predicted_distance_gate: float = 115.0
    ball_max_misses: int = 7
    ball_predicted_draw_misses: int = 3
    ball_max_uncertainty: float = 2200.0
    ball_base_roi: int = 190
    ball_velocity_roi_gain: float = 2.4
    ball_miss_roi_gain: float = 34.0
    ball_max_roi: int = 420
    ball_optical_flow_max_frames: int = 2
    ball_optical_flow_win_size: tuple[int, int] = (21, 21)
    ball_optical_flow_max_level: int = 2
    ball_optical_flow_quality: float = 0.01
    ball_optical_flow_min_distance: int = 2
    ball_optical_flow_block_size: int = 5
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


class PlayerTrackerOpenCV:
    def __init__(self, config: SceneConfig, top_mask: np.ndarray, bottom_mask: np.ndarray) -> None:
        self.config = config
        self.top_mask = top_mask
        self.bottom_mask = bottom_mask
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.player_bg_history,
            varThreshold=config.player_bg_var_threshold,
            detectShadows=config.player_bg_detect_shadows,
        )
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_open_kernel)
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_close_kernel)
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.player_dilate_kernel)
        self.color_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_color_open_kernel)
        self.color_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.player_color_close_kernel)
        self.tracks: dict[int, PlayerTrack] = {}
        self.next_track_id = 1
        self.last_output_counts: Optional[tuple[int, int]] = None

    def update(self, frame: np.ndarray, frame_index: int) -> tuple[list[PlayerTrack], int, int]:
        fg_mask = self.bg_subtractor.apply(frame)
        if frame_index < self.config.player_warmup_frames:
            self._age_tracks(frame_index, [])
            return self.active_tracks(frame_index)

        _, fg_mask = cv2.threshold(fg_mask, 210, 255, cv2.THRESH_BINARY)
        court_mask = cv2.bitwise_or(self.top_mask, self.bottom_mask)
        fg_mask = cv2.bitwise_and(fg_mask, court_mask)
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            self.open_kernel,
            iterations=self.config.player_morph_iterations,
        )
        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            self.close_kernel,
            iterations=self.config.player_morph_iterations,
        )
        fg_mask = cv2.dilate(fg_mask, self.dilate_kernel, iterations=1)

        motion_blobs = self._extract_player_blobs(fg_mask, frame.shape)
        color_blobs = self._extract_color_blobs(frame)
        blobs = self._merge_blobs(motion_blobs + color_blobs)
        self._associate(blobs, frame, frame_index)
        active, track_team_a_count, track_team_b_count = self.active_tracks(frame_index)
        color_team_a_count = min(sum(1 for blob in color_blobs if blob.side == "top"), self.config.max_players_per_team)
        color_team_b_count = min(sum(1 for blob in color_blobs if blob.side == "bottom"), self.config.max_players_per_team)
        team_a_count, team_b_count = self._stabilize_counts(
            frame_index,
            track_team_a_count,
            track_team_b_count,
            color_team_a_count,
            color_team_b_count,
        )
        return active, team_a_count, team_b_count

    def _extract_player_blobs(self, fg_mask: np.ndarray, frame_shape: tuple[int, int, int]) -> list[PlayerBlob]:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs: list[PlayerBlob] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = clip_bbox((x, y, w, h), frame_shape)
            x, y, w, h = bbox
            if w < self.config.player_min_width or w > self.config.player_max_width:
                continue
            if h <= 0 or h / max(float(w), 1.0) < 1.1:
                continue
            if w / max(float(h), 1.0) > self.config.player_max_aspect_ratio:
                continue

            footpoint = bbox_footpoint(bbox)
            if point_in_mask(footpoint, self.top_mask):
                side = "top"
                min_height = self.config.player_min_height_top
                area_min = self.config.top_player_area_min
                area_max = self.config.top_player_area_max
            elif point_in_mask(footpoint, self.bottom_mask):
                side = "bottom"
                min_height = self.config.player_min_height_bottom
                area_min = self.config.bottom_player_area_min
                area_max = self.config.bottom_player_area_max
            else:
                continue

            if h < min_height or area < area_min or area > area_max:
                continue

            fill_ratio = area / max(float(w * h), 1.0)
            if fill_ratio < self.config.player_min_fill_ratio:
                continue

            blobs.append(
                PlayerBlob(
                    bbox=bbox,
                    center=bbox_center(bbox),
                    footpoint=footpoint,
                    side=side,
                )
            )
        return blobs

    def _extract_color_blobs(self, frame: np.ndarray) -> list[PlayerBlob]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        jersey_mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.config.yellow_player_lower, self.config.yellow_player_upper),
            cv2.inRange(hsv, self.config.blue_player_lower, self.config.blue_player_upper),
        )
        white_mask = cv2.inRange(hsv, self.config.white_player_lower, self.config.white_player_upper)
        color_mask = cv2.bitwise_or(jersey_mask, white_mask)
        court_mask = cv2.bitwise_or(self.top_mask, self.bottom_mask)
        color_mask = cv2.bitwise_and(color_mask, court_mask)
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_OPEN,
            self.color_open_kernel,
            iterations=1,
        )
        color_mask = cv2.morphologyEx(
            color_mask,
            cv2.MORPH_CLOSE,
            self.color_close_kernel,
            iterations=2,
        )
        return self._extract_player_blobs(color_mask, frame.shape)

    def _merge_blobs(self, blobs: list[PlayerBlob]) -> list[PlayerBlob]:
        merged: list[PlayerBlob] = []
        for blob in blobs:
            replaced = False
            for index, existing in enumerate(merged):
                if existing.side != blob.side:
                    continue
                if bbox_iou(existing.bbox, blob.bbox) > 0.45 or distance_between(existing.center, blob.center) < 28.0:
                    existing_area = existing.bbox[2] * existing.bbox[3]
                    blob_area = blob.bbox[2] * blob.bbox[3]
                    if blob_area > existing_area:
                        merged[index] = blob
                    replaced = True
                    break
            if not replaced:
                merged.append(blob)
        return merged

    def _associate(self, blobs: list[PlayerBlob], frame: np.ndarray, frame_index: int) -> None:
        track_predictions: dict[int, tuple[int, int]] = {}
        for track_id, track in self.tracks.items():
            prediction = track.kalman.predict()
            track_predictions[track_id] = (int(round(prediction[0][0])), int(round(prediction[1][0])))

        unmatched_tracks = set(self.tracks.keys())
        used_blobs: set[int] = set()

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
                    pred_center = track_predictions.get(track_id, track.center)
                    center_distance = distance_between(blob.center, pred_center)
                    foot_distance = distance_between(blob.footpoint, track.footpoint)
                    iou = bbox_iou(blob.bbox, track.bbox)
                    if center_distance > self.config.player_match_distance and iou < self.config.player_iou_match_threshold:
                        continue
                    score = center_distance + (foot_distance * 0.35) - (iou * 120.0)
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
            )
            self._update_votes(track, frame)
            self.tracks[track.track_id] = track
            self.next_track_id += 1

        self._age_tracks(frame_index, list(unmatched_tracks))

    def _update_track(self, track: PlayerTrack, blob: PlayerBlob, frame: np.ndarray, frame_index: int) -> None:
        measurement = np.array([[np.float32(blob.center[0])], [np.float32(blob.center[1])]])
        track.kalman.correct(measurement)
        track.bbox = blob.bbox
        track.center = blob.center
        track.footpoint = blob.footpoint
        track.side = blob.side
        track.last_seen = frame_index
        track.misses = 0
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

    def _age_tracks(self, frame_index: int, unmatched_tracks: list[int]) -> None:
        for track_id in unmatched_tracks:
            if track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            track.misses += 1

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if frame_index - track.last_seen > self.config.player_track_ttl
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def active_tracks(self, frame_index: int) -> tuple[list[PlayerTrack], int, int]:
        active = [
            track
            for track in self.tracks.values()
            if frame_index - track.last_seen <= self.config.player_active_ttl and track.stable_team is not None
        ]
        team_a_count = min(sum(1 for track in active if track.stable_team == "team_a"), self.config.max_players_per_team)
        team_b_count = min(sum(1 for track in active if track.stable_team == "team_b"), self.config.max_players_per_team)
        return active, team_a_count, team_b_count

    def _stabilize_counts(
        self,
        frame_index: int,
        track_team_a_count: int,
        track_team_b_count: int,
        color_team_a_count: int,
        color_team_b_count: int,
    ) -> tuple[int, int]:
        if self.last_output_counts is None and frame_index >= self.config.player_warmup_frames:
            self.last_output_counts = (6, 6)

        team_a_candidate = min(max(track_team_a_count, min(color_team_a_count + 1, 6)), 6)
        team_b_candidate = min(max(track_team_b_count, min(color_team_b_count + 1, 6)), 6)

        if self.last_output_counts is None:
            return team_a_candidate, team_b_candidate

        last_a, last_b = self.last_output_counts
        if team_a_candidate <= max(2, last_a - 3):
            team_a_candidate = last_a
        if team_b_candidate <= max(2, last_b - 3):
            team_b_candidate = last_b

        team_a_candidate = int(np.clip(team_a_candidate, 0, self.config.max_players_per_team))
        team_b_candidate = int(np.clip(team_b_candidate, 0, self.config.max_players_per_team))
        self.last_output_counts = (team_a_candidate, team_b_candidate)
        return self.last_output_counts


class BallTrackerOpenCV:
    def __init__(self, config: SceneConfig, search_mask: np.ndarray) -> None:
        self.config = config
        self.search_mask = search_mask
        self.kalman = self._build_kalman()
        self.initialized = False
        self.last_confirmed_center: Optional[tuple[int, int]] = None
        self.last_confirmed_radius: float = 0.0
        self.miss_count = 0
        self.trail: deque[tuple[int, int]] = deque(maxlen=config.trail_length)
        self.previous_gray: Optional[np.ndarray] = None
        self.previous_ball_gray: Optional[np.ndarray] = None
        self.previous_ball_points: Optional[np.ndarray] = None

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

    def update(self, frame: np.ndarray, player_boxes: list[tuple[int, int, int, int]]) -> BallState:
        prediction = self._predict()
        predicted_center = None if prediction is None else (int(round(prediction[0])), int(round(prediction[1])))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        motion_mask, gray = self._build_motion_mask(frame)

        candidates = self._find_candidates(frame, hsv, motion_mask, player_boxes, predicted_center)
        candidate = self._choose_candidate(candidates, predicted_center)
        if candidate is not None:
            return self._confirm_candidate(frame, gray, candidate)

        flow_state = self._optical_flow_bridge(gray, predicted_center)
        if flow_state is not None:
            return flow_state

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
        yellow_mask = cv2.inRange(hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
        blue_mask = cv2.inRange(hsv, self.config.ball_blue_lower, self.config.ball_blue_upper)
        combined_mask = cv2.bitwise_or(yellow_mask, blue_mask)
        combined_mask = cv2.bitwise_and(combined_mask, self.search_mask)
        motion_color_mask = cv2.bitwise_and(combined_mask, motion_mask)

        candidates = self._candidates_from_mask(
            motion_color_mask,
            frame,
            hsv,
            motion_mask,
            player_boxes,
            "motion",
        )

        if predicted_center is not None and point_in_mask(predicted_center, self.search_mask):
            roi_size = self._roi_size()
            roi_frame, offset = crop_roi(frame, predicted_center, roi_size)
            if roi_frame.size > 0:
                roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
                roi_motion = motion_mask[offset[1] : offset[1] + roi_frame.shape[0], offset[0] : offset[0] + roi_frame.shape[1]]
                roi_search = self.search_mask[offset[1] : offset[1] + roi_frame.shape[0], offset[0] : offset[0] + roi_frame.shape[1]]
                roi_yellow = cv2.inRange(roi_hsv, self.config.ball_yellow_lower, self.config.ball_yellow_upper)
                roi_blue = cv2.inRange(roi_hsv, self.config.ball_blue_lower, self.config.ball_blue_upper)
                roi_mask = cv2.bitwise_and(cv2.bitwise_or(roi_yellow, roi_blue), roi_search)
                roi_mask = cv2.bitwise_or(roi_mask, cv2.bitwise_and(roi_mask, roi_motion))
                candidates.extend(
                    self._candidates_from_mask(
                        roi_mask,
                        roi_frame,
                        roi_hsv,
                        roi_motion,
                        player_boxes,
                        "roi",
                        offset,
                    )
                )
        return candidates

    def _candidates_from_mask(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
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
            inside_player = any(
                bbox_iou(bbox, player_bbox) > 0.1 or (
                    player_bbox[0] <= center[0] <= player_bbox[0] + player_bbox[2]
                    and player_bbox[1] <= center[1] <= player_bbox[1] + player_bbox[3]
                )
                for player_bbox in player_boxes
            )

            if (
                yellow_ratio < self.config.ball_color_ratio_min
                and blue_ratio < self.config.ball_color_ratio_min
            ):
                continue
            if motion_ratio < self.config.ball_motion_ratio_min and source == "motion":
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
                )
            )
        return candidates

    def _choose_candidate(
        self,
        candidates: list[BallCandidate],
        predicted_center: Optional[tuple[int, int]],
    ) -> Optional[BallCandidate]:
        best_candidate: Optional[BallCandidate] = None
        best_score = float("-inf")

        for candidate in candidates:
            score = candidate.circularity * 2.8
            score += min(candidate.motion_ratio / max(self.config.ball_motion_ratio_min * 3.0, 1e-6), 1.0) * self.config.ball_motion_bonus
            if candidate.yellow_ratio >= self.config.ball_color_ratio_min:
                score += 0.4
            if candidate.blue_ratio >= self.config.ball_color_ratio_min:
                score += 0.4
            if (
                candidate.yellow_ratio >= self.config.ball_color_ratio_min
                and candidate.blue_ratio >= self.config.ball_color_ratio_min
            ):
                score += self.config.ball_dual_color_bonus
            if candidate.source == "roi":
                score += 0.45

            if predicted_center is not None:
                distance = distance_between(candidate.center, predicted_center)
                gate = self.config.ball_predicted_distance_gate if candidate.source == "roi" else self.config.ball_distance_gate
                if distance > gate:
                    continue
                score += max(0.0, 1.0 - (distance / max(gate, 1.0))) * 2.4
                if candidate.inside_player and distance > self.config.ball_inside_player_gate:
                    continue
            if candidate.inside_player:
                score -= self.config.ball_inside_player_penalty

            if best_candidate is None or score > best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def _confirm_candidate(self, frame: np.ndarray, gray: np.ndarray, candidate: BallCandidate) -> BallState:
        measurement = np.array([[np.float32(candidate.center[0])], [np.float32(candidate.center[1])]])
        if not self.initialized:
            self.kalman.statePost = np.array(
                [[np.float32(candidate.center[0])], [np.float32(candidate.center[1])], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.initialized = True
        else:
            self.kalman.correct(measurement)

        if self.last_confirmed_center is not None:
            jump_distance = distance_between(candidate.center, self.last_confirmed_center)
            if self.miss_count > 1 or jump_distance > (self.config.ball_predicted_distance_gate * 1.15):
                self.trail.clear()

        self.last_confirmed_center = candidate.center
        self.last_confirmed_radius = candidate.radius
        self.miss_count = 0
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
            maxCorners=6,
            qualityLevel=self.config.ball_optical_flow_quality,
            minDistance=self.config.ball_optical_flow_min_distance,
            blockSize=self.config.ball_optical_flow_block_size,
        )
        if points is None:
            self.previous_ball_gray = patch.copy()
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
            or self.miss_count >= self.config.ball_optical_flow_max_frames
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
        if predicted_center is not None and distance_between(center, predicted_center) > self.config.ball_predicted_distance_gate:
            return None

        self.previous_ball_gray = gray.copy()
        self.previous_ball_points = valid.reshape(-1, 1, 2).astype(np.float32)
        return BallState(center, max(4.0, self.last_confirmed_radius), "predicted")

    def _handle_missing(self, prediction: Optional[np.ndarray]) -> BallState:
        self.miss_count += 1
        if prediction is None:
            self.trail.clear()
            self.previous_ball_points = None
            self.previous_ball_gray = None
            return BallState(None, None, "missing")

        center = (int(round(prediction[0])), int(round(prediction[1])))
        uncertainty = float(np.trace(self.kalman.errorCovPre[:2, :2]))
        if (
            self.miss_count <= self.config.ball_predicted_draw_misses
            and uncertainty <= self.config.ball_max_uncertainty
            and point_in_mask(center, self.search_mask)
        ):
            return BallState(center, max(4.0, self.last_confirmed_radius), "predicted")

        if self.miss_count > self.config.ball_max_misses or uncertainty > self.config.ball_max_uncertainty:
            self.initialized = False
            self.last_confirmed_center = None
            self.previous_ball_points = None
            self.previous_ball_gray = None
            self.trail.clear()
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
    parser = argparse.ArgumentParser(description="OpenCV-first volleyball tracker for Task 6.")
    parser.add_argument("--input", type=Path, default=phase6_dir / "Volleyball.mp4", help="Input video path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Volleyball_annotated_opencv.mp4",
        help="Output video path.",
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

    player_tracker = PlayerTrackerOpenCV(config, top_court_mask, bottom_court_mask)
    ball_tracker = BallTrackerOpenCV(config, ball_search_mask)

    fps_estimate = fps
    last_tick = time.perf_counter()
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            active_players, team_a_count, team_b_count = player_tracker.update(frame, frame_index)
            player_boxes = [player.bbox for player in active_players]
            ball_state = ball_tracker.update(frame, player_boxes)

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
                cv2.imshow("Volleyball Tracking OpenCV", annotated)
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
