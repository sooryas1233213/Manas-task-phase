"""
Volleyball ball tracker v6 (trajectory-first, OpenCV + optional ML recovery).

Pipeline (high level):
  1. v5's ObservationCollectorV5 scans the video and produces per-frame ball candidates
     (motion / color / MOG) plus optional global stabilization transforms.
  2. v5's build_player_states yields torso suppression boxes so ball blobs on players
     can be penalized or rejected.
  3. This module refines candidates, runs a Kalman filter + state machine (TRACK /
     DEGRADED / RECOVER / LOST), optionally reranks recovery seeds with a RandomForest
     trained from YOLO-style labels, then interpolates gaps and prunes spike outliers.
  4. A second pass draws players, ball, and overlay onto the output video.

Important: Per-frame detection logic lives in volleyball_tracker_opencv_v5.py; v6 focuses
on sieving, temporal support, tracking, recovery, and post-processing.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Reuse v5 data types, observation collection, drawing, geometry, and the v5 Kalman
# template (v6 overrides build_ball_kalman below for its own 4-state model).
from volleyball_tracker_opencv_v5 import (
    BallCandidate,
    BallDebugInfo as BaseBallDebugInfo,
    BallFrameState,
    BallTrajectoryTrackerV5,
    FrameObservation,
    ObservationCollectorV5,
    PlayerFrameState,
    RecoveryEvent,
    SceneConfigV5 as BaseSceneConfigV5,
    TrajectoryPoint,
    apply_transform,
    bbox_center,
    bbox_iou,
    build_ball_kalman,
    build_player_states,
    clip_bbox,
    distance_between,
    draw_ball,
    draw_overlay,
    draw_players,
    expected_ball_radius,
    kalman_predicted_center,
    point_in_mask,
    trajectory_fit_residual,
)

# polyfit can emit RankWarning on nearly degenerate fits; safe to ignore for interpolation.
warnings.filterwarnings("ignore", category=np.RankWarning)


@dataclass
class SceneConfigV5(BaseSceneConfigV5):
    """Tuning knobs for v6 ball pipeline (extends v5 scene config with ball-specific limits)."""

    # MOG / candidate generation (passed through v5 where applicable).
    ball_mog_history: int = 500
    # After sieving, keep at most this many candidates per frame (speed + stability).
    ball_candidate_keep_per_frame: int = 8
    # Expected ball area from radius profile must fall in [min,max] * expected_area.
    ball_area_scale_min: float = 0.25
    ball_area_scale_max: float = 6.0
    ball_max_aspect_ratio: float = 2.0
    ball_min_compactness: float = 0.25
    ball_min_circularity: float = 0.04
    # Temporal seeding: neighbor displacement must be plausible for dt (pixels * dt).
    ball_seed_min_progress: float = 7.0
    ball_seed_max_progress: float = 140.0
    ball_track_search_gap: int = 2
    # Max distance from Kalman prediction to accept a measurement (TRACK mode).
    ball_max_gate_per_step: float = 92.0
    ball_segment_min_confirmed: int = 4
    ball_segment_merge_gap: int = 5
    # Interpolation: medium gaps use polynomial fit if gap <= this and residuals OK.
    ball_interp_medium_gap: int = 7
    # Reject candidate if polynomial trajectory residual is huge (likely wrong blob).
    ball_poly_outlier_limit: float = 24.0
    # Debug: flag "long gap" segments for CSV / contact sheet when gap >= this many frames.
    ball_ml_gap_trigger: int = 9
    # After a miss, coast on Kalman prediction for this many frames before RECOVER.
    ball_coast_frames: int = 2
    # DEGRADED mode widens the acceptance gate by this factor vs TRACK.
    ball_degraded_gate_scale: float = 1.7
    # Present for parity with later variants; TRACK/DEGRADED use gate_scale args instead.
    ball_recover_gate_scale: float = 2.5
    # Large innovation (px) between prediction and chosen blob → treat as possible touch/spike.
    ball_touch_innovation_threshold: float = 34.0
    # How much RandomForest P(ball) adds to candidate / anchor scores during recovery.
    ball_rf_probability_weight: float = 12.0
    # In RECOVER, ignore RF-weak blobs unless they have temporal support.
    ball_rf_min_probability: float = 0.42
    # When re-seeding, still consider blobs within this radius of last prediction.
    ball_recover_fullframe_gate: float = 180.0
    # Spike / outlier detection: jumps larger than this (px) trigger pruning or skip interp.
    ball_large_jump_threshold: float = 130.0
    # Training: cap negatives at positive_count * ratio to balance the RF dataset.
    ball_rf_negative_ratio: int = 6


@dataclass
class BallDebugInfo(BaseBallDebugInfo):
    """Debug metadata beyond v5: frames where rendered position jumps suspiciously."""

    large_jump_frames: list[int] = field(default_factory=list)


def kalman_measurement_distance(kalman: cv2.KalmanFilter, center: tuple[int, int]) -> float:
    """
    Mahalanobis distance from a pixel measurement to the Kalman predicted measurement.

    Uses innovation covariance S = H P^- H^T + R so a candidate far from prediction in
    a direction the filter is already uncertain about is penalized less than a direction
    with low predicted variance. Falls back to Euclidean residual if S is singular.
    """
    measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]], dtype=np.float32)
    predicted_measurement = kalman.measurementMatrix @ kalman.statePre
    residual = measurement - predicted_measurement
    innovation_cov = (
        kalman.measurementMatrix @ kalman.errorCovPre @ kalman.measurementMatrix.T
    ) + kalman.measurementNoiseCov
    try:
        inv_cov = np.linalg.inv(innovation_cov)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(residual))
    distance = float(np.sqrt((residual.T @ inv_cov @ residual)[0, 0]))
    return distance


def build_ball_kalman(
    start_center: tuple[int, int],
    next_center: tuple[int, int],
    delta_frames: int = 1,
) -> cv2.KalmanFilter:
    """
    4-state constant-velocity Kalman: state = [x, y, vx, vy], measurement = [x, y].

    Seeded with two positions delta_frames apart so initial velocity matches the
    anchor pair from seeding / recovery. v6 overrides v5's build_ball_kalman import.
    """
    dt = float(max(delta_frames, 1))
    kalman = cv2.KalmanFilter(4, 2)
    # x' = x + vx*dt, y' = y + vy*dt, velocities unchanged (constant-velocity model).
    kalman.transitionMatrix = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.processNoiseCov = np.diag([0.2, 0.2, 0.9, 0.9]).astype(np.float32)
    kalman.measurementNoiseCov = np.diag([16.0, 16.0]).astype(np.float32)
    kalman.errorCovPost = np.eye(4, dtype=np.float32) * 24.0  # loose initial uncertainty
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


def load_ball_label_map(
    dataset_root: Path,
    frame_shape: tuple[int, int, int],
) -> dict[int, tuple[tuple[int, int], tuple[int, int, int, int], float]]:
    """
    Load labels (class cx cy w h normalized) from dataset_root/labels/*/*.txt.

    Filename stem must end with _<frame_index> to map to video frame. Returns per-frame
    ground-truth center, bbox, and radius for RF training and for building radius profile.
    """
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
    """
    Expected ball radius as a function of image row (perspective: ball looks smaller high).

    Bins labeled ball radii by vertical position, takes median per bin, interpolates
    across the full height. If too few labels, falls back to v5 expected_ball_radius().
    """
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
    """
    Optional RandomForest reranker used only during RECOVER / anchor selection.

    Trains on per-candidate feature rows: positives are blobs near GT ball in labeled
    frames; negatives are other candidates. Cached to joblib with FEATURE_VERSION so
    feature schema changes invalidate the cache. When disabled or data missing, recovery
    falls back to motion_seed scoring only.
    """

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
        """Load cached RF if valid; else train from labels + candidates and save artifact."""
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
            except Exception as exc:
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
        """Return candidate_id -> P(class=ball) for the given frame; empty if no model."""
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
        # Normalized position, size vs expected disk area, shape/color cues, detector flags,
        # temporal support, and torso overlap — mirrors what v6 already uses heuristically.
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
        # One positive per labeled frame (closest blob to GT); all other candidates negative.
        # Subsample negatives to avoid swamping the forest (ball_rf_negative_ratio).
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
        except Exception as exc:
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


class BallTrajectoryTrackerV6(BallTrajectoryTrackerV5):
    """
    v6 ball track builder: tighter candidate sieve, temporal support, Kalman + FSM,
    optional RF recovery, gap interpolation, and jump pruning.

    Extends v5's tracker base for shared helpers (_point_from_candidate, masks, etc.)
    but replaces the core per-frame loop with _track_frames and v6-specific scoring.
    """

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
        # Order matters: candidates first (RF may train from them), then track, smooth, debug.
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
        """Radius prior at row y from label-derived profile, else v5 heuristic."""
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
        """
        Per frame: base geometric/color sieve, rank by local_quality, keep top K,
        then temporal_support across neighbors, then drop weak banner-region false positives.
        """
        filtered: list[list[BallCandidate]] = []
        for frame_index, observation in enumerate(self.observations):
            suppression_boxes = self.player_states[frame_index].suppression_boxes if frame_index < len(self.player_states) else []
            frame_candidates = [candidate for candidate in observation.ball_candidates if self._passes_base_sieve(candidate, suppression_boxes)]
            for candidate in frame_candidates:
                candidate.local_quality = self._local_quality_v6(candidate)
            frame_candidates.sort(key=lambda item: item.local_quality, reverse=True)
            filtered.append(frame_candidates[: self.config.ball_candidate_keep_per_frame])
        self.candidates_by_frame = filtered

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            for candidate in frame_candidates:
                candidate.support_count, candidate.support_progress = self._temporal_support_v6(frame_index, candidate)

        for frame_index, frame_candidates in enumerate(self.candidates_by_frame):
            self.candidates_by_frame[frame_index] = [
                candidate
                for candidate in frame_candidates
                if not (
                    candidate.center[1] < self.config.ball_top_banner_guard_y
                    and candidate.support_count <= 0
                    and not candidate.source_median
                )
            ][: self.config.ball_candidate_keep_per_frame]

    def _passes_base_sieve(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        # Area vs expected disk, aspect ratio, compactness/circularity; large blobs on
        # torso (suppression) rejected unless small enough to still be the ball.
        expected_radius = self._expected_radius(candidate.center[1])
        expected_area = math.pi * expected_radius * expected_radius
        ratio = candidate.aspect_ratio
        if candidate.area < max(3.0, expected_area * self.config.ball_area_scale_min):
            return False
        if candidate.area > max(26.0, expected_area * self.config.ball_area_scale_max):
            return False
        if ratio > self.config.ball_max_aspect_ratio or (1.0 / max(ratio, 1e-6)) > self.config.ball_max_aspect_ratio:
            return False
        if candidate.compactness < self.config.ball_min_compactness or candidate.circularity < self.config.ball_min_circularity:
            return False
        if self._candidate_inside_suppression(candidate, suppression_boxes) and candidate.radius > expected_radius * 0.85:
            return False
        return True

    def _local_quality_v6(self, candidate: BallCandidate) -> float:
        """Heuristic blob goodness for ranking (not the same as Kalman association score)."""
        expected_radius = self._expected_radius(candidate.center[1])
        score = (
            candidate.circularity * 4.2
            + candidate.compactness * 3.0
            + max(0.0, 2.4 - abs(candidate.radius - expected_radius))
            + candidate.weak_yellow_ratio * 1.2
        )
        if candidate.source_median:
            score += 1.6
        if candidate.source_mog:
            score += 1.0
        if candidate.source_median and candidate.source_mog:
            score += 1.8
        return score

    def _temporal_support_v6(self, frame_index: int, candidate: BallCandidate) -> tuple[int, float]:
        """
        Count nearby frames (±gap) with a consistent neighbor blob (distance/radius);
        support_count and support_progress feed scoring and RF gating.
        """
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
        """
        Main temporal loop after bg_warmup_frames.

        Modes:
          SEARCH_INIT / LOST — find two-frame anchor, init Kalman, jump to TRACK.
          TRACK — strict association; on failure coast once then DEGRADED.
          DEGRADED — wider gate + relaxed shape rules; coast up to ball_coast_frames else RECOVER.
          RECOVER — RF + motion anchor or keep coasting; too many misses → LOST (Kalman cleared).

        touch_boost_frames: after a large innovation "touch", temporarily raise process noise
        so the filter can re-accelerate faster on the next frames.
        """
        states = [BallFrameState() for _ in range(len(self.observations))]
        confirmed_points: list[TrajectoryPoint] = []
        kalman: Optional[cv2.KalmanFilter] = None
        mode = "SEARCH_INIT"
        misses = 0
        last_radius = self.config.ball_top_radius
        touch_boost_frames = 0

        for frame_index in range(self.config.bg_warmup_frames, len(self.observations)):
            predicted_center = None
            if kalman is not None:
                self._set_process_noise(kalman, touch_boost_frames > 0)
                predicted_center = kalman_predicted_center(kalman.predict())
                if touch_boost_frames > 0:
                    touch_boost_frames -= 1

            if mode in {"SEARCH_INIT", "LOST"}:
                # Cold start: no Kalman yet; anchor pairs a blob with a future consistent blob.
                anchor = self._find_anchor_v6(frame_index, predicted_center, recover=False)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    states[frame_index] = BallFrameState(point.center, point.radius, "confirmed", point.confidence, source)
                    confirmed_points.append(point)
                    last_radius = candidate.radius
                    misses = 0
                    mode = "TRACK"
                continue

            if mode == "TRACK":
                # Normal operation: tight gating and scoring; single miss → coast + DEGRADED.
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
                # Try to re-lock with enlarged gate and looser blob shape thresholds.
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
                # RF scores candidates here; successful anchor reinitializes Kalman from pair.
                anchor = self._find_anchor_v6(frame_index, predicted_center, recover=True)
                if anchor is not None:
                    candidate, partner_frame, partner_candidate, source = anchor
                    kalman = build_ball_kalman(candidate.center, partner_candidate.center, max(partner_frame - frame_index, 1))
                    point = super()._point_from_candidate(frame_index, candidate, "confirmed")
                    states[frame_index] = BallFrameState(point.center, point.radius, "confirmed", point.confidence, source)
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
        """Higher Q after touch: allow faster velocity changes (spike / dig) on next steps."""
        diag = [0.8, 0.8, 3.2, 3.2] if boosted else [0.2, 0.2, 0.9, 0.9]
        kalman.processNoiseCov = np.diag(diag).astype(np.float32)

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
        """
        Pick best candidate inside gate, kalman.correct, append trajectory.

        Returns (success, touch_event, candidate). touch_event is True when innovation
        (prediction vs chosen blob) is large — heuristic for player contact / sudden motion.
        """
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
        states[frame_index] = BallFrameState(point.center, point.radius, "confirmed", point.confidence, point.source)
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
        # TRACK/DEGRADED path: recovery_scores unused (None); reserved for symmetry with recovery.
        best_choice = None
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
        return None if best_choice is None or best_choice[1] < min_score else best_choice

    def _candidate_mode_allowed_v6(
        self,
        candidate: BallCandidate,
        suppression_boxes: list[tuple[int, int, int, int]],
        relaxed: bool,
    ) -> bool:
        # DEGRADED relaxes aspect ratio / compactness floors so motion-blurred blobs can match.
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
        if candidate.compactness < compactness_floor or candidate.circularity < circularity_floor:
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
        # Combine blob quality, Mahalanobis consistency, trajectory poly residual, velocity
        # continuity, size prior, torso penalty, detector bonuses, and RF prob (0 in TRACK).
        innovation = distance_between(predicted_center, candidate.center)
        mahalanobis = kalman_measurement_distance(kalman, candidate.center)
        candidate_point = super()._point_from_candidate(frame_index, candidate, "candidate")
        trajectory_penalty = trajectory_fit_residual(confirmed_points[-self.config.ball_poly_window :], candidate_point)
        # Huge poly residual but small Mahalanobis → likely wrong blob along smooth arc; reject.
        if trajectory_penalty > self.config.ball_poly_outlier_limit and mahalanobis < self.config.ball_touch_innovation_threshold:
            return None, innovation
        velocity_penalty = 0.0
        if len(confirmed_points) >= 2:
            last = confirmed_points[-1]
            previous = confirmed_points[-2]
            delta_frames = max(1, last.frame_index - previous.frame_index)
            expected_vx = (last.center[0] - previous.center[0]) / delta_frames
            expected_vy = (last.center[1] - previous.center[1]) / delta_frames
            velocity_penalty = abs(candidate.center[0] - last.center[0] - expected_vx) + abs(candidate.center[1] - last.center[1] - expected_vy)
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
            + candidate.support_count * 3.2
            + source_bonus
            + recovery_probability * self.config.ball_rf_probability_weight
            - mahalanobis * 2.4
            - velocity_penalty * 0.16
            - size_penalty
            - trajectory_penalty * 0.9
            - torso_penalty
        )
        return score, innovation

    def _find_anchor_v6(
        self,
        frame_index: int,
        predicted_center: Optional[tuple[int, int]],
        recover: bool,
    ) -> Optional[tuple[BallCandidate, int, BallCandidate, str]]:
        """
        Find (seed, future_frame, partner, source) with consistent forward motion.

        recover=True enables RF probabilities and stricter filtering of low-P blobs without
        temporal support; source becomes recover_rf when RF is confident enough.
        """
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
        best_choice = None
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
                + candidate.support_count * 4.5
                + partner_candidate.support_count * 2.0
                + rf_probability * self.config.ball_rf_probability_weight
                - torso_penalty
                - distance_penalty
            )
            if score > best_score:
                best_score = score
                source = "recover_rf" if recover and rf_probability >= self.config.ball_rf_min_probability else "motion_seed"
                best_choice = (candidate, partner_frame, partner_candidate, source)
        return None if best_choice is None or best_score < 18.0 else best_choice

    def _find_confirmation_partner_v6(
        self,
        frame_index: int,
        seed: BallCandidate,
    ) -> Optional[tuple[int, BallCandidate]]:
        """Earliest future frame within seed distance band with best local score."""
        best_choice = None
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
                score = candidate.local_quality * 5.5 + candidate.support_count * 3.0 - distance * 0.08
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
        # Kalman-only placeholder: low confidence; skip if OOB mask or inside torso.
        if predicted_center is None or not point_in_mask(predicted_center, self.ball_mask) or self._center_hits_torso(frame_index, predicted_center):
            return
        states[frame_index] = BallFrameState(predicted_center, radius, "coast", 0.18, "kalman")

    def _recovery_roi_v6(self, predicted_center: Optional[tuple[int, int]]) -> tuple[int, int, int, int]:
        """Debug ROI around prediction (uses ball_ml_roi_margin from v5 config)."""
        if predicted_center is None:
            return (0, 0, self.frame_shape[1], self.frame_shape[0])
        half = self.config.ball_ml_roi_margin
        return clip_bbox((predicted_center[0] - half, predicted_center[1] - half, half * 2, half * 2), self.frame_shape)

    def _interpolate_states(self, states: list[BallFrameState]) -> None:
        """
        Fill gaps between confirmed points: short gaps linear blend; medium gaps use
        linear x + quadratic y poly fit over a local confirmed window if mean residual OK.
        Skips steps that would land on a torso. Does not bridge huge jumps (large_jump_threshold).
        """
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
            if distance_between(previous_state.center, current_state.center) > self.config.ball_large_jump_threshold * max(1, gap):
                continue
            if gap <= self.config.ball_interp_short_gap:
                for step in range(1, gap + 1):
                    ratio = step / float(gap + 1)
                    frame_index = previous_index + step
                    center = (
                        int(round(previous_state.center[0] * (1.0 - ratio) + current_state.center[0] * ratio)),
                        int(round(previous_state.center[1] * (1.0 - ratio) + current_state.center[1] * ratio)),
                    )
                    if self._center_hits_torso(frame_index, center):
                        continue
                    states[frame_index] = BallFrameState(
                        center=center,
                        radius=previous_state.radius * (1.0 - ratio) + current_state.radius * ratio,
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
                if float(np.mean(super()._point_residuals(window_points))) > self.config.ball_interp_residual_limit:
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
        """
        Remove isolated spikes: frame far from both neighbors but neighbors agree with each
        other, or a dangling far segment. Preserves recover_rf frames (trust recovery).
        """
        valid_frames = [index for index, state in enumerate(states) if state.center is not None and state.status in {"confirmed", "interpolated"}]
        threshold = self.config.ball_large_jump_threshold
        to_clear: set[int] = set()
        for position, frame_index in enumerate(valid_frames):
            state = states[frame_index]
            if state.source == "recover_rf":
                continue
            previous_index = valid_frames[position - 1] if position > 0 else None
            next_index = valid_frames[position + 1] if position + 1 < len(valid_frames) else None
            previous_far = previous_index is not None and distance_between(states[previous_index].center, state.center) > threshold
            next_far = next_index is not None and distance_between(state.center, states[next_index].center) > threshold
            if previous_far and next_far and previous_index is not None and next_index is not None:
                if distance_between(states[previous_index].center, states[next_index].center) <= threshold * 1.25:
                    to_clear.add(frame_index)
                    continue
            if previous_far and (next_index is None or next_far):
                to_clear.add(frame_index)
        for frame_index in to_clear:
            states[frame_index] = BallFrameState()

    def _finalize_debug_v6(self, states: list[BallFrameState]) -> None:
        """Populate long_gaps, large_jump_frames, overlap_frames, flagged_frames for export."""
        gap_start: Optional[int] = None
        flagged: set[int] = set()
        previous_rendered: Optional[tuple[int, int]] = None
        for index, state in enumerate(states):
            if state.center is None:
                if gap_start is None:
                    gap_start = index
                continue
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


def save_debug_artifacts(
    debug_dir: Path,
    input_path: Path,
    transforms: list[np.ndarray],
    ball_states: list[BallFrameState],
    debug_info: BallDebugInfo,
) -> None:
    """Write ball_report.csv and a thumbnail contact sheet for flagged / interesting frames."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    csv_path = debug_dir / "ball_report.csv"
    recovery_frames = {event.frame_index for event in debug_info.recovery_events}
    overlap_frames = set(debug_info.overlap_frames)
    large_jump_frames = set(debug_info.large_jump_frames)
    gap_frames: set[int] = set()
    for start, end in debug_info.long_gaps:
        gap_frames.update(range(start, end + 1))
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "status", "confidence", "source", "recovery_ml", "torso_overlap", "long_gap", "large_jump"])
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
                    int(frame_index in large_jump_frames),
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
        cv2.putText(transformed, f"F{frame_index} {state.status}", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if state.center is not None and state.radius is not None:
            cv2.circle(transformed, state.center, max(3, int(round(state.radius))), (0, 255, 0), 2)
        images.append(cv2.resize(transformed, (320, 180)))
    capture.release()
    if not images:
        return
    columns = 3
    rows = int(math.ceil(len(images) / columns))
    blank = np.zeros_like(images[0])
    images.extend(blank.copy() for _ in range(rows * columns - len(images)))
    sheet = cv2.vconcat([cv2.hconcat(images[row * columns : (row + 1) * columns]) for row in range(rows)])
    cv2.imwrite(str(debug_dir / "contact_sheet.png"), sheet)


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI defaults assume repo layout: task phase 6 video and tp6 output path."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Trajectory-first volleyball tracker v6")
    parser.add_argument("--input", type=Path, default=root / "task phase 6" / "Volleyball.mp4", help="Path to the input volleyball video.")
    parser.add_argument("--output", type=Path, default=root / "tp6" / "Volleyball_annotated_opencv_v6.mp4", help="Path to the output annotated video.")
    parser.add_argument("--display", action="store_true", help="Display the annotated frames during rendering.")
    parser.add_argument(
        "--stabilize",
        choices=("auto", "on", "off"),
        default="auto",
        help="Enable ORB-based global motion stabilization before tracking.",
    )
    parser.add_argument(
        "--disable-ball-recovery-ml",
        action="store_true",
        help="Disable the recovery-only RandomForest reranker and keep v6 fully OpenCV-driven.",
    )
    parser.add_argument("--debug-dir", type=Path, default=None, help="Optional directory for v6 debug artifacts.")
    return parser


def main() -> None:
    """
    End-to-end: probe video → collect observations (v5) → players → v6 track → render.

    Recovery model artifact: tp6/_artifacts/ball_recovery_rf.joblib; training labels:
    task phase 6/datasets/ball (optional). Use --disable-ball-recovery-ml for OpenCV-only recovery.
    """
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
    recovery_model = BallRecoveryModelV5(
        root / "tp6" / "_artifacts" / "ball_recovery_rf.joblib",
        root / "task phase 6" / "datasets" / "ball",
        config,
        args.disable_ball_recovery_ml,
    )
    ball_states, debug_info = BallTrajectoryTrackerV6(
        config,
        frame_shape,
        fps,
        observations,
        player_states,
        args.input,
        transforms,
        recovery_model,
    ).build()

    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen input video: {args.input}")

    render_start = time.time()
    frame_index = 0
    stabilization_text = "Stab: on" if motion_probe.enabled else f"Stab: off ({motion_probe.mean_translation:.2f}px)"
    while True:
        ok, frame = capture.read()
        if not ok or frame_index >= len(observations):
            break
        annotated = apply_transform(frame, transforms[frame_index]).copy()
        draw_players(annotated, player_states[frame_index].visible_tracks)
        draw_ball(annotated, ball_states, frame_index)
        elapsed = max(time.time() - render_start, 1e-6)
        draw_overlay(
            annotated,
            player_states[frame_index].team_a_count,
            player_states[frame_index].team_b_count,
            (frame_index + 1) / elapsed,
            stabilization_text,
        )
        writer.write(annotated)
        if args.display:
            cv2.imshow("Volleyball Tracker OpenCV V6", annotated)
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
