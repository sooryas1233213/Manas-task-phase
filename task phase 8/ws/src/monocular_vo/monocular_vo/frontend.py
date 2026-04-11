from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    max_features: int
    min_tracked_points: int
    min_feature_distance_px: float
    shi_tomasi_quality_level: float
    shi_tomasi_block_size: int
    lk_win_size: int
    lk_max_level: int
    lk_max_iterations: int
    lk_epsilon: float
    feature_border_px: int


@dataclass(frozen=True)
class TrackResult:
    previous_points: np.ndarray
    current_points: np.ndarray
    rejected_previous_points: np.ndarray
    rejected_current_points: np.ndarray
    total_input_points: int


def _empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def detect_features(image_gray: np.ndarray, config: TrackingConfig) -> np.ndarray:
    mask = np.full(image_gray.shape[:2], 255, dtype=np.uint8)
    border = max(0, int(config.feature_border_px))
    if border > 0 and image_gray.shape[0] > border * 2 and image_gray.shape[1] > border * 2:
        mask[:border, :] = 0
        mask[-border:, :] = 0
        mask[:, :border] = 0
        mask[:, -border:] = 0

    features = cv2.goodFeaturesToTrack(
        image_gray,
        maxCorners=int(config.max_features),
        qualityLevel=float(config.shi_tomasi_quality_level),
        minDistance=float(config.min_feature_distance_px),
        blockSize=int(config.shi_tomasi_block_size),
        mask=mask,
        useHarrisDetector=False,
    )

    if features is None:
        return _empty_points()
    return features.reshape(-1, 2).astype(np.float32)


def track_features(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_points: np.ndarray,
    config: TrackingConfig,
) -> TrackResult:
    if previous_points.size == 0:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            rejected_previous_points=_empty_points(),
            rejected_current_points=_empty_points(),
            total_input_points=0,
        )

    previous_points_2d = previous_points.reshape(-1, 2).astype(np.float32)
    lk_previous = previous_points_2d.reshape(-1, 1, 2)
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        lk_previous,
        None,
        winSize=(int(config.lk_win_size), int(config.lk_win_size)),
        maxLevel=int(config.lk_max_level),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(config.lk_max_iterations),
            float(config.lk_epsilon),
        ),
    )

    if current_points is None or status is None:
        return TrackResult(
            previous_points=_empty_points(),
            current_points=_empty_points(),
            rejected_previous_points=previous_points_2d,
            rejected_current_points=previous_points_2d.copy(),
            total_input_points=int(previous_points_2d.shape[0]),
        )

    current_points_2d = current_points.reshape(-1, 2).astype(np.float32)
    status_mask = status.reshape(-1).astype(bool)
    finite_mask = np.isfinite(previous_points_2d).all(axis=1) & np.isfinite(current_points_2d).all(axis=1)
    keep_mask = status_mask & finite_mask

    border = float(max(0, int(config.feature_border_px)))
    width = float(current_gray.shape[1])
    height = float(current_gray.shape[0])
    previous_in_bounds = (
        (previous_points_2d[:, 0] >= border)
        & (previous_points_2d[:, 0] < width - border)
        & (previous_points_2d[:, 1] >= border)
        & (previous_points_2d[:, 1] < height - border)
    )
    current_in_bounds = (
        (current_points_2d[:, 0] >= border)
        & (current_points_2d[:, 0] < width - border)
        & (current_points_2d[:, 1] >= border)
        & (current_points_2d[:, 1] < height - border)
    )
    keep_mask &= previous_in_bounds & current_in_bounds

    return TrackResult(
        previous_points=previous_points_2d[keep_mask],
        current_points=current_points_2d[keep_mask],
        rejected_previous_points=previous_points_2d[~keep_mask],
        rejected_current_points=current_points_2d[~keep_mask],
        total_input_points=int(previous_points_2d.shape[0]),
    )
